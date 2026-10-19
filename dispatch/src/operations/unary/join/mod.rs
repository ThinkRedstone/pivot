//! Parallel hash join: a radix-partitioned build phase populating a shared
//! directory + arena, and a prefetch-pipelined probe phase emitting the joined
//! rows.
//!
//! Build and probe are separate roots in one dataflow. Every probe-input root
//! is gated on a publication flag set after all build partition jobs complete,
//! then the probe reads the populated [`JoinTable`].
//!
//! Output layout: the probe-side columns listed in
//! [`JoinSpec::probe_output_indices`] (in list order) followed by the listed
//! build-side columns. The join is an equi-join whose key columns are read,
//! hashed, and compared through one of the [`JoinKey`] shapes, in one of the
//! [`JoinKind`]s below.
//!
//! A build-side outer join ([`JoinKind::BuildOuter`]) also emits every build
//! row no probe row matched, its probe columns null-filled. Which rows those
//! are is only known once every worker has stopped probing, so the probe marks
//! each matched build row in a shared flag array and the workers scan it
//! together behind a barrier ([`UnmatchedScan`]).
//!
//! A probe-side semi join ([`JoinKind::ProbeSemi`]) emits a probe row once if
//! the build side holds its key at all. Each probe row's candidates live in one
//! arena range, so the probe stops walking that range at the first key that
//! matches; nothing else about the phases changes, and no cross-worker state is
//! involved.
//!
//! A probe-side outer join ([`JoinKind::ProbeOuter`]) also emits every probe
//! row nothing matched, its build columns null-filled. A probe row's matches
//! all surface while its own batch is probed, so each worker settles its own
//! batches with no cross-worker state (see the probe module doc).
//!
//! A probe-side anti join ([`JoinKind::ProbeAnti`]) emits exactly the probe
//! rows a probe-side outer join would pad, and nothing else: it runs the same
//! miss tracking, but a matched pair only settles its probe row's fate and
//! never becomes an output row.
//!
//! A build-side anti join ([`JoinKind::BuildAnti`]) is the same inversion of
//! the build-side outer join: the flag array and unmatched scan run
//! unchanged, only the unmatched build rows reach the output, and they carry
//! no probe columns.
//!
//! A build-side semi join ([`JoinKind::BuildSemi`]) runs that same flag array
//! and scan with the polarity flipped: the scan emits the flagged build rows,
//! one output row per build row however many probe rows matched it, again as
//! build columns alone.
//!
//! A mark join ([`JoinKind::ProbeMark`]) emits every probe row with a
//! `mark` boolean column instead of filtering: TRUE on a match, FALSE on a
//! miss, NULL where SQL's three-valued `IN` cannot call the miss FALSE. It
//! classifies rows with the probe-side miss tracking and the semi join's
//! first-match exit, and emits batch-at-a-time with no accumulation.

mod build;
mod build_rows;
mod directory;
mod factory;
mod key_bitset;
pub use factory::JoinRecordBatchOperatorFactory;
pub use key_bitset::{KeyBitset, KeyBitsetSlot};
mod keys;
pub use keys::{DynamicRowKey, JoinKey, PackedKey, SingleColumnKey};
mod match_outputter;
mod probe;
mod range;
pub(crate) use range::create_range_join_factories;
pub use range::{RangeCompare, RangeJoinSpec};
mod residual_filter;
mod row_arena;

use std::cell::UnsafeCell;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::Field;

use crate::boundary_slot::BoundarySlot;
use crate::memory::MultiSlabBuffer;
use crate::operations::unary::join::build_rows::BuildRows;
use crate::operations::unary::join::directory::JoinDirectory;
use crate::operations::unary::join::row_arena::RowArena;
pub(crate) use factory::create_for_workers as create_join_factories;
pub(crate) use factory::create_normalizing_for_workers;

/// Which rows a join emits.
#[derive(Debug, Clone)]
pub enum JoinKind {
    /// One output row per matching (probe row, build row) pair.
    Inner,
    /// Every pair an [`Inner`](JoinKind::Inner) emits, plus one row per build
    /// row nothing matched, its probe columns null-filled.
    BuildOuter,
    /// Every pair an [`Inner`](JoinKind::Inner) emits, plus one row per probe
    /// row nothing matched (a null key, an empty build side, or every pair
    /// rejected by the residual), its build columns null-filled.
    ProbeOuter,
    /// One output row per probe row that has at least one matching build row,
    /// with no duplicates for a probe row that matches several. That is also
    /// why such a join emits no build columns: there is no single build row to
    /// take them from, so [`JoinSpec::build_output_indices`] must be empty.
    ProbeSemi,
    /// One output row per probe row that has no surviving match (a null key,
    /// an empty build side, or every pair rejected by the residual): exactly
    /// the rows a [`ProbeOuter`](JoinKind::ProbeOuter) would pad. There is no
    /// build row to take columns from, so
    /// [`JoinSpec::build_output_indices`] must be empty.
    ProbeAnti,
    /// One output row per build row that has no surviving match (a null key
    /// or every pair rejected by the residual): exactly the rows a
    /// [`BuildOuter`](JoinKind::BuildOuter) would pad. There is no probe row
    /// to take columns from, so [`JoinSpec::probe_output_indices`] must be
    /// empty.
    BuildAnti,
    /// One output row per build row that has a surviving match, with no
    /// duplicates for a build row several probe rows matched: exactly the
    /// build rows a [`BuildAnti`](JoinKind::BuildAnti) join withholds. That
    /// is also why such a join emits no probe columns: there is no single
    /// probe row to take them from, so [`JoinSpec::probe_output_indices`]
    /// must be empty.
    BuildSemi,
    /// One output row per probe row, its listed probe columns followed by a
    /// nullable `mark` boolean column: TRUE when some build row matched,
    /// FALSE when none did, and NULL for the unmatched rows whose miss is
    /// three-valued (a null probe key against a non-empty build side, or a
    /// null-keyed build row that could have matched). SQL's `IN (...)`
    /// semantics, which is what DuckDB compiles into this kind. Exactly one
    /// key column per side, no build columns
    /// ([`JoinSpec::build_output_indices`] must be empty), and no residual:
    /// the marker's three-valued logic is defined per key comparison.
    ProbeMark,
}

/// How a join is configured beyond the key shape it is instantiated for.
#[derive(Debug, Clone)]
pub struct JoinSpec {
    /// Indices of probe input columns used as join keys.
    pub probe_key_indices: Vec<usize>,
    /// Indices of build input columns used as aligned join keys.
    pub build_key_indices: Vec<usize>,
    /// Indices of probe input columns emitted first.
    pub probe_output_indices: Vec<usize>,
    /// Indices of build input columns emitted second.
    pub build_output_indices: Vec<usize>,
    /// The fields the listed probe columns take in the output, in
    /// `probe_output_indices` order. Stated by the caller rather than read
    /// off a probed batch so every worker shapes identical output whether or
    /// not it ever received a batch, and so an outer join's probe columns can
    /// be nullable regardless of the input's declared nullability.
    pub probe_fields: Vec<Field>,
    /// The fields the listed build columns take in the output, in
    /// `build_output_indices` order.
    pub build_fields: Vec<Field>,
    /// Which rows reach the output.
    pub kind: JoinKind,
    /// An optional predicate evaluated over key-matched pairs.
    ///
    /// The residual declares the probe and build columns needed for evaluation.
    /// A candidate pair counts as a match only when the predicate returns `TRUE`.
    pub residual_filters: Option<JoinResidualSpec>,
    /// Filters the build side publishes for sibling probe-side scans: when
    /// the build seals, each listed key column's minimum and maximum are
    /// published into the paired slots, which those scans read to prune row
    /// groups whose key range falls entirely outside the bounds.
    pub build_filters: Vec<JoinBuildFilter>,
}

/// One filter a join build publishes when it seals: the bounds of one build
/// key column into a pair of [`BoundarySlot`]s, and the key set itself
/// as a [`KeyBitset`] when its shape allows one, all read by consumer scans.
#[derive(Debug, Clone)]
pub struct JoinBuildFilter {
    /// The build input column whose values are bounded (a join key column).
    pub build_column: usize,
    /// Receives the smallest non-null build key; consumers compare with `>=`.
    pub min_slot: Arc<BoundarySlot>,
    /// Receives the largest non-null build key; consumers compare with `<=`.
    pub max_slot: Arc<BoundarySlot>,
    /// Receives the sealed key set as an exact key bitset; consumers
    /// drop rows whose key it does not hold.
    pub key_bitset_slot: Arc<KeyBitsetSlot>,
}

/// One evaluation instance of a join's residual predicate: batch of paired
/// rows in, one keep/reject boolean per pair out (a NULL rejects, matching
/// SQL's treatment of a non-TRUE condition).
pub type JoinResidualFn = Box<dyn FnMut(&RecordBatch) -> BooleanArray + Send>;

/// A residual-predicate factory and the minimal input projection it evaluates.
/// Each probe worker gets its own evaluator because evaluation is stateful.
/// The evaluator's batch contains `probe_column_indices` first, followed by
/// `build_column_indices`, in the order listed here.
#[derive(Clone)]
pub struct JoinResidualSpec {
    evaluator: Arc<dyn Fn() -> JoinResidualFn + Send + Sync>,
    probe_column_indices: Arc<[usize]>,
    build_column_indices: Arc<[usize]>,
}

impl JoinResidualSpec {
    pub fn new(
        evaluator: Arc<dyn Fn() -> JoinResidualFn + Send + Sync>,
        probe_column_indices: Vec<usize>,
        build_column_indices: Vec<usize>,
    ) -> Self {
        Self {
            evaluator,
            probe_column_indices: probe_column_indices.into(),
            build_column_indices: build_column_indices.into(),
        }
    }
}

impl fmt::Debug for JoinResidualSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JoinResidualSpec")
            .field("probe_column_indices", &self.probe_column_indices)
            .field("build_column_indices", &self.build_column_indices)
            .finish_non_exhaustive()
    }
}

/// The cross-worker state of the build-row flag scan a build-side outer,
/// anti, or semi join runs once probing ends: the first two emit the rows
/// whose flag stayed clear, a semi join the rows whose flag was set.
pub(crate) struct UnmatchedScan {
    /// Probe workers that may still mark a matched build row. Each decrements
    /// it once, on entering `finish`, after which that worker never consumes
    /// again; at zero the flags are final and the scan below can start.
    probes_finished: AtomicUsize,
    /// The next build row batch the scan hands out, so workers claim disjoint
    /// stretches of the flag array.
    cursor: AtomicUsize,
}

impl UnmatchedScan {
    fn new(worker_count: usize) -> Self {
        Self {
            probes_finished: AtomicUsize::new(worker_count),
            cursor: AtomicUsize::new(0),
        }
    }
}

/// Interior-mutable storage shared by partitioned build workers.
pub(crate) struct JoinCell<T>(UnsafeCell<T>);

impl<T> JoinCell<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    pub(crate) fn get(&self) -> *mut T {
        self.0.get()
    }
}

// Build workers mutate disjoint directory partitions and arena ranges. The API
// keeps the raw cell private so only that partitioned build protocol can write.
unsafe impl<T: Send> Send for JoinCell<T> {}
unsafe impl<T: Send> Sync for JoinCell<T> {}

/// Shared hash-table state handed from the build factories to the probe
/// factories. `keys` and `rows` are parallel arenas indexed by the directory's
/// slot cursors: the full-width join key, and the row's id into `build_rows`,
/// the stored build rows. All fields are populated by the build phase and
/// published to probe only after every partition job has run.
#[derive(Clone)]
pub(crate) struct JoinTable<K> {
    pub(crate) directory: Arc<JoinCell<JoinDirectory>>,
    pub(crate) keys: Arc<JoinCell<MultiSlabBuffer<K>>>,
    pub(crate) rows: Arc<JoinCell<RowArena>>,
    pub(crate) build_rows: Arc<JoinCell<BuildRows>>,
    /// Whether any build worker saw a null key. The final worker at the build
    /// gather barrier writes it before the table is published; probes read it
    /// only after that publication. A mark join's misses are NULL rather than
    /// FALSE when this is true (the null could have matched).
    pub(crate) build_saw_null_key: Arc<JoinCell<bool>>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::{Int32Type, Int64Type};
    use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};

    use crate::memory::init_test_free_pool;
    use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
    use crate::operations::unary::test_utils::CollectSender;
    use crate::operations::{Unary, UnaryFactory};

    use super::build::JoinBuildConsumer;
    use super::factory;
    use super::keys::{DynamicRowKey, JoinKey, PackedKey, SingleColumnKey};

    fn int64_batch(keys: &[i64]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(keys.to_vec()))],
        )
        .unwrap()
    }

    fn keyed_names_batch(keys: &[i64], names: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("key", DataType::Int64, false),
                Field::new("name", DataType::Utf8View, false),
            ])),
            vec![
                Arc::new(Int64Array::from(keys.to_vec())),
                Arc::new(StringViewArray::from(names.to_vec())),
            ],
        )
        .unwrap()
    }

    type Int64Key = SingleColumnKey<arrow_array::types::Int64Type>;

    fn extract_consumer<K: JoinKey, const TRACK_MATCHED_BUILD_ROWS: bool>(
        breaker: PipelineBreaker<RecordBatch, (), JoinBuildConsumer<K, TRACK_MATCHED_BUILD_ROWS>>,
    ) -> JoinBuildConsumer<K, TRACK_MATCHED_BUILD_ROWS> {
        match breaker {
            PipelineBreaker::Consuming(c) => c,
            _ => unreachable!(),
        }
    }

    struct JoinResult {
        batches: Vec<RecordBatch>,
    }

    impl JoinResult {
        fn rows(&self) -> usize {
            self.batches.iter().map(|b| b.num_rows()).sum()
        }
    }

    fn build_and_probe(
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
    ) -> JoinResult {
        run_join::<Int64Key, false, false, false>(
            super::JoinKind::Inner,
            build_worker_batches,
            probe_batches,
            None,
            vec![0],
        )
    }

    /// Build-side outer: every build row reaches the output, with null probe
    /// columns when nothing matched.
    fn build_and_probe_outer(
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
        probe_fields: Vec<Field>,
    ) -> JoinResult {
        run_join::<Int64Key, true, false, false>(
            super::JoinKind::BuildOuter,
            build_worker_batches,
            probe_batches,
            Some(probe_fields),
            vec![0],
        )
    }

    /// Probe-side outer: every probe row reaches the output, with null build
    /// columns when nothing matched.
    fn build_and_probe_probe_outer(
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
    ) -> JoinResult {
        run_join::<Int64Key, false, true, false>(
            super::JoinKind::ProbeOuter,
            build_worker_batches,
            probe_batches,
            None,
            vec![0],
        )
    }

    /// Probe-side anti: exactly the probe rows nothing matched reach the
    /// output, as probe columns alone.
    fn build_and_probe_anti(
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
    ) -> JoinResult {
        run_join::<Int64Key, false, true, true>(
            super::JoinKind::ProbeAnti,
            build_worker_batches,
            probe_batches,
            None,
            vec![0],
        )
    }

    /// Build-side anti: exactly the build rows no probe row matched reach the
    /// output, as build columns alone.
    fn build_and_probe_build_anti(
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
    ) -> JoinResult {
        run_join::<Int64Key, true, false, true>(
            super::JoinKind::BuildAnti,
            build_worker_batches,
            probe_batches,
            None,
            vec![0],
        )
    }

    /// Build-side semi: exactly the build rows some probe row matched reach
    /// the output, once each, as build columns alone.
    fn build_and_probe_build_semi(
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
    ) -> JoinResult {
        run_join::<Int64Key, true, false, true>(
            super::JoinKind::BuildSemi,
            build_worker_batches,
            probe_batches,
            None,
            vec![0],
        )
    }

    /// Run one join to completion: one build worker per entry of
    /// `build_worker_batches`, and as many probe workers, all of whose `finish`
    /// is driven (the unmatched pass only starts once every one has arrived).
    /// The probe batches all go to the first, so the rest exercise a worker
    /// reaching that pass with no batch of its own. Both sides key on
    /// `key_columns`. `kind` must agree with the const parameters, which the
    /// factory asserts (the consts alone cannot tell a build-side semi from a
    /// build-side anti).
    fn run_join<K: JoinKey, const BUILD_OUTER: bool, const PROBE_OUTER: bool, const ANTI: bool>(
        kind: super::JoinKind,
        build_worker_batches: Vec<Vec<RecordBatch>>,
        probe_batches: Vec<RecordBatch>,
        probe_fields: Option<Vec<Field>>,
        key_columns: Vec<usize>,
    ) -> JoinResult {
        init_test_free_pool(64);
        let workers = build_worker_batches.len();
        // A join preserving one side alone emits only that side's columns.
        let probe_column_count = if ANTI && BUILD_OUTER {
            0
        } else {
            probe_batches[0].num_columns()
        };
        let build_column_count = if ANTI && PROBE_OUTER {
            0
        } else {
            build_worker_batches[0][0].num_columns()
        };
        let probe_fields = probe_fields.unwrap_or_else(|| {
            probe_batches[0]
                .schema()
                .fields()
                .iter()
                .take(probe_column_count)
                .map(|f| f.as_ref().clone())
                .collect()
        });
        // A probe-side outer join pads build columns with NULLs, whatever
        // their input nullability. An anti join has no build columns at all.
        let build_fields: Vec<Field> = build_worker_batches[0][0]
            .schema()
            .fields()
            .iter()
            .take(build_column_count)
            .map(|f| {
                f.as_ref()
                    .clone()
                    .with_nullable(f.is_nullable() || PROBE_OUTER)
            })
            .collect();
        let spec = super::JoinSpec {
            probe_key_indices: key_columns.clone(),
            build_key_indices: key_columns,
            probe_output_indices: (0..probe_column_count).collect(),
            build_output_indices: (0..build_column_count).collect(),
            probe_fields,
            build_fields,
            kind,
            residual_filters: None,
            build_filters: Vec::new(),
        };
        let (builds, probes, _) =
            factory::create_for_workers::<K, BUILD_OUTER, false, PROBE_OUTER, ANTI, false>(
                spec, workers,
            );

        let mut consumers: Vec<_> = builds
            .into_iter()
            .map(|f| extract_consumer(f.build_unary()))
            .collect();

        let mut void = CollectSender::<()>::new();
        for (consumer, batches) in consumers.iter_mut().zip(&build_worker_batches) {
            for batch in batches {
                consumer.consume(batch.clone(), &mut void).unwrap();
            }
        }

        let mut outputters: Vec<_> = consumers
            .into_iter()
            .enumerate()
            .filter_map(|(worker_index, c)| {
                crate::worker::WORKER_IDX.set(worker_index);
                c.into_outputter().unwrap()
            })
            .collect();
        crate::worker::WORKER_IDX.set(0);

        loop {
            let mut all_done = true;
            for o in &mut outputters {
                if !o.output(&mut void).unwrap() {
                    all_done = false;
                }
            }
            if all_done {
                break;
            }
        }

        let mut probers: Vec<_> = probes.into_iter().map(|p| p.build_unary()).collect();
        let mut sender = CollectSender::new();
        let mut test_io = crate::io::TestOperatorIO::default();
        let mut io = test_io.io();
        for batch in probe_batches {
            probers[0].consume(batch, &mut sender, &mut io).unwrap();
        }
        loop {
            let mut all_done = true;
            for probe in &mut probers {
                if !probe.finish(&mut sender).unwrap() {
                    all_done = false;
                }
            }
            if all_done {
                break;
            }
        }
        JoinResult {
            batches: sender.items,
        }
    }

    fn collect_i64_column(results: &[RecordBatch], col: usize) -> Vec<i64> {
        results
            .iter()
            .flat_map(|b| {
                b.column(col)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect()
    }

    #[test]
    fn single_key_match() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[20])],
        );

        let keys = collect_i64_column(&r.batches, 1);
        assert_eq!(keys, vec![20]);
    }

    #[test]
    fn multiple_key_matches() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30, 40, 50])]],
            vec![int64_batch(&[20, 40])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![20, 40]);
    }

    #[test]
    fn no_matches() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[99, 100])],
        );

        assert!(r.batches.is_empty());
    }

    #[test]
    fn duplicate_build_keys_produce_multiple_matches() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 10, 20])]],
            vec![int64_batch(&[10])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![10, 10]);
    }

    #[test]
    fn multiple_build_batches() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20]), int64_batch(&[30, 40])]],
            vec![int64_batch(&[20, 30])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![20, 30]);
    }

    #[test]
    fn two_build_workers() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20])], vec![int64_batch(&[30, 40])]],
            vec![int64_batch(&[10, 30])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![10, 30]);
    }

    #[test]
    fn empty_build_no_matches() {
        let r = build_and_probe(vec![vec![int64_batch(&[])]], vec![int64_batch(&[10])]);

        assert!(r.batches.is_empty());
    }

    #[test]
    fn empty_probe_no_output() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[])],
        );

        assert!(r.batches.is_empty());
    }

    #[test]
    fn duplicate_probe_keys() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20])]],
            vec![int64_batch(&[10, 10, 10])],
        );

        let keys = collect_i64_column(&r.batches, 1);
        assert_eq!(keys, vec![10, 10, 10]);
    }

    #[test]
    fn many_to_many() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 10])]],
            vec![int64_batch(&[10, 10])],
        );

        let keys = collect_i64_column(&r.batches, 1);
        assert_eq!(keys.len(), 4);
        assert!(keys.iter().all(|&k| k == 10));
    }

    #[test]
    fn multiple_probe_batches() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[10]), int64_batch(&[30])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![10, 30]);
    }

    #[test]
    fn all_keys_match() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[1, 2, 3, 4, 5])]],
            vec![int64_batch(&[1, 2, 3, 4, 5])],
        );

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn probe_handles_matches_duplicates_and_misses() {
        let build_batches = vec![vec![int64_batch(&[10, 10, 20, 30, 40])]];
        let probe_batches = vec![int64_batch(&[99, 10, 30, 99, 10])];

        let r = build_and_probe(build_batches, probe_batches);

        let mut keys = collect_i64_column(&r.batches, 1);
        keys.sort();
        assert_eq!(keys, vec![10, 10, 10, 10, 30]);
    }

    #[test]
    fn probe_columns_pass_through() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![keyed_names_batch(&[99, 20, 99, 30], &["a", "b", "c", "d"])],
        );

        // Output: probe columns (key, name), then build columns (key).
        let batch = &r.batches[0];
        assert_eq!(batch.num_columns(), 3);
        let names: Vec<&str> = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        assert_eq!(names, vec!["b", "d"]);
        assert_eq!(collect_i64_column(&r.batches, 2), vec![20, 30]);
    }

    #[test]
    fn build_columns_are_gathered() {
        let r = build_and_probe(
            vec![vec![keyed_names_batch(&[10, 20, 30], &["x", "y", "z"])]],
            vec![int64_batch(&[30, 10])],
        );

        // Output: probe columns (key), then build columns (key, name).
        let batch = &r.batches[0];
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(collect_i64_column(&r.batches, 0), vec![30, 10]);
        assert_eq!(collect_i64_column(&r.batches, 1), vec![30, 10]);
        let names: Vec<&str> = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        assert_eq!(names, vec!["z", "x"]);
    }

    #[test]
    fn build_rows_span_workers_and_batches() {
        let r = build_and_probe(
            vec![
                vec![
                    keyed_names_batch(&[1, 2], &["a", "b"]),
                    keyed_names_batch(&[3], &["c"]),
                ],
                vec![keyed_names_batch(&[4, 5], &["d", "e"])],
            ],
            vec![int64_batch(&[5, 3, 1])],
        );

        let mut pairs: Vec<(i64, String)> = r
            .batches
            .iter()
            .flat_map(|b| {
                let keys = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                let names = b
                    .column(2)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                (0..b.num_rows())
                    .map(|i| (keys.value(i), names.value(i).to_string()))
                    .collect::<Vec<_>>()
            })
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                (1, "a".to_string()),
                (3, "c".to_string()),
                (5, "e".to_string())
            ]
        );
    }

    #[test]
    fn keys_beyond_u32_do_not_alias() {
        // Two keys that collide when truncated to 32 bits.
        let low = 7;
        let high = 7 + (1i64 << 32);
        let r = build_and_probe(
            vec![vec![int64_batch(&[high])]],
            vec![int64_batch(&[low, high])],
        );

        assert_eq!(collect_i64_column(&r.batches, 0), vec![high]);
    }

    #[test]
    fn null_keys_never_match() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
        let build = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![Some(10), None]))],
        )
        .unwrap();
        let probe = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![None, Some(10), None]))],
        )
        .unwrap();

        let r = build_and_probe(vec![vec![build]], vec![probe]);

        let total: usize = r.batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn straddling_drain_remainder_loses_no_pairs() {
        // A drain remainder left by one probe batch plus the next batch's
        // full drain accumulates more than one output batch before emitting.
        // Every pair must still come through; consumers handle the oversized
        // batch themselves.
        let repeated = 3_000usize;
        let distinct = 9_000usize;
        let mut build_keys = vec![10i64; repeated];
        build_keys.extend(100i64..100 + distinct as i64);
        let build_batches = build_keys
            .chunks(crate::RECORD_BATCH_SIZE)
            .map(int64_batch)
            .collect();
        let probe_distinct: Vec<i64> = (100..100 + distinct as i64).collect();

        let r = build_and_probe(
            vec![build_batches],
            vec![int64_batch(&[10]), int64_batch(&probe_distinct)],
        );

        assert_eq!(r.rows(), repeated + distinct);
    }

    /// The q20 corruption shape end to end: a join whose straddling drains
    /// emit a larger-than-one-output batch, feeding a coalescing filter. The
    /// filter's accumulator must chunk the oversized append itself; before it
    /// did, the overflow wrote past its slabs and the tail rows came out with
    /// garbage values.
    #[test]
    fn oversized_join_output_through_a_filter_keeps_values() {
        use crate::operations::unary::filter::{
            FilterFactory, RowDelivery, RowSelection, SelectionScratch,
        };
        use crate::operations::unary::test_utils::run_unary_to_completion;

        let repeated = 3_000usize;
        let distinct = 9_000usize;
        let mut build_keys = vec![10i64; repeated];
        build_keys.extend(100i64..100 + distinct as i64);
        let build_batches = build_keys
            .chunks(crate::RECORD_BATCH_SIZE)
            .map(int64_batch)
            .collect();
        let probe_distinct: Vec<i64> = (100..100 + distinct as i64).collect();
        let joined = build_and_probe(
            vec![build_batches],
            vec![int64_batch(&[10]), int64_batch(&probe_distinct)],
        );
        assert!(
            joined
                .batches
                .iter()
                .any(|b| b.num_rows() > crate::RECORD_BATCH_SIZE),
            "the join emits at least one oversized batch, or this exercises nothing"
        );

        // Keep all but the first row of each batch: a whole-batch selection
        // would be forwarded untouched, and only partial selections coalesce
        // through the accumulator.
        let drop_first = FilterFactory(
            |b: &RecordBatch, _: &mut _, scratch: &mut SelectionScratch| {
                scratch.positions.clear();
                scratch.positions.extend(1..b.num_rows() as u32);
                RowSelection::Indices
            },
            RowDelivery::Coalesced,
        )
        .build_unary();
        // Same-schema filler rows below the emit threshold first, so the
        // join's oversized batch lands on a part-full accumulator, as q20's
        // did.
        let filler = 8_000usize;
        let filler_keys: Vec<i64> = (1_000_000..1_000_000 + filler as i64).collect();
        let filler_schema = joined.batches[0].schema();
        let filler_columns: Vec<arrow_array::ArrayRef> = (0..filler_schema.fields().len())
            .map(|_| {
                std::sync::Arc::new(arrow_array::Int64Array::from(filler_keys.clone()))
                    as arrow_array::ArrayRef
            })
            .collect();
        let filler_batch = RecordBatch::try_new(filler_schema, filler_columns).unwrap();

        let mut inputs = vec![filler_batch];
        inputs.extend(joined.batches);
        let mut expected: Vec<i64> = inputs
            .iter()
            .flat_map(|b| collect_i64_column(std::slice::from_ref(b), 0).split_off(1))
            .collect();
        expected.sort();

        let filtered = run_unary_to_completion(drop_first, inputs);

        let mut keys = collect_i64_column(&filtered, 0);
        keys.sort();
        assert_eq!(keys, expected);
    }

    #[test]
    fn build_semi_probe_batch_larger_than_pair_buffer() {
        // A probe batch bigger than the collector's pair buffers must still
        // flag every matched build row across the buffer's drain cycles.
        let n = 20_000;
        let keys: Vec<i64> = (0..n as i64).collect();
        let r =
            build_and_probe_build_semi(vec![vec![int64_batch(&keys)]], vec![int64_batch(&keys)]);

        assert_eq!(r.rows(), n);
    }

    #[test]
    fn build_semi_large_build_large_probe_batch() {
        // The build side is large enough for a multi-slab directory and the
        // probe batch overflows the pair buffers twice, with repeating keys.
        let build_n: i64 = 1_000_000;
        let build_keys: Vec<i64> = (0..build_n).collect();
        let build_batches: Vec<Vec<RecordBatch>> = vec![
            build_keys
                .chunks(crate::RECORD_BATCH_SIZE)
                .map(int64_batch)
                .collect(),
        ];
        let probe_n = 19_400usize;
        let probe_keys: Vec<i64> = (0..probe_n as i64).map(|i| (i * 733) % build_n).collect();
        let expected: usize = {
            let mut distinct = probe_keys.clone();
            distinct.sort_unstable();
            distinct.dedup();
            distinct.len()
        };

        let r = build_and_probe_build_semi(build_batches, vec![int64_batch(&probe_keys)]);

        assert_eq!(r.rows(), expected);
    }

    #[test]
    fn output_exceeding_batch_capacity_is_chunked() {
        // One probe row matching more build rows than one output batch holds.
        let n = crate::RECORD_BATCH_SIZE + 100;
        let build_keys = vec![10i64; n];
        let r = build_and_probe(
            vec![vec![int64_batch(&build_keys)]],
            vec![int64_batch(&[10])],
        );

        let total: usize = r.batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, n);
        assert!(
            r.batches
                .iter()
                .all(|b| b.num_rows() <= crate::RECORD_BATCH_SIZE)
        );
    }

    #[test]
    fn probe_row_multiplicity_matches_build_duplicates() {
        let r = build_and_probe(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[99, 20, 99, 30])],
        );

        let mut keys = collect_i64_column(&r.batches, 0);
        keys.sort();
        assert_eq!(keys, vec![20, 30]);
    }

    #[test]
    fn large_build_exercises_multiple_partitions() {
        let keys: Vec<i64> = (0..1000).collect();
        let probe_keys: Vec<i64> = (500..600).collect();

        let r = build_and_probe(
            vec![vec![int64_batch(&keys)]],
            vec![int64_batch(&probe_keys)],
        );

        let mut matched = collect_i64_column(&r.batches, 1);
        matched.sort();
        assert_eq!(matched, probe_keys);
    }

    #[test]
    fn empty_build_still_completes_and_matches_nothing() {
        let r = build_and_probe(vec![vec![int64_batch(&[])]], vec![int64_batch(&[10])]);

        assert!(r.batches.is_empty());
    }

    fn nullable_key_field() -> Vec<Field> {
        vec![Field::new("key", DataType::Int64, true)]
    }

    /// Every output row as (probe key, build key), each null-aware.
    fn collect_key_pairs(results: &[RecordBatch]) -> Vec<(Option<i64>, Option<i64>)> {
        let mut pairs: Vec<(Option<i64>, Option<i64>)> = results
            .iter()
            .flat_map(|batch| {
                let key = |col: usize, row: usize| {
                    let column = batch
                        .column(col)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    column.is_valid(row).then(|| column.value(row))
                };
                (0..batch.num_rows())
                    .map(|row| (key(0, row), key(1, row)))
                    .collect::<Vec<_>>()
            })
            .collect();
        pairs.sort();
        pairs
    }

    #[test]
    fn unmatched_build_rows_come_out_with_null_probe_columns() {
        let r = build_and_probe_outer(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[20])],
            nullable_key_field(),
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(None, Some(10)), (None, Some(30)), (Some(20), Some(20))]
        );
    }

    #[test]
    fn every_build_row_survives_a_probe_side_that_matches_nothing() {
        let r = build_and_probe_outer(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[99])],
            nullable_key_field(),
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(None, Some(10)), (None, Some(20)), (None, Some(30))]
        );
    }

    #[test]
    fn a_build_row_matched_twice_is_not_also_reported_unmatched() {
        let r = build_and_probe_outer(
            vec![vec![int64_batch(&[10])]],
            vec![int64_batch(&[10, 10])],
            nullable_key_field(),
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(Some(10), Some(10)), (Some(10), Some(10))]
        );
    }

    #[test]
    fn null_keyed_build_rows_are_emitted_unmatched() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
        let build = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![Some(10), None]))],
        )
        .unwrap();

        let r = build_and_probe_outer(
            vec![vec![build]],
            vec![int64_batch(&[10])],
            nullable_key_field(),
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(None, None), (Some(10), Some(10))]
        );
    }

    #[test]
    fn the_unmatched_pass_covers_every_build_worker() {
        let r = build_and_probe_outer(
            vec![vec![int64_batch(&[1, 2])], vec![int64_batch(&[3, 4])]],
            vec![int64_batch(&[3])],
            nullable_key_field(),
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![
                (None, Some(1)),
                (None, Some(2)),
                (None, Some(4)),
                (Some(3), Some(3))
            ]
        );
    }

    #[test]
    fn an_outer_build_larger_than_one_batch_emits_every_row() {
        let build_keys: Vec<i64> = (0..crate::RECORD_BATCH_SIZE as i64 * 2 + 5).collect();
        let r = build_and_probe_outer(
            vec![vec![int64_batch(&build_keys)]],
            vec![int64_batch(&[7])],
            nullable_key_field(),
        );

        assert_eq!(r.rows(), build_keys.len());
    }

    #[test]
    fn unmatched_probe_rows_come_out_with_null_build_columns() {
        let r = build_and_probe_probe_outer(
            vec![vec![int64_batch(&[20])]],
            vec![int64_batch(&[10, 20, 30])],
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(Some(10), None), (Some(20), Some(20)), (Some(30), None)]
        );
    }

    #[test]
    fn a_probe_row_matching_several_build_rows_is_not_also_padded() {
        let r = build_and_probe_probe_outer(
            vec![vec![int64_batch(&[10, 10])]],
            vec![int64_batch(&[10])],
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(Some(10), Some(10)), (Some(10), Some(10))]
        );
    }

    #[test]
    fn null_keyed_probe_rows_are_emitted_padded() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
        let probe = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![None, Some(10)]))],
        )
        .unwrap();

        let r = build_and_probe_probe_outer(vec![vec![int64_batch(&[10])]], vec![probe]);

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(None, None), (Some(10), Some(10))]
        );
    }

    #[test]
    fn an_empty_build_side_pads_every_probe_row() {
        let r =
            build_and_probe_probe_outer(vec![vec![int64_batch(&[])]], vec![int64_batch(&[1, 2])]);

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(Some(1), None), (Some(2), None)]
        );
    }

    #[test]
    fn a_probe_side_larger_than_one_batch_pads_every_miss() {
        let probe_keys: Vec<i64> = (0..crate::RECORD_BATCH_SIZE as i64 * 2 + 5).collect();

        let r = build_and_probe_probe_outer(
            vec![vec![int64_batch(&[7])]],
            vec![int64_batch(&probe_keys)],
        );

        assert_eq!(r.rows(), probe_keys.len());
    }

    /// Every output row's probe key, None for a null, in emission order.
    fn collect_nullable_keys(results: &[RecordBatch]) -> Vec<Option<i64>> {
        results
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn an_anti_join_emits_only_the_probe_rows_nothing_matched() {
        let r = build_and_probe_anti(
            vec![vec![int64_batch(&[20])]],
            vec![int64_batch(&[10, 20, 30])],
        );

        assert_eq!(collect_nullable_keys(&r.batches), vec![Some(10), Some(30)]);
    }

    #[test]
    fn a_probe_row_matching_several_build_rows_is_never_emitted() {
        let r = build_and_probe_anti(
            vec![vec![int64_batch(&[10, 10])]],
            vec![int64_batch(&[10, 5])],
        );

        assert_eq!(collect_nullable_keys(&r.batches), vec![Some(5)]);
    }

    #[test]
    fn every_copy_of_a_repeated_unmatched_probe_row_is_emitted() {
        let r = build_and_probe_anti(vec![vec![int64_batch(&[1])]], vec![int64_batch(&[5, 5])]);

        assert_eq!(collect_nullable_keys(&r.batches), vec![Some(5), Some(5)]);
    }

    #[test]
    fn null_keyed_probe_rows_count_as_unmatched_in_an_anti_join() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
        let probe = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![None, Some(10), Some(30)]))],
        )
        .unwrap();

        let r = build_and_probe_anti(vec![vec![int64_batch(&[10])]], vec![probe]);

        assert_eq!(collect_nullable_keys(&r.batches), vec![None, Some(30)]);
    }

    #[test]
    fn an_empty_build_side_makes_an_anti_join_emit_every_probe_row() {
        let r = build_and_probe_anti(vec![vec![int64_batch(&[])]], vec![int64_batch(&[1, 2])]);

        assert_eq!(collect_nullable_keys(&r.batches), vec![Some(1), Some(2)]);
    }

    #[test]
    fn a_probe_side_larger_than_one_batch_emits_every_anti_miss() {
        let probe_keys: Vec<i64> = (0..crate::RECORD_BATCH_SIZE as i64 * 2 + 5).collect();

        let r = build_and_probe_anti(
            vec![vec![int64_batch(&[7])]],
            vec![int64_batch(&probe_keys)],
        );

        assert_eq!(r.rows(), probe_keys.len() - 1);
    }

    #[test]
    fn a_build_anti_join_emits_only_the_build_rows_nobody_matched() {
        let r = build_and_probe_build_anti(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[20])],
        );

        let mut keys = collect_nullable_keys(&r.batches);
        keys.sort();
        assert_eq!(keys, vec![Some(10), Some(30)]);
    }

    #[test]
    fn a_build_row_matched_twice_is_not_emitted_by_a_build_anti_join() {
        let r = build_and_probe_build_anti(
            vec![vec![int64_batch(&[10, 20])]],
            vec![int64_batch(&[10, 10])],
        );

        assert_eq!(collect_nullable_keys(&r.batches), vec![Some(20)]);
    }

    #[test]
    fn null_keyed_build_rows_come_out_of_a_build_anti_join() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
        let build = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![None, Some(10), Some(30)]))],
        )
        .unwrap();

        let r = build_and_probe_build_anti(vec![vec![build]], vec![int64_batch(&[10])]);

        let mut keys = collect_nullable_keys(&r.batches);
        keys.sort();
        assert_eq!(keys, vec![None, Some(30)]);
    }

    #[test]
    fn a_probe_side_matching_nothing_makes_a_build_anti_join_emit_every_build_row() {
        let r = build_and_probe_build_anti(
            vec![vec![int64_batch(&[1, 2, 3])]],
            vec![int64_batch(&[99])],
        );

        let mut keys = collect_nullable_keys(&r.batches);
        keys.sort();
        assert_eq!(keys, vec![Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn the_build_anti_unmatched_pass_covers_every_build_worker() {
        let r = build_and_probe_build_anti(
            vec![vec![int64_batch(&[1, 2])], vec![int64_batch(&[3, 4])]],
            vec![int64_batch(&[2, 3])],
        );

        let mut keys = collect_nullable_keys(&r.batches);
        keys.sort();
        assert_eq!(keys, vec![Some(1), Some(4)]);
    }

    #[test]
    fn a_build_side_larger_than_one_batch_emits_every_unmatched_build_row() {
        let build_keys: Vec<i64> = (0..crate::RECORD_BATCH_SIZE as i64 * 2 + 5).collect();

        let r = build_and_probe_build_anti(
            vec![vec![int64_batch(&build_keys)]],
            vec![int64_batch(&[7])],
        );

        assert_eq!(r.rows(), build_keys.len() - 1);
    }

    #[test]
    fn a_build_semi_join_emits_only_the_build_rows_somebody_matched() {
        let r = build_and_probe_build_semi(
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[20])],
        );

        assert_eq!(collect_nullable_keys(&r.batches), vec![Some(20)]);
    }

    #[test]
    fn a_build_row_matched_twice_comes_out_of_a_build_semi_join_once() {
        let r = build_and_probe_build_semi(
            vec![vec![int64_batch(&[10, 20])]],
            vec![int64_batch(&[10, 10])],
        );

        assert_eq!(collect_nullable_keys(&r.batches), vec![Some(10)]);
    }

    #[test]
    fn null_keyed_build_rows_stay_out_of_a_build_semi_join() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
        let build = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![None, Some(10), Some(30)]))],
        )
        .unwrap();

        let r = build_and_probe_build_semi(vec![vec![build]], vec![int64_batch(&[10])]);

        assert_eq!(collect_nullable_keys(&r.batches), vec![Some(10)]);
    }

    #[test]
    fn a_probe_side_matching_nothing_makes_a_build_semi_join_emit_nothing() {
        let r = build_and_probe_build_semi(
            vec![vec![int64_batch(&[1, 2, 3])]],
            vec![int64_batch(&[99])],
        );

        assert_eq!(r.rows(), 0);
    }

    #[test]
    fn the_build_semi_scan_covers_every_build_worker() {
        let r = build_and_probe_build_semi(
            vec![vec![int64_batch(&[1, 2])], vec![int64_batch(&[3, 4])]],
            vec![int64_batch(&[2, 3])],
        );

        let mut keys = collect_nullable_keys(&r.batches);
        keys.sort();
        assert_eq!(keys, vec![Some(2), Some(3)]);
    }

    #[test]
    fn a_build_side_larger_than_one_batch_emits_every_matched_build_row() {
        let build_keys: Vec<i64> = (0..crate::RECORD_BATCH_SIZE as i64 * 2 + 5).collect();

        let r = build_and_probe_build_semi(
            vec![vec![int64_batch(&build_keys)]],
            vec![int64_batch(&[7])],
        );

        assert_eq!(collect_nullable_keys(&r.batches), vec![Some(7)]);
    }

    fn pair_key_batch(first_keys: &[i64], second_keys: &[i32]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("k0", DataType::Int64, false),
                Field::new("k1", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int64Array::from(first_keys.to_vec())),
                Arc::new(Int32Array::from(second_keys.to_vec())),
            ],
        )
        .unwrap()
    }

    /// Every output row's probe-side key pair.
    fn collect_probe_key_pairs(results: &[RecordBatch]) -> Vec<(i64, i32)> {
        let mut pairs: Vec<(i64, i32)> = results
            .iter()
            .flat_map(|batch| {
                let first = collect_i64_column(std::slice::from_ref(batch), 0);
                let second = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec();
                first.into_iter().zip(second)
            })
            .collect();
        pairs.sort();
        pairs
    }

    #[test]
    fn a_packed_pair_matches_only_when_both_columns_match() {
        let build = pair_key_batch(&[1, 1, 2], &[10, 20, 30]);
        let probe = pair_key_batch(&[1, 2, 7], &[20, 30, 30]);

        let r = run_join::<PackedKey<(Int64Type, Int32Type)>, false, false, false>(
            super::JoinKind::Inner,
            vec![vec![build]],
            vec![probe],
            None,
            vec![0, 1],
        );

        assert_eq!(collect_probe_key_pairs(&r.batches), vec![(1, 20), (2, 30)]);
    }

    #[test]
    fn a_packed_triple_needs_all_three_columns_equal() {
        let triple_batch = |a: &[i64], b: &[i64], c: &[i32]| {
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("k0", DataType::Int64, false),
                    Field::new("k1", DataType::Int64, false),
                    Field::new("k2", DataType::Int32, false),
                ])),
                vec![
                    Arc::new(Int64Array::from(a.to_vec())),
                    Arc::new(Int64Array::from(b.to_vec())),
                    Arc::new(Int32Array::from(c.to_vec())),
                ],
            )
            .unwrap()
        };
        let build = triple_batch(&[1, 1, 2], &[10, 10, 20], &[100, 101, 200]);
        let probe = triple_batch(&[1, 1, 3], &[10, 10, 30], &[101, 999, 300]);

        let r = run_join::<PackedKey<(Int64Type, Int64Type, Int32Type)>, false, false, false>(
            super::JoinKind::Inner,
            vec![vec![build]],
            vec![probe],
            None,
            vec![0, 1, 2],
        );

        assert_eq!(r.rows(), 1);
        assert_eq!(collect_i64_column(&r.batches, 0), vec![1]);
    }

    #[test]
    fn a_packed_pair_with_a_null_in_either_key_column_never_matches() {
        let nullable_pair_batch = |a: Vec<Option<i64>>, b: Vec<Option<i32>>| {
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("k0", DataType::Int64, true),
                    Field::new("k1", DataType::Int32, true),
                ])),
                vec![Arc::new(Int64Array::from(a)), Arc::new(Int32Array::from(b))],
            )
            .unwrap()
        };
        let build = nullable_pair_batch(vec![Some(1), Some(1)], vec![Some(10), None]);
        let probe =
            nullable_pair_batch(vec![Some(1), Some(1), None], vec![Some(10), None, Some(10)]);

        let r = run_join::<PackedKey<(Int64Type, Int32Type)>, false, false, false>(
            super::JoinKind::Inner,
            vec![vec![build]],
            vec![probe],
            None,
            vec![0, 1],
        );

        assert_eq!(r.rows(), 1);
    }

    fn string_key_batch(keys: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "key",
                DataType::Utf8View,
                false,
            )])),
            vec![Arc::new(StringViewArray::from(keys.to_vec()))],
        )
        .unwrap()
    }

    #[test]
    fn a_dynamic_string_key_joins_on_exact_text() {
        // One key long enough to live outside the view's inline bytes.
        let build = string_key_batch(&["apple", "banana", "a-key-too-long-to-inline"]);
        let probe = string_key_batch(&["banana", "durian", "a-key-too-long-to-inline"]);

        let r = run_join::<DynamicRowKey, false, false, false>(
            super::JoinKind::Inner,
            vec![vec![build]],
            vec![probe],
            None,
            vec![0],
        );

        let mut keys: Vec<String> = r
            .batches
            .iter()
            .flat_map(|batch| {
                let names = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                (0..batch.num_rows()).map(|i| names.value(i).to_string())
            })
            .collect();
        keys.sort();
        assert_eq!(keys, vec!["a-key-too-long-to-inline", "banana"]);
    }

    #[test]
    fn a_dynamic_int_and_string_pair_needs_both_columns_equal() {
        let int_str_batch = |ints: &[i64], names: &[&str]| keyed_names_batch(ints, names);
        let build = int_str_batch(&[1, 1, 2], &["a", "b", "a"]);
        let probe = int_str_batch(&[1, 2, 2], &["b", "b", "a"]);

        let r = run_join::<DynamicRowKey, false, false, false>(
            super::JoinKind::Inner,
            vec![vec![build]],
            vec![probe],
            None,
            vec![0, 1],
        );

        // Only (1, "b") and (2, "a") exist on both sides.
        let mut pairs: Vec<(i64, String)> = r
            .batches
            .iter()
            .flat_map(|batch| {
                let ints = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let names = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                (0..batch.num_rows()).map(|i| (ints.value(i), names.value(i).to_string()))
            })
            .collect();
        pairs.sort();
        assert_eq!(pairs, vec![(1, "b".to_string()), (2, "a".to_string())]);
    }

    #[test]
    fn a_dynamic_key_outer_build_emits_unmatched_rows() {
        let r = run_join::<DynamicRowKey, true, false, false>(
            super::JoinKind::BuildOuter,
            vec![vec![int64_batch(&[10, 20, 30])]],
            vec![int64_batch(&[20])],
            Some(nullable_key_field()),
            vec![0],
        );

        assert_eq!(
            collect_key_pairs(&r.batches),
            vec![(None, Some(10)), (None, Some(30)), (Some(20), Some(20))]
        );
    }

    /// Short build batches become short stored batches with gaps in the row id
    /// space; matched rows must still gather their string columns from the
    /// right chunk. The long value lives outside a view's inline bytes, so it
    /// exercises the buffer-rebasing gather path too.
    #[test]
    fn short_build_row_batches_gather_from_the_right_batch() {
        let build = vec![
            keyed_names_batch(&[1, 2, 3], &["a", "b", "c"]),
            keyed_names_batch(&[4, 5], &["d", "a-value-too-long-to-inline"]),
            keyed_names_batch(&[6, 7, 8, 9], &["f", "g", "h", "i"]),
        ];
        let probe = int64_batch(&[5, 2, 9]);

        let r = build_and_probe(vec![build], vec![probe]);

        let mut pairs: Vec<(i64, String)> = r
            .batches
            .iter()
            .flat_map(|batch| {
                let keys = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let names = batch
                    .column(2)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                (0..batch.num_rows()).map(|i| (keys.value(i), names.value(i).to_string()))
            })
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                (2, "b".to_string()),
                (5, "a-value-too-long-to-inline".to_string()),
                (9, "i".to_string())
            ]
        );
    }
}
