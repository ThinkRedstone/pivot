//! Background compaction of a table's Parquet files.
//!
//! Frequent writes litter a table with small files, and scans pay per file.
//! The [`CompacterHandle`] runs two strategies, split by file size: files
//! below half the configured target are merged as *small files*, files at
//! least half full are candidates for *layout optimization*. Both rewrite
//! their inputs into target-sized outputs and swap them into the table in a
//! single log commit. A row group larger than the target stays one
//! oversized file, since row groups are the smallest independently encoded
//! unit. Each pass tries small files first, then layout.
//!
//! # Small files
//!
//! A partition's small files are merged once their total reaches the merge
//! target, the file target with headroom for compression gains. A pile that
//! reaches the file-count threshold before the byte target is merged once
//! it is balanced: the batch must total at least five times its largest
//! file, each file weighed with a fixed cost so that many tiny files
//! qualify, and the largest file is dropped and the test repeated until the
//! rest passes or nothing useful remains. `COMPACT ... FINAL` drops both
//! guards and merges any two or more small files.
//!
//! # Layout optimization
//!
//! Layout optimization reduces the work left after min/max pruning. A scan
//! can discard a file when its sort-key range misses the predicate, then
//! discard row groups within a retained file by the same test. If many files
//! cover the same key range, file pruning retains all of them and each can
//! contribute row groups whose bounds meet the predicate. Rewriting their
//! rows in sort order cuts the shared range into files with narrower,
//! non-overlapping ranges, so a query usually retains fewer files and fewer
//! row groups.
//!
//! Candidates are files at least half the target size in one partition of a
//! table with a sort key. A pair is measured on the first sort column on
//! which the files are not both fixed at the same value. Their ranges must
//! share more than a single boundary value, and a routine pair qualifies
//! only when more than
//! [`overlap::CONTESTED_ROW_GROUP_PERCENT_BAR`] percent of each file's row
//! groups lie inside the other file's range. Missing row-group bounds count
//! as contested because they cannot prove that the group is outside it.
//!
//! A routine round starts with the file having the most qualifying partners
//! and merges a full clique, meaning that every pair in the group qualifies.
//! The default clique has [`DEFAULT_LAYOUT_CLIQUE_SIZE`] files, six at the
//! default 20 percent bar. Smaller cliques wait for more files instead of
//! causing repeated partial rewrites. The default size also makes each
//! output's range narrow enough that it normally falls below the bar against
//! an unmerged broad file, keeping successive layout scales apart. Files in
//! different partitions are never paired. Neither are files with unusable
//! manifest bounds or no rows, pairs whose ranges are disjoint or merely
//! touch, or pairs fixed at one identical value across the entire sort key.
//! These exclusions apply even to a final sweep.
//!
//! For example, the pinned distribution scenario begins with 1,000 identical
//! target-full files, each spanning 1,000 keys in ten equal row groups. With
//! six-file routine cliques it performs 472 merges and converges to:
//!
//! ```text
//! key-range span (keys)   files
//!                     5     352
//!                     6     512
//!                    28      36
//!                    29      72
//!                   167      16
//!                   168       8
//!                 1,000       4
//! ```
//!
//! Thus 864 files cover only five or six adjacent keys, with progressively
//! fewer files at the wider leftover scales. A narrow query can prune almost
//! all of those files before using row-group bounds inside the few it keeps,
//! instead of visiting all 1,000 original full-range files.
//!
//! `COMPACT ... FINAL` still requires a measurable pair that shares a real
//! key interval, but drops the contested-row-group bar and the full-clique
//! requirement. It takes a seed and its strongest directly overlapping
//! partners, up to the same group-size cap, and keeps merging until no such
//! group remains. This lets a broad file be rewritten with narrower files
//! nested inside its range even when those narrow files do not all overlap
//! one another.
//!
//! # Merging and committing
//!
//! The compacter is **location-agnostic**: candidates come from the table's
//! manifest (paths, sizes, partition values, and stats; no directory scanning), and the merge (read,
//! upload, swap) runs through [`compact_table_files`]. A table under an `s3://`
//! database root compacts through the exact same code path as a local one.
//!
//! The swapped-out inputs are not deleted with the swap: a query that loaded the
//! prior version is still reading them, and the swap's Delta `Remove` actions
//! are their durable tombstones. The objects stay in place until the vacuum
//! sweep ages them out.
//!
//! Merging is **one dataflow** on the dispatch worker pool: the scan stages
//! decode the inputs' row groups, the encode stages repack them into full-size
//! row groups, and the upload stage writes each merged file over the shared
//! io_uring ring, the same operators an INSERT uses. The compacter's own task
//! only drives that dataflow (from a blocking thread) and, once the uploads
//! land, commits the swap.
//!
//! Crash safety comes from the table log, not from ordering tricks: a file is
//! part of the table iff the current log version lists it. The merged files are
//! uploaded, then swapped in for the inputs in one Delta commit (labelled a
//! rearrangement, not a data change); a crash before the commit leaves only
//! *orphan* objects (unlogged merged files), never a double read.
//!
//! Merging always writes into the table's own directory, so compacting a file
//! the table adopted (`with_pre_existing_parquets`, which leaves the file where its owner
//! put it) copies those rows under the database root. The adopted original is
//! dropped from the log but never deleted: it is not the table's to delete, and
//! the vacuum sweep only reclaims files under the table's own location. A table
//! that adopts a directory of small files and then compacts them therefore ends
//! up storing that data twice, once in the user's directory and once in its own.
//! That is the price of never writing into a directory the user owns.
//!
//! A [`PivotlakeDatastore`] owns one [`CompacterHandle`] actor. Configured timer ticks and
//! explicit `COMPACT` commands share its queue, so their rewrites never race in
//! one process. A timed round reloads each table to its latest log version;
//! commands carry the table snapshot captured by their query transaction. The
//! actor stops before the datastore's dispatch pool shuts down.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

use crate::manifest::DeltaFileEntry;
use crate::{CatalogTable, FileRef, PivotlakeDatastore, TableFile, scalar_values_equal};
use object_storage::ObjectPath;
use planner::catalog::SchemaQualifiedTableName;

mod overlap;

/// Rows per row group in a merged file.
const ROW_GROUP_ROWS: usize = 128 * 1024;

/// Default compacted-file size target. Files smaller than half this size enter
/// small-file compaction; files at least half this size are full enough to
/// enter layout optimization instead.
pub const DEFAULT_COMPACT_BYTES: u64 = 64 * 1024 * 1024;

/// Default headroom for compression gains: a group whose input bytes only just
/// reach the desired output size can encode to another undersized file.
const DEFAULT_MERGE_TARGET_MULTIPLIER: f64 = 1.3;

/// Default count at which a sub-target pile may take the balance-check path.
pub const DEFAULT_MIN_FILES_TO_MERGE: usize = 100;

/// Per-file fixed cost in the balance test. This prevents many similarly small
/// files from being rejected merely because their byte total is low.
const FIXED_FILE_COST_BYTES: f64 = (5 * 1024 * 1024) as f64;
const MIN_BALANCE_RATIO: f64 = 5.0;

/// Default cadence for re-checking the tables' logs. Candidates only change
/// when a flush commits a new version, so seconds-scale is plenty.
pub const DEFAULT_COMPACT_POLL: Duration = Duration::from_secs(30);

/// Default number of merges a datastore runs at once. A merge holds its whole
/// input decoded on the memory ring until its output is encoded, and the
/// decoded form is many times the input (a merge of 300 MB of compressed
/// OTLP logs with variant columns pinned about 6 GB). Several merges landing
/// together can take the entire ring, and then every read that needs a
/// buffer, the merges' own footer loads and the ingest included, fails with
/// nothing left to evict. One merge at a time keeps that footprint bounded;
/// a deployment that has the memory to spare raises it in its config.
pub const DEFAULT_COMPACT_PARALLELISM: usize = 1;

/// Default size of the clique a routine round merges, and the cap on a
/// final sweep's connected group: one more than the bar's reciprocal, the
/// narrowest merge that keeps the layout's scales apart. A clique merge's outputs each span about `1 / clique_size` of the
/// inputs' range, so a leftover input holds at most `total / width` of its
/// row groups inside any output, below the bar's `total * percent / 100 + 1`;
/// the extra one is a full row group of slack against cut drift. Narrower
/// merges let leftovers and outputs merge across scales (files of four row
/// groups need only one contested group and break at width four). Wider
/// merges settle rows in fewer rewrites but hold every input's decoded rows
/// in memory.
pub const DEFAULT_LAYOUT_CLIQUE_SIZE: usize = 100 / overlap::CONTESTED_ROW_GROUP_PERCENT_BAR + 1;

/// Default cadence for reloading the tables from the store. This bounds how
/// stale a query's view of externally committed data in a shared remote store
/// is, so it trades freshness against the store's listing traffic.
pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(30);

/// Default accumulated input size for a small-file merge.
pub fn default_merge_target_bytes(target_bytes: u64) -> u64 {
    (target_bytes as f64 * DEFAULT_MERGE_TARGET_MULTIPLIER) as u64
}

/// What background maintenance a [`PivotlakeDatastore`] runs for itself once opened.
/// The datastore spawns its own tasks from this on the ambient tokio runtime;
/// they stop when its dispatch pool begins shutting down.
#[derive(Clone)]
pub struct MaintenanceConfig {
    /// How often the datastore reloads its tables from the store (new Delta
    /// versions and new files' footers), so externally committed data in a
    /// shared remote store becomes visible to queries.
    pub refresh_interval: Duration,
    /// Compaction settings, or `None` to run no compacter (a read-only server
    /// or a remote-store deployment where another process owns compaction).
    pub compaction: Option<CompactionConfig>,
    /// Vacuum settings, or `None` to run no vacuumer (a read-only server, or a
    /// remote-store deployment where another process owns physical cleanup).
    /// Deletes tombstoned data files and superseded commit JSONs past their
    /// retention.
    pub vacuum: Option<crate::vacuum::VacuumConfig>,
}

/// Tuning for a datastore's self-managed compaction loop.
#[derive(Clone)]
pub struct CompactionConfig {
    /// Compaction output target. Half of it is the boundary between small-file
    /// and layout compaction; a single encoded row group may exceed it (see
    /// [`DEFAULT_COMPACT_BYTES`]).
    pub target_bytes: u64,
    /// Accumulated small-file bytes that trigger a merge.
    pub merge_target_bytes: u64,
    /// Count at which a sub-target pile may take the balance-check path.
    pub min_files: usize,
    /// How many merges a round may have in flight at once. Selection stays
    /// serial whatever this is; only the rewrites overlap.
    pub max_concurrent_merges: usize,
    /// Files one layout merge rewrites together, at least two; see
    /// [`DEFAULT_LAYOUT_CLIQUE_SIZE`].
    pub layout_clique_size: usize,
    /// How often to re-check the tables' logs for newly-accumulated files.
    pub poll_interval: Duration,
}

/// Handle to the datastore's single compaction actor. Both periodic maintenance
/// and explicit `COMPACT` statements run through that actor, so two compaction
/// rounds in one process never overlap. Within one round, selection is still
/// serial while the rewrites it starts run concurrently.
#[derive(Clone)]
pub struct CompacterHandle {
    commands: mpsc::UnboundedSender<CompactionCommand>,
}

/// Which pass a round runs: the routine one, which keeps every guard, or
/// the final sweep, which drops them and settles the table.
#[derive(Clone, Copy, Eq, PartialEq)]
enum CompactionPass {
    Routine,
    Final,
}

struct CompactionCommand {
    name: SchemaQualifiedTableName,
    table: CatalogTable,
    pass: CompactionPass,
    result: oneshot::Sender<crate::Result<u64>>,
}

pub(crate) struct CompacterActor {
    /// Output-size target; half of it is the small/layout boundary.
    target_bytes: u64,
    merge_target_bytes: u64,
    min_files: usize,
    /// How many merges a round may have in flight at once.
    max_concurrent_merges: usize,
    /// Files one layout merge rewrites together, at least two; see
    /// [`DEFAULT_LAYOUT_CLIQUE_SIZE`].
    layout_clique_size: usize,
    /// How often to re-check the tables' logs, or `None` when only explicit
    /// commands drive this actor.
    poll_interval: Option<Duration>,
    datastore: Arc<PivotlakeDatastore>,
    commands: mpsc::UnboundedReceiver<CompactionCommand>,
    /// Each table's pair overlaps, remembered from round to round so a
    /// pass only measures the files it has not seen before (see
    /// [`overlap::OverlapCache`]).
    overlap_caches: HashMap<uuid::Uuid, overlap::OverlapCache>,
}

impl CompacterHandle {
    pub(crate) fn new(
        target_bytes: u64,
        merge_target_bytes: u64,
        min_files: usize,
        max_concurrent_merges: usize,
        layout_clique_size: usize,
        poll_interval: Option<Duration>,
        datastore: Arc<PivotlakeDatastore>,
    ) -> (Self, CompacterActor) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            Self { commands: sender },
            CompacterActor {
                target_bytes,
                merge_target_bytes,
                min_files: min_files.max(2),
                max_concurrent_merges: max_concurrent_merges.max(1),
                layout_clique_size: layout_clique_size.max(2),
                poll_interval,
                datastore,
                commands: receiver,
                overlap_caches: HashMap::new(),
            },
        )
    }

    /// Enqueue one table from the caller's transaction snapshot; the round
    /// refreshes it to the latest version as it begins and keeps merging until
    /// nothing qualifies. `FINAL` drops the small-file guards and the
    /// contested bar and compacts each partition to one contiguous flow, see
    /// the module docs.
    pub(crate) async fn compact(
        &self,
        name: SchemaQualifiedTableName,
        table: CatalogTable,
        final_sweep: bool,
    ) -> crate::Result<u64> {
        let (result, answer) = oneshot::channel();
        let pass = if final_sweep {
            CompactionPass::Final
        } else {
            CompactionPass::Routine
        };
        self.commands
            .send(CompactionCommand {
                name,
                table,
                pass,
                result,
            })
            .map_err(|_| crate::Error::CompacterStopped)?;
        answer.await.map_err(|_| crate::Error::CompacterStopped)?
    }
}

impl CompacterActor {
    /// Run timer ticks and commands serially. The first configured tick fires
    /// immediately, preserving the background compacter's startup sweep, and
    /// each later one fires the poll interval after the previous round
    /// finished: a round runs until its tables have no candidate left, however
    /// long that takes, and the interval is the pause between rounds.
    pub(crate) async fn run(mut self) {
        let mut tick = self.poll_interval.map(|d| {
            let mut t = tokio::time::interval(d);
            t.set_missed_tick_behavior(MissedTickBehavior::Skip);
            t
        });

        loop {
            tokio::select! {
                biased;
                command = self.commands.recv() => match command {
                    Some(command) => {
                        let res = self
                            .compact_table(&command.name, command.table, command.pass)
                            .await;
                        let _ = command.result.send(res.map(|(_, count)| count));
                    },
                    None => return,
                },
                _ = async {
                    match tick.as_mut() {
                        Some(t) => t.tick().await,
                        None => std::future::pending().await,
                    }
                } => {
                    self.compact_all().await;
                    if let Some(t) = tick.as_mut() {
                        t.reset();
                    }
                },
            }
        }
    }

    async fn compact_all(&mut self) {
        let tables = self.datastore.tables();
        let live: HashSet<uuid::Uuid> = tables.iter().map(|(_, table)| table.id()).collect();
        self.overlap_caches.retain(|id, _| live.contains(id));
        for (name, table) in tables {
            if let Err(error) = self
                .compact_table(&name, table, CompactionPass::Routine)
                .await
            {
                error!(table = %name, error = %error, "compaction round failed");
            }
        }
    }

    /// Compact `table` until it has no candidate left: the round refreshes the
    /// table to the latest committed version once as it begins, then each pass
    /// starts a small-file group if one qualifies, else the best overlapping
    /// large-file group. Up to `max_concurrent_merges` of those rewrites run at
    /// once, each holding its inputs reserved until it commits, so selection
    /// works around the merges already in flight. A pass with nothing to start
    /// and nothing in flight ends the round. Returns the table as it stands and
    /// the number of passes.
    ///
    /// The round works off its one refreshed copy rather than reloading the
    /// log per pass: each of its own merges is folded in as it commits, so
    /// files landing meanwhile (an INSERT in this process, or another writer)
    /// wait for the next round. Only a commit conflict, another writer having
    /// swapped a selected file out first, reloads the copy, and the round
    /// carries on from what it finds.
    async fn compact_table(
        &mut self,
        name: &SchemaQualifiedTableName,
        mut table: CatalogTable,
        pass: CompactionPass,
    ) -> crate::Result<(CatalogTable, u64)> {
        let mut passes = 0;
        let mut in_flight = JoinSet::new();
        let mut reserved = HashSet::new();
        table.refresh()?;
        loop {
            passes += 1;
            let id = table.id();

            while in_flight.len() < self.max_concurrent_merges {
                let Some((inputs, kind)) = self.next_batch(&table, pass, &reserved) else {
                    break;
                };
                reserved.extend(inputs.iter().map(|file| file.path.clone()));
                in_flight.spawn(compact_batch(
                    Arc::clone(&self.datastore),
                    name.clone(),
                    id,
                    inputs,
                    kind,
                    self.target_bytes,
                ));
            }

            if in_flight.is_empty() {
                return Ok((table, passes));
            }

            // Every exit from here on drains first: a merge body runs on the
            // blocking pool, which a dropped handle does not stop.
            let (reserved_paths, outcome) = match in_flight.join_next().await {
                Some(Ok(merge)) => merge,
                Some(Err(error)) => {
                    drain_merges(&mut in_flight).await;
                    return Err(crate::Error::CompactionJobPanicked(error.to_string()));
                }
                None => unreachable!("the set is not empty"),
            };
            for path in &reserved_paths {
                reserved.remove(path);
            }
            match outcome {
                Ok(outputs) => table.apply_file_swap(&reserved_paths, outputs),
                Err(crate::Error::CommitConflict { .. }) => {
                    // The merge's outputs are already deleted; reload the copy
                    // to see what the conflicting writer left and keep going.
                    if let Err(error) = table.refresh() {
                        drain_merges(&mut in_flight).await;
                        return Err(error);
                    }
                }
                Err(error) => {
                    drain_merges(&mut in_flight).await;
                    return Err(error);
                }
            }
        }
    }

    /// The next merge a pass can start: a small-file group if one qualifies,
    /// else the best overlapping large-file group grown from the most
    /// contested pair. `reserved` holds the files the merges already in
    /// flight are rewriting, which no second merge may claim.
    fn next_batch(
        &mut self,
        table: &CatalogTable,
        pass: CompactionPass,
        reserved: &HashSet<ObjectPath>,
    ) -> Option<(Vec<FileRef>, MergeKind)> {
        if let Some(inputs) = self.next_small_batch(table, pass, reserved) {
            return Some((inputs, MergeKind::SmallFiles));
        }
        let inputs = self.next_layout_optimization(table, reserved, pass)?;
        Some((inputs, MergeKind::LayoutOptimization))
    }

    /// Pick one group of files below the target size from a single partition.
    /// A normal merge stops as soon as its input reaches the configured merge
    /// target. A pile that reaches the configured file-count threshold first
    /// uses the balance test, repeatedly discarding its largest file until the
    /// remainder is balanced enough to merge or no useful group remains.
    fn next_small_batch(
        &self,
        table: &CatalogTable,
        pass: CompactionPass,
        reserved: &HashSet<ObjectPath>,
    ) -> Option<Vec<FileRef>> {
        let mut by_partition: Vec<(Option<crate::PartitionValues>, Vec<&DeltaFileEntry>)> =
            Vec::new();
        for entry in table.file_entries() {
            if !is_small_file(entry.file.size, self.target_bytes)
                || reserved.contains(&entry.file.path)
            {
                continue;
            }
            match by_partition
                .iter_mut()
                .find(|(partition, _)| partition_values_equal(partition, &entry.partition))
            {
                Some((_, files)) => files.push(entry),
                None => by_partition.push((entry.partition.clone(), vec![entry])),
            }
        }

        by_partition.into_iter().find_map(|(_, files)| match pass {
            CompactionPass::Routine => {
                small_file_batch(files, self.merge_target_bytes, self.min_files)
            }
            CompactionPass::Final => (files.len() >= 2)
                .then(|| files.into_iter().map(|entry| entry.file.clone()).collect()),
        })
    }

    /// The next group of at-least-half-full files one merge should rewrite
    /// together, chosen within one partition by the table's
    /// [`overlap::OverlapCache`]: a full clique in a routine round, any
    /// connected group in a final sweep. Reserved files stay in view so
    /// their remembered overlaps survive a failed merge; selection passes
    /// their pairs over.
    fn next_layout_optimization(
        &mut self,
        table: &CatalogTable,
        reserved: &HashSet<ObjectPath>,
        pass: CompactionPass,
    ) -> Option<Vec<FileRef>> {
        table.sort_by().first()?;

        let mut by_partition: Vec<(
            Option<crate::PartitionValues>,
            Vec<overlap::LayoutCandidate<'_>>,
        )> = Vec::new();
        for file in table.files() {
            if is_small_file(file.entry.file.size, self.target_bytes) {
                continue;
            }
            let candidate = overlap::LayoutCandidate::from_table_file(file, table.sort_by());
            match by_partition
                .iter_mut()
                .find(|(partition, _)| partition_values_equal(partition, &file.entry.partition))
            {
                Some((_, files)) => files.push(candidate),
                None => by_partition.push((file.entry.partition.clone(), vec![candidate])),
            }
        }

        let partitions: Vec<&[overlap::LayoutCandidate<'_>]> = by_partition
            .iter()
            .map(|(_, files)| files.as_slice())
            .collect();
        let cache = self.overlap_caches.entry(table.id()).or_default();
        cache.observe(&partitions, table.sort_by());
        let group = cache.select_group(reserved, pass, self.layout_clique_size)?;
        let files_by_path: HashMap<&str, &FileRef> = by_partition
            .iter()
            .flat_map(|(_, candidates)| candidates.iter())
            .map(|candidate| (candidate.entry.file.path.as_str(), &candidate.entry.file))
            .collect();
        Some(
            group
                .iter()
                .map(|path| files_by_path[path.as_str()].clone())
                .collect(),
        )
    }
}

/// Rewrite `inputs` into target-sized outputs and swap them into the table in
/// one commit. Returns the input paths beside the outcome so the round can
/// release their reservations however the merge went; a committed merge's
/// outcome carries the output files for the round to fold into its table copy.
async fn compact_batch(
    datastore: Arc<PivotlakeDatastore>,
    name: SchemaQualifiedTableName,
    id: uuid::Uuid,
    inputs: Vec<FileRef>,
    kind: MergeKind,
    target_bytes: u64,
) -> (Vec<ObjectPath>, crate::Result<Vec<TableFile>>) {
    let reserved_paths: Vec<ObjectPath> = inputs.iter().map(|file| file.path.clone()).collect();
    let input_sizes: Vec<u64> = inputs.iter().map(|file| file.size).collect();
    // `spawn_blocking` needs a `'static` closure, so the merge takes the
    // datastore handle rather than a borrow of it.
    let merged = match tokio::task::spawn_blocking(move || {
        compact_table_files(&datastore, id, &inputs, ROW_GROUP_ROWS, target_bytes)
    })
    .await
    {
        Ok(Ok(merged)) => merged,
        Ok(Err(error)) => return (reserved_paths, Err(error)),
        Err(error) => {
            return (
                reserved_paths,
                Err(crate::Error::CompactionJobPanicked(error.to_string())),
            );
        }
    };
    let output_sizes: Vec<u64> = merged.iter().map(|file| file.file_ref().size).collect();
    info!(
        table = %name,
        kind = kind.label(),
        inputs = input_sizes.len(),
        input_sizes = ?input_sizes,
        input_bytes = input_sizes.iter().sum::<u64>(),
        outputs = output_sizes.len(),
        output_sizes = ?output_sizes,
        output_bytes = output_sizes.iter().sum::<u64>(),
        "compacted batch"
    );
    (reserved_paths, Ok(merged))
}

/// Await the merges still in flight, reporting rather than propagating what
/// they return: a merge body runs on the blocking pool, where dropping its
/// handle does not stop it, so a round that is giving up still waits for its
/// own rewrites to finish before the next one selects.
async fn drain_merges(in_flight: &mut JoinSet<(Vec<ObjectPath>, crate::Result<Vec<TableFile>>)>) {
    while let Some(joined) = in_flight.join_next().await {
        match joined {
            Ok((_, Err(error))) => warn!(error = %error, "compaction merge failed"),
            Err(error) => warn!(error = %error, "compaction merge panicked"),
            Ok((_, Ok(_))) => {}
        }
    }
}

#[derive(Clone, Copy)]
enum MergeKind {
    SmallFiles,
    LayoutOptimization,
}

impl MergeKind {
    fn label(self) -> &'static str {
        match self {
            Self::SmallFiles => "small files",
            Self::LayoutOptimization => "layout optimization",
        }
    }
}

/// Integer file sizes below the mathematical half-target are genuinely small.
/// `div_ceil` makes a five-byte target classify sizes 0, 1, and 2 as small,
/// while an exact half is always full enough for layout work.
fn is_small_file(size: u64, target_bytes: u64) -> bool {
    size < target_bytes.div_ceil(2)
}

fn small_file_batch(
    files: Vec<&DeltaFileEntry>,
    merge_target_bytes: u64,
    min_files: usize,
) -> Option<Vec<FileRef>> {
    let mut total = 0u64;
    let mut batch = Vec::with_capacity(files.len());
    for file in files {
        total += file.file.size;
        batch.push(file);
        if batch.len() >= 2 && total >= merge_target_bytes {
            return Some(batch.into_iter().map(|entry| entry.file.clone()).collect());
        }
    }

    if batch.len() < min_files.max(2) {
        return None;
    }

    // A few disproportionately large files can make the whole pile a poor
    // merge. Remove the current largest and run the same balance test again on
    // everything that remains; the count threshold is only the entry condition.
    while batch.len() >= 2 {
        let largest = batch
            .iter()
            .enumerate()
            .max_by_key(|(_, entry)| entry.file.size)
            .map(|(index, _)| index)
            .expect("a non-empty batch has a largest file");
        let largest_size = batch[largest].file.size;
        let ratio = (total as f64 + FIXED_FILE_COST_BYTES * batch.len() as f64)
            / (largest_size as f64 + FIXED_FILE_COST_BYTES);
        if ratio >= MIN_BALANCE_RATIO {
            return Some(batch.into_iter().map(|entry| entry.file.clone()).collect());
        }
        total -= largest_size;
        batch.swap_remove(largest);
    }
    None
}

/// Merge `inputs` of the table with identity `id` into fresh target-sized files
/// and swap them in for the inputs, in one log commit labelled a rearrangement.
/// `inputs` must share one partition tuple; the caller batches them so. Returns
/// the merged files with their footers in hand, so a caller holding its own
/// table copy can fold the swap in without reloading the log.
///
/// The merge itself (decode, re-encode, upload) runs off a plain read copy of
/// the table with no commit lock held, so it never blocks a concurrent INSERT's
/// commit; only the swap that follows takes it. If that swap fails for any
/// reason, including another writer having swapped the inputs out first, the
/// uncommitted merge outputs are deleted before the error is returned.
///
/// `max_output_file_size` is a compressed row-group body target. Parquet
/// headers and footers are excluded, and an individually oversized row group
/// is emitted alone.
///
/// Blocking store I/O, so callers run it on the blocking pool.
pub fn compact_table_files(
    datastore: &PivotlakeDatastore,
    id: uuid::Uuid,
    inputs: &[FileRef],
    target_rows_per_group: usize,
    max_output_file_size: u64,
) -> Result<Vec<TableFile>, crate::Error> {
    let table = datastore
        .table_handle_by_id(&id)
        .ok_or_else(|| crate::Error::TableNotFound(id.to_string()))?;
    let merged = table.merge_files(inputs, target_rows_per_group, max_output_file_size)?;

    let outputs = merged.clone();
    let removed: Vec<ObjectPath> = inputs.iter().map(|file| file.path.clone()).collect();
    if let Err(error) = datastore.commit_to_table(id, &removed, merged, false) {
        for file in &outputs {
            if let Err(cleanup_error) = table.delete_data_file(&file.file_ref().path) {
                warn!(
                    commit_error = %error,
                    cleanup_error = %cleanup_error,
                    file = %file.file_ref().path,
                    "compaction: deleting output after commit failure failed (orphan left)"
                );
            }
        }
        return Err(error);
    }
    Ok(outputs)
}

fn partition_values_equal(
    left: &Option<crate::PartitionValues>,
    right: &Option<crate::PartitionValues>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => scalar_values_equal(left, right),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PivotlakeDatastore;
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
    use parquet::arrow::ArrowWriter;
    use parquet_engine::ParquetTable;
    use planner::catalog::SchemaQualifiedTableName;
    use std::path::Path;

    const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

    fn file_entry(path: &str, size: u64) -> DeltaFileEntry {
        DeltaFileEntry::new(FileRef {
            path: ObjectPath::new(path),
            size,
        })
    }

    #[test]
    fn only_files_strictly_below_half_target_are_small() {
        assert!(is_small_file(49, 100));
        assert!(!is_small_file(50, 100));
        assert!(!is_small_file(80, 100));

        assert!(is_small_file(2, 5));
        assert!(!is_small_file(3, 5));
    }

    #[test]
    fn small_files_merge_at_configured_target() {
        let files = [file_entry("a", 70), file_entry("b", 59), file_entry("c", 1)];

        let batch = small_file_batch(files.iter().collect(), 130, 100).unwrap();

        assert_eq!(
            batch.iter().map(|file| file.size).collect::<Vec<_>>(),
            [70, 59, 1]
        );
    }

    #[test]
    fn small_file_fallback_removes_largest_until_balanced() {
        const MIB: u64 = 1024 * 1024;
        let mut files = vec![file_entry("outlier", 500 * MIB)];
        files.extend((0..99).map(|index| file_entry(&format!("small-{index}"), MIB)));

        let batch = small_file_batch(files.iter().collect(), 2 * 1024 * MIB, 100).unwrap();

        assert_eq!(batch.len(), 99, "the check continues below 100 files");
        assert!(batch.iter().all(|file| file.path.as_str() != "outlier"));
    }

    #[test]
    fn small_file_fallback_uses_configured_minimum() {
        let files: Vec<_> = (0..5)
            .map(|index| file_entry(&format!("small-{index}"), 1))
            .collect();

        let allowed = small_file_batch(files.iter().collect(), 100, 5);
        let waiting = small_file_batch(files.iter().collect(), 100, 6);

        assert_eq!(allowed.unwrap().len(), 5);
        assert!(waiting.is_none());
    }

    /// Write `values` as a small SNAPPY Parquet file (single Int64 `Timestamp`
    /// column) into `dir` under `file_name`. A table created over `dir` picks it
    /// up as a small input for the compacter to merge.
    fn write_parquet_file(dir: &Path, file_name: &str, values: Vec<i64>) {
        let rows = values.len().max(1);
        write_parquet_file_with_row_group_size(dir, file_name, values, rows);
    }

    /// A file whose rows are cut into row groups of `max_row_group_size`, so a
    /// test file of a few rows still carries the many row groups that layout
    /// selection counts.
    fn write_parquet_file_with_row_group_size(
        dir: &Path,
        file_name: &str,
        values: Vec<i64>,
        max_row_group_size: usize,
    ) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "Timestamp",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(values)) as _],
        )
        .unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        // SNAPPY, like every pivot-written file -- the decompressor expects it.
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .set_max_row_group_row_count(Some(max_row_group_size))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// A sorted range with incompressible non-key payloads. Two seeds write
    /// the same key range at almost the same byte size while retaining distinct
    /// rows, which models two 80%-full overlapping files without footer bytes
    /// dominating the test.
    fn write_wide_parquet_file(dir: &Path, file_name: &str, rows: usize, seed: u64) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("Timestamp", DataType::Int64, false),
            Field::new("Payload0", DataType::Int64, false),
            Field::new("Payload1", DataType::Int64, false),
            Field::new("Payload2", DataType::Int64, false),
            Field::new("Payload3", DataType::Int64, false),
        ]));
        let keys: Vec<i64> = (1..=rows as i64).collect();
        let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(keys))];
        for column in 0..4 {
            let payload: Vec<i64> = (0..rows)
                .map(|row| {
                    let mut value = seed ^ row as u64 ^ ((column as u64) << 48);
                    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    value ^= value >> 31;
                    value as i64
                })
                .collect();
            columns.push(Arc::new(Int64Array::from(payload)));
        }
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn write_partitioned_parquet_file(
        dir: &Path,
        file_name: &str,
        timestamps: Vec<i64>,
        partition: i64,
    ) {
        let row_count = timestamps.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("Timestamp", DataType::Int64, false),
            Field::new("Partition", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps)) as _,
                Arc::new(Int64Array::from(vec![partition; row_count])) as _,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// `CREATE TABLE <name> (Timestamp Int64) WITH (with_pre_existing_parquets = dir)`
    /// against `datastore`, the way the server would run it -- or, when `dir` is
    /// `None`, an empty table adopting nothing.
    fn create_table(
        datastore: &Arc<PivotlakeDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        dir: Option<&Path>,
        sorted: bool,
    ) {
        use catalog::datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let mut options = match dir {
            Some(dir) => std::collections::HashMap::from([(
                "with_pre_existing_parquets".to_string(),
                dir.to_str().unwrap().to_string(),
            )]),
            None => std::collections::HashMap::new(),
        };
        if sorted {
            options.insert("sort_by".to_string(), "Timestamp".to_string());
        }
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![Column {
                name: "Timestamp".to_string(),
                col_type: planner::types::Type::Int64,
            }],
            options,
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatcher)
            .unwrap()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
    }

    fn create_wide_sorted_table(
        datastore: &Arc<PivotlakeDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        dir: &Path,
    ) {
        use catalog::datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![
                Column {
                    name: "Timestamp".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload0".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload1".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload2".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Payload3".to_string(),
                    col_type: planner::types::Type::Int64,
                },
            ],
            options: std::collections::HashMap::from([
                (
                    "with_pre_existing_parquets".to_string(),
                    dir.to_str().unwrap().to_string(),
                ),
                ("sort_by".to_string(), "Timestamp".to_string()),
            ]),
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatcher)
            .unwrap()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
    }

    /// Create an empty table partitioned by its `Partition` column.
    fn create_partitioned_table(
        datastore: &Arc<PivotlakeDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
    ) {
        use catalog::datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![
                Column {
                    name: "Timestamp".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Partition".to_string(),
                    col_type: planner::types::Type::Int64,
                },
            ],
            options: std::collections::HashMap::from([(
                "partition_by".to_string(),
                "Partition".to_string(),
            )]),
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatcher)
            .unwrap()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
    }

    fn partition_value(value: i64) -> crate::PartitionValues {
        std::collections::HashMap::from([(
            "Partition".to_string(),
            Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef),
        )])
    }

    /// Refresh `name` to the latest committed manifest (a writer evolves a
    /// cloned-out handle, so the datastore's own copy lags until a refresh), then
    /// return its current row groups.
    fn fresh_parquet(datastore: &PivotlakeDatastore, name: &str) -> Arc<ParquetTable> {
        let mut table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
            .expect("table exists");
        table.refresh().expect("manifest reload");
        table.build_scan_view(&[], &[]).expect("build scan view")
    }

    /// Drive one full compaction sweep to completion on a temporary runtime.
    fn run_one_sweep(target_bytes: u64, datastore: Arc<PivotlakeDatastore>) {
        run_one_sweep_with(
            target_bytes,
            DEFAULT_COMPACT_PARALLELISM,
            DEFAULT_LAYOUT_CLIQUE_SIZE,
            datastore,
        )
    }

    /// [`run_one_sweep`] with layout cliques kept to pairs, for scenarios
    /// where two files are the full clique.
    fn run_one_sweep_with_pair_layout(target_bytes: u64, datastore: Arc<PivotlakeDatastore>) {
        run_one_sweep_with(target_bytes, DEFAULT_COMPACT_PARALLELISM, 2, datastore)
    }

    /// [`run_one_sweep`] with the merges in flight and the layout merge width
    /// spelled out.
    fn run_one_sweep_with(
        target_bytes: u64,
        max_concurrent_merges: usize,
        layout_clique_size: usize,
        datastore: Arc<PivotlakeDatastore>,
    ) {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let tables = datastore.tables();
                let (compacter, actor) = CompacterHandle::new(
                    target_bytes,
                    default_merge_target_bytes(target_bytes),
                    DEFAULT_MIN_FILES_TO_MERGE,
                    max_concurrent_merges,
                    layout_clique_size,
                    None,
                    datastore,
                );
                let actor = tokio::spawn(actor.run());
                for (name, table) in tables {
                    compacter.compact(name, table, false).await.unwrap();
                }
                actor.abort();
            });
    }

    /// The size knobs one compaction command runs under.
    struct CompacterSizes {
        target_bytes: u64,
        merge_target_bytes: u64,
        min_files: usize,
        layout_clique_size: usize,
    }

    impl CompacterSizes {
        /// The defaults for `target_bytes`.
        fn defaults(target_bytes: u64) -> Self {
            Self {
                target_bytes,
                merge_target_bytes: default_merge_target_bytes(target_bytes),
                min_files: DEFAULT_MIN_FILES_TO_MERGE,
                layout_clique_size: DEFAULT_LAYOUT_CLIQUE_SIZE,
            }
        }

        /// The defaults with the small-file strategy switched off, so a
        /// scenario exercises layout optimization alone.
        fn without_small_files(target_bytes: u64) -> Self {
            Self {
                merge_target_bytes: u64::MAX,
                min_files: usize::MAX,
                ..Self::defaults(target_bytes)
            }
        }
    }

    fn run_command(
        sizes: CompacterSizes,
        datastore: Arc<PivotlakeDatastore>,
        name: SchemaQualifiedTableName,
        table: CatalogTable,
        pass: CompactionPass,
    ) -> u64 {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let (compacter, actor) = CompacterHandle::new(
                    sizes.target_bytes,
                    sizes.merge_target_bytes,
                    sizes.min_files,
                    DEFAULT_COMPACT_PARALLELISM,
                    sizes.layout_clique_size,
                    None,
                    datastore,
                );
                let actor = tokio::spawn(actor.run());
                let sweeps = compacter
                    .compact(name, table, pass == CompactionPass::Final)
                    .await
                    .unwrap();
                actor.abort();
                sweeps
            })
    }

    /// The command's snapshot is refreshed before selecting, so a file
    /// committed after it was captured still joins the merge.
    #[test]
    fn command_refreshes_the_supplied_table_snapshot() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_parquet_file(&adopted, "a.parquet", vec![1]);
        write_parquet_file(&adopted, "b.parquet", vec![2]);
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            false,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let captured = datastore.table_handle(&name).unwrap();
        // Each file is small (under half the target), two of them stay under
        // the merge target (1.3 times it), three of them reach it: the merge
        // only happens once the refresh brings `c.parquet` in.
        let target = captured.file_refs()[0].size * 22 / 10;
        write_parquet_file(&adopted, "c.parquet", vec![3]);
        let mut latest = captured.clone();
        latest
            .append_data_file(
                ObjectPath::new("c.parquet"),
                &std::fs::read(adopted.join("c.parquet")).unwrap(),
                None,
            )
            .unwrap();
        datastore.refresh_from_store().unwrap();

        let sweeps = run_command(
            CompacterSizes::defaults(target),
            datastore.clone(),
            name.clone(),
            captured,
            CompactionPass::Routine,
        );

        assert_eq!(sweeps, 2, "one merging pass and one that finds nothing");
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 1);
        dispatch.exit();
    }

    /// A round does not stop at one pair: it keeps merging, folding each
    /// committed merge into its view, until no large pair overlaps any more.
    #[test]
    fn a_round_keeps_merging_until_no_pair_overlaps() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![1_000, 1_100]);
        write_parquet_file(&adopted_dir, "d.parquet", vec![1_000, 1_100]);
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let files = datastore.table_files(&name).unwrap();
        let target = files.iter().map(|file| file.size).min().unwrap();

        run_one_sweep_with_pair_layout(target, datastore.clone());

        let files = datastore.table_files(&name).unwrap();
        assert!(
            files.iter().all(|file| !matches!(
                file.path.name(),
                "a.parquet" | "b.parquet" | "c.parquet" | "d.parquet"
            )),
            "both overlapping pairs were rewritten in one round: {files:?}"
        );
        dispatch.exit();
    }

    /// Three overlapping pairs are rewritten with three merges in flight. The
    /// reservations keep every merge off the others' inputs, so all swaps
    /// commit and every row survives.
    #[test]
    fn concurrent_merges_never_share_an_input() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        for (file, rows) in [
            ("a.parquet", vec![0, 100]),
            ("b.parquet", vec![0, 100]),
            ("c.parquet", vec![1_000, 1_100]),
            ("d.parquet", vec![1_000, 1_100]),
            ("e.parquet", vec![2_000, 2_100]),
            ("f.parquet", vec![2_000, 2_100]),
        ] {
            write_parquet_file(&adopted_dir, file, rows);
        }
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let files = datastore.table_files(&name).unwrap();
        let target = files.iter().map(|file| file.size).min().unwrap();

        run_one_sweep_with(target, 3, 2, datastore.clone());

        let files = datastore.table_files(&name).unwrap();
        assert!(
            files
                .iter()
                .all(|file| file.path.name().starts_with("pivot-")),
            "all three pairs were rewritten: {files:?}"
        );
        let rows: i64 = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(rows, 12, "no input was merged twice or dropped");
        dispatch.exit();
    }

    #[test]
    fn final_command_bypasses_small_file_guards() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let encoded = tempfile::tempdir().unwrap();
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_partitioned_table(&datastore, dispatch.dispatcher(), "events");
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut table = datastore.table_handle(&name).unwrap();
        for (file, partition) in [("a", 1), ("b", 1), ("c", 2), ("d", 2)] {
            let file = format!("{file}.parquet");
            write_partitioned_parquet_file(encoded.path(), &file, vec![partition], partition);
            table
                .append_data_file(
                    ObjectPath::new(&file),
                    &std::fs::read(encoded.path().join(&file)).unwrap(),
                    Some(partition_value(partition)),
                )
                .unwrap();
        }
        datastore.refresh_from_store().unwrap();
        let table = datastore.table_handle(&name).unwrap();
        let target = table
            .file_refs()
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);

        let routine_sweeps = run_command(
            CompacterSizes::without_small_files(target),
            datastore.clone(),
            name.clone(),
            table.clone(),
            CompactionPass::Routine,
        );
        let routine_files = datastore.table_handle(&name).unwrap().file_refs().len();
        let final_sweeps = run_command(
            CompacterSizes::without_small_files(target),
            datastore.clone(),
            name.clone(),
            table,
            CompactionPass::Final,
        );

        assert_eq!(routine_sweeps, 1);
        assert_eq!(routine_files, 4);
        assert_eq!(final_sweeps, 3);
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 2);
        dispatch.exit();
    }

    /// The row-group rule is the whole layout bar: a pair merges when more
    /// than a fifth of each file's row groups overlaps the other file, and
    /// files that only reach each other's boundary value never merge, not even
    /// under the final command.
    #[test]
    fn layout_merges_need_overlapping_row_groups_on_both_sides() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        // One row per row group. The first two files share seven of their
        // twenty groups; the last two meet at 119 and share nothing.
        write_parquet_file_with_row_group_size(&adopted, "a.parquet", (0..20).collect(), 1);
        write_parquet_file_with_row_group_size(&adopted, "b.parquet", (12..32).collect(), 1);
        write_parquet_file_with_row_group_size(&adopted, "c.parquet", (100..120).collect(), 1);
        write_parquet_file_with_row_group_size(&adopted, "d.parquet", (119..139).collect(), 1);
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        // Keep every input out of the small-file path.
        let target = table
            .file_refs()
            .iter()
            .map(|file| file.size)
            .min()
            .unwrap()
            .saturating_mul(2);

        run_command(
            CompacterSizes {
                layout_clique_size: 2,
                ..CompacterSizes::without_small_files(target)
            },
            datastore.clone(),
            name.clone(),
            table,
            CompactionPass::Routine,
        );
        let after_routine = datastore.table_handle(&name).unwrap();
        let routine_files: Vec<String> = after_routine
            .file_refs()
            .iter()
            .map(|file| file.path.name().to_string())
            .collect();
        let final_sweeps = run_command(
            CompacterSizes {
                layout_clique_size: 2,
                ..CompacterSizes::without_small_files(target)
            },
            datastore.clone(),
            name.clone(),
            after_routine,
            CompactionPass::Final,
        );

        assert_eq!(routine_files.len(), 3);
        assert!(routine_files.contains(&"c.parquet".to_string()));
        assert!(routine_files.contains(&"d.parquet".to_string()));
        assert_eq!(
            final_sweeps, 1,
            "the final command finds nothing the routine round left behind"
        );
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 3);
        dispatch.exit();
    }

    /// End-to-end compaction: several registered small files merge into one, the
    /// datastore swaps to the merged file in one log commit, and it decodes back to
    /// all the original rows.
    #[test]
    fn compaction_merges_registered_files_and_swaps_datastore() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![1, 2, 3]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![4, 5]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![6, 7, 8, 9]);

        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            false,
        );
        assert_eq!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .len(),
            3
        );

        let total: u64 = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        let target = (total as f64 / DEFAULT_MERGE_TARGET_MULTIPLIER).floor() as u64;
        run_one_sweep(target, datastore.clone());

        assert_eq!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .len(),
            1
        );
        let parquet = fresh_parquet(&datastore, "events");
        assert_eq!(parquet.row_groups().len(), 1);
        assert_eq!(parquet.row_groups()[0].num_rows, 9);
        assert!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .iter()
                .all(|f| f.path.as_str().starts_with("pivot-"))
        );

        dispatch.exit();
    }

    /// Small-file compaction uses the same output target as layout work. A
    /// three-file batch whose encoded row groups need two target-sized files is
    /// split rather than becoming one oversized replacement.
    #[test]
    fn small_file_compaction_splits_outputs_at_target() {
        const ROWS_PER_INPUT: usize = 64_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "a.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "b.parquet", ROWS_PER_INPUT, 29);
        write_wide_parquet_file(&adopted, "c.parquet", ROWS_PER_INPUT, 47);

        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let inputs = table.file_refs();
        let target = inputs
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);
        assert!(inputs.iter().all(|file| is_small_file(file.size, target)));
        let before_version = table.version();

        run_one_sweep(target, datastore.clone());

        let compacted = datastore.table_handle(&name).unwrap();
        assert_eq!(compacted.version(), before_version + 1);
        let outputs = compacted.file_refs();
        assert_eq!(outputs.len(), 2, "target={target}, outputs={outputs:?}");
        assert!(
            outputs
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );
        let row_count: i64 = compacted
            .build_scan_view(&[], &[])
            .unwrap()
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(row_count, (3 * ROWS_PER_INPUT) as i64);
        dispatch.exit();
    }

    #[test]
    fn layout_optimization_merges_the_highest_overlap_large_pair() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![1_000, 1_100]);

        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let files = datastore.table_files(&name).unwrap();
        let target = files.iter().map(|file| file.size).min().unwrap();
        run_one_sweep_with_pair_layout(target, datastore.clone());

        let files = datastore.table_files(&name).unwrap();
        assert_eq!(files.len(), 2);
        assert!(
            files.iter().any(|file| file.path.name() == "c.parquet"),
            "the disjoint file is not part of the chosen pair"
        );
        assert!(
            files
                .iter()
                .all(|file| file.path.name() != "a.parquet" && file.path.name() != "b.parquet")
        );

        dispatch.exit();
    }

    /// Two 80%-full files are layout work, not small-file work. Their globally
    /// sorted rewrite is too large for one target file, so it becomes two
    /// balanced, disjoint ranges which neither strategy selects again.
    #[test]
    fn overlapping_eighty_percent_files_become_two_stable_range_files() {
        // Exactly four normal compaction row groups across the two inputs, so
        // the encoded result has a natural balanced boundary for two files.
        const ROWS_PER_INPUT: usize = 256_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "left.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "right.parquet", ROWS_PER_INPUT, 29);

        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let inputs = table.file_refs();
        assert_eq!(inputs.len(), 2);
        let largest_input = inputs.iter().map(|file| file.size).max().unwrap();
        let target = largest_input.saturating_mul(5).div_ceil(4);
        for input in &inputs {
            let fullness = input.size as f64 / target as f64;
            assert!(
                (0.75..=0.81).contains(&fullness),
                "input {} is {:.1}% of target",
                input.path,
                fullness * 100.0
            );
        }

        // Verify that an effectively unbounded encoding, not merely the sum of the
        // input sizes, requires two files under this target.
        let unbounded = table
            .merge_files(&inputs, ROW_GROUP_ROWS, u64::MAX)
            .unwrap();
        assert_eq!(unbounded.len(), 1);
        let unbounded_size = unbounded[0].file_ref().size;
        for file in unbounded {
            table.delete_data_file(&file.file_ref().path).unwrap();
        }
        assert!(unbounded_size > target);
        assert!(unbounded_size <= target * 2);

        let before_version = table.version();
        run_one_sweep_with_pair_layout(target, datastore.clone());

        let compacted = datastore.table_handle(&name).unwrap();
        assert_eq!(compacted.version(), before_version + 1);
        let outputs = compacted.file_refs();
        assert_eq!(
            outputs.len(),
            2,
            "target={target}, unbounded={unbounded_size}, outputs={outputs:?}"
        );
        assert!(outputs.iter().all(|file| file.size <= target));
        assert!(outputs.iter().all(|file| !is_small_file(file.size, target)));
        assert!(
            outputs
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );

        let mut ranges: Vec<(i64, i64)> = compacted
            .file_entries()
            .map(|entry| {
                let stats = entry.stats.as_ref().unwrap();
                let scalar = |values: &std::collections::HashMap<String, ArrayRef>| {
                    values["Timestamp"]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0)
                };
                (scalar(&stats.min_values), scalar(&stats.max_values))
            })
            .collect();
        ranges.sort_unstable();
        assert_eq!(ranges.first().unwrap().0, 1);
        assert_eq!(ranges.last().unwrap().1, ROWS_PER_INPUT as i64);
        assert!(ranges[0].1 < ranges[1].0, "ranges are disjoint: {ranges:?}");

        let row_count: i64 = compacted
            .build_scan_view(&[], &[])
            .unwrap()
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(row_count, (2 * ROWS_PER_INPUT) as i64);

        let log = db
            .path()
            .join(compacted.location())
            .join("_delta_log")
            .join(format!("{:020}.json", compacted.version()));
        let actions: Vec<serde_json::Value> = std::fs::read_to_string(log)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.get("remove").is_some())
                .count(),
            2
        );
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.get("add").is_some())
                .count(),
            2
        );

        let stable_version = compacted.version();
        let stable_paths: std::collections::HashSet<_> =
            outputs.iter().map(|file| file.path.clone()).collect();
        run_one_sweep_with_pair_layout(target, datastore.clone());
        let after_normal = datastore.table_handle(&name).unwrap();
        assert_eq!(after_normal.version(), stable_version);
        assert_eq!(
            after_normal
                .file_refs()
                .into_iter()
                .map(|file| file.path)
                .collect::<std::collections::HashSet<_>>(),
            stable_paths
        );
        let final_sweeps = run_command(
            CompacterSizes {
                layout_clique_size: 2,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name,
            after_normal,
            CompactionPass::Final,
        );
        assert_eq!(final_sweeps, 1);
        assert_eq!(
            datastore.table_handle_by_id(&table.id()).unwrap().version(),
            stable_version
        );

        dispatch.exit();
    }

    /// Three mutually overlapping 80%-full files are exactly the configured
    /// clique, so a single merge and one commit rewrite all of them together
    /// into sorted, range-disjoint files that no later pass selects again.
    #[test]
    fn layout_optimization_merges_a_group_of_three_in_one_pass() {
        const ROWS_PER_INPUT: usize = 256_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "first.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "second.parquet", ROWS_PER_INPUT, 29);
        write_wide_parquet_file(&adopted, "third.parquet", ROWS_PER_INPUT, 47);
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let inputs = table.file_refs();
        assert_eq!(inputs.len(), 3);
        let target = inputs
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(5)
            .div_ceil(4);
        assert!(inputs.iter().all(|file| !is_small_file(file.size, target)));
        let before_version = table.version();

        run_one_sweep_with(target, DEFAULT_COMPACT_PARALLELISM, 3, datastore.clone());

        let compacted = datastore.table_handle(&name).unwrap();
        assert_eq!(
            compacted.version(),
            before_version + 1,
            "one merge rewrites all three inputs"
        );
        let outputs = compacted.file_refs();
        assert_eq!(outputs.len(), 2, "target={target}, outputs={outputs:?}");
        assert!(outputs.iter().all(|file| file.size <= target));
        assert!(outputs.iter().all(|file| !is_small_file(file.size, target)));
        let mut ranges: Vec<(i64, i64)> = compacted
            .file_entries()
            .map(|entry| {
                let stats = entry.stats.as_ref().unwrap();
                let scalar = |values: &std::collections::HashMap<String, ArrayRef>| {
                    values["Timestamp"]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0)
                };
                (scalar(&stats.min_values), scalar(&stats.max_values))
            })
            .collect();
        ranges.sort_unstable();
        assert_eq!(ranges.first().unwrap().0, 1);
        assert_eq!(ranges.last().unwrap().1, ROWS_PER_INPUT as i64);
        for window in ranges.windows(2) {
            assert!(window[0].1 < window[1].0, "ranges are disjoint: {ranges:?}");
        }
        let row_count: i64 = compacted
            .build_scan_view(&[], &[])
            .unwrap()
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(row_count, (3 * ROWS_PER_INPUT) as i64);
        let log = db
            .path()
            .join(compacted.location())
            .join("_delta_log")
            .join(format!("{:020}.json", compacted.version()));
        let actions: Vec<serde_json::Value> = std::fs::read_to_string(log)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            actions
                .iter()
                .filter(|action| action.get("remove").is_some())
                .count(),
            3
        );

        run_one_sweep_with(target, DEFAULT_COMPACT_PARALLELISM, 3, datastore.clone());
        assert_eq!(
            datastore.table_handle(&name).unwrap().version(),
            compacted.version()
        );

        dispatch.exit();
    }

    /// Two overlapping large files are less than the clique of three, so a
    /// routine round stands them deliberately; the final sweep settles them.
    #[test]
    fn routine_round_stands_a_partial_clique_and_the_final_sweep_settles_it() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![0, 100]);
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        let target = table
            .file_refs()
            .iter()
            .map(|file| file.size)
            .min()
            .unwrap();
        let before_version = table.version();

        run_one_sweep_with(target, DEFAULT_COMPACT_PARALLELISM, 3, datastore.clone());
        let after_routine = datastore.table_handle(&name).unwrap();
        let final_sweeps = run_command(
            CompacterSizes {
                layout_clique_size: 3,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name.clone(),
            after_routine.clone(),
            CompactionPass::Final,
        );

        assert_eq!(after_routine.version(), before_version);
        assert!(final_sweeps <= 3);
        let settled = datastore.table_handle(&name).unwrap();
        assert_eq!(settled.version(), before_version + 1);
        assert!(
            settled
                .file_refs()
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );
        let rows: i64 = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(rows, 4);
        dispatch.exit();
    }

    /// A wide file covers several mutually disjoint narrow files, none of
    /// which contests more than a fifth of its row groups: no pair clears
    /// the bar, so routine rounds stand still, and the final sweep's relaxed
    /// connected group settles the whole neighborhood into files that share
    /// no stretch of the key.
    #[test]
    fn final_sweep_settles_a_hub_no_routine_round_can_see() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_parquet_file_with_row_group_size(&adopted, "hub.parquet", (0..20).collect(), 1);
        write_parquet_file(&adopted, "s1.parquet", vec![2, 3]);
        write_parquet_file(&adopted, "s2.parquet", vec![8, 9]);
        write_parquet_file(&adopted, "s3.parquet", vec![14, 15]);
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();
        // Keep every input, satellites included, out of the small-file path.
        let target = table
            .file_refs()
            .iter()
            .map(|file| file.size)
            .min()
            .unwrap()
            .saturating_mul(2);
        let before_version = table.version();

        run_one_sweep(target, datastore.clone());
        let after_routine = datastore.table_handle(&name).unwrap();
        let final_sweeps = run_command(
            CompacterSizes::without_small_files(target),
            datastore.clone(),
            name.clone(),
            after_routine.clone(),
            CompactionPass::Final,
        );

        assert_eq!(after_routine.version(), before_version);
        assert!(final_sweeps <= 4, "relaxed merging converges quickly");
        let settled = datastore.table_handle(&name).unwrap();
        assert!(settled.version() > before_version);
        assert!(
            settled
                .file_refs()
                .iter()
                .all(|file| file.path.name().starts_with("pivot-"))
        );
        let mut ranges: Vec<(i64, i64)> = settled
            .file_entries()
            .map(|entry| {
                let stats = entry.stats.as_ref().unwrap();
                let scalar = |values: &std::collections::HashMap<String, ArrayRef>| {
                    values["Timestamp"]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0)
                };
                (scalar(&stats.min_values), scalar(&stats.max_values))
            })
            .collect();
        ranges.sort_unstable();
        for window in ranges.windows(2) {
            assert!(
                window[0].1 <= window[1].0,
                "no two files share a stretch of the key: {ranges:?}"
            );
        }
        let rows: i64 = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|group| group.num_rows)
            .sum();
        assert_eq!(rows, 26);
        dispatch.exit();
    }

    /// The byte target is applied only between independently encoded row
    /// groups. A group that crosses it is emitted alone without changing the
    /// normal compaction row-group size or re-encoding the input.
    #[test]
    fn one_oversized_row_group_remains_one_output_file() {
        const ROWS_PER_INPUT: usize = 8_000;
        let dispatch = Dispatch::spin_up(2, 8 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_wide_parquet_file(&adopted, "left.parquet", ROWS_PER_INPUT, 11);
        write_wide_parquet_file(&adopted, "right.parquet", ROWS_PER_INPUT, 29);

        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_wide_sorted_table(&datastore, dispatch.dispatcher(), "events", &adopted);
        let table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();
        let inputs = table.file_refs();
        let target = inputs
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(5)
            .div_ceil(4);

        let outputs = table.merge_files(&inputs, ROW_GROUP_ROWS, target).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].file_ref().size > target);
        for output in outputs {
            table.delete_data_file(&output.file_ref().path).unwrap();
        }
        dispatch.exit();
    }

    /// Partitioned compaction never mixes partitions, and both the replacement
    /// Adds and input tombstones retain the partition values needed to reload a
    /// valid Delta snapshot.
    #[test]
    fn compaction_preserves_partitioned_table_metadata_and_rows() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let encoded = tempfile::tempdir().unwrap();
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_partitioned_table(&datastore, dispatch.dispatcher(), "events");

        // Two inputs in each partition. If batching crossed the partition
        // boundary, the merged file could not truthfully carry one constant.
        let inputs = [
            ("one-a.parquet", 1),
            ("one-b.parquet", 1),
            ("two-a.parquet", 2),
            ("two-b.parquet", 2),
        ];
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut table = datastore.table_handle(&name).unwrap();
        for (file_name, value) in inputs {
            write_partitioned_parquet_file(
                encoded.path(),
                file_name,
                vec![value * 10, value * 10 + 1],
                value,
            );
            table
                .append_data_file(
                    ObjectPath::new(file_name),
                    &std::fs::read(encoded.path().join(file_name)).unwrap(),
                    Some(partition_value(value)),
                )
                .unwrap();
        }
        let before_version = table.version();
        assert_eq!(table.file_refs().len(), 4);
        assert!(datastore.refresh_from_store().unwrap());

        let target = table
            .file_refs()
            .into_iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);
        run_command(
            CompacterSizes {
                merge_target_bytes: 0,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name.clone(),
            table,
            CompactionPass::Routine,
        );

        // One round merges each partition's small files in turn; a second round
        // then finds nothing left to do.
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 2);
        run_command(
            CompacterSizes {
                merge_target_bytes: 0,
                ..CompacterSizes::defaults(target)
            },
            datastore.clone(),
            name.clone(),
            datastore.table_handle(&name).unwrap(),
            CompactionPass::Routine,
        );

        // Reload from Delta rather than trusting the in-memory swap. Exactly one
        // output per partition remains, and every original row is still present.
        let mut reloaded = datastore.table_handle(&name).unwrap();
        reloaded.refresh().unwrap();
        assert_eq!(reloaded.file_refs().len(), 2);
        let partitions = reloaded.file_partitions();
        assert_eq!(partitions.len(), 2);
        for expected in [partition_value(1), partition_value(2)] {
            assert!(partitions.iter().any(|(_, actual)| {
                actual
                    .as_ref()
                    .is_some_and(|actual| scalar_values_equal(actual, &expected))
            }));
        }
        let parquet = reloaded.build_scan_view(&[], &[]).unwrap();
        assert_eq!(
            parquet
                .row_groups()
                .iter()
                .map(|group| group.num_rows)
                .sum::<i64>(),
            8
        );

        // Inspect every compaction commit. Each removes and adds files from one
        // partition only, and Kernel serialized that value into both actions.
        let log_dir = db.path().join(reloaded.location()).join("_delta_log");
        let mut compacted_partitions = std::collections::BTreeSet::new();
        for version in before_version + 1..=reloaded.version() {
            let commit =
                std::fs::read_to_string(log_dir.join(format!("{version:020}.json"))).unwrap();
            let actions: Vec<serde_json::Value> = commit
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let remove_values: std::collections::BTreeSet<_> = actions
                .iter()
                .filter_map(|action| action.get("remove"))
                .map(|remove| remove["partitionValues"]["Partition"].as_str().unwrap())
                .collect();
            assert_eq!(
                actions
                    .iter()
                    .filter(|action| action.get("remove").is_some())
                    .count(),
                2
            );
            let add_values: std::collections::BTreeSet<_> = actions
                .iter()
                .filter_map(|action| action.get("add"))
                .map(|add| add["partitionValues"]["Partition"].as_str().unwrap())
                .collect();
            assert_eq!(
                actions
                    .iter()
                    .filter(|action| action.get("add").is_some())
                    .count(),
                1
            );
            assert_eq!(remove_values.len(), 1, "one partition per compacted batch");
            assert_eq!(
                add_values, remove_values,
                "replacement stays in its partition"
            );
            compacted_partitions.extend(remove_values.into_iter().map(str::to_string));
        }
        assert_eq!(
            compacted_partitions,
            std::collections::BTreeSet::from(["1".to_string(), "2".to_string()])
        );

        dispatch.exit();
    }

    /// Compaction is location-agnostic: a table whose adopted files were listed
    /// **store-relative** (a directory under the database root, resolved through
    /// the store -- the same path a remote `s3://` root takes) compacts through
    /// the exact same code, with writes and deletes going through the table's
    /// store handle.
    #[test]
    fn compaction_works_on_store_relative_tables() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();

        // Two small files under a prefix inside the database root.
        write_parquet_file(&adopted_dir, "a.parquet", vec![1, 2]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![3]);

        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        // A relative adoption path: the files are read at `events` under the
        // store root, rather than at an absolute path of their own.
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            false,
        );
        let table_dir = db.path().join(
            datastore
                .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .location(),
        );

        let target = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .iter()
            .map(|f| f.size)
            .max()
            .unwrap()
            .saturating_mul(2)
            .saturating_add(1);
        let name = SchemaQualifiedTableName::in_default_schema("events");
        run_command(
            CompacterSizes::defaults(target),
            datastore.clone(),
            name.clone(),
            datastore.table_handle(&name).unwrap(),
            CompactionPass::Final,
        );

        // One merged object replaced the two inputs, in the store and in the
        // datastore's table view.
        let files = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.as_str().starts_with("pivot-"));
        let on_disk: Vec<_> = std::fs::read_dir(&table_dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".parquet"))
            .collect();
        // The merged object is written to the table's own directory; the two
        // adopted inputs are left where they were.
        assert!(on_disk.contains(&files[0].path.as_str().to_string()));
        assert!(adopted_dir.join("a.parquet").exists());
        assert!(adopted_dir.join("b.parquet").exists());

        let rows = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows)
            .sum::<i64>();
        assert_eq!(rows, 3);

        dispatch.exit();
    }
}
