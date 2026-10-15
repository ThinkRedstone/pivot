//! Work-stealing source that feeds row groups into the fetching pipeline.
//!
//! [`RowGroupInjectorFactory`] pre-loads every row group from a
//! [`ParquetTable`] into one shared [`Injector`] queue per NUMA node. Each
//! worker gets its own [`RowGroupInjector`] (via [`RootChannelFactory`]) that
//! claims row groups from its node's queue on demand.
//!
//! Row groups are assigned to nodes by a stable hash of their identity (file +
//! index within the file), so the *same* row group lands on the *same* node
//! query after query. Everything a claim produces (the read into the
//! compressed cache, decompression, decode) happens on the claiming worker,
//! so sticky assignment keeps a row group's cached bytes on one node and its
//! next scan node-local. A worker whose node's queue runs dry claims from
//! other nodes' queues rather than idle (the stolen row group is still read
//! and decoded entirely on the claiming node; only its cached bytes end up
//! remote for a later query).

use crate::types::metadata::{QueryRowGroupMetadata, RowGroupMetadata, RowSelection};
use crate::types::projection::Projection;
use crate::types::table::ParquetTable;
use arrow_array::{ArrayRef, Scalar};
use crossbeam_deque::{Injector, Steal};
use dispatch::{Receiver, RootChannelFactory};
use std::cmp::{Ordering as CmpOrdering, Reverse};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{RowGroupFilter, ScanOrder};

/// The NUMA node a row group is assigned to, by a hash of its stable identity
/// (its file and its index within that file). Deterministic across queries so
/// assignment is sticky for as long as the file stays open, and uniform enough
/// to split any table's row groups evenly across nodes.
fn affinity_node(row_group: &RowGroupMetadata, node_count: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    row_group.open_file.hash(&mut hasher);
    row_group.file_row_group_idx.hash(&mut hasher);
    (hasher.finish() % node_count as u64) as usize
}

use crate::types::table::scalar_lt;

/// Steal order for the row groups of `table` under a Top-N's [`ScanOrder`]:
/// indices sorted so the most-promising row group (smallest `min` ascending,
/// largest `max` descending) is handed out first, making the dynamic boundary
/// tighten before the bulk of the data is touched. Row groups lacking the stat
/// can't be pruned (missing stats are never eliminated), so they must be read
/// regardless — they go last. Reordering is purely an optimization: any order is
/// correct (the Top-N re-sorts), this just minimizes how much gets read.
fn steal_order(table: &ParquetTable, order: &ScanOrder) -> Vec<usize> {
    let n = table.row_groups.len();
    let mut keyed: Vec<(usize, Scalar<ArrayRef>)> = Vec::with_capacity(n);
    let mut unkeyed: Vec<usize> = Vec::new();
    for idx in 0..n {
        let stat = table.row_groups[idx]
            .column_statistics(order.column_idx)
            .and_then(|s| if order.descending { s.max() } else { s.min() });
        match stat {
            Some(scalar) => keyed.push((idx, scalar)),
            None => unkeyed.push(idx),
        }
    }
    keyed.sort_by(|(_, a), (_, b)| {
        // Smallest `min` first when ascending; largest `max` first when
        // descending (reverse the ascending comparison).
        let ord = if scalar_lt(a, b) {
            CmpOrdering::Less
        } else if scalar_lt(b, a) {
            CmpOrdering::Greater
        } else {
            CmpOrdering::Equal
        };
        if order.descending { ord.reverse() } else { ord }
    });
    let mut result: Vec<usize> = keyed.into_iter().map(|(idx, _)| idx).collect();
    result.extend(unkeyed);
    result
}

/// Order a plain scan's row groups for handout, by two keys:
///
/// 1. Most decompressed-cache-resident pages first, so the scan consumes them
///    before its own churn can evict them - without this, a repeated scan asks
///    for each row group exactly when its cached pages are the oldest thing in
///    the ring, and hits nothing.
/// 2. Ties (a fully cold table in particular, where every count is zero) break
///    to the largest projected compressed size. A row group's whole decode is
///    pinned to the worker that claims it, so a scan's wall time is the
///    makespan of placing unequal-cost row groups onto workers; handing out the
///    biggest first is the classic longest-processing-time rule - the
///    expensive row groups start immediately and the small ones fill the gaps
///    behind them. With any other order, a large row group claimed behind
///    another job sets a whole query's critical path (worst when the row-group
///    count is close to the worker count, where one unlucky claim adds an
///    entire extra job to the tail). Cache-resident row groups jumping the LPT
///    queue can't set that critical path: their decode skips the expensive
///    decompression, so they are the cheap tail-filler jobs anyway.
fn plain_scan_order(table: &ParquetTable, projection: &Projection) -> Vec<usize> {
    let mut order: Vec<usize> = (0..table.row_groups.len()).collect();
    let projected_size = |idx: usize| -> i64 {
        let row_group = &table.row_groups[idx];
        projection
            .indices()
            .iter()
            .map(|&col| row_group.columns[col].total_compressed_size)
            .sum()
    };
    // The cache counters move concurrently (other queries insert and evict
    // pages mid-sort), so each must be read exactly once - a per-comparison
    // re-read hands the sort an inconsistent ordering. sort_by_cached_key
    // guarantees the single read.
    order.sort_by_cached_key(|&idx| {
        Reverse((
            table.row_groups[idx]
                .live_decompressed_pages
                .load(Ordering::Relaxed),
            projected_size(idx),
        ))
    });
    order
}

/// Factory that populates one [`Injector`] per NUMA node with the table's row
/// groups (split by [`affinity_node`]) and produces [`RowGroupInjector`]
/// receivers for each worker.
#[derive(Clone)]
pub struct RowGroupInjectorFactory {
    row_group_queues: Arc<Vec<Injector<QueryRowGroupMetadata>>>,
    filter: Option<RowGroupFilter>,
    speculation: Option<Arc<SpeculationGate>>,
}

/// The allowance on outstanding claims while the boundary has not yet proven
/// it can prune (see [`RowGroupInjector::acquire_speculation_ticket`]). The
/// floor stays flat until admissions outgrow it: letting the allowance rise
/// from the very first admission instead widens the pre-convergence window,
/// measured as a heavy tail on gated scans (occasional runs admit several
/// times the floor before the boundary catches up).
const SPECULATION_ALLOWANCE_FLOOR: usize = 16;

/// Shared state for throttling a Top-N scan's claims while its boundary
/// converges.
///
/// A Top-N scan's dynamic boundary starts empty and only *tightens* as the
/// operator consumes rows, so every claim admitted early is judged by a
/// boundary weaker than the one a moment later - and a claim is a read that
/// cannot be retracted. Crucially, a boundary that has a value is not yet a
/// boundary worth trusting: the first published value comes from whichever
/// batch happened to consume first, and letting the whole pool claim the
/// entire scan the instant one appears prunes against that lottery ticket
/// (sometimes near-optimal, sometimes keeping half the table).
///
/// The gate therefore never "opens": every claim holds a ticket, capped by an
/// allowance on *outstanding* claims (claimed but not yet fully decoded,
/// tracked by the scan's shared counter). Row groups are handed out
/// most-promising-first, so the throttled front claims are exactly the ones
/// the boundary needs to converge, and each admitted round tightens it before
/// deeper row groups are judged. Once the boundary is tight enough to prune
/// the next row group, the sorted order means the entire remaining queue
/// prunes too - pruned claims release their ticket immediately, so draining
/// the tail is a cheap stats check per row group, not a read. A boundary that
/// never arms (or a scan the boundary cannot prune) converges to an
/// unthrottled scan through the allowance ramp (see
/// [`RowGroupInjector::acquire_speculation_ticket`]).
pub struct SpeculationGate {
    /// Row groups claimed but not yet fully decoded, scan-wide (incremented on
    /// claim here, decremented as each row group's last decode range is
    /// decoded, or when it is pruned).
    outstanding: Arc<AtomicUsize>,
    /// Claims admitted so far. Every admitted claim raises the allowance on
    /// outstanding claims (half this count, floored at
    /// [`SPECULATION_ALLOWANCE_FLOOR`]); pruned claims don't, which is what
    /// keeps a converged scan at the floor while the ramp opens unprunable
    /// scans geometrically.
    admitted_claims: AtomicUsize,
    /// Row groups still queued, scan-wide (see
    /// [`SpeculationGate::note_claimed`]).
    remaining: AtomicUsize,
}

impl SpeculationGate {
    /// Count one row group leaving the queues; broadcast-wake the pool when
    /// the last one goes. Workers the gate turned away park without having
    /// finished their pipelines (a denied claim is a no-op pass,
    /// indistinguishable from an idle one), and neither the drain nor the
    /// finish cascade wakes them on its own - pruned claims send nothing
    /// downstream, and the finish protocol needs *every* worker to run its own
    /// finalization. Without this wake the query hangs with the pool parked
    /// one step short of done.
    fn note_claimed(&self) {
        if self.remaining.fetch_sub(1, Ordering::Relaxed) == 1 {
            dispatch::waker::waker_set().notify_all();
        }
    }
}

impl RowGroupInjectorFactory {
    /// Creates a new factory, pushing each of `table`'s row groups into its
    /// affinity node's queue (`node_count` queues in total). When `filter` is
    /// set, each row group is offered to it on claim and skipped if it returns
    /// `false`. When `scan_order` is set (a Top-N boundary on one key), row
    /// groups are pushed in that key's order so the most-promising are claimed
    /// first and the boundary prunes the rest; otherwise decompressed-cache-covered
    /// row groups go first, largest-first within a tie (see
    /// [`plain_scan_order`]).
    pub fn new(
        table: &Arc<ParquetTable>,
        projection: Projection,
        filter: Option<RowGroupFilter>,
        scan_order: Option<ScanOrder>,
        outstanding_row_groups: Arc<AtomicUsize>,
        node_count: usize,
    ) -> Self {
        let row_group_queues: Arc<Vec<Injector<QueryRowGroupMetadata>>> =
            Arc::new((0..node_count).map(|_| Injector::new()).collect());
        let order = match &scan_order {
            Some(order) => steal_order(table, order),
            None => plain_scan_order(table, &projection),
        };
        let row_groups: Vec<QueryRowGroupMetadata> = order
            .iter()
            .map(|&idx| QueryRowGroupMetadata::new(table, idx, RowSelection::All))
            .collect();
        let speculation = scan_order.map(|_| {
            Arc::new(SpeculationGate {
                outstanding: outstanding_row_groups,
                admitted_claims: AtomicUsize::new(0),
                remaining: AtomicUsize::new(row_groups.len()),
            })
        });
        for row_group in row_groups {
            let node = affinity_node(row_group.get_metadata(), node_count);
            row_group_queues[node].push(row_group);
        }
        Self {
            row_group_queues,
            filter,
            speculation,
        }
    }
}

impl RootChannelFactory<QueryRowGroupMetadata> for RowGroupInjectorFactory {
    type Receiver = RowGroupInjector;

    /// Runs on the worker thread, so the receiver can snapshot which node's
    /// queue is local to it.
    fn build(self) -> Self::Receiver {
        RowGroupInjector {
            numa_node_idx: dispatch::worker::current_node(),
            row_group_queues: self.row_group_queues,
            filter: self.filter,
            speculation: self.speculation,
        }
    }
}

/// A [`Receiver`] that claims row groups from the per-node [`Injector`]
/// queues: its own node's queue on the hot path, other nodes' queues only
/// when its own runs dry.
pub struct RowGroupInjector {
    row_group_queues: Arc<Vec<Injector<QueryRowGroupMetadata>>>,
    /// Index of this worker's node's queue in `row_group_queues`.
    numa_node_idx: usize,
    filter: Option<RowGroupFilter>,
    speculation: Option<Arc<SpeculationGate>>,
}

impl Receiver<QueryRowGroupMetadata> for RowGroupInjector {
    fn is_empty(&self) -> bool {
        self.row_group_queues.iter().all(|queue| queue.is_empty())
    }

    /// Pull eagerly from this node's own queue. The injector is the *only*
    /// source of row groups, so it must be reachable from `run_cpu_work` (which
    /// calls `try_recv`), otherwise row groups are claimed only in the worker's
    /// idle branch, one at a time, and a high-latency remote scan never builds
    /// read-ahead (in-flight reads peak at 1 per worker). A single non-spinning
    /// attempt keeps the hot loop from spinning on `Steal::Retry`; the next
    /// iteration retries. A pruned row group also returns `None` (it has been
    /// removed; the next call advances).
    fn try_recv(&self) -> Option<QueryRowGroupMetadata> {
        // This pre-check is a plain load; once the queue drains (every pass
        // for the rest of the query) it avoids an epoch-pinning steal per
        // call.
        let queue = &self.row_group_queues[self.numa_node_idx];
        if queue.is_empty() {
            return None;
        }
        let ticket = self.acquire_speculation_ticket()?;
        match queue.steal() {
            Steal::Success(row_group) => self.admit_unless_pruned(row_group, ticket),
            // Dropping the ticket releases its reservation; the next call
            // retries.
            Steal::Retry | Steal::Empty => None,
        }
    }

    /// Pull on the idle path: drain this node's queue first, then fall back
    /// to other nodes' queues so a node that finished its share keeps its
    /// CPUs busy instead of waiting out the tail.
    fn steal(&self) -> Option<QueryRowGroupMetadata> {
        let queue_count = self.row_group_queues.len();
        // A ticket is acquired only once a non-empty queue is found; a lost
        // steal race or a queue drained under us carries it to the next
        // attempt rather than bouncing the reservation off the gate.
        let mut ticket = None;
        for offset in 0..queue_count {
            let queue = &self.row_group_queues[(self.numa_node_idx + offset) % queue_count];
            while !queue.is_empty() {
                let held = match ticket.take() {
                    Some(held) => held,
                    None => self.acquire_speculation_ticket()?,
                };
                match queue.steal() {
                    Steal::Success(row_group) => {
                        // A pruned row group released the ticket; keep
                        // claiming (back through the gate) rather than
                        // handing the worker a no-op.
                        if let Some(request) = self.admit_unless_pruned(row_group, held) {
                            return Some(request);
                        }
                    }
                    Steal::Retry => ticket = Some(held),
                    Steal::Empty => {
                        ticket = Some(held);
                        break;
                    }
                }
            }
        }
        None
    }
}

/// A reserved slot in the speculation allowance, held from queue pop to claim
/// admission (see [`RowGroupInjector::acquire_speculation_ticket`]). Dropping
/// the ticket releases the reservation, so every claim path that produces no
/// row group gives its slot back without further ceremony.
enum SpeculationTicket<'a> {
    /// The scan is not throttled (no Top-N boundary): claims need no
    /// reservation.
    Unthrottled,
    /// A claim's reservation in the speculation allowance.
    Reserved(&'a SpeculationGate),
}

impl Drop for SpeculationTicket<'_> {
    fn drop(&mut self) {
        if let SpeculationTicket::Reserved(gate) = self {
            gate.outstanding.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl SpeculationTicket<'_> {
    /// Hand the reservation over to the claimed row group: it becomes the
    /// claim's outstanding count, released when the row group is decoded
    /// instead of by drop.
    fn transfer_to_row_group(self) {
        std::mem::forget(self);
    }
}

impl RowGroupInjector {
    /// Admit a claimed row group, or `None` (and release the ticket) when the
    /// row-group filter proves it holds no matching row.
    fn admit_unless_pruned(
        &self,
        row_group: QueryRowGroupMetadata,
        ticket: SpeculationTicket<'_>,
    ) -> Option<QueryRowGroupMetadata> {
        if let Some(gate) = &self.speculation {
            gate.note_claimed();
        }
        if let Some(filter) = &self.filter
            && !filter(row_group.get_metadata())
        {
            return None;
        }
        if let SpeculationTicket::Reserved(gate) = &ticket {
            // The ticket becomes the claim's outstanding count (released when
            // the row group is decoded). While throttled the
            // pool is mostly parked (gated claims are no-op passes) and a
            // claim, unlike a channel send, wakes nobody on its own; waking
            // one worker per admitted claim lets the working set grow with
            // the allowance. The wake prefers this node but spills to the
            // others when nobody here is parked: every other wake during the
            // throttled phase is node-local, so a purely local wake can leave
            // an entire node parked for the whole phase whenever the other
            // node wins the early tickets (a self-reinforcing race: awake
            // workers claim freed tickets before parked ones hear of them),
            // while an unconditional cross-node wake invites remote workers
            // to claim, and decode through, the other node's cached bytes
            // even when local workers were available.
            gate.admitted_claims.fetch_add(1, Ordering::Relaxed);
            dispatch::waker::waker_set().notify_one_near(self.numa_node_idx);
        }
        ticket.transfer_to_row_group();
        Some(row_group)
    }

    /// Reserve a slot in the speculation allowance, or `None` when the scan is
    /// currently throttled (the allowance already claimed). The reservation
    /// happens *before* the queue pop and atomically (reserve, then check), so
    /// a burst of workers racing the gate cannot collectively overshoot it:
    /// each one either holds a counted slot or backs off. A plain load screens
    /// out the already-throttled case first, so denied passes (every spinning
    /// worker's, while the gate is closed) don't write the shared counter.
    ///
    /// The allowance ramps instead of staying a fixed cap: every *admitted*
    /// claim raises it, so a scan whose boundary can't keep up - a selective
    /// filter above it, a boundary with no publisher, statistics it can't
    /// prune - opens up geometrically instead of trickling forever, and a
    /// boundary that never helps converges to an unthrottled scan. Pruned
    /// claims release their ticket without raising the allowance, so a
    /// well-converged boundary keeps the scan at the small cap while the
    /// remaining queue drains as cheap stats checks.
    fn acquire_speculation_ticket(&self) -> Option<SpeculationTicket<'_>> {
        let Some(speculation) = self.speculation.as_deref() else {
            return Some(SpeculationTicket::Unthrottled);
        };
        let allowance = SPECULATION_ALLOWANCE_FLOOR
            .max(speculation.admitted_claims.load(Ordering::Relaxed) / 2);
        if speculation.outstanding.load(Ordering::Relaxed) >= allowance {
            return None;
        }
        if speculation.outstanding.fetch_add(1, Ordering::Relaxed) >= allowance {
            speculation.outstanding.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        Some(SpeculationTicket::Reserved(speculation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::dummy_row_group;

    fn table_of(count: usize) -> Arc<ParquetTable> {
        Arc::new(ParquetTable::new(
            (0..count).map(|_| dummy_row_group()).collect(),
        ))
    }

    #[test]
    fn cache_covered_row_groups_are_fed_first() {
        let table = table_of(3);
        table.row_groups[1]
            .live_decompressed_pages
            .store(5, Ordering::Relaxed);
        table.row_groups[2]
            .live_decompressed_pages
            .store(2, Ordering::Relaxed);

        let order = plain_scan_order(&table, &Projection::columns([]));

        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn equal_size_ties_keep_file_order() {
        let table = table_of(4);

        let order = plain_scan_order(&table, &Projection::columns([]));

        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    fn sized_row_group(compressed_size: i64) -> Arc<RowGroupMetadata> {
        let dummy = dummy_row_group();
        Arc::new(RowGroupMetadata {
            open_file: dummy.open_file.clone(),
            schema: dummy.schema.clone(),
            columns: vec![crate::types::metadata::ColumnChunkMeta {
                codec: crate::thrift::general::CompressionCodec::SNAPPY,
                dictionary_page_offset: None,
                data_page_offset: 0,
                total_compressed_size: compressed_size,
                total_uncompressed_size: compressed_size,
                max_def_level: 0,
                physical_type: 0,
                fixed_len_byte_width: None,
                data_pages_all_dictionary: false,
                absent: false,
            }],
            statistics: Arc::default(),
            num_rows: 0,
            file_row_group_idx: 0,
            live_decompressed_pages: Arc::new(AtomicUsize::new(0)),
        })
    }

    #[test]
    fn a_cold_table_is_claimed_largest_first() {
        let table = Arc::new(ParquetTable::new(vec![
            sized_row_group(10),
            sized_row_group(30),
            sized_row_group(20),
        ]));

        let order = plain_scan_order(&table, &Projection::columns([0]));

        assert_eq!(order, vec![1, 2, 0]);
    }
}
