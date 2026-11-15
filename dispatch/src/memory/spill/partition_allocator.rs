//! Per-partition storage for rows that may have to be spilled.
//!
//! Each partition appends rows into a slab of its own; a [`Layout`] says how
//! big a row is and how many rows a slab holds. Slabs are allocated out of
//! 2MB ring buffers that the partitions of a group share,
//! laid out as a [`PartitionTree`]: every group starts at the root buffer, and
//! a node whose buffer is full hands its groups to its children, so buffers
//! come to be shared by fewer partitions as the data grows.
//!
//! A buffer is sealed, and can be spilled, once no partition is still writing
//! into any slab of it. That is read off the buffer's reference count: every slab holds its
//! buffer, and the allocator records every slab it has closed. A partition's
//! open slab is the only other holder, so a buffer referenced by exactly its
//! closed slabs and the allocator's own record has no open slab left.

use crate::io::{IORequesterError, LocalFile};
use crate::memory::slab::SlabBacking;
use crate::memory::spill::codec::{compress_spilled_rows, get_most_compressed_len};
use crate::memory::spill::layout::Layout;
use crate::memory::spill::partition_tree::{
    PARTITIONS, PartitionTree, group_of_partition, is_leaf,
};
use crate::memory::spill::{
    SPILL_BLOCK_SIZE, SpillWrite, SpilledBuffer, SpilledSlab, Spiller, padded_to_spill_block,
};
use crate::memory::{BUFFER_SIZE, FileBytes, FrozenSlab, Slab, WriteBuffer, memory_ctx};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

thread_local! {
    /// Where a worker compresses the slabs of one write before it knows how
    /// much ring memory they need.
    static COMPRESSED_ROWS_SCRATCH: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// How many ring buffers a worker's operator keeps for compressed rows on
/// their way to disk.
const COMPRESSED_ROWS_BUFFER_COUNT: usize = 32;

/// The ring buffers compressed rows are copied into and written from.
///
/// Compressing rows takes memory for the result, and rows are spilled when
/// memory has run out. So the buffers are taken once, when rows are first
/// spilled and the reserve still has memory to give, and are then kept: a
/// buffer is filled from its start again as soon as every write out of it has
/// landed. How much can be spilled at once is bounded by the buffers that are
/// free, which also bounds the compressed rows in flight.
struct CompressedRowsBuffers {
    buffers: Vec<CompressedRowsBuffer>,
}

struct CompressedRowsBuffer {
    buffer: Arc<WriteBuffer>,
    /// Bytes handed out from the buffer's start since it was last empty.
    taken: usize,
}

impl CompressedRowsBuffers {
    fn take_from_pool() -> Self {
        Self {
            buffers: (0..COMPRESSED_ROWS_BUFFER_COUNT)
                .map(|_| CompressedRowsBuffer {
                    buffer: Arc::new(memory_ctx().get_write_buffer(false)),
                    taken: 0,
                })
                .collect(),
        }
    }

    /// Start from the beginning again in every buffer that no write holds
    /// any more: every write out of it has landed.
    fn start_over_in_landed_buffers(&mut self) {
        for compressed_rows_buffer in &mut self.buffers {
            if Arc::strong_count(&compressed_rows_buffer.buffer) == 1 {
                compressed_rows_buffer.taken = 0;
            }
        }
    }

    /// How many buffers of rows can be compressed into the buffers that are
    /// empty now. Rows that do not compress take slightly more room than they
    /// had, and a write does not straddle two buffers, so each buffer of rows
    /// is allowed two empty ones.
    fn count_buffers_it_can_compress(&self) -> usize {
        let empty_buffers = self
            .buffers
            .iter()
            .filter(|compressed_rows_buffer| compressed_rows_buffer.taken == 0)
            .count();
        empty_buffers / 2
    }

    /// `len` bytes, a multiple of the spill block size, from the first buffer
    /// with room for them.
    fn allocate(&mut self, len: usize) -> Slab {
        let compressed_rows_buffer = self
            .buffers
            .iter_mut()
            .find(|compressed_rows_buffer| compressed_rows_buffer.taken + len <= BUFFER_SIZE)
            .expect("no more rows are compressed at once than their buffers have room for");
        // SAFETY: `taken + len` is inside the buffer, checked just above.
        let start = unsafe {
            compressed_rows_buffer
                .buffer
                .ptr
                .add(compressed_rows_buffer.taken)
        };
        compressed_rows_buffer.taken += len;
        Slab::new(
            start,
            len,
            SlabBacking::Working(compressed_rows_buffer.buffer.clone()),
        )
    }
}

/// A slab compressed into the scratch for the write being gathered.
struct PackedSlab<'a> {
    /// The position, among the buffers being spilled, of the one it is in.
    spilled_buffer_idx: usize,
    /// Where its compressed bytes begin in the scratch, and so in the write.
    start: usize,
    compressed_len: usize,
    slab: &'a FrozenSlab,
}

/// Have `write` count as in flight for `spilled_buffer`, which it carries
/// rows of, unless it already does.
fn count_write_against_buffer(write: &mut SpillWrite, spilled_buffer: &SpilledBuffer) {
    let already_counted = write
        .buffers_writes_in_flight
        .iter()
        .any(|counted| Arc::ptr_eq(counted, &spilled_buffer.writes_in_flight));
    if !already_counted {
        spilled_buffer
            .writes_in_flight
            .fetch_add(1, Ordering::Release);
        write
            .buffers_writes_in_flight
            .push(spilled_buffer.writes_in_flight.clone());
    }
}

/// The slab a partition is appending into, and where it came from, so that
/// once full it can be recorded against its buffer. How far it is filled is
/// kept in the partition's [`RowCursor`].
struct OpenSlab {
    slab: Slab,
    /// Rows the slab holds when full.
    rows: usize,
    /// The tree node of the partition group slot the slab was allocated from.
    partition_group_slot: usize,
    /// The ring slot of that buffer, which identifies it within the node.
    buffer_slot_idx: usize,
}

/// Where a partition's next row goes, and how many more fit in its open slab.
///
/// The scatter loop hops between thousands of partitions in hash order, so
/// what it touches per partition is a cache-resident working set and every
/// extra line costs a stalled access. The per-row step therefore reads and
/// writes only this cursor, 16 bytes, with the open slab's handle and
/// provenance kept apart in [`Partition`] for the once-per-slab open and
/// close. `rows_left` is zero exactly when the partition has no open slab: a
/// slab is closed on the reserve that fills it.
#[derive(Clone, Copy)]
struct RowCursor {
    next_row: *mut u8,
    rows_left: usize,
}

struct Partition {
    open_slab: Option<OpenSlab>,
    /// Slabs this partition has filled, which the layout sizes the next by.
    slabs_filled: usize,
    /// The partition's first and last closed slab in the allocator's list,
    /// `NO_SLAB` while it has none; the slabs chain from first to last.
    first_closed: u32,
    last_closed: u32,
    /// Rows in the partition's closed slabs.
    rows_closed: usize,
    /// The file this partition's spilled rows are appended to, once it has
    /// spilled for the first time.
    spill_file: Option<LocalFile>,
    /// Bytes appended to the spill file so far, which is where the next
    /// spilled slab's rows will begin in it.
    bytes_spilled: u64,
}

/// A slab closed for writing, in the worker's list of them (see
/// [`PartitionAllocator::into_partitions`]). A partition's slabs chain through
/// `next`, in the order they were closed, so the list is grouped by partition
/// without ever being rearranged.
pub struct SlabEntry {
    /// The slab, taken when it is spilled, and when the merge releases it.
    pub slab: Option<FrozenSlab>,
    pub rows: usize,
    pub partition: usize,
    /// The partition's next closed slab, or `NO_SLAB`.
    pub next: u32,
}

/// The end of a partition's slab chain.
pub const NO_SLAB: u32 = u32::MAX;

/// A closed slab taken out of the slab list to be spilled.
struct ClosedSlab {
    partition: usize,
    slab: FrozenSlab,
}

/// A buffer the partitions under one tree node share, and the full slabs taken
/// from it so far.
struct PartitionGroupBuffer {
    buffer: Arc<WriteBuffer>,
    /// The closed slabs allocated from the buffer, as indices into the
    /// allocator's slab list.
    closed: Vec<u32>,
}

impl PartitionGroupBuffer {
    fn new(buffer: Arc<WriteBuffer>) -> Self {
        Self {
            buffer,
            closed: Vec::new(),
        }
    }

    /// Whether no partition is still writing into this buffer. Each closed
    /// slab holds one reference and this record holds one; any further
    /// reference is a partition's open slab, or the node still allocating
    /// from the buffer.
    fn is_sealed(&self) -> bool {
        Arc::strong_count(&self.buffer) == self.closed.len() + 1
    }
}

/// One tree node: the buffers of the partition groups that point at it, and
/// whether slabs are still being allocated from it.
enum PartitionGroupSlot {
    /// No partition under this node has needed a slab yet.
    Unused,
    /// Slabs are allocated from `current`, which is also the last of
    /// `buffers`, from `next_allocation_offset` onwards.
    Allocating {
        current: Arc<WriteBuffer>,
        next_allocation_offset: usize,
        buffers: Vec<PartitionGroupBuffer>,
    },
    /// An inner node whose buffer filled up, so its groups moved to its
    /// children. It keeps its buffers until they are spilled and evicted.
    Split { buffers: Vec<PartitionGroupBuffer> },
}

/// A slab and the ring slot of the buffer it was allocated from.
struct AllocatedSlab {
    slab: Slab,
    buffer_slot_idx: usize,
}

/// What a partition group slot answered when asked for a slab.
enum SlabAllocation {
    Allocated(AllocatedSlab),
    /// The slot's current buffer has no room for another slab.
    BufferFull,
}

impl PartitionGroupSlot {
    fn buffers_mut(&mut self) -> &mut Vec<PartitionGroupBuffer> {
        match self {
            PartitionGroupSlot::Allocating { buffers, .. }
            | PartitionGroupSlot::Split { buffers } => buffers,
            PartitionGroupSlot::Unused => {
                unreachable!("a slab was allocated from an unused partition group slot")
            }
        }
    }

    /// Allocate `size` bytes aligned to `align` from the current buffer,
    /// taking a first buffer if the slot has none yet. The slab starts on a
    /// spill block boundary and the buffer is used up to the next one after
    /// it.
    fn allocate_slab(&mut self, size: usize, align: usize) -> SlabAllocation {
        if let PartitionGroupSlot::Unused = self {
            self.start_new_buffer();
        }
        let PartitionGroupSlot::Allocating {
            current,
            next_allocation_offset,
            ..
        } = self
        else {
            unreachable!("groups only point at partition group slots that are allocating")
        };
        let start = next_allocation_offset.next_multiple_of(align.max(SPILL_BLOCK_SIZE));
        let end = start + padded_to_spill_block(size);
        if end > BUFFER_SIZE {
            return SlabAllocation::BufferFull;
        }
        *next_allocation_offset = end;
        // `start..end` lies inside the 2MB buffer, checked just above.
        let ptr = unsafe { current.ptr.add(start) };
        SlabAllocation::Allocated(AllocatedSlab {
            slab: Slab::new(ptr, size, SlabBacking::Working(current.clone())),
            buffer_slot_idx: current.slot_idx,
        })
    }

    /// Take a fresh buffer and allocate `size` bytes aligned to `align` from
    /// it. For a slot whose current buffer is full, or has none yet.
    fn allocate_slab_from_new_buffer(&mut self, size: usize, align: usize) -> AllocatedSlab {
        self.start_new_buffer();
        match self.allocate_slab(size, align) {
            SlabAllocation::Allocated(allocated) => allocated,
            SlabAllocation::BufferFull => {
                unreachable!("a slab that fits in a buffer fits in a fresh one")
            }
        }
    }

    /// Start allocating from a fresh buffer, as the slot's first or in place of
    /// a full current one.
    fn start_new_buffer(&mut self) {
        let fresh = Arc::new(memory_ctx().get_write_buffer(false));
        match self {
            PartitionGroupSlot::Unused => {
                *self = PartitionGroupSlot::Allocating {
                    buffers: vec![PartitionGroupBuffer::new(fresh.clone())],
                    current: fresh,
                    next_allocation_offset: 0,
                };
            }
            PartitionGroupSlot::Allocating {
                current,
                next_allocation_offset,
                buffers,
            } => {
                buffers.push(PartitionGroupBuffer::new(fresh.clone()));
                *current = fresh;
                *next_allocation_offset = 0;
            }
            PartitionGroupSlot::Split { .. } => {
                unreachable!("a split partition group slot never allocates again")
            }
        }
    }

    /// Stop allocating here for good, letting go of the current buffer so it
    /// can become sealed. For an inner node whose buffer is full, before
    /// its groups move to its children.
    fn stop_allocating(&mut self) {
        let PartitionGroupSlot::Allocating { buffers, .. } = self else {
            unreachable!("only a partition group slot that is allocating can stop")
        };
        *self = PartitionGroupSlot::Split {
            buffers: std::mem::take(buffers),
        };
    }
}

/// A slab of one partition's rows, as handed out when the allocator is taken
/// apart for reading.
pub struct PartitionSlab {
    pub slab: FrozenSlab,
    /// Rows written into the slab: all it holds for a closed slab, fewer for
    /// the one a partition was still filling.
    pub rows: usize,
}

/// One partition's share of what the allocator hands out when it is taken
/// apart: where its slabs sit in the slab list
/// [`into_partitions`](PartitionAllocator::into_partitions) returns, and the
/// file its spilled rows went to, if it ever spilled.
pub struct PartitionContents {
    /// The partition's first slab in the list, `NO_SLAB` if it has none.
    pub first_slab: u32,
    /// Rows in the partition's slabs.
    pub rows: usize,
    pub spill_file: Option<LocalFile>,
}

/// Stores rows by partition, in slabs allocated from buffers the partitions
/// of a group share. See the module documentation.
pub struct PartitionAllocator<L: Layout> {
    layout: L,
    tree: PartitionTree<PartitionGroupSlot>,
    partitions: Box<[Partition]>,
    /// Each partition's cursor into its open slab, indexed like `partitions`.
    cursors: Box<[RowCursor]>,
    /// Where compressed rows wait for their writes to land, once rows have
    /// been spilled.
    compressed_rows_buffers: Option<CompressedRowsBuffers>,
    /// Every closed slab, in the order they closed, each chained to its
    /// partition's next (see [`SlabEntry`]).
    closed: Vec<SlabEntry>,
}

impl<L: Layout> PartitionAllocator<L> {
    /// An allocator for partitions `0..partition_count`. The per-partition
    /// state is sized to that count rather than to the `PARTITIONS` ceiling,
    /// since every worker builds one of these per query and the whole set
    /// is written and torn down with it.
    pub fn new(layout: L, partition_count: usize) -> Self {
        assert!(
            partition_count <= PARTITIONS,
            "{partition_count} partitions, more than the {PARTITIONS} the allocator holds"
        );
        Self {
            layout,
            tree: PartitionTree::new(|| PartitionGroupSlot::Unused),
            partitions: std::iter::repeat_with(|| Partition {
                open_slab: None,
                slabs_filled: 0,
                first_closed: NO_SLAB,
                last_closed: NO_SLAB,
                rows_closed: 0,
                spill_file: None,
                bytes_spilled: 0,
            })
            .take(partition_count)
            .collect(),
            cursors: vec![
                RowCursor {
                    next_row: std::ptr::null_mut(),
                    rows_left: 0,
                };
                partition_count
            ]
            .into_boxed_slice(),
            compressed_rows_buffers: None,
            closed: Vec::new(),
        }
    }

    /// Reserve the next row of `partition` for the caller to fill in place.
    /// The slab the row is in is closed as soon as this row fills it, so the
    /// buffer is let go of at once rather than when the partition next
    /// receives a row; the row itself stays valid, the ring memory does not
    /// move.
    ///
    /// This is the scatter loop's inner step, so it stays small enough to
    /// inline into that loop and touches only the partition's cursor: opening
    /// and closing a slab, which happen once per slab rather than once per
    /// row, are kept out of line.
    #[inline(always)]
    pub fn reserve(&mut self, partition: usize) -> &mut L::Row {
        if self.cursors[partition].rows_left == 0 {
            self.open_slab_for(partition);
        }
        let row_size = self.layout.row_size();
        let cursor = &mut self.cursors[partition];
        let row = cursor.next_row;
        // The slab holds `rows_left` more rows of the layout's size past this
        // one, aligned for them.
        cursor.next_row = unsafe { row.add(row_size) };
        cursor.rows_left -= 1;
        if cursor.rows_left == 0 {
            self.close_open_slab(partition);
        }
        unsafe { self.layout.row_mut(row) }
    }

    /// Give `partition` a fresh slab to append into, and point its cursor at
    /// the slab's first row.
    #[cold]
    #[inline(never)]
    fn open_slab_for(&mut self, partition: usize) {
        let mut slab = self.allocate_slab_for(partition);
        self.cursors[partition] = RowCursor {
            next_row: slab.slab.as_mut_slice().as_mut_ptr(),
            rows_left: slab.rows,
        };
        self.partitions[partition].open_slab = Some(slab);
    }

    /// Close the slab `partition` has just filled.
    #[cold]
    #[inline(never)]
    fn close_open_slab(&mut self, partition: usize) {
        let full = self.partitions[partition]
            .open_slab
            .take()
            .expect("the slab the row was just reserved in is still open");
        self.close(partition, full);
    }

    /// Rows written into `partition`'s open slab so far; zero without one.
    fn open_slab_len(&self, partition: usize) -> usize {
        match &self.partitions[partition].open_slab {
            Some(open) => open.rows - self.cursors[partition].rows_left,
            None => 0,
        }
    }

    /// Take the allocator apart for reading, once no more rows will be
    /// written: every slab in one list, each partition's slabs chained in the
    /// order they were closed with its open slab last, and each partition's
    /// first slab and row count, indexed by partition. The list is the one
    /// the slabs were recorded in as they closed, so the worker builds
    /// nothing here: a worker scatters into a thousand partitions and more,
    /// and every worker flushes at once, on the query's critical path. A
    /// spilled slab's entry stays in its chain with no slab; its rows are
    /// the evictor's.
    pub fn into_partitions(self) -> (Vec<SlabEntry>, Vec<PartitionContents>) {
        let partition_count = self.partitions.len();
        let open_slab_lens: Vec<usize> = (0..partition_count)
            .map(|partition| self.open_slab_len(partition))
            .collect();
        let Self {
            partitions,
            mut closed,
            ..
        } = self;
        let contents = partitions
            .into_vec()
            .into_iter()
            .zip(open_slab_lens)
            .enumerate()
            .map(|(partition_idx, (mut partition, open_slab_len))| {
                if let Some(open_slab) = partition.open_slab.take()
                    && open_slab_len > 0
                {
                    let entry = closed.len() as u32;
                    closed.push(SlabEntry {
                        slab: Some(open_slab.slab.freeze()),
                        rows: open_slab_len,
                        partition: partition_idx,
                        next: NO_SLAB,
                    });
                    if partition.last_closed == NO_SLAB {
                        partition.first_closed = entry;
                    } else {
                        closed[partition.last_closed as usize].next = entry;
                    }
                    partition.last_closed = entry;
                    partition.rows_closed += open_slab_len;
                }
                PartitionContents {
                    first_slab: partition.first_closed,
                    rows: partition.rows_closed,
                    spill_file: partition.spill_file,
                }
            })
            .collect();
        (closed, contents)
    }

    /// Spill up to `count` sealed buffers through `spiller` and return them
    /// for the caller to keep until they are evicted: their rows stay
    /// readable in memory until then. Fewer are returned, or none, when fewer
    /// are sealed. Buffers are taken in the tree's spill order (see
    /// [`PartitionTree::walk_spill_order`]), oldest first within a node. A
    /// node's current buffer is never sealed, so it is never taken.
    ///
    /// Each partition's rows in the buffers taken go out compressed, packed
    /// into as few writes as hold them, so taking several buffers at once
    /// makes for fewer and larger writes than taking them one by one. No more
    /// buffers are taken than the allocator's buffers for compressed rows
    /// have room for now. A buffer is on disk once every write carrying rows
    /// of it has landed.
    ///
    /// A partition spilling for the first time gets its file opened through
    /// `spiller` here. Should that fail, the buffers stay where they were.
    pub fn spill(
        &mut self,
        spiller: &mut Spiller,
        count: usize,
    ) -> Result<Vec<SpilledBuffer>, IORequesterError> {
        let compressed_rows_buffers = self
            .compressed_rows_buffers
            .get_or_insert_with(CompressedRowsBuffers::take_from_pool);
        compressed_rows_buffers.start_over_in_landed_buffers();
        let count = count.min(compressed_rows_buffers.count_buffers_it_can_compress());
        // Where the sealed buffers are: each one's partition group slot in
        // the tree, and its position among that slot's buffers.
        let mut sealed_buffer_locations: Vec<(usize, usize)> = Vec::new();
        if count > 0 {
            let _: ControlFlow<()> =
                self.tree
                    .walk_spill_order(|partition_group_slot_idx, partition_group_slot| {
                        let slot_buffers = match partition_group_slot {
                            PartitionGroupSlot::Allocating { buffers, .. }
                            | PartitionGroupSlot::Split { buffers } => buffers,
                            PartitionGroupSlot::Unused => return ControlFlow::Continue(()),
                        };
                        for (buffer_idx, buffer) in slot_buffers.iter().enumerate() {
                            if !buffer.is_sealed() {
                                continue;
                            }
                            sealed_buffer_locations.push((partition_group_slot_idx, buffer_idx));
                            if sealed_buffer_locations.len() == count {
                                return ControlFlow::Break(());
                            }
                        }
                        ControlFlow::Continue(())
                    });
        }
        for &(partition_group_slot_idx, buffer_idx) in &sealed_buffer_locations {
            let slot_buffers = self.tree.node_mut(partition_group_slot_idx).buffers_mut();
            for &entry in &slot_buffers[buffer_idx].closed {
                let partition_idx = self.closed[entry as usize].partition;
                let partition = &mut self.partitions[partition_idx];
                if partition.spill_file.is_none() {
                    partition.spill_file = Some(spiller.open_spill_file(partition_idx)?);
                }
            }
        }

        // Take the buffers out of the tree, last first: the locations of one
        // slot are in rising order, so the ones still to take stay where the
        // walk found them. Every closed slab is gathered with the position,
        // in `spilled_buffers`, of the buffer it came from.
        let mut spilled_buffers: Vec<SpilledBuffer> =
            Vec::with_capacity(sealed_buffer_locations.len());
        let mut closed_slabs: Vec<(usize, ClosedSlab)> = Vec::new();
        for &(partition_group_slot_idx, buffer_idx) in sealed_buffer_locations.iter().rev() {
            let PartitionGroupBuffer {
                buffer,
                closed: buffer_closed,
            } = self
                .tree
                .node_mut(partition_group_slot_idx)
                .buffers_mut()
                .remove(buffer_idx);
            let spilled_buffer_idx = spilled_buffers.len();
            for entry in buffer_closed {
                let entry = &mut self.closed[entry as usize];
                let slab = entry
                    .slab
                    .take()
                    .expect("a closed slab is spilled once, from the buffer it was allocated in");
                // The rows now count as spilled, through the handle the
                // spilled buffer gets for them, rather than as rows in the
                // partition's slabs.
                self.partitions[entry.partition].rows_closed -= entry.rows;
                closed_slabs.push((
                    spilled_buffer_idx,
                    ClosedSlab {
                        partition: entry.partition,
                        slab,
                    },
                ));
            }
            spilled_buffers.push(SpilledBuffer {
                writes_in_flight: Arc::new(AtomicUsize::new(0)),
                buffer,
                slabs: Vec::new(),
                partition_group_slot: partition_group_slot_idx,
            });
        }

        // Each partition's rows go out together, so bring its slabs
        // together. Sorting is stable, so the slabs a partition has in one
        // buffer keep the order they were filled in.
        closed_slabs.sort_by_key(|(_, closed_slab)| closed_slab.partition);
        let mut writes: Vec<SpillWrite> = Vec::new();
        for partition_slabs in
            closed_slabs.chunk_by(|(_, slab), (_, next)| slab.partition == next.partition)
        {
            let partition = partition_slabs[0].1.partition;
            self.gather_compressed_writes(
                partition,
                partition_slabs,
                &mut spilled_buffers,
                &mut writes,
            );
        }
        spiller.spill(writes, spilled_buffers.len())?;
        // Back in spill order, which is the order the evictor gives them up in.
        spilled_buffers.reverse();
        Ok(spilled_buffers)
    }

    /// The writes of `partition_slabs`, the slabs `partition` has in the
    /// buffers being spilled: each slab compressed by itself, packed one
    /// after another into a stretch of memory that goes out as one write,
    /// padded at its end to a block boundary. A new write is started when the
    /// next slab might not fit a ring buffer with the ones before it.
    fn gather_compressed_writes(
        &mut self,
        partition: usize,
        partition_slabs: &[(usize, ClosedSlab)],
        spilled_buffers: &mut [SpilledBuffer],
        writes: &mut Vec<SpillWrite>,
    ) {
        COMPRESSED_ROWS_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let scratch_len = get_most_compressed_len(BUFFER_SIZE);
            if scratch.len() < scratch_len {
                scratch.resize(scratch_len, 0);
            }
            // The slabs packed into the scratch so far for the write being
            // gathered: where each one's compressed bytes are in it.
            let mut packed_slabs: Vec<PackedSlab> = Vec::new();
            let mut packed_len = 0;
            for (spilled_buffer_idx, ClosedSlab { slab, .. }) in partition_slabs {
                let most_compressed_len = get_most_compressed_len(slab.size());
                if packed_len > 0 && packed_len + most_compressed_len > BUFFER_SIZE {
                    writes.push(self.finish_compressed_write(
                        partition,
                        &scratch[..packed_len],
                        &packed_slabs,
                        spilled_buffers,
                    ));
                    packed_slabs.clear();
                    packed_len = 0;
                }
                let compressed_len = compress_spilled_rows(
                    slab.as_slice(),
                    &mut scratch[packed_len..packed_len + most_compressed_len],
                );
                packed_slabs.push(PackedSlab {
                    spilled_buffer_idx: *spilled_buffer_idx,
                    start: packed_len,
                    compressed_len,
                    slab,
                });
                packed_len += compressed_len;
            }
            writes.push(self.finish_compressed_write(
                partition,
                &scratch[..packed_len],
                &packed_slabs,
                spilled_buffers,
            ));
        });
    }

    /// The write of `packed`, the compressed bytes of `packed_slabs`, copied
    /// into the allocator's buffers for compressed rows, where the write
    /// holds them until it lands.
    fn finish_compressed_write(
        &mut self,
        partition: usize,
        packed: &[u8],
        packed_slabs: &[PackedSlab],
        spilled_buffers: &mut [SpilledBuffer],
    ) -> SpillWrite {
        let padded_len = padded_to_spill_block(packed.len());
        assert!(
            padded_len <= BUFFER_SIZE,
            "{} bytes of compressed rows do not fit one write",
            packed.len()
        );
        let mut compressed_rows = self
            .compressed_rows_buffers
            .as_mut()
            .expect("rows are compressed only once their buffers have been taken")
            .allocate(padded_len);
        compressed_rows.as_mut_slice()[..packed.len()].copy_from_slice(packed);
        let mut write = self.start_write(partition);
        self.partitions[partition].bytes_spilled += padded_len as u64;
        write.rows.push_slab(compressed_rows);
        for packed_slab in packed_slabs {
            let spilled_buffer = &mut spilled_buffers[packed_slab.spilled_buffer_idx];
            count_write_against_buffer(&mut write, spilled_buffer);
            spilled_buffer.slabs.push(SpilledSlab {
                spill_partition_id: partition,
                file_offset: write.offset + packed_slab.start as u64,
                stored_len: packed_slab.compressed_len,
                slab: packed_slab.slab.downgrade(),
            });
        }
        write
    }

    /// An empty write to `partition`'s file, right after everything the
    /// partition spilled before.
    fn start_write(&self, partition: usize) -> SpillWrite {
        SpillWrite {
            file: self.partitions[partition]
                .spill_file
                .clone()
                .expect("a partition's file is opened before its rows are gathered"),
            offset: self.partitions[partition].bytes_spilled,
            rows: FileBytes::new(),
            buffers_writes_in_flight: Vec::new(),
        }
    }

    /// Close a full slab, recording it against the buffer it was allocated from.
    fn close(&mut self, partition: usize, full: OpenSlab) {
        assert_eq!(
            self.cursors[partition].rows_left, 0,
            "only full slabs are closed"
        );
        self.partitions[partition].slabs_filled += 1;
        let group_buffer = self
            .tree
            .node_mut(full.partition_group_slot)
            .buffers_mut()
            .iter_mut()
            .rev()
            .find(|group_buffer| group_buffer.buffer.slot_idx == full.buffer_slot_idx)
            .expect(
                "a slab's buffer stays recorded at its partition group slot until it is spilled",
            );
        let entry = self.closed.len() as u32;
        group_buffer.closed.push(entry);
        self.closed.push(SlabEntry {
            slab: Some(full.slab.freeze()),
            rows: full.rows,
            partition,
            next: NO_SLAB,
        });
        let partition = &mut self.partitions[partition];
        if partition.last_closed == NO_SLAB {
            partition.first_closed = entry;
        } else {
            self.closed[partition.last_closed as usize].next = entry;
        }
        partition.last_closed = entry;
        partition.rows_closed += full.rows;
    }

    /// Allocate a fresh slab for `partition` from its group's slot. When the
    /// slot's current buffer is full, a leaf takes a new buffer, and an inner
    /// node hands its groups to its children and the slab comes from the
    /// child's first buffer. A slab fits in one buffer, so a fresh buffer
    /// always has room for it.
    fn allocate_slab_for(&mut self, partition: usize) -> OpenSlab {
        let group = group_of_partition(partition);
        let rows = self
            .layout
            .slab_rows(self.partitions[partition].slabs_filled);
        assert!(rows > 0, "a slab must hold at least one row");
        let size = rows * self.layout.row_size();
        let align = self.layout.row_align();
        // A slab is spilled as one compressed block inside one write, and
        // rows that do not compress come out slightly larger than they went
        // in.
        assert!(
            padded_to_spill_block(get_most_compressed_len(size)) <= BUFFER_SIZE,
            "a slab of {rows} rows must fit in one buffer, and in one write once compressed"
        );
        assert!(
            SPILL_BLOCK_SIZE.is_multiple_of(align),
            "a row aligned to {align} bytes cannot start every slab on a spill block boundary"
        );
        let node = self.tree.node_of_group(group);
        let (node, allocated) = match self.tree.node_mut(node).allocate_slab(size, align) {
            SlabAllocation::Allocated(allocated) => (node, allocated),
            SlabAllocation::BufferFull if is_leaf(node) => {
                let allocated = self
                    .tree
                    .node_mut(node)
                    .allocate_slab_from_new_buffer(size, align);
                (node, allocated)
            }
            SlabAllocation::BufferFull => {
                self.tree.node_mut(node).stop_allocating();
                self.tree.split(node);
                let child = self.tree.node_of_group(group);
                let allocated = self
                    .tree
                    .node_mut(child)
                    .allocate_slab_from_new_buffer(size, align);
                (child, allocated)
            }
        };
        OpenSlab {
            slab: allocated.slab,
            rows,
            partition_group_slot: node,
            buffer_slot_idx: allocated.buffer_slot_idx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{IORequester, OperatorIO};
    use crate::memory::init_test_free_pool;
    use crate::memory::spill::TestSpillDirectory;
    use crate::memory::spill::layout::{Dynamic, Fixed};
    use crate::memory::spill::partition_tree::PARTITIONS_PER_GROUP;
    use crate::stats::StatsCollector;
    use crate::worker::WORKER_IDX;
    use std::cell::RefCell;
    use std::sync::atomic::Ordering;

    /// What a `Spiller` borrows, with a spill directory of the test's own so
    /// its spill files clash with no other test's.
    struct SpillerParts {
        dataflow: usize,
        spill_directory: TestSpillDirectory,
        requester: RefCell<IORequester>,
        stats: StatsCollector,
        next_request_id: usize,
    }

    impl SpillerParts {
        fn new(dataflow: usize) -> Self {
            Self {
                dataflow,
                spill_directory: TestSpillDirectory::new(),
                requester: RefCell::new(IORequester::default()),
                stats: StatsCollector::disabled(),
                next_request_id: 0,
            }
        }

        fn spiller(&mut self) -> Spiller<'_> {
            let io = OperatorIO::new(
                &self.requester,
                self.dataflow,
                0,
                &mut self.stats,
                &mut self.next_request_id,
            );
            Spiller::new(io, &self.spill_directory)
        }

        /// Complete every spill write issued so far, counting each off its
        /// buffer as the worker does.
        fn finish_writes(&mut self) {
            let mut requester = self.requester.borrow_mut();
            while memory_ctx().has_spill_writes_in_flight() {
                requester.wait().unwrap();
                requester.collect_completions().unwrap();
                memory_ctx().complete_landed_spill_writes(&mut requester);
            }
        }

        fn file_len(&self, partition: usize) -> u64 {
            let path = self.spill_directory.partition_file_path(
                self.dataflow,
                WORKER_IDX.get(),
                0,
                partition,
            );
            std::fs::metadata(path).unwrap().len()
        }
    }

    impl Drop for SpillerParts {
        fn drop(&mut self) {
            self.finish_writes();
        }
    }

    const ROWS_PER_SLAB: usize = 64;

    /// `u64` rows in slabs of `ROWS_PER_SLAB`.
    fn small_rows() -> PartitionAllocator<Fixed<u64>> {
        PartitionAllocator::new(Fixed::new(ROWS_PER_SLAB * 8), PARTITIONS)
    }

    /// A row large enough that a buffer holds only four slabs of
    /// `ROWS_PER_SLAB`, so a few reserves fill it. Only its size and the
    /// bytes it leaves in memory matter.
    #[derive(Clone, Copy)]
    struct LargeRow {
        _bytes: [u8; BUFFER_SIZE / (4 * ROWS_PER_SLAB)],
    }

    fn large_row(fill: u8) -> LargeRow {
        LargeRow {
            _bytes: [fill; BUFFER_SIZE / (4 * ROWS_PER_SLAB)],
        }
    }

    fn large_rows() -> PartitionAllocator<Fixed<LargeRow>> {
        PartitionAllocator::new(Fixed::new(BUFFER_SIZE / 4), PARTITIONS)
    }

    /// A free pool of `row_buffers` buffers for rows, and the buffers a
    /// spilling allocator keeps for compressed rows besides.
    fn init_free_pool_for_spilling(row_buffers: usize) {
        init_test_free_pool(row_buffers + COMPRESSED_ROWS_BUFFER_COUNT);
    }

    fn buffers_of_slot(
        allocator: &mut PartitionAllocator<impl Layout>,
        node: usize,
    ) -> &Vec<PartitionGroupBuffer> {
        allocator.tree.node_mut(node).buffers_mut()
    }

    /// Reserve a row in the first partition of each of five groups: four
    /// slabs fill the root buffer, and the fifth finds it full and splits the
    /// root.
    fn fill_and_split_root(allocator: &mut PartitionAllocator<Fixed<LargeRow>>) {
        for group in 0..5 {
            allocator
                .reserve(group * PARTITIONS_PER_GROUP)
                .write(large_row(1));
        }
    }

    /// Fill the four slabs in the root buffer, so it becomes sealed.
    fn fill_root_slabs(allocator: &mut PartitionAllocator<Fixed<LargeRow>>) {
        for group in 0..4 {
            for _ in 1..ROWS_PER_SLAB {
                allocator
                    .reserve(group * PARTITIONS_PER_GROUP)
                    .write(large_row(1));
            }
        }
    }

    #[test]
    fn a_partition_fills_one_slab_before_starting_the_next() {
        init_test_free_pool(1);
        let mut allocator = small_rows();

        for value in 0..ROWS_PER_SLAB as u64 + 1 {
            allocator.reserve(3).write(value);
        }

        let closed = buffers_of_slot(&mut allocator, 0)[0].closed.clone();
        assert_eq!(closed.len(), 1);
        let entry = &allocator.closed[closed[0] as usize];
        assert_eq!(entry.partition, 3);
        let values: Vec<u64> = entry
            .slab
            .as_ref()
            .unwrap()
            .as_slice()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| u64::from_ne_bytes(*bytes))
            .collect();
        assert_eq!(values, (0..ROWS_PER_SLAB as u64).collect::<Vec<_>>());
        assert_eq!(allocator.open_slab_len(3), 1);
    }

    #[test]
    #[should_panic(expected = "in one write once compressed")]
    fn a_slab_that_might_not_fit_one_write_once_compressed_is_refused() {
        init_test_free_pool(1);
        let mut allocator: PartitionAllocator<Fixed<u64>> =
            PartitionAllocator::new(Fixed::new(BUFFER_SIZE), PARTITIONS);

        allocator.reserve(0).write(1);
    }

    #[test]
    fn a_full_slab_is_closed_on_the_reserve_that_fills_it() {
        init_test_free_pool(1);
        let mut allocator = small_rows();

        for value in 0..ROWS_PER_SLAB as u64 {
            allocator.reserve(0).write(value);
        }

        assert!(allocator.partitions[0].open_slab.is_none());
        assert_eq!(allocator.partitions[0].slabs_filled, 1);
        assert_eq!(buffers_of_slot(&mut allocator, 0)[0].closed.len(), 1);
    }

    #[test]
    fn a_dynamic_layout_hands_out_byte_rows_in_growing_slabs() {
        init_test_free_pool(1);
        let layout = Dynamic::new(24, 8, 2 * 24, 8 * 24);
        let mut allocator = PartitionAllocator::new(layout, PARTITIONS);

        for row in 0..6u8 {
            allocator.reserve(0).fill(row);
        }

        let closed: Vec<&FrozenSlab> = buffers_of_slot(&mut allocator, 0)[0]
            .closed
            .clone()
            .iter()
            .map(|&entry| allocator.closed[entry as usize].slab.as_ref().unwrap())
            .collect();
        let slab_sizes: Vec<usize> = closed.iter().map(|slab| slab.size()).collect();
        assert_eq!(slab_sizes, [2 * 24, 4 * 24]);
        assert_eq!(closed[1].as_slice()[0..24], [2u8; 24]);
        assert_eq!(closed[1].as_slice()[72..96], [5u8; 24]);
        assert!(allocator.partitions[0].open_slab.is_none());
    }

    #[test]
    fn partitions_of_every_group_share_the_root_buffer_at_first() {
        init_test_free_pool(1);
        let mut allocator = small_rows();

        allocator.reserve(0).write(1);
        allocator.reserve(PARTITIONS - 1).write(2);

        assert_eq!(buffers_of_slot(&mut allocator, 0).len(), 1);
        let first = allocator.partitions[0].open_slab.as_ref().unwrap();
        let last = allocator.partitions[PARTITIONS - 1]
            .open_slab
            .as_ref()
            .unwrap();
        assert_eq!(first.buffer_slot_idx, last.buffer_slot_idx);
    }

    #[test]
    fn an_inner_node_whose_buffer_is_full_splits_its_groups_between_its_children() {
        init_test_free_pool(3);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);

        allocator.reserve(PARTITIONS - 1).write(large_row(2));

        assert_eq!(allocator.tree.node_of_group(0), 1);
        assert_eq!(allocator.tree.node_of_group(63), 2);
        assert!(matches!(
            allocator.tree.node_mut(0),
            PartitionGroupSlot::Split { .. }
        ));
        let fifth_group = allocator.partitions[4 * PARTITIONS_PER_GROUP]
            .open_slab
            .as_ref()
            .unwrap();
        let last_group = allocator.partitions[PARTITIONS - 1]
            .open_slab
            .as_ref()
            .unwrap();
        assert_eq!(fifth_group.partition_group_slot, 1);
        assert_eq!(last_group.partition_group_slot, 2);
        assert_ne!(fifth_group.buffer_slot_idx, last_group.buffer_slot_idx);
    }

    #[test]
    fn a_leaf_whose_buffer_is_full_moves_to_a_new_buffer() {
        init_test_free_pool(16);
        let mut allocator = large_rows();
        // Partition 0 alone: four slabs per buffer, one buffer per inner node
        // on the way down to leaf 63, then two buffers at the leaf.
        let inner_nodes_on_path = 6;
        let slabs = 4 * inner_nodes_on_path + 5;

        for _ in 0..slabs * ROWS_PER_SLAB {
            allocator.reserve(0).write(large_row(3));
        }

        assert_eq!(allocator.tree.node_of_group(0), 63);
        assert_eq!(buffers_of_slot(&mut allocator, 63).len(), 2);
    }

    #[test]
    fn a_buffer_is_sealed_only_once_every_slab_in_it_is_full() {
        init_test_free_pool(3);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);
        let before = buffers_of_slot(&mut allocator, 0)[0].is_sealed();

        fill_root_slabs(&mut allocator);

        let after = buffers_of_slot(&mut allocator, 0)[0].is_sealed();
        assert!(!before);
        assert!(after);
    }

    #[test]
    fn a_buffer_still_being_allocated_from_is_never_sealed() {
        init_test_free_pool(1);
        let mut allocator = small_rows();

        for value in 0..ROWS_PER_SLAB as u64 {
            allocator.reserve(0).write(value);
        }

        assert!(!buffers_of_slot(&mut allocator, 0)[0].is_sealed());
    }

    #[test]
    fn spill_writes_a_sealed_buffer_once_with_one_write_per_partition() {
        init_free_pool_for_spilling(3);
        let mut parts = SpillerParts::new(9_000_010);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);
        fill_root_slabs(&mut allocator);

        let spilled = allocator
            .spill(&mut parts.spiller(), 1)
            .unwrap()
            .pop()
            .unwrap();
        let again = allocator.spill(&mut parts.spiller(), 1).unwrap();
        let writes_issued = spilled.writes_in_flight.load(Ordering::Acquire);
        parts.finish_writes();

        assert_eq!(writes_issued, 4);
        assert_eq!(spilled.writes_in_flight.load(Ordering::Acquire), 0);
        let partitions: Vec<usize> = spilled.slabs.iter().map(|s| s.spill_partition_id).collect();
        assert_eq!(partitions, [0, 64, 128, 192]);
        assert!(spilled.slabs.iter().all(|s| s.file_offset == 0));
        assert!(
            spilled
                .slabs
                .iter()
                .all(|s| s.slab.upgrade().unwrap().as_slice()[0] == 1)
        );
        // The writes have landed and let go; the spilled buffer is the one
        // holder left.
        assert_eq!(Arc::strong_count(&spilled.buffer), 1);
        assert!(again.is_empty());
        assert!(buffers_of_slot(&mut allocator, 0).is_empty());
        // A slab of identical rows compresses to a fraction of a block.
        for group in 0..4 {
            assert_eq!(
                parts.file_len(group * PARTITIONS_PER_GROUP),
                SPILL_BLOCK_SIZE as u64
            );
        }
    }

    #[test]
    fn spill_opens_a_partitions_file_the_first_time_and_reuses_it_after() {
        init_free_pool_for_spilling(8);
        let mut parts = SpillerParts::new(9_000_011);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);
        fill_root_slabs(&mut allocator);
        allocator
            .spill(&mut parts.spiller(), 1)
            .unwrap()
            .pop()
            .unwrap();
        let file_after_first = allocator.partitions[0].spill_file.clone().unwrap();
        // Partition 0's group moved to the root's left child, whose buffer
        // already holds another partition's open slab, so three more slabs
        // fill it. The four after those fill the next node's buffer by
        // themselves, and one more row moves on, leaving that buffer sealed.
        for _ in 0..(3 + 4) * ROWS_PER_SLAB + 1 {
            allocator.reserve(0).write(large_row(2));
        }

        let second = allocator
            .spill(&mut parts.spiller(), 1)
            .unwrap()
            .pop()
            .unwrap();
        parts.finish_writes();

        assert_eq!(second.writes_in_flight.load(Ordering::Acquire), 0);
        assert_eq!(
            allocator.partitions[0].spill_file.clone().unwrap(),
            file_after_first
        );
        // The second spill's four slabs are packed one after another in one
        // write, which starts on the block after the first spill's write.
        let block = SPILL_BLOCK_SIZE as u64;
        let offsets: Vec<u64> = second.slabs.iter().map(|s| s.file_offset).collect();
        let ends: Vec<u64> = second
            .slabs
            .iter()
            .map(|s| s.file_offset + s.stored_len as u64)
            .collect();
        assert_eq!(offsets[0], block);
        assert_eq!(offsets[1..], ends[..3]);
        assert_eq!(parts.file_len(0), 2 * block);
        assert!(allocator.partitions[64].spill_file.is_some());
        assert!(allocator.partitions[1].spill_file.is_none());
    }

    #[test]
    fn buffers_spilled_together_share_one_write_per_partition() {
        init_free_pool_for_spilling(8);
        let mut parts = SpillerParts::new(9_000_013);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);
        fill_root_slabs(&mut allocator);
        // Partition 0's group moved to the root's left child, whose buffer
        // already holds another partition's open slab, so three more slabs
        // fill it. The four after those fill the next node's buffer by
        // themselves, and one more row moves on, leaving that buffer sealed
        // beside the root's.
        for _ in 0..(3 + 4) * ROWS_PER_SLAB + 1 {
            allocator.reserve(0).write(large_row(2));
        }

        let spilled = allocator.spill(&mut parts.spiller(), 2).unwrap();
        let writes_issued = parts.next_request_id;
        let writes_per_buffer: Vec<usize> = spilled
            .iter()
            .map(|buffer| buffer.writes_in_flight.load(Ordering::Acquire))
            .collect();
        parts.finish_writes();

        // Partition 0 has four slabs in one buffer and one in the root's, and
        // the root's other three slabs are of three more partitions.
        assert_eq!(spilled.len(), 2);
        assert_eq!(writes_issued, 4);
        assert_eq!(writes_per_buffer, [1, 4]);
        assert!(
            spilled
                .iter()
                .all(|buffer| buffer.writes_in_flight.load(Ordering::Acquire) == 0)
        );
        assert_eq!(parts.file_len(0), SPILL_BLOCK_SIZE as u64);
    }

    #[test]
    fn spill_takes_a_deeper_buffer_before_the_root() {
        init_free_pool_for_spilling(8);
        let mut parts = SpillerParts::new(9_000_012);
        let mut allocator = large_rows();
        fill_and_split_root(&mut allocator);
        fill_root_slabs(&mut allocator);
        // Do the same to the root's right child, which groups 32..64 moved to.
        for group in 32..37 {
            allocator
                .reserve(group * PARTITIONS_PER_GROUP)
                .write(large_row(2));
        }
        for group in 32..36 {
            for _ in 1..ROWS_PER_SLAB {
                allocator
                    .reserve(group * PARTITIONS_PER_GROUP)
                    .write(large_row(2));
            }
        }

        allocator
            .spill(&mut parts.spiller(), 1)
            .unwrap()
            .pop()
            .unwrap();
        let after_first: Vec<bool> = [32 * PARTITIONS_PER_GROUP, 0]
            .iter()
            .map(|&partition| allocator.partitions[partition].spill_file.is_some())
            .collect();
        allocator
            .spill(&mut parts.spiller(), 1)
            .unwrap()
            .pop()
            .unwrap();
        let after_second = allocator.partitions[0].spill_file.is_some();

        assert_eq!(after_first, [true, false]);
        assert!(after_second);
    }
}
