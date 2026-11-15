//! Keeping spilled buffers until their ring slots are actually needed.
//!
//! Spilling writes a buffer's rows to disk; it does not free the buffer. The
//! buffer stays resident, and its rows readable, in the [`PartitionEvictor`],
//! laid out in the same [`PartitionTree`] its slabs were allocated from. The
//! memory context reaches the evictor through the [`Evict`] handle an operator
//! registers at build time, and takes a slot only when it has none to give:
//! the first buffer, in spill order, whose writes have all landed and that no
//! merge job is reading.
//!
//! The evictor is the one holder of its buffers. The merge gets weak handles
//! to the spilled slabs ([`spilled_slabs_by_partition`]), upgrades the ones it
//! reads for as long as it reads them, and falls back to the partition file
//! for a slab whose buffer is gone, so eviction goes on through the merge
//! without the two ever having to agree on anything but a reference count.
//!
//! [`spilled_slabs_by_partition`]: PartitionEvictor::spilled_slabs_by_partition

use crate::memory::evict::Evict;
use crate::memory::spill::partition_tree::PartitionTree;
use crate::memory::spill::{SpilledBuffer, SpilledSlab};
use crate::memory::{WriteBuffer, memory_ctx};
use std::cell::RefCell;
use std::ops::ControlFlow;
use std::rc::Rc;
use std::sync::Arc;

/// Spilled buffers awaiting eviction, each under the partition group slot it
/// was allocated from, and the slabs of the buffers already evicted, whose
/// rows exist only in the partition files now.
pub struct PartitionEvictor {
    tree: PartitionTree<Vec<SpilledBuffer>>,
    evicted: Vec<SpilledSlab>,
}

/// The evictor as the operator and the memory context share it. The two never
/// hold it at the same time: the operator pushes into it between allocations,
/// and the memory context evicts from it inside an allocation, when the
/// operator holds no borrow.
pub type SharedPartitionEvictor = Rc<RefCell<PartitionEvictor>>;

impl PartitionEvictor {
    pub fn new() -> Self {
        Self {
            tree: PartitionTree::new(Vec::new),
            evicted: Vec::new(),
        }
    }

    /// Keep `spilled` until it is evicted.
    pub fn push(&mut self, spilled: SpilledBuffer) {
        self.tree
            .node_mut(spilled.partition_group_slot)
            .push(spilled);
    }

    /// Give up the first buffer, in spill order, whose rows are all on disk
    /// and that nothing else holds, or `None` when there is no such buffer
    /// right now: every spilled buffer still has writes in flight, or a merge
    /// job is reading it.
    pub fn evict(&mut self) -> Option<WriteBuffer> {
        let candidate = self.tree.walk_spill_order(|_, spilled_buffers| {
            match spilled_buffers
                .iter()
                .position(|spilled| spilled.is_on_disk() && Arc::strong_count(&spilled.buffer) == 1)
            {
                Some(idx) => ControlFlow::Break(spilled_buffers.remove(idx)),
                None => ControlFlow::Continue(()),
            }
        });
        let ControlFlow::Break(SpilledBuffer {
            writes_in_flight,
            buffer,
            slabs,
            partition_group_slot,
        }) = candidate
        else {
            return None;
        };
        let mut write_buffer = match Arc::try_unwrap(buffer) {
            Ok(write_buffer) => write_buffer,
            // A merge job upgraded its handle between the count above and
            // here; the buffer stays, and the caller asks again.
            Err(buffer) => {
                self.tree
                    .node_mut(partition_group_slot)
                    .push(SpilledBuffer {
                        writes_in_flight,
                        buffer,
                        slabs,
                        partition_group_slot,
                    });
                return None;
            }
        };
        // From here on these rows can only be read from the partition files.
        self.evicted.extend(slabs);
        // The handle still says whether the slot was zeroed when the buffer
        // was taken for rows; it has been filled with them since. The next
        // taker gets a dirty buffer, so one that wants zeros zeroes it.
        write_buffer.zeroed = false;
        memory_ctx()
            .ring()
            .set_slot_zeroed(write_buffer.slot_idx, false);
        memory_ctx().record_spilled_buffers_released(1);
        Some(write_buffer)
    }

    /// Every spilled slab, by partition, for the merge: the slab's handle
    /// upgrades for as long as its buffer is resident, and the slab's rows are
    /// at the given offset of the partition's file for when it is not. Slabs
    /// of buffers already evicted come with a handle that no longer upgrades.
    /// An empty list when nothing spilled, which is the usual case, so a
    /// flush then builds nothing for it.
    pub fn spilled_slabs_by_partition(&mut self, partitions: usize) -> Vec<Vec<SpilledSlab>> {
        if self.evicted.is_empty() && self.tree.nodes().iter().all(Vec::is_empty) {
            return Vec::new();
        }
        let mut by_partition: Vec<Vec<SpilledSlab>> = (0..partitions).map(|_| Vec::new()).collect();
        let _: ControlFlow<()> = self.tree.walk_spill_order(|_, spilled_buffers| {
            for slab in spilled_buffers.iter().flat_map(|spilled| &spilled.slabs) {
                by_partition[slab.spill_partition_id].push(slab.clone());
            }
            ControlFlow::Continue(())
        });
        for slab in &self.evicted {
            by_partition[slab.spill_partition_id].push(slab.clone());
        }
        by_partition
    }
}

impl Default for PartitionEvictor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PartitionEvictor {
    /// An evictor dropped while still holding spilled buffers, as a failed
    /// dataflow's is, frees them with it; they stop counting as spilled, or
    /// the memory context would go on seeing slots to reclaim that are gone.
    fn drop(&mut self) {
        let mut held = 0;
        let _: ControlFlow<()> = self.tree.walk_spill_order(|_, spilled_buffers| {
            held += spilled_buffers.len();
            ControlFlow::Continue(())
        });
        if held > 0 {
            memory_ctx().record_spilled_buffers_released(held);
        }
    }
}

impl Evict for SharedPartitionEvictor {
    fn evict(&self) -> Option<WriteBuffer> {
        self.borrow_mut().evict()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::FileRange;
    use crate::memory::slab::SlabBacking;
    use crate::memory::{Slab, init_test_free_pool};
    use std::sync::atomic::AtomicUsize;

    /// A spilled buffer with one 1KB slab of partition 0, filed under
    /// `partition_group_slot`, with `writes_in_flight` of its writes still to
    /// land. Counted as spilled with the memory context, as the spiller would
    /// have.
    fn spilled(partition_group_slot: usize, writes_in_flight: usize) -> SpilledBuffer {
        memory_ctx().record_buffers_spilled(1);
        let buffer = Arc::new(memory_ctx().get_write_buffer(false));
        let slab = Slab::new(buffer.ptr, 1024, SlabBacking::Working(buffer.clone())).freeze();
        SpilledBuffer {
            writes_in_flight: Arc::new(AtomicUsize::new(writes_in_flight)),
            slabs: vec![SpilledSlab {
                spill_partition_id: 0,
                file_offset: 0,
                stored_len: slab.size(),
                slab: slab.downgrade(),
            }],
            buffer,
            partition_group_slot,
        }
    }

    #[test]
    fn nothing_is_evicted_while_a_buffers_writes_are_in_flight() {
        init_test_free_pool(1);
        let mut evictor = PartitionEvictor::new();
        evictor.push(spilled(0, 1));

        let evicted = evictor.evict();

        assert!(evicted.is_none());
    }

    #[test]
    fn an_evicted_slabs_handle_no_longer_upgrades_but_still_says_where_its_rows_are() {
        init_test_free_pool(2);
        let mut evictor = PartitionEvictor::new();
        evictor.push(spilled(0, 0));
        evictor.push(spilled(0, 1));

        evictor.evict().unwrap();
        let handles = evictor.spilled_slabs_by_partition(1).remove(0);
        let resident: Vec<bool> = handles
            .iter()
            .map(|handle| handle.slab.upgrade().is_some())
            .collect();
        let ranges: Vec<FileRange> = handles
            .iter()
            .map(|handle| FileRange::new(handle.file_offset as usize, handle.slab.size()))
            .collect();

        assert_eq!(resident, [true, false]);
        assert_eq!(ranges, [FileRange::new(0, 1024), FileRange::new(0, 1024)]);
    }

    #[test]
    fn a_buffer_whose_writes_landed_is_given_up_whole() {
        init_test_free_pool(1);
        let mut evictor = PartitionEvictor::new();
        let on_disk = spilled(0, 0);
        let slot_idx = on_disk.buffer.slot_idx;
        evictor.push(on_disk);

        let evicted = evictor.evict();
        let again = evictor.evict();

        assert_eq!(evicted.map(|buffer| buffer.slot_idx), Some(slot_idx));
        assert!(again.is_none());
    }

    #[test]
    fn eviction_takes_a_deeper_slot_before_the_root() {
        init_test_free_pool(2);
        let mut evictor = PartitionEvictor::new();
        let at_root = spilled(0, 0);
        let at_right_child = spilled(2, 0);
        let (root_slot, child_slot) = (at_root.buffer.slot_idx, at_right_child.buffer.slot_idx);
        evictor.push(at_root);
        evictor.push(at_right_child);

        let first = evictor.evict().unwrap();
        let second = evictor.evict().unwrap();

        assert_eq!(first.slot_idx, child_slot);
        assert_eq!(second.slot_idx, root_slot);
    }

    #[test]
    fn a_buffer_whose_slab_a_reader_holds_is_not_evicted_until_it_is_let_go() {
        init_test_free_pool(1);
        let mut evictor = PartitionEvictor::new();
        evictor.push(spilled(0, 0));
        let handle = evictor.spilled_slabs_by_partition(1).remove(0).remove(0);

        let held = handle.slab.upgrade();
        let while_held = evictor.evict();
        drop(held);
        let after = evictor.evict();

        assert!(while_held.is_none());
        assert!(after.is_some());
    }

    #[test]
    fn the_shared_handle_evicts_through_the_evict_trait() {
        init_test_free_pool(1);
        let shared: SharedPartitionEvictor = Rc::new(RefCell::new(PartitionEvictor::new()));
        shared.borrow_mut().push(spilled(5, 0));
        let handle: Box<dyn Evict> = Box::new(shared.clone());

        let evicted = handle.evict();

        assert!(evicted.is_some());
    }
}
