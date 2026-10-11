use crate::memory::BUFFER_SIZE;
use crate::memory::slab::Slab;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// Typed buffer backed by one or more [`Slab`]s, supporting allocations that span
/// multiple 2MB `WriteBuffer`s.
///
/// Created via [`super::SlabAllocator::create_multi_slab_buffer`]. The backing slabs are
/// *separate* 2MB allocations, not one contiguous run of memory, so each slab holds a whole
/// number of elements — `BUFFER_SIZE / size_of::<T>()` of them — with the few leftover bytes
/// at each slab's tail unused. Element `i` therefore lives in slab `i / elems_per_slab` at
/// offset `(i % elems_per_slab) * size_of::<T>()`. (Indexing the slabs as one contiguous byte
/// run instead — `byte_offset >> 21` — would let any element whose size doesn't divide 2MB
/// straddle a slab boundary and read past the slab into unrelated memory.) `elems_per_slab` is
/// a constant for a given `T`, so the div/mod lower to a multiply+shift.
///
/// Supports `Index<usize>` and `IndexMut<usize>` for typed element access. The buffer does
/// not track its logical length — callers must track how many elements have been written. This
/// means accessing indexes that have not yet been set is UB.
pub struct MultiSlabBuffer<T> {
    slabs: Vec<Slab>,
    _phantom: PhantomData<T>,
}

impl<T> MultiSlabBuffer<T> {
    /// Wraps existing slabs into a `MultiSlabBuffer`. Each slab must start at offset 0
    /// of its `WriteBuffer` for indexing to work correctly.
    pub fn new(slabs: Vec<Slab>) -> Self {
        Self {
            slabs,
            _phantom: Default::default(),
        }
    }

    /// Returns a raw pointer to the element at `index`.
    ///
    /// Each slab holds `elems_per_slab = BUFFER_SIZE / size_of::<T>()` elements packed from its
    /// start, so the element lives in slab `index / elems_per_slab` at byte offset
    /// `(index % elems_per_slab) * size_of::<T>()` — never straddling a slab boundary even when
    /// `size_of::<T>()` doesn't divide `BUFFER_SIZE`.
    pub fn ptr_at_index(&self, index: usize) -> *mut T {
        let elems_per_slab = BUFFER_SIZE / size_of::<T>();
        let slab_idx = index / elems_per_slab;
        let offset_in_slab = (index % elems_per_slab) * size_of::<T>();
        unsafe { self.slabs[slab_idx].ptr.add(offset_in_slab) as *mut T }
    }

    /// The number of elements from `index` to the end of the slab holding it:
    /// how far a contiguous write starting at `ptr_at_index(index)` may run.
    pub fn elements_left_in_slab(&self, index: usize) -> usize {
        let elems_per_slab = BUFFER_SIZE / size_of::<T>();
        elems_per_slab - index % elems_per_slab
    }

    /// Zero every backing slab, returning the buffer to its as-created state. Used
    /// to reuse a hash table's storage after its entries have been drained
    /// elsewhere, without reallocating from the slab pool.
    pub fn zero_out(&mut self) {
        for slab in &mut self.slabs {
            slab.zero_out();
        }
    }
}

impl<T> Index<usize> for MultiSlabBuffer<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.ptr_at_index(index) }
    }
}

impl<T> IndexMut<usize> for MultiSlabBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.ptr_at_index(index) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{BUFFER_SIZE, SlabAllocator, init_test_free_pool};

    #[test]
    fn write_and_read() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut buf: MultiSlabBuffer<u32> = alloc.create_multi_slab_buffer(64, true);

        buf[0] = 10;
        buf[63] = 20;

        assert_eq!(buf[0], 10);
        assert_eq!(buf[63], 20);
    }

    #[test]
    fn zeroed_on_creation() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);

        let buf: MultiSlabBuffer<u64> = alloc.create_multi_slab_buffer(256, true);

        for i in 0..256 {
            assert_eq!(buf[i], 0);
        }
    }

    #[test]
    fn elements_left_in_slab_counts_down_to_the_slab_boundary() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let per_slab = BUFFER_SIZE / size_of::<u64>();
        let buf: MultiSlabBuffer<u64> = alloc.create_multi_slab_buffer(per_slab + 10, true);

        assert_eq!(buf.elements_left_in_slab(0), per_slab);
        assert_eq!(buf.elements_left_in_slab(per_slab - 1), 1);
        assert_eq!(buf.elements_left_in_slab(per_slab), per_slab);
        assert_eq!(buf.elements_left_in_slab(per_slab + 3), per_slab - 3);
    }

    #[test]
    fn spans_multiple_buffers() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let elements = BUFFER_SIZE / size_of::<u64>() + 100;

        let mut buf: MultiSlabBuffer<u64> = alloc.create_multi_slab_buffer(elements, true);
        buf[0] = 1;
        buf[elements - 1] = 2;

        assert_eq!(buf[0], 1);
        assert_eq!(buf[elements - 1], 2);
    }

    /// An element whose size does NOT divide `BUFFER_SIZE` (e.g. a 24-byte hash-table
    /// `Entry`) must not straddle a slab boundary: 2MB / 24 = 87381.33, so treating the
    /// separate slabs as one contiguous byte run made element 87381 read past slab 0 into
    /// unrelated memory, corrupting its tail (high-cardinality GROUP BY counts). Write a
    /// distinct pattern to every element across the boundary and read it all back.
    #[test]
    fn non_divisor_element_size_does_not_straddle_slab_boundary() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        #[repr(C)]
        struct Entry {
            hash: u64,
            key: u64,
            value: u64,
        }
        assert_ne!(
            BUFFER_SIZE % size_of::<Entry>(),
            0,
            "test needs a non-divisor size"
        );

        init_test_free_pool(8);
        let mut alloc = SlabAllocator::new(true);
        let elems_per_slab = BUFFER_SIZE / size_of::<Entry>();
        // Span well past the first slab boundary, including the straddling index.
        let n = elems_per_slab + 50;
        let mut buf: MultiSlabBuffer<Entry> = alloc.create_multi_slab_buffer(n, true);

        let pat = |i: usize| Entry {
            hash: i as u64 + 1,
            key: (i as u64) ^ 0xA5A5_5A5A,
            value: i as u64 * 7,
        };
        for i in 0..n {
            buf[i] = pat(i);
        }
        for i in 0..n {
            assert_eq!(buf[i], pat(i), "corruption at index {i}");
        }
    }
}
