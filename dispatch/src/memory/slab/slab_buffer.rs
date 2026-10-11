use crate::memory::slab::Slab;
use arrow_buffer::Buffer;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;
use std::sync::Arc;

/// Typed buffer backed by a single [`Slab`] (must fit within one 2MB `WriteBuffer`).
///
/// Created via [`super::SlabAllocator::create_slab_buffer`]. Cheaper to index than [`super::MultiSlabBuffer`]
/// since there's no slab lookup — just a single pointer offset.
///
/// Supports `Index<usize>` and `IndexMut<usize>` for typed element access. Like `MultiSlabBuffer`,
/// does not track its logical length — callers must track how many elements have been written, and
/// it is UB to access an index not yet set.
pub struct SlabBuffer<T> {
    slab: Slab,
    _phantom: PhantomData<T>,
}

impl<T> SlabBuffer<T> {
    /// Wraps a slab into a typed `SlabBuffer`.
    pub fn new(slab: Slab) -> Self {
        Self {
            slab,
            _phantom: Default::default(),
        }
    }

    /// Returns a raw pointer to the element at `index`.
    /// Simple pointer arithmetic — no slab lookup needed.
    #[inline(always)]
    pub fn ptr_at_index(&self, index: usize) -> *mut T {
        let byte_offset = index * size_of::<T>();
        unsafe { self.slab.ptr.add(byte_offset) as *mut T }
    }

    /// The number of elements from `index` to the end of the buffer: how far a
    /// contiguous write starting at `ptr_at_index(index)` may run. `index` must
    /// not exceed the buffer's capacity.
    pub fn elements_left(&self, index: usize) -> usize {
        self.slab.size / size_of::<T>() - index
    }

    /// Hand the buffer's first `byte_len` bytes to Arrow as a zero-copy
    /// [`Buffer`]. The slab rides in the buffer's allocation `Arc`, so the
    /// memory returns to the pool when the last downstream reference drops.
    pub(crate) fn into_buffer(self, byte_len: usize) -> Buffer {
        let ptr = NonNull::new(self.slab.ptr).unwrap();
        // SAFETY: the slab owns at least `byte_len` bytes at `ptr` and lives as
        // long as the returned buffer via the custom-allocation `Arc`.
        unsafe { Buffer::from_custom_allocation(ptr, byte_len, Arc::new(self.slab)) }
    }
}

impl<T> Index<usize> for SlabBuffer<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.ptr_at_index(index) }
    }
}

impl<T> IndexMut<usize> for SlabBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.ptr_at_index(index) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{SlabAllocator, init_test_free_pool};

    #[test]
    fn write_and_read() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut buf: SlabBuffer<u64> = alloc.create_slab_buffer(16, true);

        buf[0] = 42;
        buf[15] = 99;

        assert_eq!(buf[0], 42);
        assert_eq!(buf[15], 99);
    }

    #[test]
    fn zeroed_on_creation() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);

        let buf: SlabBuffer<u64> = alloc.create_slab_buffer(128, true);

        for i in 0..128 {
            assert_eq!(buf[i], 0);
        }
    }

    #[test]
    fn elements_left_counts_down_to_the_buffer_capacity() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let buf: SlabBuffer<u64> = alloc.create_slab_buffer(128, true);

        assert_eq!(buf.elements_left(0), 128);
        assert_eq!(buf.elements_left(100), 28);
        assert_eq!(buf.elements_left(128), 0);
    }
}
