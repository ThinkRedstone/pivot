//! Builders that accumulate values on slab memory and finish into an Arrow
//! array with zero copies. [`ArrayBuilder`] is the interface the parquet page
//! decoders and the GROUP BY output write through; [`PrimitiveBuilder`] builds
//! fixed-width primitive arrays, [`BooleanBuilder`] boolean arrays and
//! [`ViewBuilder`] byte-view arrays, and [`ValidityBuilder`] the null bitmap
//! that goes with any of them.

mod boolean_builder;
mod primitive_builder;
mod validity_builder;
mod view_builder;

use arrow_array::ArrayRef;
use arrow_buffer::Buffer;
pub use boolean_builder::BooleanBuilder;
pub use primitive_builder::PrimitiveBuilder;
pub use validity_builder::ValidityBuilder;
pub use view_builder::ViewBuilder;

use crate::memory::SlabAllocator;

/// Accumulates values into engine memory and produces a finished Arrow array.
pub trait ArrayBuilder {
    /// `Default` is the element written under null slots: the slab is not
    /// zeroed on allocation, and downstream kernels (and unsafe array
    /// constructors) may touch masked slots, so they must hold a valid value.
    type Element: Copy + Default;

    /// Creates a builder pre-allocated for `capacity` elements.
    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self;

    /// Number of elements pushed so far.
    fn len(&self) -> usize;

    /// Whether no elements have been pushed yet.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends `element` repeated `amount` times (used by RLE runs).
    fn push(&mut self, element: &Self::Element, amount: usize);

    /// Returns a mutable slice of `count` uninitialised slots at the end of
    /// the buffer, advancing the length. Callers must fill every slot.
    fn spare_mut(&mut self, count: usize) -> &mut [Self::Element];

    /// Consumes the builder and returns the finished Arrow array.
    fn into_array(self, null_buffer: Option<Buffer>) -> ArrayRef;
}
