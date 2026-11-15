//! The [`Cached`] contract a value type meets to live in a
//! [`Cache<V>`](super::Cache).

use crate::memory::slab::FrozenSlab;

/// A value a [`Cache<V>`](super::Cache) can hold.
///
/// A cached value is position-independent data living in ring slots: no
/// pointers into itself and no heap. At rest the cache keeps only where its
/// bytes are plus a small heap-side [`Meta`](Self::Meta); on a hit it pins
/// those bytes and rebuilds the value over them. The value then holds the
/// pins, and dropping it releases them.
pub trait Cached: Sized + Send + Sync + 'static {
    /// What the cache keeps on the heap beside the placements: whatever
    /// rebuilding the value needs that is not in the bytes themselves.
    type Meta: Send + Sync + 'static;

    /// Rebuild the value over its bytes. `slabs` are the slabs the builder
    /// carved into the fill, in carving order, pinned.
    fn from_slabs(meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self;
}
