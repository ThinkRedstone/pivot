//! A worker's bump pointer over the ring slot it is currently packing entries
//! of one cache into. Each worker owns one cursor per cache (see
//! [`MemoryContext::fill_cursor`](crate::memory::context::MemoryContext::fill_cursor)).
//!
//! ```text
//!   slot ────────────────────────────────────────────────────────┐
//!   │ entry │ entry │ entry │ entry │        free               │
//!   └───────────────────────┬───────┴───────────────────────────┘
//!                           next_byte
//! ```
//!
//! The slot is held as a reader pin, so entries already packed into it stay
//! readable and the evictor cannot take it while the worker keeps filling its
//! tail. Rotating to a fresh slot is the cache's job, since each cache resets
//! its own per-slot metadata first; the cursor only records where it is.

use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use crate::memory::write_buffer::WriteBuffer;
use std::sync::Arc;

pub struct FillCursor {
    /// The slot being packed, pinned. `None` before the first entry and after
    /// the cursor is released.
    pin: Option<Arc<ReadBuffer>>,
    slot: usize,
    /// The next free byte in the slot, `0..=BUFFER_SIZE`.
    next_byte: usize,
}

impl FillCursor {
    /// A cursor pointing nowhere: the next entry rotates to a fresh slot.
    pub fn empty() -> Self {
        Self {
            pin: None,
            slot: 0,
            next_byte: 0,
        }
    }

    /// Whether the cursor must rotate before handing out another byte.
    pub(crate) fn is_exhausted(&self) -> bool {
        self.pin.is_none() || self.next_byte == BUFFER_SIZE
    }

    /// Bytes left in the slot, zero when pointing nowhere.
    pub(crate) fn remaining(&self) -> usize {
        if self.pin.is_none() {
            0
        } else {
            BUFFER_SIZE - self.next_byte
        }
    }

    pub(crate) fn slot(&self) -> usize {
        self.slot
    }

    pub(crate) fn next_byte(&self) -> usize {
        self.next_byte
    }

    /// The pin on the slot being packed. Only meaningful after a rotation.
    pub(crate) fn pin(&self) -> Arc<ReadBuffer> {
        self.pin
            .clone()
            .expect("the cursor was rotated onto a slot")
    }

    /// Drop the pin, making the slot ordinarily evictable. It stays bound to
    /// its cache; the next entry rotates to a fresh slot.
    pub(crate) fn release(&mut self) {
        self.pin = None;
    }

    /// Point at `write_buffer`, a fresh slot held exclusively whose cache
    /// metadata the caller has just reset, and publish it as readable.
    pub(crate) fn rotate_to(&mut self, write_buffer: WriteBuffer) {
        self.slot = write_buffer.slot_idx;
        self.next_byte = 0;
        self.pin = Some(Arc::new(ReadBuffer::from(write_buffer))); // used = 1, Release
    }

    /// Take the next `len` bytes, returning their offset in the slot. `len`
    /// must not exceed `remaining`.
    pub(crate) fn take(&mut self, len: usize) -> usize {
        debug_assert!(len <= self.remaining());
        let offset = self.next_byte;
        self.next_byte += len;
        offset
    }
}
