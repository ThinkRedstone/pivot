//! A slot's validity bitmap: which 4 KB blocks of it have been read in.
//!
//! The compressed cache publishes an extent before its bytes arrive (so other
//! readers can follow the fill in flight), so unlike a derived cache it needs a
//! per-block "landed" flag beside the map. The memory ordering that makes the
//! in-place fill sound lives here.

use crate::memory::cache::compressed::extent::BLOCKS_PER_SLOT;
use std::sync::atomic::{AtomicU64, Ordering};

/// Number of `u64` words covering a slot's blocks (8 for 512 blocks).
const WORDS: usize = BLOCKS_PER_SLOT / 64;

#[derive(Default)]
pub(crate) struct ValidBitmap([AtomicU64; WORDS]);

impl ValidBitmap {
    /// Is slot block `slot_block` present? `Acquire` pairs with `set`'s
    /// `Release`, so a reader that observes the bit sees the block's bytes.
    pub fn is_set(&self, slot_block: usize) -> bool {
        self.0[slot_block / 64].load(Ordering::Acquire) & (1 << (slot_block % 64)) != 0
    }

    /// Whether every block in `[first, first + count)` is present.
    pub fn all_set(&self, first: usize, count: usize) -> bool {
        (first..first + count).all(|slot_block| self.is_set(slot_block))
    }

    /// Mark slot blocks `[first, first + count)` present. `Release` so it only
    /// becomes visible after the bytes have landed in the slot.
    pub fn set(&self, first: usize, count: usize) {
        for slot_block in first..first + count {
            self.0[slot_block / 64].fetch_or(1 << (slot_block % 64), Ordering::Release);
        }
    }

    /// Reset every block to absent. Sound only while the slot is held
    /// exclusively.
    pub fn clear(&self) {
        for word in &self.0 {
            word.store(0, Ordering::Relaxed);
        }
    }
}
