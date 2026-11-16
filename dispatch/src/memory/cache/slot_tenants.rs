//! [`SlotTenants`]: per ring slot, the keys of every entry with bytes in it.
//!
//! The reverse of a cache's [`FileMap`](super::file_map::FileMap): the map answers
//! "where is this entry", the tenant list answers "what is in this slot", which
//! is what an evictor holding only a slot index needs.
//!
//! One lockless list per slot. A slot's list is written only by the single
//! worker filling the slot, while that worker holds its fill pin, and taken
//! only by the evictor after `try_write` succeeded, which requires every pin
//! to be gone. The pin drop's `Release` and `try_write`'s `Acquire` are the
//! happens-before, so the lists need no lock.
//!
//! A list may hold stale keys: an entry that was dropped another way, lost a
//! duplicate-insert race, or whose build failed after its slabs were carved.
//! The evictor filters them by checking each key against the map.

use std::cell::UnsafeCell;

pub struct SlotTenants<Key> {
    lists: Box<[UnsafeCell<Vec<Key>>]>,
}

// SAFETY: each list is touched by exactly one party at a time under the
// pin/exclusivity discipline in the module doc.
unsafe impl<Key: Send> Send for SlotTenants<Key> {}
unsafe impl<Key: Send> Sync for SlotTenants<Key> {}

impl<K> SlotTenants<K> {
    /// One list per ring slot.
    pub fn new(slot_count: usize) -> Self {
        Self {
            lists: (0..slot_count)
                .map(|_| UnsafeCell::new(Vec::new()))
                .collect(),
        }
    }

    /// Record `key` as having bytes in `slot`. Filler only, under the fill pin.
    pub fn push(&self, slot: usize, key: K) {
        self.list_mut(slot).push(key);
    }

    /// Take `slot`'s list, leaving it empty. Evictor only, under `try_write`.
    pub fn take(&self, slot: usize) -> Vec<K> {
        std::mem::take(self.list_mut(slot))
    }

    /// Forget `slot`'s list. Either party, while holding the slot exclusively.
    pub fn clear(&self, slot: usize) {
        self.list_mut(slot).clear();
    }

    #[allow(clippy::mut_from_ref)] // interior mutability; see the module doc
    fn list_mut(&self, slot: usize) -> &mut Vec<K> {
        unsafe { &mut *self.lists[slot].get() }
    }
}
