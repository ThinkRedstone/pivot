//! Naming a cache: its [`CacheId`] and the [`CachePolicy`] that says how hard
//! its slots are to drop.

/// Index of one cache in the [`Caches`](super::Caches) registry.
///
/// Stored per ring slot in the clock's owner byte as `index + 1`, with 0
/// meaning "no cache owns this slot", so it is deliberately one byte: a
/// revolution of the clock walks every slot's record.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CacheId(u8);

impl CacheId {
    /// At most this many caches can be registered: one owner byte per slot,
    /// minus the free marker.
    pub const MAX_CACHES: usize = u8::MAX as usize;

    pub(crate) const fn new(index: u8) -> Self {
        Self(index)
    }

    /// Position in the registry's cache list.
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// How the clock stores this id in a slot's owner byte.
    pub(crate) fn to_owner_byte(self) -> u8 {
        self.0 + 1
    }

    /// The id a slot's owner byte names, `None` for the free marker.
    pub(crate) fn from_owner_byte(byte: u8) -> Option<Self> {
        byte.checked_sub(1).map(Self)
    }
}

/// How a cache's slots behave in the shared CLOCK.
///
/// Every slot loses one life per pass of its cache's hand; a cache whose
/// entries should outlive others' at equal recency gets more lives to spend.
#[derive(Clone, Debug)]
pub struct CachePolicy {
    /// What the cache is called in system tables and logs.
    pub name: &'static str,
    /// Lives a slot starts with when the cache binds it.
    pub lives_at_insert: u8,
    /// Lives a slot gains each time one of its entries is read.
    pub lives_per_touch: u8,
    /// Lives a slot gains per read once a sibling cache dropped its copy of
    /// the same bytes, granted once at that moment too. Higher than
    /// `lives_per_touch`, so a copy that is still being read and is closer to
    /// being the last one climbs while equally hot backed copies sink. Only
    /// meaningful for a cache with siblings.
    pub lives_at_reinforce: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_owner_byte_round_trips_and_keeps_zero_for_free() {
        let id = CacheId::new(3);

        let byte = id.to_owner_byte();

        assert_eq!(CacheId::from_owner_byte(byte), Some(id));
        assert_eq!(CacheId::from_owner_byte(0), None);
    }
}
