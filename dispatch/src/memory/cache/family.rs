//! A family: caches holding the same bytes in different forms.
//!
//! ```text
//!   family "pages"
//!   ┌──────────────┬─────────┐
//!   │ compressed   │ 30 %    │
//!   │ decompressed │ 70 %    │
//!   └──────────────┴─────────┘
//! ```
//!
//! Two things follow from being in one family, and nothing else does:
//!
//! - **Shares.** Each member has a target share of the family's slots. When
//!   the family is to give a slot, the evictor moves the hand of a member
//!   holding more than its target, so the balance between the forms is held
//!   to the targets instead of being whatever their lifetimes add up to.
//!   Shares are judged within the family only; the family's total against the
//!   rest of the ring is left to recency and weight.
//! - **Reinforcement.** When a member drops an entry, every sibling's entry
//!   over the same file range is reinforced: with one copy gone the others are
//!   closer to being the last, and losing the last turns the next read into IO.
//!
//! A family with a single member has neither; it is just a cache with a weight.

use crate::memory::cache::id::CacheId;

/// One member of a family and the share of the family's slots it should hold.
#[derive(Clone, Copy, Debug)]
pub struct FamilyMember {
    pub cache: CacheId,
    pub target_share_pct: u8,
}

/// The caches that hold one kind of data in its different forms.
#[derive(Clone, Debug)]
pub struct FamilySpec {
    members: Vec<FamilyMember>,
}

impl FamilySpec {
    /// `members` are `(cache, target share percent)`; the shares must sum to
    /// 100.
    pub fn new(members: impl IntoIterator<Item = (CacheId, u8)>) -> Self {
        let members: Vec<FamilyMember> = members
            .into_iter()
            .map(|(cache, target_share_pct)| FamilyMember {
                cache,
                target_share_pct,
            })
            .collect();
        assert!(!members.is_empty(), "a family needs at least one cache");
        let total: u32 = members
            .iter()
            .map(|member| member.target_share_pct as u32)
            .sum();
        assert_eq!(total, 100, "a family's target shares must sum to 100");
        Self { members }
    }

    pub fn members(&self) -> impl Iterator<Item = CacheId> + '_ {
        self.members.iter().map(|member| member.cache)
    }

    pub fn count_members(&self) -> usize {
        self.members.len()
    }

    pub fn contains(&self, id: CacheId) -> bool {
        self.members().any(|member| member == id)
    }

    /// Every member but `id`.
    pub fn siblings_of(&self, id: CacheId) -> impl Iterator<Item = CacheId> + '_ {
        self.members().filter(move |member| *member != id)
    }

    /// The member whose hand the evictor should move next, `None` when no
    /// member has a slot to give.
    ///
    /// The first choice is the first member holding strictly more than its
    /// target share of the family's slots, and the last member when none
    /// does. Members are added from the raw bytes to the forms derived from
    /// them, so on the boundary it is a derived form that gives way: losing
    /// it costs rebuilding it, not IO. The strictness also keeps the choice
    /// from flipping back and forth on the boundary.
    ///
    /// `sweeps_given_up` is how many times this eviction has swept a member
    /// for as long as it could give something without getting a slot. Each
    /// moves the choice on by one member, round and round. A member that
    /// `can_give` says holds nothing the evictor could take is passed over at
    /// once.
    ///
    /// `slots_held` reports how many slots a member holds in the region being
    /// evicted from. It is read while peers bind and release slots, so the
    /// choice is as approximate as the counts.
    pub fn choose_member_to_evict(
        &self,
        slots_held: impl Fn(CacheId) -> usize,
        can_give: impl Fn(CacheId) -> bool,
        sweeps_given_up: usize,
    ) -> Option<CacheId> {
        let family_slots: usize = self.members().map(&slots_held).sum();
        let first_over_target = self.members.iter().position(|member| {
            slots_held(member.cache) * 100 > member.target_share_pct as usize * family_slots
        });
        let first_choice = first_over_target.unwrap_or(self.members.len() - 1);
        (0..self.members.len())
            .map(|turn| {
                let member = (first_choice + sweeps_given_up + turn) % self.members.len();
                self.members[member].cache
            })
            .find(|&member| can_give(member))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPRESSED: CacheId = CacheId::new(0);
    const DECOMPRESSED: CacheId = CacheId::new(1);
    const DECODED: CacheId = CacheId::new(2);

    fn pages() -> FamilySpec {
        FamilySpec::new([(COMPRESSED, 30), (DECOMPRESSED, 70)])
    }

    fn held(compressed: usize, decompressed: usize) -> impl Fn(CacheId) -> usize {
        move |id| {
            if id == COMPRESSED {
                compressed
            } else {
                decompressed
            }
        }
    }

    fn everyone_can_give(_: CacheId) -> bool {
        true
    }

    #[test]
    #[should_panic(expected = "sum to 100")]
    fn shares_must_sum_to_one_hundred() {
        FamilySpec::new([(COMPRESSED, 30), (DECOMPRESSED, 60)]);
    }

    #[test]
    fn the_raw_bytes_give_while_over_their_target_share() {
        let family = pages();

        let chosen = family.choose_member_to_evict(held(1, 1), everyone_can_give, 0); // 50%

        assert_eq!(chosen, Some(COMPRESSED));
    }

    #[test]
    fn the_derived_form_gives_while_the_raw_bytes_are_under_their_target_share() {
        let family = pages();

        let chosen = family.choose_member_to_evict(held(1, 10), everyone_can_give, 0); // 9%

        assert_eq!(chosen, Some(DECOMPRESSED));
    }

    #[test]
    fn exactly_on_target_the_derived_form_gives() {
        let family = pages();

        let chosen = family.choose_member_to_evict(held(3, 7), everyone_can_give, 0);

        assert_eq!(chosen, Some(DECOMPRESSED));
    }

    #[test]
    fn a_member_with_nothing_to_give_is_passed_over() {
        let family = pages();

        let chosen = family.choose_member_to_evict(held(1, 1), |id| id == DECOMPRESSED, 0);

        assert_eq!(chosen, Some(DECOMPRESSED));
    }

    #[test]
    fn each_sweep_given_up_moves_the_choice_to_the_next_member() {
        let family = pages();

        let after_one = family.choose_member_to_evict(held(1, 1), everyone_can_give, 1);
        let after_two = family.choose_member_to_evict(held(1, 1), everyone_can_give, 2);

        assert_eq!(after_one, Some(DECOMPRESSED));
        assert_eq!(after_two, Some(COMPRESSED));
    }

    #[test]
    fn a_family_with_nothing_to_give_has_no_member_to_evict() {
        let family = pages();

        let chosen = family.choose_member_to_evict(held(1, 1), |_| false, 0);

        assert_eq!(chosen, None);
    }

    #[test]
    fn a_single_member_family_gives_from_its_member() {
        let family = FamilySpec::new([(DECODED, 100)]);

        let chosen = family.choose_member_to_evict(|_| 5, everyone_can_give, 0);

        assert_eq!(chosen, Some(DECODED));
    }

    #[test]
    fn with_three_members_the_first_one_over_its_target_gives() {
        let family = FamilySpec::new([(COMPRESSED, 20), (DECOMPRESSED, 50), (DECODED, 30)]);
        let slots = |id: CacheId| match id {
            COMPRESSED => 1,
            DECOMPRESSED => 6,
            _ => 3,
        }; // decompressed 60% of 10

        let chosen = family.choose_member_to_evict(slots, everyone_can_give, 0);

        assert_eq!(chosen, Some(DECOMPRESSED));
    }

    #[test]
    fn siblings_exclude_the_cache_itself() {
        let family = pages();

        let siblings: Vec<_> = family.siblings_of(COMPRESSED).collect();

        assert_eq!(siblings, vec![DECOMPRESSED]);
    }
}
