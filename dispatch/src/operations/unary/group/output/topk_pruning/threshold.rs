//! The shared k-th best exact group value that merge jobs prune against.

use crate::operations::unary::group::GroupLimit;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Shared lower bound on the pushed top-k's k-th best exact group value.
///
/// Every fully merged group's weight is offered from whichever worker merged
/// its partition; once `cap` groups have been seen, the smallest retained
/// weight is the true global k-th best so far, which partition jobs compare
/// their bound against to skip themselves. Global rather than per-worker on
/// purpose: the k best groups land in partitions spread across the pool, so
/// no single worker's local top-k ever tightens to the real k-th value.
///
/// Offers are gated on a relaxed load of the current minimum, so after the
/// first `cap` groups almost every offer is one uncontended atomic read.
pub struct TopKThreshold {
    /// The k-th best weight once `cap` groups have been offered; 0 before,
    /// which no bound is below, so nothing is skipped prematurely.
    kth_best: AtomicU64,
    /// Min-heap of the `cap` largest weights offered so far.
    top: Mutex<BinaryHeap<Reverse<u64>>>,
    cap: usize,
    /// Groups emitted so far under a plain (unordered) pushed LIMIT. Any
    /// `limit` groups satisfy it, so once this reaches the limit every
    /// remaining merge job skips itself.
    emitted_groups: AtomicUsize,
}

impl TopKThreshold {
    /// A threshold for the pushed limit; an absent limit gets a
    /// zero-capacity threshold that never fills and never skips anything.
    pub fn for_limit(output_limit: Option<GroupLimit>) -> Self {
        let cap = match output_limit {
            Some(GroupLimit::TopK { limit, .. }) => limit,
            _ => 0,
        };
        Self {
            kth_best: AtomicU64::new(0),
            top: Mutex::new(BinaryHeap::with_capacity(cap.min(1 << 20))),
            cap,
            emitted_groups: AtomicUsize::new(0),
        }
    }

    /// Records `groups` more emitted under a plain pushed LIMIT.
    #[inline]
    pub fn add_emitted_groups(&self, groups: usize) {
        self.emitted_groups.fetch_add(groups, Ordering::Relaxed);
    }

    /// Groups emitted so far under a plain pushed LIMIT.
    #[inline]
    pub fn emitted_groups(&self) -> usize {
        self.emitted_groups.load(Ordering::Relaxed)
    }

    /// Offers one fully merged group's weight.
    #[inline(always)]
    pub fn offer(&self, weight: u64) {
        // Full and not an improvement: the common case after warmup.
        let kth = self.kth_best.load(Ordering::Relaxed);
        if kth != 0 && weight <= kth {
            return;
        }
        self.offer_under_lock(weight);
    }

    /// Adds a weight that may raise the k-th best.
    ///
    /// Never inlined. The caller is a loop over every group of a table, and
    /// the lock and the heap inside it would otherwise take the registers the
    /// loop keeps its own values in.
    #[inline(never)]
    fn offer_under_lock(&self, weight: u64) {
        let mut top = self.top.lock().unwrap();
        if top.len() < self.cap {
            top.push(Reverse(weight));
            if top.len() == self.cap {
                self.kth_best
                    .store(top.peek().unwrap().0, Ordering::Relaxed);
            }
        } else if self.cap > 0 && weight > top.peek().unwrap().0 {
            *top.peek_mut().unwrap() = Reverse(weight);
            self.kth_best
                .store(top.peek().unwrap().0, Ordering::Relaxed);
        }
    }

    /// The current lower bound on the k-th best group value (0 until `cap`
    /// groups have been offered).
    #[inline]
    pub fn kth_best(&self) -> u64 {
        self.kth_best.load(Ordering::Relaxed)
    }
}
