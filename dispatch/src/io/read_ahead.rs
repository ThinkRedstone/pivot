//! Bounds the remote bytes one worker has asked for and not yet started on.
//!
//! A worker's budget is what its share of the network card receives in the
//! time a remote read takes. A worker holding that much asked for is either
//! keeping its share of the card busy or never short of data:
//!
//! - When it processes data faster than its share of the card delivers, what
//!   arrives is taken up at once, so the whole budget is on its way, and a
//!   budget on its way arrives at exactly the worker's share of the card's
//!   rate.
//! - When it processes data slower than that, each row group it starts on
//!   frees room for one more, so it asks at the rate it processes. The rest
//!   of the budget is ready for it before it is needed: received, or held by
//!   the store until the worker reads its sockets.
//!
//! The read time is the store's, and it is not one number: a store answering
//! slower needs more asked for to keep the same rate, and what was asked for
//! while it was slow arrives on top of new requests once it speeds up.
//!
//! The process's receive rate limit still decides when each read goes out (see
//! [`super::receive_rate`]). This only decides how much a worker asks for.

use crate::worker::WORKER_IDX;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// How long a remote read is taken to last, from its request to its last byte.
/// Taking it longer than it is has a worker ask for more than the card can
/// deliver meanwhile, and the receive rate limit holds the excess back. Taking
/// it shorter leaves the card idle. Read from `PIVOT_REMOTE_READ_TIME_MS`.
const DEFAULT_REMOTE_READ_TIME_MILLIS: u64 = 100;

/// One worker's budget of remote bytes asked for and not yet started on.
pub struct ReadAheadBudget {
    budget_bytes: usize,
    /// The bytes charged and not yet released. Shared with every charge,
    /// because another worker may be the one to start on a row group this
    /// worker asked for.
    outstanding_bytes: Arc<AtomicUsize>,
    /// The worker the budget belongs to.
    worker_idx: usize,
}

impl ReadAheadBudget {
    /// The budget of one of `worker_count` workers sharing a network card that
    /// receives `bytes_per_second`.
    pub fn for_receive_rate(
        bytes_per_second: u64,
        read_time: Duration,
        worker_count: usize,
    ) -> Self {
        let card_bytes = bytes_per_second as u128 * read_time.as_nanos() / 1_000_000_000;
        Self::of_bytes(card_bytes as usize / worker_count)
    }

    /// The budget of the worker this runs on: its share of the rate the
    /// process requests remote reads at.
    pub(crate) fn for_this_worker() -> Self {
        let read_time_millis = crate::env::get_env_var_with_default(
            "PIVOT_REMOTE_READ_TIME_MS",
            DEFAULT_REMOTE_READ_TIME_MILLIS,
        );
        Self::for_receive_rate(
            super::receive_rate::process_receive_rate_limiter().bytes_per_second(),
            Duration::from_millis(read_time_millis),
            crate::worker::NUM_WORKERS.get(),
        )
    }

    /// A budget of `budget_bytes` for the worker this runs on.
    pub fn of_bytes(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            outstanding_bytes: Arc::new(AtomicUsize::new(0)),
            worker_idx: WORKER_IDX.get(),
        }
    }

    /// Whether the worker has asked for less than its budget. One more charge
    /// may then take it past the budget: a worker asking for too little leaves
    /// the card idle, and one asking for too much only has reads held back.
    /// A worker that has asked for nothing always has room, so a budget
    /// smaller than one read still lets reads out one at a time.
    pub fn has_room(&self) -> bool {
        let outstanding_bytes = self.outstanding_bytes.load(Ordering::Relaxed);
        outstanding_bytes < self.budget_bytes || outstanding_bytes == 0
    }

    /// Count `bytes` against the budget until the returned charge is dropped.
    pub fn charge(&self, bytes: usize) -> ReadAheadCharge {
        self.outstanding_bytes.fetch_add(bytes, Ordering::Relaxed);
        ReadAheadCharge {
            outstanding_bytes: self.outstanding_bytes.clone(),
            bytes,
            worker_idx: self.worker_idx,
        }
    }
}

/// Bytes counted against a worker's [`ReadAheadBudget`]. Dropping it gives
/// them back, from whichever worker drops it.
pub struct ReadAheadCharge {
    outstanding_bytes: Arc<AtomicUsize>,
    bytes: usize,
    worker_idx: usize,
}

impl Drop for ReadAheadCharge {
    fn drop(&mut self) {
        self.outstanding_bytes
            .fetch_sub(self.bytes, Ordering::Relaxed);
        // The budget's worker may be asleep with nothing of its own to wake
        // it, and it now has room to ask for more.
        if WORKER_IDX.get() != self.worker_idx {
            crate::waker::waker_set().notify_worker(self.worker_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_budget_has_room_until_its_charges_reach_it() {
        let budget = ReadAheadBudget::of_bytes(100);

        let _first = budget.charge(60);
        let had_room_below_budget = budget.has_room();
        let _second = budget.charge(60);

        assert!(had_room_below_budget);
        assert!(!budget.has_room());
    }

    #[test]
    fn dropping_a_charge_gives_its_bytes_back() {
        let budget = ReadAheadBudget::of_bytes(100);
        let charge = budget.charge(150);

        drop(charge);

        assert!(budget.has_room());
    }

    /// A budget smaller than any read still lets one read out at a time.
    #[test]
    fn an_empty_budget_has_room_while_nothing_is_charged() {
        let budget = ReadAheadBudget::of_bytes(0);

        let had_room_with_nothing_charged = budget.has_room();
        let _charge = budget.charge(1);

        assert!(had_room_with_nothing_charged);
        assert!(!budget.has_room());
    }

    #[test]
    fn a_worker_gets_its_share_of_what_the_card_receives_during_a_read() {
        let budget = ReadAheadBudget::for_receive_rate(1_000_000, Duration::from_millis(100), 4);

        let _charge = budget.charge(24_999);
        let had_room_below_share = budget.has_room();
        let _last_byte = budget.charge(1);

        assert!(had_room_below_share);
        assert!(!budget.has_room());
    }
}
