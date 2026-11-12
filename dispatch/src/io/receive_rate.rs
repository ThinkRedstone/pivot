//! Bounds how fast remote reads are requested by what the network card can
//! receive.
//!
//! Requesting more than the card may carry does not make bytes arrive sooner.
//! The excess is dropped on the way in, the store resends it, and a response
//! that loses its packets waits on retransmission timers. So every worker
//! takes the length of a range read from one shared allowance before issuing
//! it, and holds the read back while the allowance is owing.
//!
//! The allowance grows with time at the card's rate and never past a ceiling,
//! a few milliseconds of that rate. Over any stretch of time the reads issued
//! therefore add up to no more than the card receives in it, plus one ceiling
//! and one read: reads go out at once after a quiet spell, and at the card's
//! rate from then on. Nothing has to tick for the allowance to grow. It is kept as the moment
//! it will next be full, which taking from it pushes into the future and the
//! passing of time alone brings back.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// How much of the card's rate the allowance holds at most, as a time. Enough
/// that workers starting together are not made to queue behind each other for
/// no reason, and small enough that what is released at once is a burst the
/// card absorbs.
const DEFAULT_ALLOWANCE_CEILING_MILLIS: u64 = 2;

/// The share of the card's rate remote reads may be requested at, in percent.
/// Just below the whole of it, because the limit counts the bytes of each
/// read's body and the connection carries a little more around them (TLS,
/// HTTP and TCP framing, about 2%). Asked for the whole rate in body bytes,
/// the card is sent more than it may carry and drops packets for as long as
/// the reads keep coming. Read from `PIVOT_NETWORK_RATE_PERCENT`.
const DEFAULT_RATE_PERCENT: u64 = 97;

const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// The rate the network card is taken to receive at when nothing says what it
/// receives: no rate was configured and the machine is not an instance type
/// with a published one. Low enough to be safe on most cards, so a machine
/// with a faster one reads remote data slower than it could until its rate is
/// set.
pub const ASSUMED_NETWORK_BITS_PER_SECOND: u64 = 5_000_000_000;

/// The allowance every worker of the process takes remote reads from. Set once,
/// before the workers start; left unset, it is made for a card receiving
/// [`ASSUMED_NETWORK_BITS_PER_SECOND`] when the first worker asks for it.
static PROCESS_LIMITER: OnceLock<Arc<ReceiveRateLimiter>> = OnceLock::new();

/// Limit the rate remote reads are requested at to what a network card
/// receiving `bits_per_second` can take. The first call decides for the
/// process: it has one card, and the workers read the limit as they start.
///
/// `PIVOT_NETWORK_RATE_PERCENT` scales the rate and `PIVOT_NETWORK_BURST_MS`
/// sets the allowance ceiling, for tuning on a given machine.
pub fn limit_network_receive_rate(bits_per_second: u64) {
    PROCESS_LIMITER.get_or_init(|| Arc::new(ReceiveRateLimiter::for_card(bits_per_second)));
}

/// The process's limiter: the one [`limit_network_receive_rate`] set, or one
/// for a card of the assumed rate if nothing set it. Each worker also sizes
/// its budget of remote bytes from it (see [`super::read_ahead`]). Only Linux
/// holds reads back by it.
pub(crate) fn process_receive_rate_limiter() -> Arc<ReceiveRateLimiter> {
    PROCESS_LIMITER
        .get_or_init(|| {
            tracing::warn!(
                assumed_bits_per_second = ASSUMED_NETWORK_BITS_PER_SECOND,
                "the network card's receive rate was not set, remote reads are limited to an \
                 assumed rate"
            );
            Arc::new(ReceiveRateLimiter::for_card(
                ASSUMED_NETWORK_BITS_PER_SECOND,
            ))
        })
        .clone()
}

/// A shared allowance of bytes that refills at a fixed rate up to a ceiling.
pub struct ReceiveRateLimiter {
    bytes_per_second: u64,
    /// The ceiling as the time it takes to refill from empty.
    ceiling_nanos: u64,
    /// What the nanosecond stamps below count from.
    started_at: Instant,
    /// When the allowance will be back at its ceiling if nothing more is
    /// taken, in nanoseconds since `started_at`. In the past while it is full.
    full_at_nanos: AtomicU64,
}

impl ReceiveRateLimiter {
    pub fn new(bytes_per_second: u64, ceiling: Duration) -> Self {
        assert!(
            bytes_per_second > 0,
            "a receive rate limit needs a rate above zero"
        );
        Self {
            bytes_per_second,
            ceiling_nanos: ceiling.as_nanos() as u64,
            started_at: Instant::now(),
            full_at_nanos: AtomicU64::new(0),
        }
    }

    /// The limiter for a network card receiving `bits_per_second`: the share
    /// of that rate remote reads may be requested at, with the allowance
    /// ceiling, both as the environment sets them.
    fn for_card(bits_per_second: u64) -> Self {
        let rate_percent = crate::env::get_env_var_with_default(
            "PIVOT_NETWORK_RATE_PERCENT",
            DEFAULT_RATE_PERCENT,
        );
        let ceiling_millis = crate::env::get_env_var_with_default(
            "PIVOT_NETWORK_BURST_MS",
            DEFAULT_ALLOWANCE_CEILING_MILLIS,
        );
        Self::new(
            bits_per_second / 8 * rate_percent / 100,
            Duration::from_millis(ceiling_millis),
        )
    }

    /// The rate the allowance refills at.
    pub(crate) fn bytes_per_second(&self) -> u64 {
        self.bytes_per_second
    }

    /// Take `len` bytes from the allowance as of `now`, or report how long
    /// until a read may be taken.
    ///
    /// A read is taken whenever the allowance is not owing, whatever its
    /// length, and leaves the allowance owing whatever it did not cover. What
    /// goes out at once is therefore the ceiling plus one read at most, and a
    /// long read waits no longer than a short one: were a read to need the
    /// allowance to cover it, a stream of short reads from other workers
    /// would keep the allowance below a long one for as long as it lasted.
    pub fn try_take(&self, len: usize, now: Instant) -> Result<(), Duration> {
        let now_nanos = now.saturating_duration_since(self.started_at).as_nanos() as u64;
        let cost_nanos = (len as u128 * NANOS_PER_SECOND / self.bytes_per_second as u128) as u64;
        let mut full_at_nanos = self.full_at_nanos.load(Ordering::Relaxed);
        loop {
            if full_at_nanos > now_nanos + self.ceiling_nanos {
                return Err(Duration::from_nanos(
                    full_at_nanos - now_nanos - self.ceiling_nanos,
                ));
            }
            let refills_from_nanos = full_at_nanos.max(now_nanos);
            match self.full_at_nanos.compare_exchange_weak(
                full_at_nanos,
                refills_from_nanos + cost_nanos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(current) => full_at_nanos = current,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEGABYTE: usize = 1_000_000;

    /// 100 MB/s with a 10 ms ceiling: the allowance holds 1 MB.
    fn build_limiter() -> ReceiveRateLimiter {
        ReceiveRateLimiter::new(100 * MEGABYTE as u64, Duration::from_millis(10))
    }

    #[test]
    fn a_full_allowance_covers_reads_up_to_its_ceiling() {
        let limiter = build_limiter();
        let now = Instant::now();

        let first_half = limiter.try_take(MEGABYTE / 2, now);
        let second_half = limiter.try_take(MEGABYTE / 2, now);

        assert_eq!(first_half, Ok(()));
        assert_eq!(second_half, Ok(()));
    }

    #[test]
    fn a_read_is_taken_until_the_allowance_is_owing() {
        let limiter = build_limiter();
        let now = Instant::now();
        limiter.try_take(MEGABYTE, now).unwrap();

        let from_an_empty_allowance = limiter.try_take(MEGABYTE / 4, now);
        let from_an_owing_allowance = limiter.try_take(1, now);

        assert_eq!(from_an_empty_allowance, Ok(()));
        assert_eq!(from_an_owing_allowance, Err(Duration::from_micros(2500)));
    }

    #[test]
    fn what_the_allowance_owes_is_paid_off_at_the_rate() {
        let limiter = build_limiter();
        let now = Instant::now();
        limiter.try_take(MEGABYTE, now).unwrap();
        limiter.try_take(MEGABYTE / 4, now).unwrap();

        let just_before = limiter.try_take(1, now + Duration::from_micros(2499));
        let once_paid_off = limiter.try_take(1, now + Duration::from_micros(2500));

        assert_eq!(just_before, Err(Duration::from_micros(1)));
        assert_eq!(once_paid_off, Ok(()));
    }

    #[test]
    fn a_quiet_spell_refills_no_further_than_the_ceiling() {
        let limiter = build_limiter();
        let much_later = Instant::now() + Duration::from_secs(60);
        limiter.try_take(MEGABYTE, much_later).unwrap();
        limiter.try_take(1, much_later).unwrap();

        let past_the_ceiling = limiter.try_take(1, much_later);

        assert_eq!(past_the_ceiling, Err(Duration::from_nanos(10)));
    }

    #[test]
    fn a_long_read_waits_no_longer_than_a_short_one() {
        let limiter = build_limiter();
        let now = Instant::now();
        limiter.try_take(MEGABYTE, now).unwrap();
        limiter.try_take(MEGABYTE / 4, now).unwrap();

        let long_read = limiter.try_take(3 * MEGABYTE, now);
        let short_read = limiter.try_take(1, now);

        assert_eq!(long_read, short_read);
        assert_eq!(long_read, Err(Duration::from_micros(2500)));
    }

    #[test]
    fn a_read_longer_than_the_ceiling_is_paid_off_before_the_next() {
        let limiter = build_limiter();
        let now = Instant::now();

        let oversized = limiter.try_take(3 * MEGABYTE, now);
        let next = limiter.try_take(1, now + Duration::from_millis(19));

        assert_eq!(oversized, Ok(()));
        assert_eq!(next, Err(Duration::from_millis(1)));
    }
}
