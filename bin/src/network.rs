//! What the machine's network card may carry, shared by `pivot server` and
//! `pivot open`: a configured rate, or the published rate of the instance type
//! the process runs on. Dispatch limits how fast remote reads are requested
//! to it.

use std::str::FromStr;

use serde::Deserialize;
use tracing::{info, warn};

/// A network rate as the config file or the command line spells it: a number
/// of bits per second with a decimal unit, `50gbit` or `750mbit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bandwidth {
    pub bits_per_second: u64,
}

impl FromStr for Bandwidth {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let (number, bits_per_unit) = if let Some(number) = input.strip_suffix("gbit") {
            (number, 1_000_000_000.0)
        } else if let Some(number) = input.strip_suffix("mbit") {
            (number, 1_000_000.0)
        } else {
            return Err(format!(
                "`{input}` is not a network rate: write it as `50gbit` or `750mbit`"
            ));
        };
        let amount: f64 = number
            .trim()
            .parse()
            .map_err(|_| format!("`{input}` is not a network rate: `{number}` is not a number"))?;
        if amount.is_nan() || amount <= 0.0 {
            return Err(format!("`{input}` is not a network rate above zero"));
        }
        Ok(Self {
            bits_per_second: (amount * bits_per_unit).round() as u64,
        })
    }
}

impl<'de> Deserialize<'de> for Bandwidth {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// Limit how fast remote reads are requested to what the machine's network
/// card receives: `configured`, or else the baseline rate published for the
/// instance type the process runs on. Call before dispatch starts its workers.
///
/// With neither known, as on a machine that is not a cloud instance, the card
/// is taken to receive [`dispatch::ASSUMED_NETWORK_BITS_PER_SECOND`], with a
/// warning: a faster card reads remote data slower than it could until its
/// rate is set.
pub fn limit_remote_reads_to_network_bandwidth(configured: Option<Bandwidth>) {
    if let Some(bandwidth) = configured {
        info!(
            bits_per_second = bandwidth.bits_per_second,
            "remote reads limited to the configured network bandwidth"
        );
        dispatch::limit_network_receive_rate(bandwidth.bits_per_second);
        return;
    }
    match object_storage::look_up_instance_bandwidth(object_storage::INSTANCE_METADATA_ENDPOINT) {
        Ok(bandwidth) => {
            info!(
                baseline_bits_per_second = bandwidth.baseline_bits_per_second,
                burst_bits_per_second = bandwidth.burst_bits_per_second,
                "remote reads limited to the instance type's baseline network bandwidth"
            );
            dispatch::limit_network_receive_rate(bandwidth.baseline_bits_per_second);
        }
        Err(error) => {
            warn!(
                %error,
                assumed_bits_per_second = dispatch::ASSUMED_NETWORK_BITS_PER_SECOND,
                "network bandwidth unknown, remote reads are limited to an assumed rate; \
                 set the real one with `network_bandwidth` or `--network-bandwidth`"
            );
            dispatch::limit_network_receive_rate(dispatch::ASSUMED_NETWORK_BITS_PER_SECOND);
        }
    }
}
