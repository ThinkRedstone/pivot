//! The network bandwidth of the machine this process runs on.
//!
//! A network card does not report how much it may carry: on a cloud instance
//! the limit is an allowance the provider enforces outside the guest, and the
//! driver's link speed is not readable. What can be read is the instance type,
//! from the instance metadata service, and each type's bandwidth is published.
//! [`look_up_instance_bandwidth`] asks the service for the type and finds it in
//! the table compiled in from `instance_bandwidth.tsv`.
//!
//! The table holds one line per EC2 instance type: its name, then the
//! baseline and the burst bandwidth of its default network card in Gbit/s. It
//! is the output of
//!
//! ```text
//! aws ec2 describe-instance-types --query 'InstanceTypes[].[InstanceType,
//!     NetworkInfo.NetworkCards[0].BaselineBandwidthInGbps,
//!     NetworkInfo.NetworkCards[0].PeakBandwidthInGbps]' --output text
//! ```
//!
//! over the regions that between them offer every type, sorted and
//! deduplicated. Regenerate it when new instance types appear.

use std::time::Duration;
use thiserror::Error;

/// Where the EC2 instance metadata service answers from inside an instance.
pub const INSTANCE_METADATA_ENDPOINT: &str = "http://169.254.169.254";

/// How long the metadata service gets to answer. It is link-local, so a real
/// one answers in a millisecond; off EC2 nothing does, and startup should not
/// wait on that for long.
const METADATA_TIMEOUT: Duration = Duration::from_secs(1);

/// How long connecting to the metadata service may take. Where the address
/// leads nowhere a connection attempt hangs, and every start of the process
/// would wait this out.
const METADATA_CONNECT_TIMEOUT: Duration = Duration::from_millis(250);

const INSTANCE_BANDWIDTH_TABLE: &str = include_str!("instance_bandwidth.tsv");

/// What an instance's default network card may carry, in each direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstanceBandwidth {
    /// What the instance can sustain indefinitely.
    pub baseline_bits_per_second: u64,
    /// What it can reach for a while on accumulated credit. Equal to the
    /// baseline for types that do not burst.
    pub burst_bits_per_second: u64,
}

#[derive(Debug, Error)]
pub enum InstanceBandwidthError {
    #[error("the instance metadata service at {endpoint} did not answer: {reason}")]
    MetadataUnavailable { endpoint: String, reason: String },
    #[error("no bandwidth is known for instance type `{0}`")]
    UnknownInstanceType(String),
}

/// The bandwidth of the instance this process runs on, from the instance
/// metadata service at `metadata_endpoint` (see [`INSTANCE_METADATA_ENDPOINT`]).
pub fn look_up_instance_bandwidth(
    metadata_endpoint: &str,
) -> Result<InstanceBandwidth, InstanceBandwidthError> {
    let instance_type = fetch_instance_type(metadata_endpoint)?;
    find_instance_type_bandwidth(&instance_type)
}

/// The published bandwidth of `instance_type`, such as `c8gn.4xlarge`.
pub fn find_instance_type_bandwidth(
    instance_type: &str,
) -> Result<InstanceBandwidth, InstanceBandwidthError> {
    INSTANCE_BANDWIDTH_TABLE
        .lines()
        .map(|line| line.split('\t'))
        .find_map(|mut fields| {
            (fields.next() == Some(instance_type)).then(|| InstanceBandwidth {
                baseline_bits_per_second: parse_gigabits(fields.next()),
                burst_bits_per_second: parse_gigabits(fields.next()),
            })
        })
        .ok_or_else(|| InstanceBandwidthError::UnknownInstanceType(instance_type.to_string()))
}

/// A table field in Gbit/s as bits per second. The table is compiled in, so a
/// field that does not parse is a broken build, not something to recover from.
fn parse_gigabits(field: Option<&str>) -> u64 {
    let gigabits: f64 = field
        .and_then(|field| field.parse().ok())
        .expect("every line of the instance bandwidth table has two rates in Gbit/s");
    (gigabits * 1e9).round() as u64
}

/// Ask the metadata service which instance type this is. The service hands
/// out a session token first and answers only requests that carry it.
fn fetch_instance_type(metadata_endpoint: &str) -> Result<String, InstanceBandwidthError> {
    let unavailable = |reason: String| InstanceBandwidthError::MetadataUnavailable {
        endpoint: metadata_endpoint.to_string(),
        reason,
    };
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(METADATA_CONNECT_TIMEOUT)
        .timeout(METADATA_TIMEOUT)
        .build();

    let token = agent
        .put(&format!("{metadata_endpoint}/latest/api/token"))
        .set("X-aws-ec2-metadata-token-ttl-seconds", "60")
        .call()
        .map_err(|error| unavailable(error.to_string()))?
        .into_string()
        .map_err(|error| unavailable(error.to_string()))?;
    let instance_type = agent
        .get(&format!(
            "{metadata_endpoint}/latest/meta-data/instance-type"
        ))
        .set("X-aws-ec2-metadata-token", &token)
        .call()
        .map_err(|error| unavailable(error.to_string()))?
        .into_string()
        .map_err(|error| unavailable(error.to_string()))?;

    Ok(instance_type.trim().to_string())
}
