//! Operators and channels: the parallel execution toolkit.
//!
//! This module contains everything that executes within a worker's
//! `DataFlow`:
//!
//! - **[`Operator`]** — The trait every node in a dataflow implements. An operator
//!   is a stateful object that can do CPU work, request disk IO, process IO completions,
//!   steal work from peers, and signal when it is finished.
//!
//! - **[`channels`]** — Sender/receiver pairs that connect adjacent operators in a
//!   dataflow. Three flavors: work-stealing (default), mpsc (final output), and
//!   return-to-worker (worker-affinity routing).
//!
//! - **[`nullary`]** — Source-like operators with no input channel that can emit
//!   output or perform side effects directly.
//!
//! - **[`unary`]** — The [`UnaryOperator`], which reads from one
//!   input channel and writes to one output channel, applying a [`Unary`]
//!   transform. Most query stages (filter, project, count, etc.) are built as unary
//!   operators. The [`UnaryOperatorFactory`] creates them during the factory build step
//!
//! # Operator lifecycle
//!
//! Operators are created during the factory build step on the worker thread (not before —
//! some hold `Rc` or other non-`Send` state). Once created, the worker's event loop
//! repeatedly calls:
//!
//! 1. [`run_cpu_work`](Operator::run_cpu_work) — does one unit of CPU work and
//!    may append logical requests to its [`OperatorIO`].
//!    Returns [`WorkStatus::Ran`] if it did anything, [`WorkStatus::Pending`] if no work was available
//!    to do.
//!
//! 2. The worker drains every node's `OperatorIO`, resolves logical reads through
//!    the decompressed and compressed caches, allocates missing extents, and
//!    submits the resulting physical IO. It delivers complete logical reads via
//!    [`process_read_response`](Operator::process_read_response); write and upload
//!    completions retain their transport-specific callbacks.
//!
//! 3. [`try_finish`](Operator::try_finish) — called when the input channel is drained
//!    and all sibling operators (across workers) have also drained. The operator does
//!    any final work (e.g. emit aggregation results) and returns `true` when fully done.
//!
//! 4. [`try_steal_work`](Operator::try_steal_work) — called when the worker is
//!    idle. The operator attempts to steal from a peer worker's input channel.

use crate::data_flow::WorkStatus;
use crate::io::{FsWriteRequest, HttpUploadRequest, OperatorIO, ReadResponse};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use thiserror::Error;

pub mod channels;

pub mod cte;
pub use cte::{CteFactory, CteScanFactory};

pub mod in_memory;
pub use in_memory::{Forward, InjectorSourceFactory};

pub mod streamed;
pub use streamed::{ChannelInputFull, ChannelInputSender, ChannelSourceFactory};

pub mod nullary;
pub use nullary::*;

pub mod unary;
pub use unary::*;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Unary(#[from] unary::Error),
    #[error("{0}")]
    Nullary(#[from] nullary::Error),
    #[error("{0}")]
    Channel(#[from] channels::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Outcome of an operator's [`try_finish`](Operator::try_finish).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FinishStatus {
    /// Fully finished. The node drops the operator on hearing this, so it is
    /// never asked again.
    Done,
    /// Still producing final output and needs to be re-driven — e.g. a pipeline
    /// breaker draining its outputter over several calls. The worker keeps
    /// driving it (via `run_cpu_work`) rather than parking; it doesn't wake
    /// peers, because the output's own downstream `send`s already do.
    Working,
    /// Not ready to finish: input still draining, or waiting on a sibling to
    /// catch up. The worker may park — a sibling's finish will wake it.
    Pending,
}

/// A single node in a `DataFlow`.
///
/// Each operator runs on one worker thread, and is dropped as soon as it reports
/// [`FinishStatus::Done`] rather than at the end of the dataflow.
/// Sibling operators (the same stage on different workers) share the same identifier,
/// enabling work stealing and coordinated finishing (e.g. a shared atomic counter for
/// `Count`).
///
/// CPU work and IO are intentionally separate: the worker can saturate the IO queue
/// while running CPU work from a different operator, hiding disk latency.
pub trait Operator {
    /// Try to consume one item from the input and produce output.
    /// Returns [`WorkStatus::Ran`] if work was done, [`Pending`](WorkStatus::Pending) otherwise.
    fn run_cpu_work(&mut self, io: &mut OperatorIO) -> Result<WorkStatus>;

    /// Handle a completed logical read. Dispatch has already resolved cache
    /// hits and committed any physical reads before making this callback.
    fn process_read_response(
        &mut self,
        _io: &mut OperatorIO,
        _response: ReadResponse,
    ) -> Result<()> {
        unreachable!()
    }

    /// Handle a completed filesystem write.
    fn process_fs_write_response(&mut self, request: FsWriteRequest) -> Result<()>;

    /// Handle a completed HTTP upload. Called by the worker when the upload
    /// finishes.
    fn process_http_upload_response(&mut self, request: HttpUploadRequest) -> Result<()>;

    /// Attempt to finish. Returns a [`FinishStatus`] telling the worker whether
    /// the operator is done, still working (re-drive, don't park), or not yet
    /// ready. May be called many times until it reports [`FinishStatus::Done`].
    fn try_finish(&mut self) -> Result<FinishStatus>;

    /// Try to steal work from a peer worker's channel. Default: no stealing.
    fn try_steal_work(&mut self, _io: &mut OperatorIO) -> Result<WorkStatus> {
        Ok(WorkStatus::Pending)
    }

    /// A flag this operator raises to ask the `DataFlow` to abandon everything
    /// *upstream* of it, used by `LIMIT` to stop the scan once it has buffered
    /// enough rows, without disturbing operators downstream of it (e.g. a
    /// `GROUP BY` over a `LIMIT` subquery).
    ///
    /// Returning `Some` is a *capability* declaration, made once at build time:
    /// the `DataFlow` records the flag and polls only it, so operators that
    /// never cancel upstream (the default `None`) cost nothing in the hot loop.
    /// The flag's *value* is the runtime trigger: `false` until the operator
    /// decides its upstream is no longer needed.
    fn upstream_cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        None
    }
}
