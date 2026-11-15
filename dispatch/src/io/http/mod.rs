//! HTTP(S) reads as part of the per-core io_uring, driven by
//! [`IORequester`](super::IORequester).
//!
//! Reads of [`Remote`](super::OpenFile::Remote) cache regions are served by
//! issuing HTTP `Range` requests whose body lands directly in the pinned cache
//! slot — exactly like a disk read, just over TLS instead of `pread`. The
//! transport ([`HttpEngine`]) is ring-less: on Linux it submits its socket SQEs
//! onto the *same* per-core io_uring the file reads use (see [`backend`]); on
//! other platforms it falls back to synchronous `std::net` + rustls.
//!
//! Unlike a disk read (one SQE → one CQE), an HTTPS range request is a multi-step
//! exchange (connect → TLS handshake → request → response). That state machine,
//! plus per-host keep-alive pooling, lives in [`backend`]; the requester only
//! tracks which dataflow each in-flight request belongs to and commits the block
//! once its body has fully landed.

use crate::io::RemoteFile;
use crate::memory::FileBytes;
use std::sync::Arc;
use thiserror::Error;

mod backend;
mod http1;
mod proto;
mod tls;

pub use tls::default_client_config;

#[cfg(target_os = "linux")]
pub(crate) use backend::HTTP_TAG;
#[cfg(all(unix, not(target_os = "linux")))]
pub(crate) use backend::HttpCompletion;
pub(crate) use backend::HttpEngine;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Proto(#[from] proto::ProtoError),
    #[error("{0}")]
    Http1(#[from] http1::Http1Error),
    #[error("tls error: {0}")]
    Tls(#[from] rustls::Error),
    #[error("invalid dns name: {0}")]
    InvalidDnsName(#[from] rustls::pki_types::InvalidDnsNameError),
}

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

/// A single remote range read: fetch `[offset, offset + len)` of `remote` into
/// `dest` (a pointer into the pinned cache slot, kept alive by the issuing
/// [`MissingExtent`](crate::memory::cache::MissingExtent)'s pin).
///
/// `Clone` so a transient transport failure can re-issue the same read on a
/// fresh connection (see the engine's retry path); the clone aliases the same
/// pinned `dest`, which is sound because only one attempt is ever in flight.
#[derive(Clone)]
pub(crate) struct RemoteRead {
    pub remote: Arc<RemoteFile>,
    pub offset: u64,
    pub len: usize,
    pub dest: *mut u8,
}

impl RemoteRead {
    /// Expected response-body length for this block-aligned range, based on the
    /// object size recorded when `RemoteFile` was opened.
    ///
    /// A range wholly inside the object should return `len` bytes. A range
    /// overlapping EOF should return only the bytes remaining after `offset`.
    /// An offset at or beyond EOF yields zero, though a conforming server will
    /// normally reject that range with `416`.
    ///
    /// The response parser requires this exact length. A mismatch indicates a
    /// truncated response or that the remote object no longer matches its recorded
    /// size.
    pub fn expected_body_len(&self) -> usize {
        (self.len as u64).min(self.remote.size().saturating_sub(self.offset)) as usize
    }
}

// SAFETY: `dest` points into a cache slot kept alive for the read's whole
// lifetime by the originating block's `Arc<ReadBuffer>` pin; only this read
// writes to `[dest, dest+len)` (currently-invalid sub-blocks), so moving the
// descriptor into the engine is sound. Mirrors the file path's PendingRead.
unsafe impl Send for RemoteRead {}

/// A whole-object upload. The body is reference counted so the operator-owned
/// bytes remain stable while socket SQEs point into them, and is held as runs so
/// it can live on the ring; the transport sends one run at a time.
#[derive(Clone)]
pub(crate) struct RemoteUpload {
    pub remote: Arc<RemoteFile>,
    pub data: Arc<FileBytes>,
}

#[derive(Clone)]
pub(crate) enum RemoteRequest {
    Read(RemoteRead),
    Upload(RemoteUpload),
}

impl RemoteRequest {
    pub fn remote(&self) -> &Arc<RemoteFile> {
        match self {
            Self::Read(r) => &r.remote,
            Self::Upload(r) => &r.remote,
        }
    }

    pub fn request_head(&self) -> Vec<u8> {
        let remote = self.remote();
        let auth = remote.auth_header();
        match self {
            Self::Read(read) => proto::build_range_get(
                remote.host_header(),
                remote.request_target(),
                read.offset,
                read.len,
                auth.as_deref(),
            ),
            Self::Upload(upload) => proto::build_upload(
                remote.host_header(),
                remote.request_target(),
                upload.data.len(),
                auth.as_deref(),
            ),
        }
    }
}

// The read pointer has the safety argument above; uploads contain only Arc-owned
// data and are Send without an unsafe implementation of their own.
unsafe impl Send for RemoteRequest {}
