//! Throughput microbenchmark for the HTTP(S) range-read path.
//!
//! Stands up a **loopback** backend and measures how fast the HTTP engine
//! (io_uring on Linux, blocking `std::net` elsewhere) can pull object-store
//! ranges into cache slots — i.e. "how many times can we hit it, and at what
//! byte rate".
//!
//! # Why loopback
//!
//! Over loopback there is essentially no network wait, so wall-clock ≈ the
//! engine's *CPU* cost: the recv/send syscalls, the TLS record crypto, and the
//! response-decode bookkeeping. That makes this the sensitive test for "did the
//! sans-IO [`BodyDecoder`](dispatch) migration add any copy/CPU overhead": a
//! real-S3 benchmark is network-bound and would *mask* exactly that. For an A/B
//! vs a baseline build, run under `perf stat` and compare **instructions
//! retired** (robust to the wall-clock noise that plagues this box) — the loop's
//! constant cache/eviction cost cancels in the diff.
//!
//! # Scenarios
//! - `http_range/plaintext` — the zero-copy recv-straight-into-slot path; the
//!   most sensitive to a stray copy.
//! - `http_range/tls` — rustls decrypt into the slot; the realistic S3 path.
//!
//! Each request fetches a `BLOCK`-byte range at an ever-advancing offset, so it
//! is always a cache **miss** that drives a real HTTP fetch (the backend
//! synthesises the body for any range). A batch submits `READS` requests
//! concurrently (the engine pools one keep-alive connection per request and
//! reuses them across iterations), then drains them — so steady state measures
//! the pooled, keep-alive hot path.
//!
//! # Running
//! ```sh
//! cargo bench --bench http_range --features test-util
//! # one scenario, looped for perf attach:
//! cargo bench --bench http_range --features test-util -- http_range/tls --profile-time 20
//! # A/B instruction counts (see module docs):
//! perf stat -e instructions,cycles -- \
//!   cargo bench --bench http_range --features test-util -- http_range/plaintext --profile-time 10
//! ```
//! Tunables (env): `PIVOT_HTTP_BENCH_BLOCK` (bytes/request, default = one cache
//! region), `PIVOT_HTTP_BENCH_READS` (concurrent requests per batch, default 32).
//!
//! NOTE: under `perf`, raise the locked-memory limit (`ulimit -l unlimited`),
//! else io_uring setup can hit `ENOMEM`.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use criterion::{BatchSize, Criterion, Throughput, black_box};
use url::Url;

use dispatch::io::{
    FileRange, IORequester, OpenFile, PendingReadRequest, ReadRequestId, RemoteFile,
};
use dispatch::memory::init_test_free_pool;
use dispatch::{BUFFER_SIZE, StatsCollector, memory_ctx};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Bytes per range request. Default = one cache region, so each request is a
/// single block fetch whose cost is dominated by the body path (what the
/// migration touched) rather than per-request head overhead.
fn block_bytes() -> usize {
    env_usize("PIVOT_HTTP_BENCH_BLOCK", BUFFER_SIZE)
}

/// Concurrent requests per measured batch (kept well under the 128-slot test
/// ring so the pinned working set always fits).
fn reads_per_batch() -> usize {
    env_usize("PIVOT_HTTP_BENCH_READS", 32)
}

// ---------------------------------------------------------------------------
// Loopback backend: a forever-serving HTTP/1.1 range responder
// ---------------------------------------------------------------------------

/// Accepts a TLS client cert / signature unconditionally — the backend uses a
/// self-signed cert and the bench only cares about transport throughput.
#[derive(Debug)]
struct NoVerify;

impl rustls::client::danger::ServerCertVerifier for NoVerify {
    fn verify_server_cert(
        &self,
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &[rustls::pki_types::CertificateDer<'_>],
        _: &rustls::pki_types::ServerName<'_>,
        _: &[u8],
        _: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        _: &[u8],
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(
        &self,
        _: &[u8],
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn client_config() -> Arc<rustls::ClientConfig> {
    Arc::new(
        rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoVerify))
        .with_no_client_auth(),
    )
}

fn server_config() -> Arc<rustls::ServerConfig> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
    let cert_der = cert.cert.der().clone();
    let key_der = rustls::pki_types::PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());
    Arc::new(
        rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![cert_der],
            rustls::pki_types::PrivateKeyDer::Pkcs8(key_der),
        )
        .unwrap(),
    )
}

/// Serve range requests forever on one connection: read each request head, parse
/// its `Range`, and reply `206` with exactly that many bytes. The body is a
/// reusable zero buffer — content is irrelevant to the transport cost, and not
/// re-filling it keeps the server off the critical path so the *client* engine
/// is what we measure.
fn serve_conn<S: Read + Write>(mut s: S) {
    let mut scratch = vec![0u8; 64 * 1024];
    let zeros = vec![0u8; 4 * 1024 * 1024];
    loop {
        let head = match read_head(&mut s, &mut scratch) {
            Some(h) => h,
            None => return, // client closed
        };
        let (start, end) = match parse_range(&head) {
            Some(r) => r,
            None => return,
        };
        let len = end - start + 1;
        let resp = format!(
            "HTTP/1.1 206 Partial Content\r\n\
             Content-Length: {len}\r\n\
             Content-Range: bytes {start}-{end}/1000000000000\r\n\
             Connection: keep-alive\r\n\r\n"
        );
        if s.write_all(resp.as_bytes()).is_err() {
            return;
        }
        let mut remaining = len;
        while remaining > 0 {
            let n = remaining.min(zeros.len());
            if s.write_all(&zeros[..n]).is_err() {
                return;
            }
            remaining -= n;
        }
        let _ = s.flush();
    }
}

/// Read up to and including the `\r\n\r\n` head terminator. `None` on EOF/error.
fn read_head<S: Read>(s: &mut S, scratch: &mut [u8]) -> Option<Vec<u8>> {
    let mut buf = Vec::with_capacity(256);
    loop {
        let n = s.read(scratch).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&scratch[..n]);
        if let Some(pos) = find_crlf_crlf(&buf) {
            buf.truncate(pos + 4);
            return Some(buf);
        }
    }
}

fn find_crlf_crlf(b: &[u8]) -> Option<usize> {
    b.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Parse the inclusive `[start, end]` from a `Range: bytes=START-END` header.
fn parse_range(head: &[u8]) -> Option<(usize, usize)> {
    let text = std::str::from_utf8(head).ok()?;
    let line = text
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("range:"))?;
    let spec = line.split('=').nth(1)?.trim();
    let mut parts = spec.split('-');
    let start = parts.next()?.trim().parse().ok()?;
    let end = parts.next()?.trim().parse().ok()?;
    Some((start, end))
}

/// Spawn a backend; returns the bound port. Each accepted connection is handled
/// on its own thread, forever, so the engine's keep-alive pool can hold many
/// concurrent connections.
fn spawn_backend(tls: bool) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let cfg = tls.then(server_config);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(tcp) = stream else { continue };
            tcp.set_nodelay(true).ok();
            match &cfg {
                Some(cfg) => {
                    let cfg = cfg.clone();
                    thread::spawn(move || {
                        if let Ok(conn) = rustls::ServerConnection::new(cfg) {
                            serve_conn(rustls::StreamOwned::new(conn, tcp));
                        }
                    });
                }
                None => {
                    thread::spawn(move || serve_conn(tcp));
                }
            }
        }
    });
    port
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Submit `reads` logical ranges at advancing (always-missing) offsets, then
/// drive them all to completion. The request tracker pins their cache lookups
/// until the completed logical responses are discarded. Returns bytes fetched.
fn run_batch(
    requester: &mut IORequester,
    stats: &mut StatsCollector,
    loc: &OpenFile,
    reads: usize,
    block: usize,
    offset: &AtomicU64,
) -> usize {
    for _ in 0..reads {
        let off = offset.fetch_add(block as u64, Ordering::Relaxed) as usize;
        requester
            .request_read(
                0,
                0,
                PendingReadRequest {
                    id: ReadRequestId(off),
                    open_file: loc.clone(),
                    locations: vec![FileRange::new(off, block)],
                },
                stats,
            )
            .unwrap();
    }

    while requester.has_pending() {
        requester.wait().unwrap();
        requester.completions().unwrap();
    }
    requester.cancel_dataflow(0);
    reads * block
}

fn bench_transport(c: &mut Criterion, name: &str, tls: bool) {
    let port = spawn_backend(tls);
    let scheme = if tls { "https" } else { "http" };
    let url = Url::parse(&format!("{scheme}://127.0.0.1:{port}/obj")).unwrap();
    let remote = Arc::new(RemoteFile::open(url, None, 1_000_000_000_000).unwrap());
    let loc = OpenFile::Remote(remote.clone());
    memory_ctx().compressed_cache().open_entry(loc.clone());

    let mut requester = IORequester::with_http_config(client_config());
    let mut stats = StatsCollector::disabled();
    let block = block_bytes();
    let reads = reads_per_batch();
    let offset = AtomicU64::new(0);

    // Warm up the keep-alive pool (first batch pays connect + TLS handshake).
    run_batch(&mut requester, &mut stats, &loc, reads, block, &offset);

    let mut g = c.benchmark_group("http_range");
    g.throughput(Throughput::Bytes((reads * block) as u64));
    g.bench_function(name, |b| {
        // UNTIMED setup: recycle the previous batch's committed regions back to
        // the free pool, so each timed batch is `reads` genuine cache misses that
        // fit the pool (no eviction). `clear()` touches only the compressed cache, not
        // the engine's keep-alive connection pool — so connect/handshake stay
        // amortised and we measure the steady-state request hot path.
        b.iter_batched(
            || {
                memory_ctx().compressed_cache().clear();
            },
            |_| {
                let bytes = run_batch(&mut requester, &mut stats, &loc, reads, block, &offset);
                black_box(bytes);
            },
            BatchSize::PerIteration,
        );
    });
    g.finish();
}

fn main() {
    let reads = reads_per_batch();
    let block = block_bytes();
    eprintln!(
        "http range bench: {reads} concurrent reads/batch, {} KiB/read",
        block / 1024
    );
    // Single-worker test context on this thread (its own 128-slot ring); the
    // standalone IORequester drives the HTTP engine against the loopback backend.
    init_test_free_pool(128);

    let mut c = Criterion::default().configure_from_args();
    bench_transport(&mut c, "plaintext", false);
    bench_transport(&mut c, "tls", true);
    c.final_summary();
}
