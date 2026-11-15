//! A persistent, disk-backed second tier for remote (HTTP) object reads.
//!
//! The in-memory [`CompressedCache`](crate::memory::cache::CompressedCache) caches 2 MB
//! regions of every file - local or remote - in RAM. For remote objects that's
//! the *only* thing standing between a query and a network round-trip to S3/GCS.
//! This disk cache adds a tier *below* RAM and *above* HTTP: fetched byte ranges
//! are mirrored into a local file per object, so a later read of the same range
//! (in this process or after a restart) comes off local disk instead of the
//! network. On restart [`open`](DiskCache::open) reloads the existing files into
//! the cache's *tracked, budgeted* set, so they're reused yet can't leak.
//!
//! ## Where it plugs in
//!
//! Entirely inside [`IORequester`](crate::io::IORequester): when a remote
//! [`MissingExtent`](crate::memory::cache::MissingExtent) needs filling, the
//! requester asks the disk cache which sub-ranges are already on disk
//! ([`Object::split_into_segments`]). Present sub-ranges are read from the cache file
//! (a plain filesystem read into the same pinned slot); absent ones are fetched
//! over HTTP and, once they land, written back to the cache file. The in-memory
//! cache and the operators never know any of this happened.
//!
//! ## Layout
//!
//! One **sparse** file per object under the cache directory, named by a stable
//! 128-bit hash of the object's identity (its authority + path, never the
//! presign query). The file mirrors the object's byte layout: object byte `X`
//! lives at file offset `X`, so only fetched ranges consume disk. A 128-bit digest makes
//! a filename collision between two distinct objects astronomically unlikely, so
//! the name alone identifies the object - no on-disk identity check needed.
//!
//! ## Validity
//!
//! Which 4 KB blocks are resident is tracked by an in-memory [`BlockBitmap`] per
//! object. On Linux that bitmap is reseeded from the file's hole structure
//! (`SEEK_DATA`/`SEEK_HOLE`), so the cache survives restarts with the data file as
//! its own index: [`open`](DiskCache::open) reloads every existing file into the
//! tracked, budgeted set up front and evicts back to budget, so a prior run's
//! cache is reused but still bounded (leaving the files untracked instead would
//! put them beyond the budget's reach - an unbounded disk leak). On other
//! platforms (dev only) the bitmap can't be reseeded, so old files are dropped at
//! open instead.
//!
//! ## Eviction
//!
//! Whole-object LRU under two limits: a resident-byte budget (caps disk use) and
//! a max object count (caps in-memory entries and, critically, open fds - one per
//! object; the byte budget alone wouldn't bound these, since many small objects
//! stay under it while the count grows). Only *idle* objects are evicted - ones
//! no in-flight read or write-back still references (tracked by the object's
//! `Arc` strong count) - so an actively-read object is never pulled out from
//! under its readers. Both are therefore soft caps the in-flight working set may
//! briefly exceed.

use crate::io::RemoteFile;
use ahash::HashMap;
use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use tracing::warn;

/// Disk-cache block granularity - matches the compressed cache's `SUB_BLOCK_SIZE` and
/// the direct-I/O alignment. Every cached range is a whole number of these.
const BLOCK_SIZE: usize = 4096;

/// Hash an object identity to its 128-bit cache filename key: BLAKE3 truncated to
/// its first 128 bits. BLAKE3 is a fixed, version- and platform-stable spec (so a
/// file is found again after a restart) and cryptographic, so even adversarial
/// object keys can't be *crafted* to collide onto one file: hitting a specific
/// identity needs a second-preimage (~2^128, infeasible). The key alone
/// identifies the object - no stored identity to verify against.
fn hash_identity(identity: &str) -> u128 {
    let digest = blake3::hash(identity.as_bytes());
    let mut first16 = [0u8; 16];
    first16.copy_from_slice(&digest.as_bytes()[..16]);
    u128::from_le_bytes(first16)
}

/// One contiguous sub-range of a requested block, tagged with whether it is
/// already resident on disk. Offsets are relative to the request's start.
pub struct Segment {
    pub rel_offset: usize,
    pub len: usize,
    pub present: bool,
}

/// A fixed atomic bitmap over a file's 4 KB blocks: bit `b` set means block `b`
/// (bytes `[b*BLOCK_SIZE, (b+1)*BLOCK_SIZE)`) is resident on disk. Sized to the
/// object up front from its known length, so it never grows and its bits flip
/// lock-free - like the compressed cache's `ValidBitmap`.
struct BlockBitmap {
    words: Box<[AtomicU64]>,
}

impl BlockBitmap {
    /// A bitmap covering `num_blocks` blocks, all absent.
    fn new(num_blocks: usize) -> Self {
        let words = num_blocks.div_ceil(64);
        let words = (0..words).map(|_| AtomicU64::new(0)).collect::<Vec<_>>();
        Self {
            words: words.into_boxed_slice(),
        }
    }

    /// Is block `block` resident? `Acquire` pairs with `set_range`'s `Release`.
    fn is_set(&self, block: usize) -> bool {
        self.words
            .get(block / 64)
            .is_some_and(|w| w.load(Ordering::Acquire) & (1 << (block % 64)) != 0)
    }

    /// Mark blocks `[first, first + count)` resident; returns how many were
    /// *newly* set (so the caller can account for bytes added). `Release` so a
    /// reader that observes a bit knows the write-back that set it has completed.
    /// Blocks past the object's declared size are ignored (a reseed of a file
    /// that grew across runs could surface them).
    fn set_range(&self, first: usize, count: usize) -> usize {
        let mut added = 0;
        for block in first..first + count {
            let bit = 1 << (block % 64);
            if let Some(word) = self.words.get(block / 64)
                && word.fetch_or(bit, Ordering::Release) & bit == 0
            {
                added += 1;
            }
        }
        added
    }

    /// Bytes currently resident (set bits × block size).
    fn resident_bytes(&self) -> u64 {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as u64)
            .sum::<u64>()
            * BLOCK_SIZE as u64
    }
}

/// A single cached remote object: its sparse mirror file plus the bitmap of
/// which blocks are resident.
pub struct Object {
    file: Arc<File>,
    path: PathBuf,
    present: BlockBitmap,
    /// Resident bytes attributed to this object (for the budget). The cache's
    /// global total is summed from these under its lock, so this is the single
    /// source of truth for the object's size; `evicted` then stops accounting.
    bytes: AtomicU64,
    /// CLOCK-free LRU: the global tick at the last access.
    last_used: AtomicU64,
    /// Set once the object has been unlinked; a write-back completing afterwards
    /// must not re-add its bytes to the (already-adjusted) global total.
    evicted: AtomicBool,
}

impl Object {
    /// The cache file descriptor. Valid for as long as the caller holds the
    /// `Arc<Object>` it came from - even across an eviction, since the eviction
    /// only unlinks the path while outstanding `Arc`s keep the file open.
    pub fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }

    /// Split `[file_offset, file_offset + len)` into maximal present/absent runs
    /// against the resident bitmap. `len` and `file_offset` are `BLOCK_SIZE`
    /// multiples (guaranteed by the caller - a `MissingExtent` is always block
    /// aligned).
    pub fn split_into_segments(&self, file_offset: usize, len: usize) -> Vec<Segment> {
        debug_assert_eq!(file_offset % BLOCK_SIZE, 0);
        debug_assert_eq!(len % BLOCK_SIZE, 0);
        let present = &self.present;
        let first_block = file_offset / BLOCK_SIZE;
        let blocks = len / BLOCK_SIZE;

        let mut segments = Vec::new();
        let mut run_start = 0usize;
        let mut run_present = present.is_set(first_block);
        for i in 1..blocks {
            let p = present.is_set(first_block + i);
            if p != run_present {
                segments.push(Segment {
                    rel_offset: run_start * BLOCK_SIZE,
                    len: (i - run_start) * BLOCK_SIZE,
                    present: run_present,
                });
                run_start = i;
                run_present = p;
            }
        }
        segments.push(Segment {
            rel_offset: run_start * BLOCK_SIZE,
            len: (blocks - run_start) * BLOCK_SIZE,
            present: run_present,
        });
        segments
    }

    /// Mark `[file_offset, file_offset + len)` resident after a write-back has
    /// durably landed. Returns the bytes newly added to this object's resident set
    /// (zero if the object was evicted meanwhile, or the blocks were already
    /// present) so the caller can fold them into the cache's global budget.
    pub fn mark_present(&self, file_offset: usize, len: usize) -> u64 {
        if self.evicted.load(Ordering::Acquire) {
            return 0;
        }
        let first_block = file_offset / BLOCK_SIZE;
        let count = len / BLOCK_SIZE;
        let added = self.present.set_range(first_block, count) as u64 * BLOCK_SIZE as u64;
        self.bytes.fetch_add(added, Ordering::Relaxed);
        added
    }
}

pub struct DiskCache {
    dir: PathBuf,
    /// Resident-byte budget (caps disk usage).
    byte_budget: u64,
    /// Max number of cached objects (caps in-memory entries *and* open file
    /// descriptors - one per object). The byte budget alone doesn't bound these:
    /// many small objects stay under it while the count, and the fds, grow
    /// unbounded under churn.
    max_objects: usize,
    objects: RwLock<HashMap<u128, Arc<Object>>>,
    /// Resident-byte total for the budget, maintained incrementally: opens and
    /// write-backs add, eviction subtracts. Exact in normal operation; a `clear`
    /// racing an in-flight write-back can leave it high, which `needs_recompute`
    /// flags for the next eviction to reconcile.
    total_bytes: AtomicU64,
    /// Set by [`clear`](Self::clear) - the only operation that drops objects whose
    /// write-back bytes may still be in flight - to tell the next eviction to
    /// reconcile `total_bytes` against the live set once, healing that drift.
    needs_recompute: AtomicBool,
    tick: AtomicU64,
}

impl DiskCache {
    /// Open the disk cache at `dir` with a resident-byte budget and a max object
    /// count (which bounds open fds - keep it below the process's fd limit). The
    /// server builds one from its config file's `disk_cache` section and hands it to
    /// [`Dispatch::spin_up`](crate::Dispatch::spin_up).
    pub fn open(dir: PathBuf, byte_budget: u64, max_objects: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        let cache = Self {
            dir,
            byte_budget,
            max_objects,
            objects: RwLock::new(HashMap::default()),
            total_bytes: AtomicU64::new(0),
            needs_recompute: AtomicBool::new(false),
            tick: AtomicU64::new(0),
        };
        // Reload a previous run's files so the cache persists across restarts - and,
        // crucially, do it by *tracking* them. The budgets only govern objects in
        // `objects`, and `evict_locked` only reclaims tracked ones; a file left on
        // disk but never re-tracked is invisible to the budget and can never be
        // evicted. Under churn (ingest writing new files, compaction deleting them)
        // the objects backing deleted remote files are never re-read, so leaving
        // them untracked leaked them across every restart - unbounded, until the
        // disk filled. Registering
        // every existing file up front, then evicting to budget, keeps a restart's
        // cache warm yet bounded.
        #[cfg(target_os = "linux")]
        cache.reload_existing()?;
        // Off Linux there's no SEEK_HOLE reseed to recover an old file's resident
        // bytes, so we can't account or reuse it - drop them rather than track a
        // file we'd score as empty (which would leave its real bytes off the budget).
        #[cfg(not(target_os = "linux"))]
        for entry in std::fs::read_dir(&cache.dir)?.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
        Ok(cache)
    }

    /// Register every cache file already on disk (a previous run's) as a tracked
    /// object, then evict back down to the byte and object budgets. Tracking is
    /// what makes them reclaimable - an untracked file is beyond the budget's reach
    /// and would leak across restarts. Files load oldest-first so the newest take
    /// the highest LRU recency and survive the trim, and evicting after each insert
    /// holds the open-fd count to `max_objects` even over a huge directory.
    #[cfg(target_os = "linux")]
    fn reload_existing(&self) -> std::io::Result<()> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&self.dir)?.flatten() {
            let path = entry.path();
            // Cache files are named `{key:032x}`; anything else isn't one of ours.
            let Some(key) = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| u128::from_str_radix(n, 16).ok())
            else {
                let _ = std::fs::remove_file(&path);
                continue;
            };
            let mtime = entry.metadata().and_then(|m| m.modified()).ok();
            files.push((key, path, mtime));
        }
        // Oldest first: each newer file then takes a higher LRU tick, so when the
        // running total tips over budget the eviction below sheds the oldest.
        files.sort_by_key(|(_, _, mtime)| *mtime);

        let mut objects = self.objects.write().unwrap();
        for (key, path, _) in files {
            let Ok(size) = std::fs::metadata(&path).map(|m| m.len()) else {
                continue;
            };
            // `create_object` opens this same path (it derives it from the key),
            // reseeds the bitmap from the file's holes, and counts its resident
            // bytes - exactly what a live miss does.
            match self.create_object(key, size) {
                Ok(obj) => {
                    self.total_bytes
                        .fetch_add(obj.bytes.load(Ordering::Relaxed), Ordering::Relaxed);
                    obj.last_used.store(self.advance_tick(), Ordering::Relaxed);
                    objects.insert(key, obj);
                    evict_locked(
                        &self.total_bytes,
                        &self.needs_recompute,
                        self.byte_budget,
                        self.max_objects,
                        &mut objects,
                    );
                }
                // A file we can't open or reseed is useless - drop it.
                Err(_) => {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        Ok(())
    }

    fn advance_tick(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::Relaxed)
    }

    /// Get (or lazily create) the cache object for `remote`, bumping its LRU
    /// recency. Returns `None` when the object's cache file can't be opened.
    pub fn open_object(&self, remote: &RemoteFile) -> Option<Arc<Object>> {
        let identity = remote.cache_identity();
        let key = hash_identity(identity);

        // Fast path: already cached - a read lock so concurrent hits don't
        // serialize.
        if let Some(obj) = self.objects.read().unwrap().get(&key) {
            obj.last_used.store(self.advance_tick(), Ordering::Relaxed);
            return Some(obj.clone());
        }

        // Miss: open the file and reseed its bitmap *without* holding the lock -
        // this does I/O (open + SEEK_HOLE) and must not block other workers.
        let obj = match self.create_object(key, remote.size()) {
            Ok(obj) => obj,
            Err(e) => {
                warn!(identity, "disk cache: cannot use object: {e}");
                return None;
            }
        };
        obj.last_used.store(self.advance_tick(), Ordering::Relaxed);

        // Insert under the write lock, unless another worker beat us to it.
        let mut objects = self.objects.write().unwrap();
        if let Some(existing) = objects.get(&key) {
            existing
                .last_used
                .store(self.advance_tick(), Ordering::Relaxed);
            return Some(existing.clone());
        }
        // Account the object's reseeded bytes up front so the lock-free counter
        // stays accurate; `evict_locked` then trims only if a cap is exceeded.
        self.total_bytes
            .fetch_add(obj.bytes.load(Ordering::Relaxed), Ordering::Relaxed);
        objects.insert(key, obj.clone());
        evict_locked(
            &self.total_bytes,
            &self.needs_recompute,
            self.byte_budget,
            self.max_objects,
            &mut objects,
        );
        Some(obj)
    }

    /// Fold a write-back's just-landed range into the cache: mark it resident on
    /// `object` and add the new bytes to the global budget, evicting if that
    /// pushes us over. The object owns its own resident-byte count; the cache owns
    /// the global total, so the delta is applied here rather than inside `Object`.
    pub(crate) fn mark_resident(&self, object: &Object, file_offset: usize, len: usize) {
        let added = object.mark_present(file_offset, len);
        if added > 0 {
            self.total_bytes.fetch_add(added, Ordering::Relaxed);
            self.enforce_budget();
        }
    }

    /// Bring the cache back within its byte budget by evicting idle objects. Cheap
    /// when already under it (one atomic load, no lock). Called after a write-back
    /// grows the resident set. The object-count cap is enforced at
    /// [`open_object`](Self::open_object) instead - that's the only place the count
    /// grows.
    pub(crate) fn enforce_budget(&self) {
        if self.total_bytes.load(Ordering::Relaxed) <= self.byte_budget {
            return;
        }
        evict_locked(
            &self.total_bytes,
            &self.needs_recompute,
            self.byte_budget,
            self.max_objects,
            &mut self.objects.write().unwrap(),
        );
    }

    /// Drop every cached object - unlink its file and forget it - so subsequent
    /// reads miss the disk cache and re-fetch from the network. Returns the number
    /// of objects dropped. Backs the `drop_cache()` benchmarking hook for true
    /// cold reads. An in-flight read keeps its file alive through its own `Arc`
    /// (the fd outlives the unlink). A write-back landing concurrently can add its
    /// bytes back after we reset the total, so we flag `needs_recompute`; the next
    /// eviction reconciles the total against the live set, healing that drift.
    pub fn clear(&self) -> usize {
        let mut objects = self.objects.write().unwrap();
        let dropped = objects.len();
        for (_, obj) in objects.drain() {
            obj.evicted.store(true, Ordering::Release);
            let _ = std::fs::remove_file(&obj.path);
        }
        self.total_bytes.store(0, Ordering::Relaxed);
        self.needs_recompute.store(true, Ordering::Relaxed);
        dropped
    }

    fn create_object(&self, key: u128, size: u64) -> std::io::Result<Arc<Object>> {
        // The 128-bit key names the file; a collision with a different object is
        // astronomically unlikely, so opening it (creating on a miss, reusing it
        // across restarts) needs no identity check.
        let path = self.dir.join(format!("{key:032x}"));
        let file = open_cache_file(&path)?;

        // Size the bitmap to the object exactly, then recover which blocks are
        // already on disk (a prior run's data) from the file's hole structure.
        let present = BlockBitmap::new((size as usize).div_ceil(BLOCK_SIZE));
        reseed_bitmap(&present, &file);
        let resident_bytes = present.resident_bytes();

        Ok(Arc::new(Object {
            file: Arc::new(file),
            path,
            present,
            bytes: AtomicU64::new(resident_bytes),
            last_used: AtomicU64::new(0),
            evicted: AtomicBool::new(false),
        }))
    }
}

/// Evict least-recently-used **idle** objects until *both* limits are met: the
/// resident-byte `byte_budget` and the `max_objects` count (which bounds open
/// fds). The caller holds the `objects` write lock.
///
/// "Idle" = `Arc::strong_count == 1`: only the map references it, so no in-flight
/// read or write-back is using its file. Because clones can only be made under
/// the lock we hold, the count can't grow underneath us, so an idle object is
/// safe to drop (closing its fd). An object actively being read therefore can't
/// be evicted - both limits are soft caps the in-flight working set may briefly
/// exceed.
fn evict_locked(
    total_bytes: &AtomicU64,
    needs_recompute: &AtomicBool,
    byte_budget: u64,
    max_objects: usize,
    objects: &mut HashMap<u128, Arc<Object>>,
) {
    // A `clear()` may have left `total_bytes` overstated (a write-back's add that
    // landed after it reset the total to zero). It flags that, so reconcile to the
    // live-set sum once here - at most once per `clear()`, so the common,
    // unflagged path stays O(1).
    if needs_recompute.swap(false, Ordering::Relaxed) {
        let live: u64 = objects
            .values()
            .map(|o| o.bytes.load(Ordering::Relaxed))
            .sum();
        total_bytes.store(live, Ordering::Relaxed);
    }
    while total_bytes.load(Ordering::Relaxed) > byte_budget || objects.len() > max_objects {
        let victim = objects
            .iter()
            .filter(|(_, o)| Arc::strong_count(o) == 1)
            .min_by_key(|(_, o)| o.last_used.load(Ordering::Relaxed))
            .map(|(k, _)| *k);
        let Some(victim) = victim else { break }; // everything left is in use
        let obj = objects.remove(&victim).unwrap();
        obj.evicted.store(true, Ordering::Release);
        total_bytes.fetch_sub(obj.bytes.load(Ordering::Relaxed), Ordering::Relaxed);
        let _ = std::fs::remove_file(&obj.path);
    }
}

/// Open a cache file for direct, uncached read/write. Direct I/O keeps the
/// scheduler's "a read that returns means real disk work happened" invariant
/// (see [`crate::io`]); writes/reads are always 4 KB aligned so the alignment
/// constraints are met. `truncate(false)` is deliberate: an existing file's
/// bytes are the persistent cache and must be preserved (reseeded), never wiped.
fn open_cache_file(path: &Path) -> std::io::Result<File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(libc::O_DIRECT)
            .open(path)
    }

    #[cfg(target_os = "macos")]
    {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        if rc == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(file)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
    }
}

/// Mark `present`'s blocks resident from the file's hole structure via
/// `SEEK_DATA`/`SEEK_HOLE` - the data file is its own index, so the cache
/// survives restarts. Only the allocated (data) extents are marked present.
#[cfg(target_os = "linux")]
fn reseed_bitmap(present: &BlockBitmap, file: &File) {
    use nix::unistd::{Whence, lseek};
    let mut off: i64 = 0;
    // SEEK_DATA returns the next offset >= `off` holding data, or ENXIO once
    // past the last data extent (ending the loop).
    while let Ok(data) = lseek(file, off, Whence::SeekData) {
        let hole = match lseek(file, data, Whence::SeekHole) {
            Ok(h) => h,
            Err(_) => break,
        };
        // Mark the whole 4 KB blocks the [data, hole) extent covers.
        let first = data as usize / BLOCK_SIZE;
        let last = hole as usize / BLOCK_SIZE; // exclusive
        if last > first {
            present.set_range(first, last - first);
        }
        off = hole;
    }
}

#[cfg(not(target_os = "linux"))]
fn reseed_bitmap(_present: &BlockBitmap, _file: &File) {}

thread_local! {
    /// This worker thread's handle to the shared disk cache (or `None`). Set once
    /// at worker startup so worker-thread code - e.g. the `drop_cache()` SQL
    /// function - can reach it without threading it through every call site, the
    /// same pattern as the worker waker.
    static WORKER_DISK_CACHE: RefCell<Option<Arc<DiskCache>>> = const { RefCell::new(None) };
}

/// Install this worker thread's handle to the shared disk cache. Called once by
/// the worker at startup.
pub(crate) fn install_worker_disk_cache(cache: Option<Arc<DiskCache>>) {
    WORKER_DISK_CACHE.with(|c| *c.borrow_mut() = cache);
}

/// Drop everything in this worker's disk cache (see [`DiskCache::clear`]),
/// returning the number of objects dropped (0 when no disk cache is configured).
/// Backs `drop_cache()` so it forces true cold reads from the network too.
pub fn clear_disk_cache() -> usize {
    WORKER_DISK_CACHE.with(|c| c.borrow().as_ref().map_or(0, |dc| dc.clear()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::RemoteFile;
    use url::Url;

    fn cache(dir: &std::path::Path) -> DiskCache {
        // Generous byte + count budgets; the budget-specific tests set their own.
        DiskCache::open(dir.to_path_buf(), 1 << 30, 1 << 20).unwrap()
    }

    fn remote(path: &str) -> RemoteFile {
        RemoteFile::open(
            Url::parse(&format!("http://127.0.0.1:1{path}")).unwrap(),
            None,
            1 << 20,
        )
        .unwrap()
    }

    #[test]
    fn segments_report_resident_runs_and_holes() {
        let dir = tempfile::tempdir().unwrap();
        let object = cache(dir.path()).open_object(&remote("/o")).unwrap();
        object.mark_present(0, BLOCK_SIZE);
        object.mark_present(2 * BLOCK_SIZE, BLOCK_SIZE);

        let segments = object.split_into_segments(0, 3 * BLOCK_SIZE);

        let shape: Vec<_> = segments
            .iter()
            .map(|s| (s.rel_offset, s.len, s.present))
            .collect();
        assert_eq!(
            shape,
            vec![
                (0, BLOCK_SIZE, true),
                (BLOCK_SIZE, BLOCK_SIZE, false),
                (2 * BLOCK_SIZE, BLOCK_SIZE, true),
            ]
        );
    }

    #[test]
    fn an_untouched_range_is_a_single_hole() {
        let dir = tempfile::tempdir().unwrap();
        let object = cache(dir.path()).open_object(&remote("/o")).unwrap();

        let segments = object.split_into_segments(0, 3 * BLOCK_SIZE);

        let shape: Vec<_> = segments
            .iter()
            .map(|s| (s.rel_offset, s.len, s.present))
            .collect();
        assert_eq!(shape, vec![(0, 3 * BLOCK_SIZE, false)]);
    }

    #[test]
    fn re_marking_resident_blocks_adds_no_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let object = cache(dir.path()).open_object(&remote("/o")).unwrap();

        let first = object.mark_present(0, 2 * BLOCK_SIZE);
        let again = object.mark_present(0, 2 * BLOCK_SIZE);

        assert_eq!(first, 2 * BLOCK_SIZE as u64);
        assert_eq!(again, 0); // idempotent, so the budget can't drift
    }

    #[test]
    fn the_same_object_reuses_one_cache_file() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache(dir.path());
        let remote = remote("/same");

        let first = cache.open_object(&remote).unwrap();
        let second = cache.open_object(&remote).unwrap();

        assert_eq!(first.fd(), second.fd());
    }

    fn resident(cache: &DiskCache, path: &str) -> bool {
        cache
            .open_object(&remote(path))
            .unwrap()
            .split_into_segments(0, BLOCK_SIZE)[0]
            .present
    }

    /// Open `path`, mark its first block resident (folding it into the budget),
    /// then drop the handle so the object is idle and eligible for eviction.
    fn prime(cache: &DiskCache, path: &str) {
        let object = cache.open_object(&remote(path)).unwrap();
        cache.mark_resident(&object, 0, BLOCK_SIZE);
    }

    #[test]
    fn a_hole_marked_present_becomes_resident() {
        let dir = tempfile::tempdir().unwrap();
        let object = cache(dir.path()).open_object(&remote("/o")).unwrap();

        object.mark_present(0, BLOCK_SIZE);

        assert!(object.split_into_segments(0, BLOCK_SIZE)[0].present);
    }

    #[test]
    fn over_budget_evicts_the_least_recently_used_idle_object() {
        let dir = tempfile::tempdir().unwrap();
        // One block of byte budget, no object-count limit (isolating the byte cap).
        let cache =
            DiskCache::open(dir.path().to_path_buf(), BLOCK_SIZE as u64, usize::MAX).unwrap();
        prime(&cache, "/a");
        prime(&cache, "/b");

        cache.enforce_budget();

        assert!(!resident(&cache, "/a")); // the LRU object was evicted
        assert!(resident(&cache, "/b"));
    }

    #[test]
    fn an_object_still_in_use_is_never_evicted() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(dir.path().to_path_buf(), 0, usize::MAX).unwrap();
        let in_use = cache.open_object(&remote("/held")).unwrap();
        cache.mark_resident(&in_use, 0, BLOCK_SIZE);
        prime(&cache, "/idle");

        cache.enforce_budget();

        assert!(in_use.split_into_segments(0, BLOCK_SIZE)[0].present); // held, so it survived
        assert!(!resident(&cache, "/idle"));
    }

    /// The object-count cap evicts idle LRU objects even when the byte budget is
    /// nowhere near hit - bounding open fds / in-memory entries under churn. With
    /// a huge byte budget but a 2-object cap, opening a 3rd object must drop the
    /// least-recently-used one (so its file is unlinked, leaving 2 on disk).
    #[test]
    fn over_object_count_evicts_idle_objects() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(dir.path().to_path_buf(), 1 << 30, 2).unwrap();
        // The count cap is byte-independent, so just opening each object (which
        // creates its file) drives the eviction.
        for name in ["/a", "/b", "/c"] {
            cache.open_object(&remote(name)).unwrap();
        }

        let files = std::fs::read_dir(dir.path()).unwrap().count();

        assert_eq!(files, 2, "the 2-object cap should leave 2 cache files");
    }

    /// `clear` (the `drop_cache()` hook) drops every object and unlinks its file,
    /// so a later read starts cold.
    #[test]
    fn clear_drops_all_objects_and_unlinks_files() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache(dir.path());
        cache
            .open_object(&remote("/a"))
            .unwrap()
            .mark_present(0, BLOCK_SIZE);
        cache
            .open_object(&remote("/b"))
            .unwrap()
            .mark_present(0, BLOCK_SIZE);

        let dropped = cache.clear();

        assert_eq!(dropped, 2);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
        assert!(!resident(&cache, "/a")); // re-opens cold (file was unlinked)
    }

    /// A previous run's files are reloaded into the tracked, budgeted set, so the
    /// cache persists across a restart - and, being tracked, they stay bounded:
    /// eviction trims the directory back to the byte budget instead of the files
    /// leaking untracked forever.
    #[cfg(target_os = "linux")]
    #[test]
    fn open_reloads_previous_files_and_bounds_them_to_budget() {
        let dir = tempfile::tempdir().unwrap();
        // Three cache files of one resident block each, as a prior run would leave.
        for k in 0..3u128 {
            std::fs::write(dir.path().join(format!("{k:032x}")), vec![1u8; BLOCK_SIZE]).unwrap();
        }

        // A two-block budget: reopen keeps the two newest, evicts the LRU one.
        let cache =
            DiskCache::open(dir.path().to_path_buf(), 2 * BLOCK_SIZE as u64, usize::MAX).unwrap();

        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
        assert_eq!(
            cache.total_bytes.load(Ordering::Relaxed),
            2 * BLOCK_SIZE as u64
        );
    }

    /// Off Linux there's no SEEK_HOLE reseed to recover a file's resident bytes, so
    /// a prior run's files are dropped at open rather than tracked as empty.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn open_drops_previous_files_without_seek_hole() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("0000000000000000000000000000002a"),
            b"stale",
        )
        .unwrap();

        let _cache = DiskCache::open(dir.path().to_path_buf(), 1 << 30, 1 << 20).unwrap();

        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
