//! The generic cache and the caches together: packing, lookups, reclaim,
//! target shares, reinforcement, and eviction across families. Every test
//! installs its own single-worker memory context with the standard setup.

use super::*;
use crate::io::{LocalFile, OpenFile};
use crate::memory::BUFFER_SIZE;
use crate::memory::context::{init_test_free_pool, memory_ctx};
use bytes::Bytes;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// One shared local file, for tests that key by offset alone.
fn shared_file() -> OpenFile {
    use std::sync::OnceLock;
    static FILE: OnceLock<LocalFile> = OnceLock::new();
    OpenFile::Local(
        FILE.get_or_init(|| LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
            .clone(),
    )
}

/// A fresh, independent local file (distinct key bucket).
fn new_file() -> OpenFile {
    OpenFile::Local(LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
}

fn key(offset: usize) -> CacheKey {
    CacheKey::new(shared_file(), offset, 1)
}

fn key_in(file: &OpenFile, offset: usize, span: usize) -> CacheKey {
    CacheKey::new(file.clone(), offset, span)
}

fn pages() -> &'static PageCaches {
    memory_ctx().page_caches()
}

fn decompressed() -> &'static Cache<DecompressedPage> {
    &pages().decompressed
}

fn compressed() -> &'static CompressedCache {
    &pages().compressed
}

fn clock() -> &'static crate::memory::Clock {
    memory_ctx().clock()
}

fn header(tag: u8) -> Vec<Bytes> {
    vec![Bytes::from(vec![tag])]
}

fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

/// What a reinforced slot of `cache` gains per touch.
fn lives_at_reinforce(cache: &dyn RingCache) -> u8 {
    memory_ctx().caches().policy(cache.id()).lives_at_reinforce
}

/// Build a `len`-byte page filled with `pattern` straight into `cache`.
fn insert_page(
    cache: &Cache<DecompressedPage>,
    key: CacheKey,
    len: usize,
    pattern: u8,
) -> DecompressedPage {
    insert_page_counting(cache, key, len, pattern, counter())
}

fn insert_page_counting(
    cache: &Cache<DecompressedPage>,
    key: CacheKey,
    len: usize,
    pattern: u8,
    live_pages: Arc<AtomicUsize>,
) -> DecompressedPage {
    cache
        .insert(key, |fill| {
            for mut slab in fill.get_bytes(len) {
                slab.as_mut_slice().fill(pattern);
            }
            Ok::<_, Infallible>(DecompressedPageMeta::new(header(pattern), live_pages))
        })
        .unwrap()
}

fn all_bytes_are(page: &DecompressedPage, expected: u8) -> bool {
    page.data
        .iter()
        .flat_map(|b| b.iter())
        .all(|&b| b == expected)
}

fn total_len(page: &DecompressedPage) -> usize {
    page.data.iter().map(Bytes::len).sum()
}

/// The slot this worker's cursor for `cache` points at (where small entries pack).
fn cursor_slot(cache: &dyn RingCache) -> usize {
    memory_ctx().fill_cursor(cache.id()).slot()
}

fn release_cursor(cache: &dyn RingCache) {
    memory_ctx().fill_cursor(cache.id()).release();
}

/// Cache one whole slot of `shared_file()`'s compressed bytes at slot-aligned offset
/// `slot_number * BUFFER_SIZE`, touched a few times since, so it holds lives
/// like any extent an earlier query used. Returns its slot.
fn insert_compressed_slot(slot_number: usize) -> usize {
    let lookups = compressed().get(&shared_file(), slot_number * BUFFER_SIZE, BUFFER_SIZE);
    let missing = lookups[0].missing_extent().unwrap();
    missing.commit();
    let slot = memory_ctx().fill_cursor(compressed().id()).slot();
    for _ in 0..3 {
        clock().touch(slot);
    }
    slot
}

/// Cache `[offset, offset + len)` of `shared_file()` compressed and committed. Returns
/// the slot it was packed into.
fn insert_compressed(offset: usize, len: usize) -> usize {
    let lookups = compressed().get(&shared_file(), offset, len);
    let missing = lookups[0].missing_extent().unwrap();
    missing.commit();
    memory_ctx().fill_cursor(compressed().id()).slot()
}

/// Insert one decompressed page over `[offset, offset + len)` of `shared_file()` that
/// takes a whole slot by itself, then release the cursor so it is evictable.
fn insert_decompressed_slot(offset: usize, len: usize) -> usize {
    insert_page(
        decompressed(),
        key_in(&shared_file(), offset, len),
        BUFFER_SIZE,
        0,
    );
    let slot = cursor_slot(decompressed());
    release_cursor(decompressed());
    slot
}

fn compressed_has_misses(offset: usize, len: usize) -> bool {
    compressed()
        .get(&shared_file(), offset, len)
        .iter()
        .any(|l| l.missing_extent().is_some())
}

/// Take every slot still in the free pool as working memory, so the next
/// request has to evict.
fn take_every_free_slot() -> Vec<crate::memory::WriteBuffer> {
    std::iter::from_fn(|| memory_ctx().pop_free_idx(true))
        .map(|idx| memory_ctx().ring().try_write(idx).unwrap())
        .collect()
}

// -- Cache<V> on its own --

#[test]
fn get_misses_for_an_absent_key() {
    init_test_free_pool(1);

    assert!(decompressed().get(&key(0)).is_none());
}

#[test]
fn two_pages_pack_into_one_slot() {
    init_test_free_pool(4);

    insert_page(decompressed(), key(0), 100, 0xAA);
    insert_page(decompressed(), key(100), 100, 0xBB);

    let a = decompressed().get(&key(0)).unwrap();
    let b = decompressed().get(&key(100)).unwrap();
    assert_eq!(clock().count_owned(decompressed().id()), 1);
    assert!(all_bytes_are(&a, 0xAA));
    assert!(all_bytes_are(&b, 0xBB));
}

#[test]
fn a_straddling_page_reads_back_its_bytes() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), BUFFER_SIZE - 7, 0x11); // leave a 7-byte tail

    let built = insert_page(decompressed(), key(1000), 100, 0x22); // 7 + 93 across two slots

    let hit = decompressed().get(&key(1000)).unwrap();
    assert_eq!(hit.data.len(), 2, "two slabs across the slot boundary");
    assert_eq!(total_len(&hit), 100);
    assert!(all_bytes_are(&hit, 0x22));
    assert!(all_bytes_are(&built, 0x22));
}

#[test]
fn a_hit_is_ring_backed_and_copy_free() {
    init_test_free_pool(2);
    let built = insert_page(decompressed(), key(0), 8, 0);

    let hit = decompressed().get(&key(0)).unwrap();

    assert_eq!(hit.data[0].as_ptr(), built.data[0].as_ptr());
}

#[test]
fn a_hit_carries_the_header_stored_beside_it() {
    init_test_free_pool(2);
    insert_page(decompressed(), key(0), 8, 9);

    let hit = decompressed().get(&key(0)).unwrap();

    assert_eq!(hit.header, header(9));
}

#[test]
fn a_different_span_at_the_same_offset_misses() {
    init_test_free_pool(2);
    insert_page(decompressed(), key_in(&shared_file(), 0, 100), 8, 0);

    assert!(
        decompressed()
            .get(&key_in(&shared_file(), 0, 200))
            .is_none()
    );
}

#[test]
fn a_lost_dup_race_returns_readable_bytes_and_keeps_the_incumbent() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), 50, 0xAA);

    let dup = insert_page(decompressed(), key(0), 50, 0xBB);

    assert!(all_bytes_are(&dup, 0xBB));
    assert!(all_bytes_are(&decompressed().get(&key(0)).unwrap(), 0xAA));
}

#[test]
fn a_failed_build_caches_nothing_and_leaves_the_slot_recyclable() {
    init_test_free_pool(4);
    let failed = decompressed().insert(key(0), |fill| {
        fill.get_bytes(64);
        Err::<DecompressedPageMeta, &str>("decompress failed")
    });
    let slot = cursor_slot(decompressed());
    release_cursor(decompressed());

    let reclaimed = decompressed().reclaim(slot);

    assert!(matches!(failed, Err("decompress failed")));
    assert!(reclaimed.is_some(), "only a stale tenant, freely recycled");
    assert!(decompressed().is_empty());
}

#[test]
fn the_live_page_count_tracks_residency() {
    init_test_free_pool(4);
    let live = counter();
    insert_page_counting(decompressed(), key(0), 8, 0, live.clone());
    insert_page_counting(decompressed(), key(0), 8, 0, live.clone()); // loses the race
    let while_cached = live.load(std::sync::atomic::Ordering::Relaxed);

    decompressed().invalidate(&shared_file());

    assert_eq!(while_cached, 1);
    assert_eq!(live.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn reclaim_evicts_every_tenant_of_the_slot() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), 100, 1);
    insert_page(decompressed(), key(100), 100, 2);
    let slot = cursor_slot(decompressed());
    release_cursor(decompressed());

    let reclaimed = decompressed().reclaim(slot).unwrap();

    assert_eq!(reclaimed.buffer.slot_idx, slot);
    assert_eq!(reclaimed.dropped.len(), 2);
    assert!(decompressed().get(&key(0)).is_none());
    assert!(decompressed().get(&key(100)).is_none());
    assert!(decompressed().is_empty());
}

#[test]
fn evicting_one_slot_of_a_straddler_kills_the_entry_whole() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), BUFFER_SIZE - 7, 0x11);
    insert_page(decompressed(), key(1000), 100, 0x22); // 7 in slot A, 93 in slot B
    let second_slot = cursor_slot(decompressed());
    release_cursor(decompressed());

    let reclaimed = decompressed().reclaim(second_slot);

    assert!(reclaimed.is_some());
    assert!(
        decompressed().get(&key(1000)).is_none(),
        "straddler died whole"
    );
    assert!(decompressed().get(&key(0)).is_some());
}

#[test]
fn a_stale_partner_slot_recycles_cleanly() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), BUFFER_SIZE - 7, 0x11);
    let first_slot = cursor_slot(decompressed());
    insert_page(decompressed(), key(1000), 100, 0x22);
    let second_slot = cursor_slot(decompressed());
    release_cursor(decompressed());
    decompressed().reclaim(second_slot); // strands the straddler's bytes in first_slot

    let reclaimed = decompressed().reclaim(first_slot);

    assert!(reclaimed.is_some(), "stale tenant filtered, slot recycled");
    assert!(decompressed().is_empty());
}

#[test]
fn reclaiming_a_stale_orphan_spares_the_reinserted_entry() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), 64, 0xAA);
    let orphan_slot = cursor_slot(decompressed());
    decompressed().invalidate(&shared_file()); // the entry is gone; its tenant is now stale
    release_cursor(decompressed()); // force the re-insert into a fresh slot
    insert_page(decompressed(), key(0), 64, 0xBB);
    release_cursor(decompressed());

    let reclaimed = decompressed().reclaim(orphan_slot).unwrap();

    assert_eq!(reclaimed.buffer.slot_idx, orphan_slot);
    assert_eq!(clock().owner(orphan_slot), None);
    assert!(decompressed().get(&key(0)).is_some());
}

#[test]
fn evicting_a_files_last_entry_releases_its_file_handle() {
    init_test_free_pool(2);
    let open_file = new_file();
    let OpenFile::Local(file) = &open_file else {
        unreachable!()
    };
    insert_page(decompressed(), key_in(&open_file, 0, 1), 8, 0);
    let slot = cursor_slot(decompressed());
    release_cursor(decompressed());

    decompressed().reclaim(slot);

    assert_eq!(file.strong_count(), 1);
}

#[test]
fn reclaim_is_none_for_an_unowned_slot() {
    init_test_free_pool(1);

    assert!(decompressed().reclaim(0).is_none());
}

#[test]
fn invalidate_drops_only_the_matching_file() {
    init_test_free_pool(4);
    let a = new_file();
    let b = new_file();
    insert_page(decompressed(), key_in(&a, 0, 1), 8, 0);
    insert_page(decompressed(), key_in(&b, 0, 1), 8, 0);

    decompressed().invalidate(&a);

    assert!(decompressed().get(&key_in(&a, 0, 1)).is_none());
    assert!(decompressed().get(&key_in(&b, 0, 1)).is_some());
}

#[test]
fn clear_drops_everything() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), 8, 0);
    insert_page(decompressed(), key(100), 8, 0);

    let dropped = decompressed().clear();

    assert_eq!(dropped, 2);
    assert!(decompressed().is_empty());
    assert_eq!(clock().count_owned(decompressed().id()), 0);
}

#[test]
fn a_disabled_cache_builds_into_working_memory_and_never_hits() {
    init_test_free_pool(2);
    let cache: Cache<DecompressedPage> = Cache::with_enabled(CacheId::new(0), 128, false);

    let built = insert_page(&cache, key(0), 8, 7);

    assert!(all_bytes_are(&built, 7));
    assert!(cache.get(&key(0)).is_none());
    assert!(cache.is_empty());
}

#[test]
fn a_disabled_cache_answers_a_range_lookup_with_one_gap() {
    init_test_free_pool(2);
    let cache: Cache<DecompressedPage> = Cache::with_enabled(CacheId::new(0), 128, false);
    insert_page(&cache, key_in(&shared_file(), 0, 10), 8, 9);

    let segments = cache.get_range(&shared_file(), 0, 10);

    assert_eq!(shapes(&segments), vec![Shape::Gap(0, 10)]);
}

#[test]
fn is_empty_tracks_the_entry_count() {
    init_test_free_pool(2);
    assert!(decompressed().is_empty());

    insert_page(decompressed(), key(0), 8, 0);
    let while_cached = decompressed().is_empty();
    decompressed().invalidate(&shared_file());

    assert!(!while_cached);
    assert!(decompressed().is_empty());
}

// -- get_range --

#[derive(Debug, PartialEq, Eq)]
enum Shape {
    Gap(usize, usize),
    Cached(usize, usize, u8),
}

fn shapes(segments: &[Segment<DecompressedPage>]) -> Vec<Shape> {
    segments
        .iter()
        .map(|s| match s {
            Segment::Gap { offset, len } => Shape::Gap(*offset, *len),
            Segment::Cached {
                offset,
                span,
                value,
            } => Shape::Cached(*offset, *span, value.header[0][0]),
        })
        .collect()
}

#[test]
fn get_range_over_an_empty_cache_is_one_gap() {
    init_test_free_pool(1);

    let segments = decompressed().get_range(&shared_file(), 100, 50);

    assert_eq!(shapes(&segments), vec![Shape::Gap(100, 50)]);
}

#[test]
fn get_range_finds_a_hit_surrounded_by_gaps() {
    init_test_free_pool(2);
    insert_page(decompressed(), key_in(&shared_file(), 10, 5), 8, 7);
    release_cursor(decompressed());

    let segments = decompressed().get_range(&shared_file(), 0, 20);

    assert_eq!(
        shapes(&segments),
        vec![
            Shape::Gap(0, 10),
            Shape::Cached(10, 5, 7),
            Shape::Gap(15, 5)
        ]
    );
}

#[test]
fn get_range_has_no_gap_between_adjacent_hits() {
    init_test_free_pool(2);
    insert_page(decompressed(), key_in(&shared_file(), 0, 5), 8, 1);
    insert_page(decompressed(), key_in(&shared_file(), 5, 5), 8, 2);
    release_cursor(decompressed());

    let segments = decompressed().get_range(&shared_file(), 0, 10);

    assert_eq!(
        shapes(&segments),
        vec![Shape::Cached(0, 5, 1), Shape::Cached(5, 5, 2)]
    );
}

// -- the caches together --

#[test]
fn evict_takes_decompressed_while_compressed_is_under_target() {
    init_test_free_pool(16);
    insert_compressed(0, 4096);
    release_cursor(compressed());
    for i in 0..10 {
        insert_decompressed_slot((10 + i) * 4096, 4096); // compressed is 1 of 11: 9%
    }

    memory_ctx().evict();

    assert!(
        !compressed_has_misses(0, 4096),
        "compressed is under its share and keeps its extent"
    );
    assert_eq!(clock().count_owned(decompressed().id()), 9);
}

#[test]
fn evict_takes_compressed_while_over_its_target_share() {
    init_test_free_pool(8);
    insert_compressed(0, 4096);
    release_cursor(compressed());
    insert_decompressed_slot(10 * 4096, 4096); // 1 of each: compressed 50%, over target

    memory_ctx().evict();

    assert_eq!(clock().count_owned(compressed().id()), 0);
    assert_eq!(clock().count_owned(decompressed().id()), 1);
}

/// Compressed is on its target, so decompressed is the one to give, but
/// readers hold every decompressed slot. Compressed gives instead of the ring
/// looking exhausted.
#[test]
fn working_memory_takes_compressed_slots_when_every_decompressed_one_is_pinned() {
    init_test_free_pool(128);
    for slot_number in 0..30 {
        insert_compressed_slot(slot_number); // 30 of 100: exactly on target
    }
    release_cursor(compressed());
    let _pinned: Vec<DecompressedPage> = (0..70)
        .map(|i| {
            insert_decompressed_slot((100 + i) * BUFFER_SIZE, 4096);
            decompressed()
                .get(&key_in(&shared_file(), (100 + i) * BUFFER_SIZE, 4096))
                .unwrap()
        })
        .collect();
    let _working_memory = take_every_free_slot();

    let _buffer = memory_ctx().get_write_buffer(true);

    assert_eq!(clock().count_owned(compressed().id()), 29);
}

/// The cache over its share has a slot with lives to spend beside a pinned
/// zero-life one: its hand keeps aging that slot and takes it once drained,
/// and its sibling is left alone the whole time.
#[test]
fn an_ageing_slot_gives_way_before_the_sibling_is_swept() {
    init_test_free_pool(8);
    let pinned_slot = insert_compressed(0, BUFFER_SIZE);
    release_cursor(compressed());
    let _reader = memory_ctx().ring().try_read(pinned_slot).unwrap();
    insert_compressed_slot(1); // a second compressed slot, holding more lives
    release_cursor(compressed());
    insert_decompressed_slot(10 * 4096, 4096); // compressed 2 of 3: over target

    drop(memory_ctx().evict());

    assert_eq!(
        clock().count_owned(compressed().id()),
        1,
        "the ageing slot gave way"
    );
    assert_eq!(
        clock().count_owned(decompressed().id()),
        1,
        "the sibling was left alone"
    );
    assert_eq!(clock().owner(pinned_slot), Some(compressed().id()));
    assert_eq!(clock().hand_position(decompressed().id(), 0), 0);
}

/// Compressed is over its share, but readers hold every compressed slot. Its
/// hand is left where it is and decompressed gives at once, instead of the
/// ring looking exhausted.
#[test]
fn a_cache_whose_every_slot_is_pinned_is_not_swept() {
    init_test_free_pool(8);
    let mut readers = Vec::new();
    for offset in [0, BUFFER_SIZE] {
        let slot = insert_compressed(offset, BUFFER_SIZE);
        release_cursor(compressed());
        readers.push(memory_ctx().ring().try_read(slot).unwrap());
    }
    for i in 0..3 {
        insert_decompressed_slot((10 + i) * 4096, 4096); // compressed 2 of 5: over target
    }

    drop(memory_ctx().evict());

    assert_eq!(
        clock().count_owned(compressed().id()),
        2,
        "the pinned slots survive"
    );
    assert_eq!(
        clock().count_owned(decompressed().id()),
        2,
        "decompressed gave"
    );
    assert_eq!(clock().hand_position(compressed().id(), 0), 0);
}

/// Each cache's slots age by passes of its own hand. Evictions that take from
/// decompressed leave the compressed slots their lives, however many there
/// are.
#[test]
fn evicting_from_one_cache_does_not_age_its_sibling() {
    init_test_free_pool(16);
    let compressed_slot = insert_compressed(0, 4096);
    release_cursor(compressed());
    for i in 0..10 {
        insert_decompressed_slot((10 + i) * 4096, 4096); // compressed is 1 of 11: 9%
    }
    let lives_before = clock().lives(compressed_slot);

    for _ in 0..5 {
        drop(memory_ctx().evict());
    }

    assert_eq!(clock().count_owned(decompressed().id()), 5);
    assert_eq!(clock().lives(compressed_slot), lives_before);
}

#[test]
fn evicting_a_compressed_extent_reinforces_its_decompressed_twin() {
    init_test_free_pool(8);
    insert_compressed(0, 4096);
    release_cursor(compressed());
    let twin_slot = insert_decompressed_slot(0, 4096); // same file range
    assert_eq!(clock().lives(twin_slot), 1);

    memory_ctx().evict(); // compressed at 50% is the one to go

    assert_eq!(clock().count_owned(compressed().id()), 0);
    assert_eq!(
        clock().lives(twin_slot),
        1 + lives_at_reinforce(decompressed())
    );
}

#[test]
fn evicting_a_decompressed_page_reinforces_its_compressed_source() {
    init_test_free_pool(8);
    let source_slot = insert_compressed(0, 4096);
    release_cursor(compressed());
    let page_slot = insert_decompressed_slot(0, 4096);
    let lives_before = clock().lives(source_slot);

    memory_ctx()
        .caches()
        .reclaim(decompressed().id(), page_slot)
        .unwrap();

    assert_eq!(
        clock().lives(source_slot),
        lives_before + lives_at_reinforce(compressed())
    );
}

/// The member to evict is chosen from whatever counts the caller supplies,
/// which the evictor makes its own region's: a family over target in one
/// region and under in another gives from a different member in each.
#[test]
fn the_member_to_evict_is_judged_by_the_counts_of_the_region_being_swept() {
    init_test_free_pool(1);
    let pages = memory_ctx().caches().family_of(compressed().id());
    let compressed_heavy_region = |id: CacheId| {
        if id == compressed().id() { 3 } else { 1 }
    };
    let decompressed_heavy_region = |id: CacheId| {
        if id == compressed().id() { 1 } else { 3 }
    };

    let first = pages.choose_member_to_evict(compressed_heavy_region, |_| true, 0);
    let second = pages.choose_member_to_evict(decompressed_heavy_region, |_| true, 0);

    assert_eq!(first, Some(compressed().id()));
    assert_eq!(second, Some(decompressed().id()));
}
