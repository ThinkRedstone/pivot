//! The compressed cache on its own: packing, lookups, fills, failure, and
//! reclaim. How it behaves beside its family lives in the cache module's
//! cross-cache tests.

use super::*;
use crate::io::LocalFile;
use crate::memory::context::{init_test_free_pool, memory_ctx};
use bytes::Bytes;

/// A local file used as the cache key throughout these tests. One shared open
/// file, so every call keys the same bucket.
fn shared_file() -> OpenFile {
    use std::sync::OnceLock;
    static FILE: OnceLock<LocalFile> = OnceLock::new();
    OpenFile::Local(
        FILE.get_or_init(|| LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
            .clone(),
    )
}

/// A fresh, independent local file (distinct bucket).
fn new_file() -> OpenFile {
    OpenFile::Local(LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
}

fn cache() -> &'static CompressedCache {
    &memory_ctx().page_caches().compressed
}

/// Drop this worker's fill pin so its current fill slot becomes evictable.
fn release_fill_cursor() {
    memory_ctx().fill_cursor(cache().id()).release();
}

/// A deterministic byte for absolute file offset `off`, so cache contents are
/// predictable for any slice regardless of where they were packed.
fn pattern_at(off: usize) -> u8 {
    off.wrapping_mul(7).wrapping_add(1) as u8
}

/// Read the pattern into every missing block of `lookups` and commit it.
fn fill_pattern(lookups: &[CacheLookup]) {
    for lookup in lookups {
        if let Some(block) = lookup.missing_extent() {
            for i in 0..block.len() {
                unsafe { *block.dest().add(i) = pattern_at(block.file_offset() + i) };
            }
            block.commit();
        }
    }
}

fn has_misses(lookups: &[CacheLookup]) -> bool {
    lookups.iter().any(|l| l.missing_extent().is_some())
}

/// Concatenate every lookup's bytes in file order.
fn assemble(lookups: Vec<CacheLookup>) -> Vec<u8> {
    let mut out = Vec::new();
    for lookup in lookups {
        out.extend_from_slice(&lookup.into_data());
    }
    out
}

fn assert_pattern(bytes: &[u8], file_start: usize) {
    for (i, &b) in bytes.iter().enumerate() {
        assert_eq!(b, pattern_at(file_start + i), "byte {i}");
    }
}

fn missing_extents(lookups: &[CacheLookup]) -> Vec<MissingExtent> {
    lookups
        .iter()
        .filter_map(|l| l.missing_extent().cloned())
        .collect()
}

fn slot_of(missing: &MissingExtent) -> usize {
    missing.extent().slot_idx as usize
}

#[test]
fn an_abandoned_fill_fails_its_followers_and_frees_the_range() {
    init_test_free_pool(8);
    let file = new_file();
    let owner = cache().get(&file, 0, BLOCK_SIZE);
    let follower = cache().get(&file, 0, BLOCK_SIZE);
    let followed = follower[0].missing_extent().unwrap().clone();
    assert!(owner[0].missing_extent().unwrap().is_fill_owner());
    assert!(!followed.is_fill_owner());

    drop(owner);

    assert!(followed.has_failed());
    let fresh = cache().get(&file, 0, BLOCK_SIZE);
    assert!(fresh[0].missing_extent().unwrap().is_fill_owner());
}

#[test]
fn a_committed_fill_survives_its_owner_being_dropped() {
    init_test_free_pool(8);
    let file = new_file();
    let owner = cache().get(&file, 0, BLOCK_SIZE);
    let followed = cache().get(&file, 0, BLOCK_SIZE)[0]
        .missing_extent()
        .unwrap()
        .clone();
    fill_pattern(&owner);

    drop(owner);

    assert!(followed.is_committed());
    assert!(!followed.has_failed());
    assert!(!has_misses(&cache().get(&file, 0, BLOCK_SIZE)));
}

#[test]
fn miss_then_fill_then_hit() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());

    let miss = cache().get(&shared_file(), 0, 10);
    assert!(has_misses(&miss), "fresh range should miss");
    fill_pattern(&miss);
    drop(miss);

    let hit = cache().get(&shared_file(), 0, 10);
    assert!(!has_misses(&hit), "filled range should hit");
    assert_pattern(&assemble(hit), 0);
}

#[test]
fn an_unwinding_reclaim_leaves_the_slot_cached_and_out_of_the_pool() {
    // Poison the outer map lock so reclaim unwinds mid-probe; the slot the
    // cache still references must not fall into the free pool. The pool's
    // only slot becomes the fill slot, so a leak into the pool is visible.
    init_test_free_pool(1);
    cache().invalidate(&shared_file());
    let miss = cache().get(&shared_file(), 0, 10);
    fill_pattern(&miss);
    let slot = slot_of(miss[0].missing_extent().unwrap());
    drop(miss);
    release_fill_cursor();
    cache().poison_file_map();

    let reclaim = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache().reclaim(slot);
    }));

    assert!(reclaim.is_err(), "reclaim unwinds on the poisoned lock");
    assert_eq!(memory_ctx().pop_free_idx(false), None);
    assert_eq!(memory_ctx().clock().owner(slot), Some(cache().id()));
    assert!(memory_ctx().ring().try_write(slot).is_some());
}

#[test]
fn coalesces_contiguous_missing_blocks_into_one_extent() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());

    let miss = cache().get(&shared_file(), 0, 2 * BLOCK_SIZE + 1);

    let extents = missing_extents(&miss);
    assert_eq!(extents.len(), 1);
    assert_eq!(extents[0].extent().block_count, 3);
    assert_eq!(extents[0].len(), 3 * BLOCK_SIZE);
}

#[test]
fn partial_validity_only_reports_the_holes() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    let first = cache().get(&shared_file(), 0, 10);
    fill_pattern(&first);
    drop(first);

    let again = cache().get(&shared_file(), 0, 2 * BLOCK_SIZE + 1);

    let extents = missing_extents(&again);
    assert_eq!(extents.len(), 1);
    assert_eq!(extents[0].file_offset(), BLOCK_SIZE);
    assert_eq!(extents[0].extent().block_count, 2);
}

#[test]
fn consecutive_misses_pack_into_one_slot() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());

    let a = cache().get(&shared_file(), 0, 10);
    let b = cache().get(&shared_file(), 50 * BLOCK_SIZE, 50 * BLOCK_SIZE + 10);

    let a = a[0].missing_extent().unwrap();
    let b = b[0].missing_extent().unwrap();
    assert_eq!(slot_of(a), slot_of(b), "should share the fill slot");
    assert_eq!(a.extent().first_slot_block, 0);
    assert_eq!(b.extent().first_slot_block, 1);
}

#[test]
fn unrelated_files_pack_into_one_slot() {
    init_test_free_pool(8);
    let a = new_file();
    let b = new_file();

    let ra = cache().get(&a, 0, BLOCK_SIZE);
    let rb = cache().get(&b, 0, BLOCK_SIZE);

    assert_eq!(
        slot_of(ra[0].missing_extent().unwrap()),
        slot_of(rb[0].missing_extent().unwrap()),
    );
}

#[test]
fn a_read_past_the_fill_buffer_uses_a_new_slot() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());

    let miss = cache().get(&shared_file(), 0, BLOCKS_PER_SLOT * BLOCK_SIZE + 1);

    let extents = missing_extents(&miss);
    assert_eq!(extents.len(), 2);
    assert_eq!(extents[0].extent().block_count as usize, BLOCKS_PER_SLOT);
    assert_eq!(extents[1].extent().block_count, 1);
    assert_ne!(slot_of(&extents[0]), slot_of(&extents[1]));
    fill_pattern(&miss);
    assert_pattern(&assemble(miss), 0);
}

#[test]
fn overlap_dedup_only_fetches_the_holes() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    let first = cache().get(&shared_file(), 0, 4 * BLOCK_SIZE);
    fill_pattern(&first);
    drop(first);

    let second = cache().get(&shared_file(), 2 * BLOCK_SIZE, 4 * BLOCK_SIZE);

    let extents = missing_extents(&second);
    assert_eq!(extents.len(), 1);
    assert_eq!(extents[0].file_offset(), 4 * BLOCK_SIZE);
    assert_eq!(extents[0].extent().block_count, 2);
}

#[test]
fn hit_reassembles_bytes_across_lookups() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    let miss = cache().get(&shared_file(), BLOCK_SIZE - 5, BLOCK_SIZE + 10);
    fill_pattern(&miss);
    drop(miss);

    let bytes = assemble(cache().get(&shared_file(), BLOCK_SIZE - 5, BLOCK_SIZE + 10));

    assert_pattern(&bytes, BLOCK_SIZE - 5);
    assert_eq!(bytes.len(), BLOCK_SIZE + 10);
}

#[test]
fn hit_returns_the_exact_requested_byte_range() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    let miss = cache().get(&shared_file(), 0, BLOCK_SIZE);
    fill_pattern(&miss);
    drop(miss);

    let bytes = assemble(cache().get(&shared_file(), 3, 6));

    assert_eq!(bytes.len(), 6);
    assert_pattern(&bytes, 3);
}

#[test]
fn committed_bytes_serve_an_overlapping_later_lookup() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    let first = cache().get(&shared_file(), 0, 2 * BLOCK_SIZE);
    fill_pattern(&first);
    drop(first);

    let bytes = assemble(cache().get(&shared_file(), BLOCK_SIZE / 2, BLOCK_SIZE));

    assert_pattern(&bytes, BLOCK_SIZE / 2);
    assert_eq!(bytes.len(), BLOCK_SIZE);
}

#[test]
fn into_data_yields_the_freshly_filled_range() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());

    let miss = cache().get(&shared_file(), 10, BLOCK_SIZE);
    fill_pattern(&miss);
    let bytes = assemble(miss);

    assert_eq!(bytes.len(), BLOCK_SIZE);
    assert_pattern(&bytes, 10);
}

#[test]
fn interleaved_gaps_split_then_fill_to_a_full_hit() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    fill_pattern(&cache().get(&shared_file(), 0, 1));
    fill_pattern(&cache().get(&shared_file(), 2 * BLOCK_SIZE, 2 * BLOCK_SIZE + 1));

    let gaps = cache().get(&shared_file(), 0, 4 * BLOCK_SIZE);
    fill_pattern(&gaps);
    drop(gaps);
    let bytes = assemble(cache().get(&shared_file(), 0, 4 * BLOCK_SIZE));

    assert_pattern(&bytes, 0);
}

#[test]
fn a_partially_cached_range_keeps_the_cached_bytes() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    fill_pattern(&cache().get(&shared_file(), 0, 1));

    let m = cache().get(&shared_file(), 0, 2 * BLOCK_SIZE);

    let extents = missing_extents(&m);
    assert_eq!(extents.len(), 1);
    assert_eq!(extents[0].file_offset(), BLOCK_SIZE);
    fill_pattern(&m);
    assert_pattern(&assemble(m), 0);
}

#[test]
fn get_assembles_a_range_spanning_cached_and_uncached_lookups() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    fill_pattern(&cache().get(&shared_file(), 0, BLOCK_SIZE)); // [0, BLOCK_SIZE) cached, [BLOCK_SIZE, 2*BLOCK_SIZE) not

    let parts = cache().get(&shared_file(), 0, 2 * BLOCK_SIZE);

    assert!(has_misses(&parts), "the second lookup is a hole");
    fill_pattern(&parts);
    assert_pattern(&assemble(parts), 0);
}

#[test]
fn reopening_a_file_forgets_its_cached_extents() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    fill_pattern(&cache().get(&shared_file(), 0, BLOCK_SIZE));

    memory_ctx()
        .caches()
        .invalidate_reopened_file(&shared_file());

    assert!(has_misses(&cache().get(&shared_file(), 0, BLOCK_SIZE)));
}

#[test]
fn get_write_buffer_evicts_compressed_data_when_the_pool_is_empty() {
    init_test_free_pool(1);
    cache().invalidate(&shared_file());
    fill_pattern(&cache().get(&shared_file(), 0, BLOCK_SIZE));
    release_fill_cursor();

    let write = memory_ctx().get_write_buffer(false);

    assert_eq!(memory_ctx().clock().count_owned(cache().id()), 0);
    drop(write);
    assert!(has_misses(&cache().get(&shared_file(), 0, BLOCK_SIZE)));
}

/// The evictor must not reclaim a ring slot still parked in the free pool.
#[test]
fn evict_does_not_steal_a_slot_from_the_free_pool() {
    init_test_free_pool(2);
    cache().invalidate(&shared_file());
    fill_pattern(&cache().get(&shared_file(), 0, BLOCK_SIZE));
    release_fill_cursor();

    let evicted = memory_ctx().evict();

    let still_pooled: Vec<usize> =
        std::iter::from_fn(|| memory_ctx().pop_free_idx(false)).collect();
    assert!(!still_pooled.contains(&evicted.slot_idx));
}

#[test]
fn evicting_a_slot_drops_all_tenant_extents() {
    init_test_free_pool(16);
    let files: Vec<OpenFile> = (0..3).map(|_| new_file()).collect();
    for f in &files {
        fill_pattern(&cache().get(f, 0, BLOCK_SIZE)); // all pack into one fill slot
    }
    release_fill_cursor();

    memory_ctx().evict();

    for f in &files {
        assert!(
            !cache().holds_file(f),
            "an evicted extent's file entry leaked"
        );
    }
}

#[test]
fn reopening_a_file_nothing_is_cached_of_keeps_no_handle_to_it() {
    init_test_free_pool(4);
    let file = new_file();

    cache().invalidate(&file);

    assert!(!cache().holds_file(&file));
}

#[test]
fn a_recycled_slot_makes_stale_placements_miss() {
    init_test_free_pool(4);
    cache().invalidate(&shared_file());
    fill_pattern(&cache().get(&shared_file(), 0, BLOCK_SIZE));
    release_fill_cursor();

    memory_ctx().evict();

    assert!(has_misses(&cache().get(&shared_file(), 0, BLOCK_SIZE)));
}

#[test]
fn reading_a_file_after_its_entry_was_pruned_does_not_panic() {
    init_test_free_pool(8);
    let f = new_file();
    fill_pattern(&cache().get(&f, 0, BLOCK_SIZE));
    release_fill_cursor();
    memory_ctx().evict();
    assert!(!cache().holds_file(&f));

    let lookups = cache().get(&f, 0, BLOCK_SIZE);

    assert!(has_misses(&lookups));
    assert!(cache().holds_file(&f));
}

#[test]
fn clear_forgets_everything_and_re_reads() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    fill_pattern(&cache().get(&shared_file(), 0, BLOCK_SIZE));

    let dropped = cache().clear();

    assert!(dropped >= 1);
    assert!(has_misses(&cache().get(&shared_file(), 0, BLOCK_SIZE)));
}

#[test]
fn a_hit_is_a_zero_copy_view_of_the_slot() {
    init_test_free_pool(8);
    cache().invalidate(&shared_file());
    let miss = cache().get(&shared_file(), 0, BLOCK_SIZE);
    let dest = miss[0].missing_extent().unwrap().dest();
    fill_pattern(&miss);
    drop(miss);

    let hit: Bytes = cache()
        .get(&shared_file(), 0, BLOCK_SIZE)
        .pop()
        .unwrap()
        .into_data();

    assert_eq!(hit.as_ptr(), dest as *const u8);
}
