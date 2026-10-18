//! Open-addressing hash table with linear probing, optimized for GROUP BY.
//!
//! See [`HashTable`] for the full design rationale.

use crate::memory::{BUFFER_SIZE, Slab, SlabAllocator};
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::TableReader;
use crate::operations::unary::group::values::AggregationValue;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use tracing::debug;

/// Maximum fraction of slots that can be occupied before the table is
/// considered undersized. Used by [`AggregatedTable`](super::AggregatedTable)
/// to decide when to stop inserting and create a new table.
pub const MAX_LOAD_FACTOR: f64 = 0.7;

/// A `LiveKey` is a key that has *not* yet been persisted, and therefore needs to explicitly be
/// persisted if it is saved within the HashTable. The HashTable may decide not to persist a key
/// if, for example, the same key already exists.
pub trait LiveKey {
    type Persisted: PersistedKey;

    /// Compare this live key against a persisted key in the table.
    fn eq_persisted(&self, other: &Self::Persisted) -> bool;

    /// Save the key
    fn persist(self) -> Self::Persisted;
}

/// A key that has been persisted in the HashTable. Any PersistedKey can be referenced from within
/// the HashTable as long as the HashTable is alive
pub trait PersistedKey: Copy + Clone + Default {
    /// Whether this key holds out-of-line data (an arena string blob) that
    /// `eq_persisted` will chase. `true` for string-bearing keys, `false` for
    /// fixed-width integer keys. Used to decide whether the merge's blob prefetch
    /// is worth its per-row cost (it is only when there's actually a cold blob).
    const HAS_BLOB: bool = false;

    /// Prefetch the arena blob this key's `eq_persisted` will read, hiding the
    /// scattered cache miss of a non-inline string before the `memcmp`. No-op for
    /// keys with no out-of-line data (integers, inline strings).
    #[inline(always)]
    fn prefetch_blob(&self, _arena: &SharedArena) {}
}

/// We reimplement LiveKey for any PersistentKey to allow saving an already persisted key within
/// the HashTable (for example, if it was saved in another HashTable's arena previously)
impl<P: PersistedKey + PartialEq> LiveKey for P {
    type Persisted = Self;

    #[inline(always)]
    fn eq_persisted(&self, other: &Self) -> bool {
        self == other
    }

    fn persist(self) -> Self::Persisted {
        self
    }
}

/// Borrowed view of an occupied table entry.
///
/// Entries have a runtime stride, so they cannot be represented by references
/// to one statically sized Rust struct.
pub struct EntryView<'a, K, S: ?Sized> {
    /// The stored hash (never 0 for an occupied entry).
    pub hash: u64,
    /// The persisted group key.
    pub key: &'a K,
    /// The group's stored aggregation value.
    pub stored: &'a S,
}

/// Maximum number of entries allowed for a table with `len` total slots.
fn max_load_for_len(len: usize) -> usize {
    (len as f64 * MAX_LOAD_FACTOR).round() as usize
}

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Computes a reciprocal used to replace division by `d` with a widening
/// multiply: `n / d == (n * reciprocal(d)) >> 64`.
///
/// Exact for this table's ranges: with `m = ceil(2^64 / d) = (2^64 + e) / d`
/// (`0 <= e < d`), `n * m / 2^64 = n/d + n*e/(d * 2^64)`, and the error term
/// stays below `n / 2^64`. The true quotient's fractional part is at most
/// `1 - 1/d`, so rounding can change the floor only if `n >= 2^64 / d`.
/// Here `d < 2^21`, making the result exact for every table slot index.
pub(super) fn reciprocal(d: u64) -> u64 {
    assert!(d >= 2, "an entry never fills half a slab");
    ((1u128 << 64).div_ceil(d as u128)) as u64
}

/// Divides `n` using a reciprocal returned by [`reciprocal`].
#[inline(always)]
pub(super) fn fast_div(n: usize, m: u64) -> usize {
    (((n as u128) * (m as u128)) >> 64) as usize
}

/// Computes slab bases adjusted by their first global entry offset.
///
/// With these bases, an entry address is `bases[slab] + index * stride`.
/// Adjusted addresses may numerically precede an allocation, but only the
/// reconstructed in-bounds address is dereferenced.
fn adjusted_bases(slabs: &[Slab], entries_per_slab: usize, stride: usize) -> Vec<usize> {
    slabs
        .iter()
        .enumerate()
        .map(|(slab_index, slab)| {
            (slab.ptr as usize).wrapping_sub(slab_index * entries_per_slab * stride)
        })
        .collect()
}

/// Runtime field offsets, stride, and alignment for one table entry.
pub(super) struct EntryLayout {
    pub(super) hash_offset: usize,
    pub(super) key_offset: usize,
    pub(super) value_offset: usize,
    pub(super) stride: usize,
    pub(super) align: usize,
}

/// Returns the byte stride for one entry of this key and value type.
pub fn entry_stride<K, V: AggregationValue + ?Sized>(ctx: &V::SharedContext) -> usize {
    entry_layout::<K, V>(V::storage_metadata(ctx)).stride
}

pub(super) fn entry_layout<K, V: AggregationValue + ?Sized>(
    metadata: V::StorageMetadata,
) -> EntryLayout {
    // Alignment and size for hash, key, and value.
    let fields = [
        (align_of::<u64>(), size_of::<u64>()),
        (align_of::<K>(), size_of::<K>()),
        (V::stored_align(), V::stored_size(metadata)),
    ];
    // A fixed comparison network lets this layout constant-fold for specialized
    // dynamic arities. Strict comparisons preserve field order on ties.
    let mut order = [0usize, 1, 2];
    if fields[order[1]].0 > fields[order[0]].0 {
        order.swap(0, 1);
    }
    if fields[order[2]].0 > fields[order[1]].0 {
        order.swap(1, 2);
    }
    if fields[order[1]].0 > fields[order[0]].0 {
        order.swap(0, 1);
    }
    let mut offsets = [0usize; 3];
    let mut cursor = 0;
    for &f in &order {
        cursor = align_up(cursor, fields[f].0);
        offsets[f] = cursor;
        cursor += fields[f].1;
    }
    let align = fields.iter().map(|&(a, _)| a).max().unwrap();
    let stride = align_up(cursor, align);
    assert!(stride <= BUFFER_SIZE, "entry stride exceeds one slab");
    EntryLayout {
        hash_offset: offsets[0],
        key_offset: offsets[1],
        value_offset: offsets[2],
        stride,
        align,
    }
}

/// Prefetch the cache line at `ptr` into L1 (x86 `T0` / ARM `pldl1keep`).
#[inline(always)]
fn prefetch_l1_line(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    // `prfm` is a hint with no architecturally-observable memory effect, so
    // `nomem` is correct and lets the compiler schedule freely around it in the
    // hot probe loop; the lint's pointer-with-nomem heuristic is a false positive.
    #[allow(clippy::pointers_in_nomem_asm_block)]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nomem, nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}

/// Prefetch the cache line at `ptr` into L2 only (x86 `T1` / ARM `pldl2keep`).
#[inline(always)]
fn prefetch_l2_line(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T1 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::pointers_in_nomem_asm_block)]
    unsafe {
        std::arch::asm!("prfm pldl2keep, [{0}]", in(reg) ptr, options(nomem, nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}

/// Linear-probing hash table for GROUP BY.
///
/// # Memory Layout
///
/// Each table computes one entry layout from its key type and aggregation
/// signature. Entries never cross a 2 MB slab boundary.
///
/// ```text
/// slab
/// +----------------+----------------+---------+----------------+
/// | entry 0        | entry 1        |   ...   | unused tail    |
/// | hash | key | V | hash | key | V |         | < one stride   |
/// +----------------+----------------+---------+----------------+
/// ```
///
/// A [`Dynamic`](crate::operations::unary::group::values::Dynamic) value
/// makes the stride a runtime value because its cell count
/// is fixed per query rather than per Rust type. Fixed signatures use the same
/// layout algorithm and retain their compact representation.
///
/// The full hash is stored beside the key and value. Hash zero is the empty
/// sentinel; a real zero hash is remapped to one. Capacity is a power of two,
/// so linear probing wraps with a mask.
///
/// # Top-bit placement
///
/// Initial slots use the high bits of the hash. Merge partitions use the same
/// prefix, so each partition maps to a contiguous slot range even when source
/// tables have different capacities:
///
/// ```text
/// hash prefix P
///      |
///      +-- 128-slot table: slots [P * 2, P * 2 + 2)
///      +-- 256-slot table: slots [P * 4, P * 4 + 4)
/// ```
///
/// This lets one merge job scan only its ranges. Storing the full hash also
/// avoids hashing large keys again during resize and merge.
pub struct HashTable<K: PersistedKey, V: AggregationValue + ?Sized> {
    slot_mask: usize,
    slot_shift: u32,
    collisions: usize,
    /// Left-shift applied to hash before computing the slot index.
    /// Set to `PARTITIONS.trailing_zeros()` for merge maps so that the partition
    /// bits (top N) are stripped and the next top bits drive slot placement.
    hash_left_shift: u32,
    /// Whole entries per 2MB slab (`BUFFER_SIZE / entry_stride`); entries never
    /// straddle a slab boundary.
    entries_per_slab: usize,
    /// Reciprocal used by [`fast_div`] to find a slab without hardware division.
    entries_per_slab_reciprocal: u64,
    /// Bytes per entry.
    entry_stride: usize,
    hash_offset: usize,
    /// Strictest field alignment, for allocating replacement slabs on resize.
    entry_alignment: usize,
    /// Value layout metadata stored once per table.
    storage_metadata: V::StorageMetadata,
    /// Slab addresses adjusted by their first global entry offset.
    adjusted_slab_bases: Vec<usize>,
    slabs: Vec<Slab>,
    length: usize,
    max_load: usize,
    // A raw-pointer marker: a plain `(K, V)` tuple would require `V: Sized`.
    _phantom: PhantomData<(K, *const V)>,
}

unsafe impl<K: PersistedKey, V: AggregationValue + ?Sized> Send for HashTable<K, V> {}

impl<K: PersistedKey, V: AggregationValue + ?Sized> HashTable<K, V> {
    /// Creates a zeroed table with power-of-two capacity.
    pub fn new(
        allocator: &mut SlabAllocator,
        expected_capacity: usize,
        hash_left_shift: u32,
        ctx: &V::SharedContext,
    ) -> Self {
        let storage_metadata = V::storage_metadata(ctx);
        let layout = entry_layout::<K, V>(storage_metadata);
        let entries_per_slab = BUFFER_SIZE / layout.stride;
        let slabs = allocator.create_strided_slabs(expected_capacity, layout.stride, layout.align);
        HashTable {
            slot_mask: expected_capacity - 1,
            length: 0,
            max_load: max_load_for_len(expected_capacity),
            hash_left_shift,
            slot_shift: u64::BITS - expected_capacity.trailing_zeros(),
            entries_per_slab,
            entries_per_slab_reciprocal: reciprocal(entries_per_slab as u64),
            entry_stride: layout.stride,
            hash_offset: layout.hash_offset,
            entry_alignment: layout.align,
            storage_metadata,
            adjusted_slab_bases: adjusted_bases(&slabs, entries_per_slab, layout.stride),
            slabs,
            _phantom: PhantomData,
            collisions: 0,
        }
    }

    /// Clears entries and counters while retaining the allocated slabs.
    pub fn clear(&mut self) {
        for slab in &mut self.slabs {
            slab.zero_out();
        }
        self.length = 0;
        self.collisions = 0;
    }

    /// Total number of slots (occupied + empty). Always a power of 2.
    pub fn capacity(&self) -> usize {
        self.slot_mask + 1
    }

    /// Returns the number of entries currently stored in the table.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Cumulative number of probe-chain collisions since the last resize.
    pub fn collisions(&self) -> usize {
        self.collisions
    }

    /// Maps a hash to a slot index using the selected high bits.
    #[inline(always)]
    fn slot_for(&self, hash: u64) -> usize {
        ((hash << self.hash_left_shift) >> self.slot_shift) as usize
    }

    /// Returns an entry address using the supplied adjusted slab bases.
    ///
    /// Entries never cross slab boundaries. [`fast_div`] identifies the slab,
    /// and the adjusted base allows the byte address to use the global index
    /// directly.
    #[inline(always)]
    fn entry_ptr_in(&self, adjusted_slab_bases: &[usize], index: usize) -> *mut u8 {
        let slab_index = fast_div(index, self.entries_per_slab_reciprocal);
        adjusted_slab_bases[slab_index].wrapping_add(index * self.entry_stride) as *mut u8
    }

    /// The address of this table's entry at `index`.
    #[inline(always)]
    fn entry_ptr(&self, index: usize) -> *mut u8 {
        self.entry_ptr_in(&self.adjusted_slab_bases, index)
    }

    /// Returns an iterator over all non-empty entries in the table.
    ///
    /// Iteration order is arbitrary (based on slot positions, not insertion order).
    /// Empty slots (hash == 0) are skipped automatically.
    pub fn iter(&self, start_offset: usize) -> HashTableIterator<'_, K, V> {
        HashTableIterator {
            reader: self.reader::<0>(),
            idx: start_offset,
        }
    }

    /// Returns `true` if the table has exceeded its [`MAX_LOAD_FACTOR`] threshold.
    pub fn undersized(&self) -> bool {
        self.len() > self.max_load
    }

    /// Returns a copyable reader over this table's slots and entries.
    ///
    /// `N` is the specialized slot count of the surrounding
    /// [`dispatch_arity`](AggregationValue::dispatch_arity) body: a nonzero
    /// `N` derives the reader's layout from that constant so its field
    /// offsets and stride arithmetic constant-fold, while `N == 0` (the
    /// runtime-arity fallback, and every fixed-size value) reads the table's
    /// stored metadata.
    #[inline(always)]
    pub(crate) fn reader<const N: usize>(&self) -> TableReader<'_, K, V> {
        let metadata = if N == 0 {
            self.storage_metadata
        } else {
            V::metadata_for_arity::<N>()
        };
        unsafe { self.reader_with_unchecked_lifetime(metadata) }
    }

    /// Builds a reader without tying its lifetime to this particular borrow.
    ///
    /// The prober already holds an exclusive table borrow for the resulting
    /// lifetime. It uses this constructor so the copied reader and that
    /// mutable borrow can live in the same struct.
    ///
    /// # Safety
    ///
    /// The returned reader must not outlive this table. It must be replaced
    /// before accessing entries after any operation that changes the table's
    /// slabs, capacity, or entry layout.
    #[inline(always)]
    unsafe fn reader_with_unchecked_lifetime<'table>(
        &self,
        metadata: V::StorageMetadata,
    ) -> TableReader<'table, K, V> {
        let layout = entry_layout::<K, V>(metadata);
        debug_assert_eq!(layout.stride, self.entry_stride);
        let entries_per_slab = BUFFER_SIZE / layout.stride;
        unsafe {
            TableReader::new(
                layout,
                entries_per_slab,
                self.slot_mask,
                self.slot_shift,
                self.hash_left_shift,
                &self.adjusted_slab_bases,
                metadata,
            )
        }
    }

    /// Creates a probing handle with a local layout snapshot.
    #[inline(always)]
    pub fn prober(&mut self) -> Prober<'_, K, V> {
        self.prober_with_metadata(self.storage_metadata)
    }

    /// Creates a probing handle for an arity-specialized loop.
    #[inline(always)]
    pub fn prober_with_metadata(&mut self, metadata: V::StorageMetadata) -> Prober<'_, K, V> {
        let reader = unsafe { self.reader_with_unchecked_lifetime(metadata) };
        Prober {
            table: self,
            reader,
        }
    }

    /// Rehash all entries into fresh zeroed slabs of `new_size` slots.
    ///
    /// Every occupied entry is re-probed into the larger table and copied whole
    /// (`stride` bytes: hash, key, and stored value alike).
    ///
    /// The collision counter is reset to `self.length` so that post-resize
    /// collision tracking reflects only new probing pressure, not historical
    /// collisions from the smaller table.
    ///
    /// # Panics
    ///
    /// The underlying allocation may panic if memory allocation fails.
    #[cold]
    pub fn resize(&mut self, allocator: &mut SlabAllocator, new_size: usize) {
        debug!("Resizing map to {:?}...", new_size);
        let new_slabs =
            allocator.create_strided_slabs(new_size, self.entry_stride, self.entry_alignment);
        // Keep the replaced slabs alive until the copy loop below has read
        // every old entry; `old_adjusted_slab_bases` points into them.
        let _old_slabs = mem::replace(&mut self.slabs, new_slabs);
        let old_adjusted_slab_bases = mem::replace(
            &mut self.adjusted_slab_bases,
            adjusted_bases(&self.slabs, self.entries_per_slab, self.entry_stride),
        );
        let old_slot_mask = self.slot_mask;
        self.collisions = self.length;
        self.slot_mask = new_size - 1;
        self.slot_shift = u64::BITS - new_size.trailing_zeros();

        for idx in 0..=old_slot_mask {
            let old_entry = self.entry_ptr_in(&old_adjusted_slab_bases, idx);
            let hash = unsafe { *(old_entry.add(self.hash_offset) as *const u64) };
            if hash != 0 {
                let mut new_idx = self.slot_for(hash);
                loop {
                    let new_entry = self.entry_ptr(new_idx);
                    if unsafe { *(new_entry.add(self.hash_offset) as *const u64) } == 0 {
                        unsafe {
                            ptr::copy_nonoverlapping(old_entry, new_entry, self.entry_stride)
                        };
                        break;
                    }
                    new_idx = (new_idx + 1) & self.slot_mask;
                }
            }
        }

        self.max_load = max_load_for_len(new_size);
    }
}

/// Mutable table handle that keeps the probe reader local across many probes.
pub struct Prober<'t, K: PersistedKey, V: AggregationValue + ?Sized> {
    table: &'t mut HashTable<K, V>,
    reader: TableReader<'t, K, V>,
}

impl<K: PersistedKey, V: AggregationValue + ?Sized> Prober<'_, K, V> {
    /// Prefetches the initial slot and two following cache lines into L1.
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let ptr = self.reader.entry_ptr(self.reader.slot_for(hash)) as *const u8;
        prefetch_l1_line(ptr);
        prefetch_l1_line(ptr.wrapping_add(64));
        prefetch_l1_line(ptr.wrapping_add(128));
    }

    /// Prefetches the initial slot into L2 for a longer lookahead.
    #[inline(always)]
    pub fn prefetch_l2(&self, hash: u64) {
        prefetch_l2_line(self.reader.entry_ptr(self.reader.slot_for(hash)) as *const u8);
    }

    /// See [`HashTable::undersized`].
    #[inline(always)]
    pub fn undersized(&self) -> bool {
        self.table.undersized()
    }

    /// Inserts or merges a stored partial value from another table.
    #[inline(always)]
    pub fn merge_from<const COUNT_COLLISIONS: bool, L>(
        &mut self,
        hash: u64,
        key: L,
        source: &V,
        ctx: &V::SharedContext,
    ) where
        L: LiveKey<Persisted = K>,
    {
        self.probe_fold::<COUNT_COLLISIONS, L, &V, _, _>(
            hash,
            key,
            source,
            |source, stored| stored.copy_from(source),
            |source, stored| stored.merge_from(source, ctx),
        );
    }

    /// Grows the table fourfold after it crosses the load threshold.
    #[inline(always)]
    pub fn grow_if_full(&mut self, allocator: &mut SlabAllocator, capacity: &mut usize) {
        if self.table.undersized() {
            *capacity *= 4;
            self.table.resize(allocator, *capacity);
            // Preserve constant metadata across the uncommon resize branch.
            self.reader = unsafe {
                self.table
                    .reader_with_unchecked_lifetime(self.reader.storage_metadata)
            };
        }
    }

    /// Doubles the table when collision pressure exceeds the given ratio.
    #[inline(always)]
    pub fn resize_on_collisions(&mut self, allocator: &mut SlabAllocator, collision_ratio: usize) {
        if self.table.collisions() > self.table.len() * collision_ratio {
            let new_size = self.table.capacity() << 1;
            self.table.resize(allocator, new_size);
            // See `grow_if_full` for why this uses the prober's metadata.
            self.reader = unsafe {
                self.table
                    .reader_with_unchecked_lifetime(self.reader.storage_metadata)
            };
        }
    }

    /// Probes for a key, then seeds an empty slot or updates the matching slot.
    ///
    /// `context` is moved into whichever closure runs. This avoids a second
    /// lookup and lets update paths, such as string MIN and MAX, skip work when
    /// the new row does not change the group.
    ///
    /// Hash zero is remapped to one because zero marks an empty slot. With
    /// `COUNT_COLLISIONS`, each occupied non-match increments the table's
    /// collision counter.
    #[inline(always)]
    pub fn probe_fold<const COUNT_COLLISIONS: bool, L, X, S, U>(
        &mut self,
        mut hash: u64,
        key: L,
        context: X,
        seed: S,
        update: U,
    ) where
        L: LiveKey<Persisted = K>,
        S: FnOnce(X, &mut V),
        U: FnOnce(X, &mut V),
    {
        let reader = self.reader;
        // hash == 0 is our empty sentinel, so remap actual zero hashes to 1.
        if hash == 0 {
            hash = 1;
        }
        let mut idx = reader.slot_for(hash);
        loop {
            let entry = reader.entry_ptr(idx);
            unsafe {
                let hash_ptr = entry.add(reader.hash_offset) as *mut u64;
                if *hash_ptr == 0 {
                    *hash_ptr = hash;
                    (entry.add(reader.key_offset) as *mut K).write(key.persist());
                    self.table.length += 1;
                    seed(
                        context,
                        V::from_entry_mut(entry.add(reader.value_offset), reader.storage_metadata),
                    );
                    return;
                }
                if *hash_ptr == hash
                    && key.eq_persisted(&*(entry.add(reader.key_offset) as *const K))
                {
                    update(
                        context,
                        V::from_entry_mut(entry.add(reader.value_offset), reader.storage_metadata),
                    );
                    return;
                }
            }
            if COUNT_COLLISIONS {
                self.table.collisions += 1;
            }
            idx = (idx + 1) & reader.slot_mask;
        }
    }
}

/// An iterator over the non-empty entries in a `BaseHashTable`.
///
/// Created by [`HashTable::iter`]. Yields [`EntryView`]s of entries where
/// `hash != 0` in arbitrary order (based on slot positions, not insertion order).
pub struct HashTableIterator<'a, K: PersistedKey, V: AggregationValue + ?Sized> {
    reader: TableReader<'a, K, V>,
    idx: usize,
}

impl<'a, K: PersistedKey, V: AggregationValue + ?Sized> Iterator for HashTableIterator<'a, K, V> {
    type Item = EntryView<'a, K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let reader = self.reader;
        while self.idx < reader.slot_mask + 1 {
            // One address computation per slot: the hash check and the yielded
            // view read the same entry pointer.
            let entry = reader.entry_ptr(self.idx);
            self.idx += 1;
            unsafe {
                let hash = *(entry.add(reader.hash_offset) as *const u64);
                if hash != 0 {
                    return Some(reader.view(entry, hash));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::values::{
        AggregationSlot, AggregationValue, ArityBody, ValueColumnBuilder,
    };
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::Field;

    /// A minimal additive value for exercising the probe mechanics: merging two
    /// of these sums their counts.
    #[derive(Copy, Clone, Default, Debug, PartialEq)]
    struct Count(usize);

    impl AggregationValue for Count {
        type Owned = Self;
        type StorageMetadata = ();
        type Reader<'b> = ();
        type SharedContext = ();
        type ColumnBuilder = CountColumnBuilder;
        type SortKey = i64;
        type WorkerContext = ();
        fn make_reader(_batch: &RecordBatch, _slots: &[AggregationSlot]) {}
        fn storage_metadata(_ctx: &()) {}
        fn metadata_for_arity<const N: usize>() {}
        fn dispatch_arity<Ret>(_metadata: (), body: impl ArityBody<Ret>) -> Ret {
            body.run::<0>()
        }
        fn stored_size(_metadata: ()) -> usize {
            size_of::<Self>()
        }
        fn stored_align() -> usize {
            align_of::<Self>()
        }
        unsafe fn from_entry<'a>(ptr: *const u8, _metadata: ()) -> &'a Self {
            unsafe { &*(ptr as *const Self) }
        }
        unsafe fn from_entry_mut<'a>(ptr: *mut u8, _metadata: ()) -> &'a mut Self {
            unsafe { &mut *(ptr as *mut Self) }
        }
        fn seed(&mut self, _reader: &(), _idx: usize, _wc: &mut ()) {
            *self = Count(1);
        }
        fn update(&mut self, _reader: &(), _idx: usize, _wc: &mut (), _ctx: &()) {
            self.0 += 1;
        }
        fn merge_from(&mut self, source: &Self, _ctx: &()) {
            self.0 += source.0;
        }
        fn copy_from(&mut self, source: &Self) {
            *self = *source;
        }
        fn sort_key(&self, _slot: usize) -> i64 {
            self.0 as i64
        }
        fn to_owned(&self, _ctx: &(), _wc: &mut Option<()>) -> Self {
            *self
        }
    }

    /// Empty output columns for the test [`Count`] value (the probe tests never
    /// materialise output).
    struct CountColumnBuilder;

    impl ValueColumnBuilder for CountColumnBuilder {
        type Value = Count;
        type Context = ();
        fn with_capacity(_allocator: &mut SlabAllocator, _rows: usize, _context: &()) -> Self {
            CountColumnBuilder
        }
        fn push(&mut self, _value: &Count) {}
        fn push_stored(&mut self, _stored: &Count) {}
        fn finish(self, _context: &()) -> (Vec<Field>, Vec<ArrayRef>) {
            (Vec::new(), Vec::new())
        }
    }

    type TestTable = HashTable<u64, Count>;

    fn new_table(allocator: &mut SlabAllocator, capacity: usize) -> TestTable {
        HashTable::new(allocator, capacity, 0, &())
    }

    #[test]
    fn reciprocal_division_matches_hardware_division() {
        // Representative entries-per-slab divisors: non-divisors of 2MB (24B
        // and 176B entries), exact powers of two (8B/16B entries), and odd ones.
        let divisors = [2u64, 3, 24, 176, 11915, 87381, 131072, 262144];

        for d in divisors {
            let m = reciprocal(d);
            for n in (0..1usize << 26).step_by(65_537).chain([
                0,
                d as usize - 1,
                d as usize,
                d as usize + 1,
                (1 << 40) - 1,
                1 << 40,
            ]) {
                assert_eq!(fast_div(n, m), n / d as usize, "n={n} d={d}");
            }
        }
    }

    #[test]
    fn insert_single_entry() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<false, _>(42, 100u64, &Count(1), &());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.key, 100);
        assert_eq!(*entry.stored, Count(1));
    }

    #[test]
    fn merge_duplicate_keys_sums_values() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<false, _>(42, 100u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(42, 100u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(42, 100u64, &Count(1), &());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.stored, Count(3));
    }

    #[test]
    fn distinct_keys_same_hash_both_stored() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<false, _>(42, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(42, 2u64, &Count(1), &());

        assert_eq!(table.len(), 2);
        let keys: Vec<u64> = table.iter(0).map(|e| *e.key).collect();
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
    }

    #[test]
    fn hash_zero_is_remapped_and_retrievable() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<false, _>(0, 99u64, &Count(1), &());

        assert_eq!(table.len(), 1);
        let entry = table.iter(0).next().unwrap();
        assert_eq!(*entry.key, 99);
        assert_ne!(entry.hash, 0);
    }

    #[test]
    fn undersized_triggers_at_load_factor() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        let max_load = (16.0 * MAX_LOAD_FACTOR).round() as usize;

        for i in 0..max_load {
            table
                .prober()
                .merge_from::<false, _>(i as u64 + 1, i as u64, &Count(1), &());
            assert!(!table.undersized());
        }

        table
            .prober()
            .merge_from::<false, _>(max_load as u64 + 1, max_load as u64, &Count(1), &());

        assert!(table.undersized());
    }

    #[test]
    fn collision_counting() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<true, _>(42, 1u64, &Count(1), &());
        assert_eq!(table.collisions(), 0);

        table
            .prober()
            .merge_from::<true, _>(42, 2u64, &Count(1), &());
        assert_eq!(table.collisions(), 1);
    }

    #[test]
    fn collision_counting_disabled() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);

        table
            .prober()
            .merge_from::<false, _>(42, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(42, 2u64, &Count(1), &());

        assert_eq!(table.collisions(), 0);
    }

    #[test]
    fn iter_skips_empty_slots() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 128);

        table
            .prober()
            .merge_from::<false, _>(1, 10u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(2, 20u64, &Count(1), &());

        let entries: Vec<_> = table.iter(0).collect();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|e| e.hash != 0));
    }

    #[test]
    fn resize_preserves_all_entries() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        for i in 0..8u64 {
            table
                .prober()
                .merge_from::<false, _>(i + 1, i, &Count(1), &());
        }

        table.resize(&mut allocator, 32);

        assert_eq!(table.len(), 8);
        assert_eq!(table.capacity(), 32);
        for i in 0..8u64 {
            let found = table.iter(0).any(|e| *e.key == i);
            assert!(found, "key {} missing after resize", i);
        }
    }

    #[test]
    fn resize_resets_collision_counter() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        table
            .prober()
            .merge_from::<true, _>(42, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<true, _>(42, 2u64, &Count(1), &());
        let pre_resize_collisions = table.collisions();
        assert!(pre_resize_collisions > 0);

        table.resize(&mut allocator, 32);

        assert_eq!(table.collisions(), table.len());
    }

    #[test]
    fn many_entries_all_retrievable() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 256);

        for i in 0..100u64 {
            table
                .prober()
                .merge_from::<false, _>(i + 1, i, &Count(1), &());
        }

        assert_eq!(table.len(), 100);
        for i in 0..100u64 {
            let found = table.iter(0).any(|e| *e.key == i && e.stored.0 == 1);
            assert!(found, "key {} missing or wrong value", i);
        }
    }

    #[test]
    fn wrap_around_probing() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        let last_slot_hash = u64::MAX;

        table
            .prober()
            .merge_from::<false, _>(last_slot_hash, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(last_slot_hash, 2u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(last_slot_hash, 3u64, &Count(1), &());

        assert_eq!(table.len(), 3);
        let keys: Vec<u64> = table.iter(0).map(|e| *e.key).collect();
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn empty_table_iteration() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let table = new_table(&mut allocator, 16);

        let entries: Vec<_> = table.iter(0).collect();

        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn merge_after_resize() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        table
            .prober()
            .merge_from::<false, _>(42, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(99, 2u64, &Count(1), &());

        table.resize(&mut allocator, 32);
        table
            .prober()
            .merge_from::<false, _>(42, 1u64, &Count(1), &());
        table
            .prober()
            .merge_from::<false, _>(200, 3u64, &Count(1), &());

        assert_eq!(table.len(), 3);
        let merged = table.iter(0).find(|e| *e.key == 1).unwrap();
        assert_eq!(*merged.stored, Count(2));
        assert!(table.iter(0).any(|e| *e.key == 3));
    }

    #[test]
    fn view_at_returns_correct_slot() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 16);
        table
            .prober()
            .merge_from::<false, _>(42, 100u64, &Count(1), &());

        let reader = table.reader::<0>();
        let occupied: Vec<usize> = (0..table.capacity())
            .filter(|&i| reader.hash_at(i) != 0)
            .collect();

        assert_eq!(occupied.len(), 1);
        assert_eq!(*reader.view_at(occupied[0]).key, 100);
    }

    #[test]
    fn iter_with_start_offset() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(true);
        let mut table = new_table(&mut allocator, 128);
        for i in 0..20u64 {
            table
                .prober()
                .merge_from::<false, _>(i + 1, i, &Count(1), &());
        }

        let all_count = table.iter(0).count();
        let from_offset_count = table.iter(10).count();

        assert!(from_offset_count < all_count);
        assert_eq!(all_count, 20);
    }
}
