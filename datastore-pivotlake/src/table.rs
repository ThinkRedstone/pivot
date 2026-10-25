//! The catalog's in-memory projection of one Delta table snapshot.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Error;
use crate::manifest::{
    ColumnStatFilter, DeltaFileEntry, FileStats, PartitionEqFilter, PartitionValues, scalar_equal,
};
use arrow_array::{Array, ArrayRef, Datum, Scalar};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::DataType;
use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use crossbeam_deque::{Injector, Steal};
use dispatch::{DataFlowDispatcher, Projection};
use object_storage::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use parquet_engine::{ParquetTable, RowGroupMetadata};
use planner::catalog::{Column, SchemaQualifiedTableName};

/// One data file of a table, materialized: its Delta log entry
/// ([`DeltaFileEntry`] -- identity, partition tuple, stats) paired with the row
/// groups read from its footer (file-local order). This is the single per-file
/// record the catalog holds: the entry is the metadata that commits to the log,
/// the row groups the content that scans.
#[derive(Clone)]
pub struct TableFile {
    pub(super) entry: DeltaFileEntry,
    pub(super) row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl TableFile {
    /// Pair a file's Delta log entry with the row groups read from its footer.
    /// The footer fetcher builds one with a bare [`DeltaFileEntry::new`] (identity
    /// only); the catalog then joins in the log entry's partition/stats.
    ///
    /// Derive the file's stats from its footers when the entry carries none, so
    /// file-level range pruning always has min/max bounds. Usually the entry
    /// already has them -- a fresh INSERT records them, and a reload reads the
    /// log's persisted stats -- so this covers only a file with neither: one
    /// adopted at CREATE, or whose log entry recorded no stats.
    pub(crate) fn new(mut entry: DeltaFileEntry, row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        if entry.stats.is_none() && !row_groups.is_empty() {
            entry.stats = Some(Arc::new(parquet_engine::aggregate_file_stats(&row_groups)));
        }
        Self { entry, row_groups }
    }

    /// The file's store identity (path and size), for a caller that names the
    /// file outside the catalog module (reporting a merge's outputs, deleting
    /// one whose commit failed).
    pub(crate) fn file_ref(&self) -> &FileRef {
        &self.entry.file
    }

    /// What this file's bytes hold decoded, summed over its column chunks.
    fn uncompressed_size(&self) -> u64 {
        self.row_groups
            .iter()
            .flat_map(|row_group| row_group.columns.iter())
            .map(|chunk| chunk.total_uncompressed_size.max(0) as u64)
            .sum()
    }
}

/// The catalog's master record of one table: its declared schema, partitioning,
/// version, and active `TableFile`s, each pairing a Delta log entry with its
/// materialized row groups.
///
/// This is a cloneable snapshot value. Copies handed out are read views and may
/// drift as commits land; [`refresh`](Self::refresh) reconciles one to the latest
/// version. Writers instead go through
/// `PivotlakeDatastore::commit_to_table`, following the table's commit-lock
/// protocol. `store` and `location`
/// let any copy persist and reload itself.
#[derive(Clone)]
pub struct CatalogTable {
    /// The table's durable identity from the pivotlake manifest, minted once at
    /// creation and stable across renames and every commit. The catalog indexes
    /// by this; Delta Kernel independently owns the log's `metaData.id`.
    id: uuid::Uuid,
    /// Where the table's Parquet data lives in the object store.
    location: ObjectPath,
    /// The Delta snapshot this copy is at: the version its `files` were built
    /// from, and the base every commit from this copy is written on top of. It is
    /// this copy's alone: a refresh advances it, a commit replaces it with the one
    /// the commit produced, and neither reads a log another copy is holding.
    snapshot: Arc<delta_kernel::Snapshot>,
    /// The table's declared columns (schema).
    columns: Vec<Column>,
    /// Partition columns, in order; empty = unpartitioned.
    partition_by: Vec<String>,
    /// Sort columns, in order; empty = unsorted.
    sort_by: Vec<String>,
    /// The active files: each a log entry (identity + partition + stats) paired
    /// with its materialized footers.
    pub(super) files: Vec<TableFile>,
    /// The files the log retired but still remembers, each by the Unix ms its
    /// commit retired it: what vacuum ages a retired file by, so the retention
    /// window runs from the moment the file stopped being referenced rather
    /// than from its storage mtime. Rebuilt from the log alongside `files`,
    /// which bounds it to what the log still carries, and extended by every
    /// commit this copy makes, which drops the entries a window older than
    /// itself.
    tombstones: HashMap<ObjectPath, u64>,
    store: Arc<dyn ObjectStore>,
    /// The pool a reload/commit fetches footers on, so the mutators need no
    /// dispatcher passed in.
    dispatcher: DataFlowDispatcher,
    /// The Kernel engine this table's log is read and written through, shared
    /// with every other table of the datastore that opened it.
    engine: crate::log::DeltaEngine,
    /// This exists only for throughput: Delta log compare-and-swap already ensures
    /// correctness, but concurrent in-process writers can all start at `V` and race
    /// for `V + 1`, forcing every loser to replay the log and fetch the winner's
    /// footers before retrying.
    commit_lock: Arc<std::sync::Mutex<()>>,
}

impl CatalogTable {
    /// Reassemble a persisted table from its table-level metadata and the per-file
    /// [`TableFile`]s (log entry + footers) just built for it — the reopen path.
    /// Does not persist anything; the log it was loaded from is already durable.
    #[allow(clippy::too_many_arguments)] // an internal constructor; each field is needed
    pub(super) fn new(
        id: uuid::Uuid,
        location: ObjectPath,
        snapshot: Arc<delta_kernel::Snapshot>,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        files: Vec<TableFile>,
        tombstones: HashMap<ObjectPath, u64>,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
        engine: crate::log::DeltaEngine,
    ) -> Self {
        Self {
            id,
            location,
            snapshot,
            columns,
            partition_by,
            sort_by,
            files,
            tombstones,
            store,
            dispatcher,
            engine,
            commit_lock: Arc::default(),
        }
    }

    /// Create a brand-new table from freshly-read footers by atomically writing
    /// Delta version 0 with its protocol, metadata, and initial `Add` actions.
    /// The `files` arrive with bare entries (identity only); files discovered at
    /// CREATE carry no partition metadata, and a partitioned INSERT records it on
    /// the files it writes later.
    #[allow(clippy::too_many_arguments)] // an internal constructor; each field is needed
    pub(super) fn create_new(
        id: uuid::Uuid,
        location: ObjectPath,
        loaded: Vec<parquet_engine::FileRowGroups>,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        store: Arc<dyn ObjectStore>,
        dispatcher: DataFlowDispatcher,
        engine: crate::log::DeltaEngine,
    ) -> crate::Result<Self> {
        // Pair each discovered file with a bare log entry; `TableFile::new`
        // derives its stats from the footers.
        let mut files: Vec<TableFile> = loaded
            .into_iter()
            .map(|file| TableFile::new(DeltaFileEntry::new(file.file), file.row_groups))
            .collect();
        // For a partitioned table, a discovered file whose partition columns are
        // constant belongs to that partition; stamp it from the file's stats so it
        // is recorded in the log and prunes like a written file.
        if !partition_by.is_empty() {
            for file in &mut files {
                let partition = file
                    .entry
                    .stats
                    .as_ref()
                    .and_then(|stats| constant_partition(stats, &partition_by));
                file.entry.partition = partition;
            }
        }
        let entries: Vec<DeltaFileEntry> = files.iter().map(|file| file.entry.clone()).collect();
        // Version 0's commit yields the snapshot the new table starts at, so the
        // log it just wrote is not read back to open it.
        let snapshot = crate::log::initialize_table(
            &engine,
            store.as_ref(),
            &location,
            &columns,
            &partition_by,
            &sort_by,
            &entries,
        )?;
        Ok(Self {
            id,
            location,
            snapshot,
            columns,
            partition_by,
            sort_by,
            files,
            tombstones: HashMap::new(),
            store,
            dispatcher,
            engine,
            commit_lock: Arc::default(),
        })
    }

    /// An owned handle to this table's
    /// [commit lock](field@Self::commit_lock).
    pub(super) fn commit_lock(&self) -> Arc<std::sync::Mutex<()>> {
        self.commit_lock.clone()
    }

    /// Advance this copy to the latest committed version, reconciling its files
    /// to it. Returns whether it advanced; `Ok(false)` means this copy was
    /// already current, which costs one log listing and no file reads (Kernel
    /// updates the snapshot in hand rather than rebuilding it).
    pub fn refresh(&mut self) -> crate::Result<bool> {
        let Some(state) = crate::log::refresh_table(&self.snapshot, &self.engine)? else {
            return Ok(false);
        };
        let crate::log::DeltaTableState {
            snapshot,
            columns,
            partition_by,
            sort_by,
            file_entries,
            tombstones,
        } = state;
        // Reconcile the files first: it is the only step that can fail, and it
        // leaves this copy untouched when it does, so a copy never ends up at a
        // version whose files it does not hold.
        self.rebuild_files(file_entries, &columns)?;
        self.snapshot = snapshot;
        self.columns = columns;
        self.partition_by = partition_by;
        self.sort_by = sort_by;
        self.tombstones = tombstones;
        Ok(true)
    }

    /// Fold a file swap this process already committed through the datastore
    /// into this read copy without a log read: drop `removed` and adopt
    /// `added`, whose footers the writer holds. The copy's snapshot stays at
    /// its version; a later [`refresh`](Self::refresh) reconciles it to the
    /// log and keeps the adopted footers.
    pub(crate) fn apply_file_swap(&mut self, removed: &[ObjectPath], added: Vec<TableFile>) {
        self.files
            .retain(|file| !removed.contains(&file.entry.file.path));
        self.files.extend(added);
    }

    /// Commit file changes against this copy's snapshot. Each attempt first
    /// verifies that every `removed` path is still active, returning
    /// [`Error::CommitConflict`] if another commit removed one. After a successful
    /// log write, the already-loaded `added` files fold directly into this copy.
    /// A version conflict refreshes the copy and retries.
    ///
    /// In-process callers follow the table's
    /// [commit-lock protocol](field@Self::commit_lock), so this retry path
    /// normally handles contention from another process on a shared remote
    /// store.
    ///
    /// Runs off the dispatch workers: the retry's `refresh` drives a footer-fetch
    /// dataflow, and every commit path commits from a blocking thread, never a
    /// pinned worker (which would deadlock driving a dataflow from inside one).
    pub(crate) fn commit_files(
        &mut self,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
        data_change: bool,
    ) -> crate::Result<()> {
        loop {
            if let Some(missing) = removed.iter().find(|path| {
                !self
                    .files
                    .iter()
                    .any(|f| f.entry.file.path.as_str() == path.as_str())
            }) {
                return Err(Error::CommitConflict {
                    location: self.location.as_str().to_string(),
                    file: missing.to_string(),
                });
            }
            let removed_entries: Vec<DeltaFileEntry> = removed
                .iter()
                .map(|path| {
                    self.files
                        .iter()
                        .find(|file| file.entry.file.path.as_str() == path.as_str())
                        .expect("removed paths were validated above")
                        .entry
                        .clone()
                })
                .collect();
            let entries: Vec<DeltaFileEntry> = added.iter().map(|f| f.entry.clone()).collect();
            let retired_at_ms = crate::vacuum::now_unix_ms();
            if let Some(committed) = crate::log::commit_file_changes(
                &self.engine,
                &self.snapshot,
                &removed_entries,
                &entries,
                data_change,
            )? {
                self.snapshot = committed;
                self.files.retain(|f| !removed.contains(&f.entry.file.path));
                self.files.extend(added);
                self.record_retirements(removed, retired_at_ms);
                return Ok(());
            }
            self.refresh()?;
        }
    }

    /// Fold the files a commit from this copy retired into its tombstones, dated
    /// `retired_at_ms`: a clock reading taken just before the transaction was
    /// opened, back to back with the one Kernel stamps their `Remove` tombstones
    /// with, so the copy dates them within milliseconds of what a reload from
    /// the log would. The same reading retires the tombstones a full retention
    /// window older than this commit: any sweep run after it sees them past the
    /// window, so they date nothing, and dropping them is what bounds a writer's
    /// copy, which only ever extends its tombstones between reloads, to one
    /// window's worth of retirements.
    fn record_retirements(&mut self, removed: &[ObjectPath], retired_at_ms: u64) {
        if removed.is_empty() {
            return;
        }
        let cutoff = retired_at_ms.saturating_sub(self.deleted_file_retention().as_millis() as u64);
        self.tombstones.retain(|_, at| *at > cutoff);
        self.tombstones
            .extend(removed.iter().map(|path| (path.clone(), retired_at_ms)));
    }

    /// Commit bare log `entries` whose footers are not yet in hand: like
    /// [`commit_files`](Self::commit_files), but the added files' footers are read
    /// only if the commit wins (a losing swap errors before touching its output).
    /// Used by test/seed paths; production writers (INSERT, compaction) already
    /// hold the footers and use `commit_files`. Only its `test-support` caller
    /// ([`test_support`](crate::test_support)) needs it, so it is gated with that
    /// feature to stay out of a plain library build.
    #[cfg(feature = "test-support")]
    pub(crate) fn commit_entries(
        &mut self,
        removed: &[ObjectPath],
        added: &[DeltaFileEntry],
        data_change: bool,
    ) -> crate::Result<()> {
        loop {
            if let Some(missing) = removed.iter().find(|path| {
                !self
                    .files
                    .iter()
                    .any(|f| f.entry.file.path.as_str() == path.as_str())
            }) {
                return Err(Error::CommitConflict {
                    location: self.location.as_str().to_string(),
                    file: missing.to_string(),
                });
            }
            let removed_entries: Vec<DeltaFileEntry> = removed
                .iter()
                .map(|path| {
                    self.files
                        .iter()
                        .find(|file| file.entry.file.path.as_str() == path.as_str())
                        .expect("removed paths were validated above")
                        .entry
                        .clone()
                })
                .collect();
            let retired_at_ms = crate::vacuum::now_unix_ms();
            if let Some(committed) = crate::log::commit_file_changes(
                &self.engine,
                &self.snapshot,
                &removed_entries,
                added,
                data_change,
            )? {
                // We won; only now read the added files' footers. They are built
                // whole before this copy moves onto the committed version, so a
                // failed read leaves it where it was rather than at a version
                // whose files it does not hold.
                let footers = self.fetch_footers(added, &self.columns)?;
                let committed_files: Vec<TableFile> = added
                    .iter()
                    .zip(footers)
                    .map(|(entry, footer)| TableFile::new(entry.clone(), footer.row_groups))
                    .collect();
                self.snapshot = committed;
                self.files.retain(|f| !removed.contains(&f.entry.file.path));
                self.files.extend(committed_files);
                self.record_retirements(removed, retired_at_ms);
                return Ok(());
            }
            self.refresh()?;
        }
    }

    /// Reconcile `self.files` to the log `entries` of a version whose schema is
    /// `columns`: keep the footers this copy already read (carrying forward their
    /// sort bounds, which the log does not persist), fetch the footers for files
    /// not yet held, and drop files no longer present.
    ///
    /// The new file list is built whole before `self.files` is touched, so a
    /// footer fetch that fails leaves this copy exactly as it was. A copy left
    /// holding a version's files only partially would never recover: its next
    /// refresh finds that version already current and reconciles nothing.
    fn rebuild_files(
        &mut self,
        entries: Vec<DeltaFileEntry>,
        columns: &[Column],
    ) -> crate::Result<()> {
        let held: HashMap<&ObjectPath, &TableFile> = self
            .files
            .iter()
            .map(|file| (&file.entry.file.path, file))
            .collect();
        let to_fetch: Vec<DeltaFileEntry> = entries
            .iter()
            .filter(|entry| !held.contains_key(&entry.file.path))
            .cloned()
            .collect();
        // One footer per file not held, in the order `entries` names them.
        let mut fetched = self.fetch_footers(&to_fetch, columns)?.into_iter();

        let mut files = Vec::with_capacity(entries.len());
        for entry in entries {
            let row_groups = match held.get(&entry.file.path) {
                Some(held_file) => held_file.row_groups.clone(),
                None => {
                    fetched
                        .next()
                        .expect("a footer is fetched for every file not held")
                        .row_groups
                }
            };
            files.push(TableFile::new(entry, row_groups));
        }
        self.files = files;
        Ok(())
    }

    /// Fetch the footers for `entries` — the row groups of the files this copy
    /// does not yet hold, one per entry in the order of `entries`.
    /// Each file's parsed schema is reconciled against `columns`, the schema
    /// declared by the version the entries come from.
    fn fetch_footers(
        &self,
        entries: &[DeltaFileEntry],
        columns: &[Column],
    ) -> crate::Result<Vec<parquet_engine::FileRowGroups>> {
        let to_fetch: Vec<DataFile> = entries
            .iter()
            .map(|e| {
                e.file
                    .clone()
                    .into_data_file(self.store.as_ref(), &self.location)
            })
            .collect::<object_storage::Result<_>>()?;
        Ok(parquet_engine::load_file_row_groups(
            &self.dispatcher,
            &to_fetch,
            parquet_engine::TableColumns::by_name(columns.to_vec()),
        )?)
    }

    pub fn files(&self) -> &[TableFile] {
        &self.files
    }

    /// The table's durable identity from the pivotlake manifest, stable across
    /// renames and commits. The catalog indexes by this.
    pub fn id(&self) -> uuid::Uuid {
        self.id
    }

    /// The table's partition columns, in order (empty = unpartitioned). A
    /// partitioning writer routes each row to a file by these columns' values.
    pub fn partition_by(&self) -> &[String] {
        &self.partition_by
    }

    /// The table's sort columns, in order (empty = unsorted). A writer sorts each
    /// file's rows by these before encoding.
    pub fn sort_by(&self) -> &[String] {
        &self.sort_by
    }

    /// The table's current files as [`FileRef`]s — what a compacter scans to pick
    /// merge candidates, and names in a compaction swap.
    pub fn file_refs(&self) -> Vec<FileRef> {
        self.files.iter().map(|f| f.entry.file.clone()).collect()
    }

    /// The current committed file entries, including the partition tuple and
    /// file-level statistics used to choose compaction work.
    pub(crate) fn file_entries(&self) -> impl Iterator<Item = &DeltaFileEntry> {
        self.files.iter().map(|file| &file.entry)
    }

    /// This table as the cross-datastore catalog describes it, under the `name`
    /// it is currently indexed by: the columns it declares with what they cost,
    /// and the files it holds.
    pub(crate) fn metadata(&self, name: SchemaQualifiedTableName) -> DatastoreTableMetadata {
        let column_bytes = self.column_bytes();
        let columns = self
            .columns
            .iter()
            .enumerate()
            .map(|(position, column)| {
                let (bytes, bytes_uncompressed) = column_bytes[position];
                DatastoreColumnMetadata {
                    name: column.name.clone(),
                    column_type: column.col_type.clone(),
                    position,
                    bytes,
                    bytes_uncompressed,
                    is_partition_key: self.partition_by.contains(&column.name),
                    is_sort_key: self.sort_by.contains(&column.name),
                }
            })
            .collect();

        let files = self
            .files
            .iter()
            .map(|file| DatastoreFileMetadata {
                path: file.entry.file.path.as_str().to_string(),
                bytes: file.entry.file.size,
                bytes_uncompressed: file.uncompressed_size(),
                partition: self.format_partition(&file.entry.partition),
                min_max_stats: format_min_max_stats(file.entry.stats.as_deref()),
            })
            .collect();

        DatastoreTableMetadata {
            name,
            id: self.id.to_string(),
            columns,
            sort_by: self.sort_by.clone(),
            partition_by: self.partition_by.clone(),
            total_rows: self
                .files
                .iter()
                .flat_map(|file| file.row_groups.iter())
                .map(|row_group| row_group.num_rows.max(0) as u64)
                .sum(),
            bytes: self.files.iter().map(|file| file.entry.file.size).sum(),
            bytes_uncompressed: self.files.iter().map(TableFile::uncompressed_size).sum(),
            files,
        }
    }

    /// What each declared column costs across the table's committed files: the
    /// bytes it occupies in storage, and what they hold decoded. Chunks are per
    /// leaf, so a nested column is charged for every leaf it spans; a file that
    /// does not carry a column contributes nothing to it.
    fn column_bytes(&self) -> Vec<(u64, u64)> {
        let position_by_name: HashMap<&str, usize> = self
            .columns
            .iter()
            .enumerate()
            .map(|(position, column)| (column.name.as_str(), position))
            .collect();

        let mut totals = vec![(0, 0); self.columns.len()];
        for row_group in self.files.iter().flat_map(|file| file.row_groups.iter()) {
            let mut leaf = 0;
            for field in row_group.schema.fields() {
                let leaves = parquet_engine::leaf_count(field);
                if let Some(&position) = position_by_name.get(field.name().as_str()) {
                    for chunk in &row_group.columns[leaf..leaf + leaves] {
                        totals[position].0 += chunk.total_compressed_size.max(0) as u64;
                        totals[position].1 += chunk.total_uncompressed_size.max(0) as u64;
                    }
                }
                leaf += leaves;
            }
        }
        totals
    }

    /// A file's partition as the catalog reports it: `column=value` pairs in the
    /// table's partition order, comma-separated. Empty for an unpartitioned
    /// table, and for a file adopted before the table was partitioned, which
    /// carries no tuple (or none for some of the columns).
    fn format_partition(&self, partition: &Option<PartitionValues>) -> String {
        let Some(values) = partition else {
            return String::new();
        };
        self.partition_by
            .iter()
            .filter_map(|column| {
                let (array, _) = values.get(column)?.get();
                let formatter = ArrayFormatter::try_new(array, &FormatOptions::default()).ok()?;
                Some(format!("{column}={}", formatter.value(0)))
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Each committed file paired with the typed partition tuple recorded for
    /// it (or `None`), reflecting the current committed version.
    pub fn file_partitions(&self) -> Vec<(ObjectPath, Option<PartitionValues>)> {
        self.files
            .iter()
            .map(|f| (f.entry.file.path.clone(), f.entry.partition.clone()))
            .collect()
    }

    /// A scannable [`ParquetTable`] over just the `wanted` files (matched by
    /// path) — the compaction read view: feed it to
    /// [`table_input`](parquet_engine::table_input) to decode their rows.
    pub fn parquet_table_for(&self, wanted: &[FileRef]) -> Arc<ParquetTable> {
        let want: HashSet<&ObjectPath> = wanted.iter().map(|f| &f.path).collect();
        let row_groups = self
            .files
            .iter()
            .filter(|f| want.contains(&f.entry.file.path))
            .flat_map(|f| f.row_groups.iter().cloned())
            .collect();
        Arc::new(ParquetTable::new(row_groups))
    }

    /// Delete a data file (a compaction input swapped out of the manifest).
    /// `path` resolves against the table's location like any [`FileRef`] path,
    /// and must land inside it: that directory is the only storage the table
    /// owns.
    ///
    /// A file the table *adopted* (`with_pre_existing_parquets`) lives wherever its owner
    /// put it, so it is never the table's to delete: compaction may copy its rows
    /// into the table's own storage and drop it from the log, but the object
    /// itself stays. The test is where the file lands rather than how the path is
    /// spelled, so a path that names the table's own directory outright is still
    /// the table's to delete.
    pub fn delete_data_file(&self, path: &ObjectPath) -> crate::Result<()> {
        let resolved = self.location.resolve(path);
        if !self.is_in_own_storage(&resolved)? {
            return Err(crate::Error::DeletingOutsideStorage {
                location: self.location.as_str().to_string(),
                file: path.as_str().to_string(),
            });
        }
        self.store.delete(&resolved)?;
        Ok(())
    }

    /// Whether `key` names an object under the table's own location. Both sides
    /// are taken to their store-root-absolute form first, so a key that resolves
    /// into the table's directory counts however it was written.
    fn is_in_own_storage(&self, key: &ObjectPath) -> crate::Result<bool> {
        let owned = self.store.absolute_key(&self.location)?;
        let owned = format!("{}/", owned.as_str().trim_end_matches('/'));
        Ok(self
            .store
            .absolute_key(key)?
            .as_str()
            .starts_with(owned.as_str()))
    }

    /// How long a file this table no longer references is kept before vacuum may
    /// delete it: the table's `delta.deletedFileRetentionDuration`, or the
    /// datastore's 4-hour default when unset. Vacuum ages unreferenced files
    /// against this window.
    pub fn deleted_file_retention(&self) -> std::time::Duration {
        crate::log::deleted_file_retention(&self.snapshot)
    }

    /// The files this copy's version no longer references but the log retired
    /// inside the retention window, each by the Unix ms it was retired. Held
    /// with the version, so a caller wanting the latest refreshes first.
    pub fn tombstones(&self) -> &HashMap<ObjectPath, u64> {
        &self.tombstones
    }

    /// Delete the files in this table's `_delta_log` that a checkpoint has made
    /// redundant and are past the table's `delta.logRetentionDuration`
    /// (evaluated against `now_ms`). Writing a checkpoint stays inline on the
    /// commit path; this is the periodic cleanup the vacuum sweep drives.
    /// The `_delta_log` naming convention lives with the rest of the Delta-format
    /// code in the Delta-format module; returns how many files were deleted.
    pub fn cleanup_log(&self, now_ms: u64) -> crate::Result<usize> {
        Ok(crate::log::cleanup_log(
            self.store.as_ref(),
            &self.location,
            &self.snapshot,
            now_ms,
        )?)
    }

    /// The Parquet data objects physically present under the table location,
    /// each with its storage modification time (Unix ms) — what vacuum's orphan
    /// sweep lists to find files that no log action references. Data files are
    /// flat under the location (partition values live in the log, not the path),
    /// so a one-level listing is complete; non-`.parquet` objects (the
    /// `_delta_log`) are excluded. Paths are relative to the location, matching
    /// [`file_refs`](Self::file_refs) and what [`delete_data_file`](Self::delete_data_file)
    /// expects.
    pub fn list_data_files(&self) -> crate::Result<Vec<(ObjectPath, u64)>> {
        Ok(self
            .store
            .list(&self.location)?
            .objects
            .into_iter()
            .filter(|object| object.file.path.as_str().ends_with(".parquet"))
            .map(|object| (object.file.path, object.modified_unix_ms))
            .collect())
    }

    /// The Delta version this copy is at, read off the snapshot it holds.
    /// Monotonic per table; used to decide whether a published copy is newer than
    /// the catalog's.
    pub fn version(&self) -> u64 {
        self.snapshot.version()
    }

    /// A flat scan view of the files a query's pushed-down predicates cannot rule
    /// out — every surviving file's row groups concatenated in manifest order,
    /// where a row group's global index is simply its position. A file survives
    /// when its recorded partition tuple can still match every `partition_filters`
    /// *and* its Parquet stats can still match every `stat_filters`. Empty filters
    /// keep every file.
    ///
    /// Read-only: it is built entirely from the row groups this copy already
    /// holds, so it does **no** I/O. Every surviving file's footer must have
    /// been fetched (the refresh path keeps `files` synced to the manifest); a
    /// missing one is an error, never a silently narrower scan.
    ///
    /// The row group's global index is its position in this returned flat list, so
    /// a scan and its materialize must build it from the *same* filters (they do:
    /// both go through the binding's predicates) to address the same groups.
    pub fn build_scan_view(
        &self,
        partition_filters: &[PartitionEqFilter],
        stat_filters: &[ColumnStatFilter],
    ) -> crate::Result<Arc<ParquetTable>> {
        let mut row_groups = Vec::new();
        for file in self
            .files
            .iter()
            .filter(|f| {
                f.entry
                    .maybe_matches_partition(&self.partition_by, partition_filters)
            })
            .filter(|f| f.entry.maybe_matches_stats(stat_filters))
        {
            row_groups.extend(file.row_groups.iter().cloned());
        }
        Ok(Arc::new(ParquetTable::new(row_groups)))
    }

    /// The table's columns (schema), as the planner's [`Column`]s.
    pub fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    /// Whether each column (in [`columns`](Self::columns) order) can hold SQL
    /// NULLs, computed from the loaded footers rather than the declared schema:
    /// the planner uses the flags to route between the branch-free and the
    /// null-aware execution paths, so they must reflect whether the data can
    /// actually hold NULLs. An empty table reports every column nullable.
    pub fn nullability(&self) -> Vec<bool> {
        self.columns
            .iter()
            .map(|column| self.files.is_empty() || self.column_may_hold_nulls(&column.name))
            .collect()
    }

    /// Whether any loaded row group can hold a NULL in `name`; a file missing
    /// the column entirely reads as all-NULL.
    fn column_may_hold_nulls(&self, name: &str) -> bool {
        self.files
            .iter()
            .flat_map(|file| file.row_groups.iter())
            .any(|rg| match rg.schema.index_of(name) {
                Ok(field_idx) => rg.column_may_hold_nulls(field_idx),
                Err(_) => true,
            })
    }

    /// The declared schema as one shareable slice: what a footer load
    /// reconciles each file's parsed schema against.
    fn declared_columns(&self) -> Arc<[Column]> {
        self.columns.clone().into()
    }

    /// Where the table's data lives, relative to the database root (an absolute
    /// path escapes to the store root). Combine with the store's own root to
    /// know the physical location.
    pub fn location(&self) -> &str {
        self.location.as_str()
    }

    pub(crate) fn object_location(&self) -> &ObjectPath {
        &self.location
    }

    pub(crate) fn store(&self) -> Arc<dyn ObjectStore> {
        self.store.clone()
    }

    /// Re-encode `inputs`' rows into fresh target-sized files and return them,
    /// uploaded but not yet part of the table: the read half of a compaction,
    /// which [`compact_table_files`](crate::compact_table_files) then swaps in
    /// for the inputs. The merged files are written over the
    /// shared io_uring ring by the same upload operators an INSERT uses, and
    /// their row groups come straight from the writer's own footer metadata, so
    /// no footer is re-read.
    ///
    /// Nothing here touches the log, so it runs off any read copy that holds the
    /// inputs, and a merge of a large batch never stalls another writer's commit.
    /// `inputs` must share one partition tuple; the caller batches them so.
    pub(crate) fn merge_files(
        &self,
        inputs: &[FileRef],
        target_rows_per_group: usize,
        max_output_file_size: u64,
    ) -> crate::Result<Vec<TableFile>> {
        // The selector guarantees that all inputs share one partition, so its
        // recorded tuple can key every decoded batch without repartitioning it.
        let parquet = self.parquet_table_for(inputs);
        let columns = parquet.schema().fields().len();
        let scan = parquet_engine::table_input(
            &self.dispatcher,
            &parquet,
            Projection::all(columns),
            false,
        );
        let partition = inputs.first().and_then(|input| {
            self.files
                .iter()
                .find(|file| file.entry.file.path == input.path)
                .expect("a compaction input belongs to this table")
                .entry
                .partition
                .clone()
        });
        let max_output_file_size = usize::try_from(max_output_file_size).unwrap_or(usize::MAX);
        let uploaded_files = Arc::new(Injector::new());
        let encoded = parquet_engine::writing::encode_compaction_batches_spec(
            scan,
            parquet.schema().clone(),
            partition,
            Arc::from(self.partition_by()),
            Arc::from(self.sort_by()),
            target_rows_per_group,
            max_output_file_size,
        );
        let spec = super::insert_sink::upload_files_spec(
            self.store(),
            self.location.clone(),
            self.declared_columns(),
            uploaded_files.clone(),
            encoded,
        );
        // Drive encode → upload to completion; the emitted row-count batch is
        // ignored, and the uploaded files arrive on `uploaded_files`.
        let outcome = spec.collect().map(|_| ()).map_err(crate::Error::Merge);

        let mut added: Vec<TableFile> = Vec::new();
        loop {
            match uploaded_files.steal() {
                Steal::Success(uploaded) => {
                    let super::insert_sink::UploadedFile {
                        file,
                        partition,
                        row_groups,
                    } = uploaded;
                    let stats = Some(std::sync::Arc::new(parquet_engine::aggregate_file_stats(
                        &row_groups,
                    )));
                    let entry = DeltaFileEntry {
                        file,
                        partition,
                        stats,
                    };
                    added.push(TableFile::new(entry, row_groups));
                }
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }

        // A merge that failed part way has uploaded some of its outputs. They
        // will never be committed, so delete them rather than leave orphans.
        if let Err(error) = outcome {
            for file in &added {
                if let Err(cleanup_error) = self.delete_data_file(&file.entry.file.path) {
                    tracing::warn!(
                        merge_error = %error,
                        cleanup_error = %cleanup_error,
                        file = %file.entry.file.path,
                        "compaction: deleting an output of a merge that will not commit failed (orphan left)"
                    );
                }
            }
            return Err(error);
        }
        Ok(added)
    }
}

/// Render a file's typed column bounds as a stable JSON object. Each column is
/// one field containing its `min` and `max`; values stay JSON numbers/booleans
/// where their Arrow type has that shape, while strings and temporal values are
/// quoted. A column whose pair cannot be rendered is omitted.
fn format_min_max_stats(stats: Option<&FileStats>) -> String {
    let Some(stats) = stats else {
        return "{}".to_string();
    };
    let mut names: Vec<_> = stats.min_values.keys().collect();
    names.sort_unstable();

    let fields = names
        .into_iter()
        .filter_map(|name| {
            let min = format_bound(stats.min_values.get(name)?)?;
            let max = format_bound(stats.max_values.get(name)?)?;
            let name = serde_json::to_string(name).expect("a Rust string is valid JSON");
            Some(format!(r#"{name}:{{"min":{min},"max":{max}}}"#))
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{fields}}}")
}

/// Render the single value held by a file statistic as one JSON value.
fn format_bound(array: &ArrayRef) -> Option<String> {
    if array.len() != 1 || array.is_null(0) {
        return None;
    }
    let formatter = ArrayFormatter::try_new(array.as_ref(), &FormatOptions::default()).ok()?;
    let rendered = formatter.value(0).to_string();
    match array.data_type() {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _) => serde_json::from_str::<serde_json::Value>(&rendered)
            .ok()
            .map(|_| rendered),
        _ => serde_json::to_string(&rendered).ok(),
    }
}

/// The partition a file with these `stats` belongs to: each partition column
/// mapped to the value it holds when that value is constant across the file (its
/// min equals its max). Returns `None` if any partition column is absent from the
/// stats or varies within the file, so a file that does not sit in a single
/// partition is left unstamped rather than mislabeled.
fn constant_partition(stats: &FileStats, partition_by: &[String]) -> Option<PartitionValues> {
    let mut values = PartitionValues::new();
    for column in partition_by {
        let min = Scalar::new(ArrayRef::clone(stats.min_values.get(column)?));
        let max = Scalar::new(ArrayRef::clone(stats.max_values.get(column)?));
        if scalar_equal(&min, &max) != Some(true) {
            return None;
        }
        values.insert(column.clone(), min);
    }
    Some(values)
}
