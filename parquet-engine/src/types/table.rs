//! BoundTable above parquet files
//!
//! A [`ParquetTable`] is the engine's handle to one or more Parquet files that
//! together form a logical table. Construction reads and parses each file's
//! Thrift footer, converts the Parquet schema to Arrow, and collects
//! `RowGroupMetadata` entries with globally unique row-group indices.

use crate::thrift::footer::{FileMetaData, PageEncodingStats, Statistics};
use crate::thrift::general::{Encoding, PageType};
use crate::thrift::parquet_thrift::{ReadThrift, ThriftSliceInputProtocol};
use crate::types::columns::{ColumnResolution, TableColumns};
use crate::types::leaves::leaf_count;
use crate::types::metadata::{ColumnChunkMeta, FileLeafStatistics, RowGroupMetadata};
use arrow_array::builder::{BinaryViewBuilder, StringViewBuilder};
use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Decimal64Array, Decimal128Array, PrimitiveArray, Scalar,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use dispatch::DataFlowDispatcher;
use object_storage::DataFile;
use planner::catalog::Column;
use planner::types::Type;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, LazyLock};
use std::{fs, io};
use thiserror::Error;

static EMPTY_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| SchemaRef::new(Schema::empty()));

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    IO(#[from] std::io::Error),
    #[error("Column not found {0}")]
    ColumnNotFound(String),
    #[error("fetching row-group metadata: {0}")]
    Materialize(String),
    #[error("invalid parquet footer: {0}")]
    InvalidFooter(String),
    /// A Parquet/arrow type with no mapping in either direction (an arrow type
    /// the writer can't emit, or a Parquet leaf the reader can't decode).
    #[error("unsupported type: {0}")]
    UnsupportedType(String),
    /// A file's stored column type contradicts the table's declared schema.
    #[error("column '{column}' is declared {declared} but the file stores {file_type}")]
    DeclaredTypeMismatch {
        column: String,
        declared: Type,
        file_type: DataType,
    },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A logical table backed by one or more Parquet files.
///
/// Holds a flat, globally-indexed list of `RowGroupMetadata` entries spanning
/// every file. Cheaply shareable via `Arc` because the metadata is read-only
/// after construction.
#[derive(Clone)]
pub struct ParquetTable {
    pub(crate) row_groups: Vec<Arc<RowGroupMetadata>>,
    /// The table's Arrow schema, captured at construction so it survives even
    /// when every row group is pruned away (e.g. predicate-stats pruning removes
    /// all of them). Deriving it from `row_groups[0]` instead would yield an
    /// empty schema for a fully-pruned table, panicking any scan that projects a
    /// column.
    schema: SchemaRef,
}

impl Debug for ParquetTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("table")
    }
}

impl ParquetTable {
    /// Wraps pre-built row group metadata into a table. The schema is taken from
    /// the first row group (or empty when there are none) and then retained, so
    /// later pruning of all row groups does not lose it.
    pub fn new(row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        let schema = row_groups
            .first()
            .map(|rg| rg.schema.clone())
            .unwrap_or_else(|| EMPTY_SCHEMA.clone());
        Self { row_groups, schema }
    }

    /// A table of `schema` without row groups: a scan whose files were all
    /// pruned before their footers were read still has its columns.
    pub fn empty(schema: SchemaRef) -> Self {
        Self {
            row_groups: Vec::new(),
            schema,
        }
    }

    /// Read-only view of this table's row groups.
    pub fn row_groups(&self) -> &[Arc<RowGroupMetadata>] {
        &self.row_groups
    }

    /// Mutable access to this table's row groups. Used by callers that need
    /// to prune the row-group set in place (e.g. catalog-side filter pushdown).
    pub fn row_groups_mut(&mut self) -> &mut Vec<Arc<RowGroupMetadata>> {
        &mut self.row_groups
    }

    /// Build a table from an explicit, ordered list of local files.
    ///
    /// Drives the metadata-fetch dataflow, so it must run on the **coordinator**
    /// (the thread holding `dispatcher`), not inside a `run_on_worker` closure —
    /// a nested dataflow would deadlock the worker pool.
    ///
    /// `declared_columns` is the table's declared schema, reconciled with
    /// each file's own schema (see [`apply_declared_types`]); pass `&[]` when
    /// nothing was declared.
    pub fn from_files<P: AsRef<Path>>(
        dispatcher: &DataFlowDispatcher,
        paths: &[P],
        declared_columns: &[Column],
    ) -> Result<Self> {
        let files = paths
            .iter()
            .map(|p| {
                let path = p.as_ref();
                let size = fs::metadata(path)?.len();
                Ok(DataFile::local(path.to_path_buf(), size))
            })
            .collect::<io::Result<Vec<_>>>()?;
        Self::from_locations(dispatcher, files, declared_columns)
    }

    /// Read every file's footer in parallel (the metadata-fetch dataflow) and
    /// assemble the row groups. The locations may freely mix local and remote
    /// files. Same coordinator requirement as [`from_files`](Self::from_files).
    pub fn from_locations(
        dispatcher: &DataFlowDispatcher,
        files: Vec<DataFile>,
        declared_columns: &[Column],
    ) -> Result<Self> {
        let loaded = crate::metadata::load_file_row_groups(
            dispatcher,
            &files,
            TableColumns::by_name(declared_columns),
        )
        .map_err(|e| Error::Materialize(e.to_string()))?;
        let row_groups = loaded
            .iter()
            .flat_map(|f| f.row_groups.iter().cloned())
            .collect();
        Ok(Self::new(row_groups))
    }

    /// How many rows the table holds, summed over its row groups' footers less
    /// their deleted rows: the exact count with no data pages read.
    pub fn total_rows(&self) -> i64 {
        self.row_groups
            .iter()
            .map(|row_group| row_group.live_rows())
            .sum()
    }

    /// The `column`'s lower and upper bound over every row group, folded from
    /// the row-group statistics, or `None` when any row group lacks a bound
    /// (or the table has none): only a complete set of bounds answers a
    /// query's MIN/MAX without a scan. A bound held by a row group handed
    /// deleted rows is no bound either, as the rows holding it may be the
    /// deleted ones; a row group handed none has none. The scalars carry the
    /// column's physical storage type.
    pub fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        if self.row_groups.is_empty() {
            return None;
        }
        // Each bound with the row group holding it; a strictly tighter bound
        // replaces the one held, so an equal one keeps its row group.
        let mut min: Option<(Scalar<ArrayRef>, &RowGroupMetadata)> = None;
        let mut max: Option<(Scalar<ArrayRef>, &RowGroupMetadata)> = None;
        for row_group in &self.row_groups {
            let stats = row_group.column_statistics(column)?;
            let (group_min, group_max) = (stats.min()?, stats.max()?);
            min = Some(match min {
                Some(held) if !scalar_lt(&group_min, &held.0) => held,
                _ => (group_min, row_group),
            });
            max = Some(match max {
                Some(held) if !scalar_lt(&held.0, &group_max) => held,
                _ => (group_max, row_group),
            });
        }
        let (min, min_row_group) = min?;
        let (max, max_row_group) = max?;
        if min_row_group.deleted_rows.is_some() || max_row_group.deleted_rows.is_some() {
            return None;
        }
        Some((min, max))
    }

    /// Returns the table's Arrow schema. Preserved across pruning, so a table
    /// whose row groups were all eliminated still reports its real schema.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }
}

/// `a < b` for two single-value scalar bounds, compared in their shared
/// physical type. A null, a type mismatch or a kernel error reads as `false`,
/// so folding or ordering a column's per-row-group bounds gets a well-defined,
/// never-panicking answer.
pub(crate) fn scalar_lt(a: &Scalar<ArrayRef>, b: &Scalar<ArrayRef>) -> bool {
    arrow_ord::cmp::lt(a, b)
        .is_ok_and(|result| result.len() == 1 && result.is_valid(0) && result.value(0))
}

/// How much of a file's tail to fetch when probing for the footer. Almost every
/// footer fits, so it's a single read; a larger footer falls back to an exact
/// second read.
pub(crate) const FOOTER_PROBE_BYTES: usize = 64 * 1024;

/// Parse raw thrift footer bytes into this file's row groups (file-local
/// indices), tying each to `open_file` for the column-chunk reads that follow.
/// `table_columns` is the table's declared schema and how the file's columns
/// match it (see [`resolve_column_layout`]).
pub(crate) fn row_groups_from_footer(
    footer: &[u8],
    open_file: dispatch::io::OpenFile,
    table_columns: &TableColumns,
) -> Result<Vec<RowGroupMetadata>> {
    let file_meta = parse_footer_thrift(footer)?;
    row_groups_from_metadata(file_meta, open_file, table_columns)
}

/// Build the per-row-group metadata from a parsed footer and the (local or
/// remote) open file. Shared by the local and remote readers and by the writer's
/// upload path, which already holds the footer metadata it wrote and so skips the
/// parse. Row groups carry only their *file-local* index; the global index is the
/// row group's eventual position in the table's flat list.
///
/// The file's leaves are read in the file's own order, then laid out as
/// `table_columns` resolves them: a row group's `schema`, `columns` and
/// statistics all follow that layout, so every reader addresses a column the
/// same way whatever the file stored.
pub(crate) fn row_groups_from_metadata(
    file_meta: FileMetaData,
    open_file: dispatch::io::OpenFile,
    table_columns: &TableColumns,
) -> Result<Vec<RowGroupMetadata>> {
    let (file_schema, file_field_ids, leaf_infos) = schema_elements_to_arrow(&file_meta.schema)?;
    let file_leaf_count = leaf_infos.len();
    let ColumnLayout {
        schema,
        leaf_sources,
    } = resolve_column_layout(file_schema, &file_field_ids, table_columns)?;
    let schema = Arc::new(schema);
    // A row group lists one chunk per table leaf, in leaf order. The file's
    // own chunk list is that already when its leaves are exactly the table's,
    // in order; a file with leaves past the table's would add chunks no table
    // leaf addresses.
    let layout_is_identity = leaf_sources.len() == file_leaf_count
        && leaf_sources
            .iter()
            .enumerate()
            .all(|(leaf, source)| *source == LeafSource::File(leaf));
    // Statistics belong to leaf chunks, not top-level columns.
    let leaves = super::leaves::leaf_fields(schema.fields());

    let row_group_count = file_meta.row_groups.len();
    // Statistics are gathered per file leaf rather than per row group, so each
    // leaf's bounds decode into one array covering the whole file. See
    // [`FileStatistics`].
    let mut leaf_stats: Vec<Vec<Option<Statistics>>> = (0..file_leaf_count)
        .map(|_| Vec::with_capacity(row_group_count))
        .collect();
    let mut leaf_physical_types = vec![0_i32; file_leaf_count];
    let mut row_groups = Vec::with_capacity(row_group_count);

    for (i, rg) in file_meta.row_groups.into_iter().enumerate() {
        // Chunks are per leaf; a row group with a different count is a
        // malformed footer and would index out of bounds below.
        if rg.columns.len() != file_leaf_count {
            return Err(Error::InvalidFooter(format!(
                "row group {i} has {} column chunks but the schema has {} leaves",
                rg.columns.len(),
                file_leaf_count
            )));
        }
        let mut file_chunks: Vec<ColumnChunkMeta> = rg
            .columns
            .into_iter()
            .enumerate()
            .map(|(j, cc)| {
                let meta = cc.meta_data.expect("missing column metadata");
                let physical_type = meta.physical_type;
                leaf_physical_types[j] = physical_type;
                leaf_stats[j].push(meta.statistics);
                let data_pages_all_dictionary = meta.dictionary_page_offset.is_some()
                    && data_pages_all_dictionary(meta.encoding_stats.as_deref());
                ColumnChunkMeta {
                    codec: meta.codec,
                    dictionary_page_offset: meta.dictionary_page_offset,
                    data_page_offset: meta.data_page_offset,
                    total_compressed_size: meta.total_compressed_size,
                    total_uncompressed_size: meta.total_uncompressed_size,
                    max_def_level: leaf_infos[j].def_level,
                    physical_type,
                    fixed_len_byte_width: leaf_infos[j].type_length,
                    data_pages_all_dictionary,
                    absent: false,
                }
            })
            .collect();
        // The collect ran in place over the buffer the footer parse allocated
        // for the chunks, which is several times wider per element than what is
        // kept. The surplus lives as long as the table does, so release it.
        file_chunks.shrink_to_fit();
        // Lay the chunks out as the resolved schema orders its leaves.
        let columns: Vec<ColumnChunkMeta> = if layout_is_identity {
            file_chunks
        } else {
            leaf_sources
                .iter()
                .map(|source| match source {
                    LeafSource::File(leaf) => file_chunks[*leaf].clone(),
                    LeafSource::Absent => ColumnChunkMeta::absent(),
                })
                .collect()
        };
        row_groups.push((rg.num_rows, columns));
    }

    let row_counts: Vec<i64> = row_groups.iter().map(|(num_rows, _)| *num_rows).collect();
    let statistics = Arc::new(
        leaf_sources
            .iter()
            .enumerate()
            .map(|(leaf, source)| match source {
                LeafSource::File(file_leaf) => decode_leaf_statistics(
                    std::mem::take(&mut leaf_stats[*file_leaf]),
                    leaves[leaf].data_type(),
                    leaf_physical_types[*file_leaf],
                ),
                LeafSource::Absent => Some(absent_leaf_statistics(&row_counts)),
            })
            .collect::<Vec<_>>(),
    );

    let mut file_row_offset = 0u64;
    Ok(row_groups
        .into_iter()
        .enumerate()
        .map(|(i, (num_rows, columns))| {
            let row_group = RowGroupMetadata {
                open_file: open_file.clone(),
                schema: schema.clone(),
                columns,
                statistics: statistics.clone(),
                num_rows,
                file_row_offset,
                deleted_rows: None,
                file_row_group_idx: i,
                live_decompressed_pages: Arc::new(AtomicUsize::new(0)),
            };
            file_row_offset += num_rows as u64;
            row_group
        })
        .collect())
}

/// The statistics of a leaf the file has no chunk for: every row is NULL, so
/// each row group's null count is its row count and there are no bounds.
fn absent_leaf_statistics(row_counts: &[i64]) -> FileLeafStatistics {
    FileLeafStatistics {
        min: None,
        max: None,
        null_counts: row_counts.iter().map(|rows| Some(*rows)).collect(),
        distinct_counts: row_counts.iter().map(|_| Some(0)).collect(),
    }
}

/// Where a row group's leaf comes from, in the layout
/// [`resolve_column_layout`] produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeafSource {
    /// The file's leaf at this index, in the file's own leaf order.
    File(usize),
    /// A leaf the file has no chunk for; it reads as NULL.
    Absent,
}

/// A file's schema as the table sees it, and where each of its leaves comes
/// from.
#[derive(Debug)]
struct ColumnLayout {
    schema: Schema,
    /// One entry per leaf of `schema`, in leaf order.
    leaf_sources: Vec<LeafSource>,
}

/// Lay a file's columns out as `table_columns` resolves them against the
/// declared schema (see [`ColumnResolution`]), then reconcile the declared
/// column types with what the file stores ([`apply_declared_types`]).
/// `file_field_ids` is each top-level file column's Parquet field id, in
/// `file_schema` order.
fn resolve_column_layout(
    file_schema: Schema,
    file_field_ids: &[Option<i32>],
    table_columns: &TableColumns,
) -> Result<ColumnLayout> {
    let field_ids = match table_columns.resolution() {
        ColumnResolution::ByName => {
            let leaf_count = super::leaves::leaf_fields(file_schema.fields()).len();
            return Ok(ColumnLayout {
                schema: apply_declared_types(file_schema, table_columns.columns())?,
                leaf_sources: (0..leaf_count).map(LeafSource::File).collect(),
            });
        }
        ColumnResolution::ByFieldId(field_ids) => field_ids,
    };

    let file_fields = file_schema.fields();
    // A file written without field ids is matched by name; one written with
    // them is matched by id alone, so a renamed column still finds its data.
    let mut file_column_by_id = HashMap::with_capacity(file_field_ids.len());
    for (index, field_id) in file_field_ids.iter().enumerate() {
        let Some(field_id) = field_id else { continue };
        if let Some(first) = file_column_by_id.insert(*field_id, index) {
            return Err(Error::InvalidFooter(format!(
                "columns '{}' and '{}' both carry field id {field_id}",
                file_fields[first].name(),
                file_fields[index].name()
            )));
        }
    }
    let file_column_of_declared: Vec<Option<usize>> = if file_column_by_id.is_empty() {
        table_columns
            .columns()
            .iter()
            .map(|declared| {
                file_fields
                    .iter()
                    .position(|field| field.name().eq_ignore_ascii_case(&declared.name))
            })
            .collect()
    } else {
        field_ids
            .iter()
            .map(|field_id| file_column_by_id.get(field_id).copied())
            .collect()
    };

    let mut fields = Vec::with_capacity(table_columns.columns().len());
    let mut leaf_sources = Vec::new();
    for (declared, file_column) in table_columns.columns().iter().zip(file_column_of_declared) {
        match file_column {
            Some(index) => {
                let file_field = &file_fields[index];
                fields.push(Arc::new(
                    file_field.as_ref().clone().with_name(declared.name.clone()),
                ));
                leaf_sources
                    .extend(super::leaves::leaf_range(file_fields, index).map(LeafSource::File));
            }
            None => {
                let data_type = planner::types::physical_arrow_type(&declared.col_type);
                let field = Arc::new(Field::new(declared.name.clone(), data_type, true));
                leaf_sources.extend(std::iter::repeat_n(LeafSource::Absent, leaf_count(&field)));
                fields.push(field);
            }
        }
    }
    Ok(ColumnLayout {
        schema: apply_declared_types(Schema::new(fields), table_columns.columns())?,
        leaf_sources,
    })
}

/// Returns `true` when `encoding_stats` proves every *data* page in the chunk
/// is dictionary-encoded (`PLAIN_DICTIONARY` / `RLE_DICTIONARY`). Requires at
/// least one data page and no data page using any other encoding. Returns
/// `false` when stats are absent — a missing signal is not proof, so we must
/// not prune.
fn data_pages_all_dictionary(encoding_stats: Option<&[PageEncodingStats]>) -> bool {
    let Some(stats) = encoding_stats else {
        return false;
    };
    let mut saw_data_page = false;
    for s in stats {
        if s.page_type != PageType::DATA_PAGE && s.page_type != PageType::DATA_PAGE_V2 {
            continue;
        }
        if s.count == 0 {
            continue;
        }
        saw_data_page = true;
        if s.encoding != Encoding::PLAIN_DICTIONARY && s.encoding != Encoding::RLE_DICTIONARY {
            return false;
        }
    }
    saw_data_page
}

/// Decode one leaf's Parquet `Statistics` across a file's row groups into the
/// column-wise [`FileLeafStatistics`], using `data_type` to choose the physical-bytes
/// to Arrow conversion. `physical_type` disambiguates a decimal's byte encoding,
/// which follows the column's physical storage rather than its arrow type.
///
/// Prefer the modern `min_value` / `max_value` fields and fall back to the
/// legacy `min` / `max` only when those aren't populated. Returns `None` when no
/// row group recorded anything for the leaf.
fn decode_leaf_statistics(
    chunks: Vec<Option<Statistics>>,
    data_type: &DataType,
    physical_type: i32,
) -> Option<FileLeafStatistics> {
    let mut min_bytes = Vec::with_capacity(chunks.len());
    let mut max_bytes = Vec::with_capacity(chunks.len());
    let mut null_counts = Vec::with_capacity(chunks.len());
    let mut distinct_counts = Vec::with_capacity(chunks.len());
    let mut recorded = false;
    for chunk in chunks {
        let (min, max, null_count, distinct_count) = match chunk {
            Some(stats) => (
                stats.min_value.or(stats.min),
                stats.max_value.or(stats.max),
                stats.null_count,
                stats.distinct_count,
            ),
            None => (None, None, None, None),
        };
        recorded |=
            min.is_some() || max.is_some() || null_count.is_some() || distinct_count.is_some();
        min_bytes.push(min);
        max_bytes.push(max);
        null_counts.push(null_count);
        distinct_counts.push(distinct_count);
    }
    if !recorded {
        return None;
    }
    Some(FileLeafStatistics {
        min: decode_bounds(&min_bytes, data_type, physical_type),
        max: decode_bounds(&max_bytes, data_type, physical_type),
        null_counts,
        distinct_counts,
    })
}

/// One leaf's bound for every row group as a single array, in file order, null
/// where a row group recorded none or its bytes don't match the expected width.
/// `None` for a type this doesn't decode, or when no row group recorded the
/// bound at all, which leaves the leaf out of range pruning.
fn decode_bounds(
    values: &[Option<Vec<u8>>],
    data_type: &DataType,
    physical_type: i32,
) -> Option<ArrayRef> {
    if values.iter().all(Option::is_none) {
        return None;
    }

    match data_type {
        DataType::Boolean => Some(Arc::new(
            values
                .iter()
                .map(|bytes| bytes.as_deref().and_then(|b| b.first().map(|v| *v != 0)))
                .collect::<BooleanArray>(),
        )),
        DataType::Int8 => Some(primitive_bounds::<Int8Type>(values, |bytes| {
            read_le::<4>(bytes)
                .map(i32::from_le_bytes)
                .and_then(|v| i8::try_from(v).ok())
        })),
        DataType::UInt8 => Some(primitive_bounds::<UInt8Type>(values, |bytes| {
            read_le::<4>(bytes)
                .map(u32::from_le_bytes)
                .and_then(|v| u8::try_from(v).ok())
        })),
        DataType::Int16 => Some(primitive_bounds::<Int16Type>(values, |bytes| {
            read_le::<4>(bytes)
                .map(i32::from_le_bytes)
                .and_then(|v| i16::try_from(v).ok())
        })),
        DataType::UInt16 => Some(primitive_bounds::<UInt16Type>(values, |bytes| {
            read_le::<4>(bytes)
                .map(u32::from_le_bytes)
                .and_then(|v| u16::try_from(v).ok())
        })),
        DataType::Int32 => Some(primitive_bounds::<Int32Type>(values, |bytes| {
            read_le::<4>(bytes).map(i32::from_le_bytes)
        })),
        DataType::UInt32 => Some(primitive_bounds::<UInt32Type>(values, |bytes| {
            read_le::<4>(bytes).map(u32::from_le_bytes)
        })),
        DataType::Int64 => Some(primitive_bounds::<Int64Type>(values, |bytes| {
            read_le::<8>(bytes).map(i64::from_le_bytes)
        })),
        DataType::UInt64 => Some(primitive_bounds::<UInt64Type>(values, |bytes| {
            read_le::<8>(bytes).map(u64::from_le_bytes)
        })),
        DataType::Float32 => Some(primitive_bounds::<Float32Type>(values, |bytes| {
            read_le::<4>(bytes).map(f32::from_le_bytes)
        })),
        DataType::Float64 => Some(primitive_bounds::<Float64Type>(values, |bytes| {
            read_le::<8>(bytes).map(f64::from_le_bytes)
        })),
        DataType::Utf8View => {
            let mut builder = StringViewBuilder::with_capacity(values.len())
                .with_fixed_block_size(bounds_block_size(values));
            for bytes in values {
                match bytes.as_deref().and_then(|b| std::str::from_utf8(b).ok()) {
                    Some(value) => builder.append_value(value),
                    None => builder.append_null(),
                }
            }
            Some(Arc::new(builder.finish()))
        }
        DataType::BinaryView => {
            let mut builder = BinaryViewBuilder::with_capacity(values.len())
                .with_fixed_block_size(bounds_block_size(values));
            for bytes in values {
                match bytes.as_deref() {
                    Some(value) => builder.append_value(value),
                    None => builder.append_null(),
                }
            }
            Some(Arc::new(builder.finish()))
        }
        // Temporal stats share their physical int's encoding (Date32 the i32
        // days, Timestamp(Microsecond) the i64 count).
        DataType::Date32 => Some(primitive_bounds::<Date32Type>(values, |bytes| {
            read_le::<4>(bytes).map(i32::from_le_bytes)
        })),
        DataType::Timestamp(TimeUnit::Microsecond, zone) => {
            let array = values
                .iter()
                .map(|bytes| {
                    bytes
                        .as_deref()
                        .and_then(|b| read_le::<8>(b).map(i64::from_le_bytes))
                })
                .collect::<TimestampMicrosecondArray>()
                .with_timezone_opt(zone.clone());
            Some(Arc::new(array))
        }
        // A decimal's unscaled integer follows the column's physical storage;
        // the array's carrier follows the column's arrow type.
        DataType::Decimal64(precision, scale) => {
            let array = values
                .iter()
                .map(|bytes| {
                    bytes
                        .as_deref()
                        .and_then(|b| decimal_stat_value(b, physical_type))
                        .and_then(|value| i64::try_from(value).ok())
                })
                .collect::<Decimal64Array>()
                .with_precision_and_scale(*precision, *scale)
                .ok()?;
            Some(Arc::new(array))
        }
        DataType::Decimal128(precision, scale) => {
            let array = values
                .iter()
                .map(|bytes| {
                    bytes
                        .as_deref()
                        .and_then(|b| decimal_stat_value(b, physical_type))
                })
                .collect::<Decimal128Array>()
                .with_precision_and_scale(*precision, *scale)
                .ok()?;
            Some(Arc::new(array))
        }
        _ => None,
    }
}

/// Read `N` bytes as a little-endian fixed-width primitive. Returns `None` if
/// the byte slice doesn't have exactly `N` bytes.
fn read_le<const N: usize>(bytes: &[u8]) -> Option<[u8; N]> {
    bytes.try_into().ok()
}

/// One leaf's bounds as a primitive column: `decode` reads a single row group's
/// statistic bytes, and a row group with no bytes, or bytes the decoder rejects,
/// becomes a null.
fn primitive_bounds<T: ArrowPrimitiveType>(
    values: &[Option<Vec<u8>>],
    decode: impl Fn(&[u8]) -> Option<T::Native>,
) -> ArrayRef {
    Arc::new(
        values
            .iter()
            .map(|bytes| bytes.as_deref().and_then(&decode))
            .collect::<PrimitiveArray<T>>(),
    )
}

/// Data-block width for a leaf's column of view bounds: the bytes they hold
/// between them.
///
/// A view builder left on its default sizing reserves its first block by
/// doubling from 8 KiB, and the finished buffer keeps that capacity rather than
/// shrinking to the bytes written. One leaf's bounds are a handful of short
/// values held for as long as the file's metadata, so sizing the block to them
/// keeps the cost proportional to what it stores. Values of 12 bytes or fewer
/// are packed into the view itself and never touch a block, but the width still
/// has to be non-zero.
fn bounds_block_size(values: &[Option<Vec<u8>>]) -> u32 {
    let total: usize = values.iter().flatten().map(Vec::len).sum();
    u32::try_from(total).unwrap_or(u32::MAX).max(1)
}

/// A decimal statistic's unscaled integer, decoded per the column's physical
/// storage: INT32/INT64 stats are little-endian, FIXED_LEN_BYTE_ARRAY stats
/// are big-endian two's complement in the declared length.
fn decimal_stat_value(bytes: &[u8], physical_type: i32) -> Option<i128> {
    use crate::thrift::general::Type as PhysicalType;
    if physical_type == PhysicalType::INT32 as i32 {
        bytes
            .try_into()
            .ok()
            .map(i32::from_le_bytes)
            .map(i128::from)
    } else if physical_type == PhysicalType::INT64 as i32 {
        bytes
            .try_into()
            .ok()
            .map(i64::from_le_bytes)
            .map(i128::from)
    } else if physical_type == PhysicalType::FIXED_LEN_BYTE_ARRAY as i32 {
        i128_from_be_bytes(bytes)
    } else {
        None
    }
}

/// Sign-extend a big-endian two's-complement integer of up to 16 bytes into
/// an `i128`. Returns `None` for byte lengths a `Decimal128` cannot hold.
fn i128_from_be_bytes(bytes: &[u8]) -> Option<i128> {
    if bytes.is_empty() || bytes.len() > 16 {
        return None;
    }
    let fill = if bytes[0] & 0x80 != 0 { 0xff } else { 0 };
    let mut buf = [fill; 16];
    buf[16 - bytes.len()..].copy_from_slice(bytes);
    Some(i128::from_be_bytes(buf))
}

/// Deserialises raw Thrift bytes into a [`FileMetaData`].
fn parse_footer_thrift(buf: &[u8]) -> Result<FileMetaData> {
    let mut prot = ThriftSliceInputProtocol::new(buf);
    FileMetaData::read_thrift(&mut prot)
        .map_err(|e| Error::IO(io::Error::new(io::ErrorKind::InvalidData, e.to_string())))
}

/// Per-leaf schema facts collected while walking the footer's schema
/// elements, in depth-first (column-chunk) order.
#[derive(Debug)]
struct LeafSchemaInfo {
    /// Maximum definition level (one per optional ancestor plus the leaf).
    def_level: i16,
    /// The element's `type_length` (the byte width of a FIXED_LEN_BYTE_ARRAY
    /// value).
    type_length: Option<i32>,
}

/// Converts footer schema elements into an Arrow schema, each top-level
/// column's Parquet field id (in schema order), and per-leaf schema facts.
fn schema_elements_to_arrow(
    elements: &[crate::thrift::footer::SchemaElement],
) -> Result<(Schema, Vec<Option<i32>>, Vec<LeafSchemaInfo>)> {
    if elements.is_empty() {
        return Err(Error::IO(io::Error::new(
            io::ErrorKind::InvalidData,
            "empty schema",
        )));
    }
    let mut leaf_infos = Vec::new();
    let mut cursor = 1; // element 0 is the root group
    let mut fields = Vec::new();
    let mut field_ids = Vec::new();
    for _ in 0..elements[0].num_children.unwrap_or(0) {
        field_ids.push(elements.get(cursor).and_then(|element| element.field_id));
        let (field, next) = parse_schema_element(elements, cursor, 0, 0, &mut leaf_infos)?;
        fields.push(field);
        cursor = next;
    }
    Ok((Schema::new(fields), field_ids, leaf_infos))
}

/// Deepest schema nesting accepted. Real schemas stay far below this (a
/// shredded variant adds two levels per path segment); the cap only exists so
/// a malformed or hostile footer can't overflow the stack through
/// [`parse_schema_element`]'s per-level recursion, which would abort the
/// process rather than fail the load.
const MAX_SCHEMA_DEPTH: usize = 128;

/// Reconcile a file's parsed schema with the table's declared column types,
/// matched by column name: a file may carry more columns than the table
/// declares (an ingest mapping can write a superset), and those pass through
/// untouched.
///
/// A declared VARCHAR may retype unannotated binary data as text. A declared
/// VARIANT accepts any struct shape because shredding adds fields per file.
///
/// Other declared types must match the file's storage type. Undeclared file
/// fields are unchanged.
fn apply_declared_types(schema: Schema, declared_columns: &[Column]) -> Result<Schema> {
    // Keyed case-insensitively: DuckDB resolves identifiers that way, so a
    // file written by another tool may spell a declared column differently
    // and must still reconcile with it.
    let declared_by_name: std::collections::HashMap<String, &Type> = declared_columns
        .iter()
        .map(|c| (c.name.to_lowercase(), &c.col_type))
        .collect();
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            let Some(declared) = declared_by_name.get(&field.name().to_lowercase()) else {
                return Ok(field.clone());
            };
            let file_type = field.data_type();
            if **declared == Type::Utf8 && *file_type == DataType::BinaryView {
                return Ok(Arc::new(
                    field.as_ref().clone().with_data_type(DataType::Utf8View),
                ));
            }
            // A pivot timestamp counts microseconds (see `Type::Timestamp`),
            // and both timestamp types share that INT64 storage: the file's
            // `is_adjusted_to_utc` flag and the declaration disagree only
            // about zone-ness, which the declaration decides. So a declared
            // timestamp column accepts any microsecond-timestamp leaf, and
            // also a bare INT64 one (which says nothing about its own unit),
            // re-labelling it to the declared type; that is how a file written
            // by another engine, or by an older pivot, reads. Any other
            // timestamp unit was rejected when the footer was read.
            if matches!(declared, Type::Timestamp | Type::TimestampTz)
                && matches!(
                    file_type,
                    DataType::Int64 | DataType::Timestamp(TimeUnit::Microsecond, _)
                )
            {
                return Ok(Arc::new(
                    field
                        .as_ref()
                        .clone()
                        .with_data_type(planner::types::physical_arrow_type(declared)),
                ));
            }
            let matches_declared = match declared {
                Type::Variant => matches!(file_type, DataType::Struct(_)),
                _ => *file_type == planner::types::physical_arrow_type(declared),
            };
            if matches_declared {
                Ok(field.clone())
            } else {
                Err(Error::DeclaredTypeMismatch {
                    column: field.name().clone(),
                    declared: (*declared).clone(),
                    file_type: file_type.clone(),
                })
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Schema::new(fields))
}

/// Parse the subtree rooted at `elements[idx]`. `parent_def` is the definition
/// level contributed by ancestors (each optional ancestor adds one). Returns
/// the Arrow field and the index just past this subtree, appending one entry
/// per leaf to `leaf_infos` in depth-first (column-chunk) order.
fn parse_schema_element(
    elements: &[crate::thrift::footer::SchemaElement],
    idx: usize,
    parent_def: i16,
    depth: usize,
    leaf_infos: &mut Vec<LeafSchemaInfo>,
) -> Result<(Field, usize)> {
    use crate::thrift::footer::LogicalType;
    // The cursor is driven by each group's *claimed* child count, so a
    // malformed footer can point past the element list; fail the load
    // instead of panicking mid-fetch.
    let Some(elem) = elements.get(idx) else {
        return Err(Error::InvalidFooter(
            "schema group claims more children than the footer holds".into(),
        ));
    };
    if depth >= MAX_SCHEMA_DEPTH {
        return Err(Error::InvalidFooter(format!(
            "schema nesting exceeds {MAX_SCHEMA_DEPTH} levels"
        )));
    }
    // repetition_type: 0=REQUIRED, 1=OPTIONAL, 2=REPEATED.
    // REPEATED elements (the LIST/MAP encodings) need repetition-level
    // decoding, which the page decoder doesn't do; accepting them here would
    // silently misdecode their pages.
    if elem.repetition_type == Some(2) {
        return Err(Error::UnsupportedType(format!(
            "repeated field '{}' (LIST/MAP columns are not supported)",
            elem.name
        )));
    }
    let nullable = elem.repetition_type == Some(1);
    let def_level = parent_def + nullable as i16;

    match elem.num_children {
        Some(n) if n > 0 => {
            // Children can't outnumber the elements after this one; a bigger
            // claim is malformed (and would size the Vec from hostile input).
            if n as usize > elements.len() - idx - 1 {
                return Err(Error::InvalidFooter(
                    "schema group claims more children than the footer holds".into(),
                ));
            }
            let is_variant = elem.logical_type == Some(LogicalType::Variant);
            let mut children = Vec::with_capacity(n as usize);
            let mut cursor = idx + 1;
            for _ in 0..n {
                let (child, next) =
                    parse_schema_element(elements, cursor, def_level, depth + 1, leaf_infos)?;
                children.push(child);
                cursor = next;
            }
            let field = Field::new(&elem.name, DataType::Struct(children.into()), nullable);
            // Re-mark the group so a reconstructed variant re-tags VARIANT if
            // the table is re-written (e.g. by compaction).
            let field = if is_variant {
                field.with_metadata(variant_extension_metadata())
            } else {
                field
            };
            Ok((field, cursor))
        }
        _ => {
            leaf_infos.push(LeafSchemaInfo {
                def_level,
                type_length: elem.type_length,
            });
            let data_type = super::arrow_map::parquet_to_arrow(elem)?;
            Ok((Field::new(&elem.name, data_type, nullable), idx + 1))
        }
    }
}

/// The canonical Arrow extension type name for a Parquet variant. The reader
/// stamps it onto a VARIANT group's field and the writer looks for it to decide
/// which struct columns to write back as VARIANT, so both name it from here.
pub const VARIANT_EXTENSION_NAME: &str = "arrow.parquet.variant";

/// The Arrow field metadata that marks a struct as a Parquet variant.
pub fn variant_extension_metadata() -> std::collections::HashMap<String, String> {
    use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
    [
        (
            EXTENSION_TYPE_NAME_KEY.to_owned(),
            VARIANT_EXTENSION_NAME.to_owned(),
        ),
        (EXTENSION_TYPE_METADATA_KEY.to_owned(), String::new()),
    ]
    .into()
}

/// Whether `field` is a variant column — a struct carrying the Arrow variant
/// extension tag, stamped either by [`parse_schema_element`] when reading a
/// VARIANT group back or by a producer building one to write.
pub fn is_variant_field(field: &Field) -> bool {
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    matches!(field.data_type(), DataType::Struct(_))
        && field
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .map(String::as_str)
            == Some(VARIANT_EXTENSION_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::deleted_rows;
    use crate::types::metadata::ColumnStatistics;
    use arrow_array::{
        BinaryViewArray, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, Scalar,
        StringViewArray,
    };
    use arrow_schema::Field;
    use dispatch::{DataFlowDispatcher, Dispatch};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, ZstdLevel};
    use parquet::file::properties::{EnabledStatistics, WriterProperties};
    use std::fs::File;
    use std::sync::{Arc, OnceLock};
    use tempfile::TempDir;

    /// A shared one-worker dispatch pool for this module's tests, so
    /// `ParquetTable::from_files` can drive its metadata-fetch dataflow.
    fn test_dispatcher() -> &'static DataFlowDispatcher {
        static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
        DISPATCH
            .get_or_init(|| Dispatch::spin_up(1, 32, None))
            .dispatcher()
    }

    fn write_parquet(batch: &RecordBatch, stats: EnabledStatistics) -> (TempDir, ParquetTable) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data.parquet");
        let props = WriterProperties::builder()
            .set_statistics_enabled(stats)
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props))
                .unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        let table = ParquetTable::from_files(test_dispatcher(), &[path], &[]).unwrap();
        (dir, table)
    }

    fn col_stats(table: &ParquetTable, col: usize) -> ColumnStatistics<'_> {
        table.row_groups[0]
            .leaf_statistics(col)
            .expect("stats should be present")
    }

    /// The single-value array behind one of a row group's bounds.
    fn bound(bound: Option<Scalar<ArrayRef>>) -> ArrayRef {
        bound.expect("bound should be present").into_inner()
    }

    /// A table of two row groups, `[1, 9, 9]` and `[2, 9, 9]`, with the rows
    /// `deleted` of each row group marked deleted.
    fn two_row_groups_with_deleted(deleted: [&[u32]; 2]) -> (TempDir, ParquetTable) {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 9, 9, 2, 9, 9]));
        let batch = RecordBatch::try_from_iter([("v", values)]).unwrap();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data.parquet");
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(3))
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props))
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let mut table = ParquetTable::from_files(test_dispatcher(), &[path], &[]).unwrap();
        for (row_group, rows) in table.row_groups_mut().iter_mut().zip(deleted) {
            let row_group = Arc::get_mut(row_group).unwrap();
            row_group.deleted_rows = deleted_rows(row_group.file_row_offset, rows);
        }
        (dir, table)
    }

    fn int64_bound(bound: Scalar<ArrayRef>) -> i64 {
        bound
            .into_inner()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }

    #[test]
    fn a_bound_held_by_a_row_group_with_deleted_rows_is_no_bound() {
        let (_dir, deleted_in_second) = two_row_groups_with_deleted([&[], &[0]]);
        let (_dir, deleted_in_first) = two_row_groups_with_deleted([&[0], &[]]);

        let (min, max) = deleted_in_second
            .column_min_max(0)
            .expect("the first row group holds both bounds");
        assert_eq!((int64_bound(min), int64_bound(max)), (1, 9));
        assert!(deleted_in_first.column_min_max(0).is_none());
    }

    #[test]
    fn total_rows_leaves_out_deleted_rows() {
        let (_dir, table) = two_row_groups_with_deleted([&[0, 2], &[1]]);

        assert_eq!(table.total_rows(), 3);
    }

    #[test]
    fn binary_view_statistics_decode_as_raw_bytes() {
        // Setup
        let values: ArrayRef = Arc::new(BinaryViewArray::from(vec![
            &[0_u8][..],
            &[2_u8][..],
            &[1_u8][..],
        ]));
        let batch = RecordBatch::try_from_iter([("value", values)]).unwrap();

        // Execute
        let (_dir, table) = write_parquet(&batch, EnabledStatistics::Chunk);
        let stats = col_stats(&table, 0);

        // Assert
        let min = bound(stats.min());
        let max = bound(stats.max());
        assert_eq!(
            min.as_any()
                .downcast_ref::<BinaryViewArray>()
                .unwrap()
                .value(0),
            &[0]
        );
        assert_eq!(
            max.as_any()
                .downcast_ref::<BinaryViewArray>()
                .unwrap()
                .value(0),
            &[2]
        );
    }

    #[test]
    fn row_group_column_metadata_keeps_no_surplus_capacity() {
        // Setup
        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let batch = RecordBatch::try_from_iter([("value", values)]).unwrap();

        // Execute
        let (_dir, table) = write_parquet(&batch, EnabledStatistics::Chunk);

        // Assert
        let columns = &table.row_groups[0].columns;
        assert_eq!(columns.capacity(), columns.len());
    }

    fn enc_stat(page_type: PageType, encoding: Encoding, count: i32) -> PageEncodingStats {
        PageEncodingStats {
            page_type,
            encoding,
            count,
        }
    }

    fn column(name: &str, col_type: Type) -> Column {
        Column {
            name: name.to_string(),
            col_type,
        }
    }

    /// A column the table declares VARCHAR whose file leaf is unannotated
    /// binary is retyped to text; every other pairing is left alone.
    #[test]
    fn declared_varchar_retypes_an_unannotated_binary_column() {
        let schema = Schema::new(vec![
            Field::new("url", DataType::BinaryView, false),
            Field::new("id", DataType::Int64, false),
        ]);

        let declared = [column("url", Type::Utf8), column("id", Type::Int64)];
        let reconciled = apply_declared_types(schema, &declared).unwrap();

        assert_eq!(*reconciled.field(0).data_type(), DataType::Utf8View);
        assert_eq!(*reconciled.field(1).data_type(), DataType::Int64);
    }

    /// Matching is by column name: file columns the table doesn't declare
    /// pass through untouched (an ingest mapping can write a superset, in any
    /// order), and an empty declaration changes nothing.
    #[test]
    fn undeclared_file_columns_pass_through_untouched() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::BinaryView, false),
            Field::new("b", DataType::BinaryView, false),
        ]);

        let reconciled = apply_declared_types(schema.clone(), &[column("b", Type::Utf8)]).unwrap();
        let untouched = apply_declared_types(schema, &[]).unwrap();

        assert_eq!(*reconciled.field(0).data_type(), DataType::BinaryView);
        assert_eq!(*reconciled.field(1).data_type(), DataType::Utf8View);
        assert_eq!(*untouched.field(0).data_type(), DataType::BinaryView);
    }

    /// A declared VARIANT column accepts any struct shape (shredding widens
    /// the stored struct per file).
    #[test]
    fn declared_variant_accepts_any_struct_shape() {
        let doc = Field::new(
            "d",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("value", DataType::BinaryView, true),
                    Field::new("age", DataType::Int64, true),
                ]
                .into(),
            ),
            true,
        );
        let schema = Schema::new(vec![doc.clone()]);

        let reconciled = apply_declared_types(schema, &[column("d", Type::Variant)]).unwrap();

        assert_eq!(reconciled.field(0), &doc);
    }

    /// A file whose stored type contradicts the declaration fails the load,
    /// naming the offending column and both types.
    #[test]
    fn declared_type_mismatch_fails_naming_the_column() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("price", DataType::Utf8View, false),
        ]);

        let declared = [column("id", Type::Int64), column("price", Type::Float64)];
        let err = apply_declared_types(schema, &declared).unwrap_err();

        let message = err.to_string();
        assert!(
            message.contains("'price'")
                && message.contains("Float64")
                && message.contains("Utf8View"),
            "unhelpful mismatch error: {message}"
        );
    }

    /// Both timestamp storage flavors (pivot's plain INT64 count of
    /// microseconds and an annotated `Timestamp(Microsecond)`) reconcile to the
    /// one arrow type the executor reads, so a table mixing such files emits
    /// uniformly-typed batches.
    #[test]
    fn declared_timestamp_reconciles_both_storage_flavors() {
        let micros = DataType::Timestamp(TimeUnit::Microsecond, None);
        let schema = Schema::new(vec![
            Field::new("ts_plain", DataType::Int64, false),
            Field::new("ts_annotated", micros.clone(), false),
        ]);
        let declared = [
            column("ts_plain", Type::Timestamp),
            column("ts_annotated", Type::Timestamp),
        ];

        let reconciled = apply_declared_types(schema, &declared).unwrap();

        assert_eq!(*reconciled.field(0).data_type(), micros);
        assert_eq!(*reconciled.field(1).data_type(), micros);
    }

    /// DuckDB resolves identifiers case-insensitively, so a declared column
    /// must reconcile with a file column spelled in another case.
    #[test]
    fn declared_columns_match_file_columns_case_insensitively() {
        let schema = Schema::new(vec![Field::new("URL", DataType::BinaryView, false)]);

        let reconciled = apply_declared_types(schema, &[column("url", Type::Utf8)]).unwrap();

        assert_eq!(*reconciled.field(0).data_type(), DataType::Utf8View);
    }

    fn schema_element(
        name: &str,
        repetition_type: Option<i32>,
        num_children: Option<i32>,
        physical_type: Option<i32>,
    ) -> crate::thrift::footer::SchemaElement {
        crate::thrift::footer::SchemaElement {
            physical_type,
            type_length: None,
            repetition_type,
            name: name.to_string(),
            num_children,
            converted_type: None,
            scale: None,
            precision: None,
            field_id: None,
            logical_type: None,
        }
    }

    /// A REPEATED element (the LIST/MAP encoding) fails the load cleanly:
    /// the page decoder has no repetition-level support, so accepting it
    /// would silently misdecode.
    #[test]
    fn repeated_schema_elements_fail_the_load() {
        let elements = vec![
            schema_element("root", None, Some(1), None),
            schema_element("tags", Some(1), Some(1), None),
            schema_element("list", Some(2), Some(1), None),
            schema_element("element", Some(1), None, Some(2)),
        ];

        let err = schema_elements_to_arrow(&elements).unwrap_err();

        assert!(err.to_string().contains("not supported"), "{err}");
    }

    /// A group claiming more children than the footer holds is malformed and
    /// fails the load instead of indexing out of bounds.
    #[test]
    fn overclaimed_child_counts_fail_the_load() {
        let elements = vec![
            schema_element("root", None, Some(3), None),
            schema_element("a", Some(1), None, Some(2)),
        ];

        let err = schema_elements_to_arrow(&elements).unwrap_err();

        assert!(err.to_string().contains("more children"), "{err}");
    }

    /// Nesting past the cap fails the load instead of overflowing the stack
    /// (which would abort the process, not unwind).
    #[test]
    fn absurdly_deep_nesting_fails_the_load() {
        let mut elements = vec![schema_element("root", None, Some(1), None)];
        for i in 0..=MAX_SCHEMA_DEPTH {
            elements.push(schema_element(&format!("g{i}"), Some(1), Some(1), None));
        }
        elements.push(schema_element("leaf", Some(1), None, Some(2)));

        let err = schema_elements_to_arrow(&elements).unwrap_err();

        assert!(err.to_string().contains("nesting"), "{err}");
    }

    /// Pruning away every row group keeps the table's schema, so a scan that
    /// projects a column still builds. Regression: a fully-pruned table reported
    /// an empty schema, panicking the decoder build on the projected column.
    #[test]
    fn schema_survives_pruning_every_row_group() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![2])),
            ],
        )
        .unwrap();
        let (_dir, mut table) = write_parquet(&batch, EnabledStatistics::Chunk);
        let schema = table.schema().clone();

        table.row_groups_mut().clear();

        assert_eq!(table.schema(), &schema);
        assert_eq!(table.schema().fields().len(), 2);
    }

    /// All data pages dictionary-encoded → prunable.
    #[test]
    fn all_dict_when_data_pages_are_dictionary() {
        let stats = [
            enc_stat(PageType::DICTIONARY_PAGE, Encoding::PLAIN_DICTIONARY, 1),
            enc_stat(PageType::DATA_PAGE, Encoding::PLAIN_DICTIONARY, 3),
        ];
        assert!(data_pages_all_dictionary(Some(&stats)));
    }

    /// A single non-dictionary (PLAIN) data page disables pruning, even
    /// alongside dictionary-encoded data pages.
    #[test]
    fn not_all_dict_when_a_data_page_is_plain() {
        let stats = [
            enc_stat(PageType::DICTIONARY_PAGE, Encoding::PLAIN, 1),
            enc_stat(PageType::DATA_PAGE, Encoding::PLAIN_DICTIONARY, 2),
            enc_stat(PageType::DATA_PAGE, Encoding::PLAIN, 1),
        ];
        assert!(!data_pages_all_dictionary(Some(&stats)));
    }

    /// Missing encoding stats are not proof of anything → not prunable.
    #[test]
    fn not_all_dict_when_stats_absent() {
        assert!(!data_pages_all_dictionary(None));
    }

    /// A dictionary page with no data pages is not prunable (nothing to prove).
    #[test]
    fn not_all_dict_when_no_data_pages() {
        let stats = [enc_stat(
            PageType::DICTIONARY_PAGE,
            Encoding::PLAIN_DICTIONARY,
            1,
        )];
        assert!(!data_pages_all_dictionary(Some(&stats)));
    }

    /// End-to-end: write a parquet file with several column types, then read it
    /// back via `ParquetTable::from_files` and verify the decoded min/max
    /// stats match the values we wrote.
    #[test]
    fn decodes_min_max_from_real_parquet_file() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i32", DataType::Int32, false),
            Field::new("i64", DataType::Int64, false),
            Field::new("f64", DataType::Float64, false),
            Field::new("name", DataType::Utf8View, false),
            Field::new("flag", DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![3, -7, 42, 0])),
                Arc::new(Int64Array::from(vec![100i64, -200, 300, 50])),
                Arc::new(Float64Array::from(vec![1.5, -1.25, 3.75, 0.0])),
                Arc::new(StringViewArray::from(vec![
                    "banana", "apple", "zebra", "kiwi",
                ])),
                Arc::new(BooleanArray::from(vec![true, false, true, false])),
            ],
        )
        .unwrap();

        let (_dir, table) = write_parquet(&batch, EnabledStatistics::Chunk);

        let i32_stats = col_stats(&table, 0);
        let i32_min = bound(i32_stats.min());
        let i32_max = bound(i32_stats.max());
        assert_eq!(
            i32_min
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            -7
        );
        assert_eq!(
            i32_max
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            42
        );

        let i64_stats = col_stats(&table, 1);
        let i64_min = bound(i64_stats.min());
        let i64_max = bound(i64_stats.max());
        assert_eq!(
            i64_min
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            -200
        );
        assert_eq!(
            i64_max
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            300
        );

        let f64_stats = col_stats(&table, 2);
        let f64_min = bound(f64_stats.min());
        let f64_max = bound(f64_stats.max());
        assert_eq!(
            f64_min
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(0),
            -1.25
        );
        assert_eq!(
            f64_max
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(0),
            3.75
        );

        let name_stats = col_stats(&table, 3);
        let name_min = bound(name_stats.min());
        let name_max = bound(name_stats.max());
        assert_eq!(
            name_min
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .value(0),
            "apple"
        );
        assert_eq!(
            name_max
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .value(0),
            "zebra"
        );

        let bool_stats = col_stats(&table, 4);
        let bool_min = bound(bool_stats.min());
        let bool_max = bound(bool_stats.max());
        assert!(
            !bool_min
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(0)
        );
        assert!(
            bool_max
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(0)
        );
    }

    #[test]
    fn row_groups_share_file_wide_bound_arrays() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1, 2, 10, 20, 100, 200]))],
        )
        .unwrap();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("groups.parquet");
        let props = WriterProperties::builder()
            .set_statistics_enabled(EnabledStatistics::Chunk)
            .set_max_row_group_row_count(Some(2))
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let table = ParquetTable::from_files(test_dispatcher(), &[path], &[]).unwrap();

        assert!(
            table
                .row_groups
                .windows(2)
                .all(|pair| Arc::ptr_eq(&pair[0].statistics, &pair[1].statistics))
        );
        let file_stats = table.row_groups[0].statistics[0].as_ref().unwrap();
        assert_eq!(file_stats.min.as_ref().unwrap().len(), 3);
        assert_eq!(file_stats.max.as_ref().unwrap().len(), 3);

        let bounds: Vec<(i64, i64)> = table
            .row_groups
            .iter()
            .map(|rg| {
                let stats = rg.leaf_statistics(0).expect("stats should be present");
                let read = |array: ArrayRef| {
                    array
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0)
                };
                (read(bound(stats.min())), read(bound(stats.max())))
            })
            .collect();
        assert_eq!(bounds, vec![(1, 2), (10, 20), (100, 200)]);
    }

    #[test]
    fn no_statistics_when_writer_disables_them() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();

        let (_dir, table) = write_parquet(&batch, EnabledStatistics::None);

        assert!(table.row_groups[0].leaf_statistics(0).is_none());
    }

    fn declared_by_field_id(columns: Vec<(&str, Type, i32)>) -> TableColumns {
        let (columns, field_ids): (Vec<Column>, Vec<i32>) = columns
            .into_iter()
            .map(|(name, col_type, field_id)| (column(name, col_type), field_id))
            .unzip();
        TableColumns::by_field_id(columns, field_ids)
    }

    fn field_names(schema: &Schema) -> Vec<&str> {
        schema.fields().iter().map(|f| f.name().as_str()).collect()
    }

    #[test]
    fn field_ids_rename_and_reorder_file_columns() {
        let file = Schema::new(vec![
            Field::new("b_old", DataType::Int64, true),
            Field::new("a_old", DataType::Int32, false),
        ]);
        let declared = declared_by_field_id(vec![("a", Type::Int32, 1), ("b", Type::Int64, 2)]);

        let layout = resolve_column_layout(file, &[Some(2), Some(1)], &declared).unwrap();

        assert_eq!(field_names(&layout.schema), ["a", "b"]);
        assert_eq!(
            layout.leaf_sources,
            [LeafSource::File(1), LeafSource::File(0)]
        );
    }

    #[test]
    fn a_declared_column_the_file_lacks_is_an_absent_nullable_leaf() {
        let file = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let declared = declared_by_field_id(vec![("a", Type::Int32, 1), ("c", Type::Utf8, 3)]);

        let layout = resolve_column_layout(file, &[Some(1)], &declared).unwrap();

        assert_eq!(field_names(&layout.schema), ["a", "c"]);
        assert_eq!(*layout.schema.field(1).data_type(), DataType::Utf8View);
        assert!(layout.schema.field(1).is_nullable());
        assert_eq!(
            layout.leaf_sources,
            [LeafSource::File(0), LeafSource::Absent]
        );
    }

    #[test]
    fn a_file_column_past_the_declared_ones_has_no_chunk_in_the_row_group() {
        let field_id = |id: &str| {
            std::collections::HashMap::from([("PARQUET:field_id".to_string(), id.to_string())])
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false).with_metadata(field_id("1")),
            Field::new("lineage", DataType::Int64, true).with_metadata(field_id("9")),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int64Array::from(vec![10, 20])),
            ],
        )
        .unwrap();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data.parquet");
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file = DataFile::local(path.clone(), fs::metadata(&path).unwrap().len());

        let loaded = crate::metadata::load_file_row_groups(
            test_dispatcher(),
            &[file],
            declared_by_field_id(vec![("a", Type::Int32, 1)]),
        )
        .unwrap();

        let row_group = &loaded[0].row_groups[0];
        assert_eq!(field_names(&row_group.schema), ["a"]);
        assert_eq!(row_group.columns.len(), 1);
    }

    #[test]
    fn an_undeclared_file_column_is_dropped() {
        let file = Schema::new(vec![
            Field::new("retired", DataType::Int64, true),
            Field::new("a", DataType::Int32, false),
        ]);
        let declared = declared_by_field_id(vec![("a", Type::Int32, 1)]);

        let layout = resolve_column_layout(file, &[Some(9), Some(1)], &declared).unwrap();

        assert_eq!(field_names(&layout.schema), ["a"]);
        assert_eq!(layout.leaf_sources, [LeafSource::File(1)]);
    }

    #[test]
    fn a_file_without_field_ids_matches_by_name() {
        let file = Schema::new(vec![
            Field::new("B", DataType::Int64, true),
            Field::new("a", DataType::Int32, false),
        ]);
        let declared = declared_by_field_id(vec![("a", Type::Int32, 1), ("b", Type::Int64, 2)]);

        let layout = resolve_column_layout(file, &[None, None], &declared).unwrap();

        assert_eq!(field_names(&layout.schema), ["a", "b"]);
        assert_eq!(
            layout.leaf_sources,
            [LeafSource::File(1), LeafSource::File(0)]
        );
    }

    #[test]
    fn duplicate_field_ids_in_a_file_are_rejected() {
        let file = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("a_copy", DataType::Int32, false),
        ]);
        let declared = declared_by_field_id(vec![("a", Type::Int32, 1)]);

        let error = resolve_column_layout(file, &[Some(1), Some(1)], &declared).unwrap_err();

        assert!(matches!(error, Error::InvalidFooter(_)), "{error}");
    }

    #[test]
    fn an_absent_variant_column_spans_absent_leaves() {
        let file = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let declared = declared_by_field_id(vec![("a", Type::Int32, 1), ("v", Type::Variant, 2)]);

        let layout = resolve_column_layout(file, &[Some(1)], &declared).unwrap();

        assert_eq!(field_names(&layout.schema), ["a", "v"]);
        assert!(layout.schema.field(1).is_nullable());
        assert_eq!(
            layout.leaf_sources,
            [LeafSource::File(0), LeafSource::Absent, LeafSource::Absent]
        );
    }

    /// A file whose columns carry field ids, written in an order that differs
    /// from the declared one and lacking a declared scalar and a declared
    /// variant column.
    fn write_evolved_file(dir: &TempDir) -> Arc<ParquetTable> {
        let with_id = |field: Field, id: i32| {
            field.with_metadata(std::collections::HashMap::from([(
                "PARQUET:field_id".to_string(),
                id.to_string(),
            )]))
        };
        let schema = Arc::new(Schema::new(vec![
            with_id(Field::new("b_old", DataType::Int64, true), 2),
            with_id(Field::new("a_old", DataType::Int32, false), 1),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![10, 20, 30])),
                Arc::new(Int32Array::from(vec![1, 2, 3])),
            ],
        )
        .unwrap();
        let path = dir.path().join("evolved.parquet");
        let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let size = std::fs::metadata(&path).unwrap().len();

        let declared = declared_by_field_id(vec![
            ("a", Type::Int32, 1),
            ("c", Type::Int64, 3),
            ("b", Type::Int64, 2),
            ("v", Type::Variant, 4),
        ]);
        let loaded = crate::load_file_row_groups(
            test_dispatcher(),
            &[DataFile::local(path, size)],
            declared.clone(),
        )
        .unwrap();
        let row_groups = loaded
            .into_iter()
            .flat_map(|file| file.row_groups)
            .collect();
        Arc::new(ParquetTable::new(row_groups))
    }

    fn scan(table: &Arc<ParquetTable>, column_indices: Vec<usize>) -> RecordBatch {
        let projection = dispatch::Projection {
            column_indices,
            extracts: Vec::new(),
        };
        let batches = crate::table_input(test_dispatcher(), table, projection, false)
            .collect()
            .unwrap();
        arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    #[test]
    fn a_file_scans_in_the_declared_layout_with_absent_columns_null() {
        let dir = TempDir::new().unwrap();
        let table = write_evolved_file(&dir);

        let batch = scan(&table, vec![0, 1, 2]);

        assert_eq!(field_names(batch.schema().as_ref()), ["a", "c", "b"]);
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[1, 2, 3]
        );
        assert_eq!(batch.column(1).null_count(), 3);
        assert_eq!(
            batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[10, 20, 30]
        );
    }

    #[test]
    fn an_absent_variant_column_scans_as_null_rows() {
        let dir = TempDir::new().unwrap();
        let table = write_evolved_file(&dir);

        let batch = scan(&table, vec![3]);

        let variant = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap();
        assert_eq!(variant.len(), 3);
        assert_eq!(variant.null_count(), 3);
        assert_eq!(variant.column(0).null_count(), 3);
    }

    #[test]
    fn an_extract_from_an_absent_variant_column_is_null() {
        let dir = TempDir::new().unwrap();
        let table = write_evolved_file(&dir);
        let projection = dispatch::Projection {
            column_indices: vec![3, 3],
            extracts: vec![
                Some(dispatch::VariantExtract {
                    path: vec!["x".to_string()],
                    as_type: Some(DataType::Int64),
                }),
                Some(dispatch::VariantExtract {
                    path: vec!["x".to_string()],
                    as_type: None,
                }),
            ],
        };

        let batches = crate::table_input(test_dispatcher(), &table, projection, false)
            .collect()
            .unwrap();
        let batch = arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.column(0).data_type(), &DataType::Int64);
        assert_eq!(batch.column(0).null_count(), 3);
        assert!(matches!(batch.column(1).data_type(), DataType::Struct(_)));
        assert_eq!(batch.column(1).null_count(), 3);
    }

    #[test]
    fn a_projection_of_only_absent_columns_yields_null_rows_without_reading() {
        let dir = TempDir::new().unwrap();
        let table = write_evolved_file(&dir);
        let projection = dispatch::Projection {
            column_indices: vec![1],
            extracts: Vec::new(),
        };

        let request = crate::types::requests::RowGroupRequest::from(
            crate::types::metadata::QueryRowGroupMetadata::new(
                &table,
                0,
                crate::types::metadata::RowSelection::All,
            ),
            &projection,
        );
        let batch = scan(&table, vec![1]);

        assert!(request.reads_nothing());
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.column(0).null_count(), 3);
    }

    /// A scan leaves a row group's pages in the decompressed cache, so a rescan
    /// plans them as already resolved and reads only what is missing; the rows
    /// it decodes from those pages are the ones the first scan read.
    #[test]
    fn a_rescan_decodes_the_same_rows_from_already_decompressed_pages() {
        let batch = RecordBatch::try_from_iter([(
            "v",
            Arc::new(Int64Array::from_iter_values(0..20_000)) as _,
        )])
        .unwrap();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("compressed.parquet");
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props))
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let table = Arc::new(ParquetTable::from_files(test_dispatcher(), &[path], &[]).unwrap());
        let first = scan(&table, vec![0]);

        let second = scan(&table, vec![0]);

        assert_eq!(first.column(0).as_ref(), batch.column(0).as_ref());
        assert_eq!(second.column(0).as_ref(), first.column(0).as_ref());
    }

    #[test]
    fn an_absent_column_reports_every_row_null_in_its_statistics() {
        let dir = TempDir::new().unwrap();
        let table = write_evolved_file(&dir);

        let stats = table.row_groups[0].column_statistics(1).unwrap();

        assert_eq!(stats.null_count, Some(3));
        assert!(stats.min().is_none() && stats.max().is_none());
    }
}
