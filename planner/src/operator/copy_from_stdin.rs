//! [`CopyFromStdin`] — the `COPY <table> FROM STDIN` statement, and its
//! lowering into an ingest dataflow.

use crate::catalog::{BoundTable, Error as CatalogError};
use crate::expression::json_to_canonical_variant;
use crate::operator::copy_arrow::ArrowFrameDecoder;
use crate::operator::copy_text::{TextFormat, TextFrameDecoder};
use crate::types::{Type, physical_arrow_type, type_from_physical};
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, RecordBatch, StringArray, new_null_array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use dispatch::BuildContext;
use dispatch::{Built, BuiltUnary};
use dispatch::{
    ChannelInputSender, DataFlowDispatcher, RecordBatchOperatorSpec, Unary, UnaryError,
    UnaryFactory, UnaryResult, channel_input, stealable,
};
use std::fmt;
use std::sync::Arc;

/// `COPY <table> [(columns)] FROM STDIN [WITH (FORMAT text | arrow, ...)]`:
/// load rows arriving over the client protocol into a table.
///
/// A statement, not a query: its rows arrive later over the connection's
/// copy-in sub-protocol, so a plan never compiles it. The engine reads it off
/// the plan ([`Plan::as_copy_from_stdin`](crate::Plan::as_copy_from_stdin))
/// and lowers it through [`compile_ingest`](Self::compile_ingest), which must
/// run inside the transaction the statement was planned in: the bound table
/// stages its writes there. Everything knowable without the rows is validated
/// at plan time; the rows themselves are checked per batch as they arrive
/// (width and casts).
#[derive(Debug)]
pub struct CopyFromStdin {
    /// The bound target table, resolved by DuckDB's binder like an INSERT's.
    pub table: Box<dyn BoundTable>,
    /// The explicit column list resolved to table column positions, empty
    /// when the statement targets every column.
    pub columns: Vec<usize>,
    /// The validated data format.
    pub format: CopyFormat,
}

/// The validated data format of a COPY FROM STDIN: PostgreSQL's text format
/// (the protocol's default, what psql sends) or an Arrow IPC stream. Any
/// other format is rejected at plan time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyFormat {
    /// Tab-separated text rows; see [`TextFormat`].
    Text(TextFormat),
    /// An Arrow IPC stream; batches conform to the table schema by position.
    ArrowIpc,
}

impl CopyFormat {
    /// Validate a statement's format name and raw option list (names as
    /// written, each with its bound constant values; a bare flag has none).
    pub fn resolve(
        format: Option<&str>,
        options: &[(String, Vec<String>)],
    ) -> Result<Self, String> {
        match format {
            None | Some("text") => Ok(Self::Text(TextFormat::from_options(options)?)),
            Some("arrow") => {
                if let Some((name, _)) = options.first() {
                    return Err(format!(
                        "COPY option \"{}\" is not valid for FORMAT arrow",
                        name.to_lowercase()
                    ));
                }
                Ok(Self::ArrowIpc)
            }
            Some(other) => Err(format!(
                "COPY FORMAT {other} is not supported; use text or arrow"
            )),
        }
    }
}

/// The sequential half of one copy format: reassembles the protocol's byte
/// frames into chunks for the dataflow's workers. A partial chunk is carried
/// across frames, so a chunk may span any number of frames and one frame may
/// complete several.
///
/// A format is its decoder and the [`CopyChunk`]s it yields, nothing else.
pub trait FrameDecoder: Send {
    /// Take the next protocol frame. The previous frame's chunks have all
    /// been drained through [`next_chunk`](Self::next_chunk) by then.
    fn push_frame(&mut self, frame: Bytes);

    /// The next chunk the frames so far complete. An error means the stream
    /// is malformed.
    fn next_chunk(&mut self) -> Result<Option<Box<dyn CopyChunk>>, String>;

    /// The stream is over: whatever is still held, or an error when it was
    /// cut mid-chunk.
    fn finish(&mut self) -> Result<Option<Box<dyn CopyChunk>>, String>;
}

/// The worker half of one copy format: a unit of the stream that a
/// [`FrameDecoder`] cut, carrying what it takes to finish decoding it.
pub trait CopyChunk: Send {
    /// Decode into a batch of the columns the statement feeds, in the
    /// statement's order.
    fn into_batch(self: Box<Self>) -> Result<RecordBatch, String>;
}

impl Clone for CopyFromStdin {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone_box(),
            columns: self.columns.clone(),
            format: self.format.clone(),
        }
    }
}

impl fmt::Display for CopyFromStdin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reference = self.table.table_reference();
        write!(
            f,
            "CopyFromStdin({}.{}.{}",
            reference.datastore, reference.schema, reference.table
        )?;
        if !self.columns.is_empty() {
            let columns = self.table.columns();
            let names: Vec<String> = self
                .columns
                .iter()
                .map(|&index| columns[index].name.clone())
                .collect();
            write!(f, " ({})", names.join(", "))?;
        }
        match &self.format {
            CopyFormat::Text(_) => write!(f, " format: text")?,
            CopyFormat::ArrowIpc => write!(f, " format: arrow")?,
        }
        write!(f, ")")
    }
}

/// Unclaimed chunks the frontend may queue ahead of the workers.
const CHUNK_QUEUE_CAPACITY: usize = 1048;

/// The statement's column list resolved against the table: the full output
/// schema in table-column order (built once, shared by every conformed
/// batch), and where each output column's data comes from in an incoming
/// batch (`None` for columns the statement does not fill, which become NULL).
#[derive(Debug)]
struct ColumnLayout {
    schema: SchemaRef,
    source_positions: Vec<Option<usize>>,
}

impl ColumnLayout {
    /// Columns each incoming batch must have: the length of the COPY column
    /// list (or the full table width without one). Exactly the table columns
    /// the statement feeds, so it is the count of filled source positions.
    fn incoming_columns(&self) -> usize {
        self.source_positions.iter().flatten().count()
    }

    /// The incoming columns as text, in the statement's order: what a
    /// text-format row parses into before it conforms to the table.
    fn incoming_text_schema(&self) -> SchemaRef {
        let mut names = vec![None; self.incoming_columns()];
        for (field, source) in self.schema.fields().iter().zip(&self.source_positions) {
            if let Some(position) = source {
                names[*position] = Some(field.name().clone());
            }
        }
        let fields: Vec<Field> = names
            .into_iter()
            .map(|name| {
                let name = name.expect("every incoming position feeds one table column");
                Field::new(name, DataType::Utf8, true)
            })
            .collect();
        Arc::new(Schema::new(fields))
    }
}

impl CopyFromStdin {
    /// Lower this statement into its ingest dataflow: a chunk channel fanning
    /// out to per-worker decode-and-conform stages, feeding the bound table's
    /// insert sink (which stages into the transaction the statement was
    /// planned in). `on_claim` fires as workers take chunks, for producer
    /// backpressure. Returns the decoder the frontend runs over the
    /// protocol's frames, the queue its chunks go into, and the dataflow
    /// that drains it, whose single output row is the insert count.
    #[allow(clippy::type_complexity)]
    pub fn compile_ingest(
        &self,
        dispatcher: &DataFlowDispatcher,
        on_claim: Box<dyn Fn() + Send + Sync>,
    ) -> Result<
        (
            Box<dyn FrameDecoder>,
            ChannelInputSender<Box<dyn CopyChunk>>,
            RecordBatchOperatorSpec,
        ),
        CatalogError,
    > {
        let layout = Arc::new(self.build_layout().map_err(|message| {
            CatalogError::Other(Box::<dyn std::error::Error + Send + Sync>::from(message))
        })?);
        let decoder: Box<dyn FrameDecoder> = match &self.format {
            CopyFormat::Text(format) => Box::new(TextFrameDecoder::new(
                format.clone(),
                layout.incoming_text_schema(),
            )),
            CopyFormat::ArrowIpc => Box::new(ArrowFrameDecoder::new()),
        };
        let (sender, chunks) =
            channel_input::<Box<dyn CopyChunk>>(dispatcher, CHUNK_QUEUE_CAPACITY, on_claim);
        let conformers: Vec<_> = (0..dispatcher.worker_count())
            .map(|_| ConformCopyChunk {
                layout: layout.clone(),
            })
            .collect();
        let conformed = chunks
            .chain(
                stealable::<Box<dyn CopyChunk>>(dispatcher.topology())
                    .into_iter()
                    .collect(),
                conformers,
            )
            .record_batches();
        let dataflow = self.table.compile_insert(conformed, dispatcher)?;
        Ok((decoder, sender, dataflow))
    }

    /// Resolve the column list (positions resolved and validated by the
    /// binder) against the table.
    fn build_layout(&self) -> Result<ColumnLayout, String> {
        let table_columns = self.table.columns();
        let fields: Vec<Field> = table_columns
            .iter()
            .map(|column| {
                Field::new(
                    column.name.clone(),
                    physical_arrow_type(&column.col_type),
                    true,
                )
            })
            .collect();
        let mut source_positions: Vec<Option<usize>> = vec![None; table_columns.len()];
        if self.columns.is_empty() {
            for (i, source) in source_positions.iter_mut().enumerate() {
                *source = Some(i);
            }
        } else {
            // The binder guarantees valid, distinct positions; check anyway so
            // a disagreement fails loudly instead of scrambling columns.
            for (position, &index) in self.columns.iter().enumerate() {
                let slot = source_positions
                    .get_mut(index)
                    .ok_or_else(|| format!("COPY column position {index} is out of range"))?;
                if slot.is_some() {
                    return Err(format!("COPY column position {index} listed twice"));
                }
                *slot = Some(position);
            }
        }
        Ok(ColumnLayout {
            schema: Arc::new(Schema::new(fields)),
            source_positions,
        })
    }
}

fn is_text_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

/// Cast one incoming column to its table field's physical type, with a loud
/// error naming the column when a value does not convert. Text into a
/// VARIANT column parses each value as a JSON document, as a cast does, and
/// text into any other non-text column ignores surrounding whitespace.
fn cast_to_field(array: &ArrayRef, field: &Field) -> Result<ArrayRef, String> {
    if field.data_type() == array.data_type() {
        return Ok(array.clone());
    }
    let into_variant = type_from_physical(field.data_type()) == Some(Type::Variant)
        && is_text_type(array.data_type());
    if into_variant {
        return json_to_canonical_variant(array)
            .map_err(|e| format!("column \"{}\": {e}", field.name()));
    }
    let options = arrow::compute::CastOptions {
        safe: false,
        format_options: Default::default(),
    };
    let trimmed: ArrayRef;
    let array = match array.as_string_opt::<i32>() {
        Some(text) if !is_text_type(field.data_type()) => {
            trimmed = Arc::new(StringArray::from_iter(
                text.iter().map(|value| value.map(str::trim)),
            ));
            &trimmed
        }
        _ => array,
    };
    arrow::compute::cast_with_options(array, field.data_type(), &options)
        .map_err(|e| format!("column \"{}\": {e}", field.name()))
}

/// Conform one client-schema batch to the table's physical schema: columns
/// map to the statement's column list by position, cast to the table's
/// physical types, and unlisted table columns fill with NULL. Variant
/// columns arriving as variants are admitted as sent: clients are trusted to
/// encode them correctly, and a malformed document surfaces wherever it is
/// first parsed.
fn conform_batch(batch: &RecordBatch, layout: &ColumnLayout) -> Result<RecordBatch, String> {
    let incoming_columns = layout.incoming_columns();
    if batch.num_columns() != incoming_columns {
        return Err(format!(
            "COPY arrow stream has {} columns, the statement targets {incoming_columns}",
            batch.num_columns(),
        ));
    }
    let arrays = layout
        .schema
        .fields()
        .iter()
        .zip(&layout.source_positions)
        .map(|(field, source)| match source {
            Some(index) => cast_to_field(batch.column(*index), field),
            None => Ok(new_null_array(field.data_type(), batch.num_rows())),
        })
        .collect::<Result<Vec<_>, String>>()?;
    RecordBatch::try_new(layout.schema.clone(), arrays)
        .map_err(|e| format!("assembling COPY batch: {e}"))
}

/// Worker-side transform: one chunk in, one table-schema batch out.
struct ConformCopyChunk {
    layout: Arc<ColumnLayout>,
}

impl Unary<Box<dyn CopyChunk>, RecordBatch> for ConformCopyChunk {
    fn consume(
        &mut self,
        chunk: Box<dyn CopyChunk>,
        sender: &mut dyn dispatch::Sender<RecordBatch>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        let batch = chunk
            .into_batch()
            .and_then(|batch| conform_batch(&batch, &self.layout))
            .map_err(|message| UnaryError::Operator(message.into()))?;
        if batch.num_rows() > 0 {
            sender.send(batch)?;
        }
        Ok(())
    }
}

// Stateless per worker, so it serves as its own factory.
impl UnaryFactory<Box<dyn CopyChunk>, RecordBatch> for ConformCopyChunk {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}
