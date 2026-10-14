//! The PostgreSQL text format of `COPY ... FROM STDIN`: one row per line,
//! fields separated by a one-byte delimiter (tab unless the statement says
//! otherwise), NULL spelled `\N`, and backslash escapes for the bytes that
//! would otherwise end a field or a row. A line holding only `\.` ends the
//! data.
//!
//! The stream is handled in two halves. A [`TextFrameDecoder`] runs where
//! the bytes arrive and cuts them into chunks of whole rows, so that the
//! rest of the work spreads over workers: each [`TextRows`] chunk parses
//! into text columns. The one rule both halves share is where a row ends: at
//! a newline, unless a backslash escapes it.

use crate::operator::copy_from_stdin::{CopyChunk, FrameDecoder};
use arrow_array::builder::StringBuilder;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use bytes::{Bytes, BytesMut};
use std::sync::Arc;

/// The delimiter and NULL spelling of a text-format copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextFormat {
    delimiter: u8,
    null: Vec<u8>,
}

impl Default for TextFormat {
    fn default() -> Self {
        Self {
            delimiter: b'\t',
            null: b"\\N".to_vec(),
        }
    }
}

impl TextFormat {
    /// Apply a statement's `WITH (...)` options (names as written, each
    /// with its bound constant values) to the format's defaults.
    pub fn from_options(options: &[(String, Vec<String>)]) -> Result<Self, String> {
        let mut format = Self::default();
        for (name, values) in options {
            let name = name.to_lowercase();
            match name.as_str() {
                "delimiter" => {
                    let value = single_option_value(&name, values)?;
                    let [byte] = value.as_bytes() else {
                        return Err(
                            "COPY delimiter must be a single one-byte character".to_string()
                        );
                    };
                    if matches!(byte, b'\n' | b'\r') {
                        return Err(
                            "COPY delimiter cannot be newline or carriage return".to_string()
                        );
                    }
                    // These would be mistaken for an escape sequence or the
                    // end-of-data marker.
                    if matches!(byte, b'\\' | b'.' | b'a'..=b'z' | b'0'..=b'9') {
                        return Err(format!("COPY delimiter cannot be \"{value}\""));
                    }
                    format.delimiter = *byte;
                }
                "null" => {
                    let value = single_option_value(&name, values)?;
                    if value.contains(['\n', '\r']) {
                        return Err(
                            "COPY null representation cannot use newline or carriage return"
                                .to_string(),
                        );
                    }
                    format.null = value.into_bytes();
                }
                _ => {
                    return Err(format!(
                        "COPY option \"{name}\" is not valid for FORMAT text"
                    ));
                }
            }
        }
        if format.null.contains(&format.delimiter) {
            return Err("COPY delimiter must not appear in the NULL specification".to_string());
        }
        Ok(format)
    }
}

fn single_option_value(name: &str, values: &[String]) -> Result<String, String> {
    match values {
        [value] => Ok(value.clone()),
        _ => Err(format!("COPY option \"{name}\" requires a single value")),
    }
}

/// Bytes of whole rows a chunk gathers before it is handed to a worker.
const CHUNK_BYTES: usize = 1 << 20;

/// Whether the byte after `line` is escaped: an odd run of backslashes ends
/// the line, and the last one escapes what follows.
fn escapes_next_byte(line: &[u8]) -> bool {
    line.iter().rev().take_while(|&&byte| byte == b'\\').count() % 2 == 1
}

/// Whether a line (without its newline) is the end-of-data marker.
fn is_end_marker(line: &[u8]) -> bool {
    line == b"\\." || line == b"\\.\r"
}

/// The sequential half of a text copy: takes the stream's bytes as they
/// arrive and cuts them at row ends into chunks for the workers, carrying
/// a row in progress across pushes. Stops taking bytes at the end-of-data
/// marker.
#[derive(Debug, Default)]
pub struct TextRowFramer {
    /// Bytes received and not yet handed out.
    buffer: BytesMut,
    /// `buffer[..rows_end]` holds only whole rows, each with its newline.
    rows_end: usize,
    /// `buffer[..searched]` has been searched for a row end; the row in
    /// progress is `buffer[rows_end..]`.
    searched: usize,
    /// The end-of-data marker was seen; whatever follows is ignored.
    ended: bool,
}

impl TextRowFramer {
    /// Append bytes from the stream and find the rows they complete.
    pub fn push(&mut self, bytes: &[u8]) {
        if self.ended {
            return;
        }
        self.buffer.extend_from_slice(bytes);
        while let Some(offset) = memchr::memchr(b'\n', &self.buffer[self.searched..]) {
            let newline = self.searched + offset;
            self.searched = newline + 1;
            let line = &self.buffer[self.rows_end..newline];
            if escapes_next_byte(line) {
                continue;
            }
            if is_end_marker(line) {
                self.ended = true;
                self.buffer.truncate(self.rows_end);
                self.searched = self.rows_end;
                return;
            }
            self.rows_end = newline + 1;
        }
        self.searched = self.buffer.len();
    }

    /// A chunk of whole rows, once enough have gathered to be worth a
    /// worker's trip.
    pub fn next_chunk(&mut self) -> Option<Bytes> {
        (self.rows_end >= CHUNK_BYTES).then(|| self.take_rows())
    }

    /// The stream is over: the whole rows still held, plus a last row the
    /// client sent without a trailing newline.
    pub fn finish(mut self) -> Option<Bytes> {
        let unterminated = &self.buffer[self.rows_end..];
        if is_end_marker(unterminated) {
            self.buffer.truncate(self.rows_end);
        } else {
            self.rows_end = self.buffer.len();
        }
        self.searched = self.rows_end;
        (self.rows_end > 0).then(|| self.take_rows())
    }

    fn take_rows(&mut self) -> Bytes {
        let rows = self.buffer.split_to(self.rows_end).freeze();
        self.searched -= self.rows_end;
        self.rows_end = 0;
        rows
    }
}

/// The rows of a chunk, without their newlines. A chunk normally ends with
/// a newline; bytes after the last one are a final row only when there are
/// any.
fn split_rows(chunk: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut remaining = chunk;
    std::iter::from_fn(move || {
        if remaining.is_empty() {
            return None;
        }
        let mut searched = 0;
        while let Some(offset) = memchr::memchr(b'\n', &remaining[searched..]) {
            let newline = searched + offset;
            if escapes_next_byte(&remaining[..newline]) {
                searched = newline + 1;
                continue;
            }
            let row = &remaining[..newline];
            remaining = &remaining[newline + 1..];
            return Some(row);
        }
        Some(std::mem::take(&mut remaining))
    })
}

/// A row without the carriage return of a `\r\n` line ending. An escaped
/// carriage return is data and stays.
fn strip_carriage_return(row: &[u8]) -> &[u8] {
    match row.split_last() {
        Some((b'\r', rest)) if !escapes_next_byte(rest) => rest,
        _ => row,
    }
}

/// The raw (still escaped) fields of a row.
fn split_fields(row: &[u8], delimiter: u8) -> impl Iterator<Item = &[u8]> {
    let mut remaining = Some(row);
    std::iter::from_fn(move || {
        let row = remaining?;
        let mut searched = 0;
        while let Some(offset) = row
            .get(searched..)
            .and_then(|rest| memchr::memchr2(delimiter, b'\\', rest))
        {
            let found = searched + offset;
            if row[found] == b'\\' {
                // The backslash escapes the byte after it.
                searched = found + 2;
                continue;
            }
            remaining = Some(&row[found + 1..]);
            return Some(&row[..found]);
        }
        remaining = None;
        Some(row)
    })
}

/// Resolve a raw field's backslash escapes into `out`: the C-style letters,
/// `\` plus up to three octal digits, `\x` plus up to two hex digits, and
/// any other byte standing for itself. `\.` is the end-of-data marker, which
/// is only valid as a whole line and never reaches here as one.
fn unescape_field(raw: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    let mut i = 0;
    while i < raw.len() {
        let byte = raw[i];
        i += 1;
        if byte != b'\\' {
            out.push(byte);
            continue;
        }
        let Some(&escaped) = raw.get(i) else {
            return Err("COPY text row ends with an unfinished backslash escape".to_string());
        };
        i += 1;
        let resolved = match escaped {
            b'b' => 0x08,
            b'f' => 0x0c,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'v' => 0x0b,
            b'.' => {
                return Err("end-of-copy marker is not alone on its line".to_string());
            }
            b'0'..=b'7' => {
                let mut value = u32::from(escaped - b'0');
                for _ in 0..2 {
                    match raw.get(i) {
                        Some(digit @ b'0'..=b'7') => {
                            value = value * 8 + u32::from(digit - b'0');
                            i += 1;
                        }
                        _ => break,
                    }
                }
                (value & 0xff) as u8
            }
            b'x' if raw.get(i).is_some_and(u8::is_ascii_hexdigit) => {
                let mut value = 0u8;
                for _ in 0..2 {
                    match raw.get(i) {
                        Some(digit) if digit.is_ascii_hexdigit() => {
                            value = value * 16 + hex_digit_value(*digit);
                            i += 1;
                        }
                        _ => break,
                    }
                }
                value
            }
            other => other,
        };
        out.push(resolved);
    }
    Ok(())
}

fn hex_digit_value(digit: u8) -> u8 {
    match digit {
        b'0'..=b'9' => digit - b'0',
        b'a'..=b'f' => digit - b'a' + 10,
        _ => digit - b'A' + 10,
    }
}

/// Parse one chunk of whole rows into a batch of `schema`, one text column
/// per field the rows must carry. A field spelled exactly as the NULL
/// representation (before its escapes resolve) is NULL. Every row must
/// have exactly the schema's width.
pub fn parse_text_rows(
    chunk: &[u8],
    format: &TextFormat,
    schema: &SchemaRef,
) -> Result<RecordBatch, String> {
    let mut builders: Vec<StringBuilder> = schema
        .fields()
        .iter()
        .map(|_| StringBuilder::new())
        .collect();
    let mut unescaped = Vec::new();
    for row in split_rows(chunk) {
        let row = strip_carriage_return(row);
        let mut fields = split_fields(row, format.delimiter);
        for (builder, field) in builders.iter_mut().zip(schema.fields()) {
            let raw = fields
                .next()
                .ok_or_else(|| format!("missing data for column \"{}\"", field.name()))?;
            if raw == format.null.as_slice() {
                builder.append_null();
                continue;
            }
            let value = if memchr::memchr(b'\\', raw).is_some() {
                unescaped.clear();
                unescape_field(raw, &mut unescaped)?;
                unescaped.as_slice()
            } else {
                raw
            };
            let text = std::str::from_utf8(value).map_err(|_| {
                format!(
                    "column \"{}\": COPY text data is not valid UTF-8",
                    field.name()
                )
            })?;
            builder.append_value(text);
        }
        if fields.next().is_some() {
            return Err("extra data after last expected column".to_string());
        }
    }
    let columns: Vec<ArrayRef> = builders
        .into_iter()
        .map(|mut builder| Arc::new(builder.finish()) as ArrayRef)
        .collect();
    RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| format!("assembling COPY text rows: {e}"))
}

/// The text format's sequential half: frames in, chunks of whole rows out.
pub(crate) struct TextFrameDecoder {
    framer: TextRowFramer,
    format: Arc<TextFormat>,
    /// One text column per field a row must carry, in the statement's order.
    schema: SchemaRef,
}

impl TextFrameDecoder {
    pub(crate) fn new(format: TextFormat, schema: SchemaRef) -> Self {
        Self {
            framer: TextRowFramer::default(),
            format: Arc::new(format),
            schema,
        }
    }

    fn wrap_rows(&self, rows: Bytes) -> Box<dyn CopyChunk> {
        Box::new(TextRows {
            rows,
            format: self.format.clone(),
            schema: self.schema.clone(),
        })
    }
}

impl FrameDecoder for TextFrameDecoder {
    fn push_frame(&mut self, frame: Bytes) {
        self.framer.push(&frame);
    }

    fn next_chunk(&mut self) -> Result<Option<Box<dyn CopyChunk>>, String> {
        Ok(self.framer.next_chunk().map(|rows| self.wrap_rows(rows)))
    }

    /// Whole rows went out as they completed; a last row without a trailing
    /// newline is still data.
    fn finish(&mut self) -> Result<Option<Box<dyn CopyChunk>>, String> {
        let framer = std::mem::take(&mut self.framer);
        Ok(framer.finish().map(|rows| self.wrap_rows(rows)))
    }
}

/// Whole text rows, still to be parsed.
struct TextRows {
    rows: Bytes,
    format: Arc<TextFormat>,
    schema: SchemaRef,
}

impl CopyChunk for TextRows {
    fn into_batch(self: Box<Self>) -> Result<RecordBatch, String> {
        parse_text_rows(&self.rows, &self.format, &self.schema)
    }
}

#[cfg(test)]
mod tests {
    use super::{TextFormat, TextRowFramer, parse_text_rows};
    use arrow_array::cast::AsArray;
    use arrow_array::{Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use std::sync::Arc;

    fn text_schema(names: &[&str]) -> SchemaRef {
        Arc::new(Schema::new(
            names
                .iter()
                .map(|name| Field::new(*name, DataType::Utf8, true))
                .collect::<Vec<_>>(),
        ))
    }

    fn cells(batch: &RecordBatch) -> Vec<Vec<Option<String>>> {
        (0..batch.num_rows())
            .map(|row| {
                batch
                    .columns()
                    .iter()
                    .map(|column| {
                        let strings = column.as_string::<i32>();
                        strings
                            .is_valid(row)
                            .then(|| strings.value(row).to_string())
                    })
                    .collect()
            })
            .collect()
    }

    fn options(pairs: &[(&str, &[&str])]) -> Vec<(String, Vec<String>)> {
        pairs
            .iter()
            .map(|(name, values)| {
                (
                    name.to_string(),
                    values.iter().map(|value| value.to_string()).collect(),
                )
            })
            .collect()
    }

    /// Push the stream in pieces and gather everything the framer cuts.
    fn frame(pieces: &[&[u8]]) -> Vec<u8> {
        let mut framer = TextRowFramer::default();
        let mut rows = Vec::new();
        for piece in pieces {
            framer.push(piece);
            while let Some(chunk) = framer.next_chunk() {
                rows.extend_from_slice(&chunk);
            }
        }
        if let Some(chunk) = framer.finish() {
            rows.extend_from_slice(&chunk);
        }
        rows
    }

    #[test]
    fn the_framer_cuts_at_unescaped_newlines_only() {
        let rows = frame(&[b"a\\\nb\tc\n", b"d\\\\\ne", b"\n"]);

        assert_eq!(rows, b"a\\\nb\tc\nd\\\\\ne\n");
    }

    #[test]
    fn the_framer_stops_at_the_end_marker_and_keeps_an_unterminated_last_row() {
        let with_marker = frame(&[b"1\n\\.\r\n2\n"]);
        let unterminated = frame(&[b"1\n", b"2"]);
        let marker_unterminated = frame(&[b"1\n\\."]);

        assert_eq!(with_marker, b"1\n");
        assert_eq!(unterminated, b"1\n2");
        assert_eq!(marker_unterminated, b"1\n");
    }

    #[test]
    fn rows_parse_with_nulls_escapes_and_line_endings() {
        let schema = text_schema(&["a", "b"]);
        let chunk = b"1\t\\N\r\nx\\ty\\\\z\t\\\\N\n\\101\\x41\\x4g\t\\\n\n\t\n";

        let batch = parse_text_rows(chunk, &TextFormat::default(), &schema).unwrap();

        assert_eq!(
            cells(&batch),
            vec![
                vec![Some("1".into()), None],
                vec![Some("x\ty\\z".into()), Some("\\N".into())],
                vec![Some("AA\x04g".into()), Some("\n".into())],
                vec![Some("".into()), Some("".into())],
            ]
        );
    }

    #[test]
    fn options_change_the_delimiter_and_null_spelling() {
        let format =
            TextFormat::from_options(&options(&[("delimiter", &["|"]), ("null", &[""])])).unwrap();
        let schema = text_schema(&["a", "b"]);

        let batch = parse_text_rows(b"\\N|\n\\||x\n", &format, &schema).unwrap();

        assert_eq!(
            cells(&batch),
            vec![
                vec![Some("N".into()), None],
                vec![Some("|".into()), Some("x".into())],
            ]
        );
    }

    #[test]
    fn malformed_rows_and_options_are_refused_by_name() {
        let schema = text_schema(&["a", "b"]);
        let format = TextFormat::default();

        let short = parse_text_rows(b"1\n", &format, &schema).unwrap_err();
        let long = parse_text_rows(b"1\t2\t3\n", &format, &schema).unwrap_err();
        let dangling = parse_text_rows(b"1\t2\\", &format, &schema).unwrap_err();
        let not_utf8 = parse_text_rows(b"1\t\\xff\n", &format, &schema).unwrap_err();
        let two_bytes = TextFormat::from_options(&options(&[("delimiter", &["ab"])])).unwrap_err();
        let header = TextFormat::from_options(&options(&[("header", &[])])).unwrap_err();
        let overlap = TextFormat::from_options(&options(&[("null", &["\t"])])).unwrap_err();

        assert_eq!(short, "missing data for column \"b\"");
        assert_eq!(long, "extra data after last expected column");
        assert_eq!(
            dangling,
            "COPY text row ends with an unfinished backslash escape"
        );
        assert_eq!(not_utf8, "column \"b\": COPY text data is not valid UTF-8");
        assert_eq!(
            two_bytes,
            "COPY delimiter must be a single one-byte character"
        );
        assert_eq!(
            header,
            "COPY option \"header\" is not valid for FORMAT text"
        );
        assert_eq!(
            overlap,
            "COPY delimiter must not appear in the NULL specification"
        );
    }
}
