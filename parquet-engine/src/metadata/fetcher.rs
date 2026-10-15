//! Fetch stage for Parquet footers.
//!
//! The operator submits logical file ranges through `OperatorIO`; dispatch's
//! requester's tracker resolves cache hits and allocates missing compressed-cache
//! extents, then the requester immediately performs the physical reads. This
//! module owns only the footer-specific two-step state machine: probe the tail,
//! then request the exact footer when it is larger than the probe.

use super::FileRowGroups;
use crate::types::columns::TableColumns;
use crate::types::metadata::RowGroupMetadata;
use crate::types::table::{Error, FOOTER_PROBE_BYTES, Result, row_groups_from_footer};
use dispatch::io::{FileRange, OpenFile, OperatorIO, ReadRequestId, ReadResponse};
use dispatch::memory::memory_ctx;
use dispatch::{Sender, Unary};
use object_storage::{DataFile, FileRef};
use std::collections::HashMap;
use std::sync::Arc;

const PARQUET_MAGIC: [u8; 4] = *b"PAR1";

/// Local footer reads outstanding before another file is admitted. Footers are
/// small and scattered, so modest concurrency hides per-file latency.
const MAX_DISK_IN_FLIGHT: usize = 32;

/// Parse the 4-byte footer length from a file's `tail` (whose final 8 bytes are
/// `[footer_len][PAR1]`).
pub(super) fn footer_len_from_tail(tail: &[u8]) -> Result<usize> {
    if tail.len() < 8 || tail[tail.len() - 4..] != PARQUET_MAGIC {
        return Err(Error::InvalidFooter("missing PAR1 magic".to_string()));
    }
    let len = &tail[tail.len() - 8..tail.len() - 4];
    Ok(u32::from_le_bytes(len.try_into().unwrap()) as usize)
}

pub(super) struct FileRowGroupsFetcher {
    /// Footer domain state keyed by the node-local logical request id.
    in_flight: HashMap<ReadRequestId, FooterRead>,
    table_columns: TableColumns,
}

impl FileRowGroupsFetcher {
    pub(super) fn new(table_columns: TableColumns) -> Self {
        Self {
            in_flight: HashMap::new(),
            table_columns,
        }
    }

    fn request_region(
        &mut self,
        request: FooterRead,
        range: FileRange,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let id = io.read(request.open_file.clone(), [range])?;
        self.in_flight.insert(id, request);
        Ok(())
    }
}

impl Unary<DataFile, FileRowGroups> for FileRowGroupsFetcher {
    fn consume(
        &mut self,
        file: DataFile,
        _sender: &mut dyn Sender<FileRowGroups>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let DataFile {
            file: file_ref,
            source,
        } = file;
        let size = file_ref.size as usize;
        let open_file = source.open_read(file_ref.size).map_err(crate::op_err)?;

        // This is the one registration point for a newly-opened descriptor.
        // It prevents stale cache entries from an equal, reused fd serving the
        // new file. Later logical ranges must not call `open_entry`, because it
        // intentionally clears the file's old extent map.
        memory_ctx()
            .compressed_cache()
            .open_entry(open_file.clone());

        let request = FooterRead::new(file_ref, open_file, size);
        let range = request.probe_range();
        self.request_region(request, range, io)
    }

    fn ready_for_more_work(&mut self) -> bool {
        let disk_in_flight = self
            .in_flight
            .values()
            .filter(|request| matches!(request.open_file, OpenFile::Local(_)))
            .count();
        let http_in_flight = self.in_flight.len() - disk_in_flight;
        disk_in_flight < MAX_DISK_IN_FLIGHT && http_in_flight < crate::http_readahead()
    }

    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<FileRowGroups>,
        io: &mut OperatorIO,
        response: ReadResponse,
    ) -> dispatch::UnaryResult<()> {
        let mut request = self
            .in_flight
            .remove(&response.id())
            .expect("response for an unknown footer request");
        let bytes = response.into_bytes();

        match request
            .parse_region(&bytes, &self.table_columns)
            .map_err(crate::op_err)?
        {
            FooterProgress::ReadExact(range) => self.request_region(request, range, io)?,
            FooterProgress::Done(row_groups) => sender.send(FileRowGroups {
                file: request.file,
                row_groups: row_groups.into_iter().map(Arc::new).collect(),
            })?,
        }
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<FileRowGroups>) -> dispatch::UnaryResult<bool> {
        Ok(self.in_flight.is_empty())
    }
}

/// Domain state for one file's footer read. Physical block accounting lives in
/// dispatch; this remembers only how the next completed byte range is parsed.
struct FooterRead {
    file: FileRef,
    open_file: OpenFile,
    size: usize,
    reading_exact: bool,
}

impl FooterRead {
    fn new(file: FileRef, open_file: OpenFile, size: usize) -> Self {
        Self {
            file,
            open_file,
            size,
            reading_exact: false,
        }
    }

    fn probe_range(&self) -> FileRange {
        let probe = self.size.min(FOOTER_PROBE_BYTES);
        FileRange::new(self.size - probe, probe)
    }

    fn parse_region(
        &mut self,
        bytes: &[u8],
        table_columns: &TableColumns,
    ) -> Result<FooterProgress> {
        let footer = if self.reading_exact {
            bytes
        } else {
            let footer_len = footer_len_from_tail(bytes)?;
            if footer_len + 8 > self.size {
                return Err(Error::InvalidFooter(format!(
                    "footer length {footer_len} exceeds file size {}",
                    self.size
                )));
            }
            if footer_len + 8 > bytes.len() {
                self.reading_exact = true;
                return Ok(FooterProgress::ReadExact(FileRange::new(
                    self.size - 8 - footer_len,
                    footer_len,
                )));
            }
            let start = bytes.len() - 8 - footer_len;
            &bytes[start..bytes.len() - 8]
        };

        Ok(FooterProgress::Done(row_groups_from_footer(
            footer,
            self.open_file.clone(),
            table_columns,
        )?))
    }
}

enum FooterProgress {
    ReadExact(FileRange),
    Done(Vec<RowGroupMetadata>),
}
