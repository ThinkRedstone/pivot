//! The Arrow IPC format of `COPY ... FROM STDIN`. An IPC stream reads
//! sequentially, so its batches are decoded where the bytes arrive and the
//! workers only conform them to the table.

use crate::operator::copy_from_stdin::{CopyChunk, FrameDecoder};
use arrow::buffer::Buffer;
use arrow::ipc::reader::StreamDecoder;
use arrow_array::RecordBatch;
use bytes::Bytes;

/// The Arrow format's sequential half: frames in, client-schema batches out.
/// A frame is handed to the IPC decoder as is: a message contained in one
/// frame is sliced in place, and the decoded batches keep the frame's
/// allocation alive.
pub(crate) struct ArrowFrameDecoder {
    decoder: StreamDecoder,
    /// The part of the current frame not yet decoded.
    frame: Buffer,
}

impl ArrowFrameDecoder {
    pub(crate) fn new() -> Self {
        Self {
            decoder: StreamDecoder::new(),
            frame: Buffer::from(Bytes::new()),
        }
    }
}

impl FrameDecoder for ArrowFrameDecoder {
    fn push_frame(&mut self, frame: Bytes) {
        self.frame = Buffer::from(frame);
    }

    fn next_chunk(&mut self) -> Result<Option<Box<dyn CopyChunk>>, String> {
        let batch = self
            .decoder
            .decode(&mut self.frame)
            .map_err(|e| format!("decoding COPY arrow stream: {e}"))?;
        Ok(batch.map(|batch| Box::new(batch) as Box<dyn CopyChunk>))
    }

    /// Complete batches went out as they decoded, so there is nothing to
    /// flush; a stream cut mid-message is an error.
    fn finish(&mut self) -> Result<Option<Box<dyn CopyChunk>>, String> {
        self.decoder
            .finish()
            .map_err(|e| format!("COPY arrow stream ended mid-message: {e}"))?;
        Ok(None)
    }
}

/// A decoded batch is already what the workers conform.
impl CopyChunk for RecordBatch {
    fn into_batch(self: Box<Self>) -> Result<RecordBatch, String> {
        Ok(*self)
    }
}
