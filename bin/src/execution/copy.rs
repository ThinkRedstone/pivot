use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use arrow::ipc::reader::StreamDecoder;
use arrow_array::RecordBatch;
use bytes::Bytes;
use dispatch::{CancelToken, ChannelInputFull, ChannelInputSender, DataFlowHandle};
use tokio::sync::Notify;

use super::dataflow::{affected_rows, affected_rows_from_record_batch};
use super::{Error, Executor, Result};

/// A running `COPY ... FROM STDIN` ingest, the executor's whole copy surface:
/// the frontend feeds it protocol bytes and finishes or drops it, and every
/// dispatch-facing concern (decoding, backpressure, cancellation, the
/// statement's transaction) stays inside.
pub struct CopyIngest {
    /// The sequential half: reassembles the protocol's byte frames into Arrow
    /// IPC messages and yields client-schema batches.
    decoder: StreamDecoder,
    sender: ChannelInputSender<RecordBatch>,
    /// Signalled by workers claiming batches; [`push`](Self::push) sleeps on
    /// it while the queue is full.
    space_freed: Arc<Notify>,
    cancel: CancelToken,
    /// Taken by [`finish`](Self::finish); still present on drop marks an
    /// unfinished ingest, which the drop aborts.
    handle: Option<DataFlowHandle<usize>>,
    /// The transaction the statement was planned in, which its bound table
    /// stages into. Taken and committed by [`finish`](Self::finish); rolled
    /// back on drop otherwise.
    transaction: Option<Arc<dyn planner::catalog::CatalogTransaction>>,
    column_count: usize,
}

impl fmt::Debug for CopyIngest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopyIngest")
            .field("column_count", &self.column_count)
            .finish_non_exhaustive()
    }
}

impl CopyIngest {
    /// Columns each incoming batch must carry (for the frontend's
    /// copy-in response).
    pub fn column_count(&self) -> usize {
        self.column_count
    }

    /// Decode one protocol frame and feed the batches it completes to the
    /// running dataflow, sleeping while the queue is full. The decoder
    /// carries partial messages across frames, so a batch may span any number
    /// of frames (and one frame may complete several). An error means the
    /// stream is malformed; the copy cannot proceed and should be dropped.
    ///
    /// The frame arrives as refcounted [`Bytes`], so handing it to the
    /// decoder copies nothing: a message contained in one frame is sliced in
    /// place, and the decoded batches keep the frame's allocation alive.
    pub async fn push(&mut self, bytes: Bytes) -> Result<()> {
        let mut buffer = arrow::buffer::Buffer::from(bytes);
        loop {
            let batch = self
                .decoder
                .decode(&mut buffer)
                .map_err(|e| Error::Copy(format!("decoding COPY arrow stream: {e}")))?;
            let Some(mut batch) = batch else {
                return Ok(());
            };
            loop {
                // A dead dataflow claims nothing again; stop feeding it. Its
                // own error surfaces when the flow is collected at finish.
                if self.cancel.is_cancelled() {
                    return Ok(());
                }
                // Register before trying so a claim racing the failed send
                // cannot lose the notification. The timeout lets a cancelled
                // dataflow that will never claim again be observed promptly.
                let notified = self.space_freed.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                match self.sender.try_send(batch) {
                    Ok(()) => break,
                    Err(ChannelInputFull(returned)) => {
                        batch = returned;
                        let _ = tokio::time::timeout(Duration::from_millis(50), notified).await;
                    }
                }
            }
        }
    }

    /// Verify the stream ended cleanly, drain the dataflow, and commit,
    /// returning the ingested-row count. Consuming `self` makes completion
    /// single-use; every failure path drops the remains, which aborts.
    pub async fn finish(mut self) -> Result<usize> {
        // Complete batches were sent as they decoded, so there is nothing to
        // flush; a stream cut mid-message is an error.
        self.decoder
            .finish()
            .map_err(|e| Error::Copy(format!("COPY arrow stream ended mid-message: {e}")))?;
        self.sender.close();
        let handle = self.handle.take().expect("a copy finishes only once");
        let outputs = tokio::task::spawn_blocking(move || handle.collect())
            .await
            .map_err(Error::WorkerPanic)??;
        let count = affected_rows(outputs)?;
        let transaction = self
            .transaction
            .take()
            .expect("the transaction resolves only here");
        transaction.commit().await?;
        Ok(count)
    }
}

/// Dropping an unfinished ingest aborts it: cancel the dataflow, stop the
/// queue, and roll back the statement's transaction, discarding whatever the
/// copy staged. A finished ingest already took both fields, so its drop does
/// nothing.
impl Drop for CopyIngest {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.cancel.cancel();
            self.sender.close();
        }
        if let Some(transaction) = self.transaction.take() {
            transaction.rollback();
        }
    }
}

impl Executor {
    /// Launch the ingest dataflow for a planned COPY FROM STDIN: a batch
    /// channel fanning out to per-worker schema-conformance stages, feeding
    /// the target table's insert sink, staging into the statement's
    /// transaction. The returned [`CopyIngest`] owns the whole exchange.
    pub(super) async fn launch_copy_ingest(
        &self,
        statement: planner::CopyFromStdin,
        transaction: Arc<dyn planner::catalog::CatalogTransaction>,
    ) -> Result<CopyIngest> {
        let column_count = if statement.columns.is_empty() {
            statement.table.columns().len()
        } else {
            statement.columns.len()
        };
        let space_freed = Arc::new(Notify::new());
        let on_claim = {
            let space_freed = space_freed.clone();
            Box::new(move || space_freed.notify_one()) as Box<dyn Fn() + Send + Sync>
        };
        let dispatcher = self.dispatcher.clone();
        // Compile and launch on the blocking pool: the insert sink's write
        // preparation may touch the store.
        let (sender, handle) =
            tokio::task::spawn_blocking(move || -> Result<(ChannelInputSender<RecordBatch>, _)> {
                let (sender, spec) = statement.compile_ingest(&dispatcher, on_claim)?;
                Ok((
                    sender,
                    spec.map(|| affected_rows_from_record_batch).execute(),
                ))
            })
            .await
            .map_err(Error::PlannerPanic)??;
        Ok(CopyIngest {
            decoder: StreamDecoder::new(),
            cancel: handle.cancel_token(),
            sender,
            space_freed,
            handle: Some(handle),
            transaction: Some(transaction),
            column_count,
        })
    }
}
