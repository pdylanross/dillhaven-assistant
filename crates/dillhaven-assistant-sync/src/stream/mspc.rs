use crate::stream::DynItemProcessor;
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tracing::{error, instrument};

pub struct MpscStreamProcessor<Out> {
    out_rx: Receiver<Out>,
}

impl<Out: Clone + Send + Sync + 'static> MpscStreamProcessor<Out> {
    pub fn new<In: Clone + Send + Sync + 'static>(
        in_rx: Receiver<In>,
        channel_capacity: usize,
        processor: DynItemProcessor<In, Out>,
    ) -> Self {
        let (out_tx, out_rx) = channel(channel_capacity);
        process_mpsc_stream(in_rx, out_tx.clone(), processor);

        Self { out_rx }
    }

    pub fn get_result_rx(self) -> Receiver<Out> {
        self.out_rx
    }
}

#[instrument(skip(in_rx, out_tx, processor))]
fn process_mpsc_stream<In: Clone + Send + Sync + 'static, Out: Clone + Send + Sync + 'static>(
    mut in_rx: Receiver<In>,
    out_tx: Sender<Out>,
    processor: DynItemProcessor<In, Out>,
) {
    tokio::spawn(async move {
        while let Some(item) = in_rx.recv().await {
            let processed_item = processor.process(item).await;
            if let Err(e) = processed_item {
                error!("Error processing item: {}", e);
                break;
            }

            if let Some(item) = processed_item.unwrap() {
                if out_tx.send(item).await.is_err() {
                    return;
                }
            }
        }

        let finalizer = processor.done().await;
        if let Err(e) = finalizer {
            error!("Error finalizing processor: {}", e);
        } else if let Some(final_result) = finalizer.unwrap() {
            if let Err(e) = out_tx.send(final_result).await {
                error!("Error sending final result: {}", e);
            }
        }
    });
}
