use crate::stream::DynItemProcessor;
use tokio::sync::broadcast::{channel, Receiver, Sender};
use tracing::{error, instrument};

pub struct BroadcastStreamProcessor<Out> {
    out_tx: Sender<Out>,
}

impl<Out: Clone + Send + Sync + 'static> BroadcastStreamProcessor<Out> {
    pub fn new<In: Clone + Send + Sync + 'static>(
        in_rx: Receiver<In>,
        channel_capacity: usize,
        processor: DynItemProcessor<In, Out>,
    ) -> Self {
        let (out_tx, _) = channel(channel_capacity);
        process_broadcast_stream(in_rx, out_tx.clone(), processor);

        Self { out_tx }
    }

    pub fn get_result_rx(&self) -> Receiver<Out> {
        self.out_tx.subscribe()
    }
}

#[instrument(skip(in_rx, out_tx, processor))]
fn process_broadcast_stream<
    In: Clone + Send + Sync + 'static,
    Out: Clone + Send + Sync + 'static,
>(
    mut in_rx: Receiver<In>,
    out_tx: Sender<Out>,
    processor: DynItemProcessor<In, Out>,
) {
    tokio::spawn(async move {
        while let Ok(item) = in_rx.recv().await {
            let processed_item = processor.process(item).await;
            if let Err(e) = processed_item {
                error!("Error processing item: {}", e);
                break;
            }

            if let Some(item) = processed_item.unwrap() {
                if out_tx.send(item).is_err() {
                    return;
                }
            }
        }

        let finalizer = processor.done().await;
        if let Err(e) = finalizer {
            error!("Error finalizing processor: {}", e);
        } else if let Some(final_result) = finalizer.unwrap() {
            if let Err(e) = out_tx.send(final_result) {
                error!("Error sending final result: {}", e);
            }
        }
    });
}
