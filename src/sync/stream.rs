use crate::app::lifespan::LifespanManager;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::select;
use tokio::sync::broadcast::{channel, Receiver, Sender};
use tracing::{error, instrument};

pub type DynItemProcessor<In, Out> = Arc<dyn ItemProcessor<In, Out> + Send + Sync>;

#[async_trait]
pub trait ItemProcessor<In, Out> {
    async fn process(&self, item: In) -> Result<Option<Out>>;
    async fn done(&self) -> Result<Option<Out>>;
}

pub struct BroadcastStreamProcessor<Out> {
    out_tx: Sender<Out>,
}

impl<Out: Clone + Send + Sync + 'static> BroadcastStreamProcessor<Out> {
    pub fn new<In: Clone + Send + Sync + 'static>(
        in_rx: Receiver<In>,
        channel_capacity: usize,
        processor: DynItemProcessor<In, Out>,
        lifespan_manager: Arc<LifespanManager>,
    ) -> Self {
        let (out_tx, _) = channel(channel_capacity);
        process_broadcast_stream(in_rx, out_tx.clone(), processor, lifespan_manager);

        Self { out_tx }
    }

    pub fn get_result_rx(&self) -> Receiver<Out> {
        self.out_tx.subscribe()
    }
}

#[instrument(skip(in_rx, out_tx, processor, lifespan_manager))]
fn process_broadcast_stream<
    In: Clone + Send + Sync + 'static,
    Out: Clone + Send + Sync + 'static,
>(
    mut in_rx: Receiver<In>,
    out_tx: Sender<Out>,
    processor: DynItemProcessor<In, Out>,
    lifespan_manager: Arc<LifespanManager>,
) {
    tokio::spawn(async move {
        let mut shutdown_signal = lifespan_manager.get_shutdown_signaler();

        loop {
            select! {
                item = in_rx.recv() => {
                    if let Err(_) = item {
                        let finalizer = processor.done().await;
                        if let Err(e) = finalizer {
                            error!("Error finalizing processor: {}", e);
                        } else if let Some(final_result) = finalizer.unwrap() {
                            if let Err(e) = out_tx.send(final_result) {
                                error!("Error sending final result: {}", e);
                            }
                        }
                        break;
                    }
                    let item = item.unwrap();

                    let process_res = processor.process(item).await;
                    if let Err(e) = process_res {
                        error!("Error processing item: {}", e);
                        break;
                    }

                    if let Some(result) = process_res.unwrap() {
                        if let Err(e) = out_tx.send(result) {
                            error!("Error sending result: {}", e);
                            break;
                        }
                    }
                },
                res = shutdown_signal.flagged_for_shutdown() => {
                    if res {
                        break
                    }
                }
            }
        }
    });
}
