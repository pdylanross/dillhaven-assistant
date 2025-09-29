use async_trait::async_trait;
use dillhaven_assistant_sync::stream::{ItemProcessor, MpscStreamProcessor};
use kaudio::resample;
use std::sync::Arc;
use tokio::sync::mpsc::Receiver;

struct AudioResampleStreamProcessor {
    from: usize,
    to: usize,
}

pub fn new_resample_processor(
    in_rx: Receiver<Vec<f32>>,
    from: usize,
    to: usize,
) -> MpscStreamProcessor<Vec<f32>> {
    let item_processor = Arc::new(AudioResampleStreamProcessor { from, to });

    MpscStreamProcessor::new(in_rx, 10, item_processor)
}

#[async_trait]
impl ItemProcessor<Vec<f32>, Vec<f32>> for AudioResampleStreamProcessor {
    async fn process(&self, item: Vec<f32>) -> anyhow::Result<Option<Vec<f32>>> {
        Ok(Some(resample(&item, self.from, self.to)?))
    }

    async fn done(&self) -> anyhow::Result<Option<Vec<f32>>> {
        Ok(None)
    }
}
