use crate::resample::new_resample_processor;
use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, StreamConfig};
use dillhaven_assistant_sync::stream::mpsc_rx_to_mspc_tx;
use dillhaven_assistant_types::dialogue::{DialogueCoordinatorRef, DialogueMode};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tracing::{error, instrument, trace};

pub struct AudioPlayback {
    in_tx: Sender<Vec<f32>>,
    _stream: cpal::Stream,
    config: StreamConfig,
}

impl AudioPlayback {
    pub fn new(dialogue_coordinator: DialogueCoordinatorRef) -> Result<AudioPlayback> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no output device available"))?;
        let config = device.default_output_config()?;
        let mut stream_config = config.config();
        stream_config.buffer_size = cpal::BufferSize::Fixed(
            (stream_config.channels as u32) * (stream_config.sample_rate.0) * 10,
        );
        let (in_tx, in_rx) = channel(100);

        let error_callback = |err| {
            error!("an error occurred on stream: {}", err);
        };

        let stream = match config.sample_format() {
            SampleFormat::F32 => Self::build_stream::<f32>(
                &device,
                &config.config(),
                in_rx,
                error_callback,
                dialogue_coordinator,
            ),
            SampleFormat::F64 => Self::build_stream::<f64>(
                &device,
                &config.config(),
                in_rx,
                error_callback,
                dialogue_coordinator,
            ),
            sample_format => Err(anyhow::Error::msg(format!(
                "Unsupported sample format '{sample_format}'"
            ))),
        }?;

        stream.play()?;
        let config = config.config();

        Ok(AudioPlayback {
            in_tx,
            _stream: stream,
            config,
        })
    }

    pub fn get_sample_rate(&self) -> u32 {
        self.config.sample_rate.0
    }

    fn build_stream<T>(
        device: &cpal::Device,
        config: &StreamConfig,
        in_rx: Receiver<Vec<f32>>,
        error_callback: impl Fn(cpal::StreamError) + Send + 'static,
        dialogue_coordinator: DialogueCoordinatorRef,
    ) -> Result<cpal::Stream>
    where
        T: Sample + SizedSample + FromSample<f32> + Send + Sync + 'static,
    {
        let buffer = OutputStreamBuf::new(in_rx, config.sample_rate.0 as usize);
        let num_channels = config.channels as usize;
        let stream = device.build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                process_output(data, &buffer, num_channels, &dialogue_coordinator);
            },
            error_callback,
            None,
        )?;
        Ok(stream)
    }

    pub fn get_in_tx(&self) -> Sender<Vec<f32>> {
        self.in_tx.clone()
    }

    pub fn get_resampled_tx(&self, input_sample_rate: usize) -> Sender<Vec<f32>> {
        let (tx, rx) = channel(100);
        let processor =
            new_resample_processor(rx, input_sample_rate, self.get_sample_rate() as usize);

        let audio_tx = self.get_in_tx();
        mpsc_rx_to_mspc_tx(processor.get_result_rx(), audio_tx);

        tx
    }
}

#[instrument(skip(data, buf))]
fn process_output<T>(
    data: &mut [T],
    buf: &Arc<OutputStreamBuf<T>>,
    num_channels: usize,
    dialogue_coordinator: &DialogueCoordinatorRef,
) where
    T: Sample + FromSample<f32> + Send + Sync + 'static,
{
    trace!("data_len {}", data.len());
    assert_eq!(
        data.len() % num_channels,
        0,
        "data_len {} is not divisible by num_channels {}",
        data.len(),
        num_channels
    );

    let size = data.len() / num_channels;
    let mut queue = buf.front(size);
    if !queue.is_empty() && DialogueMode::Speaking != dialogue_coordinator.get_current_state() {
        dialogue_coordinator.set_current_state(DialogueMode::Speaking);
    } else if queue.is_empty() && DialogueMode::Speaking == dialogue_coordinator.get_current_state()
    {
        dialogue_coordinator.set_current_state(DialogueMode::PassiveListening);
    }

    for frame in data.chunks_mut(num_channels) {
        let value = queue.pop_front().unwrap_or(T::EQUILIBRIUM);
        for sample in frame {
            *sample = value;
        }
    }
}

#[derive(Clone)]
struct OutputStreamBuf<T>
where
    T: Sample + FromSample<f32> + Send + Sync + 'static,
{
    inner: Arc<RwLock<OutputStreamBufInner<T>>>,
}

impl<T: Sample + FromSample<f32> + Send + Sync + 'static> OutputStreamBuf<T> {
    #[instrument(skip(in_rx))]
    pub fn new(in_rx: Receiver<Vec<f32>>, sample_rate: usize) -> Arc<Self> {
        let ret = Arc::new(Self {
            inner: Arc::new(RwLock::new(OutputStreamBufInner::new(sample_rate))),
        });

        ret.clone().start_collection(in_rx);

        ret
    }

    #[instrument(skip(self, in_rx))]
    fn start_collection(self: Arc<Self>, in_rx: Receiver<Vec<f32>>) {
        tokio::spawn(async move { self.collector(in_rx).await });
    }

    #[instrument(skip(self, in_rx))]
    async fn collector(&self, mut in_rx: Receiver<Vec<f32>>) {
        while let Some(data) = in_rx.recv().await {
            trace!("received data len {}", data.len());
            let data = data.into_iter().map(|x| x.to_sample::<T>()).collect();
            let mut writer = self.inner.write().unwrap();
            writer.append(data);
        }
    }

    #[instrument(skip(self))]
    fn front(&self, size: usize) -> VecDeque<T> {
        let mut inner = self.inner.write().unwrap();
        inner.pop_front(size)
    }
}

struct OutputStreamBufInner<T>
where
    T: Sample + FromSample<f32> + Send + Sync + 'static,
{
    buf: VecDeque<T>,
}

impl<T: Sample + FromSample<f32> + Send + Sync + 'static> OutputStreamBufInner<T> {
    pub fn new(sample_rate: usize) -> Self {
        Self {
            buf: VecDeque::from(vec![T::EQUILIBRIUM; sample_rate]),
        }
    }

    #[instrument(skip(self, data))]
    pub fn append(&mut self, mut data: VecDeque<T>) {
        self.buf.append(&mut data);
        trace!("buf len {}", self.buf.len());
    }

    #[instrument(skip(self))]
    pub fn pop_front(&mut self, size: usize) -> VecDeque<T> {
        let len = self.buf.len();
        let size = if len < size { len } else { size };
        trace!("pop_front size {} len {}", size, len);
        let mut ret = VecDeque::with_capacity(size);

        if len == 0 {
            ret
        } else if len < size {
            ret.extend(self.buf.drain(..len));
            ret
        } else {
            ret.extend(self.buf.drain(..size));
            ret
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dillhaven_assistant_types::dialogue::DialogueCoordinator;
    use rand::Rng;
    use tracing::info;

    #[tokio::test]
    #[instrument]
    #[ignore] // cannot run on GHA
    async fn test_audio_playback() {
        dillhaven_assistant_observe::init();

        let data_len = 44_100 * 1000;
        let mut buf = vec![0.0; data_len];
        let mut rng = rand::rng();
        rng.fill(buf.as_mut_slice());
        let mut buf = VecDeque::from(buf);
        let dc = DialogueCoordinator::new();

        let playback = AudioPlayback::new(dc).unwrap();

        let in_tx = playback.get_in_tx();
        let test_duration = 1;
        info!("config: {:#?}", playback.config);

        tokio::spawn(async move {
            while !buf.is_empty() {
                let buf2 = buf.drain(..1_000).collect();
                in_tx.send(buf2).await.unwrap();
            }
        });

        tokio::time::sleep(std::time::Duration::from_secs(test_duration as u64)).await;
    }
}
