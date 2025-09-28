use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, SizedSample};
use once_cell::sync::Lazy;
use tokio::sync::broadcast::{self, Receiver, Sender};
use tracing::{error, info, warn};

static SAMPLE_RATE: Lazy<u32> = Lazy::new(|| {
    let host = cpal::default_host();

    // Get the default input device
    let device = host
        .default_input_device()
        .expect("No default input device available");

    device.default_input_config().unwrap().sample_rate().0
});

pub fn get_sample_rate() -> u32 {
    *SAMPLE_RATE
}

/// A struct that captures audio from the default input device
pub struct AudioCapture {
    _stream: cpal::Stream,
    receiver: Receiver<Vec<f32>>,
    sender: Sender<Vec<f32>>,
}

impl AudioCapture {
    /// Creates a new AudioCapture with the specified buffer size
    /// The audio capture starts immediately upon creation
    ///
    /// Note: For optimal speech recognition, a buffer size of 1920 is recommended
    /// as it matches the chunk size used in the reference implementation.
    pub fn new(buffer_size: usize) -> Result<Self> {
        let host = cpal::default_host();

        // Get the default input device
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("No default input device available"))?;

        info!("Using input device: {}", device.name()?);

        // Get the default config for the device
        let config = device.default_input_config()?;
        let stream_config = config.config();

        info!("Using audio config: {:?}", config);

        // Create a broadcast channel to send audio data
        // The channel capacity should be large enough to handle audio buffers
        let (sender, receiver) = broadcast::channel(500);

        // Set up the audio stream based on the sample format
        let err_fn = |err| error!("An error occurred on the audio stream: {}", err);

        let stream = match config.sample_format() {
            SampleFormat::F32 => Self::build_stream::<f32>(
                &device,
                &stream_config,
                sender.clone(),
                buffer_size,
                err_fn,
            )?,
            SampleFormat::I16 => Self::build_stream::<i16>(
                &device,
                &stream_config,
                sender.clone(),
                buffer_size,
                err_fn,
            )?,
            SampleFormat::U16 => Self::build_stream::<u16>(
                &device,
                &stream_config,
                sender.clone(),
                buffer_size,
                err_fn,
            )?,
            _ => return Err(anyhow!("Unsupported sample format")),
        };

        // Start the stream
        stream.play()?;

        Ok(AudioCapture {
            _stream: stream,
            receiver,
            sender,
        })
    }

    /// Builds an audio stream with the specified sample type
    fn build_stream<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        sender: Sender<Vec<f32>>,
        buffer_size: usize,
        err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
    ) -> Result<cpal::Stream>
    where
        T: Sample + SizedSample + Send + 'static,
    {
        let channels = config.channels as usize;
        let mut buffer = Vec::<f32>::with_capacity(buffer_size);
        let stream = device.build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                // Convert samples to f32 and add to buffer
                for frame in data.chunks(channels) {
                    let frame = frame.iter().map(|s| s.to_float_sample().to_sample::<f32>());
                    let sum = frame.sum::<f32>();
                    let avg = (sum / channels as f32).clamp(-1.0, 1.0);

                    buffer.push(avg);

                    // When the buffer is full, send it and create a new one
                    if buffer.len() >= buffer_size {
                        // Send the downsampled audio
                        let res = sender.send(buffer.clone());
                        if let Err(err) = res {
                            warn!("Failed to send audio buffer: {}", err);
                        }
                        buffer.clear();
                    }
                }
            },
            err_fn,
            None,
        )?;

        stream.play()?;
        Ok(stream)
    }

    /// Returns the next audio buffer if available
    pub fn next_buffer(&mut self) -> Option<Vec<f32>> {
        match self.receiver.try_recv() {
            Ok(buffer) => Some(buffer),
            Err(_) => None,
        }
    }

    /// Creates a new receiver that will get the same audio data
    pub fn add_receiver(&self) -> Receiver<Vec<f32>> {
        self.sender.subscribe()
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        // The stream will be dropped automatically
    }
}
