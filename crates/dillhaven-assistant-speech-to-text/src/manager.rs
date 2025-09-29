use crate::model::MoshiModel;
use crate::output_stream::TextToSpeechOutputStream;
use anyhow::{anyhow, Result};
use dillhaven_assistant_audio::audio_capture::AudioCapture;
use dillhaven_assistant_sync::lifespan::{AppState, LifespanManager};
use dillhaven_assistant_types::dialogue::{DialogueCoordinatorRef, DialogueMode};
use dillhaven_assistant_util::huggingface::HuggingFaceApiManger;
use std::sync::Arc;
use tokio::sync::broadcast::{self, Receiver, Sender};
use tokio::{select, task};
use tracing::{error, info, trace};

const BUFFER_SIZE: usize = 1920;

/// Manager for speech-to-text processing
pub struct STTManager {
    text_sender: Sender<String>,
    lifespan_manager: Arc<LifespanManager>,
    output_stream: TextToSpeechOutputStream,
    hf_manager: Arc<HuggingFaceApiManger>,
}

impl STTManager {
    /// Creates a new STTManager with the specified configuration
    pub fn new(
        lifespan_manager: Arc<LifespanManager>,
        hf_manager: Arc<HuggingFaceApiManger>,
    ) -> Result<Self> {
        let (text_sender, _) = broadcast::channel(100);
        let output_stream =
            TextToSpeechOutputStream::new(text_sender.clone(), lifespan_manager.clone());

        Ok(Self {
            text_sender,
            lifespan_manager,
            output_stream,
            hf_manager,
        })
    }

    /// Starts audio capture and processing
    pub async fn start(&self, dialogue_coordinator: DialogueCoordinatorRef) -> Result<()> {
        self.output_stream.start();

        // Clone necessary data for the processing task
        let text_sender = self.text_sender.clone();
        let lifespan_manager = self.lifespan_manager.clone();
        let hf_manager = self.hf_manager.clone();
        let mut init_barrier = lifespan_manager.get_init_barrier().await;

        // Spawn a task to process audio chunks
        task::spawn(async move {
            let model = MoshiModel::load_from_hf(hf_manager).await;
            if let Err(e) = model {
                error!("Error loading STT model: {}", e);
                lifespan_manager.crash(e);
                return;
            }

            let model = model.unwrap();

            let audio_capture = AudioCapture::new(BUFFER_SIZE);
            if let Err(e) = audio_capture {
                error!("Error starting up audio capture: {}", e);
                lifespan_manager.crash(e);
                return;
            }
            let audio_capture = audio_capture.unwrap();

            let audio_receiver = audio_capture.add_receiver();
            let mut loop_processor =
                AudioLoopProcessor::new(model, audio_receiver, text_sender, dialogue_coordinator);
            let mut app_state_rx = lifespan_manager.get_state_rx();
            init_barrier.wait();

            trace!("Audio capture and processing task started");

            loop {
                select! {
                    res = loop_processor.run_loop() => {
                        if let Err(e) = res {
                            error!("Error processing audio: {}", e);
                            lifespan_manager.crash(e);
                            break
                        }
                    },
                    event = app_state_rx.changed() => {
                        if let Err(err) = event {
                            error!("Error receiving app state: {}", err);
                            break;
                        } else {
                            let val = app_state_rx.borrow_and_update();
                            if let AppState::Shutdown = *val {
                                break
                            }
                        }
                    }
                }
            }

            trace!("Audio capture and processing task stopped");
        });

        info!("Audio capture and processing started");

        Ok(())
    }

    /// Creates a new receiver for the recognized text
    pub fn add_text_receiver(&self) -> Receiver<String> {
        self.output_stream.get_output_tx()
    }
}

struct AudioLoopProcessor {
    model: MoshiModel,
    audio_rx: Receiver<Vec<f32>>,
    text_tx: Sender<String>,
    dialogue_coordinator: DialogueCoordinatorRef,
}

impl AudioLoopProcessor {
    fn new(
        model: MoshiModel,
        audio_rx: Receiver<Vec<f32>>,
        text_tx: Sender<String>,
        dialogue_coordinator: DialogueCoordinatorRef,
    ) -> Self {
        Self {
            model,
            audio_rx,
            text_tx,
            dialogue_coordinator,
        }
    }

    pub async fn run_loop(&mut self) -> Result<()> {
        match self.audio_rx.recv().await {
            Ok(buffer) => {
                // if we're speaking right now, skip processing any input
                if let DialogueMode::Speaking = self.dialogue_coordinator.get_current_state() {
                    return Ok(());
                }
                // Get a lock on the model
                // Process the audio chunk and send recognized text
                if let Err(e) = self.model.process_chunk(&buffer, |text| {
                    let _ = self.text_tx.send(text.to_string());
                }) {
                    error!("Error processing audio chunk: {}", e);
                }
                Ok(())
            }
            Err(e) => Err(anyhow!(e)),
        }
    }
}
