use crate::piper::{PiperModel, PiperModelConfig, MODEL_PATH};
use anyhow::{anyhow, Result};
use dillhaven_assistant_audio::audio_playback::AudioPlayback;
use dillhaven_assistant_sync::lifespan::{AppState, LifespanManager};
use dillhaven_assistant_types::dialogue::DialogueCoordinatorRef;
use std::sync::Arc;
use tokio::select;
use tokio::sync::broadcast::Receiver;
use tracing::{error, instrument, trace};

pub struct TTSManager {
    lifespan_manager: Arc<LifespanManager>,
}

impl TTSManager {
    pub fn new(lifespan_manager: Arc<LifespanManager>) -> Result<Self> {
        Ok(TTSManager { lifespan_manager })
    }

    #[instrument(skip(self, output_stream))]
    pub async fn start(
        &self,
        mut output_stream: Receiver<String>,
        dialogue_coordinator: DialogueCoordinatorRef,
    ) -> Result<()> {
        let mut init_barrier = self.lifespan_manager.get_init_barrier().await;
        let mut app_state_rx = self.lifespan_manager.get_state_rx();
        let lifespan_manager = self.lifespan_manager.clone();

        let model_config = PiperModelConfig {
            config_path: MODEL_PATH.to_string(),
        };

        trace!("Starting TTSManager with piper config {:?}", model_config);

        let model = PiperModel::new(model_config)?;

        trace!("piper model loaded");

        tokio::spawn(async move {
            init_barrier.wait();
            let playback = AudioPlayback::new(dialogue_coordinator);
            if let Err(e) = playback {
                error!("Error starting audio playback: {}", e);
                lifespan_manager.crash(e);
                return;
            }

            let playback = playback.unwrap();
            let audio_tx = playback.get_resampled_tx(model.get_sample_rate());

            loop {
                select! {
                    output = output_stream.recv() => {
                        if let Ok(output) = output {
                            let piper_res = model.synth(output);
                            if let Err(e) = piper_res {
                                error!("Error synthesizing text: {}", e);
                                lifespan_manager.crash(e);
                                return
                            }

                            let piper_res = piper_res.unwrap();
                            for res in piper_res {
                                let res = res.unwrap();

                                if let Err(e) = audio_tx.send(res.samples.into_vec()).await {
                                    error!("Error sending audio: {}", e);
                                    lifespan_manager.crash(anyhow!(e));
                                    return
                                }
                            }
                        }
                    }
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
        });

        Ok(())
    }
}
