use crate::app::huggingface::HuggingFaceApiManger;
use crate::app::lifespan::{AppState, LifespanManager};
use crate::llm::model::{Model, ModelSettings, ModelVersion};
use crate::llm::prompt::PromptProcessor;
use crate::text::sentence_chunker::StreamSentenceChunker;
use anyhow::Result;
use log::warn;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::select;
use tokio::sync::broadcast::{Receiver, Sender};
use tracing::{error, trace};

pub struct LLMManager {
    hf_manager: Arc<HuggingFaceApiManger>,
    lifespan_manager: Arc<LifespanManager>,
    output_tx: Sender<String>,
}

impl LLMManager {
    pub fn new(
        hf_manager: Arc<HuggingFaceApiManger>,
        lifespan_manager: Arc<LifespanManager>,
    ) -> Result<Self> {
        let (output_tx, _) = tokio::sync::broadcast::channel(10);

        Ok(Self {
            hf_manager,
            lifespan_manager,
            output_tx,
        })
    }

    fn get_model_settings(&self) -> ModelSettings {
        ModelSettings {
            version: ModelVersion::InstructV2_9B,
            seed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("get sys time")
                .as_secs(),
            temp: Some(0.1),
            top_p: None,
        }
    }

    pub fn add_text_receiver(&self) -> Receiver<String> {
        self.output_tx.subscribe()
    }

    pub async fn run(&self, input_rx: Receiver<String>) -> Result<()> {
        let lifespan_manager = self.lifespan_manager.clone();
        let hf_manager = self.hf_manager.clone();
        let model_settings = self.get_model_settings();
        let mut init_barrier = lifespan_manager.get_init_barrier().await;
        let output_tx = self.output_tx.clone();

        tokio::spawn(async move {
            let model = Model::new(model_settings, hf_manager).await;
            if let Err(e) = model {
                log::error!("model load error: {}", e);
                lifespan_manager.crash(e);
                return;
            }

            let mut inner = LLMMangerInner::new(
                input_rx,
                model.unwrap(),
                output_tx,
                lifespan_manager.clone(),
            );
            let mut state_rx = lifespan_manager.get_state_rx();
            init_barrier.wait();

            let mut inference_errs = 0;

            loop {
                select! {
                    res = inner.run_iter() => {
                        if let Err(e) = res {
                            warn!("error in llm: {}", e);
                            inference_errs += 1;
                            if inference_errs > 10 {
                                error!("too many inference errors, crashing");
                                lifespan_manager.crash(e);
                                break
                            }
                        } else {
                            if inference_errs > 0 {
                                inference_errs = 0;
                            }
                        }
                    },
                    event = state_rx.changed() => {
                        if let Err(err) = event {
                            error!("Error receiving app state: {}", err);
                            break;
                        } else {
                            let val = state_rx.borrow_and_update();
                            match *val {
                                AppState::Shutdown => {
                                    break
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }
}

struct LLMMangerInner {
    input_rx: Receiver<String>,
    model: Model,
    output_tx: Sender<String>,
    prompt_processor: PromptProcessor,
    lifespan_manager: Arc<LifespanManager>,
}

impl LLMMangerInner {
    pub fn new(
        input_rx: Receiver<String>,
        model: Model,
        output_tx: Sender<String>,
        lifespan_manager: Arc<LifespanManager>,
    ) -> Self {
        let prompt_processor = PromptProcessor::new();

        Self {
            input_rx,
            model,
            output_tx,
            prompt_processor,
            lifespan_manager,
        }
    }

    pub async fn run_iter(&mut self) -> Result<()> {
        let res = self.input_rx.recv().await?;
        let prompt = self.prompt_processor.process(res);
        let inference_res = self.model.prompt(prompt)?;
        let chunker = StreamSentenceChunker::new_processor_with_default_tokens(
            inference_res,
            self.lifespan_manager.clone(),
        );
        let mut chunk_rx = chunker.get_result_rx();
        let tx = self.output_tx.clone();
        tokio::spawn(async move {
            while let Ok(chunk) = chunk_rx.recv().await {
                trace!("chunk: {}", chunk);

                if let Err(_) = tx.send(chunk) {
                    return;
                };
            }
        });

        Ok(())
    }
}
