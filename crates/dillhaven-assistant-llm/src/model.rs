use crate::token_stream::TokenizerHandler;
use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma2::{Config as Config2, Model as Model2};
use candle_transformers::models::gemma3::{Config as Config3, Model as Model3};
use candle_transformers::models::mimi::candle;
use dillhaven_assistant_util::candle::get_device;
use dillhaven_assistant_util::huggingface::{
    HuggingFaceApiManger, SafeTensorsIndex, load_json_file_as_async,
};
use std::fmt::Debug;
use std::fs::File;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::broadcast::Receiver;
use tracing::{info, instrument, trace, warn};

#[cfg(feature = "flash")]
const FLASH_ENABLED: bool = true;
#[cfg(not(feature = "flash"))]
const FLASH_ENABLED: bool = false;

#[derive(Debug, Clone)]
pub struct ModelSettings {
    pub version: ModelVersion,
    pub seed: u64,
    pub temp: Option<f64>,
    pub top_p: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum ModelVersion {
    InstructV2_2B,
    InstructV2_9B,
    InstructV3_1B,
    InstructV3_4B,
}

impl ModelVersion {
    pub fn to_hf_id(&self) -> String {
        match self {
            ModelVersion::InstructV2_2B => "google/gemma-2-2b-it".to_string(),
            ModelVersion::InstructV2_9B => "google/gemma-2-9b-it".to_string(),
            ModelVersion::InstructV3_1B => "google/gemma-3-1b-it".to_string(),
            ModelVersion::InstructV3_4B => "google/gemma-3-4b-it".to_string(),
        }
    }
}

pub struct Model {
    inner: GemmaModelInner,
    tokenizer: Arc<Tokenizer>,
    dev: candle::Device,
    settings: ModelSettings,

    eos_token: u32,
    eot_token: Option<u32>,
}

impl Model {
    #[instrument]
    pub async fn new(
        settings: ModelSettings,
        hf_manager: Arc<HuggingFaceApiManger>,
    ) -> Result<Self> {
        let dev = get_device()?;

        info!("Loading gemma model: {:?} with device {:?}", settings, dev);

        // download / get cached versions of required models
        let api = hf_manager.get_async_api();
        let model_id = settings.version.to_hf_id();
        let repo = api.model(model_id);

        let tokenizer_filename = repo.get("tokenizer.json").await?;
        let config_filename = repo.get("config.json").await?;
        let models = match settings.version {
            ModelVersion::InstructV2_2B
            | ModelVersion::InstructV2_9B
            | ModelVersion::InstructV3_4B => {
                let model_index = load_json_file_as_async::<SafeTensorsIndex>(
                    &repo,
                    "model.safetensors.index.json",
                )
                .await?;
                model_index.load_tensors(&repo).await?
            }
            ModelVersion::InstructV3_1B => {
                vec![repo.get("model.safetensors").await?]
            }
        };

        // load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename.as_path());
        if let Err(e) = tokenizer {
            return Err(anyhow::anyhow!("failed to load tokenizer: {:?}", e));
        }
        let tokenizer = Arc::new(tokenizer.unwrap());

        let dtype = if dev.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&models, dtype, &dev)? };

        // actually load the model
        let inner = match settings.version {
            ModelVersion::InstructV2_2B | ModelVersion::InstructV2_9B => {
                let config: Config2 = serde_json::from_reader(File::open(config_filename)?)?;
                let model = Model2::new(FLASH_ENABLED, &config, vb)?;
                GemmaModelInner::V2(model)
            }
            ModelVersion::InstructV3_1B | ModelVersion::InstructV3_4B => {
                let config: Config3 = serde_json::from_reader(File::open(config_filename)?)?;
                let model = Model3::new(FLASH_ENABLED, &config, vb)?;
                GemmaModelInner::V3(model)
            }
        };
        info!("Gemma model loaded");

        // lookup and pre-save special tokens
        let vocab = tokenizer.get_vocab(true);
        let eos_token = if let Some(tok) = vocab.get("<eos>") {
            *tok
        } else {
            return Err(anyhow::anyhow!("failed to find EOS token in tokenizer"));
        };
        let eot_token = vocab.get("<end_of_turn>").copied();

        Ok(Self {
            inner,
            tokenizer,
            settings,
            dev,
            eos_token,
            eot_token,
        })
    }

    #[instrument(skip(self))]
    pub fn prompt<S: Into<String> + Debug>(&self, prompt: S) -> Result<Receiver<String>> {
        trace!("getting model thread ready");
        let stream = TokenizerHandler::new(self.tokenizer.clone());

        let mut inner = self.inner.clone();
        let eos_token = self.eos_token;
        let eot_token = self.eot_token.unwrap_or(eos_token);
        let mut logits_processor =
            LogitsProcessor::new(self.settings.seed, self.settings.temp, self.settings.top_p);
        let dev = self.dev.clone();
        let prompt: String = prompt.into();

        let response_stream = stream.get_rx();
        trace!("model thread ready");

        tokio::spawn(async move {
            trace!("model thread started");
            let prompt = inner.format_prompt(prompt.as_str());
            trace!("prompt: {}", prompt);
            let input_tokens = stream.get_tokenizer().encode(prompt, true);
            if let Err(e) = input_tokens {
                warn!("failed to encode prompt: {:?}", e);
                return;
            }
            let mut input_tokens = input_tokens.unwrap().get_ids().to_vec();

            trace!("begin inferencing");
            for index in 0..10_000 {
                let context_size = if index > 0 { 1 } else { input_tokens.len() };
                let start_pos = input_tokens.len().saturating_sub(context_size);
                let ctxt = &input_tokens[start_pos..];

                let input = Tensor::new(ctxt, &dev);
                if let Err(e) = input {
                    warn!("failed to create input tensor: {:?}", e);
                    return;
                }

                let input = input.expect("checked already").unsqueeze(0);
                if let Err(e) = input {
                    warn!("failed to unsqueeze input tensor: {:?}", e);
                    return;
                }
                let input = input.expect("checked already");

                trace!(
                    "model forward, start_pos: {}, input_size: {:#?}",
                    start_pos,
                    input.shape()
                );
                let logits = inner.forward(&input, start_pos);
                if let Err(e) = logits {
                    warn!("failed to forward: {:?}", e);
                    return;
                }

                let logits = logits.expect("checked already");

                let logits = logits.squeeze(0);
                if let Err(e) = logits {
                    warn!("failed to squeeze logits: {:?}", e);
                    return;
                }

                let logits = logits.expect("checked already").squeeze(0);
                if let Err(e) = logits {
                    warn!("failed to squeeze logits: {:?}", e);
                    return;
                }

                let logits = logits.expect("checked already").to_dtype(DType::F32);
                if let Err(e) = logits {
                    warn!("failed to convert logits to f32: {:?}", e);
                    return;
                }

                let logits = logits.expect("checked already");

                let next_token = logits_processor.sample(&logits);
                if let Err(e) = next_token {
                    warn!("failed to sample token: {:?}", e);
                    return;
                }
                let next_token = next_token.expect("checked already");

                input_tokens.push(next_token);

                if next_token == eos_token || next_token == eot_token {
                    break;
                }

                if let Err(e) = stream.next_token(next_token).await {
                    warn!("failed to write token: {:?}", e);
                    return;
                }
            }
        });

        Ok(response_stream)
    }
}

#[derive(Clone)]
enum GemmaModelInner {
    V2(Model2),
    V3(Model3),
}

impl GemmaModelInner {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> candle::Result<Tensor> {
        match self {
            Self::V2(m) => m.forward(input_ids, pos),
            Self::V3(m) => m.forward(input_ids, pos),
        }
    }

    fn format_prompt(&self, prompt: &str) -> String {
        match self {
            GemmaModelInner::V2(_) => prompt.to_string(),
            GemmaModelInner::V3(_) => {
                format!(
                    "<start_of_turn> user\n{}<end_of_turn>\n<start_of_turn> model\n",
                    prompt
                )
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::model::{Model, ModelSettings, ModelVersion};
    use dillhaven_assistant_util::huggingface::HuggingFaceApiManger;
    use std::sync::Arc;
    use tokio::sync::broadcast::Receiver;

    async fn read_output_to_end(mut output: Receiver<String>) -> String {
        let mut resp = String::new();
        while let Ok(msg) = output.recv().await {
            resp.push_str(&msg);
        }
        resp
    }

    #[tokio::test]
    // cannot run on GHA
    #[ignore]
    pub async fn test_model_init_v3_1b() {
        let model = model_init(ModelVersion::InstructV3_1B).await;

        let resp = model.prompt("hello world").expect("failed to prompt");
        let resp2 = model.prompt("what are you").expect("failed to prompt2");

        print!("Q: hello world\nA: ");
        read_output_to_end(resp).await;
        print!("Q: what are you\nA: ");
        read_output_to_end(resp2).await;
    }

    async fn model_init(version: ModelVersion) -> Model {
        dillhaven_assistant_observe::init();

        let hf_manager = Arc::new(
            HuggingFaceApiManger::new()
                .await
                .expect("failed to init hf manager"),
        );
        let settings = ModelSettings {
            version,
            seed: 299792458,
            temp: None,
            top_p: None,
        };

        Model::new(settings, hf_manager).await.unwrap()
    }
}
