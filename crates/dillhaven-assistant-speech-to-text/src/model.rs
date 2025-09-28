use crate::config::Config;
use anyhow::Result;
use candle_core::{Device, Tensor};
use dillhaven_assistant_audio::audio_capture::get_sample_rate;
use dillhaven_assistant_util::candle::get_device;
use dillhaven_assistant_util::huggingface::HuggingFaceApiManger;
use sentencepiece::SentencePieceProcessor;
use std::sync::Arc;
use tracing::info;

pub struct MoshiModel {
    state: moshi::asr::State,
    text_tokenizer: SentencePieceProcessor,
    dev: Device,
    timestamps: bool,
    _vad: bool,
    _config: Config,
}

const HF_REPO: &str = "kyutai/stt-1b-en_fr-candle";
const MODEL_PATH: &str = "model.safetensors";
const TIMESTAMPS: bool = false;
const VAD: bool = true;

impl MoshiModel {
    pub async fn load_from_hf(hf_manager: Arc<HuggingFaceApiManger>) -> Result<Self> {
        // Initialize the model
        let dev = get_device()?;
        info!("Initializing speech-to-text model with {:?}", dev);

        // Retrieve the model files from the Hugging Face Hub
        let api = hf_manager.get_async_api();
        let repo = api.model(HF_REPO.to_string());
        let config_file = repo.get("config.json").await?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
        let tokenizer_file = repo.get(&config.tokenizer_name).await?;
        let model_file = repo.get(MODEL_PATH).await?;
        let mimi_file = repo.get(&config.mimi_name).await?;
        let is_quantized = model_file.to_str().unwrap().ends_with(".gguf");

        let text_tokenizer = SentencePieceProcessor::open(&tokenizer_file)?;

        let lm = if is_quantized {
            let vb_lm = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &model_file,
                &dev,
            )?;
            moshi::lm::LmModel::new(
                &config.model_config(VAD),
                moshi::nn::MaybeQuantizedVarBuilder::Quantized(vb_lm),
            )?
        } else {
            let dtype = dev.bf16_default_to_f32();
            let vb_lm = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &dev)?
            };
            moshi::lm::LmModel::new(
                &config.model_config(VAD),
                moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
            )?
        };

        let audio_tokenizer = moshi::mimi::load(mimi_file.to_str().unwrap(), Some(32), &dev)?;
        let asr_delay_in_tokens = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(1, asr_delay_in_tokens, 0., audio_tokenizer, lm)?;

        info!("Speech-to-text model initialized successfully");
        Ok(MoshiModel {
            state,
            _config: config,
            text_tokenizer,
            timestamps: TIMESTAMPS,
            _vad: VAD,
            dev,
        })
    }

    /// Enable or disable timestamps for recognized words
    pub fn set_timestamps(&mut self, enable: bool) {
        self.timestamps = enable;
    }

    /// Process an audio chunk and call the callback with recognized text
    ///
    /// If timestamps are enabled, the callback will be called with text in the format:
    /// "[start_time-stop_time] word"
    pub fn process_chunk<F>(&mut self, pcm: &[f32], mut callback: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        let pcm = kaudio::resample(pcm, get_sample_rate() as usize, 24000)?;
        let len = pcm.len();

        let pcm_tensor = Tensor::new(pcm, &self.dev)?
            .reshape((1, 1, ()))?
            .broadcast_as((self.state.batch_size(), 1, len))?;
        let asr_msgs = self
            .state
            .step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

        for asr_msg in asr_msgs.iter() {
            match asr_msg {
                moshi::asr::AsrMsg::Step { .. } => {}
                moshi::asr::AsrMsg::EndWord { .. } => {}
                moshi::asr::AsrMsg::Word { tokens, .. } => {
                    let word = self
                        .text_tokenizer
                        .decode_piece_ids(&tokens)
                        .unwrap_or_else(|_| String::new());

                    callback(&word);
                }
            }
        }

        Ok(())
    }
}
