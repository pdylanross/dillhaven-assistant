use anyhow::Result;
use piper_rs::synth::PiperSpeechSynthesizer;
use piper_rs::{PiperAudioResult, from_config_path};
use std::path::Path;

pub const MODEL_PATH: &str = "/home/dylan/Documents/git/dillhaven-assistant/piper-finetune/data/en_US-patrickstewartemote-medium.onnx.json";

#[derive(Debug, Clone)]
pub struct PiperModelConfig {
    pub config_path: String,
}

pub struct PiperModel {
    _config: PiperModelConfig,
    synth: PiperSpeechSynthesizer,
    sample_rate: usize,
}

impl PiperModel {
    pub fn new(config: PiperModelConfig) -> Result<Self> {
        let model = from_config_path(Path::new(&config.config_path))?;
        let sample_rate = model.audio_output_info()?.sample_rate;
        let synth = PiperSpeechSynthesizer::new(model.clone())?;

        Ok(PiperModel {
            synth,
            sample_rate,
            _config: config,
        })
    }

    pub fn get_sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub fn synth(&self, text: String) -> Result<impl Iterator<Item = PiperAudioResult>> {
        self.synth
            .synthesize_parallel(text, None)
            .map_err(|e| anyhow::anyhow!(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dillhaven_assistant_audio::audio_playback::AudioPlayback;
    use dillhaven_assistant_types::dialogue::DialogueCoordinator;

    #[tokio::test]
    // cannot run on GHA
    #[ignore]
    async fn test_piper_model() {
        let config = PiperModelConfig {
            config_path: "/home/dylan/Documents/git/dillhaven-assistant/piper-finetune/data/en_US-patrickstewartemote-medium.onnx.json".to_string(),
        };
        let model = PiperModel::new(config).unwrap();
        let text = "Rereading my original post, or makes me wonder if Rodio doesn't strip null bytes.

We're getting a bunch of Hey me too comments, which isn't very helpful. Could we get some code of what y'all are doing and see what's in common? Is this specifically to Linux, or does it apply to other OS' too? If it does apply to Linux, which Distros and audio software are y'all using?

I'm using Ubuntu and PulseAudio.".to_string();
        let result = model.synth(text).unwrap();
        let dc = DialogueCoordinator::new();

        let output = AudioPlayback::new(dc).unwrap();
        let in_tx = output.get_in_tx();

        for res in result {
            let res = res.unwrap();

            let resampled = kaudio::resample(
                res.samples.into_vec().as_slice(),
                res.info.sample_rate,
                output.get_sample_rate() as usize,
            )
            .expect("failed to resample audio");

            in_tx.send(resampled).await.expect("failed to send audio");
        }

        tokio::time::sleep(std::time::Duration::from_secs(20)).await;
    }
}
