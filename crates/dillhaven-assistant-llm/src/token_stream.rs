use anyhow::Result;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::broadcast::{Receiver, Sender, channel};
use tokio::task::yield_now;
use tracing::{instrument, trace};

pub struct TokenizerHandler {
    tokenizer: Arc<Tokenizer>,
    stream_tx: Sender<String>,
}

impl TokenizerHandler {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let (stream_tx, _) = channel(100);

        Self {
            tokenizer,
            stream_tx,
        }
    }

    pub fn get_rx(&self) -> Receiver<String> {
        self.stream_tx.subscribe()
    }

    pub fn get_tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    #[instrument(skip(self))]
    pub async fn next_token(&self, token: u32) -> Result<()> {
        let tokens = vec![token];
        let text = self.tokenizer.decode(&tokens, true);
        if let Err(e) = text {
            anyhow::bail!("{:?}", e);
        }
        let text = text.unwrap();

        if !text.is_empty() {
            trace!("sending token text: {}", text);
            self.stream_tx.send(text)?;

            // if we don't yield back to tokio, the rx's for this will never get notified
            // of the send operation.
            yield_now().await;
        }

        Ok(())
    }
}

impl Drop for TokenizerHandler {
    #[instrument(skip(self))]
    fn drop(&mut self) {
        trace!("dropping TokenizerHandler");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dillhaven_assistant_util::huggingface::HuggingFaceApiManger;
    use tokio::sync::Barrier;
    use tracing::info;

    async fn get_tokenizer() -> Tokenizer {
        let hf_manager = HuggingFaceApiManger::new().await.expect("hf api");
        let api = hf_manager.get_async_api();
        let repo = api.model("google/gemma-2-2b-it".to_string());

        let tokenizer_filename = repo
            .get("tokenizer.json")
            .await
            .expect("Failed to get tokenizer.json");

        Tokenizer::from_file(tokenizer_filename).unwrap()
    }

    #[tokio::test]
    pub async fn test_tokenizer() {
        dillhaven_assistant_observe::init();

        let tok = get_tokenizer().await;
        let stream = TokenizerHandler::new(Arc::new(tok.clone()));
        let mut tok_rx = stream.get_rx();
        let barrier = Arc::new(Barrier::new(2));

        let b2 = barrier.clone();
        tokio::spawn(async move {
            while let Ok(msg) = tok_rx.recv().await {
                info!("{:?}", msg);
            }
            b2.wait().await;
        });

        let tokens = tok
            .encode(
                "test one two three. A new sentence! some other..... stuff. {\"key\": \"value\"}",
                true,
            )
            .expect("true");
        let tokens = tokens.get_ids().to_vec();

        for tok in tokens {
            stream.next_token(tok).await.expect("no err");
        }
        drop(stream);
        barrier.wait().await;
    }
}
