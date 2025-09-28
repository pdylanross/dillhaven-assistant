//! The `TokenOutputStream` provides a mechanism to decode token IDs into text and stream the decoded
//! tokens using a broadcast channel. It is useful for scenarios where tokens are processed and
//! streamed in real-time to multiple subscribers.
//!
//! # Fields
//! * `tokenizer` - An `Arc` wrapped instance of `Tokenizer` used for decoding token IDs into text.
//! * `stream_tx` - The sender part of a `tokio::sync::broadcast` channel used to send decoded text streams.
//!
//! # Methods
//!
//! ## `new`
//! Creates a new instance of `TokenOutputStream`.
//!
//! ### Arguments
//! * `tokenizer` - An `Arc` wrapped `Tokenizer` required for token decoding.
//!
//! ### Returns
//! * An instance of `TokenOutputStream`.
//!
//! ```
//! let tokenizer = Arc::new(Tokenizer::from_file("path/to/tokenizer.json").unwrap());
//! let stream = TokenOutputStream::new(tokenizer);
//! ```
//!
//! ## `get_rx`
//! Retrieves a `Receiver` from the broadcast channel that allows clients to subscribe
//! and listen for streamed decoded tokens.
//!
//! ### Returns
//! * A `tokio::sync::broadcast::Receiver<String>` to receive decoded token text.
//!
//! ```
//! let receiver = stream.get_rx();
//! tokio::spawn(async move {
//!     while let Ok(msg) = receiver.recv().await {
//!         println!("Received: {:?}", msg);
//!     }
//! });
//! ```
//!
//! ## `next_token`
//! Decodes a given token ID into text and sends it to the broadcast channel.
//!
//! ### Arguments
//! * `token` - A `u32` token ID to decode.
//!
//! ### Returns
//! * A `Result<()>` indicating success or an error.
//!
//! ### Errors
//! * If decoding fails, it returns an `anyhow::Error`.
//! * If broadcasting the decoded text fails, it returns an error from the `tokio::sync::broadcast` channel.
//!
//! ```
//! let token_id: u32 = 1234; // Example token ID
//! stream.next_token(token_id).expect("Token should be processed and sent successfully");
//! ```
//!
//! # Example
//! ```
//! use std::sync::Arc;
//! use tokenizers::Tokenizer;
//! use tokio::sync::broadcast::Receiver;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create tokenizer instance
//!     let tokenizer = Arc::new(Tokenizer::from_file("path/to/tokenizer.json").unwrap());
//!     let stream = TokenOutputStream::new(tokenizer.clone());
//!
//!     // Subscribe to token stream
//!     let mut receiver: Receiver<String> = stream.get_rx();
//!
//!     tokio::spawn(async move {
//!         while let Ok(msg) = receiver.recv().await {
//!             println!("Received token text: {:?}", msg);
//!         }
//!     });
//!
//!     // Example token IDs to be processed
//!     let tokens = vec![123, 456, 789];
//!
//!     for token in tokens {
//!         stream.next_token(token).expect("Failed to process token");
//!     }
//! }
//! ```
//!
//! # Unit Tests
//! The `tests` module provides test coverage for the `TokenOutputStream` and its interactions with a tokenizer.
//! It demonstrates fetching a tokenizer from an external source, encoding input strings into tokens,
//! using the `TokenOutputStream` to decode tokens, and ensuring the decoded text is correctly streamed.
//!
//! ## Test: `test_tokenizer`
//! * Verifies the `TokenOutputStream` can be used to stream tokenized input text as decoded output.
//! * Tests interaction with the Hugging Face API Manager for dynamic model and tokenizer retrieval.
//! * Confirms that each token is successfully decoded and streamed.
//!
//! Uses:
//! * `observe` for logging initialization.
//! * `tokio::sync::Barrier` to synchronize concurrent testing tasks.
//! * `tokio::spawn` to launch a subscriber task for receiving streamed decoded tokens.
use anyhow::Result;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::broadcast::{channel, Receiver, Sender};
use tokio::task::yield_now;
use tracing::{instrument, trace};

pub struct TokenizerHandler {
    tokenizer: Arc<Tokenizer>,
    stream_tx: Sender<String>,
}

impl TokenizerHandler {
    /// Creates a new instance of the struct.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - An `Arc<Tokenizer>` that provides a tokenizer implementation
    ///                 shared across threads.
    ///
    /// # Returns
    ///
    /// Returns a new instance of the struct, initializing it with the provided tokenizer
    /// and creating a bounded channel with a capacity of 100 for internal use.
    ///
    /// # Example
    /// ```rust
    /// let tokenizer = Arc::new(Tokenizer::new());
    /// let instance = YourStruct::new(tokenizer);
    /// ```
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let (stream_tx, _) = channel(100);

        Self {
            tokenizer,
            stream_tx,
        }
    }

    /// Returns a new `Receiver<String>` for subscribing to the shared broadcast channel.
    ///
    /// This function allows the caller to get a new subscription to the `stream_tx` broadcast
    /// channel owned by the current instance of the struct. By subscribing, the caller can
    /// asynchronously receive `String` messages that are sent through the broadcast channel.
    ///
    /// # Returns
    ///
    /// A `Receiver<String>` which can be used to listen for messages broadcasted on this channel.
    ///
    /// # Example
    ///
    /// ```
    /// let receiver = instance.get_rx();
    ///
    /// tokio::spawn(async move {
    ///     while let Ok(message) = receiver.recv().await {
    ///         println!("Received: {}", message);
    ///     }
    /// });
    /// ```
    ///
    /// Note that each call to `get_rx` will create a new, independent subscription to the channel.
    /// All receivers subscribed to the channel will receive the same broadcasted messages unless
    /// they fall behind the broadcast queue's capacity, in which case older messages may be dropped.
    pub fn get_rx(&self) -> Receiver<String> {
        self.stream_tx.subscribe()
    }

    pub fn get_tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    /// Processes a given token, decodes it to its corresponding text, and sends the decoded text to a stream channel if it's not empty.
    ///
    /// # Arguments
    /// * `token` - A `u32` token to decode into text.
    ///
    /// # Returns
    /// * `Result<()>` - Returns `Ok(())` if processing and sending are successful, otherwise returns an error.
    ///
    /// # Errors
    /// This function will return an error in the following cases:
    /// * If the token decoding fails, an error with the decoding failure reason will be returned.
    /// * If there is an issue sending the decoded text to the stream channel, it will propagate the error.
    ///
    /// # Behavior
    /// 1. Converts the provided token into a vector of tokens.
    /// 2. Attempts to decode the token vector into a text string.
    ///    - If the decoding fails, the function returns the error wrapped in `anyhow::bail!`.
    /// 3. Checks if the resulting text is non-empty.
    ///    - If the text is non-empty, it sends the text to the `stream_tx` channel.
    /// 4. Returns `Ok(())` if all operations are successful.
    ///
    /// # Example
    /// ```rust
    /// let result = instance.next_token(42);
    /// if let Err(e) = result {
    ///     eprintln!("Error occurred: {}", e);
    /// }
    /// ```
    #[instrument(skip(self))]
    pub async fn next_token(&self, token: u32) -> Result<()> {
        let tokens = vec![token];
        let text = self.tokenizer.decode(&tokens, true);
        if let Err(e) = text {
            anyhow::bail!("{:?}", e);
        }
        let text = text.unwrap();

        if text.len() > 0 {
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
    use crate::app::huggingface::HuggingFaceApiManger;
    use crate::observe;
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

        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();
        tokenizer
    }

    #[tokio::test]
    pub async fn test_tokenizer() {
        observe::init();

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
