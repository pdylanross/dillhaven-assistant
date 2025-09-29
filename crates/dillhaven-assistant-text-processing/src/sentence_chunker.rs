//! The `StreamSentenceChunker` module provides a utility to process a stream of text messages
//! chunked at sentence boundaries based on predefined or custom delimiters.

use anyhow::Result;
use async_trait::async_trait;
use dillhaven_assistant_sync::stream::{BroadcastStreamProcessor, ItemProcessor};
use std::sync::Arc;
use tokio::sync::broadcast::Receiver;
use tokio::sync::Mutex;

pub const DEFAULT_SPLIT_TOKENS: [char; 3] = ['.', '!', '?'];

pub struct StreamSentenceChunker {
    split_tokens: Vec<char>,
    current_string: Mutex<String>,
}

#[async_trait]
impl ItemProcessor<String, String> for StreamSentenceChunker {
    async fn process(&self, item: String) -> Result<Option<String>> {
        let mut lck = self.current_string.lock().await;

        lck.push_str(&item);
        if !lck.is_empty() {
            let last = last_non_whitespace(&lck);
            if let Some(last) = last {
                if self.split_tokens.contains(&last) {
                    let ret = lck.clone().trim().to_string();
                    lck.clear();
                    return Ok(Some(ret));
                }
            }
        }
        Ok(None)
    }

    async fn done(&self) -> Result<Option<String>> {
        let lck = self.current_string.lock().await;
        let lck = lck.trim().to_string();
        if !lck.is_empty() {
            Ok(Some(lck))
        } else {
            Ok(None)
        }
    }
}

fn last_non_whitespace(s: &str) -> Option<char> {
    s.chars().rev().find(|c| !c.is_whitespace())
}

impl StreamSentenceChunker {
    pub fn new(split_tokens: Vec<char>) -> Self {
        Self {
            split_tokens,
            current_string: Mutex::new(String::new()),
        }
    }

    pub fn new_processor(
        split_tokens: Vec<char>,
        in_rx: Receiver<String>,
    ) -> BroadcastStreamProcessor<String> {
        let processor = Arc::new(Self::new(split_tokens));
        BroadcastStreamProcessor::new(in_rx, 20, processor)
    }

    pub fn new_processor_with_default_tokens(
        in_rx: Receiver<String>,
    ) -> BroadcastStreamProcessor<String> {
        Self::new_processor(DEFAULT_SPLIT_TOKENS.to_vec(), in_rx)
    }

    pub fn new_default_tokens() -> Self {
        Self::new(DEFAULT_SPLIT_TOKENS.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn send_split(message: String, tx: &tokio::sync::broadcast::Sender<String>) {
        let parts = message.split(' ');
        for p in parts {
            tx.send(format!(" {}", p)).unwrap();
        }
    }

    #[tokio::test]
    async fn test_stream_sentence_chunker() {
        dillhaven_assistant_observe::init();
        let (tx, rx) = tokio::sync::broadcast::channel(10);
        let chunker = StreamSentenceChunker::new_processor_with_default_tokens(rx);

        let mut chunk_rx = chunker.get_result_rx();

        tx.send("Hello world!".to_string()).unwrap();
        assert_eq!(chunk_rx.recv().await.unwrap(), "Hello world!");
    }

    #[tokio::test]
    async fn test_stream_sentence_chunker_multiple_sentences() {
        dillhaven_assistant_observe::init();
        let (tx, rx) = tokio::sync::broadcast::channel(10);
        let chunker = StreamSentenceChunker::new_processor_with_default_tokens(rx);
        let mut results = Vec::new();

        let mut chunk_rx = chunker.get_result_rx();

        send_split(
            "One Sentence. Two sentence! Three sentence?".to_string(),
            &tx,
        );
        drop(tx);
        drop(chunker);

        while let Ok(result) = chunk_rx.recv().await {
            results.push(result);
        }

        let expected = vec!["One Sentence.", "Two sentence!", "Three sentence?"];
        assert_eq!(results, expected);
    }
}
