use crate::app::lifespan::{AppState, LifespanManager};
use crate::app::timer::ResettableTimer;
use std::sync::Arc;
use std::time::Duration;
use tokio::select;
use tokio::sync::broadcast::{Receiver, Sender};

const INPUT_DELAY: f32 = 1.5;

pub struct TextToSpeechOutputStream {
    input_tx: Sender<String>,
    lifespan_manager: Arc<LifespanManager>,
    output_tx: Sender<String>,
}

impl TextToSpeechOutputStream {
    pub fn new(input_tx: Sender<String>, lifespan_manager: Arc<LifespanManager>) -> Self {
        let (output_tx, _) = tokio::sync::broadcast::channel(100);

        Self {
            input_tx,
            lifespan_manager,
            output_tx,
        }
    }

    pub fn get_output_tx(&self) -> Receiver<String> {
        self.output_tx.subscribe()
    }

    pub fn start(&self) {
        let timer = ResettableTimer::new(Duration::from_secs_f32(INPUT_DELAY));
        let mut input_rx = self.input_tx.subscribe();
        let output_tx = self.output_tx.clone();
        let lifespan_manager = self.lifespan_manager.clone();

        tokio::spawn(async move {
            let mut current_message = String::new();

            let mut init = lifespan_manager.get_init_barrier().await;
            init.wait();

            let mut state_rx = lifespan_manager.get_state_rx();

            loop {
                select! {
                    _ = timer.wait() => {
                        if !current_message.is_empty() {
                            output_tx.send(current_message.clone()).unwrap();
                            current_message.clear();
                        } else {
                            timer.pause().await;
                        }
                    },
                    message = input_rx.recv() => {
                        if let Err(err) = message {
                            lifespan_manager.crash(anyhow::Error::from(err));
                            break;
                        }

                        if let Ok(message) = message {
                            current_message.push_str(&message.trim_end());
                            current_message.push(' ');
                            timer.restart().await;
                        }
                    },
                    _ = state_rx.changed() => {
                        let val = state_rx.borrow_and_update();
                        if let AppState::Shutdown = *val {
                            break;
                        }
                    }
                }
            }
        });
    }
}
