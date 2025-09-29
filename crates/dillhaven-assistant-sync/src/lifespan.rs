use adaptive_barrier::PanicMode;
use std::sync::Arc;
use tokio::sync::{RwLock, watch};
use tokio::{select, signal};
use tracing::{error, info, warn};

#[derive(Debug, Clone, Copy)]
pub enum AppState {
    Initializing,
    Ready,
    Shutdown,
}

pub struct LifespanManager {
    current_state_tx: Arc<watch::Sender<AppState>>,
    init_barrier: Arc<RwLock<adaptive_barrier::Barrier>>,
    shutdown_barrier: Arc<RwLock<adaptive_barrier::Barrier>>,
}

impl Default for LifespanManager {
    fn default() -> Self {
        Self::new()
    }
}

impl LifespanManager {
    pub fn new() -> Self {
        let (current_state_tx, _) = watch::channel(AppState::Initializing);
        let init_barrier = Arc::new(RwLock::new(adaptive_barrier::Barrier::new(
            PanicMode::Poison,
        )));
        let shutdown_barrier = Arc::new(RwLock::new(adaptive_barrier::Barrier::new(
            PanicMode::Poison,
        )));

        Self {
            current_state_tx: Arc::new(current_state_tx),
            init_barrier,
            shutdown_barrier,
        }
    }

    pub fn init(&self) {
        info!("starting app");
        let state_tx = self.current_state_tx.clone();
        let mut state_rx = self.current_state_tx.subscribe();
        let init_barrier = self.init_barrier.clone();
        let shutdown_barrier = self.shutdown_barrier.clone();

        tokio::spawn(async move {
            // wait for all app initializations
            {
                init_barrier.write().await.wait();
            }

            // ensure we didn't crash during startup
            let cur_state = { *state_rx.borrow() };

            if let AppState::Shutdown = cur_state {
                let mut shutdown = shutdown_barrier.write().await;
                shutdown.wait();

                return;
            }

            // set state to ready
            state_tx
                .send(AppState::Ready)
                .expect("failed to send ready signal");
            loop {
                // handle any events that would cause a shutdown
                select! {
                    _ = signal::ctrl_c() => {
                        if let Err(e) = state_tx.send(AppState::Shutdown) {
                            warn!("err sending shutdown signal: {:?}", e)
                        }
                    }
                    err = state_rx.changed() => {
                        if let Err(e) = err {
                            warn!("err receiving state change: {:?}", e);
                            break
                        }

                        let current_state = state_rx.borrow_and_update();
                        info!("app state changed: {:?}", *current_state);
                        match *current_state {
                            AppState::Shutdown => break,
                            _ => continue,
                        }
                    }
                }
            }

            info!("app shutting down");
            if let Err(e) = state_tx.send(AppState::Shutdown) {
                warn!("err sending shutdown signal: {:?}", e)
            }

            let mut shutdown = shutdown_barrier.write().await;
            shutdown.wait();
        });
    }

    pub fn crash(&self, reason: anyhow::Error) {
        error!("app crashed: {}", reason);
        self.current_state_tx
            .send(AppState::Shutdown)
            .expect("failed to send shutdown signal");
    }

    pub fn get_state(&self) -> AppState {
        *self.current_state_tx.borrow()
    }

    pub fn get_state_rx(&self) -> watch::Receiver<AppState> {
        self.current_state_tx.subscribe()
    }

    pub async fn get_init_barrier(&self) -> adaptive_barrier::Barrier {
        let init_guard = self.init_barrier.read().await;
        init_guard.clone()
    }

    pub async fn get_shutdown_barrier(&self) -> adaptive_barrier::Barrier {
        let guard = self.shutdown_barrier.read().await;

        guard.clone()
    }

    pub fn get_shutdown_signaler(&self) -> LifespanShutdownSignaler {
        LifespanShutdownSignaler::new(self.get_state_rx())
    }
}

pub struct LifespanShutdownSignaler {
    rx: watch::Receiver<AppState>,
}

impl LifespanShutdownSignaler {
    fn new(rx: watch::Receiver<AppState>) -> Self {
        Self { rx }
    }

    pub async fn flagged_for_shutdown(&mut self) -> bool {
        let event = self.rx.changed().await;
        if event.is_err() {
            return true;
        }

        let current_val = self.rx.borrow_and_update();
        matches!(*current_val, AppState::Shutdown)
    }
}
