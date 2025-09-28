use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct ResettableTimer {
    inner: Arc<RwLock<ResettableTimerInner>>,
}

impl ResettableTimer {
    pub fn new(duration: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(ResettableTimerInner::new(duration))),
        }
    }

    pub async fn pause(&self) {
        self.inner.write().await.pause();
    }

    pub async fn set_duration(&self, duration: Duration) {
        self.inner.write().await.set_duration(duration);
    }

    pub async fn restart(&self) {
        self.inner.write().await.restart();
    }

    pub async fn wait(&self) {
        loop {
            if self.timer_is_elapsed().await {
                return;
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn timer_is_elapsed(&self) -> bool {
        self.inner.read().await.timer_is_elapsed()
    }
}

#[derive(Debug)]
struct ResettableTimerInner {
    awake_at: Option<Instant>,
    duration: Duration,
}

impl ResettableTimerInner {
    fn new(duration: Duration) -> Self {
        Self {
            awake_at: None,
            duration,
        }
    }

    pub fn set_duration(&mut self, duration: Duration) {
        self.duration = duration;
    }

    pub fn pause(&mut self) {
        self.awake_at = None;
    }

    pub fn restart(&mut self) {
        self.awake_at = Some(Instant::now() + self.duration);
    }

    pub fn timer_is_elapsed(&self) -> bool {
        if let Some(awake_at) = self.awake_at {
            if awake_at <= Instant::now() {
                return true;
            }
        }

        false
    }
}
