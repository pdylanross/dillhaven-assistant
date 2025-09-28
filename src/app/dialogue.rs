use std::cmp::PartialEq;
use std::fmt::Debug;
use std::future::Future;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use tracing::{info, instrument};

pub type DialogueCoordinatorRef = Arc<DialogueCoordinator>;

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(usize)]
pub enum DialogueMode {
    PassiveListening = 0,
    ActiveListening = 1,
    Speaking = 2,
}

impl DialogueMode {
    pub fn from_usize(val: usize) -> Option<Self> {
        match val {
            0 => Some(DialogueMode::PassiveListening),
            1 => Some(DialogueMode::ActiveListening),
            2 => Some(DialogueMode::Speaking),
            _ => None,
        }
    }
}

pub struct DialogueCoordinator {
    current_state: AtomicUsize,
}

impl Debug for DialogueCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DialogueCoordinator {{ current_state: {:?} }}",
            self.get_current_state()
        )
    }
}

impl DialogueCoordinator {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            current_state: AtomicUsize::new(DialogueMode::PassiveListening as usize),
        })
    }

    pub fn get_current_state(&self) -> DialogueMode {
        let cur = self
            .current_state
            .load(std::sync::atomic::Ordering::Relaxed);
        DialogueMode::from_usize(cur).unwrap_or(DialogueMode::PassiveListening)
    }

    pub fn set_current_state(&self, state: DialogueMode) {
        info!("Setting dialogue state to: {:?}", state);
        self.current_state
            .store(state as usize, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn set_current_state_for_duration(
        self: &Arc<Self>,
        state: DialogueMode,
        duration: std::time::Duration,
        next_state: DialogueMode,
    ) {
        self.set_current_state(state);
        let s2 = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(duration).await;
            s2.set_current_state(next_state);
        });
    }

    #[instrument(skip(f))]
    pub fn run_in_state<F, O>(&self, state: DialogueMode, mut f: F)
    where
        F: FnMut(),
    {
        let cur = self.get_current_state();
        if cur == state {
            f();
        }
    }

    #[instrument(skip(f))]
    pub async fn run_in_state_async<FN, Fut>(&self, state: DialogueMode, mut f: FN)
    where
        FN: FnMut() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let cur = self.get_current_state();
        if cur == state {
            f().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use tokio::sync::Mutex;

    #[test]
    fn test_dialogue_mode_from_usize() {
        assert_eq!(
            DialogueMode::from_usize(0),
            Some(DialogueMode::PassiveListening)
        );
        assert_eq!(
            DialogueMode::from_usize(1),
            Some(DialogueMode::ActiveListening)
        );
        assert_eq!(DialogueMode::from_usize(2), Some(DialogueMode::Speaking));
        assert_eq!(DialogueMode::from_usize(3), None);
    }

    #[test]
    fn test_new_coordinator_default_state() {
        let coordinator = DialogueCoordinator::new();
        assert_eq!(
            coordinator.get_current_state(),
            DialogueMode::PassiveListening
        );
    }

    #[test]
    fn test_set_and_get_state() {
        let coordinator = DialogueCoordinator::new();

        // Test initial state
        assert_eq!(
            coordinator.get_current_state(),
            DialogueMode::PassiveListening
        );

        // Test setting to ActiveListening
        coordinator.set_current_state(DialogueMode::ActiveListening);
        assert_eq!(
            coordinator.get_current_state(),
            DialogueMode::ActiveListening
        );

        // Test setting to Speaking
        coordinator.set_current_state(DialogueMode::Speaking);
        assert_eq!(coordinator.get_current_state(), DialogueMode::Speaking);

        // Test setting back to PassiveListening
        coordinator.set_current_state(DialogueMode::PassiveListening);
        assert_eq!(
            coordinator.get_current_state(),
            DialogueMode::PassiveListening
        );
    }

    #[test]
    fn test_run_in_state() {
        let coordinator = DialogueCoordinator::new();
        let counter = Arc::new(AtomicUsize::new(0));

        // Set state to ActiveListening
        coordinator.set_current_state(DialogueMode::ActiveListening);

        // Function should run when state matches
        {
            let counter_clone = counter.clone();
            coordinator.run_in_state::<_, ()>(DialogueMode::ActiveListening, move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Function should not run when the state doesn't match
        {
            let counter_clone = counter.clone();
            coordinator.run_in_state::<_, ()>(DialogueMode::Speaking, move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1); // Counter should not change
    }

    #[tokio::test]
    async fn test_run_in_state_async() {
        let coordinator = DialogueCoordinator::new();
        let counter = Arc::new(Mutex::new(0));

        // Set state to Speaking
        coordinator.set_current_state(DialogueMode::Speaking);

        // Function should run when state matches
        {
            let counter_clone = counter.clone();
            coordinator
                .run_in_state_async(DialogueMode::Speaking, move || {
                    let value = counter_clone.clone();
                    async move {
                        let mut lock = value.lock().await;
                        *lock += 1;
                    }
                })
                .await;
        }
        assert_eq!(*counter.lock().await, 1);

        // Function should not run when the state doesn't match
        {
            let counter_clone = counter.clone();
            coordinator
                .run_in_state_async(DialogueMode::PassiveListening, move || {
                    let value = counter_clone.clone();
                    async move {
                        let mut lock = value.lock().await;
                        *lock += 1;
                    }
                })
                .await;
        }
        assert_eq!(*counter.lock().await, 1); // Counter should not change
    }
}
