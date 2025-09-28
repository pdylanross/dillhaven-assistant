use anyhow::Result;
use dillhaven_assistant_llm::manager::LLMManager;
use dillhaven_assistant_speech_to_text::manager::STTManager;
use dillhaven_assistant_sync::lifespan::{AppState, LifespanManager};
use dillhaven_assistant_text_to_speech::manager::TTSManager;
use dillhaven_assistant_types::dialogue::{DialogueCoordinator, DialogueCoordinatorRef};
use dillhaven_assistant_util::huggingface::HuggingFaceApiManger;
use std::sync::Arc;
use tokio::runtime::Handle;
use tokio::select;
use tokio::sync::broadcast::Receiver;
use tracing::info;

pub async fn run_it() {
    // Initialize tracing
    dillhaven_assistant_observe::init();
    let current_rt = Handle::current();
    info!("Current runtime: {:?}", current_rt);

    let lifespan_manager = Arc::new(LifespanManager::new());
    let dialogue_coordinator = DialogueCoordinator::new();
    let hf_manager = Arc::new(
        HuggingFaceApiManger::new()
            .await
            .expect("Failed to initialize HuggingFace API Manager"),
    );

    let _stt_mgr = init_stt(
        lifespan_manager.clone(),
        hf_manager.clone(),
        dialogue_coordinator.clone(),
    )
    .await
    .expect("Failed to initialize STT");

    let llm_mgr = init_llm(
        lifespan_manager.clone(),
        hf_manager.clone(),
        _stt_mgr.clone(),
    )
    .await
    .expect("Failed to initialize LLM");

    let _tts_manager = init_tts(
        lifespan_manager.clone(),
        llm_mgr.add_text_receiver(),
        dialogue_coordinator.clone(),
    )
    .await
    .expect("failed to start tts");

    lifespan_manager.init();
    wait_for_shutdown(lifespan_manager).await;
}

async fn init_stt(
    lifespan_manager: Arc<LifespanManager>,
    hf_manager: Arc<HuggingFaceApiManger>,
    dialogue_coordinator: DialogueCoordinatorRef,
) -> Result<Arc<STTManager>> {
    let stt_manager = Arc::new(STTManager::new(
        lifespan_manager.clone(),
        hf_manager.clone(),
    )?);
    stt_manager.start(dialogue_coordinator).await?;

    let mut stt_stream_tx = stt_manager.add_text_receiver();
    tokio::spawn(async move {
        let mut state_rx = lifespan_manager.get_state_rx();
        loop {
            select! {
                res = stt_stream_tx.recv() => {
                    if let Ok(text) = res {
                        info!("TTS: {}", text);
                    }
                }
                _ = state_rx.changed() => {
                    let current_state = state_rx.borrow_and_update();
                    if let AppState::Shutdown = *current_state {
                        break;
                    }
                }
            }
        }
    });

    Ok(stt_manager)
}

async fn init_llm(
    lifespan_manager: Arc<LifespanManager>,
    hf_manager: Arc<HuggingFaceApiManger>,
    stt_manager: Arc<STTManager>,
) -> Result<Arc<LLMManager>> {
    let manager = Arc::new(LLMManager::new(hf_manager, lifespan_manager.clone())?);
    manager.run(stt_manager.add_text_receiver()).await?;

    let mut llm_stream_rx = manager.add_text_receiver();
    tokio::spawn(async move {
        let mut state_rx = lifespan_manager.get_state_rx();
        loop {
            select! {
                res = llm_stream_rx.recv() => {
                    if let Ok(text) = res {
                        info!("LLM resp: {}", text)
                    }
                }
                _ = state_rx.changed() => {
                    let current_state = state_rx.borrow_and_update();
                    if let AppState::Shutdown = *current_state {
                        break;
                    }
                }
            }
        }
    });

    Ok(manager)
}

async fn init_tts(
    lifespan_manager: Arc<LifespanManager>,
    text_input: Receiver<String>,
    dialogue_coordinator: DialogueCoordinatorRef,
) -> Result<Arc<TTSManager>> {
    let tts_manager = Arc::new(TTSManager::new(lifespan_manager.clone())?);

    tts_manager.start(text_input, dialogue_coordinator).await?;
    Ok(tts_manager)
}

async fn wait_for_shutdown(lifespan_manager: Arc<LifespanManager>) {
    let mut shutdown_barrier = lifespan_manager.get_shutdown_barrier().await;

    shutdown_barrier.wait();
}
