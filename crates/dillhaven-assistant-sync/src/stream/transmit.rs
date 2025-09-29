pub fn mspc_rx_to_broadcast_tx<T>(
    mut rx: tokio::sync::mpsc::Receiver<T>,
    tx: tokio::sync::broadcast::Sender<T>,
) where
    T: Send + Sync + 'static,
{
    tokio::spawn(async move {
        while let Some(val) = rx.recv().await {
            if tx.send(val).is_err() {
                return;
            }
        }
    });
}

pub fn broadcast_rx_to_mspc_tx<T>(
    mut rx: tokio::sync::broadcast::Receiver<T>,
    tx: tokio::sync::mpsc::Sender<T>,
) where
    T: Clone + Send + Sync + 'static,
{
    tokio::spawn(async move {
        while let Ok(val) = rx.recv().await {
            if tx.send(val).await.is_err() {
                return;
            }
        }
    });
}

pub fn mpsc_rx_to_mspc_tx<T>(
    mut rx: tokio::sync::mpsc::Receiver<T>,
    tx: tokio::sync::mpsc::Sender<T>,
) where
    T: Clone + Send + Sync + 'static,
{
    tokio::spawn(async move {
        while let Some(val) = rx.recv().await {
            if tx.send(val).await.is_err() {
                return;
            }
        }
    });
}
