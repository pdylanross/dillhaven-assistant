mod broadcast;
mod mspc;
mod transmit;

use async_trait::async_trait;
pub use broadcast::*;
pub use mspc::*;
pub use transmit::*;

use std::sync::Arc;

pub type DynItemProcessor<In, Out> = Arc<dyn ItemProcessor<In, Out> + Send + Sync>;

#[async_trait]
pub trait ItemProcessor<In, Out> {
    async fn process(&self, item: In) -> anyhow::Result<Option<Out>>;
    async fn done(&self) -> anyhow::Result<Option<Out>>;
}
