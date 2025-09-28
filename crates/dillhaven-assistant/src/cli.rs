use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use dillhaven_assistant_server::run_it;
use tracing::{info, instrument};

#[derive(Parser)]
#[command(about, version, author)]
#[command(propagate_version = true)]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run in server mode
    Server(ServerArgs),

    /// Run in thin-client mode
    Client(ClientArgs),
}

#[derive(Args, Debug)]
pub struct ServerArgs {}

impl ServerArgs {
    #[instrument()]
    pub async fn run(&self) -> Result<()> {
        info!("running server");
        run_it().await;
        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct ClientArgs {}

impl ClientArgs {
    #[instrument()]
    pub async fn run(&self) -> Result<()> {
        info!("running client");
        Ok(())
    }
}
