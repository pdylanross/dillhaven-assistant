mod cli;

use crate::cli::{Cli, Commands};
use anyhow::Result;
use clap::Parser;

#[tokio::main()]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    dillhaven_assistant_observe::init();
    match cli.command {
        Commands::Server(s) => {
            s.run().await?;
            Ok(())
        }
        Commands::Client(c) => {
            c.run().await?;
            Ok(())
        }
    }
}
