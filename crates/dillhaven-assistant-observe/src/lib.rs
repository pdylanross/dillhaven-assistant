use std::sync::Once;
use tracing::subscriber::set_global_default;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

static INIT: Once = Once::new();

/// Initialize tracing with pretty console output
///
/// This function sets up a tracing subscriber that formats log messages
/// and outputs them to the console. It uses a pretty format with colors
/// when outputting to a terminal.
///
/// It's safe to call this function multiple times; it will only initialize
/// tracing once.
pub fn init() {
    INIT.call_once(|| {
        // Create a formatter for console output
        let fmt_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_ansi(true)
            .pretty();

        // Create a filter that allows INFO level and above by default
        // This can be overridden by setting the RUST_LOG environment variable
        let filter_layer =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

        // Combine the layers and set as the global subscriber
        let subscriber = tracing_subscriber::registry()
            .with(filter_layer)
            .with(fmt_layer);

        // Set the subscriber as the global default
        set_global_default(subscriber).expect("Failed to set global default subscriber");

        tracing::info!("Tracing initialized with pretty console output");
    });
}
