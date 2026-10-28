//! Embedded single-datastore support and the interactive `pivot open` shell.

use std::path::PathBuf;

mod copy;
mod instance;
mod parser;
mod progress;
mod render;
mod repl;
mod target;

pub use instance::{ShellInstance, ShellLimits};
pub use repl::Error;
pub use target::ShellTarget;

/// Input the shell runs instead of starting its interactive prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShellInput {
    /// SQL statements or backslash commands given inline.
    Command(String),
    /// A file of SQL statements and backslash commands; `-` reads stdin.
    File(PathBuf),
}

/// Open an interactive SQL shell over one pivotlake datastore, named by a local
/// directory or an object-store URI.
pub fn run(datastore_location: String) -> Result<(), Error> {
    run_with_limits(
        ShellTarget::pivot(datastore_location),
        ShellLimits::default(),
        Vec::new(),
    )
}

/// Open a SQL shell over `target` with optional resource limits. With
/// `inputs`, run them in order and exit; without, start the interactive
/// prompt.
pub fn run_with_limits(
    target: ShellTarget,
    limits: ShellLimits,
    inputs: Vec<ShellInput>,
) -> Result<(), Error> {
    crate::server::raise_open_file_limit();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(repl::run_shell(target, limits, inputs))
}
