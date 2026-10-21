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
pub use target::ShellTarget;

/// Input the shell runs instead of starting its interactive prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShellScript {
    /// SQL statements or backslash commands given inline.
    Command(String),
    /// A file of SQL statements and backslash commands; `-` reads stdin.
    File(PathBuf),
}

/// How a shell run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellExit {
    Success,
    /// A script statement or command failed, which stopped every script.
    ScriptFailed,
}

/// Open an interactive SQL shell over one pivotlake datastore, named by a local
/// directory or an object-store URI.
pub fn run(datastore_location: String) -> Result<ShellExit, Box<dyn std::error::Error>> {
    run_with_limits(
        ShellTarget::pivot(datastore_location),
        ShellLimits::default(),
        Vec::new(),
    )
}

/// Open a SQL shell over `target` with optional resource limits. With
/// `scripts`, run them in order and exit; without, start the interactive
/// prompt.
pub fn run_with_limits(
    target: ShellTarget,
    limits: ShellLimits,
    scripts: Vec<ShellScript>,
) -> Result<ShellExit, Box<dyn std::error::Error>> {
    crate::server::raise_open_file_limit();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(repl::run_shell(target, limits, scripts))
}
