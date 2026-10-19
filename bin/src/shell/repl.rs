use std::future::Future;
use std::io::{self, IsTerminal, Write};
use std::time::Instant;

use crate::execution::{ExecuteOptions, STATS_VARIABLE, StatementOutput, is_truthy};
use clap::{Parser, ValueEnum};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::shell::copy::{CopyCommand, copy_from_file, parse_copy_command};
use crate::shell::parser::split_complete;
use crate::shell::progress::show_opening_progress;
use crate::shell::render::{TextBatch, render_table};
use crate::shell::{ShellInstance, ShellLimits, ShellTarget};

const HELP: &str = "\
Shell commands:
  \\q                 quit
  quit;              quit
  \\timing            toggle statement timing
  \\timing on|off     set statement timing
  \\copy <table> [(columns)] FROM '<file>' WITH (FORMAT arrow)
                     load a local Arrow IPC file into a table
  \\h, \\help         show this help

Session settings:
  SET pivot_stats = true;    print each statement's execution stats
  RESET pivot_stats;       stop printing them
";

pub(crate) async fn run_shell(
    target: ShellTarget,
    limits: ShellLimits,
) -> Result<(), Box<dyn std::error::Error>> {
    if !io::stdin().is_terminal() {
        return Err("an interactive terminal is required".into());
    }

    let instance = show_opening_progress(|| ShellInstance::open_with_limits(&target, limits))?;
    let editor = DefaultEditor::new()?;

    println!("pivot shell ({})", env!("CARGO_PKG_VERSION"));
    println!("Type \"\\help\" for help.");

    run_repl(instance.executor(), editor).await
}

async fn run_repl(
    executor: &crate::execution::Executor,
    mut editor: DefaultEditor,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buffer = String::new();
    let mut timing = false;
    let mut stats = false;
    loop {
        let prompt = if buffer.is_empty() {
            "pivot=> "
        } else {
            "pivot-> "
        };
        let (returned_editor, readline) = read_line(editor, prompt).await?;
        editor = returned_editor;
        let line = match readline {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                buffer.clear();
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(error) => return Err(error.into()),
        };

        if let Some(parsed) = parse_meta_command(&line) {
            let _ = editor.add_history_entry(line.clone());
            if handle_meta_command(parsed, executor, &mut timing, stats).await {
                break;
            }
            continue;
        }

        if !buffer.is_empty() {
            buffer.push('\n');
        }
        buffer.push_str(&line);
        let ParsedBuffer {
            statements,
            remainder,
            trailing_meta_command,
        } = parse_buffer(&buffer);
        buffer = remainder;

        let mut failed = false;
        for statement in statements {
            let _ = editor.add_history_entry(statement.clone());
            if is_quit_statement(&statement) {
                return Ok(());
            }
            if !execute_statement(executor, statement, timing, &mut stats).await {
                failed = true;
                break;
            }
        }
        if failed {
            buffer.clear();
        }
        if let Some((input, parsed)) = trailing_meta_command {
            let _ = editor.add_history_entry(input);
            if handle_meta_command(parsed, executor, &mut timing, stats).await {
                break;
            }
        }
    }
    Ok(())
}

async fn read_line(
    mut editor: DefaultEditor,
    prompt: &'static str,
) -> Result<(DefaultEditor, rustyline::Result<String>), tokio::task::JoinError> {
    tokio::task::spawn_blocking(move || {
        let result = editor.readline(prompt);
        (editor, result)
    })
    .await
}

/// The shell's backslash commands. A line's first word names the command
/// and the rest are its arguments, so adding one is a variant here and an
/// arm in `handle_meta_command`.
#[derive(Debug, Parser, PartialEq, Eq)]
#[command(
    multicall = true,
    disable_help_flag = true,
    disable_help_subcommand = true
)]
enum MetaCommand {
    /// Quit the shell.
    #[command(name = "\\q")]
    Quit,
    /// Show the shell's help.
    #[command(name = "\\help", alias = "\\h")]
    Help,
    /// Toggle statement timing, or set it.
    #[command(name = "\\timing")]
    Timing {
        #[arg(ignore_case = true)]
        state: Option<Switch>,
    },
    /// Load a local Arrow IPC file into a table.
    #[command(name = "\\copy")]
    Copy {
        /// `<table> [(columns)] FROM '<file>' [WITH (option, ...)]`, split
        /// on whitespace like every command line and rejoined for parsing.
        #[arg(
            value_name = "COPY SYNTAX",
            required = true,
            allow_hyphen_values = true
        )]
        arguments: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Switch {
    #[value(aliases = ["true", "1"])]
    On,
    #[value(aliases = ["false", "0"])]
    Off,
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedBuffer {
    statements: Vec<String>,
    remainder: String,
    trailing_meta_command: Option<(String, Result<MetaCommand, String>)>,
}

fn parse_buffer(buffer: &str) -> ParsedBuffer {
    let split = split_complete(buffer);
    let mut remainder = split.remainder;
    let trailing_meta_command = parse_meta_command(&remainder).map(|command| {
        let input = std::mem::take(&mut remainder);
        (input, command)
    });
    ParsedBuffer {
        statements: split.complete,
        remainder,
        trailing_meta_command,
    }
}

/// `None` when the line is not a meta-command; `Err` with the message to
/// print when it is one that does not parse.
fn parse_meta_command(line: &str) -> Option<Result<MetaCommand, String>> {
    let trimmed = line.trim();
    if !trimmed.starts_with('\\') {
        return None;
    }
    Some(MetaCommand::try_parse_from(trimmed.split_whitespace()).map_err(|error| error.to_string()))
}

/// Execute a parsed meta-command (or print why it did not parse) and return
/// whether the REPL should exit.
async fn handle_meta_command(
    parsed: Result<MetaCommand, String>,
    executor: &crate::execution::Executor,
    timing: &mut bool,
    stats: bool,
) -> bool {
    let command = match parsed {
        Ok(command) => command,
        Err(message) => {
            eprintln!("{message}");
            return false;
        }
    };
    match command {
        MetaCommand::Quit => true,
        MetaCommand::Help => {
            print!("{HELP}");
            false
        }
        MetaCommand::Timing { state } => {
            *timing = match state {
                Some(Switch::On) => true,
                Some(Switch::Off) => false,
                None => !*timing,
            };
            println!("Timing is {}.", if *timing { "on" } else { "off" });
            false
        }
        MetaCommand::Copy { arguments } => {
            match parse_copy_command(&arguments.join(" ")) {
                Ok(copy) => {
                    execute_copy(executor, copy, *timing, stats).await;
                }
                Err(message) => eprintln!("{message}"),
            }
            false
        }
    }
}

/// Interpret a parsed `SET`/`RESET` against the shell's stats toggle: the new
/// state when the statement names `pivot_stats`, `None` for any other
/// variable. `RESET` arrives with no value and reads as off.
fn parse_stats_toggle(name: &str, value: Option<&str>) -> Option<bool> {
    name.eq_ignore_ascii_case(STATS_VARIABLE)
        .then(|| value.is_some_and(is_truthy))
}

fn is_quit_statement(statement: &str) -> bool {
    statement
        .strip_suffix(';')
        .is_some_and(|command| command.trim().eq_ignore_ascii_case("quit"))
}

async fn execute_statement(
    executor: &crate::execution::Executor,
    sql: String,
    timing: bool,
    stats: &mut bool,
) -> bool {
    // Like the server, read the flag before the statement runs: the SET that
    // turns stats on reports no stats itself, the one that turns them off
    // still reports its own.
    let collect_stats = *stats;
    let options = ExecuteOptions {
        collect_stats,
        profile: false,
    };
    let work = async {
        match executor
            .execute_with_output::<TextBatch>(sql, options)
            .await
        {
            Ok(execution) => {
                if let StatementOutput::Set { name, value } = &execution.output
                    && let Some(enabled) = parse_stats_toggle(name, value.as_deref())
                {
                    *stats = enabled;
                }
                if collect_stats {
                    println!("{}", execution.stats.summary());
                }
                if let Err(error) = display_output(execution.output) {
                    eprintln!("pivot: writing output: {error}");
                    false
                } else {
                    true
                }
            }
            Err(error) => {
                eprintln!("ERROR:  {error}");
                false
            }
        }
    };
    run_interruptible(work, timing).await
}

/// Run a `\copy`: load its file and report the row count like a COPY does.
async fn execute_copy(
    executor: &crate::execution::Executor,
    copy: CopyCommand,
    timing: bool,
    collect_stats: bool,
) -> bool {
    let options = ExecuteOptions {
        collect_stats,
        profile: false,
    };
    let work = async {
        match copy_from_file(executor, copy, options).await {
            Ok(outcome) => {
                if collect_stats {
                    println!("{}", outcome.stats.summary());
                }
                println!("COPY {}", outcome.rows);
                true
            }
            Err(error) => {
                eprintln!("ERROR:  {error}");
                false
            }
        }
    };
    run_interruptible(work, timing).await
}

/// Drive `work` to its verdict unless Ctrl-C arrives first, in which case the
/// future is dropped, cancelling whatever it was running. Prints the elapsed
/// time when `timing` is on.
async fn run_interruptible(work: impl Future<Output = bool>, timing: bool) -> bool {
    let started = Instant::now();
    let succeeded = tokio::select! {
        succeeded = work => succeeded,
        signal = tokio::signal::ctrl_c() => {
            match signal {
                Ok(()) => {
                    eprintln!("Cancel request sent");
                    eprintln!("ERROR:  query canceled");
                }
                Err(error) => eprintln!("pivot: failed to listen for Ctrl-C: {error}"),
            }
            false
        }
    };

    if timing {
        println!("Time: {:.3} ms", started.elapsed().as_secs_f64() * 1e3);
    }
    succeeded
}

fn display_output(output: StatementOutput<TextBatch>) -> io::Result<()> {
    let rendered = match output {
        StatementOutput::Rows { columns, batches } => render_table(&columns, batches),
        StatementOutput::Command(command) => format!("{command}\n"),
        StatementOutput::Set { value, .. } => {
            if value.is_some() { "SET\n" } else { "RESET\n" }.to_string()
        }
        StatementOutput::CopyFromStdin(ingest) => {
            // A bare COPY has nothing to feed the ingest with; dropping it
            // aborts it and rolls the statement's transaction back.
            drop(ingest);
            "COPY FROM STDIN is not supported in the shell; \\copy loads a file\n".to_string()
        }
    };
    let mut stdout = io::stdout().lock();
    stdout.write_all(rendered.as_bytes())?;
    stdout.flush()
}

#[cfg(test)]
mod tests {
    use super::{
        MetaCommand, ParsedBuffer, Switch, execute_statement, is_quit_statement, parse_buffer,
        parse_meta_command, parse_stats_toggle,
    };

    #[test]
    fn parses_supported_meta_commands() {
        let quit = parse_meta_command("  \\q ");
        let short_help = parse_meta_command("\\h");
        let help = parse_meta_command("\\help");
        let toggle = parse_meta_command("\\timing");
        let off = parse_meta_command("\\timing OFF");

        assert_eq!(quit, Some(Ok(MetaCommand::Quit)));
        assert_eq!(short_help, Some(Ok(MetaCommand::Help)));
        assert_eq!(help, Some(Ok(MetaCommand::Help)));
        assert_eq!(toggle, Some(Ok(MetaCommand::Timing { state: None })));
        assert_eq!(
            off,
            Some(Ok(MetaCommand::Timing {
                state: Some(Switch::Off)
            }))
        );
    }

    #[test]
    fn rejects_meta_command_arguments_and_unknown_commands() {
        let quit_with_argument = parse_meta_command("\\q now");
        let unknown = parse_meta_command("\\dt");

        let quit_error = quit_with_argument.unwrap().unwrap_err();
        let unknown_error = unknown.unwrap().unwrap_err();
        assert!(
            quit_error.contains("unexpected argument 'now'"),
            "{quit_error}"
        );
        assert!(
            unknown_error.contains("unrecognized subcommand '\\dt'"),
            "{unknown_error}"
        );
    }

    #[test]
    fn copy_takes_the_rest_of_the_line_as_its_arguments() {
        let copy =
            parse_meta_command("\\copy people (id) from 'my rows.arrow' with (format arrow)");
        let bare = parse_meta_command("\\copy");

        assert_eq!(
            copy,
            Some(Ok(MetaCommand::Copy {
                arguments: [
                    "people",
                    "(id)",
                    "from",
                    "'my",
                    "rows.arrow'",
                    "with",
                    "(format",
                    "arrow)"
                ]
                .map(String::from)
                .to_vec()
            }))
        );
        assert!(bare.unwrap().is_err());
    }

    #[test]
    fn parses_a_meta_command_after_complete_sql_on_the_same_line() {
        assert_eq!(
            parse_buffer("SELECT 1; \\timing"),
            ParsedBuffer {
                statements: vec!["SELECT 1;".to_string()],
                remainder: String::new(),
                trailing_meta_command: Some((
                    "\\timing".to_string(),
                    Ok(MetaCommand::Timing { state: None })
                )),
            }
        );
    }

    #[test]
    fn set_statements_turn_stats_on_and_off_across_a_session() {
        let directory = tempfile::tempdir().unwrap();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let instance = runtime
            .block_on(async {
                crate::shell::ShellInstance::open_with_resources(
                    &crate::shell::ShellTarget::pivot(directory.path().to_str().unwrap()),
                    1,
                    32,
                    datastore_pivotlake::DEFAULT_REFRESH_INTERVAL,
                )
            })
            .unwrap();
        let mut stats = false;

        let on_after_set = runtime.block_on(async {
            let executor = instance.executor();
            assert!(
                execute_statement(executor, "SET pivot_stats = 1".into(), false, &mut stats).await
            );
            let on = stats;
            assert!(
                execute_statement(executor, "SET pivot_stats = 0".into(), false, &mut stats).await
            );
            on
        });

        assert!(on_after_set);
        assert!(!stats);
        drop(instance);
    }

    #[test]
    fn toggles_stats_only_for_the_pivot_stats_variable() {
        assert_eq!(parse_stats_toggle("pivot_stats", Some("1")), Some(true));
        assert_eq!(parse_stats_toggle("PIVOT_STATS", Some("on")), Some(true));
        assert_eq!(parse_stats_toggle("pivot_stats", Some("off")), Some(false));
        assert_eq!(parse_stats_toggle("pivot_stats", None), Some(false));
        assert_eq!(parse_stats_toggle("search_path", Some("1")), None);
    }

    #[test]
    fn recognizes_quit_as_a_complete_statement() {
        assert!(is_quit_statement("quit;"));
        assert!(is_quit_statement("QUIT ;"));
        assert!(!is_quit_statement("quit"));
        assert!(!is_quit_statement("SELECT 'quit;';"));
    }
}
