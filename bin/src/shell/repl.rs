use crate::execution::{ExecuteOptions, STATS_VARIABLE, StatementOutput, is_truthy};
use crate::shell::copy::{CopyCommand, copy_from_file, parse_copy_command};
use crate::shell::parser::split_complete;
use crate::shell::progress::show_opening_progress;
use crate::shell::render::{TextBatch, render_table};
use crate::shell::{ShellInput, ShellInstance, ShellLimits, ShellTarget};
use clap::{Parser, ValueEnum};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::fs::File;
use std::future::Future;
use std::io::{self, BufRead, BufReader, IsTerminal, Write};
use std::process::ExitCode;
use std::time::Instant;

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

/// Why the shell, or one line of its input, stopped.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `\q` or `quit;`; `run_shell` turns it into success.
    #[error("quit")]
    Quit,
    /// A statement or `\copy` that failed or was canceled.
    #[error("ERROR:  {0}")]
    Statement(Box<dyn std::error::Error>),
    /// A backslash command that did not parse; its message is complete as-is.
    #[error("{0}")]
    InvalidCommand(String),
    #[error("{0}")]
    Open(Box<dyn std::error::Error>),
    #[error("{input}: {source}")]
    Input { input: String, source: io::Error },
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Readline(#[from] ReadlineError),
}

impl Error {
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::Quit => ExitCode::SUCCESS,
            Self::Open(_) => ExitCode::from(2),
            Self::Statement(_) | Self::InvalidCommand(_) => ExitCode::from(3),
            Self::Input { .. } | Self::Io(_) | Self::Readline(_) => ExitCode::from(1),
        }
    }
}

pub(crate) async fn run_shell(
    target: ShellTarget,
    limits: ShellLimits,
    inputs: Vec<ShellInput>,
) -> Result<(), Error> {
    let interactive = inputs.is_empty();
    if interactive && !io::stdin().is_terminal() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "an interactive terminal is required",
        )
        .into());
    }

    let session = Session::open(&target, limits)?;
    if !interactive {
        return match run_inputs(session, inputs).await {
            Err(Error::Quit) => Ok(()),
            result => result,
        };
    }

    let editor = DefaultEditor::new()?;
    println!("pivot shell ({})", env!("CARGO_PKG_VERSION"));
    println!("Type \"\\help\" for help.");
    run_repl(session, editor).await
}

/// Run each input in order, stopping at the first that fails or quits.
async fn run_inputs(mut session: Session, inputs: Vec<ShellInput>) -> Result<(), Error> {
    for input in inputs {
        session.run_input(&input).await?;
    }
    Ok(())
}

/// A reader over the input's text, so a file of any size is read a line at
/// a time instead of all at once.
fn open_input(input: &ShellInput) -> io::Result<impl BufRead + '_> {
    let reader: Box<dyn BufRead + '_> = match input {
        ShellInput::Command(command) => Box::new(command.as_bytes()),
        ShellInput::File(path) if path.as_os_str() == "-" => Box::new(io::stdin().lock()),
        ShellInput::File(path) => Box::new(BufReader::new(File::open(path)?)),
    };
    Ok(reader)
}

/// How errors name the input they came from.
fn describe_input(input: &ShellInput) -> String {
    match input {
        // We shouldn't get IO errors reading from a CLI argument, but this is easier than
        // untangling the typing
        ShellInput::Command(command) => format!("-c '{command}'"),
        ShellInput::File(path) if path.as_os_str() == "-" => "stdin".to_string(),
        ShellInput::File(path) => path.display().to_string(),
    }
}

async fn run_repl(mut session: Session, mut editor: DefaultEditor) -> Result<(), Error> {
    loop {
        let prompt = if session.buffer.is_empty() {
            "pivot=> "
        } else {
            "pivot-> "
        };
        let (returned_editor, readline) = read_line(editor, prompt).await;
        editor = returned_editor;
        let line = match readline {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                session.buffer.clear();
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(error) => return Err(error.into()),
        };

        let mut record_history = |entry: String| {
            let _ = editor.add_history_entry(entry);
        };
        match session.feed_line(&line, &mut record_history).await {
            Ok(()) => {}
            Err(Error::Quit) => break,
            // An interactive statement failing is not fatal: report it and
            // prompt again.
            Err(error @ (Error::Statement(_) | Error::InvalidCommand(_))) => {
                eprintln!("{error}");
                session.buffer.clear();
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

async fn read_line(
    mut editor: DefaultEditor,
    prompt: &'static str,
) -> (DefaultEditor, rustyline::Result<String>) {
    tokio::task::spawn_blocking(move || {
        let result = editor.readline(prompt);
        (editor, result)
    })
    .await
    // If `readline` somehow panics in the other thread, we want to propagate that
    .unwrap_or_else(|error| std::panic::resume_unwind(error.into_panic()))
}

/// One shell session over an open datastore: the statement buffer and the
/// settings its meta-commands and `SET`s change.
struct Session {
    instance: ShellInstance,
    timing: bool,
    stats: bool,
    buffer: String,
}

impl Session {
    /// Open the datastore (behind the opening spinner) into a fresh session.
    fn open(target: &ShellTarget, limits: ShellLimits) -> Result<Self, Error> {
        let instance = show_opening_progress(|| ShellInstance::open_with_limits(target, limits))
            .map_err(Error::Open)?;
        Ok(Self {
            instance,
            timing: false,
            stats: false,
            buffer: String::new(),
        })
    }

    fn executor(&self) -> &crate::execution::Executor {
        self.instance.executor()
    }

    /// Handle one line of input: a meta-command, or SQL appended to the buffer
    /// whose complete statements then run. `record_history` receives each
    /// command as it is recognized.
    async fn feed_line(
        &mut self,
        line: &str,
        record_history: &mut impl FnMut(String),
    ) -> Result<(), Error> {
        if let Some(parsed) = parse_meta_command(line) {
            record_history(line.to_string());
            return self.handle_meta_command(parsed).await;
        }

        if !self.buffer.is_empty() {
            self.buffer.push('\n');
        }
        self.buffer.push_str(line);
        let ParsedBuffer {
            statements,
            remainder,
            trailing_meta_command,
        } = parse_buffer(&self.buffer);
        self.buffer = remainder;

        let mut result = Ok(());
        for statement in statements {
            record_history(statement.clone());
            if is_quit_statement(&statement) {
                return Err(Error::Quit);
            }
            result = self.execute_statement(statement).await;
            if result.is_err() {
                break;
            }
        }
        if let Some((input, parsed)) = trailing_meta_command {
            record_history(input);
            let meta_result = self.handle_meta_command(parsed).await;
            if meta_result.is_err() {
                result = meta_result;
            }
        }
        result
    }

    /// Declare we have no more lines to feed: this runs the statement left
    /// unterminated in the buffer, if any.
    async fn finish(&mut self) -> Result<(), Error> {
        let statement = std::mem::take(&mut self.buffer);
        if statement.is_empty() {
            Ok(())
        } else if statement.trim().eq_ignore_ascii_case("quit") {
            Err(Error::Quit)
        } else {
            self.execute_statement(statement).await
        }
    }

    /// Feed `input` line by line as it is read, then finish it.
    async fn run_input(&mut self, input: &ShellInput) -> Result<(), Error> {
        let read_failed = |source| Error::Input {
            input: describe_input(input),
            source,
        };
        let mut reader = open_input(input).map_err(read_failed)?;
        let mut line = String::new();
        while reader.read_line(&mut line).map_err(read_failed)? != 0 {
            let content = line.strip_suffix('\n').unwrap_or(&line);
            let content = content.strip_suffix('\r').unwrap_or(content);
            self.feed_line(content, &mut |_| {}).await?;
            line.clear();
        }
        self.finish().await
    }

    /// Execute a parsed meta-command.
    async fn handle_meta_command(
        &mut self,
        parsed: Result<MetaCommand, String>,
    ) -> Result<(), Error> {
        match parsed.map_err(Error::InvalidCommand)? {
            MetaCommand::Quit => Err(Error::Quit),
            MetaCommand::Help => {
                print!("{HELP}");
                Ok(())
            }
            MetaCommand::Timing { state } => {
                self.timing = match state {
                    Some(Switch::On) => true,
                    Some(Switch::Off) => false,
                    None => !self.timing,
                };
                println!("Timing is {}.", if self.timing { "on" } else { "off" });
                Ok(())
            }
            MetaCommand::Copy { arguments } => {
                let copy =
                    parse_copy_command(&arguments.join(" ")).map_err(Error::InvalidCommand)?;
                self.execute_copy(copy).await
            }
        }
    }

    async fn execute_statement(&mut self, sql: String) -> Result<(), Error> {
        // Like the server, read the flag before the statement runs: the SET that
        // turns stats on reports no stats itself, the one that turns them off
        // still reports its own.
        let collect_stats = self.stats;
        let options = ExecuteOptions {
            collect_stats,
            profile: false,
        };
        let executor = self.instance.executor();
        let stats = &mut self.stats;
        let work = async {
            let execution = executor
                .execute_with_output::<TextBatch>(sql, options)
                .await
                .map_err(|error| Error::Statement(error.into()))?;
            if let StatementOutput::Set { name, value } = &execution.output
                && let Some(enabled) = parse_stats_toggle(name, value.as_deref())
            {
                *stats = enabled;
            }
            if collect_stats {
                println!("{}", execution.stats.summary());
            }
            display_output(execution.output)?;
            Ok(())
        };
        run_interruptible(work, self.timing).await
    }

    /// Run a `\copy`: load its file and report the row count like a COPY does.
    async fn execute_copy(&self, copy: CopyCommand) -> Result<(), Error> {
        let collect_stats = self.stats;
        let options = ExecuteOptions {
            collect_stats,
            profile: false,
        };
        let work = async {
            let outcome = copy_from_file(self.executor(), copy, options)
                .await
                .map_err(|error| Error::Statement(error.into()))?;
            if collect_stats {
                println!("{}", outcome.stats.summary());
            }
            println!("COPY {}", outcome.rows);
            Ok(())
        };
        run_interruptible(work, self.timing).await
    }
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

/// Drive `work` to its result unless Ctrl-C arrives first, in which case the
/// future is dropped, cancelling whatever it was running. Prints the elapsed
/// time when `timing` is on.
async fn run_interruptible(
    work: impl Future<Output = Result<(), Error>>,
    timing: bool,
) -> Result<(), Error> {
    let started = Instant::now();
    let result = tokio::select! {
        result = work => result,
        signal = tokio::signal::ctrl_c() => {
            signal?;
            eprintln!("Cancel request sent");
            Err(Error::Statement("query canceled".into()))
        }
    };

    if timing {
        println!("Time: {:.3} ms", started.elapsed().as_secs_f64() * 1e3);
    }
    result
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
        MetaCommand, ParsedBuffer, Session, Switch, is_quit_statement, parse_buffer,
        parse_meta_command, parse_stats_toggle,
    };
    use crate::shell::{ShellLimits, ShellTarget};

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
        let mut session = runtime
            .block_on(async {
                Session::open(
                    &ShellTarget::pivot(directory.path().to_str().unwrap()),
                    ShellLimits {
                        memory: Some("64m".parse().unwrap()),
                        workers: Some(1),
                    },
                )
            })
            .unwrap();

        let on_after_set = runtime.block_on(async {
            session
                .execute_statement("SET pivot_stats = 1".into())
                .await
                .unwrap();
            let on = session.stats;
            session
                .execute_statement("SET pivot_stats = 0".into())
                .await
                .unwrap();
            on
        });

        assert!(on_after_set);
        assert!(!session.stats);
        drop(session);
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
