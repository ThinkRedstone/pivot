//! Pivot command-line entry point.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;

use bin::memory::MemoryBudget;
use bin::shell::{ShellInput, ShellTarget};
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(name = "pivot", about = "Pivot command-line tools", version)]
struct Args {
    #[command(subcommand)]
    command: PivotCommand,
}

/// Which datastore implementation `pivot open` opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DatastoreKind {
    /// A pivotlake datastore in a local directory or at an object-store URI.
    /// Also accepted as `pivot`, the kind's name before it was renamed.
    #[value(alias = "pivot")]
    Pivotlake,
    /// The read-only tables of an Iceberg REST catalog at an http(s) URI.
    Iceberg,
}

#[derive(Subcommand, Debug)]
enum PivotCommand {
    /// Open a datastore in the Pivot SQL shell.
    ///
    /// A pivotlake datastore (the default kind) is named either by a local
    /// directory, which is created if it does not exist, or by an object-store
    /// URI: s3://bucket/prefix (s3:// also spelled s3a://), gs://bucket/prefix,
    /// or file:///path.
    ///
    /// An Iceberg datastore (--kind iceberg) is named by its REST catalog's
    /// http(s) base URI and serves the catalog's tables read-only; each
    /// namespace is a schema. The catalog is
    /// authenticated to with PIVOT_ICEBERG_TOKEN (a bearer token) or
    /// PIVOT_ICEBERG_CREDENTIAL (an OAuth2 client_id:client_secret, refined by
    /// the optional PIVOT_ICEBERG_OAUTH2_SERVER_URI and
    /// PIVOT_ICEBERG_OAUTH2_SCOPE), or with nothing when neither is set. The
    /// tables' files are read from wherever the catalog says they are, with the
    /// credentials the catalog vends or those of the environment below.
    ///
    /// Object-store credentials are read from the environment. S3 takes
    /// AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY when both are set, or uses
    /// anonymous access when both are absent. Its region comes from AWS_REGION
    /// or AWS_DEFAULT_REGION when set, and is discovered with HeadBucket
    /// otherwise. An optional AWS_ENDPOINT_URL names a path-style S3-compatible
    /// endpoint such as MinIO. GCS follows Application Default Credentials:
    /// GOOGLE_APPLICATION_CREDENTIALS, then the gcloud login file, then the
    /// instance metadata server.
    Open {
        /// Local directory or object-store URI holding the datastore, or the
        /// Iceberg REST catalog's URI with --kind iceberg.
        #[arg(value_name = "DATASTORE_LOCATION")]
        datastore_location: String,

        /// Which datastore implementation the location names.
        #[arg(long, value_enum, default_value_t = DatastoreKind::Pivotlake)]
        kind: DatastoreKind,

        /// The warehouse to serve, for an Iceberg catalog that serves several.
        #[arg(long, value_name = "WAREHOUSE")]
        warehouse: Option<String>,

        /// Buffer-pool memory budget: a size (suffixes k/m/g/t, base-1024), or
        /// a share of the machine's physical memory such as 80%, the default,
        /// minus a 4 GiB reserve for allocations outside the pool.
        #[arg(long, value_name = "SIZE")]
        memory: Option<MemoryBudget>,

        /// Number of dispatch worker threads. Defaults to all available cores.
        #[arg(long, value_name = "COUNT")]
        workers: Option<NonZeroUsize>,

        /// Run COMMAND (SQL or a backslash command) and exit instead of
        /// starting the prompt. Repeatable; runs in order with every -f.
        #[arg(short = 'c', long = "command", value_name = "COMMAND")]
        commands: Vec<String>,

        /// Run the SQL and backslash commands in FILE ("-" for stdin) and exit
        /// instead of starting the prompt. Repeatable; runs in order with
        /// every -c. The first failure stops every script.
        #[arg(short = 'f', long = "file", value_name = "FILE")]
        files: Vec<PathBuf>,
    },
    /// Run the Pivot database server in the foreground.
    Server(bin::server::ServerOptions),
}

/// Why `pivot` failed. Shell errors pick their own exit code.
#[derive(Debug, thiserror::Error)]
enum Error {
    #[error(transparent)]
    ShellError(#[from] bin::shell::Error),
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error>),
}

impl Error {
    fn exit_code(&self) -> ExitCode {
        match self {
            Self::ShellError(error) => error.exit_code(),
            Self::Other(_) => ExitCode::FAILURE,
        }
    }
}

fn run() -> Result<(), Error> {
    let matches = Args::command().get_matches();
    let args = Args::from_arg_matches(&matches).unwrap_or_else(|error| error.exit());
    match args.command {
        PivotCommand::Open {
            datastore_location,
            kind,
            warehouse,
            memory,
            workers,
            commands,
            files,
        } => {
            let (_, open_matches) = matches
                .subcommand()
                .expect("clap parsed the open subcommand");
            bin::shell::run_with_limits(
                build_shell_target(kind, datastore_location, warehouse)?,
                bin::shell::ShellLimits {
                    memory,
                    workers: workers.map(NonZeroUsize::get),
                },
                order_scripts(open_matches, commands, files),
            )?;
            Ok(())
        }
        PivotCommand::Server(options) => {
            bin::server::run(options).map_err(Box::from)?;
            Ok(())
        }
    }
}

/// Interleave `-c` and `-f` in command-line order. clap keeps each flag's
/// values in order but stores the two flags separately, so their relative
/// order comes from the positions it recorded for each value.
fn order_scripts(
    open_matches: &ArgMatches,
    commands: Vec<String>,
    files: Vec<PathBuf>,
) -> Vec<ShellInput> {
    let indices_of = |id: &str| {
        open_matches
            .indices_of(id)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    };
    let mut scripts: Vec<(usize, ShellInput)> = indices_of("commands")
        .into_iter()
        .zip(commands.into_iter().map(ShellInput::Command))
        .chain(
            indices_of("files")
                .into_iter()
                .zip(files.into_iter().map(ShellInput::File)),
        )
        .collect();
    scripts.sort_by_key(|(index, _)| *index);
    scripts.into_iter().map(|(_, script)| script).collect()
}

/// What `pivot open` opens, from its arguments. A pivotlake datastore has no
/// warehouse to name.
fn build_shell_target(
    kind: DatastoreKind,
    datastore_location: String,
    warehouse: Option<String>,
) -> Result<ShellTarget, Box<dyn std::error::Error>> {
    match kind {
        DatastoreKind::Pivotlake => {
            if warehouse.is_some() {
                return Err("--warehouse applies only to --kind iceberg".into());
            }
            Ok(ShellTarget::pivot(datastore_location))
        }
        DatastoreKind::Iceberg => Ok(ShellTarget::iceberg(datastore_location, warehouse)?),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("pivot: {error}");
            error.exit_code()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Args, DatastoreKind, PivotCommand, order_scripts};
    use bin::shell::ShellInput;
    use clap::{CommandFactory, FromArgMatches, Parser};

    fn parse_open(arguments: &[&str]) -> (String, DatastoreKind, Option<String>) {
        let args = Args::try_parse_from([&["pivot", "open"], arguments].concat()).unwrap();
        let PivotCommand::Open {
            datastore_location,
            kind,
            warehouse,
            ..
        } = args.command
        else {
            panic!("open did not parse as the open command");
        };
        (datastore_location, kind, warehouse)
    }

    fn open_location(argument: &str) -> String {
        parse_open(&[argument]).0
    }

    #[test]
    fn requires_a_command_and_an_open_datastore_location() {
        assert!(Args::try_parse_from(["pivot"]).is_err());
        assert!(Args::try_parse_from(["pivot", "open"]).is_err());
        assert!(Args::try_parse_from(["pivot", "shell", "/var/lib/pivot"]).is_err());

        assert_eq!(open_location("/var/lib/pivot"), "/var/lib/pivot");
    }

    #[test]
    fn open_takes_an_object_store_uri_verbatim() {
        assert_eq!(open_location("s3://bucket/prefix"), "s3://bucket/prefix");
        assert_eq!(open_location("gs://bucket/prefix"), "gs://bucket/prefix");
    }

    #[test]
    fn open_defaults_to_the_pivotlake_kind() {
        let (_, kind, warehouse) = parse_open(&["/var/lib/pivot"]);

        assert_eq!(kind, DatastoreKind::Pivotlake);
        assert!(warehouse.is_none());
    }

    #[test]
    fn open_takes_pivot_as_the_pivotlake_kind() {
        let (_, kind, _) = parse_open(&["--kind", "pivot", "/var/lib/pivot"]);

        assert_eq!(kind, DatastoreKind::Pivotlake);
    }

    #[test]
    fn open_takes_an_iceberg_catalog_and_its_warehouse() {
        let (location, kind, warehouse) = parse_open(&[
            "--kind",
            "iceberg",
            "https://catalog.example.com/api",
            "--warehouse",
            "s3://lake/warehouse",
        ]);

        assert_eq!(location, "https://catalog.example.com/api");
        assert_eq!(kind, DatastoreKind::Iceberg);
        assert_eq!(warehouse.as_deref(), Some("s3://lake/warehouse"));
    }

    #[test]
    fn open_rejects_an_unknown_kind() {
        assert!(
            Args::try_parse_from(["pivot", "open", "--kind", "delta", "/var/lib/pivot"]).is_err()
        );
    }

    #[test]
    fn open_orders_commands_and_files_as_given() {
        let matches = Args::command()
            .try_get_matches_from(["pivot", "open", "/d", "-c", "x", "-f", "y", "-c", "z"])
            .unwrap();
        let args = Args::from_arg_matches(&matches).unwrap();
        let PivotCommand::Open {
            commands, files, ..
        } = args.command
        else {
            panic!("open did not parse as the open command");
        };

        let scripts = order_scripts(matches.subcommand().unwrap().1, commands, files);

        assert_eq!(
            scripts,
            [
                ShellInput::Command("x".into()),
                ShellInput::File("y".into()),
                ShellInput::Command("z".into()),
            ]
        );
    }

    #[test]
    fn server_requires_a_config_file() {
        assert!(Args::try_parse_from(["pivot", "server"]).is_err());
        assert!(Args::try_parse_from(["pivot", "server", "install"]).is_err());

        let args =
            Args::try_parse_from(["pivot", "server", "--config", "/etc/pivot/pivot.yaml"]).unwrap();
        assert!(matches!(args.command, PivotCommand::Server(_)));
    }
}
