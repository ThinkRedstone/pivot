//! The shell's `\copy`: a `COPY ... FROM STDIN` whose rows come from a local
//! file, the way psql's `\copy` feeds the server from the client's file
//! system.

use std::io;
use std::path::PathBuf;

use bytes::BytesMut;
use tokio::io::AsyncReadExt;

use crate::execution::{ExecuteOptions, ExecutionStats, Executor, StatementOutput};
use crate::shell::render::TextBatch;

/// Bytes handed to the ingest per read of the source file.
const CHUNK_BYTES: usize = 1 << 20;

/// A parsed `\copy`, split the way psql splits it: the engine runs a plain
/// `COPY ... FROM STDIN`, and the shell plays the part of that STDIN by
/// streaming a local file into it. The engine never sees the file.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CopyCommand {
    /// `COPY <target> FROM STDIN <options>`: the user's line with the file
    /// reference replaced by `STDIN` and everything else as written.
    pub statement: String,
    /// The local file whose bytes stand in for that STDIN.
    pub file: PathBuf,
}

const USAGE: &str = "\\copy <table> [(columns)] FROM '<file>' [WITH (option, ...)]";

/// Parse the text after `\copy`: everything before the first FROM (or TO)
/// word is the target, the word after it names the file, and the rest goes to
/// the statement untouched, so options are spelled exactly as in a `COPY`. A
/// trailing semicolon is tolerated.
pub(crate) fn parse_copy_command(arguments: &str) -> Result<CopyCommand, String> {
    let arguments = arguments.trim_end();
    let arguments = arguments.strip_suffix(';').unwrap_or(arguments);
    let (target, direction, after_direction) =
        split_at_direction(arguments).ok_or_else(parse_error)?;
    if target.is_empty() {
        return Err(parse_error());
    }
    if direction.eq_ignore_ascii_case("to") {
        return Err("\\copy: COPY TO is not supported; only FROM a file is".to_string());
    }
    let (file, options) = split_file_name(after_direction.trim_start())?;
    let options = options.trim();
    let statement = if options.is_empty() {
        format!("COPY {target} FROM STDIN")
    } else {
        format!("COPY {target} FROM STDIN {options}")
    };
    Ok(CopyCommand {
        statement,
        file: PathBuf::from(file),
    })
}

fn parse_error() -> String {
    format!("\\copy: parse error; usage: {USAGE}")
}

/// Split `text` at its first FROM or TO word, whatever its case: the trimmed
/// text before it, the word itself, and the text after it.
fn split_at_direction(text: &str) -> Option<(&str, &str, &str)> {
    let mut remaining = text;
    while let Some(start) = remaining.find(|character: char| !character.is_whitespace()) {
        let end = remaining[start..]
            .find(char::is_whitespace)
            .map_or(remaining.len(), |length| start + length);
        let word = &remaining[start..end];
        if word.eq_ignore_ascii_case("from") || word.eq_ignore_ascii_case("to") {
            let before = &text[..text.len() - remaining.len() + start];
            return Some((before.trim(), word, &remaining[end..]));
        }
        remaining = &remaining[end..];
    }
    None
}

/// The file name at the start of `text` and what follows it. The name is a
/// single-quoted string (a doubled quote inside is a literal one) or a bare
/// word; the other psql sources have no meaning here and are refused by name.
fn split_file_name(text: &str) -> Result<(String, &str), String> {
    let Some(mut remaining) = text.strip_prefix('\'') else {
        let (word, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        if word.is_empty() {
            return Err(parse_error());
        }
        if ["stdin", "pstdin", "program"]
            .iter()
            .any(|keyword| word.eq_ignore_ascii_case(keyword))
        {
            return Err(format!(
                "\\copy: FROM {} is not supported; name a file to load",
                word.to_uppercase()
            ));
        }
        return Ok((word.to_string(), rest));
    };
    let mut name = String::new();
    loop {
        let close = remaining.find('\'').ok_or_else(parse_error)?;
        name.push_str(&remaining[..close]);
        remaining = &remaining[close + 1..];
        match remaining.strip_prefix('\'') {
            Some(after_doubled_quote) => {
                name.push('\'');
                remaining = after_doubled_quote;
            }
            None => return Ok((name, remaining)),
        }
    }
}

/// Why a `\copy` did not load its file.
#[derive(Debug, thiserror::Error)]
pub(crate) enum CopyError {
    #[error(transparent)]
    Execute(#[from] crate::execution::Error),
    #[error("{path}: {source}")]
    Read {
        path: String,
        #[source]
        source: io::Error,
    },
}

/// A completed `\copy`: the rows it loaded and the statement's stats.
pub(crate) struct CopyOutcome {
    pub rows: usize,
    pub stats: ExecutionStats,
}

/// Plan the COPY, stream the file into its ingest, and commit. Dropping the
/// future midway drops the ingest, which aborts the load and rolls it back.
pub(crate) async fn copy_from_file(
    executor: &Executor,
    command: CopyCommand,
    options: ExecuteOptions,
) -> Result<CopyOutcome, CopyError> {
    let execution = executor
        .execute_with_output(command.statement, options, TextBatch::from_record_batch)
        .await?;
    let StatementOutput::CopyFromStdin(mut ingest) = execution.output else {
        unreachable!("a COPY ... FROM STDIN statement plans as a copy-in");
    };
    let path = command.file.display().to_string();
    let read_error = |source: io::Error| CopyError::Read {
        path: path.clone(),
        source,
    };
    let mut file = tokio::fs::File::open(&command.file)
        .await
        .map_err(read_error)?;
    let mut chunk = BytesMut::with_capacity(CHUNK_BYTES);
    loop {
        chunk.reserve(CHUNK_BYTES);
        let read = file.read_buf(&mut chunk).await.map_err(read_error)?;
        if read == 0 {
            break;
        }
        ingest.push(chunk.split().freeze()).await?;
    }
    let rows = ingest.finish().await?;
    Ok(CopyOutcome {
        rows,
        stats: execution.stats,
    })
}

#[cfg(test)]
mod tests {
    use super::{CopyCommand, copy_from_file, parse_copy_command};
    use crate::execution::{ExecuteOptions, StatementOutput};
    use crate::shell::{ShellInstance, ShellTarget};
    use arrow_array::{Array, Int64Array, RecordBatch};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn parsed(arguments: &str) -> CopyCommand {
        parse_copy_command(arguments).unwrap()
    }

    #[test]
    fn builds_a_copy_from_stdin_with_the_options_passed_through() {
        let command = parsed("people (id, name) from 'rows.arrow' with (format arrow);");

        assert_eq!(
            command,
            CopyCommand {
                statement: "COPY people (id, name) FROM STDIN with (format arrow)".to_string(),
                file: PathBuf::from("rows.arrow"),
            }
        );
    }

    #[test]
    fn accepts_qualified_and_quoted_names_and_bare_file_names() {
        let qualified = parsed("main.\"odd (name)\" FROM /tmp/rows.arrow");
        let quoted_file = parsed("t from 'it''s here/rows.arrow'");

        assert_eq!(
            qualified.statement,
            "COPY main.\"odd (name)\" FROM STDIN".to_string()
        );
        assert_eq!(qualified.file, PathBuf::from("/tmp/rows.arrow"));
        assert_eq!(quoted_file.file, PathBuf::from("it's here/rows.arrow"));
    }

    #[test]
    fn refuses_what_it_cannot_load() {
        let to = parse_copy_command("people to 'out.arrow'").unwrap_err();
        let stdin = parse_copy_command("people from stdin").unwrap_err();
        let no_file = parse_copy_command("people from").unwrap_err();
        let bad_direction = parse_copy_command("people into 'x'").unwrap_err();
        let nothing = parse_copy_command("").unwrap_err();
        let unterminated = parse_copy_command("people from 'x").unwrap_err();

        assert_eq!(to, "\\copy: COPY TO is not supported; only FROM a file is");
        assert_eq!(
            stdin,
            "\\copy: FROM STDIN is not supported; name a file to load"
        );
        let parse_error = "\\copy: parse error; usage: \\copy <table> [(columns)] FROM '<file>' [WITH (option, ...)]";
        assert_eq!(no_file, parse_error);
        assert_eq!(bad_direction, parse_error);
        assert_eq!(nothing, parse_error);
        assert_eq!(unterminated, parse_error);
    }

    fn write_arrow_stream(path: &std::path::Path, batch: &RecordBatch, truncate: bool) {
        let mut bytes = Vec::new();
        let mut writer =
            arrow::ipc::writer::StreamWriter::try_new(&mut bytes, &batch.schema()).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        if truncate {
            bytes.truncate(bytes.len() / 2);
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn people_batch(ids: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "id",
            arrow_schema::DataType::Int64,
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(ids))]).unwrap()
    }

    /// Open a one-worker instance in `directory` with a `people (id BIGINT)`
    /// table, and return the ids it holds after `copy` ran.
    fn ids_after_copy(
        directory: &std::path::Path,
        copy: &str,
    ) -> (Result<usize, String>, Vec<i64>) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let instance = runtime
            .block_on(async {
                ShellInstance::open_with_resources(
                    &ShellTarget::pivot(directory.to_str().unwrap()),
                    1,
                    32,
                    datastore_pivotlake::DEFAULT_REFRESH_INTERVAL,
                )
            })
            .unwrap();
        let executor = instance.executor();
        let result = runtime.block_on(async {
            executor
                .execute(
                    "CREATE TABLE people (id BIGINT)".into(),
                    ExecuteOptions::default(),
                )
                .await
                .unwrap();
            copy_from_file(executor, parsed(copy), ExecuteOptions::default())
                .await
                .map(|outcome| outcome.rows)
                .map_err(|error| error.to_string())
        });
        let ids = runtime.block_on(async {
            let select = executor
                .execute(
                    "SELECT id FROM people ORDER BY id".into(),
                    ExecuteOptions::default(),
                )
                .await
                .unwrap();
            let StatementOutput::Rows { batches, .. } = select.output else {
                panic!("SELECT did not return rows");
            };
            batches
                .iter()
                .flat_map(|batch| {
                    let ids = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    ids.values().to_vec()
                })
                .collect()
        });
        drop(instance);
        (result, ids)
    }

    #[test]
    fn loads_an_arrow_ipc_file_into_the_table() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("rows.arrow");
        write_arrow_stream(&file, &people_batch(vec![3, 1, 2]), false);

        let (result, ids) = ids_after_copy(
            directory.path(),
            &format!("people FROM '{}' WITH (FORMAT arrow)", file.display()),
        );

        assert_eq!(result, Ok(3));
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn a_truncated_file_loads_nothing() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("rows.arrow");
        write_arrow_stream(&file, &people_batch(vec![1, 2]), true);

        let (result, ids) = ids_after_copy(
            directory.path(),
            &format!("people FROM '{}' WITH (FORMAT arrow)", file.display()),
        );

        assert!(result.unwrap_err().contains("COPY arrow stream"));
        assert!(ids.is_empty());
    }

    #[test]
    fn the_format_must_be_spelled_out_like_a_copy() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("rows.arrow");
        write_arrow_stream(&file, &people_batch(vec![1]), false);

        let (result, ids) = ids_after_copy(
            directory.path(),
            &format!("people FROM '{}'", file.display()),
        );

        assert!(
            result
                .unwrap_err()
                .contains("the COPY text format is not supported yet; use WITH (FORMAT arrow)")
        );
        assert!(ids.is_empty());
    }

    #[test]
    fn a_missing_file_is_reported_by_path() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("absent.arrow");

        let (result, ids) = ids_after_copy(
            directory.path(),
            &format!("people FROM '{}' WITH (FORMAT arrow)", file.display()),
        );

        let error = result.unwrap_err();
        assert!(error.starts_with(&file.display().to_string()), "{error}");
        assert!(ids.is_empty());
    }
}
