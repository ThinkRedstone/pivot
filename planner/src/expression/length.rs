//! [`Length`] — SQL `length(string)`, the byte length of the string.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `length(string)` — the number of *bytes* in the string, as `BIGINT`.
/// Arrives as a `BOUND_FUNCTION` named `length` (also `strlen`/`len`).
///
/// This is ClickHouse's `length()` semantics (byte count). It differs from
/// DuckDB's `length`/`strlen`, which count Unicode characters; we don't use
/// DuckDB as the result oracle, so byte count is the intended answer here.
#[derive(Debug, Clone)]
pub struct Length {
    pub input: Box<Expression>,
    pub return_type: Type,
}

impl Display for Length {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "length({})", self.input)
    }
}

impl Length {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let strings = arr.as_string_view();
                // Byte length: each view keeps its string's length in its low
                // 32 bits, so this reads the views alone and never touches the
                // string payload buffers. That makes it O(rows), not O(bytes)
                // — no per-byte scan like a Unicode character count would
                // need. A null row keeps the input's null bit; whatever its
                // view holds is never looked at.
                let lengths: Vec<i64> = strings
                    .views()
                    .iter()
                    .map(|view| i64::from(*view as u32))
                    .collect();
                let lengths = Int64Array::new(lengths.into(), strings.nulls().cloned());
                ExprResult::Array(Arc::new(lengths) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, StringViewArray};
    use rstest::rstest;
    use std::sync::Arc;

    fn strs_table(testing_planner: &mut TestingPlanner) {
        testing_planner.add_table(
            "strs",
            &[(
                "s",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    "hello",
                    "héllo",
                    "",
                    "日本語abc",
                ])) as ArrayRef,
            )],
        );
    }

    #[rstest]
    fn counts_bytes(mut testing_planner: TestingPlanner) {
        strs_table(&mut testing_planner);

        let mut rows = run(&mut testing_planner, "SELECT length(s) FROM strs");

        rows.sort_by_key(|r| only_column(r).as_i64().unwrap());
        // Byte counts, not characters: "héllo" is 6 bytes (é is 2),
        // "日本語abc" is 12 (three 3-byte CJK chars + "abc").
        assert_eq!(
            rows.iter()
                .map(|r| only_column(r).as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 5, 6, 12]
        );
    }

    #[rstest]
    fn a_null_string_has_a_null_length(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "strs",
            &[(
                "s",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    Some("a string longer than twelve bytes"),
                    None,
                    Some("ab"),
                ])) as ArrayRef,
            )],
        );

        let rows = run(&mut testing_planner, "SELECT length(s) FROM strs");

        // A row whose only column is NULL comes back with no column at all.
        let mut lengths: Vec<Option<i64>> = rows
            .iter()
            .map(|row| {
                let columns = row.as_object().unwrap();
                columns.values().next().and_then(|length| length.as_i64())
            })
            .collect();
        lengths.sort();
        assert_eq!(lengths, vec![None, Some(2), Some(33)]);
    }

    #[rstest]
    fn strlen_is_an_alias(mut testing_planner: TestingPlanner) {
        strs_table(&mut testing_planner);

        let mut rows = run(&mut testing_planner, "SELECT strlen(s) FROM strs");

        rows.sort_by_key(|r| only_column(r).as_i64().unwrap());
        assert_eq!(
            rows.iter()
                .map(|r| only_column(r).as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 5, 6, 12]
        );
    }
}
