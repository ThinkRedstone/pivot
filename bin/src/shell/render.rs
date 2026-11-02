//! PostgreSQL-style text conversion and aligned table rendering.

use crate::execution::ResultColumn;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow_array::{
    Array, BooleanArray, Date32Array, Decimal64Array, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, IntervalMonthDayNanoArray, RecordBatch,
    StringArray, StringViewArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::DataType;
use planner::types::render_interval;
use unicode_width::UnicodeWidthStr;

#[derive(Debug)]
pub struct TextBatch {
    pub rows: Vec<Vec<Option<String>>>,
}

/// Turn a ring-backed Arrow batch into owned display cells while still on its
/// dispatch worker.
impl dispatch::OutputBatch for TextBatch {
    fn from_record_batch(batch: RecordBatch) -> dispatch::UnaryResult<Self> {
        let fallback_options = FormatOptions::default();
        let fallback_formatters = batch
            .columns()
            .iter()
            .map(|column| ArrayFormatter::try_new(column.as_ref(), &fallback_options).ok())
            .collect::<Vec<_>>();
        let rows = (0..batch.num_rows())
            .map(|row| {
                batch
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(column_index, column)| {
                        format_cell(
                            column.as_ref(),
                            row,
                            fallback_formatters[column_index].as_ref(),
                        )
                    })
                    .collect()
            })
            .collect();
        Ok(Self { rows })
    }
}

fn format_cell(
    array: &dyn Array,
    row: usize,
    fallback: Option<&ArrayFormatter<'_>>,
) -> Option<String> {
    if array.is_null(row) {
        return None;
    }
    let value = match array.data_type() {
        DataType::Boolean => {
            if array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(row)
            {
                "t".to_string()
            } else {
                "f".to_string()
            }
        }
        DataType::Int8 => array
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Int16 => array
            .as_any()
            .downcast_ref::<Int16Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Int32 => array
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::UInt8 => array
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::UInt16 => array
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::UInt32 => array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::UInt64 => array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Utf8View => array
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .value(row)
            .to_string(),
        DataType::Decimal64(_, _) => array
            .as_any()
            .downcast_ref::<Decimal64Array>()
            .unwrap()
            .value_as_string(row),
        DataType::Decimal128(_, _) => array
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap()
            .value_as_string(row),
        DataType::Date32 => array
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap()
            .value_as_date(row)
            .unwrap()
            .to_string(),
        DataType::Timestamp(_, None) => trim_timestamp_fraction(
            array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .value_as_datetime(row)
                .unwrap()
                .to_string(),
        ),
        // A zone-carrying timestamp is a UTC instant; render it with the
        // offset suffix a Postgres server prints under a UTC session.
        DataType::Timestamp(_, Some(_)) => format!(
            "{}+00",
            trim_timestamp_fraction(
                array
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap()
                    .value_as_datetime(row)
                    .unwrap()
                    .to_string(),
            )
        ),
        DataType::Interval(_) => render_interval(
            array
                .as_any()
                .downcast_ref::<IntervalMonthDayNanoArray>()
                .unwrap()
                .value(row),
        ),
        _ => fallback
            .map(|formatter| formatter.value(row).to_string())
            .unwrap_or_else(|| format!("{:?}", array.slice(row, 1))),
    };
    Some(value)
}

fn trim_timestamp_fraction(rendered: String) -> String {
    let Some((instant, fraction)) = rendered.split_once('.') else {
        return rendered;
    };
    let fraction = fraction.trim_end_matches('0');
    if fraction.is_empty() {
        instant.to_string()
    } else {
        format!("{instant}.{fraction}")
    }
}

#[derive(Clone, Copy)]
enum Alignment {
    Left,
    Right,
}

fn alignment(data_type: &DataType) -> Alignment {
    match data_type {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _) => Alignment::Right,
        _ => Alignment::Left,
    }
}

/// Render the default psql aligned format, including its row-count footer.
pub fn render_table(columns: &[ResultColumn], batches: Vec<TextBatch>) -> String {
    let rows = batches
        .into_iter()
        .flat_map(|batch| batch.rows)
        .collect::<Vec<_>>();
    let widths = columns
        .iter()
        .enumerate()
        .map(|(column_index, column)| {
            rows.iter().fold(display_width(&column.name), |width, row| {
                let cell_width = row
                    .get(column_index)
                    .and_then(Option::as_deref)
                    .map(max_line_width)
                    .unwrap_or(0);
                width.max(cell_width)
            })
        })
        .collect::<Vec<_>>();

    let mut output = String::new();
    if !columns.is_empty() {
        let headers = columns
            .iter()
            .zip(&widths)
            .map(|(column, &width)| center(&column.name, width))
            .collect::<Vec<_>>();
        output.push(' ');
        output.push_str(&headers.join(" | "));
        output.push_str(" \n");
        output.push_str(
            &widths
                .iter()
                .map(|width| "-".repeat(width + 2))
                .collect::<Vec<_>>()
                .join("+"),
        );
        output.push('\n');

        for row in &rows {
            render_row(&mut output, columns, &widths, row);
        }
    }
    let row_count = rows.len();
    output.push_str(&format!(
        "({row_count} {})\n\n",
        if row_count == 1 { "row" } else { "rows" }
    ));
    output
}

fn render_row(
    output: &mut String,
    columns: &[ResultColumn],
    widths: &[usize],
    row: &[Option<String>],
) {
    let cell_lines = columns
        .iter()
        .enumerate()
        .map(|(index, _)| {
            row.get(index)
                .and_then(Option::as_deref)
                .unwrap_or("")
                .split('\n')
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let height = cell_lines.iter().map(Vec::len).max().unwrap_or(1);
    for line_index in 0..height {
        let cells = columns
            .iter()
            .enumerate()
            .map(|(column_index, column)| {
                let value = cell_lines[column_index]
                    .get(line_index)
                    .copied()
                    .unwrap_or("");
                if column_index + 1 == columns.len() {
                    pad_last(value, widths[column_index], alignment(&column.data_type))
                } else {
                    pad(value, widths[column_index], alignment(&column.data_type))
                }
            })
            .collect::<Vec<_>>();
        output.push(' ');
        output.push_str(&cells.join(" | "));
        output.push('\n');
    }
}

fn center(value: &str, width: usize) -> String {
    let padding = width.saturating_sub(display_width(value));
    let left = padding / 2;
    let right = padding - left;
    format!("{}{}{}", " ".repeat(left), value, " ".repeat(right))
}

fn pad_last(value: &str, width: usize, alignment: Alignment) -> String {
    match alignment {
        Alignment::Left => value.to_string(),
        Alignment::Right => pad(value, width, alignment),
    }
}

fn pad(value: &str, width: usize, alignment: Alignment) -> String {
    let padding = width.saturating_sub(display_width(value));
    match alignment {
        Alignment::Left => format!("{value}{}", " ".repeat(padding)),
        Alignment::Right => format!("{}{value}", " ".repeat(padding)),
    }
}

fn display_width(value: &str) -> usize {
    UnicodeWidthStr::width(value)
}

fn max_line_width(value: &str) -> usize {
    value.split('\n').map(display_width).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{TextBatch, render_table};
    use crate::execution::ResultColumn;
    use arrow_schema::DataType;

    #[test]
    fn aligned_output_centers_headers_and_aligns_numbers_right() {
        let columns = vec![
            ResultColumn {
                name: "number".to_string(),
                data_type: DataType::Int64,
            },
            ResultColumn {
                name: "name".to_string(),
                data_type: DataType::Utf8,
            },
        ];
        let batches = vec![TextBatch {
            rows: vec![
                vec![Some("1".to_string()), Some("alice".to_string())],
                vec![Some("20".to_string()), None],
            ],
        }];

        let rendered = render_table(&columns, batches);

        assert_eq!(
            rendered,
            " number | name  \n--------+-------\n      1 | alice\n     20 | \n(2 rows)\n\n"
        );
    }

    #[test]
    fn empty_results_keep_their_headers() {
        let columns = vec![ResultColumn {
            name: "value".to_string(),
            data_type: DataType::Utf8,
        }];

        let rendered = render_table(&columns, Vec::new());

        assert_eq!(rendered, " value \n-------\n(0 rows)\n\n");
    }
}
