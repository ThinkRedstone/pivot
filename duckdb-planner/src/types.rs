use crate::Error;
use crate::duckdb_bridge::duckdb_types::LogicalTypeId;
use crate::duckdb_bridge::ffi;
use cxx::UniquePtr;
use std::fmt;

/// The payload that completes a parameterized DuckDB type, mirroring
/// DuckDB's own `ExtraTypeInfo`. Most ids fully describe their type and
/// carry [`ExtraTypeInfo::None`]; a parameterized id (`DECIMAL`, `STRUCT`,
/// `LIST`) carries its parameters here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtraTypeInfo {
    None,
    /// `DECIMAL(width, scale)`: total digits and fractional digits.
    Decimal {
        width: u8,
        scale: u8,
    },
    /// `STRUCT(name type, ...)`: each field's name and type, in field order.
    /// A field of an unnamed struct, such as a `(a, b)` tuple, has an empty
    /// name.
    Struct(Vec<(String, BoundLogicalType)>),
    /// `LIST(element)`: the type of each element.
    List(Box<BoundLogicalType>),
}

/// A DuckDB logical type as read off a bound plan: the type id plus the
/// extra info that completes a parameterized id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundLogicalType {
    pub id: LogicalTypeId,
    pub extra: ExtraTypeInfo,
}

impl BoundLogicalType {
    /// A type made of the id alone, for the ids that need nothing else.
    pub fn plain(id: LogicalTypeId) -> Self {
        BoundLogicalType {
            id,
            extra: ExtraTypeInfo::None,
        }
    }

    /// Decode the flat FFI type struct, which holds everything about a type
    /// except the fields of a `STRUCT`. CXX cannot pass an enum with a
    /// payload across the bridge, so the struct carries every parameter
    /// field flat and the id decides which of them are meaningful; this is
    /// the one place that mapping is interpreted.
    pub(crate) fn from_bridge(raw: ffi::BridgeLogicalType) -> Self {
        let id = LogicalTypeId::from_u8(raw.id);
        let extra = match id {
            LogicalTypeId::DECIMAL => ExtraTypeInfo::Decimal {
                width: raw.decimal_width,
                scale: raw.decimal_scale,
            },
            _ => ExtraTypeInfo::None,
        };
        BoundLogicalType { id, extra }
    }

    /// Encode into the flat FFI type struct, the inverse of `from_bridge`. The
    /// flat struct has no room for a `STRUCT`'s fields or a `LIST`'s element
    /// type, so those types are refused.
    fn to_bridge(&self) -> Result<ffi::BridgeLogicalType, Error> {
        let (decimal_width, decimal_scale) = match self.extra {
            ExtraTypeInfo::None => (0, 0),
            ExtraTypeInfo::Decimal { width, scale } => (width, scale),
            ExtraTypeInfo::Struct(_) | ExtraTypeInfo::List(_) => {
                return Err(Error::UnsupportedParameterType(self.id.clone()));
            }
        };
        Ok(ffi::BridgeLogicalType {
            id: self.id.clone() as u8,
            decimal_width,
            decimal_scale,
        })
    }

    /// Encode into the flat FFI column descriptor the DuckDB catalog glue
    /// consumes, the inverse of `from_bridge`: every parameter field is
    /// written flat, zeroed when the id has none.
    pub fn to_duckdb_column(&self, name: String) -> ffi::DuckDBColumn {
        let (decimal_width, decimal_scale) = match &self.extra {
            ExtraTypeInfo::Decimal { width, scale } => (*width, *scale),
            // The flat descriptor has no room for a struct's fields or a
            // list's element type, and no table declares a column of either.
            ExtraTypeInfo::None | ExtraTypeInfo::Struct(_) | ExtraTypeInfo::List(_) => (0, 0),
        };
        ffi::DuckDBColumn {
            name,
            duckdb_logical_type_id: self.id.clone() as u8,
            decimal_width,
            decimal_scale,
        }
    }
}

/// A constant value from the query (or a table-function argument), extracted from
/// DuckDB as a typed value rather than a string.
///
/// Each variant holds the value already decoded into the corresponding Rust type;
/// [`Other`](ScalarValue::Other) covers DuckDB types the bridge doesn't decode
/// into a typed variant and carries the logical type so the consumer can
/// report it.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    /// `HUGEINT`: a 128-bit integer.
    Int128(i128),
    Float32(f32),
    Float64(f64),
    /// `DECIMAL(width, scale)`: the unscaled 128-bit integer, so the numeric
    /// value is `value / 10^scale`.
    Decimal {
        value: i128,
        width: u8,
        scale: u8,
    },
    Utf8(String),
    /// `DATE`: days since the Unix epoch (1970-01-01).
    Date(i32),
    /// `TIMESTAMP`: microseconds since the Unix epoch.
    Timestamp(i64),
    /// `TIMESTAMP WITH TIME ZONE`: microseconds since the Unix epoch, UTC.
    TimestampTz(i64),
    /// `INTERVAL` kept as its three independent components.
    Interval {
        months: i32,
        days: i32,
        micros: i64,
    },
    /// A `VARIANT`: the document text it holds. DuckDB builds a variant from a
    /// string by wrapping the raw text, and hands it back as that same text
    /// under a variant-to-VARCHAR cast, so the consumer parses it as JSON.
    Variant(String),
    /// A `STRUCT`: each field's name and value, in field order.
    Struct(Vec<(String, ScalarValue)>),
    /// A `LIST`: its elements in order. The element type is carried separately
    /// so an empty list still has one.
    List {
        element_type: BoundLogicalType,
        elements: Vec<ScalarValue>,
    },
    /// A NULL, carrying the type the binder resolved for it (an INSERT's NULL
    /// takes the target column's type) so the consumer can build a null of the
    /// right shape. An untyped NULL keeps DuckDB's own `SQLNULL` id.
    Null(BoundLogicalType),
    /// A DuckDB type the bridge does not decode into a typed variant.
    Other(LogicalTypeId),
}

impl ScalarValue {
    /// Build the DuckDB value this scalar describes, the inverse of decoding
    /// one off a plan. [`Other`](ScalarValue::Other) names a type without a
    /// value, so it has nothing to build, and a `STRUCT` cannot be built yet.
    pub fn into_duckdb_value(self) -> Result<UniquePtr<ffi::Value>, Error> {
        Ok(match self {
            ScalarValue::Boolean(value) => ffi::build_boolean_value(value)?,
            ScalarValue::Int8(value) => ffi::build_int8_value(value)?,
            ScalarValue::Int16(value) => ffi::build_int16_value(value)?,
            ScalarValue::Int32(value) => ffi::build_int32_value(value)?,
            ScalarValue::Int64(value) => ffi::build_int64_value(value)?,
            ScalarValue::UInt8(value) => ffi::build_uint8_value(value)?,
            ScalarValue::UInt16(value) => ffi::build_uint16_value(value)?,
            ScalarValue::UInt32(value) => ffi::build_uint32_value(value)?,
            ScalarValue::UInt64(value) => ffi::build_uint64_value(value)?,
            ScalarValue::Int128(value) => {
                ffi::build_int128_value((value >> 64) as i64, value as u64)?
            }
            ScalarValue::Float32(value) => ffi::build_float32_value(value)?,
            ScalarValue::Float64(value) => ffi::build_float64_value(value)?,
            ScalarValue::Decimal {
                value,
                width,
                scale,
            } => ffi::build_decimal_value((value >> 64) as i64, value as u64, width, scale)?,
            ScalarValue::Utf8(value) => ffi::build_varchar_value(&value)?,
            ScalarValue::Date(days) => ffi::build_date_value(days)?,
            ScalarValue::Timestamp(micros) => ffi::build_timestamp_value(micros)?,
            ScalarValue::TimestampTz(micros) => ffi::build_timestamp_tz_value(micros)?,
            ScalarValue::Interval {
                months,
                days,
                micros,
            } => ffi::build_interval_value(months, days, micros)?,
            ScalarValue::Variant(text) => ffi::build_variant_value(&text)?,
            ScalarValue::List {
                element_type,
                elements,
            } => {
                let mut element_values = ffi::new_value_list()?;
                for element in elements {
                    ffi::push_value(element_values.pin_mut(), element.into_duckdb_value()?)?;
                }
                ffi::build_list_value(element_type.to_bridge()?, element_values)?
            }
            ScalarValue::Null(BoundLogicalType {
                extra: ExtraTypeInfo::List(element_type),
                ..
            }) => ffi::build_null_list_value(element_type.to_bridge()?)?,
            // A LIST or STRUCT id alone does not say what the value holds, so
            // there is no complete type to build a NULL of.
            ScalarValue::Null(BoundLogicalType {
                id: id @ (LogicalTypeId::LIST | LogicalTypeId::STRUCT),
                extra: ExtraTypeInfo::None,
            }) => return Err(Error::IncompleteParameterType(id)),
            ScalarValue::Null(logical_type) => ffi::build_null_value(logical_type.to_bridge()?)?,
            ScalarValue::Struct(_) => {
                return Err(Error::UnsupportedParameterType(LogicalTypeId::STRUCT));
            }
            ScalarValue::Other(logical_type) => {
                return Err(Error::UnsupportedParameterType(logical_type));
            }
        })
    }
}

impl fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarValue::Boolean(v) => write!(f, "{v}"),
            ScalarValue::Int8(v) => write!(f, "{v}"),
            ScalarValue::Int16(v) => write!(f, "{v}"),
            ScalarValue::Int32(v) => write!(f, "{v}"),
            ScalarValue::Int64(v) => write!(f, "{v}"),
            ScalarValue::UInt8(v) => write!(f, "{v}"),
            ScalarValue::UInt16(v) => write!(f, "{v}"),
            ScalarValue::UInt32(v) => write!(f, "{v}"),
            ScalarValue::UInt64(v) => write!(f, "{v}"),
            ScalarValue::Int128(v) => write!(f, "{v}"),
            ScalarValue::Float32(v) => write!(f, "{v}"),
            ScalarValue::Float64(v) => write!(f, "{v}"),
            ScalarValue::Decimal {
                value,
                width,
                scale,
            } => write!(f, "{value} as DECIMAL({width},{scale})"),
            ScalarValue::Utf8(v) => write!(f, "{v}"),
            ScalarValue::Date(v) => write!(f, "{v}"),
            ScalarValue::Timestamp(v) => write!(f, "{v}"),
            ScalarValue::TimestampTz(v) => write!(f, "{v}"),
            ScalarValue::Interval {
                months,
                days,
                micros,
            } => write!(f, "INTERVAL {months} months {days} days {micros} micros"),
            ScalarValue::Variant(v) => write!(f, "{v}"),
            ScalarValue::Struct(fields) => {
                f.write_str("{")?;
                for (field_idx, (name, value)) in fields.iter().enumerate() {
                    if field_idx > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{name}: {value}")?;
                }
                f.write_str("}")
            }
            ScalarValue::List { elements, .. } => {
                f.write_str("[")?;
                for (element_idx, element) in elements.iter().enumerate() {
                    if element_idx > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{element}")?;
                }
                f.write_str("]")
            }
            ScalarValue::Null(ty) => write!(f, "NULL as {:?}", ty.id),
            ScalarValue::Other(ty) => write!(f, "{ty:?}"),
        }
    }
}
