//! Hash tables used by grouped aggregation.
//!
//! [`Table`] stores one key and aggregation value per group.
//! [`AggregatedTable`] owns the tables built by a single worker and decides
//! when to partition them for parallel merging.

pub use crate::operations::unary::group::keys::KeyExtractor;
pub use crate::operations::unary::group::values::AggregationValue;

mod hash_table;
pub use hash_table::{HashTable, LiveKey, MAX_LOAD_FACTOR, PersistedKey, Prober, entry_stride};

mod table_reader;
pub(crate) use table_reader::TableReader;

mod scatter;
pub use scatter::StridedScatterRows;

mod aggregated_table;
pub use aggregated_table::{AggregatedTable, AggregatedTableOutput, PartitionBuffers, RadixConfig};

/// Initial number of slots for a new per-worker hash table.
pub const DEFAULT_CAPACITY: usize = 128;
