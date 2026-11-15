//! [`FileMap`]: a cache's index of what it holds, per file.
//!
//! ```text
//!   RwLock<HashMap<OpenFile, RwLock<Table>>>
//!              │                      │
//!              │                      └── one file's entries, keyed by file offset
//!              └── which files have anything cached at all
//! ```
//!
//! Two lock levels so a lookup contends only on its own file, and so the outer
//! map can be pruned of files that no longer hold anything: it is keyed by an
//! [`OpenFile`], which pins a descriptor or a connection, and a dead entry
//! would keep that alive for every file ever opened.
//!
//! Every access goes through a closure, so no guard ever escapes and the two
//! lock levels are always taken outer-then-inner.

use crate::io::OpenFile;
use ahash::HashMap;
use std::sync::{PoisonError, RwLock};

pub struct FileMap<Table> {
    files: RwLock<HashMap<OpenFile, RwLock<Table>>>,
}

impl<Table: Default> Default for FileMap<Table> {
    fn default() -> Self {
        Self {
            files: RwLock::new(HashMap::default()),
        }
    }
}

impl<Table: Default> FileMap<Table> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read `file`'s table, if it has one.
    pub fn with_table<R>(&self, file: &OpenFile, read: impl FnOnce(&Table) -> R) -> Option<R> {
        let files = self.files.read().unwrap();
        let table = files.get(file)?.read().unwrap();
        Some(read(&table))
    }

    /// Write `file`'s table, if it has one.
    pub fn with_table_mut<R>(
        &self,
        file: &OpenFile,
        write: impl FnOnce(&mut Table) -> R,
    ) -> Option<R> {
        let files = self.files.read().unwrap();
        let mut table = files.get(file)?.write().unwrap();
        Some(write(&mut table))
    }

    /// [`with_table_mut`](Self::with_table_mut) that goes ahead on a lock
    /// another thread poisoned. For callers that run from a `Drop` during an
    /// unwind and must still take their entry out of the map: the table is
    /// only ever read and written under its lock, so a poisoned one holds a
    /// consistent table, and leaving the entry in would be the worse outcome.
    pub fn with_table_mut_tolerating_poison<R>(
        &self,
        file: &OpenFile,
        write: impl FnOnce(&mut Table) -> R,
    ) -> Option<R> {
        let files = self.files.read().unwrap_or_else(PoisonError::into_inner);
        let mut table = files
            .get(file)?
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        Some(write(&mut table))
    }

    /// Write `file`'s table, creating it first if the file has none. The fast
    /// path takes only the outer read lock; a file that was pruned (or never
    /// seen) takes the outer write lock to create its table, with no lock
    /// upgrade in between.
    pub fn with_table_or_create<R>(
        &self,
        file: &OpenFile,
        write: impl FnOnce(&mut Table) -> R,
    ) -> R {
        {
            let files = self.files.read().unwrap();
            if let Some(table) = files.get(file) {
                return write(&mut table.write().unwrap());
            }
        }
        let mut files = self.files.write().unwrap();
        let table = files.entry(file.clone()).or_default();
        write(table.get_mut().unwrap())
    }

    /// Take `file`'s table out entirely, so later lookups miss until
    /// something is cached again. Serializes with every concurrent writer, so
    /// a racing insert either lands in a fresh table or is dropped here.
    pub fn remove(&self, file: &OpenFile) -> Option<Table> {
        self.files
            .write()
            .unwrap()
            .remove(file)
            .map(|table| table.into_inner().unwrap())
    }

    /// Drop `file`'s entry if its table is empty, re-checking under the outer
    /// write lock so a concurrent insert that just refilled it is kept.
    pub fn prune_if_empty(&self, file: &OpenFile, is_empty: impl Fn(&Table) -> bool) {
        let mut files = self.files.write().unwrap();
        if let Some(table) = files.get(file)
            && is_empty(&table.read().unwrap())
        {
            files.remove(file);
        }
    }

    /// Take every table out, leaving the map empty.
    pub fn drain(&self) -> Vec<(OpenFile, Table)> {
        std::mem::take(&mut *self.files.write().unwrap())
            .into_iter()
            .map(|(file, table)| (file, table.into_inner().unwrap()))
            .collect()
    }

    /// Visit every file's table for writing, one at a time, holding the outer
    /// read lock throughout.
    pub fn for_each_table_mut(&self, mut visit: impl FnMut(&OpenFile, &mut Table)) {
        let files = self.files.read().unwrap();
        for (file, table) in files.iter() {
            visit(file, &mut table.write().unwrap());
        }
    }

    #[cfg(test)]
    pub fn contains_file(&self, file: &OpenFile) -> bool {
        self.files.read().unwrap().contains_key(file)
    }

    /// Poison the outer lock, for tests of unwind paths.
    #[cfg(test)]
    pub(crate) fn poison(&self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = self.files.write().unwrap();
            panic!("poison the file map lock");
        }));
    }
}
