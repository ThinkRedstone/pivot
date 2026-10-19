//! Building the REST catalog client, and blocking on its calls.
//!
//! The Iceberg REST client is async. Every caller of this datastore runs on a
//! blocking thread (a query is planned on the blocking pool), so each catalog
//! call blocks on the ambient runtime, or on a runtime of this crate's own when
//! there is none (a sync test, a standalone tool).

use std::future::Future;
use std::sync::{Arc, LazyLock};

use iceberg::CatalogBuilder;
use iceberg::io::MemoryStorageFactory;
use iceberg_catalog_rest::{RestCatalog, RestCatalogBuilder};

use crate::{IcebergCatalogAuth, IcebergCatalogConfig, Result};

/// The name the client registers the catalog under: it requires one and
/// never uses it, so every catalog gets this one.
const CATALOG_NAME: &str = "pivot_catalog";

/// Build the client for the catalog `config` describes. Nothing is sent yet:
/// the client fetches the catalog's own configuration on its first call, so
/// an unreachable catalog fails there, not here. The client's own storage
/// layer is never used (every file is read through Pivot's ring), so it is
/// given an in-memory storage factory that satisfies the requirement to have
/// one.
pub(crate) fn build_rest_catalog(config: &IcebergCatalogConfig) -> Result<RestCatalog> {
    let mut properties = config.properties.clone();
    properties.insert("uri".to_string(), config.uri.clone());
    // Ask the catalog to vend storage credentials with each table it loads. A
    // catalog that cannot ignores the header; a configured `header.*`
    // property of the same name wins, so vending can be switched off per
    // catalog.
    properties
        .entry("header.X-Iceberg-Access-Delegation".to_string())
        .or_insert_with(|| "vended-credentials".to_string());
    if let Some(warehouse) = &config.warehouse {
        properties.insert("warehouse".to_string(), warehouse.clone());
    }
    match &config.auth {
        Some(IcebergCatalogAuth::Token(token)) => {
            properties.insert("token".to_string(), token.clone());
        }
        Some(IcebergCatalogAuth::OAuth2 {
            credential,
            server_uri,
            scope,
        }) => {
            properties.insert("credential".to_string(), credential.clone());
            if let Some(server_uri) = server_uri {
                properties.insert("oauth2-server-uri".to_string(), server_uri.clone());
            }
            if let Some(scope) = scope {
                properties.insert("scope".to_string(), scope.clone());
            }
        }
        None => {}
    }
    // `load` captures the current runtime before returning its future.
    let catalog = block_on(async {
        RestCatalogBuilder::default()
            .with_storage_factory(Arc::new(MemoryStorageFactory))
            .load(CATALOG_NAME, properties)
            .await
    })
    .map_err(Box::new)?;
    Ok(catalog)
}

/// Drive `future` to completion from a blocking thread: on the ambient
/// multi-thread runtime when there is one, else on [`CATALOG_RUNTIME`]. A
/// single-thread ambient runtime (a current-thread test) cannot be blocked on
/// from inside itself, so it is declined in favour of the crate's own.
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| handle.block_on(future))
        }
        _ => CATALOG_RUNTIME.block_on(future),
    }
}

/// The runtime the catalog client runs on when the process has no
/// multi-thread runtime of its own. A `static` so it is never dropped
/// (dropping an owned runtime inside an async context panics). Two workers:
/// the client's requests are few and sequential.
static CATALOG_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build the Iceberg catalog runtime")
});
