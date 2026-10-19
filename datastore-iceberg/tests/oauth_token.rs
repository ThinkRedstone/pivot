//! A catalog authenticated to with an OAuth2 credential, against a real Apache
//! Polaris whose tokens expire within seconds: the datastore must keep
//! refreshing after the token it was first issued has expired.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use datastore_iceberg::{IcebergCatalogAuth, IcebergCatalogConfig, IcebergDatastore};
use dispatch::Dispatch;
use object_storage::{ExternalStoreFactory, ObjectStore, StoreError};
use testcontainers::core::{ContainerPort, IntoContainerPort};
use testcontainers::runners::SyncRunner;
use testcontainers::{Container, GenericImage, ImageExt};

const CLIENT_ID: &str = "root";
const CLIENT_SECRET: &str = "s3cr3t";
const SCOPE: &str = "PRINCIPAL_ROLE:ALL";
const CATALOG: &str = "tokens";
/// How long a token Polaris issues stays valid.
const TOKEN_LIFETIME: Duration = Duration::from_secs(2);

/// Polaris with the root principal bootstrapped and one empty catalog, or
/// `None` (with a note) when Docker is not available, so the test skips rather
/// than fails.
fn start_polaris() -> Option<(Container<GenericImage>, String)> {
    let polaris = GenericImage::new("apache/polaris", "1.8.0")
        .with_exposed_port(ContainerPort::Tcp(8181))
        .with_env_var(
            "POLARIS_BOOTSTRAP_CREDENTIALS",
            format!("POLARIS,{CLIENT_ID},{CLIENT_SECRET}"),
        )
        .with_env_var(
            "POLARIS_AUTHENTICATION_TOKEN_BROKER_MAX_TOKEN_GENERATION",
            format!("PT{}S", TOKEN_LIFETIME.as_secs()),
        );
    let polaris = match polaris.start() {
        Ok(polaris) => polaris,
        Err(error) => {
            eprintln!("[oauth_token] skipping: Polaris unavailable: {error}");
            return None;
        }
    };
    let port = polaris.get_host_port_ipv4(8181.tcp()).unwrap();
    let uri = format!("http://localhost:{port}/api/catalog");
    let token = wait_for_token(&uri);
    // The catalog's storage is never touched: the test only lists namespaces.
    ureq::post(&format!(
        "http://localhost:{port}/api/management/v1/catalogs"
    ))
    .set("Authorization", &format!("Bearer {token}"))
    .send_json(serde_json::json!({"catalog": {
        "name": CATALOG,
        "type": "INTERNAL",
        "properties": {"default-base-location": "s3://unused/warehouse"},
        "storageConfigInfo": {
            "storageType": "S3",
            "roleArn": "arn:aws:iam::123456789012:role/unused",
            "allowedLocations": ["s3://unused/warehouse"],
        },
    }}))
    .expect("create the catalog");
    Some((polaris, uri))
}

/// Exchange the root credential for a token, polling until Polaris answers:
/// the container accepts connections a moment before it serves requests.
fn wait_for_token(uri: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let response = ureq::post(&format!("{uri}/v1/oauth/tokens")).send_form(&[
            ("grant_type", "client_credentials"),
            ("client_id", CLIENT_ID),
            ("client_secret", CLIENT_SECRET),
            ("scope", SCOPE),
        ]);
        if let Ok(response) = response {
            let body: serde_json::Value = response.into_json().unwrap();
            return body["access_token"].as_str().unwrap().to_string();
        }
        assert!(Instant::now() < deadline, "Polaris never issued a token");
        thread::sleep(Duration::from_millis(500));
    }
}

#[derive(Debug)]
struct NoStoreFactory;

impl ExternalStoreFactory for NoStoreFactory {
    fn open(&self, root_uri: &str) -> object_storage::Result<Arc<dyn ObjectStore>> {
        Err(StoreError::Config(format!("no store for `{root_uri}`")))
    }
}

#[test]
fn refresh_succeeds_after_the_first_token_expires() {
    let Some((_polaris, uri)) = start_polaris() else {
        return;
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let _runtime = runtime.enter();
    let dispatch = Dispatch::spin_up(1, 64, None);
    let config = IcebergCatalogConfig {
        uri,
        warehouse: Some(CATALOG.to_string()),
        auth: Some(IcebergCatalogAuth::OAuth2 {
            credential: format!("{CLIENT_ID}:{CLIENT_SECRET}"),
            server_uri: None,
            scope: Some(SCOPE.to_string()),
        }),
        ..IcebergCatalogConfig::default()
    };
    let datastore = IcebergDatastore::open(
        &config,
        Arc::new(NoStoreFactory),
        dispatch.dispatcher(),
        Duration::from_secs(3600),
    )
    .expect("open the datastore");

    thread::sleep(TOKEN_LIFETIME + Duration::from_secs(1));
    let refreshed = datastore.refresh();

    refreshed.expect("refresh with a fresh token");
}
