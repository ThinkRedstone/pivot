use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::thread;

use object_storage::{
    InstanceBandwidth, InstanceBandwidthError, find_instance_type_bandwidth,
    look_up_instance_bandwidth,
};

/// Serve a metadata service that hands out a session token and then names
/// `instance_type` to the request carrying it. Returns its endpoint.
fn spawn_metadata_service(instance_type: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());

    thread::spawn(move || {
        for stream in listener.incoming().take(2) {
            let mut stream = stream.unwrap();
            let head: Vec<String> = BufReader::new(&stream)
                .lines()
                .map(Result::unwrap)
                .take_while(|line| !line.is_empty())
                .collect();
            let body = if head[0].starts_with("PUT /latest/api/token") {
                "session-token"
            } else if head.iter().any(|line| line.ends_with("session-token")) {
                instance_type
            } else {
                ""
            };
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        }
    });

    endpoint
}

#[test]
fn an_instance_type_has_its_published_bandwidth() {
    let bandwidth = find_instance_type_bandwidth("c8gn.4xlarge").unwrap();

    assert_eq!(
        bandwidth,
        InstanceBandwidth {
            baseline_bits_per_second: 50_000_000_000,
            burst_bits_per_second: 50_000_000_000,
        }
    );
}

#[test]
fn a_bursting_instance_type_has_a_baseline_below_its_burst() {
    let bandwidth = find_instance_type_bandwidth("t3.micro").unwrap();

    assert_eq!(bandwidth.baseline_bits_per_second, 64_000_000);
    assert_eq!(bandwidth.burst_bits_per_second, 5_000_000_000);
}

#[test]
fn an_unknown_instance_type_is_an_error() {
    let result = find_instance_type_bandwidth("z99.huge");

    assert!(matches!(
        result,
        Err(InstanceBandwidthError::UnknownInstanceType(name)) if name == "z99.huge"
    ));
}

#[test]
fn the_bandwidth_is_looked_up_from_the_metadata_service() {
    let endpoint = spawn_metadata_service("c8g.4xlarge");

    let bandwidth = look_up_instance_bandwidth(&endpoint).unwrap();

    assert_eq!(bandwidth.baseline_bits_per_second, 7_500_000_000);
}

#[test]
fn a_silent_metadata_endpoint_is_an_error() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);

    let result = look_up_instance_bandwidth(&endpoint);

    assert!(matches!(
        result,
        Err(InstanceBandwidthError::MetadataUnavailable { .. })
    ));
}
