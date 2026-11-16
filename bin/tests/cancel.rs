//! A statement stops running once its client cancels it or goes away.

mod common;

use std::time::{Duration, Instant};

use common::{Conn, RawConn, conn, lock_serial, server_port};
use nix::sys::resource::{UsageWho, getrusage};
use nix::sys::time::TimeValLike;
use rstest::rstest;
use tokio_postgres::NoTls;
use tokio_postgres::error::SqlState;

/// A statement that keeps every worker busy for far longer than a test runs.
const LONG_RUNNING_QUERY: &str = "SELECT SUM(range) FROM range(1000000000000)";

/// How long a statement gets to start running before the test interrupts it.
const STATEMENT_HEAD_START: Duration = Duration::from_millis(300);

/// CPU time this process (the test and the server it hosts) has used so far.
fn read_process_cpu_time() -> Duration {
    let usage = getrusage(UsageWho::RUSAGE_SELF).unwrap();
    let microseconds =
        usage.user_time().num_microseconds() + usage.system_time().num_microseconds();
    Duration::from_micros(microseconds as u64)
}

/// Whether the server stops burning CPU within `deadline`: a window in which it
/// used less than a tenth of one core.
fn server_goes_idle_within(deadline: Duration) -> bool {
    const WINDOW: Duration = Duration::from_millis(200);
    let started = Instant::now();
    while started.elapsed() < deadline {
        let cpu_time_before = read_process_cpu_time();
        std::thread::sleep(WINDOW);
        if read_process_cpu_time() - cpu_time_before < WINDOW / 10 {
            return true;
        }
    }
    false
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn cancel_request_stops_the_running_statement(#[future] conn: Conn) {
    let cancel_token = conn.cancel_token();

    let (statement, cancel_request) = tokio::join!(conn.simple_query(LONG_RUNNING_QUERY), async {
        tokio::time::sleep(STATEMENT_HEAD_START).await;
        cancel_token.cancel_query(NoTls).await
    });

    cancel_request.unwrap();
    assert_eq!(
        statement.unwrap_err().code(),
        Some(&SqlState::QUERY_CANCELED)
    );
    assert!(server_goes_idle_within(Duration::from_secs(10)));
}

#[test]
fn client_disconnect_stops_the_running_statement() {
    let _serial = lock_serial();
    let mut client = RawConn::connect(server_port());
    client.query(LONG_RUNNING_QUERY);
    std::thread::sleep(STATEMENT_HEAD_START);

    drop(client);

    assert!(server_goes_idle_within(Duration::from_secs(10)));
}
