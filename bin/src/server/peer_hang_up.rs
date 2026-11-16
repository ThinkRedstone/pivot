//! Noticing that a connection's client has gone away.

use std::io;
use std::os::fd::AsFd;
use std::time::Duration;

use tokio::io::Interest;
use tokio::net::TcpStream;

/// How long a hang-up can go unnoticed.
const CHECK_INTERVAL: Duration = Duration::from_secs(1);

/// A second handle on a connection's socket that only asks whether the peer has
/// closed its end. It never reads, so it sees the hang-up behind bytes the
/// session has not consumed yet, and under TLS as well as in plaintext.
pub struct PeerHangUp {
    socket: TcpStream,
}

impl PeerHangUp {
    pub fn watch(socket: &TcpStream) -> io::Result<Self> {
        let duplicate = socket.as_fd().try_clone_to_owned()?;
        Ok(Self {
            socket: TcpStream::from_std(duplicate.into())?,
        })
    }

    /// Resolve once the peer has closed its end of the connection.
    pub async fn wait(&self) -> io::Result<()> {
        loop {
            let readiness = self.socket.ready(Interest::READABLE).await?;
            if readiness.is_read_closed() {
                return Ok(());
            }
            // The socket stays readable once bytes have arrived for the session,
            // so asking again straight away would spin.
            tokio::time::sleep(CHECK_INTERVAL).await;
        }
    }
}
