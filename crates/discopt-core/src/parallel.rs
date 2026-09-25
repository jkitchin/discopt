//! Whether this platform can actually run a parallel wave.
//!
//! Rayon builds its global thread pool lazily, on first use, and if that build
//! fails it **panics** rather than returning an error:
//!
//! ```text
//! thread '<unnamed>' panicked at rayon-core/src/registry.rs:171:10:
//! The global thread pool has not been initialized.:
//! ThreadPoolBuildError { kind: IOError(Os { code: 6, kind: WouldBlock }) }
//! ```
//!
//! That is what happens under Pyodide/emscripten, which has no pthreads. The
//! panic crosses the PyO3 boundary as `pyo3_runtime.PanicException`, which
//! derives from `BaseException` *by design* (pyo3 `src/panic.rs`: "so that it
//! will typically propagate all the way through the stack and cause the Python
//! interpreter to exit"). So no `except Exception` fallback on the Python side
//! can catch it, and every serial fallback discopt has is bypassed.
//!
//! The fix is to ask the question before calling, not to catch the panic
//! afterwards: a panic from inside a rayon *task* is a real bug and must stay
//! loud, so nothing here widens a catch. This only declines to enter the
//! parallel branch on a platform that cannot run it.
//!
//! The probe is a capability test, not a platform test. `cfg(target_os =
//! "emscripten")` would cover today's case and silently miss the next
//! threadless target (wasi, a seccomp sandbox, a thread-limited container that
//! is at its limit); spawning one thread asks exactly what rayon is about to
//! ask. It runs once per process and is then a relaxed atomic load.

use std::sync::OnceLock;

/// True when this process can spawn an OS thread, so rayon's global pool can
/// be built.
///
/// Cached: the spawn happens at most once per process. A `false` here is not
/// an error — callers take their serial path, which is always present because
/// the parallel branches are optimizations over a serial default.
pub fn threads_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| match std::thread::Builder::new().spawn(|| {}) {
        Ok(handle) => handle.join().is_ok(),
        Err(_) => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_are_available_on_a_hosted_platform() {
        // The test suite itself runs threads, so anything but `true` here means
        // the probe is broken rather than the platform being threadless.
        assert!(threads_available());
    }

    #[test]
    fn the_probe_is_cached_and_stable() {
        let first = threads_available();
        for _ in 0..1000 {
            assert_eq!(threads_available(), first);
        }
    }
}
