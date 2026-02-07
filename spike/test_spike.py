"""Validation tests for Rust-JAX architectural spike.

Tests verify:
- Round-trip latency is acceptable
- Zero-copy array transfer works
- JIT does not recompile after warmup
- vmap batching provides speedup over serial evaluation

Forces CPU backend because JAX Metal (Apple GPU) does not support int64/float64.
On CUDA systems, set JAX_PLATFORMS=cuda to test GPU path.
"""

import os
import time

# Force CPU backend before importing JAX
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import discopt_spike  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64

HAS_CUDA = False
try:
    _gpu_devs = jax.devices("gpu")
    HAS_CUDA = len(_gpu_devs) > 0
except RuntimeError:
    pass


def _build_quadratic(n_vars: int):
    """Build a jitted vmapped quadratic f(x) = x^T Q x + c^T x."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    Q = jax.random.normal(k1, (n_vars, n_vars), dtype=DTYPE)
    Q = Q @ Q.T
    c = jax.random.normal(k2, (n_vars,), dtype=DTYPE)

    @jax.jit
    def f_single(x):
        return x @ Q @ x + c @ x

    return jax.jit(jax.vmap(f_single)), Q, c


class TestZeroCopy:
    """Verify that Rust-created numpy arrays share memory with Python."""

    def test_pointer_match_f64(self):
        arr, rust_ptr = discopt_spike.create_batch_with_ptr(32, 10)
        py_ptr = discopt_spike.data_pointer(arr)
        assert rust_ptr == py_ptr, (
            f"Rust ptr {rust_ptr} != Python ptr {py_ptr}: not zero-copy"
        )

    def test_pointer_match_f32(self):
        arr, rust_ptr = discopt_spike.create_batch_f32_with_ptr(32, 10)
        py_ptr = discopt_spike.data_pointer_f32(arr)
        assert rust_ptr == py_ptr, (
            f"Rust ptr {rust_ptr} != Python ptr {py_ptr}: not zero-copy"
        )

    def test_ctypes_pointer_match(self):
        arr, rust_ptr = discopt_spike.create_batch_with_ptr(64, 50)
        np_ptr = arr.ctypes.data
        assert rust_ptr == np_ptr, (
            f"Rust ptr {rust_ptr} != ctypes ptr {np_ptr}: not zero-copy"
        )

    def test_no_copy_on_read(self):
        """Rust reading a numpy array should use same memory."""
        arr = np.random.randn(32, 10)
        original_ptr = arr.ctypes.data
        rust_ptr = discopt_spike.data_pointer(arr)
        assert original_ptr == rust_ptr


class TestJITRecompilation:
    """Verify JIT cache behavior after warmup."""

    @pytest.mark.parametrize("n_vars", [10, 50, 100])
    def test_no_recompilation_same_shape(self, n_vars):
        f_batch, _, _ = _build_quadratic(n_vars)

        # Warmup
        warmup = discopt_spike.create_batch(32, n_vars)
        result = f_batch(warmup)
        jax.block_until_ready(result)

        cache_before = f_batch._cache_size()

        # Run 10 more times with same shape
        for _ in range(10):
            arr = discopt_spike.create_batch(32, n_vars)
            result = f_batch(arr)
            jax.block_until_ready(result)

        cache_after = f_batch._cache_size()
        assert cache_before == cache_after, (
            f"JIT recompiled: cache went from {cache_before} to {cache_after}"
        )


class TestRoundTripLatency:
    """Verify round-trip latency for Rust->Python array transfer."""

    @pytest.mark.parametrize("batch_size", [32, 64, 128, 256])
    def test_create_batch_latency(self, batch_size):
        """Rust create_batch should complete in < 500us for moderate sizes."""
        n_vars = 50

        # Warmup
        _ = discopt_spike.create_batch(batch_size, n_vars)

        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            arr = discopt_spike.create_batch(batch_size, n_vars)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        median_us = np.median(times) * 1e6
        assert median_us < 500, (
            f"create_batch latency {median_us:.1f}us > 500us for "
            f"batch={batch_size}, n_vars={n_vars}"
        )

    @pytest.mark.parametrize("batch_size", [32, 64, 128])
    def test_sum_array_latency(self, batch_size):
        """Rust sum_array (Python->Rust) should be fast."""
        n_vars = 50
        arr = np.random.randn(batch_size, n_vars)

        # Warmup
        _ = discopt_spike.sum_array(arr)

        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            discopt_spike.sum_array(arr)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        median_us = np.median(times) * 1e6
        assert median_us < 100, (
            f"sum_array latency {median_us:.1f}us > 100us for "
            f"batch={batch_size}, n_vars={n_vars}"
        )


class TestBatchedSpeedup:
    """Verify that vmap batching is faster than serial evaluation."""

    def test_vmap_faster_than_serial(self):
        """vmap batch of 512 should be significantly faster than 512 serial calls."""
        n_vars = 50
        batch_size = 512
        f_batch, Q, c = _build_quadratic(n_vars)

        @jax.jit
        def f_single(x):
            return x @ Q @ x + c @ x

        arr = np.random.randn(batch_size, n_vars)

        # Warmup both paths
        _ = f_batch(arr)
        jax.block_until_ready(_)
        _ = f_single(arr[0])
        jax.block_until_ready(_)

        # Serial
        serial_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            for i in range(batch_size):
                r = f_single(arr[i])
                jax.block_until_ready(r)
            t1 = time.perf_counter()
            serial_times.append(t1 - t0)

        # Batched
        batch_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            r = f_batch(arr)
            jax.block_until_ready(r)
            t1 = time.perf_counter()
            batch_times.append(t1 - t0)

        serial_us = np.median(serial_times) * 1e6
        batch_us = np.median(batch_times) * 1e6
        speedup = serial_us / batch_us

        # On CPU, vmap should still show significant dispatch overhead reduction
        assert speedup >= 2.0, (
            f"vmap speedup only {speedup:.1f}x (serial={serial_us:.0f}us, "
            f"batch={batch_us:.0f}us). Expected >= 2x."
        )

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available")
    def test_gpu_batch_speedup(self):
        """On CUDA GPU, batch of 512 should be >= 10x faster than serial CPU."""
        n_vars = 50
        batch_size = 512
        f_batch, Q, c = _build_quadratic(n_vars)

        arr = np.random.randn(batch_size, n_vars)

        # Warmup
        _ = f_batch(arr)
        jax.block_until_ready(_)

        cpu = jax.devices("cpu")
        if not cpu:
            pytest.skip("No CPU device for baseline comparison")

        @jax.jit
        def f_single_cpu(x):
            return x @ Q @ x + c @ x

        # Serial on CPU
        _ = f_single_cpu(jax.device_put(arr[0], cpu[0]))
        jax.block_until_ready(_)

        serial_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            for i in range(batch_size):
                r = f_single_cpu(jax.device_put(arr[i], cpu[0]))
                jax.block_until_ready(r)
            t1 = time.perf_counter()
            serial_times.append(t1 - t0)

        # Batched on GPU
        batch_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            r = f_batch(arr)
            jax.block_until_ready(r)
            t1 = time.perf_counter()
            batch_times.append(t1 - t0)

        serial_us = np.median(serial_times) * 1e6
        batch_us = np.median(batch_times) * 1e6
        speedup = serial_us / batch_us

        assert speedup >= 10.0, (
            f"GPU batch speedup only {speedup:.1f}x "
            f"(serial_cpu={serial_us:.0f}us, batch_gpu={batch_us:.0f}us). "
            f"Expected >= 10x."
        )


class TestCorrectness:
    """Verify that Rust-created arrays produce correct JAX results."""

    def test_quadratic_values(self):
        """Check that vmap produces same results as manual loop."""
        n_vars = 10
        batch_size = 8
        f_batch, Q, c = _build_quadratic(n_vars)

        arr = discopt_spike.create_batch(batch_size, n_vars)
        batch_result = f_batch(arr)

        for i in range(batch_size):
            x = jnp.array(arr[i], dtype=DTYPE)
            expected = x @ Q @ x + c @ x
            np.testing.assert_allclose(
                float(batch_result[i]), float(expected), rtol=1e-10
            )

    def test_sum_array_correct(self):
        """Rust sum should match numpy sum."""
        arr = np.random.randn(64, 50)
        rust_sum = discopt_spike.sum_array(arr)
        np_sum = arr.sum()
        np.testing.assert_allclose(rust_sum, np_sum, rtol=1e-12)
