"""Benchmark script for Rust-JAX array transfer latency validation.

Measures round-trip latency of: Rust creates batch -> Python/JAX receives ->
JAX vmap evaluates f(x) = x^T Q x + c^T x -> results returned.

Note: JAX Metal (Apple GPU) backend does not support 64-bit operations and
crashes on basic PRNG. This benchmark uses CPU backend for reliable results.
On CUDA systems, set JAX_PLATFORMS=cuda to test GPU path.
"""

import os
import time

# Force CPU backend if Metal would be default (Metal is broken for this workload)
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import discopt_spike  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64


def build_quadratic(n_vars: int):
    """Build a random quadratic f(x) = x^T Q x + c^T x and return jitted vmap."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    Q = jax.random.normal(k1, (n_vars, n_vars), dtype=DTYPE)
    Q = Q @ Q.T  # Make symmetric positive-definite
    c = jax.random.normal(k2, (n_vars,), dtype=DTYPE)

    @jax.jit
    def f_single(x):
        return x @ Q @ x + c @ x

    f_batch = jax.jit(jax.vmap(f_single))
    return f_batch, Q, c


def time_round_trip(batch_size: int, n_vars: int, f_batch, n_reps: int = 100):
    """Time the Rust->JAX round-trip for a given batch and variable size."""
    # Warmup
    warmup_arr = discopt_spike.create_batch(batch_size, n_vars)
    _ = f_batch(warmup_arr)
    jax.block_until_ready(_)

    # Timed runs
    times_create = []
    times_transfer_eval = []
    times_total = []

    for _ in range(n_reps):
        t0 = time.perf_counter()
        arr = discopt_spike.create_batch(batch_size, n_vars)
        t1 = time.perf_counter()
        result = f_batch(arr)
        jax.block_until_ready(result)
        t2 = time.perf_counter()

        times_create.append(t1 - t0)
        times_transfer_eval.append(t2 - t1)
        times_total.append(t2 - t0)

    return {
        "create_us": np.median(times_create) * 1e6,
        "eval_us": np.median(times_transfer_eval) * 1e6,
        "total_us": np.median(times_total) * 1e6,
    }


def time_rust_receive(batch_size: int, n_vars: int, n_reps: int = 100):
    """Time sending a numpy array to Rust (sum_array)."""
    arr = np.random.randn(batch_size, n_vars)

    # Warmup
    _ = discopt_spike.sum_array(arr)

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        discopt_spike.sum_array(arr)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times) * 1e6


def check_zero_copy():
    """Verify that Rust-created arrays maintain zero-copy semantics."""
    arr, rust_ptr = discopt_spike.create_batch_with_ptr(32, 10)
    py_ptr = discopt_spike.data_pointer(arr)
    np_ptr = arr.ctypes.data
    return rust_ptr == py_ptr == np_ptr


def check_jit_recompilation(n_vars: int):
    """Verify that jit does not recompile after warmup for same-shaped inputs."""
    f_batch, _, _ = build_quadratic(n_vars)

    # Warmup with a specific shape
    warmup = discopt_spike.create_batch(32, n_vars)
    _ = f_batch(warmup)
    jax.block_until_ready(_)

    lowered_count_before = f_batch._cache_size()

    for _ in range(10):
        arr = discopt_spike.create_batch(32, n_vars)
        result = f_batch(arr)
        jax.block_until_ready(result)

    lowered_count_after = f_batch._cache_size()
    return lowered_count_before, lowered_count_after


def main():
    print("=" * 80)
    print("Rust<->JAX Architectural Spike Benchmark")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Compute dtype: {DTYPE}")
    print("=" * 80)

    # Zero-copy check
    zc = check_zero_copy()
    print(f"\nZero-copy verification: {'PASS' if zc else 'FAIL'}")

    # JIT recompilation check
    before, after = check_jit_recompilation(50)
    print(f"JIT cache size before extra evals: {before}, after: {after}")
    print(
        f"JIT recompilation check: "
        f"{'PASS (no recompilation)' if before == after else 'FAIL (recompiled)'}"
    )

    # Round-trip benchmarks
    batch_sizes = [1, 32, 64, 128, 256, 512, 1024]
    n_vars_list = [10, 50, 100]

    print("\n--- Rust create_batch + JAX vmap eval round-trip (median, us) ---")
    print(f"{'batch':>6} | {'n_vars':>6} | {'create':>10} | {'eval':>10} | {'total':>10}")
    print("-" * 60)

    for n_vars in n_vars_list:
        f_batch, _, _ = build_quadratic(n_vars)
        for batch_size in batch_sizes:
            stats = time_round_trip(batch_size, n_vars, f_batch, n_reps=200)
            print(
                f"{batch_size:>6} | {n_vars:>6} | "
                f"{stats['create_us']:>10.1f} | "
                f"{stats['eval_us']:>10.1f} | "
                f"{stats['total_us']:>10.1f}"
            )
        print()

    # Rust receive benchmarks
    print("--- Rust sum_array (Python->Rust, f64) latency (median, us) ---")
    print(f"{'batch':>6} | {'n_vars':>6} | {'latency':>10}")
    print("-" * 35)
    for n_vars in n_vars_list:
        for batch_size in batch_sizes:
            lat = time_rust_receive(batch_size, n_vars, n_reps=200)
            print(f"{batch_size:>6} | {n_vars:>6} | {lat:>10.1f}")
        print()

    # Serial vs batched comparison
    print("--- Serial vs vmap batched evaluation comparison ---")
    n_vars = 50
    f_batch, Q, c = build_quadratic(n_vars)

    @jax.jit
    def f_single(x):
        return x @ Q @ x + c @ x

    batch_size = 512
    arr = np.random.randn(batch_size, n_vars)

    # Warmup
    _ = f_batch(arr)
    jax.block_until_ready(_)
    _ = f_single(arr[0])
    jax.block_until_ready(_)

    # Serial
    serial_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        for i in range(batch_size):
            r = f_single(arr[i])
            jax.block_until_ready(r)
        t1 = time.perf_counter()
        serial_times.append(t1 - t0)

    # Batched
    batch_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        r = f_batch(arr)
        jax.block_until_ready(r)
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)

    serial_median = np.median(serial_times) * 1e6
    batch_median = np.median(batch_times) * 1e6
    speedup = serial_median / batch_median

    print(f"  n_vars={n_vars}, batch_size={batch_size}")
    print(f"  Serial (512 individual jit calls): {serial_median:.0f} us")
    print(f"  Batched (single vmap call):        {batch_median:.0f} us")
    print(f"  Speedup:                           {speedup:.1f}x")

    print("\n" + "=" * 80)
    print("GO/NO-GO SUMMARY")
    print("=" * 80)
    print(f"  Zero-copy:         {'GO' if zc else 'NO-GO'}")
    print(f"  JIT no-recompile:  {'GO' if before == after else 'NO-GO'}")
    print(
        f"  vmap speedup:      {speedup:.1f}x "
        f"({'GO' if speedup >= 5 else 'MARGINAL' if speedup >= 2 else 'NO-GO'})"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
