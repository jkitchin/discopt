"""#764 item-4 entry experiment — decompose the tanksize per-node cost.

Answers the >=5x/node kill criterion for the native per-node kernel WITHOUT
building weeks of plumbing: measure each component of the current Python/JAX
node directly, and the pure-Rust floor of the dominant one (OBBT probes).

Findings (2026-07-19, tanksize node LP m=187 n=257 nnz=864):
  * node wall ~= 1352 ms/node (rust_time ~= 0; the wall is Python/JAX)
  * OBBT = 110 probes/node, 35% of wall, 4.25 ms/probe in-loop
  * pure-Rust binding (arrays reused) = 1.28 ms/probe -> ~2.97 ms/probe (~70%)
    is Python marshaling a native loop deletes
  * JAX McCormick build ~33%, Python orchestration ~32% -> native ~0
  => native node ~= 145 ms/node ~= 9.3x. GREEN, kill criterion cleared.

See docs/dev/issue-764-native-node-kernel-scope.md entry experiment step 2.
"""
from __future__ import annotations
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

NL = "python/tests/data/minlplib_nl/tanksize.nl"


def decompose():
    import discopt._rust as R
    from discopt.modeling.core import from_nl

    orig = R.solve_lp_warm_csc_py
    cnt = {"n": 0}
    wall = {"t": 0.0}
    captured: dict = {}

    def wrapped(*a, **k):
        cnt["n"] += 1
        if "args" not in captured and a[1] > 50:  # m>50: a real node LP
            captured["args"] = a
            captured["kw"] = k
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            wall["t"] += time.perf_counter() - t

    R.solve_lp_warm_csc_py = wrapped
    m = from_nl(NL)
    t0 = time.time()
    r = m.solve(time_limit=40.0, max_nodes=10)
    dt = time.time() - t0
    R.solve_lp_warm_csc_py = orig

    nodes = max(r.node_count, 1)
    print(f"nodes={r.node_count} wall={dt:.2f}s -> {dt / nodes * 1000:.0f} ms/node")
    print(f"rust_time={getattr(r, 'rust_time', None)} jax_time={getattr(r, 'jax_time', None)} "
          f"python_time={getattr(r, 'python_time', None)}")
    print(f"OBBT: {cnt['n']} probes ({cnt['n'] / nodes:.0f}/node), "
          f"{wall['t']:.2f}s ({wall['t'] / dt * 100:.0f}% of wall), "
          f"{wall['t'] / max(cnt['n'], 1) * 1000:.2f} ms/probe in-loop")

    # Pure-Rust floor: hammer the raw binding with pre-built arrays (no marshaling).
    a = captured["args"]
    N = 300
    t0 = time.perf_counter()
    for _ in range(N):
        orig(*a, **captured["kw"])
    pr = (time.perf_counter() - t0) / N * 1000
    print(f"pure-Rust binding (arrays reused) = {pr:.3f} ms/probe over {N} calls "
          f"(node LP m={a[1]} n={a[2]} nnz={len(a[5])})")


if __name__ == "__main__":
    decompose()
