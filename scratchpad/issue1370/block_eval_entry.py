"""#1370 Part B entry experiment: does vectorising identical blocks pay?

CLAUDE.md §4 — the experiment runs BEFORE the implementation, with the kill
criterion stated up front:

    KILL: if one vectorised pass over K identical blocks is not at least 2x
    faster than the default evaluator's Jacobian + Lagrangian-Hessian cost at
    K = 64, Part B does not ship. The measurement is recorded either way.

The issue's supporting numbers (0.044x for a directional derivative over 64
blocks at once) were taken in JAX against the `.nl`/ASL evaluator. discopt's
default is neither: it is POUNCE's Rust AD tape. So the baseline here is that
tape on the same model, which is the comparison that decides whether the win is
*structure* (identical blocks share one pattern and one coloring) or merely
*engine* (XLA against a serial tape). Both answers are useful; only the first
justifies the build.

Model class, not instance (CLAUDE.md §2): K structurally identical dynamic
blocks — one discretized trajectory each — coupled only through a small vector
of shared parameters. That is the shape both `dae.fit.fit_trajectories` (one
collocation block per trajectory, one shared surrogate) and
`stochastic.extensive_form` (scenario blocks, shared first stage) produce, and
the shape the SCOPF case in the issue has.

**What is and is not matched between the arms.** Both arms compute the same
mathematical quantities over the same point: the constraint Jacobian and the
Lagrangian Hessian, each with respect to the block's own columns *and* the
shared ones. The baseline returns the sparse nonzeros the tape carries; the
candidate returns dense per-block blocks, because that is what a vmapped
`jacfwd`/`hessian` produces before any coloring. At these block sizes that is
the candidate's *upper* bound on work, not a handicap — but it is the reason a
marginal result here should not be read as a marginal result for a coloured
implementation.

Run:  python -u scratchpad/issue1370/block_eval_entry.py --ks 1,4,16,64
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import discopt.modeling as dm
import numpy as np
from discopt.modeling.core import Model

H_STEP = 0.1


def make_block_model(K: int, steps: int, dim: int) -> Model:
    """K identical dynamic blocks sharing one parameter vector.

    Block k carries a trajectory ``z[k]`` of ``steps`` points in ``R^dim``
    obeying an explicit Euler step whose right-hand side is a nonlinear function
    of the SHARED parameters ``w`` — the same rhs in every block, which is
    exactly the "identical blocks" hypothesis under test. Built vectorised (one
    array-valued constraint, many rows), which is also how the DAE and scenario
    builders emit theirs.
    """
    m = Model(f"blocks_K{K}")
    w = m.continuous("w", shape=(dim,), lb=-2.0, ub=2.0)
    z = m.continuous("z", shape=(K, steps, dim), lb=-10.0, ub=10.0)
    rhs = -w[None, None, :] * z[:, :-1, :] + 0.1 * dm.sin(z[:, :-1, :])
    m.subject_to(z[:, 1:, :] - z[:, :-1, :] - H_STEP * rhs == 0.0, name="dyn")
    m.minimize(dm.sum((z - 1.0) ** 2) + 0.01 * dm.sum(w**2))
    m.set_block(z, np.repeat(np.arange(K, dtype=np.int64), steps * dim).reshape(K, steps, dim))
    m.set_block(w, -1)
    return m


def template_callables(steps: int, dim: int):
    """The single block's residuals and Lagrangian, as jax functions of (zb, w)."""
    import jax.numpy as jnp

    def residuals(zb, w):
        # zb: (steps, dim) -> (steps-1, dim) flattened, matching the model's rows.
        rhs = -w[None, :] * zb[:-1, :] + 0.1 * jnp.sin(zb[:-1, :])
        return (zb[1:, :] - zb[:-1, :] - H_STEP * rhs).reshape(-1)

    def lagrangian(zb, w, lam):
        return jnp.sum((zb - 1.0) ** 2) + jnp.dot(lam, residuals(zb, w))

    return residuals, lagrangian


def _block_column_colors(ev, model, steps: int, dim: int, *, kind: str):
    """Seed vectors for a compressed pass over ONE block, and how many there are.

    The pattern is taken from the emitted NLP (``jacobian_structure`` /
    ``hessian_structure``) and restricted to block 0's columns — not assumed from
    the model source, which is the mistake that makes a "structure-aware"
    measurement measure the wrong structure. Greedy distance-1 coloring: two
    columns share a color when no row (or Hessian row) touches both. That is
    exact for a Jacobian and conservative for a symmetric Hessian, where a star
    coloring would need no more colors than this one.
    """
    from discopt.block_structure import resolve_block_structure

    bs = resolve_block_structure(model, ev)
    block0 = np.flatnonzero(bs.var_blocks == 0)
    local_of = {int(c): i for i, c in enumerate(block0)}
    nb = block0.size

    if kind == "jacobian":
        rows, cols = ev.jacobian_structure()
    else:
        rows, cols = ev.hessian_structure()
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    # Adjacency: columns sharing a row cannot share a color.
    neighbours: list[set[int]] = [set() for _ in range(nb)]
    by_row: dict[int, list[int]] = {}
    for r, c in zip(rows.tolist(), cols.tolist()):
        c_local = local_of.get(int(c))
        if c_local is None:
            continue
        by_row.setdefault(int(r), []).append(c_local)
    if kind == "hessian":
        # A Hessian entry (i, j) is itself an adjacency between two columns.
        for r, c in zip(rows.tolist(), cols.tolist()):
            a, b = local_of.get(int(r)), local_of.get(int(c))
            if a is not None and b is not None and a != b:
                neighbours[a].add(b)
                neighbours[b].add(a)
    for members in by_row.values():
        for a in members:
            for b in members:
                if a != b:
                    neighbours[a].add(b)

    color = [-1] * nb
    for c in range(nb):
        used = {color[nb_] for nb_ in neighbours[c] if color[nb_] >= 0}
        k = 0
        while k in used:
            k += 1
        color[c] = k
    n_colors = max(color) + 1 if nb else 0

    seeds = np.zeros((n_colors, nb))
    for c, k in enumerate(color):
        seeds[k, c] = 1.0
    assert nb == steps * dim, f"block 0 has {nb} columns, expected {steps * dim}"
    return seeds, n_colors


def verify_template(ev, model, residuals, lagrangian, x, lam, steps, dim) -> tuple[int, float]:
    """Prove the template computes what the model's block does, before timing it.

    A candidate that is fast because it computes something else is not a
    candidate. This reconstructs block 0's Jacobian and Lagrangian-Hessian
    entries from the TAPE and compares them against the template's, returning
    ``(entries_compared, max_abs_diff)`` — and the caller refuses to report a
    speedup when the comparison count is zero (CLAUDE.md §6) or the difference is
    real (§7: let it fail loudly rather than time a wrong function).
    """
    import jax
    import jax.numpy as jnp
    from discopt.block_structure import resolve_block_structure

    bs = resolve_block_structure(model, ev)
    cols0 = np.flatnonzero(bs.var_blocks == 0)
    rows0 = np.flatnonzero(bs.con_blocks == 0)
    col_of = {int(c): i for i, c in enumerate(cols0)}
    row_of = {int(r): i for i, r in enumerate(rows0)}

    zb = jnp.asarray(x[cols0].reshape(steps, dim))
    w = jnp.asarray(x[:dim])
    lam0 = jnp.asarray(lam[rows0])

    # Jacobian.
    jr, jc = ev.jacobian_structure()
    jv = ev.evaluate_jacobian_values(x)
    tape_J = np.zeros((rows0.size, cols0.size))
    compared = 0
    for r, c, v in zip(np.asarray(jr).tolist(), np.asarray(jc).tolist(), np.asarray(jv).tolist()):
        i, j = row_of.get(int(r)), col_of.get(int(c))
        if i is not None and j is not None:
            tape_J[i, j] = v
            compared += 1
    tmpl_J = np.asarray(jax.jacfwd(residuals, argnums=0)(zb, w)).reshape(rows0.size, cols0.size)
    diff = float(np.max(np.abs(tape_J - tmpl_J)))

    # Lagrangian Hessian (block 0's own columns; lower triangle from the tape).
    hr, hc = ev.hessian_structure()
    hv = ev.evaluate_hessian_values(x, 1.0, lam)
    tape_H = np.zeros((cols0.size, cols0.size))
    for r, c, v in zip(np.asarray(hr).tolist(), np.asarray(hc).tolist(), np.asarray(hv).tolist()):
        i, j = col_of.get(int(r)), col_of.get(int(c))
        if i is not None and j is not None:
            tape_H[i, j] = v
            tape_H[j, i] = v
            compared += 1
    tmpl_H = np.asarray(jax.hessian(lagrangian, argnums=0)(zb, w, lam0)).reshape(
        cols0.size, cols0.size
    )
    diff = max(diff, float(np.max(np.abs(tape_H - tmpl_H))))
    return compared, diff


def load_gate() -> float:
    """1-minute load average. A timing claim under load is not a measurement (§9)."""
    try:
        out = subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout
        print(f"  load: {out.strip()}", flush=True)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"  load: unavailable ({exc})", flush=True)
    return os.getloadavg()[0]


def run_one(K: int, steps: int, dim: int, reps: int, skip_dense: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from discopt._tape_nlp_evaluator import make_evaluator

    print(f"\n=== K={K} steps={steps} dim={dim} ===", flush=True)
    t0 = time.perf_counter()
    model = make_block_model(K, steps, dim)
    ev = make_evaluator(model)
    n, m_rows = ev.n_variables, ev.n_constraints
    print(f"  built in {time.perf_counter() - t0:.2f}s: n={n} m={m_rows}", flush=True)

    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    lam = rng.normal(size=m_rows)

    def baseline_jac():
        return ev.evaluate_jacobian_values(x)

    def baseline_hess():
        return ev.evaluate_hessian_values(x, 1.0, lam)

    residuals, lagrangian = template_callables(steps, dim)
    w = jnp.asarray(x[:dim])
    Z = jnp.asarray(x[dim:].reshape(K, steps, dim))
    L = jnp.asarray(lam.reshape(K, -1))

    # Jacobian of each block's rows wrt (its own columns, the shared columns).
    jac_fn = jax.jit(
        jax.vmap(
            lambda zb, wv: jax.jacfwd(residuals, argnums=(0, 1))(zb, wv),
            in_axes=(0, None),
        )
    )
    # Lagrangian Hessian of each block wrt (its own columns, the shared columns).
    hess_fn = jax.jit(
        jax.vmap(
            lambda zb, wv, lm: jax.hessian(lagrangian, argnums=(0, 1))(zb, wv, lm),
            in_axes=(0, None, 0),
        )
    )

    def candidate_jac():
        return jax.block_until_ready(jac_fn(Z, w))

    def candidate_hess():
        return jax.block_until_ready(hess_fn(Z, w, L))

    # --- the arm an implementation would actually ship: compressed, not dense.
    # Identical blocks share one sparsity pattern AND one coloring, so the whole
    # Jacobian is C_jac directional derivatives over all K blocks at once, and
    # the Hessian C_hess Hessian-vector products. The colors are computed from
    # the block's own pattern, taken from the emitted NLP rather than assumed.
    jac_seeds, n_jac_colors = _block_column_colors(ev, model, steps, dim, kind="jacobian")
    hess_seeds, n_hess_colors = _block_column_colors(ev, model, steps, dim, kind="hessian")
    Sj = jnp.asarray(jac_seeds.reshape(-1, steps, dim))
    Sh = jnp.asarray(hess_seeds.reshape(-1, steps, dim))

    jvp_all = jax.jit(
        jax.vmap(  # over colors
            jax.vmap(  # over blocks
                lambda zb, wv, v: jax.jvp(lambda zz: residuals(zz, wv), (zb,), (v,))[1],
                in_axes=(0, None, None),
            ),
            in_axes=(None, None, 0),
        )
    )
    hvp_all = jax.jit(
        jax.vmap(
            jax.vmap(
                lambda zb, wv, lm, v: jax.jvp(
                    jax.grad(lambda zz: lagrangian(zz, wv, lm)), (zb,), (v,)
                )[1],
                in_axes=(0, None, 0, None),
            ),
            in_axes=(None, None, None, 0),
        )
    )

    def compressed_jac():
        return jax.block_until_ready(jvp_all(Z, w, Sj))

    def compressed_hess():
        return jax.block_until_ready(hvp_all(Z, w, L, Sh))

    print(
        f"  block coloring: {n_jac_colors} Jacobian colors, {n_hess_colors} Hessian colors",
        flush=True,
    )

    compared, diff = verify_template(ev, model, residuals, lagrangian, x, lam, steps, dim)
    print(f"  template vs tape: {compared} entries compared, max |diff| {diff:.3e}", flush=True)
    if compared == 0:
        raise SystemExit("verification compared NOTHING; the timing below would be meaningless")
    if diff > 1e-10:
        raise SystemExit(f"template disagrees with the tape by {diff:.3e}; not a candidate")

    results: dict = {
        "K": K,
        "n": int(n),
        "m": int(m_rows),
        "jacobian_colors": int(n_jac_colors),
        "hessian_colors": int(n_hess_colors),
    }
    for name, base, cand in (
        ("jacobian", baseline_jac, compressed_jac),
        ("hessian", baseline_hess, compressed_hess),
        ("jacobian_dense", baseline_jac, candidate_jac),
        ("hessian_dense", baseline_hess, candidate_hess),
    ):
        if skip_dense and name.endswith("_dense"):
            # A dense per-block Hessian is O(n_block^2) per block; at real block
            # sizes it is a memory wall, not an informative arm.
            continue
        t_compile = time.perf_counter()
        base()
        cand()  # warm-up: pays the XLA trace/compile, which is not per-call cost
        compile_s = time.perf_counter() - t_compile
        b_samples, c_samples = [], []
        for _ in range(reps):  # interleaved A/B (§9), not arm-after-arm
            t = time.perf_counter()
            base()
            b_samples.append(time.perf_counter() - t)
            t = time.perf_counter()
            cand()
            c_samples.append(time.perf_counter() - t)
        b_med, c_med = statistics.median(b_samples), statistics.median(c_samples)
        b_sd = statistics.stdev(b_samples) if reps > 1 else 0.0
        c_sd = statistics.stdev(c_samples) if reps > 1 else 0.0
        results[name] = {
            "baseline_median_s": b_med,
            "baseline_sd_s": b_sd,
            "candidate_median_s": c_med,
            "candidate_sd_s": c_sd,
            "candidate_warmup_s": compile_s,
            "speedup": b_med / c_med if c_med > 0 else float("inf"),
        }
        print(
            f"  {name:9s} tape {b_med * 1e3:9.3f} ms (sd {b_sd * 1e3:6.3f})  "
            f"vmap {c_med * 1e3:9.3f} ms (sd {c_sd * 1e3:6.3f})  "
            f"speedup {b_med / c_med:6.2f}x   [warm-up {compile_s:.2f} s]",
            flush=True,
        )
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,4,16,64")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--kill-speedup", type=float, default=2.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument(
        "--allow-load",
        action="store_true",
        help="run under load anyway. For SHAKING OUT THE SCRIPT only — numbers "
        "produced this way are not measurements (CLAUDE.md §9) and are labelled so.",
    )
    args = ap.parse_args()

    print("#1370 Part B entry experiment", flush=True)
    load = load_gate()
    if load > 2.0 and args.allow_load:
        print(
            "WARNING: running under load with --allow-load; these numbers are NOT a "
            "measurement and must not be quoted.",
            flush=True,
        )
    elif load > 2.0:
        print(
            f"REFUSING: 1-min load average {load:.2f} > 2.0; a timing claim under load is "
            "not a measurement (CLAUDE.md §9).",
            flush=True,
        )
        return 2

    ks = [int(t) for t in args.ks.split(",")]
    out = [run_one(K, args.steps, args.dim, args.reps, args.skip_dense) for K in ks]

    # Probe-fired assertion (§6): a run that measured nothing must not exit 0.
    measured = sum(1 for r in out if "jacobian" in r and "hessian" in r)
    if measured == 0:
        print("NO MEASUREMENTS TAKEN", flush=True)
        return 1

    largest = out[-1]
    base_total = largest["jacobian"]["baseline_median_s"] + largest["hessian"]["baseline_median_s"]
    cand_total = (
        largest["jacobian"]["candidate_median_s"] + largest["hessian"]["candidate_median_s"]
    )
    combined = base_total / cand_total
    print(
        f"\nK={largest['K']}: combined J+H speedup {combined:.2f}x "
        f"(kill criterion: >= {args.kill_speedup}x)",
        flush=True,
    )
    print("VERDICT: " + ("PROCEED" if combined >= args.kill_speedup else "KILL"), flush=True)
    print(f"measurements taken: {measured}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"runs": out, "combined_speedup_at_max_K": combined}, fh, indent=2)
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
