#!/usr/bin/env python3
"""#843 graduation panel — DISCOPT_QUBO_PRIMAL (QUBO/Ising local-search primal).

The CLAUDE.md §5 differential panel for flipping the #846 seed default-ON: flag
ON vs OFF, requiring BOTH (1) *cert-clean* — zero soundness violations, and the
flag a proven no-op wherever its structure is absent — AND (2) *net-positive* —
measurably helpful on the structure-carrying class, not merely sound.

The flag's structural gate is exact and cheap (``qubo_primal.is_qubo`` +
the MIQP classification): it can only fire on an UNCONSTRAINED, ALL-BINARY,
QUADRATIC-objective model. No instance of the in-repo corpus is a QUBO, so the
cert arm here is (a) an executed structural no-fire proof over every vendored
.nl (not an assumption — counted assertions, §6), plus (b) an interleaved
ON-vs-OFF differential solve over a fast corpus subset asserting byte-identical
(status, objective, node_count).

The structure-carrying arm draws from the QUBO class itself:
  * small dense QUBOs with a brute-force oracle (soundness against the TRUE
    optimum, not just the run's own bound);
  * chimera-topology Ising instances (C_s: s×s grid of K4,4 cells, ±1
    couplers — the D-Wave topology of the minlplib ``chimera_k64ising``
    family, generated at C4/C8/C12; C12 = 1152 binary vars, the named
    instance's scale) — the class the issue names, constructed rather than
    copied so the panel measures the CLASS, not the probe instance (§2).

Soundness assertions on every solve: a MAXIMIZE incumbent must not exceed the
run's certified dual bound, nor the brute-force optimum where one exists.

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      python discopt_benchmarks/scripts/issue843_qubo_primal_graduation_panel.py \
        [--quick] [--out-dir reports]

Exit codes: 0 = panel PASS (cert-clean AND net-positive), 1 = any violation or
zero executed assertions (a probe that measured nothing must not read as a pass).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent.parent
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"

# Fast subset of the vendored corpus for the interleaved ON-vs-OFF differential
# (small instances that certify in seconds; the structural no-fire proof covers
# the WHOLE vendored corpus regardless).
_NEUTRALITY_SOLVES = [
    "gbd",
    "alan",
    "ex1222",
    "st_e13",
    "st_miqp1",
    "st_miqp2",
    "st_test1",
    "nvs04",
]


def _set_arm(arm: str) -> None:
    os.environ["DISCOPT_QUBO_PRIMAL"] = {"off": "0", "on": "1"}[arm]


def chimera_edges(s: int):
    """Chimera C_s: s×s grid of K4,4 cells (8 vars each): intra-cell bipartite
    edges + inter-cell couplers (left half vertical, right half horizontal)."""

    def idx(r: int, c: int, h: int, k: int) -> int:
        return ((r * s + c) * 2 + h) * 4 + k

    edges = []
    for r in range(s):
        for c in range(s):
            for a in range(4):
                for b in range(4):
                    edges.append((idx(r, c, 0, a), idx(r, c, 1, b)))
            if r + 1 < s:
                for a in range(4):
                    edges.append((idx(r, c, 0, a), idx(r + 1, c, 0, a)))
            if c + 1 < s:
                for b in range(4):
                    edges.append((idx(r, c, 1, b), idx(r, c + 1, 1, b)))
    return 8 * s * s, edges


def build_chimera_ising(s: int, seed: int):
    """MAXIMIZE Σ J_ij s_i s_j, J ∈ {−1,+1}, spins s = 2x − 1 expanded to the
    binary quadratic form — the expanded Σ J x_i x_j shape the minlplib
    ``chimera_k64ising`` .nl files carry."""
    import discopt.modeling as dm

    n, edges = chimera_edges(s)
    rng = np.random.default_rng(seed)
    m = dm.Model(f"chimera_c{s}_seed{seed}")
    x = [m.binary(f"x{i}") for i in range(n)]
    lin = np.zeros(n)
    const = 0.0
    terms = []
    for i, j in edges:
        jw = float(rng.choice((-1.0, 1.0)))
        terms.append(4.0 * jw * (x[i] * x[j]))
        lin[i] -= 2.0 * jw
        lin[j] -= 2.0 * jw
        const += jw
    for i in range(n):
        if lin[i]:
            terms.append(float(lin[i]) * x[i])
    m.maximize(dm.sum(terms) + const)
    return m


def build_dense_qubo(n: int, seed: int):
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    m = dm.Model(f"dense_qubo_n{n}_seed{seed}")
    x = [m.binary(f"x{i}") for i in range(n)]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(rng.integers(-3, 4))
            if w:
                terms.append(w * (x[i] * x[j]))
    m.maximize(dm.sum(terms))
    return m


def brute_force_max(model) -> float:
    from discopt._relax.nlp_evaluator import NLPEvaluator

    ev = NLPEvaluator(model)
    n = ev.n_variables
    assert n <= 16, "brute force capped at 2^16"
    return max(
        -float(ev.evaluate_objective(np.array([(k >> b) & 1 for b in range(n)], float)))
        for k in range(1 << n)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="short time limits / skip C12")
    ap.add_argument("--out-dir", type=str, default=str(_REPO / "reports"))
    args = ap.parse_args()

    import discopt.modeling as dm
    from discopt.qubo_primal import is_qubo, qubo_local_search

    executed = 0  # counted assertions (§6): zero at the end = probe measured nothing
    violations: list[str] = []
    report: dict = {"timestamp": time.strftime("%Y-%m-%dT%H-%M-%S"), "quick": args.quick}

    # ------------------------------------------------------------------ #
    # N1: structural no-fire proof over the ENTIRE vendored corpus.
    # ------------------------------------------------------------------ #
    print("== N1: structural no-fire over the vendored corpus ==", flush=True)
    n1 = 0
    for nl in sorted(_NL_DIR.glob("*.nl")):
        m = dm.from_nl(str(nl))
        fired_gate = is_qubo(m)
        seed_pt = qubo_local_search(m, deadline=time.perf_counter() + 5.0)
        if fired_gate or seed_pt is not None:
            violations.append(f"N1: {nl.name} fired the QUBO gate (is_qubo={fired_gate})")
        n1 += 1
    executed += n1
    print(f"   {n1} instances checked, 0 expected to fire", flush=True)
    report["n1_no_fire_checked"] = n1

    # ------------------------------------------------------------------ #
    # N2: interleaved ON-vs-OFF differential on the fast corpus subset —
    # byte-identical (status, objective, node_count) required.
    # ------------------------------------------------------------------ #
    print("== N2: ON-vs-OFF differential (fast corpus subset) ==", flush=True)
    tl = 30 if not args.quick else 15
    n2_rows = []
    for name in _NEUTRALITY_SOLVES:
        path = _NL_DIR / f"{name}.nl"
        if not path.exists():
            continue
        out = {}
        for arm in ("off", "on"):
            _set_arm(arm)
            m = dm.from_nl(str(path))
            t0 = time.perf_counter()
            r = m.solve(time_limit=tl)
            out[arm] = {
                "status": r.status,
                "objective": r.objective,
                "node_count": r.node_count,
                "wall": round(time.perf_counter() - t0, 2),
            }
        same = (
            out["off"]["status"] == out["on"]["status"]
            and (
                (out["off"]["objective"] is None and out["on"]["objective"] is None)
                or (
                    out["off"]["objective"] is not None
                    and out["on"]["objective"] is not None
                    and abs(out["off"]["objective"] - out["on"]["objective"]) <= 1e-9
                )
            )
            and out["off"]["node_count"] == out["on"]["node_count"]
        )
        executed += 1
        if not same:
            violations.append(f"N2: {name} differs ON vs OFF: {out}")
        n2_rows.append({"instance": name, **out, "identical": same})
        print(f"   {name}: identical={same} off={out['off']} on={out['on']}", flush=True)
    report["n2_differential"] = n2_rows

    # ------------------------------------------------------------------ #
    # S1: small dense QUBOs vs a brute-force oracle (ON arm).
    # ------------------------------------------------------------------ #
    print("== S1: small QUBOs vs brute-force oracle ==", flush=True)
    s1_rows = []
    _set_arm("on")
    for seed in range(5):
        m = build_dense_qubo(12, seed)
        opt = brute_force_max(m)
        r = m.solve(time_limit=30)
        executed += 1
        if r.objective is None:
            violations.append(f"S1: dense12 seed{seed}: ON arm found no incumbent")
        elif r.objective > opt + 1e-6:
            violations.append(f"S1 SOUNDNESS: dense12 seed{seed}: {r.objective} > optimum {opt}")
        s1_rows.append({"seed": seed, "optimum": opt, "objective": r.objective, "status": r.status})
        print(f"   dense12 seed{seed}: opt={opt} got={r.objective} ({r.status})", flush=True)
    report["s1_small_oracle"] = s1_rows

    # ------------------------------------------------------------------ #
    # S2: structure-carrying differential — chimera-topology Ising + dense
    # QUBOs, interleaved OFF/ON. Metric: incumbent presence + quality at the
    # time limit; soundness: incumbent ≤ certified dual bound (MAXIMIZE).
    # ------------------------------------------------------------------ #
    print("== S2: QUBO-class differential (chimera topology + dense) ==", flush=True)
    cases = [
        ("chimera_c4", lambda: build_chimera_ising(4, 7), 30),
        ("dense_n60", lambda: build_dense_qubo(60, 1), 30),
        ("chimera_c8", lambda: build_chimera_ising(8, 7), 60),
    ]
    if not args.quick:
        cases.append(("chimera_c12", lambda: build_chimera_ising(12, 7), 120))
    s2_rows = []
    for name, build, tl in cases:
        out = {}
        for arm in ("off", "on"):
            _set_arm(arm)
            m = build()
            t0 = time.perf_counter()
            r = m.solve(time_limit=tl)
            wall = time.perf_counter() - t0
            out[arm] = {
                "status": r.status,
                "objective": r.objective,
                "bound": r.bound,
                "node_count": r.node_count,
                "wall": round(wall, 1),
            }
            executed += 1
            if r.objective is not None and r.bound is not None and r.objective > r.bound + 1e-6:
                violations.append(
                    f"S2 SOUNDNESS: {name}[{arm}]: incumbent {r.objective} above "
                    f"dual bound {r.bound} (MAXIMIZE)"
                )
        off_obj = out["off"]["objective"]
        on_obj = out["on"]["objective"]
        better = (on_obj is not None) and (off_obj is None or on_obj >= off_obj - 1e-9)
        strictly = (on_obj is not None) and (off_obj is None or on_obj > off_obj + 1e-9)
        if not better:
            violations.append(f"S2 REGRESSION: {name}: ON incumbent worse than OFF: {out}")
        s2_rows.append(
            {"instance": name, **out, "on_not_worse": better, "on_strictly_better": strictly}
        )
        print(f"   {name}: off={out['off']} on={out['on']}", flush=True)
    report["s2_structure_differential"] = s2_rows

    # ------------------------------------------------------------------ #
    # verdict
    # ------------------------------------------------------------------ #
    n_better = sum(1 for row in s2_rows if row["on_strictly_better"])
    cert_clean = not violations
    net_positive = n_better >= 1 and all(row["on_not_worse"] for row in s2_rows)
    report["executed_assertions"] = executed
    report["violations"] = violations
    report["cert_clean"] = cert_clean
    report["net_positive"] = net_positive
    report["s2_strictly_better_count"] = n_better

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / f"issue843_graduation_panel_{report['timestamp']}.json"
    js.write_text(json.dumps(report, indent=2))

    print("\n─── verdict ───")
    print(f"executed assertions: {executed}")
    for v in violations:
        print(f"VIOLATION: {v}")
    print(f"cert-clean:   {'YES' if cert_clean else 'NO'}")
    print(f"net-positive: {'YES' if net_positive else 'NO'} ({n_better} strictly better)")
    print(f"JSON: {js}")
    if executed == 0:
        print("FATAL: zero executed assertions — the panel measured nothing (§6)")
        return 1
    return 0 if (cert_clean and net_positive) else 1


if __name__ == "__main__":
    raise SystemExit(main())
