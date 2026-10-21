# ruff: noqa: N806 -- matrix names (Q, A, X, Y, Z, M) follow the math
"""Graduation panel for ``DISCOPT_MATMUL_INTERVAL_ENCLOSURE`` (#1842; CLAUDE.md §5).

The flag gives the general 1-D / 2-D matmul (``c @ x`` with an ndarray ``c``,
``x @ x``, ``x @ Q @ x``, ``x @ Z``, ``X @ Y``) a finite interval enclosure where
it used to be ``(-inf, inf)``. A ``.nl`` or QPLIB model is built from scalar
sums and never contains a ``MatMulExpression``, so those corpora cannot reach the
flag; ``--nl-dir`` checks that directly (ON and OFF must agree exactly, and the
enclosure must never fire). The class the flag can change is models written
with ``@``, so the panel is families of them over a size x seed grid:

* ``iqp`` -- ``min x @ Q @ x - c @ x  s.t.  A @ x <= b``, ``Q`` indefinite, ``x``
  integer (brute-force oracle) or continuous;
* ``ball`` -- ``max c @ x  s.t.  x @ x <= r`` (integer: oracle);
* ``bilin`` -- ``min sum(C * (X @ Y))`` over 2-D variables (2x2 integer: oracle);
* ``rowmat`` -- ``(k,) @ (k, n)`` with both operands variable (integer: oracle);
* ``const`` -- the #1842 repro shape, ``(x + a @ M - 10)[0] + sqrt(sum(M @ M)) <= 0``
  over random ``M`` (closed-form oracle);
* ``ro_ell`` -- an ellipsoidal robust counterpart of ``p @ x <= b`` (the class
  #1842 was found on).

Per instance, OFF (``=0``, the opt-out) then ON (variable unset, the default since
graduation), each on a freshly built model, ``deterministic=True``. Checks:

* cert-clean -- every certified objective equals the oracle when there is one;
  no published bound crosses the oracle (or, without one, the best incumbent
  either arm found); every incumbent re-verifies with ``verify_point``; when
  both certify the objectives agree; no certificate is lost ON;
* net-positive -- certificates gained, nodes on instances both certify, final
  bound where neither certifies, total wall.

Proves the probe fired (§6): counts the enclosure firing per arm (ON must fire on
the instances it is credited with; OFF and the ``.nl`` arm must never fire), and
prints an executed-check count, exiting non-zero if nothing was compared.

    python -u discopt_benchmarks/scripts/issue1842_matmul_enclosure_panel.py \
        [--time-limit 20] [--nl-dir python/tests/data/minlplib_nl]

Graduation run, 2026-10-21 (main ``ce69e84e``, ``--time-limit 20``, before the
flip, so OFF was the unset default and ON was ``=1``): 60 models, 430 checks, 0
violations, the enclosure fired on every ON solve; certificates gained ON 4 (all
``const_*``, the #1842 class: OFF ``feasible`` after ~14,000 nodes, ON optimal in
11-35), lost 0; nodes on both-certified 3586 -> 3560 (iqp_int 191->181, 25->21,
49->41, 35->33, 59->55; bilin_int_2_s1 15->17); uncertified-both (iqp_cont_n12,
time limit at identical node counts) one bound tighter, one looser by 5e-3.
``.nl`` (66 in-repo files, 10 s): the enclosure fired 0 times in both arms; the
one status mismatch (cvxnonsep_psig30, OFF ``feasible`` n=79 vs ON optimal n=89)
was the time limit, since OFF rerun alone gave optimal n=89 at 8.1 s.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

FLAG = "DISCOPT_MATMUL_INTERVAL_ENCLOSURE"
FIRED = {"n": 0}


def _install_counter() -> None:
    """Count every call where the flag actually enabled the finite enclosure."""
    import discopt._relax.convexity.interval_eval as ie

    orig = ie._matmul_enclosure_enabled

    def counted() -> bool:
        on = orig()
        FIRED["n"] += bool(on)
        return on

    ie._matmul_enclosure_enabled = counted


def _set_arm(flag: str) -> None:
    """Arm ``"0"`` sets the opt-out; arm ``"1"`` is the default (variable unset)."""
    if flag == "1":
        os.environ.pop(FLAG, None)
    else:
        os.environ[FLAG] = "0"


def _grid(lo: int, hi: int, n: int) -> np.ndarray:
    return np.array(list(itertools.product(range(lo, hi + 1), repeat=n)), dtype=float)


# ── families: each returns (build, oracle or None, maximize) ─────────────────


def iqp(n: int, seed: int, integer: bool):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    Q = (B + B.T) / 2  # indefinite in general
    c = rng.normal(size=n)
    A = rng.normal(size=(2, n))
    b = np.abs(rng.normal(size=2)) + 1.0
    L = 3

    def build():
        import discopt.modeling as dm

        m = dm.Model(f"iqp{n}_s{seed}")
        if integer:
            x = m.integer("x", shape=(n,), lb=-L, ub=L)
        else:
            x = m.continuous("x", shape=(n,), lb=-L, ub=L)
        m.subject_to(A @ x <= b)
        m.minimize(x @ Q @ x - c @ x)
        return m

    oracle = None
    if integer:
        G = _grid(-L, L, n)
        ok = np.all(b + 1e-9 >= G @ A.T, axis=1)
        f = np.einsum("ij,jk,ik->i", G, Q, G) - G @ c
        oracle = float(f[ok].min())
    return build, oracle, False


def ball(n: int, seed: int, integer: bool):
    rng = np.random.default_rng(100 + seed)
    c = rng.normal(size=n)
    r = float(rng.uniform(2.0, 6.0))
    L = 3

    def build():
        import discopt.modeling as dm

        m = dm.Model(f"ball{n}_s{seed}")
        if integer:
            x = m.integer("x", shape=(n,), lb=-L, ub=L)
        else:
            x = m.continuous("x", shape=(n,), lb=-L, ub=L)
        m.subject_to(x @ x <= r)
        m.maximize(c @ x)
        return m

    if integer:
        G = _grid(-L, L, n)
        ok = np.sum(G * G, axis=1) <= r + 1e-9
        oracle = float((G @ c)[ok].max())
    else:
        # max c.x over the ball cut by the box; exact when the ball sits in the box
        oracle = float(np.sqrt(r) * np.linalg.norm(c)) if np.sqrt(r) <= L else None
    return build, oracle, True


def bilin(k: int, seed: int, integer: bool):
    rng = np.random.default_rng(200 + seed)
    C = rng.normal(size=(k, k))
    L = 2

    def build():
        import discopt.modeling as dm

        m = dm.Model(f"bilin{k}_s{seed}")
        mk = m.integer if integer else m.continuous
        X = mk("X", shape=(k, k), lb=-L, ub=L)
        Y = mk("Y", shape=(k, k), lb=-L, ub=L)
        m.subject_to(dm.sum(X) >= 1)
        m.subject_to(dm.sum(Y) <= 2)
        m.minimize(dm.sum(C * (X @ Y)))
        return m

    oracle = None
    if integer and k == 2:
        G = _grid(-L, L, 4)
        Xs = G[G.sum(axis=1) >= 1].reshape(-1, 2, 2)
        Ys = G[G.sum(axis=1) <= 2].reshape(-1, 2, 2)
        P = np.einsum("aij,bjk->abik", Xs, Ys)
        oracle = float(np.einsum("abik,ik->ab", P, C).min())
    return build, oracle, False


def rowmat(k: int, n: int, seed: int, integer: bool):
    rng = np.random.default_rng(300 + seed)
    D = rng.normal(size=n)
    c = rng.normal(size=k)
    L = 2

    def build():
        import discopt.modeling as dm

        m = dm.Model(f"rowmat{k}x{n}_s{seed}")
        mk = m.integer if integer else m.continuous
        x = mk("x", shape=(k,), lb=-L, ub=L)
        Z = mk("Z", shape=(k, n), lb=-L, ub=L)
        m.subject_to(dm.sum(Z) >= 1)
        m.minimize(dm.sum(D * (x @ Z)) + c @ x)
        return m

    oracle = None
    if integer and k * n + k <= 8:
        xs = _grid(-L, L, k)
        Zg = _grid(-L, L, k * n)
        Zs = Zg[Zg.sum(axis=1) >= 1].reshape(-1, k, n)
        val = np.einsum("ai,bij,j->ab", xs, Zs, D) + (xs @ c)[:, None]
        oracle = float(val.min())
    return build, oracle, False


def const(seed: int):
    rng = np.random.default_rng(400 + seed)
    while True:
        M = rng.normal(size=(2, 2)) * 2
        if np.sum(M @ M) >= 0:
            break
    a = rng.normal(size=2)
    opt = 10.0 - float((a @ M)[0]) - float(np.sqrt(np.sum(M @ M)))

    def build():
        import discopt.modeling as dm
        from discopt.modeling.core import Constant

        m = dm.Model(f"const_s{seed}")
        x = m.continuous("x", lb=-30, ub=30)
        row = (x + Constant(a) @ Constant(M) - 10)[0] + dm.sqrt(dm.sum(Constant(M) @ Constant(M)))
        m.subject_to(row <= 0)
        m.maximize(x)
        return m

    return build, (opt if -30 <= opt <= 30 else None), True


def ro_ell(n: int, seed: int):
    rng = np.random.default_rng(500 + seed)
    pbar = rng.uniform(0.5, 2.0, size=n)
    c = rng.uniform(0.5, 2.0, size=n)
    b = float(rng.uniform(4.0, 8.0))
    rho = 0.5

    def build():
        import discopt.modeling as dm
        from discopt.ro import EllipsoidalUncertaintySet, RobustCounterpart

        m = dm.Model(f"ro_ell{n}_s{seed}")
        x = m.integer("x", shape=(n,), lb=0, ub=3)
        p = m.parameter("p", value=pbar.copy())
        m.subject_to(p @ x <= b)
        m.maximize(c @ x)
        RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=rho)).formulate()
        return m

    # Closed-form robust row: pbar.x + rho * ||x|| <= b. The counterpart may be
    # conservative (#1835), so this oracle bounds a certified objective from above
    # but is not asserted equal; see ``oracle_is_upper``.
    G = _grid(0, 3, n)
    ok = G @ pbar + rho * np.linalg.norm(G, axis=1) <= b + 1e-9
    return build, float((G @ c)[ok].max()), True


def instances():
    out = []
    for seed in range(4):
        for n in (3, 4, 5):
            out.append((f"iqp_int_n{n}_s{seed}", *iqp(n, seed, True), False))
        for n in (4, 8, 12):
            out.append((f"iqp_cont_n{n}_s{seed}", *iqp(n, seed, False), False))
        for n in (3, 5):
            out.append((f"ball_int_n{n}_s{seed}", *ball(n, seed, True), False))
        out.append((f"ball_cont_n6_s{seed}", *ball(6, seed, False), False))
        out.append((f"bilin_int_2_s{seed}", *bilin(2, seed, True), False))
        out.append((f"bilin_cont_3_s{seed}", *bilin(3, seed, False), False))
        out.append((f"rowmat_int_2x2_s{seed}", *rowmat(2, 2, seed, True), False))
        out.append((f"rowmat_cont_3x4_s{seed}", *rowmat(3, 4, seed, False), False))
        out.append((f"const_s{seed}", *const(seed), False))
        out.append((f"ro_ell_n4_s{seed}", *ro_ell(4, seed), True))
    return out


def solve(build, flag: str, tl: float):
    from discopt.validation.feasibility import verify_point

    _set_arm(flag)
    m = build()
    FIRED["n"] = 0
    t0 = time.perf_counter()
    r = m.solve(time_limit=tl, deterministic=True)
    wall = time.perf_counter() - t0
    fired = FIRED["n"]
    bad = None
    if r.x is not None and r.status in ("optimal", "feasible"):
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in m._variables]
        )
        bad = not verify_point(m, flat).ok
    return r, wall, bad, fired


def nl_neutrality(paths, tl: float):
    from discopt.modeling.core import from_nl

    checks, viol = 0, []
    for p in paths:
        out = {}
        for f in ("0", "1"):
            _set_arm(f)
            FIRED["n"] = 0
            t0 = time.perf_counter()
            r = from_nl(str(p)).solve(time_limit=tl, deterministic=True)
            out[f] = (r, FIRED["n"], time.perf_counter() - t0)
        (r0, f0, w0), (r1, f1, w1) = out["0"], out["1"]
        checks += 2
        if f0 or f1:
            viol.append(f"{p.name}: enclosure fired on a .nl model (OFF {f0}, ON {f1})")
        same = (
            r0.status == r1.status
            and r0.node_count == r1.node_count
            and (
                r0.objective == r1.objective
                or (
                    r0.objective is not None
                    and r1.objective is not None
                    and abs(r0.objective - r1.objective) <= 1e-9 * (1 + abs(r0.objective))
                )
            )
        )
        # A budgeted exit with an incumbent reports "feasible", not "time_limit":
        # key the escape on the limit actually being reached (wall time).
        timed_out = "time_limit" in (r0.status, r1.status) or max(w0, w1) >= 0.95 * tl
        if not same and not timed_out:
            viol.append(
                f"{p.name}: OFF {r0.status} n={r0.node_count} obj={r0.objective} vs "
                f"ON {r1.status} n={r1.node_count} obj={r1.objective}"
            )
        print(
            f"NL {p.name:24s} OFF {r0.status:10s} n={r0.node_count:<6d} | ON {r1.status:10s} "
            f"n={r1.node_count:<6d} fired={f0}/{f1} same={same}",
            flush=True,
        )
    return checks, viol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--nl-dir", default=None)
    ap.add_argument("--nl-time-limit", type=float, default=10.0)
    ap.add_argument("--skip-families", action="store_true")
    args = ap.parse_args()
    import discopt
    import discopt._relax.convexity.interval_eval as ie

    assert hasattr(ie, "_matmul_enclosure_enabled"), "flag consumer absent from loaded code"
    # §8: a marker unique to the version under test. The panel's ON arm is "variable
    # unset", which is ON only once the default is graduated; on a tree from before
    # the flip both arms would be OFF.
    saved = os.environ.pop(FLAG, None)
    try:
        assert ie._matmul_enclosure_enabled() is True, (
            f"{FLAG} unset does not read ON in the loaded code ({discopt.__file__}): "
            "this is not the graduated tree"
        )
    finally:
        if saved is not None:
            os.environ[FLAG] = saved
    _install_counter()
    print(f"discopt from {discopt.__file__}; load {os.getloadavg()}", flush=True)
    tol = 1e-5
    checks = compared = 0
    viol, gained, lost, unfired = [], [], [], []
    nodes_both = {"0": 0, "1": 0}
    wall = {"0": 0.0, "1": 0.0}
    tighter = looser = 0
    node_moves = []
    if args.nl_dir:
        from pathlib import Path

        c, v = nl_neutrality(sorted(Path(args.nl_dir).glob("*.nl")), args.nl_time_limit)
        checks += c
        compared += c // 2
        viol += v
    for name, build, oracle, maximize, oracle_is_upper in [] if args.skip_families else instances():
        res = {f: solve(build, f, args.time_limit) for f in ("0", "1")}
        compared += 1
        r0, r1 = res["0"][0], res["1"][0]
        checks += 1
        if res["0"][3]:
            viol.append(f"{name}: enclosure fired with the flag OFF ({res['0'][3]})")
        if not res["1"][3]:
            unfired.append(name)
        for f in ("0", "1"):
            wall[f] += res[f][1]
            if res[f][2] is not None:
                checks += 1
                if res[f][2]:
                    viol.append(f"{name} flag={f}: published incumbent fails verify_point")
        incs = [r.objective for r in (r0, r1) if r.objective is not None and r.x is not None]
        ref = oracle
        if ref is None and incs:
            ref = max(incs) if maximize else min(incs)
        for f, r in (("0", r0), ("1", r1)):
            scale = tol * (1 + abs(ref)) if ref is not None else 0.0
            if r.bound is not None and ref is not None and np.isfinite(r.bound):
                checks += 1
                crossed = r.bound < ref - scale if maximize else r.bound > ref + scale
                if crossed and not (oracle_is_upper and maximize):
                    viol.append(f"{name} flag={f}: bound {r.bound} crosses reference {ref}")
            if r.gap_certified and oracle is not None and r.objective is not None:
                checks += 1
                if oracle_is_upper:
                    wrong = r.objective > oracle + scale
                else:
                    wrong = abs(r.objective - oracle) > scale
                if wrong:
                    viol.append(f"{name} flag={f}: certified {r.objective} vs oracle {oracle}")
        c0, c1 = bool(r0.gap_certified), bool(r1.gap_certified)
        if c0 and c1:
            checks += 1
            if abs(r0.objective - r1.objective) > tol * (1 + abs(r0.objective)):
                viol.append(f"{name}: certified objectives differ {r0.objective} vs {r1.objective}")
            nodes_both["0"] += r0.node_count
            nodes_both["1"] += r1.node_count
            if r0.node_count != r1.node_count:
                node_moves.append((name, r0.node_count, r1.node_count))
        if c0 and not c1:
            lost.append(name)
        if c1 and not c0:
            gained.append(name)
        if not c0 and not c1 and r0.bound is not None and r1.bound is not None:
            better = r1.bound < r0.bound if maximize else r1.bound > r0.bound
            worse = r1.bound > r0.bound if maximize else r1.bound < r0.bound
            tighter += bool(better and abs(r1.bound - r0.bound) > 1e-9)
            looser += bool(worse and abs(r1.bound - r0.bound) > 1e-9)
        print(
            f"{name:24s} OFF {r0.status:9s} {'C' if c0 else '-'} n={r0.node_count:<6d} "
            f"b={r0.bound!s:>12.10s} | ON {r1.status:9s} {'C' if c1 else '-'} "
            f"n={r1.node_count:<6d} b={r1.bound!s:>12.10s} fired={res['1'][3]:<5d} "
            f"oracle={oracle!s:>12.10s} wall {res['0'][1]:.1f}/{res['1'][1]:.1f} "
            f"load={os.getloadavg()[0]:.2f}",
            flush=True,
        )
    print(f"\nCOMPARED {compared}; executed checks {checks}")
    print(f"violations {len(viol)}")
    for v in viol:
        print("  VIOLATION", v)
    print(f"ON arm never fired the enclosure on {len(unfired)}: {unfired}")
    print(f"certificates gained ON {len(gained)} {gained}")
    print(f"certificates lost ON   {len(lost)} {lost}")
    print(f"nodes on both-certified: OFF {nodes_both['0']} ON {nodes_both['1']}")
    print(f"node-count changes (OFF -> ON) on both-certified: {node_moves}")
    print(f"uncertified both: ON bound tighter {tighter}, looser {looser}")
    print(f"total wall: OFF {wall['0']:.0f}s ON {wall['1']:.0f}s")
    # §6: an ON solve that never fired the enclosure measured nothing; fail loudly.
    return 0 if compared and checks and not viol and not unfired else 1


if __name__ == "__main__":
    sys.exit(main())
