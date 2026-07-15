#!/usr/bin/env python
"""P1 entry experiment — per-atom factorable-envelope tightness audit (issue #632).

Measures, deterministically and fully in-container, **where discopt's default
factorable relaxation is loose, per atom**. For each elementary atom and common
composition it builds a small model over representative boxes, solves the ROOT LP
with discopt's OWN in-house Rust simplex (``MccormickLPRelaxer.solve_at_node`` —
NOT scipy/HiGHS), and compares the bound to the true optimum (analytic, fine 1-D
scan, or exact vertex enumeration for multilinear atoms whose extrema are at box
corners). It reports the absolute and relative gap and whether the objective
obtained a real per-atom envelope or fell back to the separable-interval /
feasibility path (``fb=True`` ⇒ no genuine envelope was applied).

This is the P1 measurement of record for the plan
``docs/dev/avm-canonicalization-plan.md`` §0′ (SOTA per-atom factorable
envelopes) and produces the numbers in ``docs/dev/p1-atom-tightness-audit.md``.

Usage (from repo root, extension built)::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/p1_atom_tightness_audit.py            # atoms only
    ... p1_atom_tightness_audit.py --census    # + 62-file corpus fallback census
    ... p1_atom_tightness_audit.py --json out.json

Determinism: no randomness, no timestamps; fine scans use fixed grid sizes.

Oracle handoff: every number here is computed in-container (analytic / fine scan /
vertex enumeration). The *end-to-end certified* nvs09 root gap against
``minlplib.solu`` and the BARON side-by-side require the user's local host and are
NOT computed here (BARON is absent in-container); those are called out explicitly
in the report as handoffs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
sys.path.insert(0, str(_REPO / "python"))

_SCAN_N = 400_001  # 1-D fine-scan resolution (deterministic)


# ---------------------------------------------------------------------------
# Root-bound measurement (discopt's own in-house simplex)
# ---------------------------------------------------------------------------
class _CaptureWarnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def root_bound(model) -> tuple[str, float | None, bool]:
    """Return ``(status, lower_bound, objective_fell_back)`` for the root LP.

    ``objective_fell_back`` is True when the relaxation could not linearize the
    objective and dropped to the separable-interval / feasibility path — i.e. no
    genuine per-atom envelope was applied to the nonlinear objective. Detected by
    capturing the ``build_milp_relaxation`` warning; the module's ``_warn_once``
    dedupe cache is cleared per call so every build is observed.
    """
    from discopt._jax import milp_relaxation as milp
    from discopt._jax.mccormick_lp import MccormickLPRelaxer

    milp._warned_messages.clear()
    cap = _CaptureWarnings()
    logger = logging.getLogger("discopt._jax.milp_relaxation")
    prev_level = logger.level
    logger.addHandler(cap)
    logger.setLevel(logging.WARNING)
    try:
        lbs, ubs = [], []
        for v in model._variables:
            lbs.append(np.asarray(v.lb, dtype=np.float64).ravel())
            ubs.append(np.asarray(v.ub, dtype=np.float64).ravel())
        res = MccormickLPRelaxer(model).solve_at_node(np.concatenate(lbs), np.concatenate(ubs))
    finally:
        logger.removeHandler(cap)
        logger.setLevel(prev_level)
    fell_back = any("could not linearize the objective" in m for m in cap.messages)
    bound = float(res.lower_bound) if res.lower_bound is not None else None
    return res.status, bound, fell_back


# ---------------------------------------------------------------------------
# True-optimum oracles (in-container)
# ---------------------------------------------------------------------------
def scan_min_1d(f, lo: float, hi: float, n: int = _SCAN_N) -> float:
    xs = np.linspace(lo, hi, n)
    return float(np.min(f(xs)))


def multilinear_corner_opt(exps, lbs, ubs, coef: float, sense: str) -> float:
    """Exact optimum of ``coef * prod(x_i**e_i)`` over a box.

    Extrema of a multilinear / signed-monomial function over a box are attained
    at a vertex, so enumerating the ``2**n`` corners is exact (used only for small
    n). ``sense`` is 'min' or 'max'.
    """
    best: float | None = None
    for pt in itertools.product(*[(lo, hi) for lo, hi in zip(lbs, ubs, strict=True)]):
        val = coef * float(np.prod([p**e for p, e in zip(pt, exps, strict=True)]))
        if best is None or (sense == "min" and val < best) or (sense == "max" and val > best):
            best = val
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def _new_model():
    from discopt import Model

    return Model()


_con_counter = [0]


def _add_scalar_constraint(model, expr) -> None:
    """Attach a single scalar (in)equality constraint to *model*."""
    from discopt.modeling.sets import RangeSet

    _con_counter[0] += 1
    name = f"aux_c{_con_counter[0]}"
    model.constraint(RangeSet(1, 1), lambda i: expr, name=name)


def _gap(true_opt: float, bound: float | None) -> tuple[float | None, float | None]:
    if bound is None:
        return None, None
    abs_gap = true_opt - bound
    rel_gap = abs_gap / (1.0 + abs(true_opt))
    return abs_gap, rel_gap


# ---------------------------------------------------------------------------
# Atom experiments
# ---------------------------------------------------------------------------
def _record(rows, group, name, box, true_opt, status, bound, fell_back, sound_ref=None):
    abs_gap, rel_gap = _gap(true_opt, bound)
    # Soundness: a valid relaxation bound must not exceed the true optimum
    # (min sense) beyond numerical noise.
    unsound = bound is not None and bound > true_opt + 1e-6
    rows.append(
        {
            "group": group,
            "atom": name,
            "box": box,
            "true_opt": true_opt,
            "status": status,
            "bound": bound,
            "fell_back": fell_back,
            "abs_gap": abs_gap,
            "rel_gap": rel_gap,
            "unsound": bool(unsound),
        }
    )


def run_atom_audit() -> list[dict]:
    import discopt.modeling as dm

    rows: list[dict] = []

    # -- Group A: base univariate atoms (both directions) ------------------
    # min f exposes the convex-underestimator; min(-f) exposes the secant side.
    univ = [
        ("x**2", lambda x: x**2, lambda z: z**2, -2.0, 3.0),
        ("x**3", lambda x: x**3, lambda z: z**3, -2.0, 2.0),
        ("x**4", lambda x: x**4, lambda z: z**4, -2.0, 3.0),
        ("exp(x)", lambda x: dm.exp(x), lambda z: np.exp(z), -1.0, 2.0),
        ("log(x)", lambda x: dm.log(x), lambda z: np.log(z), 0.5, 5.0),
        ("sqrt(x)", lambda x: x**0.5, lambda z: np.sqrt(z), 0.5, 5.0),
        ("1/x", lambda x: 1.0 / x, lambda z: 1.0 / z, 0.5, 5.0),
        ("x**-2", lambda x: x ** (-2), lambda z: z**-2.0, 0.5, 5.0),
        ("x**2.5", lambda x: x**2.5, lambda z: z**2.5, 0.5, 5.0),
        ("x**0.2", lambda x: x**0.2, lambda z: z**0.2, 0.5, 5.0),
    ]
    for name, fexpr, fpy, lo, hi in univ:
        for sense in ("min", "max"):
            m = _new_model()
            x = m.continuous("x", lb=lo, ub=hi)
            if sense == "min":
                m.minimize(fexpr(x))
                true_opt = scan_min_1d(fpy, lo, hi)
            else:
                m.minimize(-fexpr(x))
                true_opt = scan_min_1d(lambda z, g=fpy: -g(z), lo, hi)
            st, b, fb = root_bound(m)
            _record(rows, "A_univariate", f"{name} [{sense}]", f"[{lo},{hi}]", true_opt, st, b, fb)

    # -- Group B: bilinear / trilinear / multilinear (interior slice) ------
    # A single linear equality forces the optimum off the vertices, where
    # recursive McCormick is exact; the residual gap is the envelope's true
    # looseness. Wide vs narrow boxes isolate the box-width dependence.
    def product_slice(varboxes, sum_rhs, sense):
        m = _new_model()
        vs = [m.continuous(f"v{i}", lb=lo, ub=hi) for i, (lo, hi) in enumerate(varboxes)]
        _add_scalar_constraint(m, sum(vs[1:], vs[0]) == sum_rhs)
        prod = vs[0]
        for v in vs[1:]:
            prod = prod * v
        # maximize the positive product on the slice (interior optimum)
        m.minimize(-prod)
        return m, vs

    prod_cases = [
        ("x*y | x+y=5", [(1.0, 4.0)] * 2, 5.0),
        ("x*y*z | sum=6", [(1.0, 4.0)] * 3, 6.0),
        ("prod5 NARROW [4,6] | sum=25", [(4.0, 6.0)] * 5, 25.0),
        ("prod5 WIDE [1,10] | sum=25", [(1.0, 10.0)] * 5, 25.0),
    ]
    for name, boxes, rhs in prod_cases:
        m, vs = product_slice(boxes, rhs, "max")
        st, b, fb = root_bound(m)
        # true optimum on the slice: equal split maximizes a positive product
        # for a fixed sum (AM-GM); verify it lies in the box.
        k = len(boxes)
        eq = rhs / k
        assert all(lo <= eq <= hi for lo, hi in boxes)
        true_max = float(eq**k)
        true_opt = -true_max  # objective is -prod
        _record(rows, "B_multilinear", f"max {name}", f"n={k}", true_opt, st, b, fb)

    # -- Group C: monomial / positive product, plain box (vertex optima) ---
    for name, fexpr, exps, lo, hi, sense in [
        ("x**2*y", lambda x, y: x**2 * y, (2, 1), 1.0, 3.0, "min"),
        ("x**2*y", lambda x, y: -(x**2 * y), (2, 1), 1.0, 3.0, "max"),
    ]:
        m = _new_model()
        x = m.continuous("x", lb=lo, ub=hi)
        y = m.continuous("y", lb=lo, ub=hi)
        m.minimize(fexpr(x, y))
        coef = 1.0 if sense == "min" else -1.0
        true_opt = coef * multilinear_corner_opt(exps, [lo, lo], [hi, hi], 1.0, sense)
        st, b, fb = root_bound(m)
        _record(rows, "C_monomial", f"{name} [{sense}]", f"[{lo},{hi}]^2", true_opt, st, b, fb)

    # -- Group D: division ------------------------------------------------
    for name, sense in [("x/y", "min"), ("x/y", "max")]:
        m = _new_model()
        x = m.continuous("x", lb=1.0, ub=4.0)
        y = m.continuous("y", lb=1.0, ub=4.0)
        coef = 1.0 if sense == "min" else -1.0
        m.minimize(coef * (x / y))
        # x/y over [1,4]^2: min 1/4 at (1,4), max 4 at (4,1); both vertices.
        true_opt = 0.25 if sense == "min" else -4.0
        st, b, fb = root_bound(m)
        _record(rows, "D_division", f"{name} [{sense}]", "[1,4]^2", true_opt, st, b, fb)

    # -- Group E: composite compositions (the crux) -----------------------
    comp = [
        ("(log(x-2))**2", lambda x: dm.log(x - 2) ** 2, lambda z: np.log(z - 2) ** 2, 3.0, 9.0),
        (
            "(log(x-2))**2+(log(10-x))**2",  # nvs09 per-variable composite
            lambda x: dm.log(x - 2) ** 2 + dm.log(10 - x) ** 2,
            lambda z: np.log(z - 2) ** 2 + np.log(10 - z) ** 2,
            3.0,
            9.0,
        ),
        (
            "exp(-2*(x-1)**2)",
            lambda x: dm.exp(-2 * (x - 1) ** 2),
            lambda z: np.exp(-2 * (z - 1) ** 2),
            -1.0,
            3.0,
        ),
        ("x**3-3*x", lambda x: x**3 - 3 * x, lambda z: z**3 - 3 * z, -2.0, 2.0),
        ("(x**2-1)**2", lambda x: (x**2 - 1) ** 2, lambda z: (z**2 - 1) ** 2, -2.0, 2.0),
        ("sin(x)**2", lambda x: dm.sin(x) ** 2, lambda z: np.sin(z) ** 2, 0.0, np.pi),
    ]
    for name, fexpr, fpy, lo, hi in comp:
        m = _new_model()
        x = m.continuous("x", lb=lo, ub=hi)
        m.minimize(fexpr(x))
        true_opt = scan_min_1d(fpy, lo, hi)
        st, b, fb = root_bound(m)
        _record(rows, "E_composite", f"min {name}", f"[{lo},{hi}]", true_opt, st, b, fb)

    # -- Group F: AVM decomposition of the nvs09 composite ----------------
    # Introduce explicit auxiliaries w=log(...) linked by equality; the squares
    # become bare-variable squares (exact envelope) and the logs get the exact
    # univariate ln envelope. This is exactly the factorable AVM decomposition
    # SOTA solvers apply; it measures what proper atomization recovers vs the
    # monolithic default (Group E) — WITHOUT any new envelope math.
    log7 = float(np.log(7.0))
    m = _new_model()
    x = m.continuous("x", lb=3.0, ub=9.0)
    w1 = m.continuous("w1", lb=0.0, ub=log7)
    w2 = m.continuous("w2", lb=0.0, ub=log7)
    _add_scalar_constraint(m, w1 == dm.log(x - 2))
    _add_scalar_constraint(m, w2 == dm.log(10 - x))
    m.minimize(w1**2 + w2**2)
    true_opt = scan_min_1d(lambda z: np.log(z - 2) ** 2 + np.log(10 - z) ** 2, 3.0, 9.0)
    st, b, fb = root_bound(m)
    _record(rows, "F_avm", "AVM w=log; min w1^2+w2^2", "[3,9]", true_opt, st, b, fb)

    return rows


# ---------------------------------------------------------------------------
# nvs09 anchor + attribution
# ---------------------------------------------------------------------------
def run_nvs09_anchor() -> list[dict]:
    import discopt.modeling as dm
    from discopt.modeling.core import from_nl

    rows: list[dict] = []
    nvs09_opt = -43.134  # minlplib.solu reference cited in lever-a plan

    # Full instance, default path.
    model = from_nl(str(_NL_DIR / "nvs09.nl"))
    st, b, fb = root_bound(model)
    rows.append(
        {
            "probe": "nvs09 full (default root LP)",
            "reference_opt": nvs09_opt,
            "status": st,
            "bound": b,
            "fell_back": fb,
            "note": "no finite bound; relaxation objective not linearizable",
        }
    )

    # Squares-only sub-objective (drop the product term).
    m = _new_model()
    vs = [m.continuous(f"x{i}", lb=3.0, ub=9.0) for i in range(10)]
    obj = 0.0 * vs[0]
    for v in vs:
        obj = obj + dm.log(v - 2) ** 2 + dm.log(10 - v) ** 2
    m.minimize(obj)
    per_var = scan_min_1d(lambda z: np.log(z - 2) ** 2 + np.log(10 - z) ** 2, 3.0, 9.0)
    st, b, fb = root_bound(m)
    rows.append(
        {
            "probe": "nvs09 squares-only (10 vars)",
            "reference_opt": 10.0 * per_var,
            "status": st,
            "bound": b,
            "fell_back": fb,
            "note": f"per-var true min {per_var:.5f}; separable interval loses all of it",
        }
    )

    # Product term (log-space target).
    m = _new_model()
    vs = [m.continuous(f"x{i}", lb=3.0, ub=9.0) for i in range(10)]
    prod = vs[0]
    for v in vs[1:]:
        prod = prod * v
    m.minimize(prod**0.2)  # min at all=3 -> (3^10)^0.2 = 9
    st, b, fb = root_bound(m)
    rows.append(
        {
            "probe": "nvs09 (prod x)^0.2 term",
            "reference_opt": 9.0,
            "status": st,
            "bound": b,
            "fell_back": fb,
            "note": "fractional power of 10-way product: NO finite bound (blocker)",
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Corpus fallback census (frequency signal for the leverage ranking)
# ---------------------------------------------------------------------------
def run_corpus_census() -> dict:
    from discopt.modeling.core import from_nl

    files = sorted(_NL_DIR.glob("*.nl"))
    fell_back, no_bound, errors = [], [], []
    for f in files:
        name = f.stem
        try:
            model = from_nl(str(f))
            _st, b, fb = root_bound(model)
            if fb:
                fell_back.append(name)
            if b is None:
                no_bound.append(name)
        except Exception as exc:  # noqa: BLE001 - census records unbuildable instances
            errors.append((name, repr(exc)[:80]))
    return {
        "n_instances": len(files),
        "objective_fell_back": fell_back,
        "no_finite_bound": no_bound,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(x, width=12, prec=5):
    if x is None:
        return "None".rjust(width)
    return f"{x:{width}.{prec}f}"


def print_atom_table(rows: list[dict]) -> None:
    print("\n=== Per-atom factorable-envelope tightness (root LP vs true optimum) ===")
    hdr = f"{'atom':<38}{'box':<14}{'true_opt':>12}{'bound':>13}{'abs_gap':>12}{'rel_gap':>10}  fb"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = " UNSOUND" if r["unsound"] else ""
        fbc = "Y" if r["fell_back"] else "."
        print(
            f"{r['atom']:<38}{r['box']:<14}{_fmt(r['true_opt'])}{_fmt(r['bound'], 13)}"
            f"{_fmt(r['abs_gap'])}{_fmt(r['rel_gap'], 10, 4)}  {fbc}{flag}"
        )


def print_nvs09(rows: list[dict]) -> None:
    print("\n=== nvs09 anchor + attribution ===")
    for r in rows:
        fbc = "Y" if r["fell_back"] else "."
        print(
            f"{r['probe']:<34} ref_opt={_fmt(r['reference_opt'], 10)} "
            f"bound={_fmt(r['bound'], 10)} fb={fbc}  {r['note']}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--census", action="store_true", help="also run the 62-file corpus fallback census"
    )
    ap.add_argument("--json", type=str, default=None, help="write full results to this JSON path")
    args = ap.parse_args()

    atom_rows = run_atom_audit()
    print_atom_table(atom_rows)
    nvs09_rows = run_nvs09_anchor()
    print_nvs09(nvs09_rows)

    census = None
    if args.census:
        census = run_corpus_census()
        print("\n=== Corpus objective-fallback census (62 vendored .nl) ===")
        print(f"instances                 : {census['n_instances']}")
        print(f"objective fell back (fb=Y): {len(census['objective_fell_back'])}")
        print(f"  {census['objective_fell_back']}")
        print(f"no finite root bound      : {len(census['no_finite_bound'])}")
        print(f"  {census['no_finite_bound']}")
        if census["errors"]:
            print(f"build errors: {census['errors']}")

    if args.json:
        out = {"atoms": atom_rows, "nvs09": nvs09_rows, "census": census}
        Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
