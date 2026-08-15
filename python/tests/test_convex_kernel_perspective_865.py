"""Issue #865 — perspective terms in the convex-kernel producer.

The hull-reformulated (``*hfsg``) family writes its disjunctive nonlinearities as
the perspective ``s·f(a/s)`` with ``s = 0.001 + 0.999·y``. Syntactically that is a
product of two non-constant subexpressions, so the pre-#865 gate rejected it as a
"bilinear product"; mathematically the perspective of a convex ``f`` is *jointly
convex* on ``s > 0``, so admitting it recognises convexity the gate missed rather
than loosening anything.

The tests below pin, in order of what actually protects the certificate:

1. **exactness** — the marshaled row equals the pristine model's row pointwise
   (the lift ``s·h(·/s) → affine + perspective`` is an algebraic identity, not an
   approximation);
2. **convexity** — every routed row satisfies the midpoint inequality, so its OA
   tangent is a valid relaxation;
3. **soundness gates** — a scale not provably positive on the box, a
   wrong-curvature perspective, and a genuine bilinear product all still fall back;
4. **no drift** — models routed before #865 marshal byte-identically (their terms
   carry an all-zero scale, i.e. the plain composite form).
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
from _optima import known_optimum

_ck = pytest.importorskip("discopt.solvers._convex_kernel")
build_convex_spec = _ck.build_convex_spec
solve_convex_tree = _ck.solve_convex_tree

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")
# BARON-confirmed optimum from minlplib.solu (=opt=); syn05hfsg is MAXIMIZE, so the
# dual bound is an UPPER bound and `bound >= opt` is the correct-side invariant
# (see test_issue759_syn05hfsg_bound_sense.py).
_SYN05HFSG_OPT = 837.7324009

_FUNC_NP = {
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log1p": np.log1p,
    "sqr": np.square,
}


def _scalar(m, expr_fn, name):
    m.constraint(dm.RangeSet(1), lambda _i: expr_fn(), name=name, fast=False)


def _row_value(d, x):
    """g(x) for a `_Decomp`, mirroring the Rust kernel's term semantics exactly."""
    v = d.const + sum(k * x[c] for c, k in d.aff.items())
    for t in d.terms:
        a = t["arg_const"] + sum(k * x[c] for c, k in t["arg_aff"].items())
        f = _FUNC_NP[t["func"]]
        if t["sc_aff"] is None:
            v += t["coeff"] * f(a)
        else:
            s = t["sc_const"] + sum(k * x[c] for c, k in t["sc_aff"].items())
            v += t["coeff"] * s * f(a / s)
    return float(v)


def _nl_decomps(model):
    """Re-run the producer's row loop: [(row_index, sign, _Decomp)] for nonlinear rows."""
    from discopt._relax.gdp_reformulate import reformulate_gdp
    from discopt._relax.model_utils import flat_variable_bounds
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = reformulate_gdp(model, method="big-m")
    lb, ub = flat_variable_bounds(m)
    lb, ub = lb.astype(float), ub.astype(float)
    ev = NLPEvaluator(m)
    n = len(lb)
    rng = np.random.default_rng(0)
    lo = np.where(np.isfinite(lb), lb, 0.0)
    hi = np.where(np.isfinite(ub), ub, lo + 5.0)
    xa = lo + rng.random(n) * (hi - lo)
    xb = lo + rng.random(n) * (hi - lo)
    lin = np.all(np.isclose(ev.evaluate_jacobian(xa), ev.evaluate_jacobian(xb), atol=1e-9), axis=1)
    offsets = _ck._flat_offsets(m)
    rows = []
    for i, con in enumerate(m._constraints):
        if lin[i]:
            continue
        sense = con.sense if isinstance(con.sense, str) else con.sense.value
        d = _ck._decompose(_ck._constraint_expr(m, i), offsets)
        sign = -1.0 if sense == ">=" else 1.0
        if sign < 0:
            d.scale(-1.0)
        rows.append((i, sign, d))
    return m, lb, ub, ev, rows


def _box_sample(lb, ub, rng, n_pts):
    lo = np.where(np.isfinite(lb), lb, -10.0)
    hi = np.where(np.isfinite(ub), ub, lo + 20.0)
    return lo + rng.random((n_pts, len(lb))) * (hi - lo)


# ── the real-corpus instance the issue is about ───────────────────────────────


def test_syn05hfsg_is_routed_after_865():
    """The `*hfsg` perspective family reaches the kernel (it did not before #865)."""
    m = dm.from_nl(os.path.join(_DATA, "syn05hfsg.nl"))
    spec = build_convex_spec(m)
    assert spec is not None, "syn05hfsg's smoothed perspective rows must be routed"
    # Its three nonlinear rows are perspectives, i.e. they carry a nonzero scale.
    assert int(np.count_nonzero(spec["term_scale_const"])) == 3


def test_syn05hfsg_certifies_the_true_optimum():
    """Certified objective == the BARON optimum, and the bound is on the sound side."""
    m = dm.from_nl(os.path.join(_DATA, "syn05hfsg.nl"))
    spec = build_convex_spec(m)
    assert spec is not None
    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=120.0)
    assert r["status"] == "optimal"
    inc, bound = r["incumbent"], r["bound"]
    tol = 1e-4 * max(1.0, abs(_SYN05HFSG_OPT))
    assert abs(inc - _SYN05HFSG_OPT) < tol, f"incumbent {inc} != {_SYN05HFSG_OPT}"
    # MAXIMIZE: the dual bound is an UPPER bound — it must never fall below the
    # true optimum (that would be a too-tight, unsound bound).
    assert bound >= _SYN05HFSG_OPT - tol, f"unsound dual bound {bound} < {_SYN05HFSG_OPT}"
    assert bound >= inc - tol, "certificate invariant: bound ≥ incumbent (max sense)"


@pytest.mark.parametrize(
    ("instance", "func", "n_pts"),
    [("syn05hfsg", "log", 120), ("clay0303hfsg", "sqr", 40)],
)
def test_perspective_lift_is_exact_and_convex(instance, func, n_pts):
    """The two properties the certificate rests on, checked over the box.

    Exactness: the marshaled row must equal the PRISTINE model's row pointwise —
    any drift would mean the lift is an approximation, not an identity.
    Convexity: the midpoint inequality must hold on every routed row, else the OA
    tangent is not a valid relaxation.

    Both properties held for `clay0303hfsg` while #879's false certificate was
    live, so passing here is necessary but NOT sufficient to admit a term class —
    see `test_clay0303hfsg_certifies_against_its_known_optimum`, which is the check
    that was missing.
    """
    model = dm.from_nl(os.path.join(_DATA, f"{instance}.nl"))
    assert build_convex_spec(model) is not None
    m, lb, ub, ev, rows = _nl_decomps(model)
    assert any(t["sc_aff"] is not None for _i, _s, d in rows for t in d.terms)
    assert {t["func"] for _i, _s, d in rows for t in d.terms} == {func}

    # Count the assertions that ACTUALLY execute. The finiteness guards below
    # would otherwise let this test pass having compared nothing (CLAUDE.md
    # "Measurement & instrumentation discipline" rule 6) — and this is the test
    # the certificate's soundness argument rests on.
    rng = np.random.default_rng(12345)
    n_exact = n_convex = 0
    X = _box_sample(lb, ub, rng, n_pts)
    for x in X:
        g = np.asarray(ev.evaluate_constraints(x), float)
        for i, sign, d in rows:
            ref, got = sign * g[i], _row_value(d, x)
            if np.isfinite(ref) and np.isfinite(got):
                n_exact += 1
                assert abs(ref - got) <= 1e-9 * max(1.0, abs(ref)), (
                    f"row {i}: marshaled {got} != model {ref}"
                )

    A, B = _box_sample(lb, ub, rng, n_pts), _box_sample(lb, ub, rng, n_pts)
    for a, b in zip(A, B):
        for lam in (0.25, 0.5, 0.75):
            mid = lam * a + (1 - lam) * b
            for i, _s, d in rows:
                gm, ga, gb = _row_value(d, mid), _row_value(d, a), _row_value(d, b)
                if not all(np.isfinite(v) for v in (gm, ga, gb)):
                    continue
                n_convex += 1
                assert gm - (lam * ga + (1 - lam) * gb) <= 1e-9 * max(1.0, abs(gm)), (
                    f"row {i} is not convex at lambda={lam}"
                )
    assert n_exact >= n_pts, f"exactness compared only {n_exact} rows — probe did not fire"
    assert n_convex >= n_pts, f"convexity compared only {n_convex} rows — probe did not fire"


# ── the quadratic inner function (#879) ───────────────────────────────────────


def test_square_inner_function_is_routed():
    """`** 2` is admitted: `x²` is convex, and its perspective `s·(a/s)² = a²/s` is
    quadratic-over-linear — the `clay*hfsg` hull shape (#879)."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    z = m.binary("z")
    _scalar(m, lambda: x**2 <= 4.0, "sq")
    m.maximize(x + z)
    spec = build_convex_spec(m)
    assert spec is not None, "a plain convex `x**2 <= c` row must be routable"
    assert list(spec["term_func"]) == [_ck._FUNC_CODE["sqr"]]

    clay = dm.from_nl(os.path.join(_DATA, "clay0303hfsg.nl"))
    clay_spec = build_convex_spec(clay)
    assert clay_spec is not None, "clay0303hfsg's quadratic perspective must be routed"
    # Every one of its 72 nonlinear terms is a `sqr` PERSPECTIVE (nonzero scale).
    assert set(clay_spec["term_func"]) == {_ck._FUNC_CODE["sqr"]}
    assert int(np.count_nonzero(clay_spec["term_scale_const"])) == len(clay_spec["term_coeff"])


def test_only_exponent_two_is_admitted():
    """Every other power is nonconvex, domain-restricted, or signomial — refuse."""
    refused = 0
    for power in (1.5, 3.0, -1.0, 0.5, 2.5):
        m = dm.Model()
        x = m.continuous("x", lb=1.0, ub=10.0)
        _scalar(m, lambda p=power: x**p <= 4.0, "pw")
        m.maximize(x)
        assert build_convex_spec(m) is None, f"power {power} must fall back"
        refused += 1
    # A non-affine base is refused too: `(log x)²` is not convex in x.
    m = dm.Model()
    x = m.continuous("x", lb=1.0, ub=10.0)
    _scalar(m, lambda: dm.log(x) ** 2 <= 4.0, "logsq")
    m.maximize(x)
    assert build_convex_spec(m) is None, "a non-affine power base must fall back"
    refused += 1
    assert refused == 6, "every refusal branch must have been exercised"


@pytest.mark.correctness
@pytest.mark.timeout(900)
def test_clay0303hfsg_is_sound_and_any_certificate_is_correct(convex_kernel_solve):
    """THE check whose absence let the #879 false certificate ship.

    Exactness and convexity of the marshaled rows both PASSED while the kernel
    reported `optimal` at 28351.42 / 36397.83 / 55092.52 — three mutually
    inconsistent values, none of them the optimum. They were incumbents, published as
    certified because the tree had silently discarded a `numerical` subtree (#871). A
    term class is only admitted once a routed instance's result is checked against a
    known optimum, so that is asserted here against the shared registry.

    **Runs in the correctness lane, not behind ``slow``.** Every CI lane excludes
    ``slow`` (``ci.yml`` 178 / 262 / 388), so as a slow test this guard would never
    execute on a PR — reproducing the exact #879 gap it exists to close.

    It does need an explicit ``timeout(900)``: the solve is ~9 s on an M-series laptop
    but **263 s** on the CI runner, which blew the lane's 120 s default. That 40x
    spread is why "it only takes 9 s" is not a safe basis for a marker decision. The
    marker must also stay above ``CONVEX_KERNEL_BUDGET_S`` so the solver's budget, not
    pytest, is what bounds the run.

    ONE solve, both facts. Asserting soundness and conditionally checking the
    certificate costs a single tree; an earlier split into two tests re-solved the same
    instance and added 251 s for nothing. That was a within-file lesson; the same
    duplication existed *across* files (two more tests in ``test_871_cut_free_retry``
    re-solved this instance), so the tree now comes from the session-scoped
    ``convex_kernel_solve`` fixture in ``conftest.py``, which records the determinism
    measurement that makes sharing information-neutral.

    Measured today: ``status='exhausted'``, incumbent 26669.109572 (the known optimum
    to 7 figures), bound 26668.921579, relative gap 7.0e-06 — inside the usual 1e-4,
    so the last step of the certificate rather than a structural failure. The
    ``Exhausted -> Optimal`` upgrade requires ``!uncertified_drop``, so a node LP is
    still exiting ``numerical`` and the certificate is CORRECTLY withheld (#871
    residual). Do NOT relax that condition to force a certificate — that manufactures
    exactly the #879 defect. If this instance starts certifying, the ``optimal`` branch
    below tightens automatically to the full #879 check.
    """
    opt = known_optimum("clay0303hfsg")
    solved = convex_kernel_solve("clay0303hfsg")
    spec, r = solved["spec"], solved["result"]
    assert spec is not None
    inc, bound, status = r["incumbent"], r["bound"], r["status"]
    tol = 1e-4 * max(1.0, abs(opt))

    checks = 0
    # MINIMIZE: the dual bound is a LOWER bound, so `bound > opt` is the unsound side
    # — the invariant #879 was believed to have broken.
    assert bound is not None, "no dual bound at all — nothing to check"
    assert bound <= opt + tol, f"UNSOUND dual bound {bound} > known optimum {opt}"
    checks += 1
    assert inc is not None, "no incumbent — this guard would be vacuous"
    assert inc >= opt - tol, f"incumbent {inc} is BELOW the known optimum {opt}"
    checks += 1
    assert bound <= inc + tol, "certificate invariant: bound <= incumbent (min sense)"
    checks += 1

    # The #879 check proper: a CERTIFIED objective must be the known optimum. Today
    # this branch does not fire (status is `exhausted`); it arms itself the moment the
    # instance starts certifying, which is when it matters.
    assert status in ("exhausted", "optimal"), f"unexpected status {status!r}"
    if status == "optimal":
        assert abs(inc - opt) < tol, (
            f"CERTIFIED objective {inc} != known optimum {opt} — a false certificate"
        )
        checks += 1
    assert checks >= 3, "soundness assertions did not all execute"


@pytest.mark.slow
def test_clay0303hfsg_root_relaxation_is_sound_not_too_tight():
    """The #879 hypothesis, falsified and pinned.

    #879 read the false certificate as an invalid (too-tight) `a²/s` relaxation,
    the tangent coefficients scaling as `a²/s² ≈ 1e6` at the `0.001` smoothing
    floor. The root node's safe bound is measured here at every separation setting:
    it is not too tight at any of them — it is *trivially weak* (0.0 against an
    optimum of 26669), which is the opposite failure. Kept so a future regression
    that genuinely does over-tighten the root is caught at the node, not the tree.
    """
    import discopt._rust as _rust

    opt = known_optimum("clay0303hfsg")
    spec = build_convex_spec(dm.from_nl(os.path.join(_DATA, "clay0303hfsg.nl")))
    assert spec is not None
    compared = 0
    for sep in (0, 2, 12):
        r = dict(_rust.solve_convex_node_py(**spec, max_sep_rounds=sep))
        compared += 1
        assert r["bound"] <= opt + 1e-6, f"sep={sep}: root safe bound {r['bound']} > {opt}"
        assert r["raw_bound"] <= opt + 1e-6, f"sep={sep}: raw root bound {r['raw_bound']} > {opt}"
    assert compared == 3, "the root bound must have been compared at every setting"


# ── soundness gates ───────────────────────────────────────────────────────────


def _perspective_model(*, scale_lb: float, sense_ge: bool = False):
    """`s·log(u/s + 1)` with `s = scale_lb + (1-scale_lb)·y`, y binary.

    With ``scale_lb > 0`` the perspective is convex and the ``≤`` row is routable;
    the caller varies ``scale_lb`` (positivity of the scale) and the row sense
    (curvature) to exercise each gate.
    """
    m = dm.Model()
    u = m.continuous("u", lb=0.0, ub=10.0)
    y = m.binary("y")
    w = m.continuous("w", lb=0.0, ub=10.0)

    def body():
        s = scale_lb + (1.0 - scale_lb) * y
        expr = (w / s - dm.log(u / s + 1.0)) * s
        return expr >= 0.0 if sense_ge else expr <= 0.0

    _scalar(m, body, "persp")
    _scalar(m, lambda: u + w <= 8.0, "lin")
    m.maximize(u + w + y)
    return m


def test_positive_scale_perspective_is_routed():
    assert build_convex_spec(_perspective_model(scale_lb=0.001)) is not None


def test_scale_not_provably_positive_falls_back():
    """`s = 0 + 1·y` touches 0 on the box → the perspective is not convex there →
    the gate must refuse rather than emit an invalid tangent."""
    assert build_convex_spec(_perspective_model(scale_lb=0.0)) is None


def test_wrong_curvature_perspective_falls_back():
    """The same row as `>=`: its ≤-normal form flips every sign, making the
    perspective term concave → not routable."""
    assert build_convex_spec(_perspective_model(scale_lb=0.001, sense_ge=True)) is None


def test_genuine_bilinear_still_falls_back():
    """A real var*var product has no perspective structure — the #865 path must not
    admit it by mistake."""
    m = dm.Model()
    a = m.continuous("a", lb=0.5, ub=5.0)
    b = m.continuous("b", lb=0.5, ub=5.0)
    z = m.binary("z")
    _scalar(m, lambda: a * b <= 4.0, "bilin")
    m.maximize(a + b + z)
    assert build_convex_spec(m) is None


def test_scaled_product_without_matching_denominator_falls_back():
    """`(w/s2 - log(u/s2 + 1)) * s1` with s1 != s2 is NOT a perspective."""
    m = dm.Model()
    u = m.continuous("u", lb=0.0, ub=10.0)
    w = m.continuous("w", lb=0.0, ub=10.0)
    y = m.binary("y")
    z = m.binary("z")
    s1 = 0.001 + 0.999 * y
    s2 = 0.001 + 0.999 * z
    _scalar(m, lambda: (w / s2 - dm.log(u / s2 + 1.0)) * s1 <= 0.0, "notpersp")
    m.maximize(u + w + y + z)
    assert build_convex_spec(m) is None


# ── no drift on models routed before #865 ─────────────────────────────────────


def test_plain_composite_terms_carry_a_zero_scale():
    """A pre-#865 routable model marshals unchanged: every term is the plain form,
    flagged by an empty scale CSR row with a zero constant."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    k = m.integer("k", lb=0, ub=3)
    _scalar(m, lambda: k - x <= 0, "kx")
    _scalar(m, lambda: dm.exp(x) <= 5.0, "expc")
    m.maximize(x + k)

    spec = build_convex_spec(m)
    assert spec is not None
    n_terms = len(spec["term_coeff"])
    assert n_terms == 1
    assert np.all(spec["term_scale_const"] == 0.0)
    assert len(spec["term_scale_cols"]) == 0
    assert list(spec["term_scale_ptr"]) == [0] * (n_terms + 1)

    # ...and it still certifies the analytic optimum ln(5) + 1.
    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=30.0)
    assert r["status"] == "optimal"
    truth = float(np.log(5.0)) + 1.0
    assert abs(r["incumbent"] - truth) < 1e-3
    assert r["bound"] >= truth - 1e-6


def test_model_solve_routes_syn05hfsg_when_flag_on(monkeypatch):
    """End-to-end: with the kernel flag on, `Model.solve()` returns the certified
    optimum, incumbent-verified against the pristine model (the #779 guard)."""
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "1")
    m = dm.from_nl(os.path.join(_DATA, "syn05hfsg.nl"))
    r = m.solve(time_limit=120, gap_tolerance=1e-4)
    tol = 1e-4 * max(1.0, abs(_SYN05HFSG_OPT))
    assert r.objective is not None
    assert abs(r.objective - _SYN05HFSG_OPT) < tol, f"objective {r.objective}"
    assert r.bound >= _SYN05HFSG_OPT - tol, f"unsound dual bound {r.bound}"
