"""#1410: the HiGHS MILP route certified a false optimum on a badly scaled model.

Found by adversarial probing. On the instance below the default route returned
``status="optimal"``, ``gap_certified=True`` and objective ``+0.00200765...`` while the
true optimum -- established by exact enumeration over the integer box, not by a reference
solution -- is ``-0.01513385...``: a certified bound 0.0171 ABOVE the optimum, which is
impossible for a valid relaxation of a minimisation. The in-house Rust backend answers
correctly on the same model, so the defect is in what the route accepts from HiGHS.

Root cause: HiGHS's simplex mis-solves the *LP* on this matrix (``highs-ipm`` and a
row-rescaled copy both answer correctly, so it is matrix scaling), and prunes the true
optimum deep in the tree. The pre-existing #1295 guard could not catch it twice over: it
inspects only columns with an open side, and its threshold missed by a factor of 1.43
(ratio 1.366e-6 against a cap of 9.537e-7). The root-bound cross-check could not either --
it compares the tree bound against the root bound, and here the root bound is perfectly
valid while the tree bound is the wrong one.

The guard that does catch it verifies HiGHS's own root-LP primal/dual pair for box dual
feasibility in unscaled doubles: 7.87e-17 on well-scaled LPs against 1.0 here.
"""

import itertools

import numpy as np
import pytest
from discopt import Model

# The counterexample, verbatim from the probe that found it.
C = [
    0.6866774929690818,
    -0.005713836158671233,
    11740.059150860803,
    0.007721491930584785,
    21.9371989038807,
]
A = [
    [
        10.023552111631327,
        -554709.2273055869,
        -31120.64874492787,
        0.0019316280314185673,
        -731961.352263406,
    ],
    [
        1445.4625401493574,
        21007.7074712139,
        -20275.753039620417,
        -499143.4063638526,
        -0.24860654636723628,
    ],
]
B = [-77796.72865470266, 19089.22626419994]
UB = [3, 4, 3, 2, 4]


def _enumerated_optimum() -> float:
    """The true optimum, by exhaustive enumeration of the integer box (4320 points)."""
    An, bn, cn = np.array(A), np.array(B), np.array(C)
    best = None
    for pt in itertools.product(*[range(u + 1) for u in UB]):
        x = np.array(pt, dtype=float)
        if np.all(An @ x <= bn):
            v = float(cn @ x)
            if best is None or v < best:
                best = v
    assert best is not None, "the enumeration found no feasible point -- probe is broken"
    return best


def _build() -> Model:
    m = Model("i1410")
    xs = [m.integer(f"x{j}", lb=0, ub=UB[j]) for j in range(5)]
    m.minimize(sum(C[j] * xs[j] for j in range(5)))
    for i in range(2):
        m.subject_to(sum(A[i][j] * xs[j] for j in range(5)) <= B[i])
    return m


def test_enumerated_optimum_is_what_the_probe_found() -> None:
    """Pin the oracle, so a later change to the model cannot quietly move the target."""
    assert _enumerated_optimum() == pytest.approx(-0.015133852704100148, abs=1e-15)


def test_no_certified_bound_above_the_true_optimum() -> None:
    """The invariant this issue is about: a certified bound may never exceed the optimum.

    This is the assertion that failed before the fix -- the route reported
    ``optimal``/``gap_certified=True`` with a bound of ``+0.002008``.
    """
    truth = _enumerated_optimum()
    res = _build().solve(time_limit=120)
    checks = 0
    assert res.status != "error", res
    if getattr(res, "gap_certified", False):
        bound = res.bound
        assert bound is not None, "a certified result must carry a bound"
        checks += 1
        assert bound <= truth + 1e-7 * (1.0 + abs(truth)), (
            f"certified bound {bound!r} exceeds the enumerated optimum {truth!r} "
            f"by {bound - truth:.6g} -- a false certificate (#1410)"
        )
        checks += 1
        assert res.objective == pytest.approx(truth, rel=1e-7, abs=1e-9), (
            f"a certified optimum must BE the optimum: got {res.objective!r}, enumerated {truth!r}"
        )
    else:
        # Declining to certify is the sound outcome; the reported bound must still be
        # a valid lower bound, and the incumbent must still be feasible.
        checks += 1
        if res.bound is not None:
            assert res.bound <= truth + 1e-7 * (1.0 + abs(truth)), (
                f"uncertified bound {res.bound!r} still exceeds the optimum {truth!r}"
            )
    assert checks > 0, "MEASURED NOTHING"


def test_rust_backend_is_correct_on_the_same_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-house backend solves it exactly -- so the model is not the problem."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    truth = _enumerated_optimum()
    res = _build().solve(time_limit=120)
    assert res.objective == pytest.approx(truth, rel=1e-9, abs=1e-12), res.objective


def test_box_dual_violation_separates_this_matrix_from_a_well_scaled_one() -> None:
    """The guard signal itself: machine-precision on a sane LP, order 1 on this one."""
    import scipy.sparse as sp
    from discopt.solvers.lp_milp_highs import (
        INF,
        StdForm,
        box_dual_violation,
        dual_violation_tolerance,
        solve_lp_std,
    )

    def slacked(c, a, b, xl, xu) -> StdForm:
        m = a.shape[0]
        mat = sp.hstack([sp.csr_matrix(a), sp.identity(m, format="csr")], format="csc")
        return StdForm.from_arrays(
            np.concatenate([c, np.zeros(m)]),
            mat,
            b,
            np.concatenate([xl, np.zeros(m)]),
            np.concatenate([xu, np.full(m, INF)]),
        )

    checks = 0
    rng = np.random.default_rng(3)
    worst_ok = 0.0
    for _ in range(25):
        n, m = 6, 3  # noqa: F841
        c = rng.normal(size=n)
        a = rng.normal(size=(m, n))
        xl = np.zeros(n)
        xu = rng.integers(1, 5, size=n).astype(float)
        b = a @ (xu / 2) + np.abs(a) @ (xu * 0.3)
        sf = slacked(c, a, b, xl, xu)
        out = solve_lp_std(sf, time_limit=30.0)
        if out.status != "optimal" or out.x is None or out.row_dual is None:
            continue
        v = box_dual_violation(sf, out.x, out.row_dual)
        assert np.isfinite(v), "the violation must never be NaN (infinite bounds)"
        checks += 1
        worst_ok = max(worst_ok, v)
    assert checks >= 10, f"MEASURED NOTHING: only {checks} well-scaled LPs solved"
    sf_bad = slacked(np.array(C), np.array(A), np.array(B), np.zeros(5), np.array(UB, float))
    out_bad = solve_lp_std(sf_bad, time_limit=60.0)
    assert out_bad.x is not None and out_bad.row_dual is not None
    v_bad = box_dual_violation(sf_bad, out_bad.x, out_bad.row_dual)
    checks += 1
    tol = dual_violation_tolerance(sf_bad)
    assert worst_ok <= tol, f"well-scaled LPs must pass the guard: {worst_ok:.3g} > {tol:.3g}"
    assert v_bad > tol, f"the #1410 matrix must fail it: {v_bad:.3g} <= {tol:.3g}"
    assert v_bad > 1e6 * worst_ok, (
        f"separation too small to be a guard: bad={v_bad:.3g} ok={worst_ok:.3g}"
    )
    assert checks > 0, "MEASURED NOTHING"
