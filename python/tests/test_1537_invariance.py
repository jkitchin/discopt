"""#1537 workstream B: certificates must be invariant under a change of variables.

``x = y - c`` and a positive row rescaling leave every model mathematically
identical, so a certified answer before and after must agree (helpers in
``_invariance.py``). The September adversarial rounds never tested this; the first
4-model probe found a P0 on plain linear MILPs (#1536).

Three layers:

* ``test_harness_*`` -- the instrument's own self-test: the rebuilt model agrees
  with the original at sampled points, so a disagreement in the panels below is
  the solver's, never the harness's (CLAUDE.md §6).
* ``test_generated_panel`` -- seeded small models of each family, every PR
  (``correctness``, not ``slow``: the ``python-correctness`` lane).
* ``test_corpus_panel`` -- the in-repo MINLPLib ``.nl`` corpus, ``slow`` +
  ``correctness``: the dispatch-only ``python-correctness-slow`` lane, run before
  a release or when touching the certificate path.

Only FALSE certificates fail a test. A *lost* certificate (the transformed solve
is honest but uncertified) is a performance defect, measured and printed but not
asserted -- e.g. today every row-scaled MILP loses its certificate to the #1295
guard (#1537 workstream C), which is honest.
"""

from __future__ import annotations

import glob
import os

import discopt.modeling as dm
import numpy as np
import pytest
from _invariance import invariance_violation, rescale_rows, translate
from discopt._tape_nlp_evaluator import cached_tape_evaluator
from discopt.validation.feasibility import verify_point

CORPUS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "*.nl")))


# ── generated families ────────────────────────────────────────────────────────


def _lin_rows(m, x, rng, n_rows=3):
    A = rng.integers(-4, 5, size=(n_rows, len(x))).astype(float)
    b = rng.integers(2, 10, size=n_rows).astype(float)
    for r in range(n_rows):
        m.subject_to(sum(A[r, j] * x[j] for j in range(len(x))) <= b[r])
    return rng.integers(-5, 6, size=len(x)).astype(float)


def _mixed(m):
    return [m.integer(f"i{k}", lb=0, ub=4) for k in range(3)] + [
        m.continuous(f"c{k}", lb=0, ub=5) for k in range(2)
    ]


def milp(seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"milp{seed}")
    x = _mixed(m)
    c = _lin_rows(m, x, rng)
    m.minimize(sum(c[j] * x[j] for j in range(5)))
    return m


def bilinear(seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"bilin{seed}")
    x = _mixed(m)
    c = _lin_rows(m, x, rng)
    m.subject_to(x[3] * x[4] <= 6)
    m.minimize(sum(c[j] * x[j] for j in range(5)) + 0.5 * x[0] * x[3] - 0.3 * x[1] * x[4])
    return m


def polynomial(seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"poly{seed}")
    x = _mixed(m)
    c = _lin_rows(m, x, rng)
    m.minimize(sum(c[j] * x[j] for j in range(5)) + 0.2 * x[3] ** 3 - x[4] ** 2)
    return m


def convex_nlp(seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"cvx{seed}")
    x = [m.continuous(f"x{k}", lb=-3, ub=3) for k in range(3)]
    a = rng.uniform(-1, 1, size=3)
    m.subject_to(x[0] + x[1] + x[2] >= 1)
    m.minimize(sum((x[k] - a[k]) ** 2 for k in range(3)) + dm.exp(0.3 * x[0]))
    return m


def maximize_milp(seed):
    m = milp(seed)
    m._objective = type(m._objective)(
        expression=-m._objective.expression, sense=dm.core.ObjectiveSense.MAXIMIZE
    )
    return m


FAMILIES = {
    "milp": milp,
    "maximize_milp": maximize_milp,
    "bilinear": bilinear,
    "polynomial": polynomial,
    "convex_nlp": convex_nlp,
}
TRANSFORMS = {
    "shift1e3": lambda m: translate(m, 1e3, seed=1),
    "shift1e6": lambda m: translate(m, 1e6, seed=2),
    "rows1e-3": lambda m: rescale_rows(m, 1e-3),
    # #1537 B names row scaling in {1e-3, 1e3, 1e6}; 1e3 was missing.
    "rows1e3": lambda m: rescale_rows(m, 1e3),
    "rows1e6": lambda m: rescale_rows(m, 1e6),
}


# ── the instrument's self-test ────────────────────────────────────────────────


def _box_samples(model, rng, k):
    lo = np.concatenate([np.ravel(v.lb) for v in model._variables])
    hi = np.concatenate([np.ravel(v.ub) for v in model._variables])
    lo, hi = np.maximum(lo, -10.0), np.minimum(hi, 10.0)
    hi = np.maximum(hi, lo)
    is_int = np.concatenate(
        [np.full(v.size, v.var_type is not dm.core.VarType.CONTINUOUS) for v in model._variables]
    )
    for _ in range(k):
        x = rng.uniform(lo, hi)
        x[is_int] = np.round(x[is_int])
        yield np.clip(x, lo, hi)


def _harness_models():
    for name, fam in FAMILIES.items():
        yield name, fam(0)
    for f in CORPUS[::6]:  # a spread of the corpus; the full set is the slow panel's job
        yield os.path.basename(f), dm.from_nl(f)


def _evaluator(model):
    ev = cached_tape_evaluator(model)
    assert ev is not None, f"{model.name}: no tape evaluator -- the self-test cannot evaluate it"
    return ev


@pytest.mark.parametrize(
    "name, model", list(_harness_models()), ids=lambda v: v if isinstance(v, str) else ""
)
def test_harness_preserves_the_model(name, model):
    """At sampled box points (feasible or not):  f_new(x + c) == f(x) and
    g_new(x + c) == g(x) under translation;  f unchanged and g_new == s * g under
    row scaling;  and ``verify_point``'s verdict is unchanged by translation."""
    rng = np.random.default_rng(0)
    shifted, scaled, s = translate(model, 1e3, seed=3), rescale_rows(model, 1e3), 1e3
    ev, ev_sh, ev_sc = _evaluator(model), _evaluator(shifted), _evaluator(scaled)
    compared = 0
    for x in _box_samples(model, rng, 8):
        f, g = ev.evaluate_objective(x), np.asarray(ev.evaluate_constraints(x))
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            continue  # outside the evaluator's domain (e.g. log of a negative)
        xs = x + shifted._invariance_shift
        ftol = 1e-7 * (1.0 + abs(f))
        gtol = 1e-7 * (1.0 + np.abs(g))
        assert ev_sh.evaluate_objective(xs) == pytest.approx(f, abs=ftol), name
        assert ev_sc.evaluate_objective(x) == pytest.approx(f, abs=ftol), name
        g_sh = np.asarray(ev_sh.evaluate_constraints(xs))
        g_sc = np.asarray(ev_sc.evaluate_constraints(x))
        assert np.all(np.abs(g_sh - g) <= gtol), (name, np.max(np.abs(g_sh - g)))
        assert np.all(np.abs(g_sc - s * g) <= s * gtol), (name, np.max(np.abs(g_sc - s * g)))
        assert verify_point(shifted, xs).ok == verify_point(model, x).ok, name
        compared += 1
    assert compared > 0, f"{name}: no sampled point in the evaluator's domain -- measured nothing"


def test_harness_refuses_rather_than_guesses():
    m = dm.Model("gdp")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 1], [x >= 9]])
    m.minimize(x)
    with pytest.raises(ValueError, match="cannot rebuild"):
        translate(m, 1e3)
    with pytest.raises(ValueError, match="positive"):
        rescale_rows(milp(0), -1.0)


# ── panels ────────────────────────────────────────────────────────────────────


class KnownFalseCertificate(AssertionError):
    """A transformed solve contradicted its certified base: the failure a
    ``KNOWN_FALSE`` / ``KNOWN_CORPUS`` xfail is allowed to absorb. Any OTHER
    failure (a base that stopped certifying, a cell that compared nothing) is a
    plain ``AssertionError`` and is NOT absorbed, so a marker cannot outlive its
    bug by failing for the wrong reason (#1546 review)."""


def _run_panel(cases, transforms, time_limit, *, require_certified_base=False):
    """Solve each case and its transforms; return ``(certified, lost, violations)``.

    ``certified`` counts comparisons whose transformed solve was CERTIFIED and
    checked against a certified base -- the number of things actually measured,
    not the number of solves (#1546 review: a cell where every certificate is
    lost used to pass while comparing nothing). Every transformed result, certified
    or not, goes through :func:`invariance_violation` (bound and incumbent too).
    """
    certified, lost, violations = 0, [], []
    for label, model in cases:
        base = model.solve(time_limit=time_limit)
        if not base.gap_certified:
            if require_certified_base:
                # A known cell exists to exercise its bug, so a base that stops
                # certifying for any REASON is a hard failure -- except running out
                # of wall clock on a loaded runner, which says nothing about the bug
                # and must not fail an xfail cell for timing (#1546 review). Keyed on
                # wall time, not status: a timed-out solve holding an incumbent
                # reports "feasible", not "time_limit" (measured on tls2).
                if base.wall_time >= 0.95 * time_limit:
                    pytest.skip(
                        f"{label}: base stopped at the {time_limit:g} s wall-time limit "
                        f"({base.status}, {base.wall_time:.1f} s)"
                    )
                raise AssertionError(f"{label}: base no longer certifies ({base.status})")
            continue  # nothing certified to compare against
        for tname, tf in transforms.items():
            transformed = tf(model)
            other = transformed.solve(time_limit=time_limit)
            why = invariance_violation(base, other, model, transformed)
            if why:
                violations.append((label, tname, why))
            elif other.gap_certified:
                certified += 1
            else:
                lost.append((label, tname, other.status))
    return certified, lost, violations


def _xfail(reason: str, raises):
    return pytest.mark.xfail(strict=True, raises=raises, reason=reason)


#: Known false results, each tied to the issue tracking it. ``strict`` + a pinned
#: ``raises``: a fix flips the xfail to a failure, and a failure for any other
#: reason is not absorbed.
KNOWN_FALSE: dict = {
    # ("polynomial", "shift1e6") was here until #1542 fixed the separable
    # objective floor's float expansion of the shifted cubic.
}


def _panel_params():
    for fam in FAMILIES:
        for tname in TRANSFORMS:
            known = KNOWN_FALSE.get((fam, tname))
            marks = [_xfail(*known)] if known else []
            yield pytest.param(fam, tname, marks=marks, id=f"{fam}-{tname}")


@pytest.mark.correctness
@pytest.mark.parametrize("family, transform", list(_panel_params()))
def test_generated_panel(family, transform):
    known = (family, transform) in KNOWN_FALSE
    cases = [(f"{family}[{s}]", FAMILIES[family](s)) for s in range(4)]
    certified, lost, violations = _run_panel(
        cases, {transform: TRANSFORMS[transform]}, time_limit=10, require_certified_base=known
    )
    print(
        f"\n{family} x {transform}: certified={certified} "
        f"violations={len(violations)} lost={len(lost)}"
    )
    for row in lost:
        print("  lost:", row)
    if violations:
        raise KnownFalseCertificate(violations)
    assert certified >= 1, (
        f"{family} x {transform}: no transformed solve certified, so nothing was "
        f"compared (lost={lost})"
    )


TRANSFORM_GROUPS = {
    "shift": {k: v for k, v in TRANSFORMS.items() if k.startswith("shift")},
    "rows": {k: v for k, v in TRANSFORMS.items() if k.startswith("rows")},
}

#: Known corpus failures, per (instance, transform group) so the group that still
#: passes keeps guarding the instance: ``(reason, the exception the failure raises)``.
#: Re-measured on main after #1548 / #1549 -- see the PR for the run.
KNOWN_CORPUS: dict = {
    # The three #1537 E cells -- ("ex14_1_9.nl", "rows"), ("syn05hfsg.nl", "shift"),
    # ("ex1225.nl", "shift") -- were here: each published an incumbent whose row
    # residual (the solver's working accuracy) passed verify_point's allowance in
    # the transformed units and failed it in the original's. Fixed by repairing the
    # published incumbent to float noise (feasibility.repair_point) and re-judging
    # its certificate on the repaired objective.
}


def _corpus_params():
    for path in CORPUS:
        for group in TRANSFORM_GROUPS:
            name = os.path.basename(path)
            known = KNOWN_CORPUS.get((name, group))
            marks = [_xfail(*known)] if known else []
            yield pytest.param(path, group, marks=marks, id=f"{name}-{group}")


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.parametrize("path, group", list(_corpus_params()))
def test_corpus_panel(path, group):
    name = os.path.basename(path)
    known = (name, group) in KNOWN_CORPUS
    cases = [(name, dm.from_nl(path))]
    certified, lost, violations = _run_panel(
        cases, TRANSFORM_GROUPS[group], time_limit=20, require_certified_base=known
    )
    for row in lost:
        print("  lost:", row)
    if violations:
        raise KnownFalseCertificate(violations)
    if certified == 0 and not lost:
        # The base did not certify within 20 s: nothing to compare. A KNOWN entry
        # never gets here: _run_panel skips it on a wall-time miss, fails otherwise.
        pytest.skip("base solve not certified within 20 s: no certificate to compare against")
