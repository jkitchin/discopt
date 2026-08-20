"""Issue #954: the NLP-BB path must verify the incumbent it returns.

Background
----------
``_solve_nlp_bb`` was the last of the five solve exit paths with no verification
of the point it returns (#952 closed ``_solve_miqp_bb`` and ``_solve_milp_bb``;
``solve_model`` and ``_try_native_spatial_kernel`` were closed by #779/#789). Its
exit was the same three steps those two had — snap the integers, unpack, return —
with nothing checking the point whose objective is reported as ``objective`` and,
on ``optimal``, as the dual ``bound`` too.

Two things separate this path from #952's:

* its rows are nonlinear, so the arbiter cannot be ``_matrix_solution_feasible``.
  The gate uses the repo's nonlinear convention instead — ``viol <= tol +
  rtol*term_scale``, ``tol=1e-6``, ``rtol=1e-9`` — the same one
  ``_relax.primal_heuristics._check_constraint_feasibility`` applies whenever a
  primal heuristic accepts an incumbent.
* the terminal refine re-solve can REPLACE the reported point after every gate
  the search ran, so the check has to be the last thing before the return.

Entry experiment (CLAUDE.md §4), run before the gate was written, over a 106-item
panel — the 40-seed family below plus every in-repo ``minlplib_nl`` instance,
driven through this path with ``nlp_bb=True`` because the in-repo corpus is mostly
nonconvex and would otherwise dispatch elsewhere. 100 entered ``_solve_nlp_bb``,
69 returned a point, 2638 comparisons against the declared rows and bounds::

    worst raw violation = 9.36e-11  (tls2, row 16)
    worst tolerance that would be needed to admit any returned point = 3.93e-14
    instances a 1e-6 gate would newly refuse = 0

Every solve here passes ``nlp_bb=True`` for the same reason the entry panel did,
and since #1059 it is load-bearing rather than merely explicit: the panel model is
convexity-certified, so a bare ``.solve()`` is now auto-routed to the MIP-NLP
family and never enters ``_solve_nlp_bb`` at all. The probes then measure nothing
and say so -- ``the refine stub never ran`` is exactly that failure, not a
regression in the gate. ``nlp_bb=True`` is the documented way to name this engine,
and the route declines whenever a caller names one.

So there was no live false primal, and the gate this adds refuses nothing that
was being returned — four orders of headroom. The defect is that nothing bounded
the excursion: its size was set by whatever the NLP backend converged to, and the
exit would equally have returned 1e-2.

What is pinned here
-------------------
1. :func:`test_returned_incumbent_is_within_declared_tolerance` — the standing
   watch, measured independently of the solver's own helpers, with an
   executed-comparison count (CLAUDE.md §6).
2. :func:`test_exit_gate_refuses_an_off_row_incumbent` — the gate FIRES. Fails
   before the change (the point came back as ``optimal``).
3. :func:`test_baseline_solve_is_unaffected` — its control: the same model with
   the same refine failure returns normally when the incumbent is not perturbed.
4. :func:`test_snap_acceptance_uses_the_exit_arbiter` — #954 item 3: the integer
   snap at this call site is accepted by the exit's own arbiter, not by the
   helper default ``feas_tol=1e-4`` (100x the declared ``abs=1e-6``), so the guard
   can no longer admit a rounding that the exit then refuses.
5. :func:`test_gate_judges_declared_rows_only` — an appended root cut (#781) is
   not a declared row and must not be able to manufacture a refusal.
6. :func:`test_arbiter_agrees_with_the_authoritative_nonlinear_check` — parity
   with ``primal_heuristics._check_constraint_feasibility`` so the new arbiter
   cannot drift away from the repo's existing one.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver as S

# The repo's declared absolute feasibility tolerance (CLAUDE.md "Key
# Constraints"); the gate is held to this, not to the helper's legacy 1e-4.
DECLARED_ABS_TOL = 1e-6


def _panel_model(seed: int) -> dm.Model:
    """A genuinely nonlinear convex MINLP — the dispatch routes it to NLP-BB.

    ``min Σ exp(0.4 xⱼ) + Σ cᵢyᵢ`` s.t. ``Σ log(1+xⱼ) >= tgt`` (a concave ``>=``
    row, so the feasible set is convex), ``xⱼ <= 4 y_{j mod 2}``, ``y`` binary,
    ``x ∈ [0,4]³``. The log row is active at every optimum, which is what makes an
    excursion off it observable.
    """
    rng = np.random.default_rng(1000 + seed)
    m = dm.Model(f"nlp954_{seed}")
    y = m.binary("y", shape=(2,))
    x = m.continuous("x", shape=(3,), lb=0.0, ub=4.0)
    c = rng.uniform(1.0, 3.0, 2)
    tgt = rng.uniform(0.8, 2.2)
    m.minimize(
        sum(dm.exp(0.4 * x[j]) for j in range(3)) + sum(float(c[i]) * y[i] for i in range(2))
    )
    m.subject_to(sum(dm.log(1.0 + x[j]) for j in range(3)) >= tgt)
    for j in range(3):
        m.subject_to(x[j] <= 4 * y[j % 2])
    return m


def _worst_violation(model: dm.Model, x_dict: dict) -> tuple[float, int]:
    """Max violation of the returned point over every declared row AND bound.

    Measured through ``NLPEvaluator.evaluate_constraints`` against the model as
    declared — deliberately not through the solver's own helpers, so this test
    cannot inherit the defect it checks for. Returns ``(worst, n_compared)``;
    ``n_compared`` is the executed-comparison count (CLAUDE.md §6).
    """
    from discopt._relax.nlp_evaluator import NLPEvaluator

    x = np.concatenate(
        [np.asarray(x_dict[v.name], dtype=np.float64).ravel() for v in model._variables]
    )
    ev = NLPEvaluator(model)
    cl, cu = (np.asarray(b, dtype=np.float64) for b in S._infer_constraint_bounds(model, ev))
    g = np.asarray(ev.evaluate_constraints(x), dtype=np.float64)
    n = min(len(g), len(cl))
    assert n > 0, "no constraint rows evaluated: this probe would measure nothing"
    lo, hi = (np.asarray(b, dtype=np.float64) for b in ev.variable_bounds)
    viols = np.concatenate([g[:n] - cu[:n], cl[:n] - g[:n], x - hi, lo - x])
    return float(np.max(viols)), int(2 * n + 2 * len(x))


@pytest.mark.correctness
@pytest.mark.slow
def test_returned_incumbent_is_within_declared_tolerance():
    """The standing watch: no NLP-BB incumbent leaves outside a declared row/bound.

    This passed before the fix too — the measured excursions are ~1e-15 — because
    the point was never to catch a live false certificate but to bound something
    nothing bounded. :func:`test_exit_gate_refuses_an_off_row_incumbent` is the
    one that fails without the gate.
    """
    seeds = 12
    compared = 0
    checks = 0
    worst = (-np.inf, -1)

    for seed in range(seeds):
        m = _panel_model(seed)
        r = m.solve(time_limit=60, nlp_bb=True)
        assert r.status == "optimal", f"seed {seed}: status {r.status}, panel assumes optimal"
        assert r.nlp_bb, f"seed {seed} did not dispatch to NLP-BB; this panel tests that path"
        assert r.x is not None
        viol, n_checked = _worst_violation(m, r.x)
        compared += 1
        checks += n_checked
        if viol > worst[0]:
            worst = (viol, seed)

    # §6: prove the probe fired rather than skipping every seed.
    assert compared == seeds, f"only {compared}/{seeds} seeds compared"
    assert checks == seeds * (2 * 4 + 2 * 5), f"unexpected comparison count {checks}"
    assert worst[0] <= DECLARED_ABS_TOL, (
        f"seed {worst[1]} returned a point {worst[0]:.6e} outside a declared row/bound, "
        f"beyond the declared abs tolerance {DECLARED_ABS_TOL:.0e}"
    )


def _patch_offrow_tree(monkeypatch, cont_slice, shift):
    """Make the tree hand back an incumbent nudged off the active log row.

    Proxies ``PyTreeManager`` and perturbs only the vector returned by
    ``incumbent()``, so the search itself is untouched and the exit gate is the
    only thing under test. Returns a dict whose ``applied`` flag lets the caller
    prove the perturbation actually happened (CLAUDE.md §6) — without it a passing
    ``pytest.raises`` could be pinning some unrelated error.
    """
    real_tree_cls = S.PyTreeManager
    state = {"applied": False}

    class _OffRowTree:
        def __init__(self, *args, **kwargs):
            self._inner = real_tree_cls(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def incumbent(self):
            inc = self._inner.incumbent()
            if inc is None:
                return None
            sol, obj = inc
            sol = np.asarray(sol, dtype=np.float64).copy()
            block = sol[cont_slice]
            if np.all(np.isfinite(block)) and float(np.max(block)) > 1e-2:
                sol[cont_slice.start + int(np.argmax(block))] += shift
                state["applied"] = True
            return sol, obj

    monkeypatch.setattr(S, "PyTreeManager", _OffRowTree)
    return state


def _disable_terminal_refine(monkeypatch):
    """Make the terminal refine/dual-recovery re-solve unavailable.

    That re-solve is best-effort and guarded (``except Exception`` -> debug log),
    and when it runs it would re-converge a perturbed point back onto the row —
    repairing the very excursion under test. Its failure is the regime the gate
    exists for: with it unavailable, nothing but the gate stands between an
    off-row point and a certified return. Returns a counter proving it fired.
    """
    calls = {"n": 0}

    def _boom(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("refine disabled for test")

    monkeypatch.setattr(S, "_solve_node_nlp_kkt", _boom)
    return calls


@pytest.mark.correctness
def test_exit_gate_refuses_an_off_row_incumbent(monkeypatch):
    """The gate fires: an incumbent off a declared row is refused, not returned.

    ``Σ log(1+xⱼ) >= tgt`` is active at the optimum, so pushing the largest ``x``
    coordinate down by 1e-2 puts the incumbent well outside that row while leaving
    it inside every variable bound and leaving the binaries integral. Before the
    fix this came back as ``optimal`` with that point as both ``objective`` and
    ``bound``; now it raises.
    """
    # Variable order is y (2 binaries) then x (3 continuous).
    state = _patch_offrow_tree(monkeypatch, slice(2, 5), -1e-2)
    refine = _disable_terminal_refine(monkeypatch)

    m = _panel_model(0)
    with pytest.raises(RuntimeError, match="NLP-BB returned an infeasible point labeled"):
        m.solve(time_limit=60, nlp_bb=True)

    assert state["applied"], "the incumbent was never perturbed; the test proved nothing"
    assert refine["n"] > 0, "the refine stub never ran; the perturbation may have been repaired"


@pytest.mark.correctness
def test_baseline_solve_is_unaffected(monkeypatch):
    """Control for the test above: same model, same disabled refine, no perturbation.

    Without this, a passing ``pytest.raises`` above could be pinning a gate that
    fires on everything.
    """
    refine = _disable_terminal_refine(monkeypatch)

    m = _panel_model(0)
    r = m.solve(time_limit=60, nlp_bb=True)

    assert refine["n"] > 0, "the refine stub never ran; this control is not comparable"
    assert r.status == "optimal", f"unperturbed solve returned {r.status}"
    assert r.x is not None
    worst, n_checked = _worst_violation(m, r.x)
    assert n_checked > 0
    assert worst <= DECLARED_ABS_TOL, f"unperturbed incumbent already {worst:.3e} off a row"


def _tight_row_fixture():
    """A model, evaluator and near-integral point whose integer snap breaks a row.

    ``10*y - x == 0`` with ``y`` integer and ``x`` continuous. The point sits
    exactly on that row at ``y = 3 - 5e-6``, which is inside the 1e-5 integrality
    window, so the snap applies and moves ``y`` by 5e-6 — a 5e-5 equality
    residual. That lands deliberately between the two tolerances under test:
    inside the retired ``feas_tol=1e-4`` and outside the declared ``abs=1e-6``.
    """
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = dm.Model("snap954")
    y = m.integer("y", lb=0, ub=5)
    x = m.continuous("x", lb=0.0, ub=100.0)
    m.minimize(x)
    m.subject_to(10.0 * y - x == 0.0)
    ev = NLPEvaluator(m)
    cl, cu = (np.asarray(b, dtype=np.float64) for b in S._infer_constraint_bounds(m, ev))
    # y is 5e-6 shy of 3 (inside the 1e-5 snap window); x sits exactly on the row.
    y_val = 3.0 - 5e-6
    point = np.array([y_val, 10.0 * y_val], dtype=np.float64)
    return m, ev, cl, cu, point


@pytest.mark.correctness
def test_snap_acceptance_uses_the_exit_arbiter():
    """#954 item 3: the snap is judged at 1e-6, not at the helper's legacy 1e-4.

    The retired call passed ``feas_tol`` at its default 1e-4 — 100x the declared
    ``abs=1e-6``. On the fixture the snap moves an equality row by 5e-5: inside
    1e-4 (so the old guard adopted the snapped point) and outside 1e-6 (so the
    exit gate this issue adds would then have refused the very point the guard
    installed). The call site now takes the exit's own arbiter, so the two can no
    longer disagree.
    """
    m, ev, cl, cu, point = _tight_row_fixture()
    int_offsets, int_sizes = [0], [1]

    rounded, _ = S._round_incumbent_integers(point, int_offsets, int_sizes)
    assert rounded[0] == 3.0, "the fixture point was not inside the 1e-5 snap window"
    assert rounded[1] == point[1], "only the integer coordinate may be snapped"

    residual = float(np.abs(ev.evaluate_constraints(rounded))[0])
    assert 1e-6 < residual < 1e-4, f"fixture residual {residual:.3e} does not separate the two"

    # The retired verdict: the helper's default admitted this snap.
    assert S._check_constraint_feasibility(ev, rounded, cl, cu, tol=1e-4), (
        "fixture no longer reproduces the 1e-4 acceptance this item is about"
    )

    # The verdict the call site now takes — the same arbiter the exit enforces.
    box = np.stack([np.array([0.0, 0.0]), np.array([5.0, 100.0])], axis=1)
    excess, where, n_cmp = S._nonlinear_point_excess(ev, rounded, cl, cu, box=box)
    assert n_cmp > 0, "the arbiter compared nothing"
    assert excess > S._NLPBB_EXIT_ABS_TOL, f"{where}: excess {excess:.3e} should exceed the gate"

    # And the unsnapped point — the one the call site keeps instead — is clean.
    excess0, _, n_cmp0 = S._nonlinear_point_excess(ev, point, cl, cu, box=box)
    assert n_cmp0 > 0
    assert excess0 <= S._NLPBB_EXIT_ABS_TOL


@pytest.mark.correctness
def test_gate_judges_declared_rows_only():
    """An appended root cut (#781) must not be able to manufacture a refusal.

    The NLP-BB path appends cut rows to the model *before* compiling the
    evaluator. Those rows are valid for integer-feasible points but deliberately
    tight, and they are not rows the user declared, so the gate is limited to the
    leading declared block.
    """
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = dm.Model("cutrows954")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.subject_to(x >= 1.0)  # declared row, satisfied at x = 2
    m.subject_to(x <= 1.5)  # stands in for an appended cut: violated at x = 2
    ev = NLPEvaluator(m)
    cl, cu = (np.asarray(b, dtype=np.float64) for b in S._infer_constraint_bounds(m, ev))
    point = np.array([2.0])

    all_rows, _, n_all = S._nonlinear_point_excess(ev, point, cl, cu)
    assert n_all == 4, f"expected 2 comparisons per row over 2 rows, got {n_all}"
    assert all_rows > S._NLPBB_EXIT_ABS_TOL, "fixture must violate the trailing row"

    declared_only, _, n_decl = S._nonlinear_point_excess(ev, point, cl, cu, n_rows=1)
    assert n_decl == 2, f"expected 2 comparisons over 1 declared row, got {n_decl}"
    assert declared_only <= S._NLPBB_EXIT_ABS_TOL, "the trailing row leaked into the verdict"


@pytest.mark.correctness
def test_arbiter_agrees_with_the_authoritative_nonlinear_check():
    """Parity with ``primal_heuristics._check_constraint_feasibility``.

    That function is what every primal heuristic on this path already holds an
    incumbent to. The gate must not be a second, differently-tuned opinion of the
    same question, so accept/reject is pinned to agree over a sweep that straddles
    the tolerance — including the term-scaled regime, where a large-magnitude row
    forgives noise an absolute-only test would reject (the prob07 lesson).
    """
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt._relax.primal_heuristics import _check_constraint_feasibility as authoritative

    compared = 0
    for scale in (1.0, 1e3, 1e5):
        m = dm.Model(f"parity954_{scale:g}")
        x = m.continuous("x", lb=0.0, ub=10.0 * scale)
        m.minimize(x)
        m.subject_to(x * x - float(scale) * x >= 0.0)
        m.subject_to(x <= float(scale))
        ev = NLPEvaluator(m)
        cl, cu = (np.asarray(b, dtype=np.float64) for b in S._infer_constraint_bounds(m, ev))
        for delta in (0.0, 1e-9, 1e-7, 1e-6, 1e-5, 1e-3):
            point = np.array([float(scale) + delta])
            mine, _, n_cmp = S._nonlinear_point_excess(ev, point, cl, cu)
            assert n_cmp > 0
            theirs = bool(authoritative(ev, point))
            assert (mine <= S._NLPBB_EXIT_ABS_TOL) == theirs, (
                f"scale {scale:g} delta {delta:g}: gate says "
                f"{mine <= S._NLPBB_EXIT_ABS_TOL} (excess {mine:.3e}), authoritative "
                f"check says {theirs}"
            )
            compared += 1

    assert compared == 18, f"parity sweep ran {compared} comparisons, expected 18"
