"""#1346: ``DISCOPT_CONVEX_KERNEL`` graduates default-ON, and the gate stops
claiming models that belong to the LP/MILP route.

**The default flip.** It was never a failed panel that kept this off. #798 proved
both §5 bars on the convex family and the 66-instance Regime-2 panel came back
cert-clean; #800's close-out then deferred graduation to #807's *SCIP wall
parity*, a bar strictly above what §5 asks (§5 scores ON against OFF, not against
SCIP). #1346 re-ran the gate and acted on it. Panel
(``issue1346_convex_kernel_graduation_panel.py``, in-repo 66-instance corpus,
``time_limit=60``, ``deterministic=True``, arms interleaved within each instance
with the order alternated by index, idle machine) -- gate 1 PASS, gate 2 PASS::

    clay0303hfsg   off  feasible/UNCERTIFIED 29911.20 (12.2% above opt)  90.2 s
                   on   optimal/CERTIFIED    26669.1096   149 nodes      21.2 s
    syn05hfsg      off  optimal  277 nodes  23.8 s -> on  optimal  2 nodes  0.01 s

**The LP/MILP refusal, which is the part with teeth.** A pure LP or MILP passes
every clause of the convexity gate trivially -- linear objective, zero nonlinear
rows -- so before this change the gate claimed all of them. Harmless while the
flag was opt-in; a routing hijack the moment it graduated, because
``Model.solve()`` consults the kernel before the HiGHS LP/MILP route and the Rust
MILP engine. 17 smoke tests failed, nearly all MILP or HiGHS-route tests.

The §5 panel did **not** catch it and could not have: all 66 in-repo instances are
MINLP ``.nl`` files, so the corpus contains no pure LP or MILP at all. A
corpus-wide panel bounds what was looked at, not what was affected. These tests
exist so the refusal cannot regress silently the way its absence did.

**What is deliberately NOT here: a counter-case guard.** A two-stage probe guard
was built for the ``watercontamination0202`` case that
``sota-parity-analysis-2026-07-27.md`` G-C records at 2001 s with no bound. Run
against the real instance, it is refused by this gate's *existing* ``nonlinear
objective`` clause in 3.8 s -- G-C's "convex/MIQP route" is the problem
classifier's route, not this kernel. The guard defended against a threat that
cannot reach here, cost +2.6 s on ``clay0303hfsg`` and bought nothing measurable,
so it was deleted rather than shipped. ``test_the_water_counter_case_is_refused``
pins the finding so the guard is not rebuilt on the same misreading.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt.solvers import _convex_kernel as ck  # noqa: E402


@pytest.fixture
def flag(monkeypatch):
    def _set(name, value):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    return _set


# --------------------------------------------------------------------------- #
# the default flip
# --------------------------------------------------------------------------- #


def test_kernel_defaults_to_on(flag):
    """Graduated by the #1346 panel: cert-clean AND net-positive."""
    flag("DISCOPT_CONVEX_KERNEL", None)
    assert ck.convex_kernel_enabled() is True


@pytest.mark.parametrize(
    "value,expected",
    [("0", False), ("", False), ("false", False), ("False", False), ("1", True), ("on", True)],
)
def test_kernel_opt_out_is_preserved(flag, value, expected):
    """§5 keeps the ``=0`` opt-out and the legacy path intact on graduation."""
    flag("DISCOPT_CONVEX_KERNEL", value)
    assert ck.convex_kernel_enabled() is expected


def test_no_guard_switch_was_left_behind():
    """The probe guard was removed, not merely disabled. A gate whose mechanism is
    gone but whose switch survives is the dead flag CLAUDE.md §3 forbids."""
    assert not hasattr(ck, "convex_kernel_guard_enabled")
    assert not hasattr(ck, "_run_guarded_tree")
    src = (ck.__file__ or "").replace(".pyc", ".py")
    assert "DISCOPT_CONVEX_KERNEL_GUARD" not in open(src).read()


# --------------------------------------------------------------------------- #
# the LP/MILP refusal
# --------------------------------------------------------------------------- #


def _pure_milp() -> dm.Model:
    m = dm.Model("pure_milp")
    x = m.integer("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(3 * x + 2 * y)
    m.subject_to(x + y >= 4)
    return m


def _pure_lp() -> dm.Model:
    m = dm.Model("pure_lp")
    a = m.continuous("a", lb=0.0, ub=10.0)
    b = m.continuous("b", lb=0.0, ub=10.0)
    m.minimize(a + b)
    m.subject_to(a + b >= 2)
    return m


def _convex_minlp() -> dm.Model:
    """Composite-of-affine convex with a genuine nonlinear row: in scope."""
    m = dm.Model("cvx")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.integer("y", lb=0, ub=4)
    z = m.continuous("z", lb=0.0, ub=50.0)
    m.minimize(z + 2 * y)
    m.subject_to((x - 3.0) ** 2 <= z)
    m.subject_to(x + y >= 3)
    return m


@pytest.mark.parametrize("build", [_pure_milp, _pure_lp], ids=["milp", "lp"])
def test_a_model_with_no_nonlinear_row_is_refused(build):
    """The regression the smoke suite caught. An outer-approximation tree with
    nothing to outer-approximate must not preempt the routes built for these."""
    assert ck.build_convex_spec(build()) is None


def test_the_refusal_names_its_reason():
    """A refusal that cannot be told apart from the others is untriageable."""
    with pytest.raises(ck.NotConvexKernel, match="no nonlinear row"):
        ck._build(_pure_milp(), None)


def _continuous_convex_nlp() -> dm.Model:
    """``minimize -x s.t. x^2 + y^2 <= r^2`` — the #1037 circle model. Linear
    objective, one convex nonlinear row, **zero integer variables**."""
    m = dm.Model("circle")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(-x)
    m.subject_to(x**2 + y**2 <= 1.0, name="ball")
    return m


def test_a_continuous_convex_nlp_is_refused():
    """The second half of the same defect, caught by CI rather than by the panel.

    The kernel's ``SolveResult`` carries no duals, so routing a continuous convex
    NLP silently drops ``constraint_duals``/``bound_duals_upper`` from a result that
    used to carry them — the regression #1037 exists to prevent. 8 tests in
    ``test_solver_duals.py`` failed on exactly this model before the refusal.
    """
    with pytest.raises(ck.NotConvexKernel, match="no integer variable"):
        ck._build(_continuous_convex_nlp(), None)
    assert ck.build_convex_spec(_continuous_convex_nlp()) is None


def test_the_continuous_convex_nlp_still_reports_duals(flag):
    """The property the refusal protects, asserted end to end rather than inferred
    from the refusal. With the kernel default-ON this model must still come back
    with real multipliers."""
    flag("DISCOPT_CONVEX_KERNEL", None)  # the graduated default
    res = _continuous_convex_nlp().solve()
    assert res.status == "optimal"
    assert res.constraint_duals is not None, "duals were withheld, not refitted"
    assert res.bound_duals_upper is not None


def test_a_convex_minlp_is_still_claimed():
    """The refusal must be keyed on 'no nonlinear row', not on integrality or size —
    otherwise it would take the graduation's own two wins with it."""
    assert ck.build_convex_spec(_convex_minlp()) is not None


@pytest.mark.parametrize("kernel", ["0", "1"])
def test_a_milp_returns_the_same_answer_either_way(flag, kernel):
    flag("DISCOPT_CONVEX_KERNEL", kernel)
    r = _pure_milp().solve(time_limit=60)
    assert r.objective == pytest.approx(8.0, abs=1e-6)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6, "UNSOUND: bound above incumbent (min)"


@pytest.mark.parametrize("kernel", ["0", "1"])
def test_a_convex_minlp_returns_the_same_answer_either_way(flag, kernel):
    """The kernel is a *route*, not a relaxation: adopted only when it certifies and
    its incumbent verifies against the pristine model (#779), so the flag may not
    move the objective or invert the bound."""
    flag("DISCOPT_CONVEX_KERNEL", kernel)
    r = _convex_minlp().solve(time_limit=60)
    # x=3 makes (x-3)^2 = 0, so z=0; x+y>=3 is then satisfied at y=0. Optimum 0.
    assert r.objective == pytest.approx(0.0, abs=1e-4)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6, "UNSOUND: bound above incumbent (min)"


def test_a_nonconvex_model_is_still_refused(flag):
    flag("DISCOPT_CONVEX_KERNEL", "1")
    m = dm.Model("bilinear")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    k = m.integer("k", lb=0, ub=2)
    m.minimize(x * y + k)
    m.subject_to(x + y + k >= 1)
    assert ck.build_convex_spec(m) is None


# --------------------------------------------------------------------------- #
# the counter-case, pinned as a finding
# --------------------------------------------------------------------------- #


def test_the_water_counter_case_is_refused_by_the_objective_clause():
    """``watercontamination0202`` -- G-C's counter-case -- never reaches this kernel.

    Measured on the real 106,711-variable instance: refused in 3.8 s by the
    *existing* ``nonlinear objective`` clause. The .nl declares ``0 1`` for
    (nonlinear constraints, nonlinear objectives), so the refusal is structural,
    not incidental.

    This test uses the same shape rather than vendoring a 5.8 MB fixture: a
    nonlinear OBJECTIVE with entirely linear constraints. If a future change admits
    nonlinear objectives, this fails — and that change then owes its own evidence
    about the 2001 s route, rather than inheriting #1346's panel.
    """
    m = dm.Model("nl_objective_linear_rows")
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.integer("y", lb=0, ub=4)
    m.minimize(dm.log(x) + y)  # nonlinear objective, linear rows — the water shape
    m.subject_to(x + y >= 2)
    with pytest.raises(ck.NotConvexKernel, match="nonlinear objective"):
        ck._build(m, None)
    assert ck.build_convex_spec(m) is None
