"""#871: a numerically-failed node LP is retried without cuts before poisoning the certificate.

The convex tree only upgrades ``Exhausted -> Optimal`` when every node is accounted
for (``!uncertified_drop``). A node LP exiting ``numerical`` proves nothing, so
poisoning is correct — but premature. The breakdown is caused by the separated pool
itself: on ``clay0303hfsg`` one node reached 307 OA rounds / 2042 tangents with
coefficient dynamism ``7.3e8`` before the factorization failed.

Re-solving that box with ``solve_node`` (K1: OA-only, no integrality separation) is a
*different, weaker* relaxation — every cut dropped can only enlarge the feasible
region — so its optimum stays a valid bound in the sense convention. Weaker, never
tighter, which is the sound direction. That is #871 step 2's retry-with-fewer-cuts,
and it is not a tolerance tweak: no threshold moves and nothing is relaxed.

Measured, ``DISCOPT_CONVEX_KERNEL=1``, oracle from ``minlplib.solu``:

==================  =====================================  =====================================
instance            before                                 after
==================  =====================================  =====================================
clay0303hfsg        ``exhausted``, bound 26668.921579      **``optimal``, 26669.109557**
syn05m              optimal 837.7324009                    unchanged
syn05hfsg           optimal 837.7324009                    unchanged
rsyn0805m04hfsg     optimal 7174.220058                    unchanged
rsyn0810m04hfsg     optimal 6581.935607                    unchanged
cvxnonsep_psig40r   ``exhausted``/``-inf`` at node 1       unchanged (see below)
==================  =====================================  =====================================

**#871 is not fully closed by this.** Its title symptom — an LP that exits ``Optimal``
while its Neumaier–Shcherbina safe bound is ``-inf`` — is a *different* face, and
``cvxnonsep_psig40r`` still shows it at node 1. Retrying that case without cuts was
implemented, measured to be a **no-op** there (its NS decline is not caused by the
separated pool), and removed rather than shipped unmeasured.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from discopt.modeling.core import ObjectiveSense  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")


def _known_optimum(name: str) -> float:
    """Read the optimum from the shared registry — never a transcribed literal."""
    from _optima import known_optimum

    return float(known_optimum(name))


@pytest.fixture
def kernel_on(monkeypatch):
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "1")


@pytest.mark.correctness
@pytest.mark.timeout(900)
def test_clay0303hfsg_certifies_after_the_cut_free_retry(kernel_on, convex_kernel_solve):
    """The instance #871 blocks now certifies, and the certified value is correct.

    Fail-before / pass-after: with the retry removed this returns ``exhausted`` with
    bound 26668.921579 (measured), so the ``optimal`` assertion is what the change
    buys. Not marked ``slow`` — every CI lane excludes ``slow`` (``ci.yml`` 178 / 262 /
    388), and a guard that cannot run is documentation. It carries an explicit
    ``timeout(900)`` because the solve is ~9 s locally but far slower on CI runners;
    it must stay above ``CONVEX_KERNEL_BUDGET_S`` so the solver's own budget, not
    pytest, is what bounds the run.

    The solve comes from the session-scoped ``convex_kernel_solve`` fixture: three
    tests across two files were re-running this identical deterministic tree, which
    is what pushed the correctness lane to 1353 s. See the fixture's docstring in
    ``conftest.py`` for the determinism measurement.
    """
    opt = _known_optimum("clay0303hfsg")
    solved = convex_kernel_solve("clay0303hfsg")
    m, spec, r = solved["model"], solved["spec"], solved["result"]
    assert m._objective.sense == ObjectiveSense.MINIMIZE, "sense assumed below"
    assert spec is not None, "clay0303hfsg must be kernel-eligible"

    inc, bound, status = r["incumbent"], r["bound"], r["status"]
    tol = 1e-4 * max(1.0, abs(opt))

    checks = 0
    assert status == "optimal", f"clay0303hfsg did not certify (status={status})"
    checks += 1
    # MINIMIZE: the dual bound is a LOWER bound, so `bound > opt` is the unsound side.
    assert bound is not None and bound <= opt + tol, f"UNSOUND bound {bound} > {opt}"
    checks += 1
    assert inc is not None and abs(inc - opt) < tol, (
        f"CERTIFIED objective {inc} != known optimum {opt} — a false certificate"
    )
    checks += 1
    assert bound <= inc + tol, "certificate invariant: bound <= incumbent (min sense)"
    checks += 1
    assert checks == 4, "soundness assertions did not all execute"


@pytest.mark.correctness
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "name",
    ["syn05m", "syn05hfsg", "clay0303hfsg"],
)
def test_kernel_certificates_stay_correct(kernel_on, name, convex_kernel_solve):
    """No instance that already certified may regress, and none may certify wrongly.

    Reads the objective SENSE from the model rather than assuming MINIMIZE: `syn*` are
    MAXIMIZE, where a valid dual bound is an UPPER bound and the incumbent lies BELOW
    the optimum. Applying the minimize direction to them reports perfectly sound
    results as violations — a mistake made once already while measuring this issue.

    Instances outside ``known_optima.toml`` or outside the vendored corpus SKIP rather
    than assert: the registry is the only ground truth CI has, and asserting against a
    transcribed literal is exactly how a wrong oracle gets published. The `syn*`
    non-regression is covered by the measurement in this module's docstring and by
    ``test_convex_kernel_perspective_865``.
    """
    path = os.path.join(_DATA, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name}.nl not vendored")
    try:
        opt = _known_optimum(name)
    except KeyError:
        # The registry is the only ground truth available in CI (minlplib.solu is not
        # vendored). Skipping is honest; asserting against a transcribed literal is how
        # a wrong oracle gets published, which happened once already on this issue.
        pytest.skip(f"no recorded optimum for {name} in known_optima.toml")
    solved = convex_kernel_solve(name)
    m, spec, r = solved["model"], solved["spec"], solved["result"]
    maximize = m._objective.sense == ObjectiveSense.MAXIMIZE
    assert spec is not None, f"{name} must be kernel-eligible"

    inc, bound, status = r["incumbent"], r["bound"], r["status"]
    tol = 1e-4 * (1.0 + abs(opt))

    checks = 0
    if bound is not None and abs(bound) != float("inf"):
        bad = (bound < opt - tol) if maximize else (bound > opt + tol)
        assert not bad, f"{name}: UNSOUND bound {bound} vs optimum {opt} (max={maximize})"
        checks += 1
    if inc is not None:
        bad = (inc > opt + tol) if maximize else (inc < opt - tol)
        assert not bad, f"{name}: incumbent {inc} beyond optimum {opt} (max={maximize})"
        checks += 1
    if status == "optimal":
        assert inc is not None and abs(inc - opt) < tol, (
            f"{name}: CERTIFIED {inc} != optimum {opt} — a false certificate"
        )
        checks += 1
    assert checks >= 2, f"{name}: only {checks} assertions fired — probe is vacuous"
