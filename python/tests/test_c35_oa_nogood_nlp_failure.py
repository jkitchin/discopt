"""C-35 regression: OA/LOA must NOT emit an unconditional no-good cut on a
*non-rigorous* NLP failure.

A no-good (integer-exclusion) cut permanently removes an integer configuration.
It is only sound when that configuration is *rigorously* proved
infeasible/dominated. The fixed-integer NLP subproblem in the OA/LOA paths is a
*local* solver that never returns a rigorous infeasibility verdict (see
``nlp_ipopt._IPOPT_STATUS_MAP`` — ``Infeasible_Problem_Detected`` maps to
``ERROR``, precisely because IPOPT sees only *local* infeasibility). Treating
"the NLP failed to converge" as "this configuration is infeasible" can:

  * exclude the optimal configuration -> a **false optimal** at a worse config, or
  * exclude every feasible configuration -> a **false infeasible**.

These tests drive the exact false-certificate class by monkeypatching the NLP
subproblem to fail non-rigorously (as a diverged / iteration-limited local NLP
would) and assert the OA/LOA path NEVER returns a *certified* optimal/infeasible
that excluded a feasible/optimal integer assignment. They call ``solve_oa`` /
``solve_gdpopt_loa`` directly (sub-second, no full B&B) and use the default
``milp_solver="auto"`` (in-house Rust simplex), so they run in CI without highspy.

Before the fix these assertions FAIL (status "infeasible"/"optimal", cert=True);
after the fix the path abstains (uncertified) and never excludes the config.
"""

import discopt.modeling as dm
import discopt.solvers.gdpopt_loa as loa_mod
import discopt.solvers.oa as oa_mod
import pytest

pytestmark = pytest.mark.smoke


def _feasible_binary_model():
    """min (x-0.5)^2 + c*y,  x in [0,1], y in {0,1}, x + y >= 0.3.

    Feasible for both y=0 (x>=0.3) and y=1. With c>0 the true optimum is
    y=0, x=0.5, obj=0. A model that is unambiguously feasible.
    """
    m = dm.Model("c35")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.binary("y")
    m.minimize((x - 0.5) ** 2 + 3.0 * y)
    m.subject_to(x + y >= 0.3)
    return m


def test_oa_no_false_infeasible_on_nonrigorous_nlp_failure(monkeypatch):
    """Every NLP fails non-rigorously -> OA must NOT certify infeasible.

    The model is feasible. If the NLP relaxation and every fixed-integer NLP
    subproblem merely *fail* (diverge), OA previously emitted no-good cuts that
    excluded every configuration and returned a certified ``infeasible``. That
    is a false infeasible.
    """
    # Simulate a global non-rigorous NLP failure (no rigorous infeasibility).
    monkeypatch.setattr(oa_mod, "_solve_nlp_relaxation", lambda *a, **k: (None, None))
    monkeypatch.setattr(oa_mod, "_solve_nlp_subproblem", lambda *a, **k: (None, None))

    result = oa_mod.solve_oa(_feasible_binary_model(), time_limit=30, max_iterations=50)

    # The model is feasible: OA must never claim CERTIFIED infeasibility here.
    assert result.status != "infeasible", (
        "OA reported a false 'infeasible' on a feasible model whose NLPs merely "
        "failed non-rigorously (no-good cut excluded feasible configurations)"
    )
    # Whatever it reports, it must not be certified.
    assert result.gap_certified is False


def test_oa_no_false_optimal_when_optimal_config_nlp_fails(monkeypatch):
    """Only the optimal config's NLP fails -> OA must NOT certify a worse optimum.

    y=0 is optimal (obj 0); y=1 is feasible but suboptimal (obj ~3). If the NLP
    for the optimal config y=0 fails non-rigorously, OA previously excluded y=0
    with a no-good cut and certified the suboptimal y=1 as ``optimal`` -> a false
    optimal. After the fix it must not certify optimality at the worse value.
    """
    orig_sub = oa_mod._solve_nlp_subproblem

    def failing_sub(evaluator, lb, ub, int_indices, x_master, nlp_solver):
        y_val = int(round(x_master[int_indices[0]]))
        if y_val == 0:  # optimal configuration's NLP "diverges"
            return None, None
        return orig_sub(evaluator, lb, ub, int_indices, x_master, nlp_solver)

    # No early integer-feasible incumbent from the relaxation.
    monkeypatch.setattr(oa_mod, "_solve_nlp_relaxation", lambda *a, **k: (None, None))
    monkeypatch.setattr(oa_mod, "_solve_nlp_subproblem", failing_sub)

    result = oa_mod.solve_oa(_feasible_binary_model(), time_limit=30, max_iterations=50)

    # Must not certify OPTIMAL at the suboptimal objective (~3) by excluding y=0.
    if result.status == "optimal":
        assert result.objective is not None and result.objective < 1.0, (
            "OA certified 'optimal' at a suboptimal objective after excluding the "
            f"true-optimal config y=0 via a no-good cut (obj={result.objective})"
        )
    # An uncertified/abstaining outcome is the correct behavior here.
    assert not (result.status == "optimal" and result.gap_certified)


def test_oa_certifies_when_infeasible_config_is_rigorous():
    """All-integer model: an infeasible config IS rigorously provable.

    ``max x+y s.t. x^2+y^2 <= 10``, x,y integers. The config (0,4) has
    x^2+y^2=16 > 10 — infeasible — but with all variables fixed the constraint
    check is a *complete*, rigorous feasibility test, so the no-good cut is
    valid and OA may (and must be able to) certify optimality at the true max=4.
    This locks the rigorous branch: the C-35 fix must NOT over-abstain when the
    exclusion is genuinely provable.
    """
    m = dm.Model("oa_int_rigorous")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.maximize(x + y)
    m.subject_to(x**2 + y**2 <= 10)
    result = oa_mod.solve_oa(m, time_limit=30, max_iterations=50)
    assert result.status == "optimal", (
        "OA over-abstained on an all-integer model where the infeasible "
        "configuration is rigorously provable (certification lost)"
    )
    assert result.objective == pytest.approx(4.0, abs=1e-3)
    assert result.gap_certified is True


def _feasible_binary_loa_model():
    m = dm.Model("c35_loa")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.binary("y")
    m.minimize((x - 0.5) ** 2 + 3.0 * y)
    m.subject_to(x + y >= 0.3)
    return m


def test_loa_no_false_infeasible_on_nonrigorous_nlp_failure(monkeypatch):
    """LOA path: same class — a non-rigorous NLP failure must not certify infeasible."""
    monkeypatch.setattr(loa_mod, "_solve_nlp_relaxation", lambda *a, **k: None)
    monkeypatch.setattr(loa_mod, "_solve_nlp_subproblem", lambda *a, **k: None)

    result = loa_mod.solve_gdpopt_loa(
        _feasible_binary_loa_model(), time_limit=30, max_iterations=50
    )

    assert result.status != "infeasible", (
        "LOA reported a false 'infeasible' on a feasible model whose NLPs merely "
        "failed non-rigorously"
    )
    assert result.gap_certified is False
