"""#1082: a stalled convex node must abstain from its bound, not be excluded.

On the convex B&B path a node whose NLP returns ``ITERATION_LIMIT`` with a
constraint-violating iterate was imported at the ``1e30`` infeasibility
sentinel -- i.e. **excluded**. The Rust tree prunes an excluded node by
``node_lb >= incumbent_value`` without any proof, which is exactly why that arm
also had to decertify the whole solve.

The observable cost is worse than a lost certificate: handing the solver a
*proven optimal* point makes it stop early. On ``tls2`` (reference optimum 5.3)
the unseeded solve proves ``optimal`` at 255 nodes, while the same solve seeded
with its own optimum returns ``feasible`` with a dual bound of 1.03.

``solve_model``'s spatial path already does the precise thing: abstain
(``nlp_lb = -inf``). ``import_results`` floors an imported bound at the node's
inherited parent bound -- valid, since a child box is a subset of its parent --
and marks the node ``bound_trusted=False``, so it is *branched*, never fathomed
and never promoted to the incumbent. The one case that proves nothing is an
untrusted node the tree had to fathom with no branch direction left, reported as
``bound_unresolved`` (#598/#467). ``DISCOPT_CONVEX_STALL_ABSTAIN=1`` adopts that
rule on the convex path. It graduated default-ON through the CLAUDE.md §5
panel; ``=0`` keeps the legacy arm.
"""

from pathlib import Path

import numpy as np
import pytest
from discopt.modeling.core import from_nl
from discopt.solver import (
    _convex_stall_abstain_enabled,
    _gap_values_converged,
    solve_model,
)

NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"
ABSTAIN_ENV = "DISCOPT_CONVEX_STALL_ABSTAIN"

# minlplib.solu: ``=opt=  tls2  5.3000000000``
TLS2_OPT = 5.3


@pytest.mark.unit
class TestFlagDefault:
    """The flag is ON since the §5 panel, and reads the documented spellings.

    The opt-out arms are the load-bearing half now: §5 graduates a flag by
    flipping the default while keeping the legacy path intact, so a regression
    that made ``=0`` a no-op would remove the escape hatch the policy requires
    -- invisibly, since the ON arm would keep passing.
    """

    def test_default_on(self, monkeypatch):
        monkeypatch.delenv(ABSTAIN_ENV, raising=False)
        assert _convex_stall_abstain_enabled() is True

    def test_empty_value_does_not_switch_off_the_graduated_default(self, monkeypatch):
        """#993's rule: an empty env value must not switch off a graduated default.

        ``os.environ.get(name, "")`` cannot distinguish unset from set-empty, so
        a graduated flag that treats ``""`` as false silently reverts for anyone
        whose shell exports the variable empty -- which is how #993 shipped.
        """
        monkeypatch.setenv(ABSTAIN_ENV, "")
        assert _convex_stall_abstain_enabled() is True

    def test_zero_is_off(self, monkeypatch):
        monkeypatch.setenv(ABSTAIN_ENV, "0")
        assert _convex_stall_abstain_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
    def test_truthy_spellings_are_on(self, monkeypatch, value):
        monkeypatch.setenv(ABSTAIN_ENV, value)
        assert _convex_stall_abstain_enabled() is True


def _solve_tls2(time_limit=180.0):
    path = NL_DIR / "tls2.nl"
    assert path.exists(), f"missing corpus instance {path}"
    model = from_nl(str(path))
    return model, solve_model(model, time_limit=time_limit)


class _InjectedStall:
    """Make exactly one convex node NLP return a violating ``ITERATION_LIMIT``.

    #1270: tls2 no longer stalls on its own (0 iteration limits seeded or not,
    and 0 across a 66-instance sweep of the in-repo corpus), so a canary that
    waits for a natural stall measures nothing. This replays the #1082 shape
    deterministically: the ``k``-th convex node solve that would have returned
    ``OPTIMAL`` instead reports ``ITERATION_LIMIT`` at a point of its box that
    violates the constraints -- the input both arms branch on.
    ``injected`` is the proof that it fired (CLAUDE.md §6).
    """

    def __init__(self, monkeypatch, k, corners):
        import discopt.solver as solver_mod
        from discopt.solvers import NLPResult, SolveStatus

        self.calls = 0
        self.injected = 0
        real = solver_mod._solve_node_nlp

        def stalled(evaluator, x0, node_lb, node_ub, constraint_bounds, options, **kw):
            r = real(evaluator, x0, node_lb, node_ub, constraint_bounds, options, **kw)
            self.calls += 1
            if (
                kw.get("convex")
                and not self.injected
                and self.calls >= k
                and r.status == SolveStatus.OPTIMAL
            ):
                cl = [c[0] for c in constraint_bounds]
                cu = [c[1] for c in constraint_bounds]
                lo = np.clip(node_lb, -1e3, 1e3)
                hi = np.clip(node_ub, -1e3, 1e3)
                points = {"lo": lo, "hi": hi, "mid": 0.5 * (lo + hi)}
                for name in corners:
                    x = points[name]
                    if not solver_mod._check_constraint_feasibility(evaluator, x, cl, cu):
                        self.injected += 1
                        return NLPResult(
                            status=SolveStatus.ITERATION_LIMIT, x=x, objective=r.objective
                        )
            return r

        monkeypatch.setattr(solver_mod, "_solve_node_nlp", stalled)


def _assert_sound(result):
    """CLAUDE.md §1: the bound never crosses the oracle or the incumbent."""
    assert result.bound <= TLS2_OPT + 1e-6, (
        f"dual bound {result.bound} exceeds the reference optimum {TLS2_OPT}"
    )
    assert result.bound <= result.objective + 1e-6
    assert abs(result.objective - TLS2_OPT) <= 1e-4 * max(1.0, abs(TLS2_OPT))


def _solve_with_stall(monkeypatch, arm, k, corners):
    monkeypatch.setenv(ABSTAIN_ENV, arm)
    stall = _InjectedStall(monkeypatch, k=k, corners=corners)
    _, result = _solve_tls2()
    assert stall.injected == 1, f"the stall never fired ({stall.calls} node solves)"
    return result


@pytest.mark.slow
@pytest.mark.correctness
class TestStalledNodeKeepsCertificate:
    """A stalled convex node must not cost the certificate (#1082).

    Until #1270 this was driven by seeding tls2 with its own optimum, which
    used to make a node stall. It no longer does, so the legacy-arm canary
    passed for a reason unrelated to the flag. The stall is now injected; the
    midpoint of the node box is the violating point, so the node stays
    branchable (its integer coordinates are fractional).
    """

    @pytest.mark.parametrize("k", [1, 20])
    def test_abstention_certifies(self, monkeypatch, k):
        result = _solve_with_stall(monkeypatch, "1", k, ("mid",))
        assert result.gap_certified is True, (
            f"stalled node cost the certificate: bound={result.bound} "
            f"obj={result.objective} status={result.status}"
        )
        assert result.status == "optimal"
        _assert_sound(result)
        assert _gap_values_converged(result.objective, result.bound, 1e-4, 1e-6)

    @pytest.mark.parametrize("k", [1, 20])
    def test_legacy_arm_still_loses_it(self, monkeypatch, k):
        """Pins the defect the flag fixes: the legacy arm excludes the node.

        This is the "fails before, passes after" half. If a later change makes
        the ``=0`` arm certify through a stall, this fails and the flag can be
        retired rather than silently kept.
        """
        result = _solve_with_stall(monkeypatch, "0", k, ("mid",))
        assert result.gap_certified is False, (
            "the legacy arm now certifies through a stalled node -- #1082 may be "
            "fixed on the default path; retire DISCOPT_CONVEX_STALL_ABSTAIN"
        )
        assert result.status == "feasible"
        _assert_sound(result)


@pytest.mark.slow
@pytest.mark.correctness
def test_unbranchable_stall_does_not_certify_an_open_gap(monkeypatch):
    """#1270: an abstaining node with no branch direction floors the tree.

    A violating box *corner* is integral, so the tree fathoms the untrusted
    node and seeds ``unresolved_floor`` at its inherited bound (~2.81). The tree
    then finishes, and ``is_finished()`` alone used to certify ``optimal`` at
    5.3 over that floor.
    """
    result = _solve_with_stall(monkeypatch, "1", 20, ("lo", "hi", "mid"))
    _assert_sound(result)
    if result.status == "optimal" or result.gap_certified:
        assert _gap_values_converged(result.objective, result.bound, 1e-4, 1e-6), (
            f"certified {result.status} with an open gap: obj={result.objective} "
            f"bound={result.bound}"
        )


@pytest.mark.slow
@pytest.mark.correctness
def test_unseeded_solve_is_unchanged_by_the_flag(monkeypatch):
    """No node stalls without the seed, so the flag must be inert there."""
    monkeypatch.setenv(ABSTAIN_ENV, "0")
    _, off = _solve_tls2()
    monkeypatch.setenv(ABSTAIN_ENV, "1")
    _, on = _solve_tls2()

    assert off.status == on.status == "optimal"
    assert off.gap_certified is on.gap_certified is True
    # Bound-neutral where the flag does not fire: node count exactly unchanged.
    assert off.node_count == on.node_count, (
        f"flag perturbed a solve with no stalled node: {off.node_count} -> {on.node_count}"
    )
    assert on.bound == pytest.approx(off.bound, abs=1e-9)
