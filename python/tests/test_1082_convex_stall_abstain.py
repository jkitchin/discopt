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
rule on the convex path; it is bound-changing, so it is default-OFF pending the
CLAUDE.md §5 differential panel.
"""

from pathlib import Path

import numpy as np
import pytest
from discopt.mo.utils import _flatten_solution
from discopt.modeling.core import from_nl
from discopt.solver import _convex_stall_abstain_enabled, solve_model

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


def _solve_tls2(seed=None, time_limit=180.0):
    path = NL_DIR / "tls2.nl"
    assert path.exists(), f"missing corpus instance {path}"
    model = from_nl(str(path))
    kwargs = {"time_limit": time_limit}
    if seed is not None:
        kwargs["initial_point"] = seed
    return model, solve_model(model, **kwargs)


def _optimal_seed():
    """A genuinely optimal point for tls2, obtained by solving it."""
    model, result = _solve_tls2()
    assert result.status == "optimal", f"unseeded tls2 did not prove optimal: {result.status}"
    seed = np.asarray(_flatten_solution(model, result.x), dtype=np.float64).reshape(-1)
    assert seed.size and np.all(np.isfinite(seed)), "seed is not a usable point"
    return seed


@pytest.mark.slow
@pytest.mark.correctness
class TestSeededSolveKeepsCertificate:
    """Seeding an optimal point must not cost the certificate."""

    def test_seeded_solve_certifies_with_abstention(self, monkeypatch):
        monkeypatch.setenv(ABSTAIN_ENV, "1")
        seed = _optimal_seed()
        _, result = _solve_tls2(seed=seed)

        # The point of the fix: the seeded solve still proves optimality.
        assert result.gap_certified is True, (
            f"seeded solve lost the certificate: bound={result.bound} "
            f"obj={result.objective} status={result.status}"
        )
        assert result.status == "optimal"

        # Soundness (CLAUDE.md §1): the dual bound never crosses the oracle,
        # and never exceeds the incumbent it is certifying.
        assert result.bound <= TLS2_OPT + 1e-6, (
            f"dual bound {result.bound} exceeds the reference optimum {TLS2_OPT}"
        )
        assert result.bound <= result.objective + 1e-6
        assert abs(result.objective - TLS2_OPT) <= 1e-4 * max(1.0, abs(TLS2_OPT))

    def test_legacy_arm_still_loses_it(self, monkeypatch):
        """Pins the defect: with the flag off, the same seed strands the bound.

        This is the "fails before, passes after" half. It asserts the *old*
        behaviour, so if a later change fixes the default path this test fails
        loudly and the flag can be retired rather than silently kept.
        """
        monkeypatch.setenv(ABSTAIN_ENV, "0")
        seed = _optimal_seed()
        _, result = _solve_tls2(seed=seed)

        assert result.gap_certified is False, (
            "the legacy arm now certifies -- #1082 may be fixed on the default "
            "path; retire DISCOPT_CONVEX_STALL_ABSTAIN instead of keeping it"
        )
        # Even uncertified, the reported bound must remain a valid dual bound.
        assert result.bound <= TLS2_OPT + 1e-6


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
