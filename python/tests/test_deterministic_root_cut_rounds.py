"""Under ``deterministic=True`` NLP-BB's root-cut stage stops on a round budget.

The stage's working terminator is its wall budget, which is role-2 and goes inert
under ``deterministic`` (#1371). Its only other limits were ``ROUNDS`` (a runaway
backstop) and a stall exit that never fires while each round gains a little. Measured
(``nlp_bb=True, deterministic=True``, 20 s): ``nvs07`` ran 169 rounds and ``nvs15``
103, each until the solve's deadline, so NLP-BB returned ``time_limit`` with ONE node
and the #844 fallback answered for it. ``DETERMINISTIC_ROUNDS`` is that budget; see
its comment in ``discopt.solvers._root_cuts`` for the sweep that set it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import discopt.modeling as dm
import discopt.solvers._root_cuts as RC
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_2262_cmir_integer_bounds as T2262  # noqa: E402

NVS07 = Path(__file__).parent / "data" / "minlplib_nl" / "nvs07.nl"


@pytest.mark.parametrize("deterministic", [False, True])
def test_nlpbb_passes_the_round_budget_for_its_mode(monkeypatch, deterministic):
    seen = []

    def stub(*args, **kwargs):
        seen.append(kwargs.get("max_rounds"))
        return RC.RootCutResult()

    monkeypatch.setattr(RC, "generate_root_cuts", stub)
    T2262._issue_model().solve(time_limit=20, nlp_bb=True, deterministic=deterministic)
    assert len(seen) == 1, seen  # rule 6: the convex IQP reaches the stage
    assert seen[0] == (RC.DETERMINISTIC_ROUNDS if deterministic else RC.ROUNDS)


def test_the_deterministic_budget_is_a_working_limit():
    assert 0 < RC.DETERMINISTIC_ROUNDS < RC.ROUNDS


@pytest.mark.slow
@pytest.mark.correctness
def test_deterministic_nvs07_is_certified_by_nlpbb_itself(monkeypatch):
    """Uncapped, the stage ran to the deadline: ``time_limit``, one node."""
    rounds = []
    orig = RC.generate_root_cuts

    def spy(*args, **kwargs):
        r = orig(*args, **kwargs)
        rounds.append((r.rounds_run, r.stop_reason))
        return r

    monkeypatch.setattr(RC, "generate_root_cuts", spy)
    r = dm.from_nl(str(NVS07)).solve(time_limit=20, nlp_bb=True, deterministic=True)
    assert rounds == [(RC.DETERMINISTIC_ROUNDS, "rounds")], rounds  # the budget stopped it
    assert r.status == "optimal" and r.gap_certified, (r.status, r.node_count)
    assert "#844" not in (r.algorithm_route or ""), r.algorithm_route  # NLP-BB answered
    assert r.node_count > 1
