"""#1296 on the Python tree drivers: a contradicted tree certifies nothing.

``TreeManager`` raises ``bound_contradicted`` when an integral node's own value comes
in below the bound it inherited from its ancestors -- a feasible point that proves an
ancestor bound invalid. The MILP driver withdraws its certificate on it
(``milp_driver.rs``, ``withdraw_certificate``). The Python drivers (spatial B&B,
NLP-BB, MIQP-BB) never read it, and the binding did not expose it.

Measured before the guard, with the tree's first import (the root) given the INVALID
bound ``opt + 1e-3`` on the random convex IQPs of ``test_keep_proved_bound``: NLP-BB
returned ``optimal`` / ``gap_certified=True`` with the contradiction raised (seed 32),
and spatial B&B reached its exit still ``optimal`` on all 34 contradicted cases.
MIQP-BB could not be driven into a witnessed contradiction this way (its heuristics
find the optimum before the root import, so an invalid root bound is pruned at once);
it shares the guard, which is tested directly below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import discopt.solver as S
import numpy as np
import pytest
from discopt._rust import PyTreeManager

sys.path.insert(0, str(Path(__file__).parent))
import test_keep_proved_bound as K  # noqa: E402

_GUARD = S._withdraw_contradicted_tree_certificate


# ── the guard itself (shared by spatial B&B, NLP-BB and MIQP-BB) ─────────────


def test_guard_leaves_an_uncontradicted_result_alone():
    out = _GUARD({"bound_contradicted": False}, "optimal", 1.0, 0.0, True, "bnb_tree", None, "t")
    assert out == ("optimal", 1.0, 0.0, True, "bnb_tree", None)


@pytest.mark.parametrize(
    "status,expected",
    [("optimal", "feasible"), ("feasible", "feasible"), ("infeasible", "unknown")],
)
def test_guard_withdraws_certificate_and_bound(status, expected):
    st, bound, gap, cert, src, stats = _GUARD(
        {"bound_contradicted": True}, status, 1.0, 0.0, True, "bnb_tree", {"k": 2.0}, "t"
    )
    assert (st, bound, gap, cert, src) == (expected, None, None, False, None)
    assert stats == {"k": 2.0, "tree/bound_contradicted": 1.0}


def test_binding_exposes_bound_contradicted():
    tree = PyTreeManager(1, [0.0], [10.0], [0], [1], "best_first")
    tree.initialize()
    assert tree.stats()["bound_contradicted"] is False


# ── end to end: an invalid root bound, witnessed by the optimal leaf ─────────


class _InflateRoot:
    """A tree whose first import (the root) carries ``inflated`` as its bound."""

    def __init__(self, tree, inflated: float):
        self._t, self._inflated, self.imports = tree, inflated, 0

    def import_results(self, ids, lbs, *a, **k):
        self.imports += 1
        if self.imports == 1:
            lbs = np.array(lbs, dtype=np.float64, copy=True)
            lbs[0] = self._inflated
        return self._t.import_results(ids, lbs, *a, **k)

    def __getattr__(self, name):
        return getattr(self._t, name)


def _solve_with_invalid_root(monkeypatch, seed: int, sub: bool, **kw):
    model, opt = K._random_convex_iqp(seed, K._box(seed, sub))
    made = []
    orig = S._tree_manager

    def make(*a, **k):
        made.append(_InflateRoot(orig(*a, **k), opt + 1e-3))  # above the optimum: invalid
        return made[-1]

    monkeypatch.setattr(S, "_tree_manager", make)
    r = model.solve(time_limit=30, deterministic=True, **kw)
    assert len(made) == 1 and made[0].imports >= 2  # the root was inflated, then searched
    assert made[0]._t.stats()["bound_contradicted"]  # rule 6: the witness fired
    return r


@pytest.mark.parametrize(
    "seed,sub,kw",
    [
        pytest.param(
            32,
            False,
            {"nlp_bb": True},
            marks=[pytest.mark.slow, pytest.mark.correctness],  # ~5 s here, ~15 s on CI
        ),
        (38, False, {"incumbent_callback": lambda *a, **k: True}),  # MIQP -> spatial B&B
    ],
    ids=["nlp_bb", "spatial"],
)
def test_contradicted_tree_does_not_certify(monkeypatch, seed, sub, kw):
    r = _solve_with_invalid_root(monkeypatch, seed, sub, **kw)
    assert r.status != "optimal" and not r.gap_certified, (r.status, r.bound)
    assert r.bound is None
    assert (r.solver_stats or {}).get("tree/bound_contradicted") == 1.0
