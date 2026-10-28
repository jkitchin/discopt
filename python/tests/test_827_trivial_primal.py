"""#827 showcase instances solve on the default path, without a trivial-point seed.

#827 added ``DISCOPT_TRIVIAL_PRIMAL``, which passed an obvious feasible point (origin,
box center, bound corners) to the root as ``initial_point``, because ``ball_mk2_30``
(optimum at the origin) and ``chimera_k64ising-01`` (no constraints) returned no
incumbent. Both now certify without it, and the MINLPLib panel of 2026-10-28 found the
seed cost 9 certificates for 0 gained, so the flag was retired (performance-plan
§103.11, data in ``docs/dev/data/827-trivial-primal/``). These tests keep the two
instances that motivated it from regressing on the default path.

``ball_mk2_30`` uses ``nlp_bb=False``: a convex instance is otherwise auto-routed to
NLP-BB, which also finds the origin, so the spatial tree the seed targeted would not
run. ``DISCOPT_CONVEX_KERNEL=0`` for the same reason (#1422: the kernel attempt can
take the whole budget on this instance). Corpus-gated.
"""

from __future__ import annotations

import os
from pathlib import Path

import discopt.modeling as dm
import pytest

BENCH = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(
    not (BENCH / "ball_mk2_30.nl").exists(),
    reason="ball_mk2_30.nl (benchmark corpus) absent",
)
def test_827_ball_mk2_spatial_tree_finds_the_origin(monkeypatch):
    """The spatial tree certifies the origin (objective 0) with no seed."""
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "0")
    r = dm.from_nl(str(BENCH / "ball_mk2_30.nl")).solve(
        time_limit=8.0, deterministic=True, nlp_bb=False
    )
    # #1422: the search must have run, or this asserts nothing about the tree.
    assert r.node_count > 0, "the spatial search never ran"
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.objective is not None and abs(r.objective) < 1e-4, r.objective


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(
    not (BENCH / "chimera_k64ising-01.nl").exists(),
    reason="chimera_k64ising-01.nl (benchmark corpus) absent",
)
def test_827_chimera_certifies_without_seed():
    """#827 family C: 1192 integers, no constraints, MAXIMIZE Ising, optimum 24.3.
    Measured 2026-10-28: feasible 24.3 at 5 s, optimal at 30 s, flag OFF."""
    r = dm.from_nl(str(BENCH / "chimera_k64ising-01.nl")).solve(time_limit=60.0)
    assert r.objective is not None, "no incumbent on chimera_k64ising-01"
    # MAXIMIZE: a feasible incumbent never exceeds the optimum.
    assert r.objective <= 24.3 + 1e-3, r.objective
    assert r.status == "optimal" and abs(r.objective - 24.3) < 1e-3, (
        r.status,
        r.objective,
        r.bound,
    )
