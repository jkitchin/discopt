"""#1236: the native spatial kernel must report its root bound and node counters.

Issue #1236 set out to diagnose an integer-heavy node-count tail and found that
one of the three instances in it — ``nvs13``, 637 nodes against SCIP's 27 — takes
the native kernel path, which returned ``root_bound=None``, ``root_gap=None`` and
an EMPTY ``solver_stats``. With only the final ``bound`` visible, "the relaxation
is loose at the root and the tree spent 637 nodes closing it" and "the root was
already tight and the nodes went somewhere else" are the same observation, so the
instance could not be diagnosed at all from Python (the issue's own words:
"this path is currently unmeasurable from Python").

The kernel computed all of it already and dropped it on the floor. Surfacing it
answered the question immediately: nvs13's root bound is about -1234 against an
optimum of -585.2 — a >100% root gap — with ZERO uncertified and ZERO undecided
nodes, so the tree is paying for relaxation looseness, not for a bound plateau
caused by node LPs that could not be certified.

Pure instrumentation: nothing here feeds back into the search, so the node count
and the certified objective are unchanged (CLAUDE.md §6).
"""

from __future__ import annotations

import math
import os

import discopt.solver as solver_mod
from discopt.modeling.core import from_nl

DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

#: Keys the kernel path must now publish on ``SolveResult.solver_stats``.
COUNTER_KEYS = (
    "tree/nodes",
    "tree/lp_solves",
    "tree/uncertified_nodes",
    "tree/undecided_nodes",
)


def _kernel_results(monkeypatch, name, **solve_kwargs):
    """Solve ``name`` and return ``(result, kernel_results)``.

    The second element is what makes the probe auditable: an empty list means the
    kernel never ran and every assertion below would be vacuous.
    """
    seen = []
    original = solver_mod._try_native_spatial_kernel

    def spy(*args, **kwargs):
        res = original(*args, **kwargs)
        if res is not None:
            seen.append(res)
        return res

    monkeypatch.setattr(solver_mod, "_try_native_spatial_kernel", spy)
    result = from_nl(os.path.join(DATA, f"{name}.nl")).solve(**solve_kwargs)
    return result, seen


def test_kernel_reports_root_bound_and_counters(monkeypatch):
    result, seen = _kernel_results(monkeypatch, "nvs13", time_limit=60.0)

    # The probe must actually exercise the kernel (CLAUDE.md §6).
    assert seen, "native kernel never supplied a result -- probe measured nothing"
    assert result.status == "optimal", result.status

    for r in seen:
        assert r.root_bound is not None, "kernel result still carries root_bound=None"
        assert math.isfinite(r.root_bound)
        # Soundness: a root bound is a valid dual bound, so it can never be
        # TIGHTER than the bound the whole tree proved.
        assert r.bound is not None
        assert r.root_bound <= r.bound + 1e-6, (
            f"root bound {r.root_bound} tighter than the final bound {r.bound}"
        )
        # ...and it never crosses the incumbent it is reported against.
        assert r.objective is not None
        assert r.root_bound <= r.objective + 1e-6

        assert r.root_gap is not None and math.isfinite(r.root_gap)
        assert r.root_gap >= 0.0
        assert r.root_time is not None and r.root_time >= 0.0

        stats = r.solver_stats or {}
        missing = [k for k in COUNTER_KEYS if k not in stats]
        assert not missing, f"kernel dropped {missing} (solver_stats={stats})"
        assert stats["tree/nodes"] == float(r.node_count)
        assert stats["tree/lp_solves"] >= 0.0
        assert stats["tree/uncertified_nodes"] >= 0.0
        assert stats["tree/undecided_nodes"] >= 0.0

    # The finding the instrumentation exists to make visible: nvs13's root
    # relaxation is loose (this is a bound-strength instance, unlike the
    # nvs02/nvs14 pair in the same issue, whose root gap is under 1%).
    assert result.root_gap is not None and result.root_gap > 0.5, (
        f"nvs13 root gap {result.root_gap} -- the instance changed; re-derive the "
        "issue #1236 diagnosis rather than relaxing this assertion"
    )


def test_root_bound_matches_a_one_node_search(monkeypatch):
    """The reported root bound is the ROOT's, not a copy of the final bound.

    A search stopped at one node reports exactly the bound the root region proved,
    so the full search's ``root_bound`` must agree with it. Spec-independent: it
    holds whether or not the root relaxation happens to be tight.
    """
    full, seen_full = _kernel_results(monkeypatch, "nvs13", time_limit=60.0)
    one, seen_one = _kernel_results(monkeypatch, "nvs13", time_limit=60.0, max_nodes=1)

    assert seen_full and seen_one, "native kernel never ran -- probe measured nothing"
    assert full.root_bound is not None and one.bound is not None
    assert one.node_count == 1, f"one-node run processed {one.node_count} nodes"
    assert abs(one.bound - full.root_bound) <= 1e-6 * (1.0 + abs(full.root_bound)), (
        f"root bound {full.root_bound} != the one-node search's bound {one.bound}"
    )
