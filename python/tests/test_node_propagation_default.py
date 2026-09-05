"""``node_propagation`` graduates to default-ON in the Rust MILP engine.

Domain propagation at each node had been plumbed through ``lp_bindings.rs`` but
left ``node_propagation=false`` in all three ``#[pyo3(signature=...)]`` blocks,
and Python never passed the keyword — so the pass existed and never ran on any
default solve.

**Graduation evidence** (CLAUDE.md §5 double bar), 38-instance MIPLIB 2017 easy
panel, ``gap_tol=1e-4``, 20 s/instance, 2 replicates, arms interleaved, both arms
on the shipped A0 cut budget so the flag is the only difference (152 comparisons):

*Bar 1 — cert-clean:* 0 bounds above the reference optimum, 0 certification
regressions, 0 objective drift beyond tolerance.

*Bar 2 — net-positive:* solved 20/38 → **21/38** (``neos-3611689-kaihu``
converts feasible → optimal); node-count geomean **0.790**; 23 instances >2 %
fewer nodes against 6 with >2 % more. Largest wins ``flugpl`` 4333 → 323,
``enlight8`` 416389 → 129509, ``neos-3611447-jijia`` 313441 → 100541. Largest
loss ``sp150x300d`` 137336 → 318714, then ``fiber`` 19207 → 23537 — the ``fiber``
regression was predicted in ``docs/dev/milp-competitiveness-plan.md`` before the
run. Load moved 6.3 → 26.8 during the panel, most of it the panel's own parallel
B&B, so per CLAUDE.md §9 the verdict rests on node counts (load-independent), not
on wall time.

The behavioral test below runs with cuts, heuristics, presolve and strong
branching **off**. That is deliberate: it isolates propagation as the only
mechanism that can move the node count, so the test measures the flag rather than
the interaction. The shipped-path evidence is the panel above, not this fixture.
"""

import numpy as np
import pytest
from discopt._rust import solve_milp_csc_py, solve_milp_py

# Big-M uncapacitated lot-sizing: `x_t <= M*y_t` is the implication propagation
# chases (y_t driven to 0 fixes x_t to 0 and cascades through the inventory
# balance). This is the structure behind `flugpl`, the panel's largest win.
_T = 32
_SEED = 1


def _lotsizing():
    rng = np.random.default_rng(_SEED)
    demand = rng.integers(2, 30, size=_T).astype(float)
    setup = rng.integers(40, 120, size=_T).astype(float)
    prod = rng.integers(1, 6, size=_T).astype(float)
    hold = rng.integers(1, 4, size=_T).astype(float)
    big_m = demand.sum()

    nv = 3 * _T
    y = lambda t: t  # noqa: E731 - setup binary
    x = lambda t: _T + t  # noqa: E731 - production
    s = lambda t: 2 * _T + t  # noqa: E731 - end-of-period inventory

    rows, rhs, equality = [], [], []
    for t in range(_T):  # s_{t-1} + x_t - s_t == d_t
        r = np.zeros(nv)
        r[x(t)] = 1.0
        r[s(t)] = -1.0
        if t:
            r[s(t - 1)] = 1.0
        rows.append(r)
        rhs.append(demand[t])
        equality.append(True)
    for t in range(_T):  # x_t - M*y_t <= 0
        r = np.zeros(nv)
        r[x(t)] = 1.0
        r[y(t)] = -big_m
        rows.append(r)
        rhs.append(0.0)
        equality.append(False)

    c = np.concatenate([setup, prod, hold])
    lo = np.zeros(nv)
    up = np.concatenate([np.ones(_T), np.full(2 * _T, big_m)])
    integrality = np.concatenate([np.ones(_T, dtype=np.int64), np.zeros(2 * _T, dtype=np.int64)])

    a = np.asarray(rows)
    m = a.shape[0]
    # Standard form: one slack per row, fixed at 0 for the equalities.
    return dict(
        c=np.concatenate([c, np.zeros(m)]),
        a=np.hstack([a, np.eye(m)]),
        b=np.asarray(rhs),
        lb=np.concatenate([lo, np.zeros(m)]),
        ub=np.concatenate([up, np.where(equality, 0.0, 1e20)]),
        integer_cols=np.concatenate([integrality, np.zeros(m, dtype=np.int64)]),
        n_struct=nv,
    )


# Everything that could also prune is off, so propagation is the only live
# mechanism and the node delta is attributable.
_ISOLATED = dict(
    max_nodes=2_000_000,
    gap_tol=1e-9,
    gmi_cuts=False,
    root_cuts=0,
    cut_rounds=0,
    heuristics=False,
    presolve=False,
    strong_branch=False,
    time_limit_s=60.0,
)


def _solve(**overrides):
    inst = _lotsizing()
    status, _x, obj, _bound, nodes, _iters = solve_milp_py(
        inst["c"],
        inst["a"],
        inst["b"],
        inst["lb"],
        inst["ub"],
        inst["integer_cols"],
        inst["n_struct"],
        0.0,
        **{**_ISOLATED, **overrides},
    )
    return status, obj, nodes


@pytest.mark.smoke
def test_node_propagation_is_live_and_sound():
    """The flag must change the search and must not change the answer."""
    off_status, off_obj, off_nodes = _solve(node_propagation=False)
    on_status, on_obj, on_nodes = _solve(node_propagation=True)

    assert off_status == "optimal", off_status
    assert on_status == "optimal", on_status

    # §6: the discriminator. If propagation were a no-op the two arms would be
    # identical and every other assertion here would be decorative.
    assert on_nodes < off_nodes, (
        f"node_propagation changed nothing: {off_nodes} nodes off vs {on_nodes} on. "
        "Either the pass stopped running or the fixture stopped exercising it."
    )
    # Soundness is the hard half: a propagation bug shows up as a *different*
    # certified optimum, not as a slower solve.
    assert on_obj == pytest.approx(off_obj, rel=1e-9, abs=1e-9), (
        f"node_propagation changed the certified optimum: {off_obj!r} -> {on_obj!r}"
    )


@pytest.mark.smoke
def test_default_solve_gets_propagation():
    """Omitting the keyword must behave exactly like passing ``True``.

    This is the graduation itself. Before the flip the default arm matched the
    ``False`` arm, which is why the pass never ran on a default solve.
    """
    default_status, default_obj, default_nodes = _solve()
    on_status, on_obj, on_nodes = _solve(node_propagation=True)
    _, _, off_nodes = _solve(node_propagation=False)

    assert (default_status, default_nodes) == (on_status, on_nodes), (
        f"default solve took {default_nodes} nodes, explicit node_propagation=True "
        f"took {on_nodes}: the default is not ON"
    )
    assert default_obj == pytest.approx(on_obj, rel=1e-9, abs=1e-9)
    # Guard against the fixture going degenerate and making the check above pass
    # for the wrong reason.
    assert default_nodes < off_nodes, (
        f"fixture no longer discriminates ({off_nodes} off vs {on_nodes} on); "
        "the default-ON assertion above is vacuous until it does"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("entry", [solve_milp_py, solve_milp_csc_py])
def test_both_milp_entry_points_default_to_propagation(entry):
    """Both bindings ship the same default.

    ``lp_bindings.rs`` carries the value in three separate ``pyo3(signature)``
    blocks; flipping some of them is the realistic regression, and a
    behavioral test that happens to route through only one entry point would
    not catch it.
    """
    signature = entry.__text_signature__
    assert signature is not None, f"{entry.__name__} exposes no signature to pin"
    assert "node_propagation=True" in signature, (
        f"{entry.__name__} does not default node_propagation to True: {signature}"
    )
