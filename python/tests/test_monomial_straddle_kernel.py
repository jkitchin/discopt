"""A straddling node box must not let the native kernel cut its own feasible points.

``mccormick_patch::monomial_aux_bounds`` took the endpoint min/max of ``x**p``. That
is the range of ``x**p`` only on a **sign-definite** box, but the spatial kernel calls
it on every node box, and branching splits a straddling root on an *interior* point —
so straddling boxes arrive routinely. ``assemble_node_lp`` intersects the returned
range tighten-only into the auxiliary column, so over ``[-10, 10]`` the aux for
``x**2`` was pinned to ``[100, 100]``, cutting off every point with ``|x| < 10``.

Observed before the fix: ``mathopt4`` returned ``status=optimal`` with
``objective == bound == 2.49e-4`` in a single node, on a MINIMIZE model whose optimum
is 0 (the origin is feasible) — a false certificate on a default-ON path.

Separately, odd ``p`` on a straddling box took the *concave* branch of
``monomial_rows``. ``x**p`` is S-shaped there, so neither hull is valid; the rows it
emitted made the node LP INFEASIBLE, which as a node result reads as "fathomed".
"""

from __future__ import annotations

import numpy as np
import pytest

_rust = pytest.importorskip("discopt._rust")

_INF = 1e20
_I64 = np.int64

# Straddling boxes reachable by branching a straddling root, plus sign-definite
# controls that must stay exactly as they were.
_BOXES = [
    (-10.0, 10.0),
    (-10.0, 9.272727272727275),
    (-2.0, 5.0),
    (-0.5, 3.0),
    (-7.0, -1.0),  # control: sign-definite
    (1.0, 7.0),  # control: sign-definite
]
_POWERS = [2, 3, 4, 5, 6]


def _admitted_range(p, lo, hi, pin):
    """Interval the kernel's node LP admits for ``s = x**p`` with ``x`` pinned.

    Two fixed rows pin ``x`` (``x <= pin`` and ``-x <= -pin``) rather than shrinking
    the column bounds, so the root box stays straddling and the kernel regenerates the
    envelope for that box exactly as it does at a node.
    """
    out = []
    for sense in (1.0, -1.0):
        res = _rust.solve_spatial_tree_py(
            n_cols=2,
            n_orig=1,
            c=np.array([0.0, sense]),
            integrality=np.zeros(2, dtype=_I64),
            global_lo=np.array([lo, -_INF]),
            global_hi=np.array([hi, _INF]),
            fixed_row_ptr=np.array([0, 1, 2], dtype=_I64),
            fixed_cols=np.array([0, 0], dtype=_I64),
            fixed_coeffs=np.array([1.0, -1.0]),
            fixed_rhs=np.array([pin, -pin]),
            term_kind=np.array([1], dtype=_I64),
            term_i=np.array([0], dtype=_I64),
            term_j=np.array([-1], dtype=_I64),
            term_out=np.array([1], dtype=_I64),
            term_p=np.array([p], dtype=_I64),
            term_coeff=np.array([0.0]),
            term_cst=np.array([0.0]),
            blf_w=np.zeros(0, dtype=_I64),
            blf_a_ptr=np.zeros(1, dtype=_I64),
            blf_a_cols=np.zeros(0, dtype=_I64),
            blf_a_coeffs=np.zeros(0),
            blf_a_const=np.zeros(0),
            blf_b_ptr=np.zeros(1, dtype=_I64),
            blf_b_cols=np.zeros(0, dtype=_I64),
            blf_b_coeffs=np.zeros(0),
            blf_b_const=np.zeros(0),
            obbt_candidates=np.zeros(0, dtype=_I64),
            max_nodes=200,
            gap_tol=1e-6,
            int_tol=1e-5,
            mccormick_tol=1e-8,
            min_box_width=1e-8,
            # The envelope is what is under test: in-tree FBBT would shrink the box to
            # the pinned point and the probe would measure nothing (CLAUDE.md §6).
            run_obbt=False,
            run_propagation=False,
            propagation_rounds=0,
            initial_incumbent=None,
            time_limit_s=10.0,
            incumbent_time_extension_s=None,
            bound_time_extension_s=None,
            cold_dual_start=False,
        )
        if res["status"] == "infeasible":
            return None
        out.append(sense * res["bound"])
    return out[0], out[1]


@pytest.mark.smoke
def test_regenerated_monomial_envelope_never_cuts_a_point_in_its_box():
    checks = 0
    cutting = []
    for p in _POWERS:
        for lo, hi in _BOXES:
            for pin in sorted({0.0, 0.5 * (lo + hi), lo, hi}):
                if not lo <= pin <= hi:
                    continue
                true_s = pin**p
                rng = _admitted_range(p, lo, hi, pin)
                checks += 1
                if rng is None:
                    cutting.append((p, lo, hi, pin, "INFEASIBLE"))
                    continue
                s_min, s_max = rng
                if s_min > true_s + 1e-6 or s_max < true_s - 1e-6:
                    cutting.append((p, lo, hi, pin, s_min, s_max, true_s))
    # CLAUDE.md §6: prove the probe fired. 62 of these cut before the fix.
    assert checks >= 100, f"probe executed only {checks} comparisons"
    assert not cutting, f"{len(cutting)} boxes cut their own feasible point: {cutting[:5]}"


@pytest.mark.smoke
def test_odd_power_on_a_straddling_root_never_reaches_the_kernel():
    """The producer must not hand the kernel an S-shaped term the spec cannot express.

    It declines because the cold build emits the 2-facet S-hull (2 rows) and the spec
    claims exactly 4 rows per monomial. That is an *incidental* gate, which is why
    ``monomial_rows`` also refuses the case directly: if the S-hull ever grew to 4
    rows the producer would start accepting, and the concave branch it used to fall
    into emits rows that make the node LP infeasible — a silently fathomed optimum.
    """
    from discopt import Model
    from discopt._relax.spatial_producer import build_spatial_kernel_spec

    straddling = Model()
    x = straddling.continuous("x", lb=-2.0, ub=3.0)
    y = straddling.continuous("y", lb=0.0, ub=5.0)
    straddling.subject_to(x**3 + y <= 4.0)
    straddling.minimize(x**3 - y)
    assert build_spatial_kernel_spec(straddling) is None, (
        "odd power on a root box spanning zero must not reach the kernel"
    )

    # Control: the SAME model on a sign-definite root IS accepted, so this asserts a
    # real discrimination and not a producer that declines everything.
    definite = Model()
    xd = definite.continuous("x", lb=0.5, ub=3.0)
    yd = definite.continuous("y", lb=0.0, ub=5.0)
    definite.subject_to(xd**3 + yd <= 4.0)
    definite.minimize(xd**3 - yd)
    assert build_spatial_kernel_spec(definite) is not None, (
        "sign-definite odd-power root must still reach the kernel"
    )
