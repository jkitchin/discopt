"""#2001: two defects of the incumbent repair (``feasibility.repair_point``, #1537 E).

1. A homogeneous row at zero (``w - v == 0`` with ``w = v = 0``, an inactive hull
   disjunct) had a noise floor that vanished with the row. The least-squares
   step leaves ``~ulp * ||step||`` of round-off on every column it moves, so the
   row came back at 4e-21, then 1.5e-36, ... 1.8e-130 -- each above
   ``32 ulp * itself`` -- and the repair rejected every iterate of a point it had
   repaired ("no step reduced the violation"). The floor now carries the step's
   own round-off.
2. The minimum-norm step leaves a column that sits a solver tolerance inside an
   active bound free, and spends the correction along it: the published
   objective drifted (cstr nt=5: 2.79188879 against the sub-NLP's 2.79188379).
   A second candidate snaps such columns onto the bound; the lower objective wins.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.validation.feasibility import _noise_floor, repair_point, verify_point


def test_homogeneous_row_at_zero_is_repaired():
    m = dm.Model("hull_disagg_repro")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=10)
    v_act = m.continuous("v_act", lb=0, ub=10)
    v_inact = m.continuous("v_inact", lb=0, ub=10)
    w_inact = m.continuous("w_inact", lb=0, ub=10)
    m.minimize(x)
    m.subject_to(x == 1.0)
    m.subject_to(x - v_act - v_inact == 0.0)
    m.subject_to(w_inact - v_inact == 0.0)
    m.subject_to(v_inact - 10.0 * y <= 0.0)
    m.subject_to(w_inact - 10.0 * y <= 0.0)
    x0 = np.array([0.0, 1.000001, 1.0, 0.0, 0.0])
    ev = make_evaluator(m)
    rep = repair_point(m, x0, evaluator=ev)
    assert rep.x is not None, rep.reason
    assert rep.excess_before > 1e-7
    assert rep.excess_after == 0.0
    v = verify_point(m, rep.x, with_objective=True, evaluator=ev)
    assert v.ok
    assert v.objective == 1.0


def test_step_round_off_floor_is_row_local_and_scales_with_the_row():
    import scipy.sparse as sp

    J = sp.csr_matrix(np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]))
    x = np.array([0.0, 0.0, 1e12])
    r = np.zeros(2)
    rhs = np.array([0.0, 1e12])
    step = np.array([1e-6, 0.0, 0.0])
    base = _noise_floor(J, x, r, rhs)
    with_step = _noise_floor(J, x, r, rhs, step)
    # Row 0 (moved) gains ulp * ||step||; row 1 (not moved) is unchanged, however
    # large an unrelated column is.
    assert base[0] == 0.0 and with_step[0] > 1e-6 * np.finfo(float).eps
    assert with_step[1] == base[1]
    np.testing.assert_allclose(_noise_floor(3.0 * J, x, r, 3.0 * rhs, step), 3.0 * with_step)


def _near_active_bound(sense):
    m = dm.Model("near_bound")
    a = m.continuous("a", lb=0.0, ub=10.0)
    b = m.continuous("b", lb=0.0, ub=10.0)
    m.subject_to(a + b == 10.0)
    if sense == "min":
        m.minimize(b)
    else:
        m.maximize(b)
    # a solver point 1e-5 inside a's active bound, row residual 2e-5
    return m, np.array([10.0 - 1e-5, 3e-5])


def test_near_active_bound_is_snapped_when_it_lowers_the_objective():
    m, x0 = _near_active_bound("min")
    ev = make_evaluator(m)
    rep = repair_point(m, x0, evaluator=ev)
    assert rep.x is not None and rep.excess_after == 0.0
    # min-norm alone: a = b = +-1e-5 -> b = 2e-5. Snapped: a = 10, b = 0.
    np.testing.assert_array_equal(rep.x, [10.0, 0.0])
    assert verify_point(m, rep.x, with_objective=True, evaluator=ev).objective == 0.0


def test_snapped_candidate_is_not_taken_when_it_is_worse():
    m, x0 = _near_active_bound("max")
    ev = make_evaluator(m)
    rep = repair_point(m, x0, evaluator=ev)
    assert rep.x is not None and rep.excess_after == 0.0
    np.testing.assert_allclose(rep.x, [10.0 - 2e-5, 2e-5], rtol=0, atol=1e-15)


# The pre-repair incumbent of cstr nt=5 big-M (#2000 id=6): a terms-tight LP point
# with equality residual 4.5e-6 and x40 3.0e-5 under its upper bound 10.
_CSTR_X0 = [
    1.0,
    1.0,
    -0.0,
    -0.0,
    -0.0,
    -0.0,
    -0.0,
    -0.0,
    -0.0,
    -0.0,
    -0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    -0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0000035527136789e-09,
    0.9900000037918929,
    2.7918607715124963,
    1.0000035527136789e-09,
    0.010000003791892947,
    -3.552713678800501e-14,
    1.0,
    0.9900000151675716,
    0.45004420054802224,
    0.5000490278055164,
    2.7918607525531915,
    0.9900000113756787,
    1.0000035527136789e-09,
    2.7918607601369145,
    0.9900000075837858,
    1.0000035527136789e-09,
    2.791860763928775,
    1.0000035527136789e-09,
    2.7918607677206357,
    0.010000015167571763,
    8.549975440048566,
    9.499971118766295,
    0.010000011375678828,
    1.0000035527136789e-09,
    0.01000000758378589,
    1.0000035527136789e-09,
    1.0000035527136789e-09,
    1.0,
    8.999974147268968,
    9.999969598550791,
    1.0000000149999433,
    -3.552713678800501e-14,
    1.0000000059999716,
    -3.552713678800501e-14,
    -3.552713678800501e-14,
    0.05000482725749417,
    0.45004420054802224,
    0.9499956787177286,
    8.549975440048566,
    0.9999954512818197,
    8.999974147268968,
    2.7918607525531915,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0000017763568413e-09,
    1.0000017763568413e-09,
    -0.33669145665324435,
    1.0000017763568413e-09,
    1.0000017763568413e-09,
    1.0000017763568413e-09,
    0.33669145665324435,
    1.0000017763568413e-09,
    1.0000017763568413e-09,
    1.0000017763568413e-09,
]


def test_cstr_bigm_repair_reaches_the_sub_nlp_objective():
    import os

    from discopt.modeling.gams_parser import parse_gams_file

    path = os.path.join(os.path.dirname(__file__), "data", "gdp_cstr", "cstr_nt5_bigm_0006.gms")
    m = parse_gams_file(path)
    ev = make_evaluator(m)
    rep = repair_point(m, np.array(_CSTR_X0), evaluator=ev)
    assert rep.x is not None and rep.excess_after == 0.0
    v = verify_point(m, rep.x, with_objective=True, evaluator=ev)
    assert v.ok
    # min-norm alone published 2.79188879; the sub-NLP seed is 2.79188379.
    assert v.objective < 2.791884
