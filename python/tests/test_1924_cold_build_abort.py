"""#1924: the cold relaxation build in ``solve_at_node`` has an effective deadline.

``build_deadline`` truncates only constraint rows, and the node loop passed no
deadline at all, so one build -- inside convexity certification -- outlived a
60 s solve by more than a minute (``chimera_mis-01``, ``t1000``,
``gastrans582_cold13``). ``solve_at_node(abort_deadline=...)`` now discards a
build that passes it, and the curvature certificate polls the deadline between
sub-boxes. The certificate also takes its Hessian over the expression's support
only; its verdicts are unchanged.
"""

import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.canonical_expr import RelaxationBuildAborted
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt._relax.milp_relaxation import _build_convexity_box, _multivar_box_curvature


def _wide_model(n_extra=200):
    m = dm.Model("wide")
    pad = m.continuous("pad", shape=(n_extra,), lb=-1, ub=1)
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    z = m.continuous("z", shape=(3,), lb=0.5, ub=2.0)
    return m, pad, x, y, z


def _box(m):
    lb = np.concatenate([np.asarray(v.lb, dtype=float).ravel() for v in m._variables])
    ub = np.concatenate([np.asarray(v.ub, dtype=float).ravel() for v in m._variables])
    return lb, ub


def _cols(m, *pairs):
    off, out = 0, {}
    for v in m._variables:
        out[id(v)] = off
        off += v.size
    return [out[id(v)] + k for v, k in pairs]


@pytest.mark.unit
@pytest.mark.parametrize(
    "build, support, verdict",
    [
        (lambda x, y, z: x * x + y * y + 0.1 * x * y, "xy", "convex"),
        (lambda x, y, z: -(x**2) - y**2 - x * y * 0.1, "xy", "concave"),
        (lambda x, y, z: x * y - z[1] ** 2, "xyz1", None),
        (lambda x, y, z: z[0] ** 2 + z[2] ** 2 + y * y + 0.2 * y * z[2], "yz02", "convex"),
        # An unbounded walk (log of a range through 0): the abstention sentinel,
        # whose compact form spans a one-index fallback support.
        (lambda x, y, z: dm.log(x - 1.0) * y, "xy", None),
    ],
)
def test_compact_hessian_decides_as_the_dense_one(build, support, verdict, monkeypatch):
    # The dependent columns sit after 200 unrelated ones (some are vector elements),
    # so the support-restricted Hessian is a small block of a large model.
    import discopt._relax.convexity.interval_ad as iad

    m, pad, x, y, z = _wide_model()
    expr = build(x, y, z)
    lb, ub = _box(m)
    pairs = {
        "xy": [(x, 0), (y, 0)],
        "xyz1": [(x, 0), (y, 0), (z, 1)],
        "yz02": [(y, 0), (z, 0), (z, 2)],
    }[support]
    idxs = sorted(_cols(m, *pairs))
    verdicts = []
    for dense in (False, True):
        if dense:
            real = iad.interval_hessian
            monkeypatch.setattr(
                iad, "interval_hessian", lambda e, mod, box=None, compact=False: real(e, mod, box)
            )
        box = _build_convexity_box(m, lb, ub)
        verdicts.append(_multivar_box_curvature(expr, m, idxs, lb, ub, box))
    assert verdicts[0] == verdicts[1] == verdict


@pytest.mark.unit
def test_curvature_certificate_polls_the_abort_deadline():
    m, pad, x, y, z = _wide_model(5)
    lb, ub = _box(m)
    idxs = _cols(m, (x, 0), (y, 0))
    box = _build_convexity_box(m, lb, ub)
    saved = dict(box)
    with pytest.raises(RelaxationBuildAborted):
        _multivar_box_curvature(
            dm.exp(x + 2 * y) + x * x, m, idxs, lb, ub, box, abort_deadline=time.perf_counter() - 1
        )
    assert all(box[v] is saved[v] for v in saved)  # narrowed entries restored


@pytest.mark.unit
def test_solve_at_node_reports_an_aborted_build():
    m = dm.Model("abort")
    x = m.continuous("x", shape=(4,), lb=-1, ub=1)
    m.minimize(dm.exp(x[0] + x[1]) * x[2] + x[3] * x[0])
    relaxer = MccormickLPRelaxer(m)
    lb, ub = _box(m)
    res = relaxer.solve_at_node(lb, ub, abort_deadline=time.perf_counter() - 1)
    assert res.status == "aborted"
    assert res.lower_bound is None
    assert relaxer.aborted_builds == 1
    # Without a deadline the same box builds and solves.
    ok = relaxer.solve_at_node(lb, ub)
    assert ok.status == "optimal" and ok.lower_bound is not None
