"""The vectorised NN emitter must build the same model, and certify (#1215).

`FullSpaceFormulation` emitted one `Constraint` per unit; it now emits one
array-valued body per layer. Two things have to hold, and only the first is a
diff:

1. the EXPORTED model is unchanged -- including row names, which only LP/MPS/GAMS
   carry;
2. the SOLVE is unchanged -- which the export diff does not show, because the
   solve path reads the arena and not the `.nl`.

The second is not hypothetical. Writing the layer as `W.T @ prev_z + b` -- the
obvious formulation -- produces a byte-identical `.nl` and turns the 1x1x1
sigmoid net below from `optimal` in 1 node into `feasible` in 121, with an
identical incumbent: `MatMulExpression` relaxes more weakly than the equivalent
expanded sum. The emitter uses `dm.sum(W.T * prev_z, axis=1)` for that reason,
and `test_the_matmul_form_would_lose_the_certificate` pins the difference so the
formulation cannot drift back.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.export import to_lp, to_nl
from discopt.modeling import Model
from discopt.nn import Activation, DenseLayer, NetworkDefinition, OffsetScaling
from discopt.nn.formulations.full_space import FullSpaceFormulation, _family_name


def _net(sizes, act="tanh"):
    rng = np.random.default_rng(7)
    layers = [
        DenseLayer(
            weights=rng.normal(0, 0.3, (sizes[i], sizes[i + 1])),
            biases=rng.normal(0, 0.1, sizes[i + 1]),
            activation=act if i < len(sizes) - 2 else "linear",
        )
        for i in range(len(sizes) - 1)
    ]
    bounds = (np.full(sizes[0], -1.0), np.full(sizes[0], 1.0))
    return NetworkDefinition(layers=layers, input_bounds=bounds)


def _embed(sizes, act, scaled):
    m = Model("nn")
    x = m.continuous("x", shape=(sizes[0],), lb=-1.0, ub=1.0)
    sc = None
    if scaled:
        sc = OffsetScaling(
            x_offset=np.full(sizes[0], 0.25),
            x_factor=np.full(sizes[0], 2.0),
            y_offset=np.full(sizes[-1], -0.5),
            y_factor=np.full(sizes[-1], 3.0),
        )
    inp, out = FullSpaceFormulation(m, _net(sizes, act), "pred", sc).build()
    m.subject_to(inp == x, name="link")
    m.minimize(out[0])
    return m


@pytest.mark.parametrize("sizes", [[3, 5, 2], [4, 8, 4, 1], [3, 4, 1], [5, 12, 12, 12, 2]])
@pytest.mark.parametrize("act", ["tanh", "sigmoid", "softplus", "linear"])
@pytest.mark.parametrize("scaled", [False, True])
def test_every_shape_exports(sizes, act, scaled):
    """Single-output nets with scaling used to fail here; see §55's two scalarizer bugs."""
    assert to_nl(_embed(sizes, act, scaled))


def test_one_constraint_per_layer_not_per_unit():
    """4-8-8-2 is THREE layers: 3 affine + 2 activation (the last is linear)."""
    m = _embed([4, 8, 8, 2], "tanh", scaled=False)
    nn_rows = [c for c in m._constraints if str(c.name).startswith("pred_")]
    assert len(nn_rows) == 5, [c.name for c in nn_rows]
    # ...while the model really is 38 scalar rows wide, so the small object count
    # is one body per layer and not a model that failed to be built.
    header = to_nl(m).split("\n")[1].split()
    affine = 8 + 8 + 2  # one row per unit of each layer's output
    activation = 8 + 8  # hidden layers only
    assert int(header[1]) == affine + activation + 4  # + the test's own link rows


@pytest.mark.parametrize("n_rows,expected", [(1, "c_0"), (2, "c"), (17, "c")])
def test_single_row_families_keep_the_loop_s_row_name(n_rows, expected):
    """A 1-row family is written under the bare name, which would rename the row."""
    assert _family_name("c", n_rows) == expected


def test_row_names_survive_in_lp():
    """LP carries row names; `.nl` does not, so only LP can show a rename."""
    text = to_lp(_embed([3, 4, 1], "linear", scaled=True))
    for name in ("pred_scale_in_0", "pred_affine_0_0", "pred_affine_1_0", "pred_scale_out_0"):
        assert f"{name}:" in text, f"{name} missing from LP output"


def _sigmoid_1x1x1():
    return NetworkDefinition(
        [
            DenseLayer(np.array([[0.8]]), np.array([0.1]), Activation.SIGMOID),
            DenseLayer(np.array([[1.2]]), np.array([-0.3]), Activation.LINEAR),
        ],
        input_bounds=(np.array([-1.0]), np.array([1.0])),
    )


@pytest.mark.slow
def test_the_embedded_net_still_certifies():
    """The emitter must reach `optimal`, not merely `feasible`."""
    m = Model("nn")
    inp, out = FullSpaceFormulation(m, _sigmoid_1x1x1(), "pred", None).build()
    m.subject_to(inp[0] == 0.09762701, name="fix")
    m.minimize(out[0])
    r = m.solve()
    assert r.status == "optimal", f"status {r.status!r}"
    assert r.objective == pytest.approx(0.353289693579, abs=1e-9)


@pytest.mark.slow
def test_the_matmul_form_would_lose_the_certificate():
    """Pins WHY the emitter avoids `@`, so the formulation cannot drift back.

    Same mathematics, same incumbent, weaker relaxation: `A @ x` does not certify
    where `dm.sum(A * x, axis=1)` does. If this ever starts passing as `optimal`,
    the matmul relaxation was fixed and the emitter may use `@` again.
    """
    W1, b1 = np.array([[0.8]]), np.array([0.1])
    W2, b2 = np.array([[1.2]]), np.array([-0.3])

    def build(matmul: bool):
        m = Model("nn")
        inp = m.continuous("inp", shape=(1,), lb=-1.0, ub=1.0)
        zh0 = m.continuous("zh0", shape=(1,), lb=-1e20, ub=1e20)
        z0 = m.continuous("z0", shape=(1,), lb=-1e20, ub=1e20)
        zh1 = m.continuous("zh1", shape=(1,), lb=-1e20, ub=1e20)
        if matmul:
            m.subject_to(zh0 == W1.T @ inp + b1, name="a0")
            m.subject_to(zh1 == W2.T @ z0 + b2, name="a1")
        else:
            m.subject_to(zh0 == dm.sum(W1.T * inp, axis=1) + b1, name="a0")
            m.subject_to(zh1 == dm.sum(W2.T * z0, axis=1) + b2, name="a1")
        m.subject_to(z0 == dm.sigmoid(zh0), name="act")
        m.subject_to(inp[0] == 0.09762701, name="fix")
        m.minimize(zh1[0])
        return m.solve()

    reduction, matmul = build(False), build(True)
    assert reduction.objective == pytest.approx(matmul.objective, abs=1e-9), (
        "the two forms disagree on the incumbent, which is a different bug"
    )
    assert reduction.status == "optimal"
    assert matmul.status == "feasible", (
        f"matmul now reports {matmul.status!r}; if the matmul relaxation was "
        "tightened, this test and the emitter's formulation should be revisited"
    )
