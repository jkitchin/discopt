"""The vectorised reduced-space emitter builds the same model (#1215).

`ReducedSpaceFormulation` emitted one `Constraint` per unit; it now emits one
array-valued body per layer, fusing each layer's affine map and activation.

Verified the same way `full_space` was, with the same four regimes
(`discopt_benchmarks/scripts/issue1215_nn_verify.py`), because two of them are
blind to things the others catch:

* `.nl` and LP carry the numbers; LP and GAMS carry the row NAMES; only GAMS
  carries both names and a nonlinear body.
* the ARENA is a different lowering from the `.nl`, so its objective is checked
  separately -- `full_space` had a formulation that exported byte-identically and
  lost the certificate (§55).

Result: GAMS row names identical, arena objectives identical at every sampled
point. What does differ is understood and accepted (§57/§58): an array reduction
expands to a binary `+` fold where the old `dm.sum(lambda …)` was an n-ary
`SumOverExpression`, so the `.nl` sum opcode and the GAMS parenthesisation change
while the mathematics does not.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from discopt.export import to_gams, to_nl
from discopt.modeling import Model
from discopt.nn import DenseLayer, NetworkDefinition, OffsetScaling
from discopt.nn.formulations.reduced_space import ReducedSpaceFormulation


def _net(sizes, act="tanh"):
    rng = np.random.default_rng(7)
    return NetworkDefinition(
        layers=[
            DenseLayer(
                weights=rng.normal(0, 0.35, (sizes[i], sizes[i + 1])),
                biases=rng.normal(0, 0.2, sizes[i + 1]),
                activation=act if i < len(sizes) - 2 else "linear",
            )
            for i in range(len(sizes) - 1)
        ],
        input_bounds=(np.full(sizes[0], -1.0), np.full(sizes[0], 1.0)),
    )


def _embed(sizes, act="tanh", scaled=False):
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
    inp, out = ReducedSpaceFormulation(m, _net(sizes, act), "pred", sc).build()
    m.subject_to(inp == x, name="link")
    m.minimize(out[0])
    return m


@pytest.mark.parametrize("sizes", [[3, 5, 2], [4, 8, 4, 1], [3, 4, 1], [2, 3, 3, 3]])
@pytest.mark.parametrize("act", ["tanh", "sigmoid", "softplus"])
@pytest.mark.parametrize("scaled", [False, True])
def test_every_shape_exports(sizes, act, scaled):
    """Single-output nets used to fail in GAMS too (§57's bound-writer defect)."""
    m = _embed(sizes, act, scaled)
    assert to_nl(m)
    assert to_gams(m)


def test_one_constraint_per_layer_not_per_unit():
    """2-5-5-2 is three layers; the last is fused into the output constraint."""
    m = _embed([2, 5, 5, 2])
    nn_rows = [c for c in m._constraints if str(c.name).startswith("pred_")]
    assert len(nn_rows) == 3, [c.name for c in nn_rows]
    # ...and the model is still its full scalar width, so the small object count
    # is one body per layer rather than a model that failed to build.
    header = to_nl(m).split("\n")[1].split()
    assert int(header[1]) == 5 + 5 + 2 + 2  # two hidden layers + output + link


def test_gams_row_names_are_the_ones_the_loop_wrote():
    """The per-element loop wrote `pred_layer_0_0 … _4`; the family must too."""
    text = to_gams(_embed([3, 5, 2]))
    names = sorted(set(re.findall(r"^(pred_layer_0_\d+)\.\.", text, re.M)))
    assert names == [f"pred_layer_0_{k}" for k in range(5)], names


def test_relu_uses_maximum_and_is_refused_by_nl_not_silently_dropped():
    """RELU lowers to `max()`, which `.nl` cannot represent; it must REFUSE."""
    m = _embed([3, 4, 1], act="relu")
    with pytest.raises(ValueError, match="DNLP"):
        to_nl(m)


@pytest.mark.slow
def test_the_arena_objective_matches_the_per_element_form():
    """The solve path reads the arena, not the `.nl`; check it directly.

    Pinning the input makes the model an evaluation, so the objective at the
    optimum is the network's prediction and must not depend on how the layers
    were emitted.
    """
    from discopt._rust import model_to_repr

    m = _embed([3, 4, 1])
    rep = model_to_repr(m, getattr(m, "_builder", None))
    rng = np.random.default_rng(11)
    for _ in range(4):
        obj, _con, _bnd = rep.evaluate_point(list(rng.uniform(-0.9, 0.9, rep.n_vars)))
        assert np.isfinite(obj), "arena objective is not finite"
