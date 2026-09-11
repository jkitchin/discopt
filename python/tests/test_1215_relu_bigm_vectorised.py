"""The vectorised relu-bigM emitter builds the same model (#1215).

`ReluBigMFormulation`'s four `build()` loops -- input scaling, the layer affine
map, smooth activations, output scaling -- now emit one array-valued body each.

**`_add_relu_constraints` is deliberately left per-unit.** It creates one binary
per *mixed* unit (`pred_q_0_3`), so vectorising it would make those an array
variable and change both their names and the model's COLUMN order -- which is how
a solver's `.sol` maps values back to variables. Its rows are also the cheap ones:
2-3 terms each, against `n_inputs` terms for every affine row, so the loops that
were vectorised are the ones whose cost grows with layer width. The measured
speedup rises from 2.1x to 7.9x as width goes 32 -> 256 precisely because of that
split (perf-plan §59).

`test_binary_variables_are_untouched` pins the decision: if someone vectorises
that block later, it fails and they have to think about column order first.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from discopt.export import to_gams, to_lp, to_nl
from discopt.modeling import Model
from discopt.nn import DenseLayer, NetworkDefinition, OffsetScaling
from discopt.nn.formulations.relu_bigm import ReluBigMFormulation


def _net(sizes, act="relu"):
    rng = np.random.default_rng(7)
    return NetworkDefinition(
        layers=[
            DenseLayer(
                weights=rng.normal(0, 0.5, (sizes[i], sizes[i + 1])),
                biases=rng.normal(0, 0.2, sizes[i + 1]),
                activation=act if i < len(sizes) - 2 else "linear",
            )
            for i in range(len(sizes) - 1)
        ],
        input_bounds=(np.full(sizes[0], -1.0), np.full(sizes[0], 1.0)),
    )


def _embed(sizes, act="relu", scaled=False):
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
    inp, out = ReluBigMFormulation(m, _net(sizes, act), "pred", sc).build()
    m.subject_to(inp == x, name="link")
    m.minimize(out[0])
    return m


@pytest.mark.parametrize("sizes", [[3, 5, 2], [4, 8, 4, 1], [3, 4, 1], [2, 3, 3, 3]])
@pytest.mark.parametrize("scaled", [False, True])
def test_every_shape_exports_to_every_format(sizes, scaled):
    m = _embed(sizes, scaled=scaled)
    assert to_nl(m)
    assert to_lp(m)  # relu-bigM is a MILP, so LP CAN represent it
    assert to_gams(m)


def test_affine_is_one_constraint_per_layer():
    m = _embed([4, 6, 6, 2])
    affine = [c for c in m._constraints if str(c.name).startswith("pred_affine")]
    assert len(affine) == 3, [c.name for c in affine]


def test_binary_variables_are_untouched():
    """The big-M block stays per-unit ON PURPOSE; see this module's docstring.

    Vectorising it would turn the per-unit binaries into an array variable,
    changing their names AND the model's column order -- which is `.sol` mapping.
    If that block is ever vectorised, this test must be revisited deliberately
    rather than silently updated.
    """
    m = _embed([4, 6, 6, 2])
    binaries = [v for v in m._variables if str(v.name).startswith("pred_q_")]
    assert binaries, "no big-M binaries were created"
    for v in binaries:
        assert v.shape == (), f"{v.name} became an array: column order moved"
    # One scalar binary per mixed unit, each individually named.
    assert len(set(v.name for v in binaries)) == len(binaries)


def test_relu_rows_still_carry_their_per_unit_names():
    text = to_lp(_embed([3, 5, 2]))
    for stem in ("pred_relu_lb", "pred_relu_ge", "pred_relu_ub", "pred_relu_bigm"):
        assert re.search(rf"{stem}_0_\d+", text), f"{stem} rows missing from LP"


def test_gams_affine_row_names_are_the_ones_the_loop_wrote():
    text = to_gams(_embed([3, 5, 2]))
    names = sorted(set(re.findall(r"^(pred_affine_0_\d+)\.\.", text, re.M)))
    assert names == [f"pred_affine_0_{k}" for k in range(5)], names


@pytest.mark.parametrize("act", ["tanh", "sigmoid", "softplus"])
def test_smooth_activations_still_work(act):
    """relu_bigm also handles smooth activations; that loop was vectorised too."""
    m = _embed([3, 4, 2], act=act)
    assert to_nl(m)
    smooth = [c for c in m._constraints if str(c.name).startswith("pred_act")]
    assert len(smooth) == 1, [c.name for c in smooth]
