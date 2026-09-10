"""``DenseLayer`` must accept a string activation, as its docstring promises.

``activation`` is typed as :class:`Activation`, but every reader and every
example passes a plain string (``"tanh"``). Nothing normalised it, so the bare
string was stored -- and every downstream membership test against a set of
``Activation`` members then classified the layer as *unsupported*.

``FullSpaceFormulation`` did exactly that and then crashed formatting its own
error message (``a.value`` on a ``str``), so a perfectly valid smooth network was
refused with ``AttributeError: 'str' object has no attribute 'value'`` instead of
being built. Both halves are fixed: the value is normalised on construction, and
the error path no longer assumes an enum.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.nn.network import Activation, DenseLayer

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("name", [a.value for a in Activation])
def test_string_activation_is_normalised(name):
    layer = DenseLayer(np.ones((2, 2)), np.zeros(2), activation=name)
    assert isinstance(layer.activation, Activation)
    assert layer.activation is Activation(name)


def test_enum_activation_is_left_alone():
    layer = DenseLayer(np.ones((2, 2)), np.zeros(2), activation=Activation.TANH)
    assert layer.activation is Activation.TANH


def test_unknown_activation_names_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown activation"):
        DenseLayer(np.ones((2, 2)), np.zeros(2), activation="frobnicate")


def test_smooth_network_with_string_activations_formulates():
    """The end-to-end regression: this used to raise ``AttributeError``."""
    import discopt.modeling as dm
    from discopt.nn import NetworkDefinition, add_predictor

    m = dm.Model("nn")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
    rng = np.random.default_rng(0)
    net = NetworkDefinition(
        [
            DenseLayer(rng.normal(size=(3, 4)) * 0.1, np.zeros(4), activation="tanh"),
            DenseLayer(rng.normal(size=(4, 1)) * 0.1, np.zeros(1), activation="tanh"),
        ],
        input_bounds=(np.zeros(3), np.ones(3)),
    )
    add_predictor(m, x, net)
    assert len(m._constraints) > 0, "formulation produced no constraints"


def test_unsupported_activation_reports_a_message_not_an_attribute_error():
    """ReLU is genuinely unsupported by the smooth formulation -- say so clearly."""
    import discopt.modeling as dm
    from discopt.nn import NetworkDefinition, add_predictor

    m = dm.Model("nn_relu")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    net = NetworkDefinition(
        [DenseLayer(np.ones((2, 2)) * 0.1, np.zeros(2), activation="relu")],
        input_bounds=(np.zeros(2), np.ones(2)),
    )
    with pytest.raises(ValueError, match="relu_bigm"):
        add_predictor(m, x, net, method="full_space")
