"""Load ONNX models into NetworkDefinition."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from discopt.nn.network import Activation, DenseLayer, NetworkDefinition

_ONNX_ACTIVATION_MAP = {
    "Relu": Activation.RELU,
    "Sigmoid": Activation.SIGMOID,
    "Tanh": Activation.TANH,
    "Softplus": Activation.SOFTPLUS,
}

_STRUCTURAL_OPS = ("Flatten", "Reshape", "Identity", "Dropout")


def load_onnx(
    path: str | Path,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> NetworkDefinition:
    """Load an ONNX model into a :class:`NetworkDefinition`.

    Supports strictly *sequential* feedforward networks with dense layers
    (Gemm/MatMul) and standard activations (ReLU, Sigmoid, Tanh, Softplus).
    The parser tracks the single data tensor threading through the graph and
    refuses (rather than silently mis-reads) any branched / residual topology,
    input-transposed Gemm, mis-oriented MatMul, or non-constant weight.

    Parameters
    ----------
    path : str or Path
        Path to the ``.onnx`` file.
    input_bounds : tuple of (np.ndarray, np.ndarray) or None
        ``(lb, ub)`` bounds on each input feature.

    Returns
    -------
    NetworkDefinition

    Raises
    ------
    ImportError
        If ``onnx`` is not installed.
    ValueError
        If the model contains unsupported operations or is not a single
        sequential data path (branch/residual topology, ``transA=1`` Gemm,
        MatMul with the weight as ``input[0]``, or a non-initializer weight).
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as e:
        raise ImportError(
            "onnx is required to load ONNX models. Install with: pip install discopt[nn]"
        ) from e

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = model.graph

    # Build initializer lookup: name -> numpy array
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    if not graph.input:
        raise ValueError("ONNX graph has no inputs")
    # The single data tensor we thread through the graph. Any consumed compute
    # node whose data input is not this tensor means the graph is not a single
    # sequential path.
    current = graph.input[0].name

    layers: list[DenseLayer] = []
    nodes = list(graph.node)
    i = 0

    while i < len(nodes):
        node = nodes[i]

        if node.op_type in ("Gemm", "MatMul"):
            _check_data_input(node, current, node.op_type)
            weights, biases = _extract_gemm(node, initializers)
            current = node.output[0]
            i += 1

            # Bias supplied by a separate Add node (MatMul + Add, or Gemm w/o C).
            if biases is None and i < len(nodes) and nodes[i].op_type == "Add":
                biases = _extract_add_bias(nodes[i], initializers, current)
                current = nodes[i].output[0]
                i += 1

            # Optional activation immediately following.
            if i < len(nodes) and nodes[i].op_type in _ONNX_ACTIVATION_MAP:
                act_node = nodes[i]
                _check_data_input(act_node, current, act_node.op_type)
                activation = _ONNX_ACTIVATION_MAP[act_node.op_type]
                current = act_node.output[0]
                i += 1
            else:
                activation = Activation.LINEAR

            if biases is None:
                biases = np.zeros(weights.shape[1], dtype=np.float64)

            layers.append(DenseLayer(weights, biases, activation))

        elif node.op_type in _STRUCTURAL_OPS:
            _check_data_input(node, current, node.op_type)
            current = node.output[0]
            i += 1
        else:
            raise ValueError(
                f"Unsupported ONNX operation: {node.op_type}. "
                f"Only sequential dense networks are supported."
            )

    if not layers:
        raise ValueError("No dense layers found in ONNX model")

    return NetworkDefinition(layers, input_bounds=input_bounds)


def _check_data_input(node, current: str, what: str) -> None:
    """Verify the node's data input is the current tensor (sequential path)."""
    if not node.input or node.input[0] != current:
        got = node.input[0] if node.input else "<none>"
        raise ValueError(
            f"non-sequential ONNX graph: {what} node {node.name!r} consumes "
            f"data tensor {got!r} but the current tensor is {current!r}. "
            f"Only a single sequential data path is supported."
        )


def _extract_gemm(node, initializers: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract weights and biases from a Gemm or MatMul node.

    For Gemm ``Y = alpha * (A[' ] @ B[' ]) + beta * C`` we fold ``alpha`` into
    the weight and ``beta`` into the bias. ``transA=1`` (transposing the data
    input) is unsupported and raises.
    """
    if node.op_type == "Gemm":
        alpha = 1.0
        beta = 1.0
        trans_a = False
        trans_b = False
        for attr in node.attribute:
            if attr.name == "alpha":
                alpha = float(attr.f)
            elif attr.name == "beta":
                beta = float(attr.f)
            elif attr.name == "transA":
                trans_a = bool(attr.i)
            elif attr.name == "transB":
                trans_b = bool(attr.i)

        if trans_a:
            raise ValueError(
                "Gemm with transA=1 (data-input transpose) is not supported; "
                "only x @ W with an untransposed data input is representable."
            )

        w_name = node.input[1]
        if w_name not in initializers:
            raise ValueError(
                f"Gemm weight input {w_name!r} is not a constant initializer; "
                f"dynamic weights are not supported."
            )
        W = np.asarray(initializers[w_name], dtype=np.float64)
        if trans_b:
            W = W.T
        W = W * alpha

        biases = None
        if len(node.input) >= 3 and node.input[2] and node.input[2] in initializers:
            biases = np.asarray(initializers[node.input[2]], dtype=np.float64) * beta

        return W, biases

    # MatMul: data flows as ``x @ W`` so the weight must be input[1].
    if node.input[0] in initializers:
        raise ValueError(
            "MatMul with the weight as input[0] would mis-orient the data flow; "
            "only x @ W (weight as input[1]) is supported."
        )
    if node.input[1] not in initializers:
        raise ValueError(
            f"MatMul has no constant weight initializer at input[1] "
            f"(inputs {list(node.input)}); dynamic weights are not supported."
        )
    W = np.asarray(initializers[node.input[1]], dtype=np.float64)
    return W, None


def _extract_add_bias(node, initializers: dict, current: str) -> np.ndarray:
    """Extract a constant bias from an Add that adds it to the current tensor.

    Raises if the Add joins the current tensor with another *computed* tensor
    (a residual / branch join) rather than a constant initializer — this is the
    silent "residual Add consumed as zero bias" path being removed.
    """
    bias = None
    other_inputs = []
    for inp in node.input:
        if inp in initializers:
            if bias is None:
                bias = np.asarray(initializers[inp], dtype=np.float64)
        else:
            other_inputs.append(inp)

    if bias is None or other_inputs != [current]:
        raise ValueError(
            f"non-sequential ONNX graph: Add node {node.name!r} does not add a "
            f"constant bias to the current tensor {current!r} (inputs "
            f"{list(node.input)}); residual/branch topology is not supported."
        )
    return bias
