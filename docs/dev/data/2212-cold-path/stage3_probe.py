#!/usr/bin/env python3
"""#2212: does #2163's third `ScaledProblem` stage still fire with the cold cascade on?

#2163 gave the Rust MILP driver its own logical-basis dual (`ScaledProblem::
logical_basis_dual`, counters `LogicalBasisDual{Attempts,Decided}`) for the node LPs
the cold primal could not decide. #2212 asks, once the engine's cold path does that
itself (`DISCOPT_LP_LOGICAL_DUAL_COLD=1`), whether that stage still fires. This
solves the four #2163 networks (`python/tests/data/relu_bigm_2163_nets.json`,
exactly as `test_2163_rust_engine_relu_bigm.py` builds them) on the Rust engine,
one arm per process (the flag is a `OnceLock`)::

    DISCOPT_PROFILE=1 DISCOPT_LP_MILP_BACKEND=rust DISCOPT_LP_LOGICAL_DUAL_COLD=0|1 \\
        python -u stage3_probe.py

One JSON line per network and an executed count (CLAUDE.md §6).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))
import discopt  # noqa: E402
import discopt.modeling as dm  # noqa: E402
from discopt._rust import profile_counters_py, profile_reset_py  # noqa: E402
from discopt.ml import NNFormulation  # noqa: E402
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition  # noqa: E402

assert Path(discopt.__file__).resolve().is_relative_to(REPO), discopt.__file__
assert os.environ.get("DISCOPT_LP_MILP_BACKEND") == "rust", "the stage lives in the Rust driver"
# #2212: `DISCOPT_LP_LOGICAL_DUAL_COLD` was RETIRED (docs/dev/flag-retirement-audit.md).
# Against a tree without it, the "ON" arm runs the same code as OFF and would read as
# "no effect" (CLAUDE.md §6), so refuse instead. Re-run from 70ae79b or earlier.
if os.environ.get("DISCOPT_LP_LOGICAL_DUAL_COLD", "").strip().lower() in ("1", "true", "on"):
    raise SystemExit(
        "DISCOPT_LP_LOGICAL_DUAL_COLD was retired (#2212); this arm would measure the "
        "default path. Check out 70ae79b or earlier to reproduce the recorded rows."
    )

NETS = json.loads((REPO / "python/tests/data/relu_bigm_2163_nets.json").read_text())

n = 0
for name, cuts in (
    ("scaled_numerical", False),
    ("snapped_incumbent", False),
    ("cuts_a", True),
    ("cuts_b", True),
):
    d = NETS[name]
    layers = [(np.array(lay["W"]), np.array(lay["b"])) for lay in d["layers"]]
    net = NetworkDefinition(
        [DenseLayer(W, b, Activation.RELU) for W, b in layers[:-1]]
        + [DenseLayer(*layers[-1], Activation.LINEAR)],
        input_bounds=(np.array(d["lo"]), np.array(d["hi"])),
    )
    m = dm.Model(f"relu_2163_{name}")
    nn = NNFormulation(m, net, strategy="relu_bigm", ideal_cuts=cuts)
    nn.formulate()
    m.minimize(dm.sum(nn.outputs))
    extra = {"cut_callback": nn.cut_callback} if cuts else {}
    profile_reset_py()
    r = m.solve(time_limit=60, threads=1, deterministic=True, **extra)
    c = profile_counters_py()
    row = {
        "net": name,
        "arm": os.environ.get("DISCOPT_LP_LOGICAL_DUAL_COLD", "0"),
        "status": str(r.status),
        "objective": r.objective,
        "nodes": r.node_count,
        "route": str(r.algorithm_route),
    }
    row.update(
        {
            k: v
            for k, v in c.items()
            if v and k.startswith(("LogicalBasisDual", "LogicalDualCold", "LpVerdictNumerical"))
        }
    )
    print(json.dumps(row), flush=True)
    n += 1
print(f"EXECUTED: {n}", flush=True)
