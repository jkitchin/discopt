"""The construction benchmark's arms must build the same model (#1215).

`scripts/bench_model_construction.py` gained a second discopt arm: the same
model in the vectorised idiom (one array-valued constraint body per form)
alongside the per-element one it always had. The two are compared directly --
`discopt-vec / discopt` is printed as "THE IDIOM GAP" -- so a divergence between
them would turn the benchmark into a comparison of two different models, which
is exactly the failure the cross-tool panel's identity gate exists to prevent.

The gate here is stronger than the panel's, because both arms are discopt and can
be diffed at the byte level: the `.nl` they write is **identical**, not merely the
same shape. That covers row order too, which is how a solver's `.sol` maps duals
back to constraints.

The Pyomo arm is checked on shape only -- a different modelling layer is not
expected to emit the same bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

pytest.importorskip("pyomo.environ", reason="the benchmark's baseline arm needs pyomo")

from scripts import bench_model_construction as bench  # noqa: E402

FORMS, INSTANCES = 4, 50
ROWS = FORMS * INSTANCES


def _discopt_nl(model):
    """`.nl` text for a benchmark model, which is built without an objective."""
    import discopt.modeling as dm
    from discopt.export import to_nl

    model.minimize(dm.sum(model._variables[0]))
    return to_nl(model)


def _pyomo_shape(model, tmp_path):
    import pyomo.environ as pyo

    model.obj = pyo.Objective(expr=sum(model.x[i] for i in model.I))
    path = str(tmp_path / "pyomo.nl")
    model.write(path, format="nl")
    header = Path(path).read_text().split("\n")[1].split()
    return int(header[1]), int(header[0])


def test_both_discopt_arms_are_registered():
    assert "discopt" in bench.ARMS and "discopt-vec" in bench.ARMS
    # Reporting only the per-element arm published the slower of two idioms as
    # "discopt"; both are default-on so that cannot happen silently again.
    assert "discopt" in bench.DEFAULT_ARMS and "discopt-vec" in bench.DEFAULT_ARMS


def test_arms_report_the_rows_they_built():
    for name in ("discopt", "discopt-vec", "pyomo"):
        _, built = bench.ARMS[name][0](FORMS, INSTANCES)
        assert built == ROWS, f"{name} reported {built} rows, expected {ROWS}"


def test_the_two_discopt_arms_write_byte_identical_nl():
    per_element, _ = bench.build_discopt(FORMS, INSTANCES)
    vectorised, _ = bench.build_discopt_vectorised(FORMS, INSTANCES)
    text = _discopt_nl(per_element)
    assert _discopt_nl(vectorised) == text
    # ...and it is a model of the expected size, so an all-empty match cannot
    # pass this test.
    header = text.split("\n")[1].split()
    assert (int(header[1]), int(header[0])) == (ROWS, 2 * INSTANCES)


def test_the_vectorised_arm_really_is_vectorised():
    """One `Constraint` per form, not per row -- otherwise the arm is a copy."""
    vectorised, _ = bench.build_discopt_vectorised(FORMS, INSTANCES)
    per_element, _ = bench.build_discopt(FORMS, INSTANCES)
    assert len(vectorised._constraints) == FORMS
    assert len(per_element._constraints) == ROWS


def test_pyomo_arm_builds_the_same_shape(tmp_path):
    pyomo_model, _ = bench.build_pyomo(FORMS, INSTANCES)
    assert _pyomo_shape(pyomo_model, tmp_path) == (ROWS, 2 * INSTANCES)
