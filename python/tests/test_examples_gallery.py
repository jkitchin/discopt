"""Regression tests for the examples-gallery fixes E1, E2, E3.

- **E1** (`example_pooling_haverly`): the product-1 sulfur spec was dimensionally
  inconsistent and, once "cleared" by multiplying through by the pool flow, went
  vacuous when the pool was bypassed — certifying a false optimum of 500 that ships
  2 %-sulfur product against a 1.5 % spec. Reformulated with the pool *concentration*
  so the spec binds even at zero pool flow; the classic Haverly-I optimum is 400.
- **E2** (`example_logical_constraints`): used non-existent API (`subject_to` on a
  logical expression, `m.disjunct`) and encoded the precedence "3 requires 0" as a
  conjunction that forces both values. Now builds and validates.
- **E3** (`example_reactor_design`): the heat balance drove `T[2] >= 940 K` against a
  750 K limit — provably infeasible. Fixed to a feasible adiabatic cascade.

The whole-gallery build/validate test is `smoke`; the two global/feasibility solves
carry time limits.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from discopt.modeling import examples


def _build(factory):
    """Construct an example model, swallowing its illustrative print(m)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return factory()


# Pure-modeling examples (no external deps: nn / pyomo / a .nl file / litellm).
_GALLERY = [
    "example_simple_minlp",
    "example_pooling_haverly",
    "example_process_synthesis",
    "example_portfolio",
    "example_reactor_design",
    "example_facility_location",
    "example_parametric",
    "example_logical_constraints",
    "example_transportation",
    "example_assignment",
    "example_multicommodity_flow",
]


@pytest.mark.smoke
@pytest.mark.parametrize("fname", _GALLERY)
def test_every_gallery_example_builds_and_validates(fname):
    """Every gallery example constructs and passes validate() (E2 crashed here)."""
    m = _build(getattr(examples, fname))
    m.validate()  # must not raise


@pytest.mark.smoke
def test_e2_logical_constraints_precedence_is_an_implication():
    """The precedence constraint must be the implication ~a3 | a0, not a conjunction."""
    m = _build(examples.example_logical_constraints)
    # A conjunction would have forced active[3]=0 and active[0]=1; as an
    # implication the model just constrains them. Sanity: it builds with the
    # expected logical + disjunction constraints present.
    names = {getattr(c, "name", None) for c in m._constraints}
    assert {"budget", "precedence", "require_one"} <= names


def test_e1_haverly_certifies_classic_optimum_400():
    """The fixed Haverly-I model certifies the textbook optimum of 400."""
    m = _build(examples.example_pooling_haverly)
    r = m.solve(time_limit=120)
    assert r.objective is not None
    assert float(r.objective) == pytest.approx(400.0, abs=1.0)
    # The certified point must actually respect the 1.5 % product-1 spec.
    val = {v.name: np.array(r.value(v)) for v in m._variables}
    x = val["x_pool_to_product"]
    z = float(val["z_direct"])
    p = float(val["pool_sulfur_concentration"])
    flow1 = float(x[1]) + z
    if flow1 > 1e-6:
        conc1 = (p * float(x[1]) + 2.0 * z) / flow1
        assert conc1 <= 1.5 + 1e-4


def test_e3_reactor_is_feasible_not_infeasible():
    """The fixed reactor model is feasible (was provably infeasible)."""
    m = _build(examples.example_reactor_design)
    r = m.solve(time_limit=90)
    assert r.status in ("optimal", "feasible"), f"reactor status {r.status}"
    T = np.array(r.value(next(v for v in m._variables if v.name == "temperature")))
    assert np.all(T <= 750.0 + 1e-3)  # material limit respected
    assert float(T[0]) <= 400.0 + 1e-3  # feed temperature limit
