"""#861 — the probe and validation boxes are generated INSIDE the model's root box.

``IncrementalMcCormickLP`` identified its structure on a synthetic probe box
(``[1, 7+k]`` / ``[-(7+k), -1]``) and validated it against synthetic comparison
boxes (fixed magnitudes, ``lb=0`` on even trials). Both ignored the model's actual
bounds, so for most models they were **unreachable**: on ``gear`` the root box is
``[12,60]^4`` and the structure was identified on ``[1,7+k]``, where the factorable
engine takes a *different decomposition route* for the ratio ``x0*x1/(x2*x3)``
(log/reciprocal columns). The structure was therefore built over a 15-column layout
while every real node builds 10, and ``_validate`` reported ``column-count
mismatch``. That was the largest decline bucket after the row-classifier fix: 14
instances, of which 11 were this artifact.

Anchoring both box families to the root box (and letting the caller pass its
*presolved* root box) makes the comparison boxes ones the tree can actually branch
into. Measured over the 81-instance corpus:

    column-count mismatch   14 -> 3      (only nvs01, nvs21, st_e17 remain)
    no valid bound / rows    5 -> 3
    admitted                36 -> 36     (0 admitted->declined flips)

The three survivors are *genuinely* box-unstable — their lifted decomposition
changes on reachable interior sub-boxes (nvs01: 11 -> 15 columns), which a
fixed-layout structure cannot reproduce. They must keep declining, and this file
pins that as the negative control: a test that only checked "more models admit"
would be satisfied by an unsound widening that stopped detecting real divergence.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
from discopt._relax.term_classifier import classify_nonlinear_terms

_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "minlplib")


def _build(model, box=None):
    return IncrementalMcCormickLP(model, classify_nonlinear_terms(model), deadline=None, box=box)


def _shifted_box_model():
    """A model whose root box is far from the origin — the shape the synthetic probe
    box misrepresented. Every variable is positive-definite and bounded away from 0,
    so a probe at ``[1, 7+k]`` sits outside the root interval entirely."""
    m = dm.Model("shifted")
    x = m.continuous("x", lb=12.0, ub=60.0)
    y = m.continuous("y", lb=12.0, ub=60.0)
    m.minimize(x * y)
    m.subject_to(x + y >= 30.0)
    return m


def test_probe_box_lies_inside_the_root_box():
    inc = _build(_shifted_box_model())
    assert inc.ok, f"declined: {inc.decline_reason}"
    lo, hi = inc._probe_lb, inc._probe_ub
    assert np.all(lo >= 12.0) and np.all(hi <= 60.0), f"probe {lo}..{hi} escapes the root box"
    assert np.all(hi > lo), "probe box must have distinct endpoints"


def test_validation_boxes_lie_inside_the_root_box():
    inc = _build(_shifted_box_model())
    assert inc.ok
    executed = 0
    for lo, hi in inc._validation_boxes():
        assert np.all(lo >= 12.0 - 1e-12), f"validation box lb {lo} below the root lb"
        assert np.all(hi <= 60.0 + 1e-12), f"validation box ub {hi} above the root ub"
        assert np.all(hi >= lo)
        executed += 1
    assert executed == 6, f"expected 6 validation boxes, got {executed}"


def test_caller_box_overrides_the_declared_bounds():
    """The caller's presolved box is what the tree branches in, so it — not the
    declared box — must anchor the probe and validation boxes."""
    m = _shifted_box_model()
    tight = (np.array([20.0, 20.0]), np.array([30.0, 30.0]))
    inc = _build(m, box=tight)
    assert inc.ok, f"declined: {inc.decline_reason}"
    assert np.all(inc._probe_lb >= 20.0) and np.all(inc._probe_ub <= 30.0)
    for lo, hi in inc._validation_boxes():
        assert np.all(lo >= 20.0 - 1e-12) and np.all(hi <= 30.0 + 1e-12)


def test_spanning_root_still_reaches_its_sign_regimes():
    """Anchoring must not cost the C-21 regime coverage: a variable whose ROOT box
    straddles zero must still be driven through negative, zero-spanning and pinned
    boxes, because real nodes reach those and the envelope must match there."""
    m = dm.Model("span")
    x = m.continuous("x", lb=-3.0, ub=4.0)
    y = m.continuous("y", lb=-2.0, ub=5.0)
    m.minimize(x * y)
    m.subject_to(x + y >= 1.0)
    inc = _build(m)
    assert inc.ok, f"declined: {inc.decline_reason}"
    regimes = set()
    for lo, hi in inc._validation_boxes():
        for i in range(inc.n):
            regimes.add(inc._box_sign_regime(float(lo[i]), float(hi[i])))
    # The regimes a spanning variable's real nodes actually reach.
    for needed in ("span", "neg", "degen", "zero_lb"):
        assert needed in regimes, f"regime {needed!r} no longer exercised (got {regimes})"


def test_infinite_root_endpoint_still_builds():
    """A half-infinite declared box must not stop the structure being identified —
    the probe needs finite endpoints, so an infinite end gets a finite stand-in
    anchored at the finite side. (Before this, such models died at the probe build
    with 'relaxation has no valid bound / no rows' and could not be judged at all.)"""
    m = dm.Model("halfinf")
    x = m.continuous("x", lb=1.0, ub=float("inf"))
    y = m.continuous("y", lb=1.0, ub=8.0)
    m.minimize(x * y)
    m.subject_to(x + y >= 3.0)
    inc = _build(m)
    lo, hi = inc._probe_lb, inc._probe_ub
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), "probe box must be finite"
    assert np.all(lo >= 1.0), "probe must respect the finite side of the root interval"


@pytest.mark.parametrize("name", ["gear", "gear2", "gear3", "ex1225", "ex1226", "st_e03"])
def test_column_count_artifact_is_gone(name):
    """These declined with ``column-count mismatch`` purely because the synthetic
    probe box took a different decomposition route. They may still decline — most
    need lifted-family patch coverage (the T5 workstream) — but never again for
    that reason."""
    path = os.path.join(_CORPUS, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name} not in the in-repo corpus")
    from discopt.modeling.core import from_nl

    inc = _build(from_nl(path))
    assert "column-count mismatch" not in (inc.decline_reason or ""), (
        f"{name} still hits the probe-box layout artifact: {inc.decline_reason}"
    )


@pytest.mark.parametrize("name", ["nvs01", "nvs21", "st_e17"])
def test_genuinely_unstable_models_still_decline(name):
    """NEGATIVE CONTROL. These models' lifted decomposition really does change on a
    reachable interior sub-box (nvs01: 11 -> 15 columns), so a fixed-layout structure
    cannot reproduce their per-node cold build. ``_validate`` must keep catching
    that. A widening that admitted these would have stopped detecting real
    patch/cold divergence — the failure this whole gate exists to prevent."""
    path = os.path.join(_CORPUS, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name} not in the in-repo corpus")
    from discopt.modeling.core import from_nl

    inc = _build(from_nl(path))
    assert not inc.ok, f"{name} was admitted despite a box-dependent decomposition"
    assert "column-count mismatch" in (inc.decline_reason or ""), (
        f"{name} declined for an unexpected reason: {inc.decline_reason}"
    )
