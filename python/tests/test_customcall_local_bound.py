"""#998 — the opaque ``CustomCall`` local-NLP path must not emit a fabricated bound.

A ``dm.custom`` body that does NOT trace soundly through MCBox (raw ``jnp``
intrinsics, a non-affine hidden division, …) has no valid node relaxation, so
``solve_model`` falls back to a single local NLP. ``_solve_continuous`` fills
``bound`` / ``gap`` / ``root_bound`` / ``root_gap`` from the NLP's own
convergence status — which for a local solver means "a KKT point was reached",
not "this is the global optimum". On a nonconvex opaque body that value is
routinely *above* the true global minimum, i.e. not a valid dual bound at all.

That is exactly the C-33/SC-1 defect already fixed for the convexity-unknown
local path; the ``CustomCall`` caller cleared ``gap_certified`` only, leaving the
fabricated bound/gap in place. An opaque body is a strictly weaker epistemic
position than "convexity unknown" — it cannot be inspected at all — so the strip
applies a fortiori.

Fails-before / passes-after: 2-D Ackley on the asymmetric box
``[-25.768, 39.768]`` has its global minimum 0.0 at the origin, but the default
start stalls at ≈ 15.06. Before the fix the opaque arm reported
``bound = 15.06`` with ``gap = 0.0`` on a problem whose optimum is 0.

The algebraic twin (same mathematics, inspectable AST → the convexity-unknown
path) is the control: the two paths must agree.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import math  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

# Ackley's global minimum on any box containing the origin.
TRUE_GLOBAL = 0.0
LB, UB = -25.768, 39.768


def _ackley(v):
    """2-D Ackley written with raw ``jnp`` intrinsics (NOT MCBox-traceable)."""
    n = v.shape[0]
    return (
        -20 * jnp.exp(-0.2 * jnp.sqrt(jnp.sum(v**2) / n))
        - jnp.exp(jnp.sum(jnp.cos(2 * math.pi * v)) / n)
        + 20
        + math.e
    )


def _opaque_model():
    """Ackley behind ``dm.custom`` — routes to the CustomCall local-NLP path."""
    m = dm.Model("ackley_customcall")
    x = m.continuous("x", shape=2, lb=LB, ub=UB)
    m.minimize(dm.custom(_ackley, name="ackley")(x))
    return m


def _algebraic_model():
    """The same mathematics as an inspectable AST — the convexity-unknown path."""
    m = dm.Model("ackley_algebraic")
    y = m.continuous("y", shape=2, lb=LB, ub=UB)
    m.minimize(
        -20 * dm.exp(-0.2 * dm.sqrt((y[0] ** 2 + y[1] ** 2) / 2))
        - dm.exp((dm.cos(2 * math.pi * y[0]) + dm.cos(2 * math.pi * y[1])) / 2)
        + 20
        + math.e
    )
    return m


def _assert_no_fabricated_bound(r, label):
    """No dual-bound claim may survive an uncertified local solve."""
    assert r.gap_certified is False, f"{label}: local NLP result certified as global"
    assert r.bound is None, f"{label}: fabricated dual bound {r.bound!r}"
    assert r.root_bound is None, f"{label}: fabricated root bound {r.root_bound!r}"
    assert r.gap is None, f"{label}: fabricated gap {r.gap!r}"
    assert r.root_gap is None, f"{label}: fabricated root gap {r.root_gap!r}"
    # "optimal" from a local NLP means only "the NLP converged". With an
    # incumbent in hand the honest verdict is "feasible": a feasible point was
    # found, global optimality was not proved.
    assert r.status == "feasible", f"{label}: local solution reported as {r.status!r}"


def _assert_bound_is_valid(r, label):
    """The property that actually matters (min sense): a *reported* dual bound
    must never exceed the true global optimum. Vacuous while the bound is
    stripped; it is the pin that survives if a future change ever routes this
    path to a genuine bound."""
    for field in ("bound", "root_bound"):
        val = getattr(r, field)
        if val is not None:
            assert val <= TRUE_GLOBAL + 1e-6, (
                f"{label}: {field}={val} exceeds the true global optimum "
                f"{TRUE_GLOBAL} — invalid dual bound"
            )


@pytest.mark.smoke
def test_customcall_local_path_strips_fabricated_bound():
    """Opaque ``dm.custom`` body: keep the incumbent, strip the bound/gap."""
    r = _opaque_model().solve(time_limit=60)

    # The feasible incumbent is kept — only the *claim* about it is withheld.
    assert r.objective is not None
    assert r.objective >= TRUE_GLOBAL - 1e-6

    _assert_no_fabricated_bound(r, "customcall")
    _assert_bound_is_valid(r, "customcall")


@pytest.mark.smoke
def test_algebraic_twin_control_agrees():
    """Control: the same mathematics on the convexity-unknown path (C-33/SC-1)."""
    r = _algebraic_model().solve(time_limit=60, skip_convex_check=True)

    _assert_no_fabricated_bound(r, "algebraic")
    _assert_bound_is_valid(r, "algebraic")


@pytest.mark.smoke
def test_customcall_and_algebraic_paths_agree():
    """The two spellings of one problem must report the same kind of answer.

    Before the fix the opaque arm reported ``bound=15.06, gap=0.0`` while the
    algebraic arm reported ``bound=None, gap=None`` — the same mathematics,
    two different soundness stories.
    """
    r_opaque = _opaque_model().solve(time_limit=60)
    r_algebraic = _algebraic_model().solve(time_limit=60, skip_convex_check=True)

    assert (r_opaque.bound is None) == (r_algebraic.bound is None)
    assert (r_opaque.gap is None) == (r_algebraic.gap is None)
    assert (r_opaque.root_bound is None) == (r_algebraic.root_bound is None)
    assert (r_opaque.root_gap is None) == (r_algebraic.root_gap is None)
    assert r_opaque.gap_certified == r_algebraic.gap_certified
    assert r_opaque.status == r_algebraic.status


@pytest.mark.smoke
def test_mcbox_traceable_customcall_still_certifies():
    """Control against over-correction: a CustomCall that DOES trace through
    MCBox goes to the global reduced-space engine and must keep its valid
    certificate — the strip above must not reach that path."""
    m = dm.Model("customcall_mcbox_convex")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    m.minimize(dm.custom(lambda v: v * v, name="sq")(x))
    r = m.solve(time_limit=60)

    assert r.status == "optimal"
    assert r.gap_certified, "MCBox-relaxable CustomCall lost its valid certificate"
    assert r.bound is not None
    assert r.bound <= r.objective + 1e-6
