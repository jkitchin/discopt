"""Issue #1351: AMP partition refinement must reach a power with an AFFINE base.

``(x - c)**2`` is lowered by ``_build_power`` into an aux carrying a sound 1-D
envelope over the base's enclosure, but it gets no ``monomial_map`` entry -- that
key *means* the bare ``x_i**p``, and ``single_orig_col`` rightly refuses a
shifted/scaled base (registering ``(2x)**2`` as ``x**2`` would be unsound). With
no channel of its own, AMP's partition on ``x`` had nothing to attach to, so
``_apply_partition_refinement`` was a no-op for the term and the dual bound stayed
pinned at the root secant forever -- `max_iter=1000` and 16 initial partitions
moved it by exactly 0.0.

``DISCOPT_AFFINE_POWER_PARTITION=1`` records the affine base alongside the atom so
the existing piecewise machinery (already used for ``f(coeff*x + const)``
intrinsics) can refine it.

These tests FAIL with the flag off (that is the bug) and pass with it on.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.smoke

# minimize sum -(x_i - c_i)^2 over [-2,2]^3, -1 <= sum x <= 3.
# Global optimum -23.5 at x = (2, -1, -2), by vertex enumeration.
_SHIFTED_SQUARES = """
import discopt.modeling as dm
m = dm.Model("p")
xs = [m.continuous(f"x{i}", lb=-2.0, ub=2.0) for i in range(3)]
m.subject_to(sum(xs) >= -1.0)
m.subject_to(sum(xs) <= 3.0)
m.minimize(sum(-((xs[i] - c) ** 2) for i, c in enumerate([-1.0, 0.5, 1.5])))
r = m.solve(solver="amp", rel_gap=1e-4, max_iter=30)
print("RESULT", r.status, float(r.objective), float(r.bound))
"""

_TRUE_OPT = -23.5


def _run(source: str, flag: str) -> tuple[str, float, float]:
    """Run in a subprocess so the env var is read fresh by the relaxation builder."""
    env = dict(os.environ)
    env["DISCOPT_AFFINE_POWER_PARTITION"] = flag
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert out.returncode == 0, f"solve failed (flag={flag}):\n{out.stderr[-2000:]}"
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            _, status, obj, bound = line.split()
            return status, float(obj), float(bound)
    raise AssertionError(f"no RESULT line (flag={flag}):\n{out.stdout[-2000:]}")


def test_affine_power_partition_certifies_shifted_squares():
    """ON closes the gap the OFF path cannot move. This is the #1351 fix."""
    off_status, off_obj, off_bound = _run(_SHIFTED_SQUARES, "0")
    on_status, on_obj, on_bound = _run(_SHIFTED_SQUARES, "1")

    # Both arms must find the true optimum as their incumbent -- the bug was never
    # in the primal.
    assert off_obj == pytest.approx(_TRUE_OPT, abs=1e-4)
    assert on_obj == pytest.approx(_TRUE_OPT, abs=1e-4)

    # The bug: OFF is stuck at the root secant and cannot certify.
    assert off_status == "feasible"
    assert off_bound == pytest.approx(-26.5, abs=1e-4)

    # The fix: ON refines the envelope and certifies.
    assert on_status == "optimal", f"expected certification, got {on_status}"
    assert on_bound > off_bound + 1.0, f"bound did not tighten: {off_bound} -> {on_bound}"


def test_affine_power_partition_bound_stays_sound():
    """SOUNDNESS (CLAUDE.md §1): a tightened bound must never pass the optimum."""
    for flag in ("0", "1"):
        status, obj, bound = _run(_SHIFTED_SQUARES, flag)
        assert bound <= _TRUE_OPT + 1e-5, (
            f"flag={flag}: dual bound {bound} exceeds the true optimum {_TRUE_OPT}"
        )
        assert bound <= obj + 1e-5, (
            f"flag={flag}: dual bound {bound} exceeds its own incumbent {obj}"
        )


def test_affine_power_partition_defaults_on():
    """GRADUATED default-ON (#1351): refinement reaches the affine base by default.

    Graduated on real-instance evidence under ``solver="amp"``: ``nvs03`` gains a
    certificate (bound 15 -> 16, ``feasible`` -> ``optimal``, oracle optimum 16.0)
    and ``nvs04`` tightens 0 -> 0.16 (oracle 0.72). No bound exceeded its reference
    optimum on any carrier.
    """
    env = dict(os.environ)
    env.pop("DISCOPT_AFFINE_POWER_PARTITION", None)
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SHIFTED_SQUARES)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "RESULT optimal" in out.stdout, (
        "with no env var set the refined path must now be taken (default-ON); "
        f"got: {out.stdout[-500:]}"
    )


def test_affine_power_partition_opt_out_restores_legacy():
    """``=0`` keeps the legacy unrefined path intact, as §5 graduation requires."""
    status, obj, bound = _run(_SHIFTED_SQUARES, "0")
    assert status == "feasible"
    assert bound == pytest.approx(-26.5, abs=1e-4)
    assert obj == pytest.approx(_TRUE_OPT, abs=1e-4)


def test_bare_monomial_path_is_unchanged_by_the_flag():
    """Control: a bare ``x**2`` model already refined, and must not move."""
    src = """
    import discopt.modeling as dm
    m = dm.Model("p")
    xs = [m.continuous(f"x{i}", lb=-2.0, ub=2.0) for i in range(3)]
    m.subject_to(sum(xs) >= -1.0)
    m.subject_to(sum(xs) <= 3.0)
    m.minimize(sum(-(xs[i] ** 2) + 2 * c * xs[i] for i, c in enumerate([-1.0, 0.5, 1.5])))
    r = m.solve(solver="amp", rel_gap=1e-4, max_iter=30)
    print("RESULT", r.status, float(r.objective), float(r.bound))
    """
    off = _run(src, "0")
    on = _run(src, "1")
    assert off[0] == on[0] == "optimal"
    assert on[1] == pytest.approx(off[1], abs=1e-6)
    assert on[2] == pytest.approx(off[2], abs=1e-6)
