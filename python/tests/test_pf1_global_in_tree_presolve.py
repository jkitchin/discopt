"""PF1 (issue #632) — the in-tree FBBT/branch-and-reduce kernel must fire on the
GLOBAL spatial B&B node loop, not only on the convex ``_solve_nlp_bb`` path.

The PF1 spike (``docs/dev/pf1-branch-and-reduce-spike-2026-07-14.md``) found the
kernel was wired *only* into ``_solve_nlp_bb`` (``solver.py`` node loop), so every
unproved spatial instance (tspn*, bchoco*, heatexch*, nvs05, …) processed every
node with **zero** ``in_tree_presolve`` invocations — a call-counter monkeypatch
read 0. These tests pin the wiring so it cannot silently disconnect again:

  * with ``in_tree_presolve_stride >= 1`` a nonconvex (spatial) MINLP records
    ``> 0`` global-loop firings, and
  * with ``in_tree_presolve_stride == 0`` it records exactly ``0`` (the pass is
    disabled), which is the pre-PF1 behaviour the spike measured.

The model is a small inline nonconvex bilinear MINLP (``x*y`` objective + integer
variable) so the solve routes through the global spatial B&B path — not a named
benchmark instance (per CLAUDE.md, named instances are gate probes, not the class
under test).

Soundness is covered elsewhere (``test_r2_branch_and_reduce.py`` feasible-point
retention; the PF0 differential + panel gates). These tests only pin the *wiring*.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import pytest
from discopt import solver


def _spatial_minlp() -> dm.Model:
    """A small nonconvex (bilinear) MINLP that routes through the global spatial
    B&B loop rather than the convex ``_solve_nlp_bb`` path."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(x + y + z >= 5)
    m.subject_to(x * y <= 6)
    m.minimize(x * y - 2 * x + z)
    return m


@pytest.mark.unit
def test_global_loop_invokes_reduce_kernel_when_stride_on():
    """stride>=1 => the global spatial B&B node loop calls the FBBT reduce kernel
    (>0 firings). This is the PF1 wiring the spike found missing (was 0)."""
    m = _spatial_minlp()
    r = m.solve(time_limit=20, in_tree_presolve_stride=1)
    assert r.status in ("optimal", "feasible", "time_limit")
    assert solver._in_tree_presolve_global_calls() > 0, (
        "global spatial B&B node loop never invoked the in-tree presolve kernel "
        "with stride=1 — PF1 wiring regressed (see #632)"
    )


@pytest.mark.unit
def test_global_loop_skips_reduce_kernel_when_stride_off():
    """stride==0 => the kernel is disabled and never fires (pre-PF1 behaviour)."""
    m = _spatial_minlp()
    m.solve(time_limit=20, in_tree_presolve_stride=0)
    assert solver._in_tree_presolve_global_calls() == 0, (
        "in-tree presolve fired on the global loop with stride=0 (must be disabled)"
    )
