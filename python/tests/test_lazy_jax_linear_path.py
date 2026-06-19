"""Guard: pure LP/MILP/QP solves must not import JAX.

discopt's JAX/XLA stack costs ~0.5-1 s of cold-start init. The LP/QP/MIQP
default (POUNCE) and the pure-Rust simplex MILP B&B solve with no JAX
involvement, so they must not pay that tax — the import path is kept JAX-free
(lazy ``_jax`` package, lazy ``deadline`` jax, JAX-free ``problem_classifier``
+ numpy ``LPData``/``QPData``, and deferred ``solver`` imports).

MIQP is included: its B&B node QP relaxations solve via POUNCE (the pure-Rust
IPM), so the whole MIQP solve is JAX-free.

MILP note: the *default* engine is now POUNCE, but the POUNCE MILP B&B shares
the JAX-based relaxation/cut infrastructure (cover/clique/GMI separation), so
that path is **not** JAX-free. The JAX-free MILP path is the pure-Rust
warm-started simplex B&B (``nlp_solver="simplex"``), which is what this guard
pins for the MILP case.

Each case runs in a *fresh subprocess* and asserts ``'jax' not in sys.modules``
after the solve, so a regression that reintroduces an eager JAX import on the
JAX-free LP/QP/MIQP path (or the simplex MILP path) fails here.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_CASES = {
    "lp": """
        m = dm.Model('lp')
        x = m.continuous('x', shape=(5,), lb=0, ub=1)
        m.minimize(dm.sum([float(i + 1) * x[i] for i in range(5)]))
        m.subject_to(dm.sum([x[i] for i in range(5)]) >= 1.5)
    """,
    "milp": """
        m = dm.Model('milp')
        x = m.binary('x', shape=(8,))
        m.maximize(dm.sum([float(i + 1) * x[i] for i in range(8)]))
        m.subject_to(dm.sum([x[i] for i in range(8)]) <= 4)
    """,
    "qp": """
        m = dm.Model('qp')
        x = m.continuous('x', shape=(5,), lb=0, ub=1)
        m.minimize(dm.sum([x[i] * x[i] for i in range(5)]) - dm.sum([x[i] for i in range(5)]))
        m.subject_to(dm.sum([x[i] for i in range(5)]) >= 1.5)
    """,
    "miqp": """
        m = dm.Model('miqp')
        x = m.continuous('x', shape=(4,), lb=0, ub=1)
        y = m.binary('y', shape=(3,))
        m.minimize(
            dm.sum([x[i] * x[i] for i in range(4)])
            - dm.sum([x[i] for i in range(4)])
            + dm.sum([float(i + 1) * y[i] for i in range(3)])
        )
        m.subject_to(dm.sum([x[i] for i in range(4)]) + dm.sum([y[i] for i in range(3)]) >= 1.5)
    """,
}

# The default (POUNCE) is JAX-free for LP/QP/MIQP; the JAX-free MILP path is
# the pure-Rust warm-started simplex B&B (the POUNCE MILP B&B uses JAX cuts).
_SOLVER = {"milp": "simplex"}

_DRIVER = """
import os
os.environ['DISCOPT_DISABLE_JAX_CACHE'] = '1'
import sys
import discopt.modeling as dm
{body}
r = m.solve(time_limit=60{solver})
assert r.status == 'optimal', r.status
print('JAX_LOADED' if 'jax' in sys.modules else 'JAX_FREE')
"""


@pytest.mark.parametrize("name", list(_CASES))
def test_linear_solve_is_jax_free(name):
    _solver = _SOLVER.get(name)
    solver_arg = f", nlp_solver={_solver!r}" if _solver else ""
    script = _DRIVER.format(body=textwrap.dedent(_CASES[name]), solver=solver_arg)
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, f"solve failed:\n{out.stderr}"
    assert out.stdout.strip().splitlines()[-1] == "JAX_FREE", (
        f"{name} solve imported JAX (cold-start regression):\n{out.stdout}\n{out.stderr}"
    )
