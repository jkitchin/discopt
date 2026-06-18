"""Guard: pure LP/MILP/QP solves must not import JAX.

discopt's JAX/XLA stack costs ~0.5-1 s of cold-start init. LP/MILP/QP are solved
by HiGHS / the pure-Rust simplex with no JAX involvement, so they must not pay
that tax — the import path is kept JAX-free (lazy ``_jax`` package, lazy
``deadline`` jax, JAX-free ``problem_classifier`` + numpy ``LPData``/``QPData``,
and deferred ``solver`` imports).

Each case runs in a *fresh subprocess* and asserts ``'jax' not in sys.modules``
after the solve, so a regression that reintroduces an eager JAX import on the
linear path fails here. (MIQP still uses the JAX QP IPM for node relaxations and
is intentionally excluded — see the parity-push notes.)
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
}

_DRIVER = """
import os
os.environ['DISCOPT_DISABLE_JAX_CACHE'] = '1'
import sys
import discopt.modeling as dm
{body}
r = m.solve(time_limit=60)
assert r.status == 'optimal', r.status
print('JAX_LOADED' if 'jax' in sys.modules else 'JAX_FREE')
"""


@pytest.mark.parametrize("name", list(_CASES))
def test_linear_solve_is_jax_free(name):
    script = _DRIVER.format(body=textwrap.dedent(_CASES[name]))
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
