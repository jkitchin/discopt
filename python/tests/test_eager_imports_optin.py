"""Guard: ``DISCOPT_EAGER_IMPORTS`` is opt-in and never changes default startup.

F7 (perf-followup-plan §2, fixed-tax trims). The first ``solve()`` in a fresh
process lazily loads ~0.4 s of solve-path modules (jax.numpy, scipy
sparse/linalg, the discopt._jax / discopt.solvers stack). Batch/CLI/benchmark
harnesses that pay import cost once can opt in to move that tax to
``import discopt`` time via ``DISCOPT_EAGER_IMPORTS=1``.

This is bound-neutral plumbing: it must never change what a *default*
``import discopt`` does. In particular the default import must stay JAX-free
(the pure LP/MILP/QP paths must not pay the multi-second JAX cold start —
see ``test_lazy_jax_linear_path.py``). These guards pin:

* default (env unset / ``=0``): ``import discopt`` succeeds and does **not**
  eager-load jax or the solve orchestrator (startup stays lazy/fast);
* opt-in (env ``=1``): ``import discopt`` succeeds and the solve-path modules
  (jax, ``discopt.solver``) are preloaded.

Each case runs in a *fresh subprocess* so ``sys.modules`` reflects only what
importing discopt did, and a regression that eager-loads on the default path
(or fails to on the opt-in path) fails here.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Assert against sys.modules right after ``import discopt`` in a clean process.
_DRIVER = """
import sys
import discopt  # noqa: F401
jax_loaded = 'jax' in sys.modules
solver_loaded = 'discopt.solver' in sys.modules
print('JAX_LOADED' if jax_loaded else 'JAX_FREE')
print('SOLVER_LOADED' if solver_loaded else 'SOLVER_LAZY')
"""


def _run(env_value: str | None) -> list[str]:
    import os

    env = dict(os.environ)
    env.pop("DISCOPT_EAGER_IMPORTS", None)
    if env_value is not None:
        env["DISCOPT_EAGER_IMPORTS"] = env_value
    out = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert out.returncode == 0, f"import discopt failed:\n{out.stderr}"
    return out.stdout.strip().splitlines()


@pytest.mark.parametrize("env_value", [None, "0"])
def test_default_import_stays_lazy(env_value):
    """Default (env unset or =0): import succeeds and stays JAX-free/lazy."""
    lines = _run(env_value)
    assert lines[0] == "JAX_FREE", (
        f"default import eager-loaded JAX (cold-start regression): {lines}"
    )
    assert lines[1] == "SOLVER_LAZY", f"default import eager-loaded the solve orchestrator: {lines}"


def test_eager_optin_preloads_solve_path():
    """Opt-in (env =1): import succeeds and preloads jax + the orchestrator."""
    lines = _run("1")
    assert lines[0] == "JAX_LOADED", f"DISCOPT_EAGER_IMPORTS=1 did not eager-load JAX: {lines}"
    assert lines[1] == "SOLVER_LOADED", (
        f"DISCOPT_EAGER_IMPORTS=1 did not eager-load discopt.solver: {lines}"
    )
