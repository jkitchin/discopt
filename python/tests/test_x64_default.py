"""discopt must default to JAX float64 unless the user opts out.

float32 silently produces ~1e-6 residual noise that breaks IPOPT tolerances and
solver correctness, so ``import discopt`` enables ``jax_enable_x64`` by default.
These run in subprocesses to control jax-vs-discopt import order (this test
process already has both imported with x64 on).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def _run(code: str, x64_env: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("JAX_ENABLE_X64", None)  # start from "unset" unless overridden
    if x64_env is not None:
        env["JAX_ENABLE_X64"] = x64_env
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )


def test_x64_on_when_discopt_imported_first():
    p = _run("import discopt, jax; print('X64', jax.config.jax_enable_x64)")
    assert p.returncode == 0, p.stderr[-400:]
    assert "X64 True" in p.stdout


def test_x64_on_when_jax_imported_first():
    # The edge case: jax reads JAX_ENABLE_X64 at its own import, so importing it
    # before discopt would leave it float32 unless discopt updates the live config.
    p = _run("import jax, discopt; print('X64', jax.config.jax_enable_x64)")
    assert p.returncode == 0, p.stderr[-400:]
    assert "X64 True" in p.stdout


def test_optout_respected():
    p = _run("import jax, discopt; print('X64', jax.config.jax_enable_x64)", x64_env="0")
    assert p.returncode == 0, p.stderr[-400:]
    assert "X64 False" in p.stdout


def test_import_discopt_does_not_force_jax_import():
    # The x64 default must stay free: importing discopt must not pull in jax.
    p = _run("import discopt, sys; print('JAX_LOADED', 'jax' in sys.modules)")
    assert p.returncode == 0, p.stderr[-400:]
    assert "JAX_LOADED False" in p.stdout
