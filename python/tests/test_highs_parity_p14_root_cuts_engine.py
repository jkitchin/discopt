"""highs-parity P1.4: NLP-BB root cuts on the Rust simplex (``DISCOPT_NLPBB_ROOT_CUTS_ENGINE``).

The Rust arm solves the root LP on the in-house simplex and takes GMI cuts from
``gomory_cuts_py`` at its vertex, projected to x-space and filtered exactly as the HiGHS
separator's. Pinned here: engine selection (an install without highspy now gets the Rust
root cuts instead of none, #1514), the exact-enumeration soundness check on the Rust arm,
and the stage deadline reaching the Rust simplex.
"""

from __future__ import annotations

import builtins

import discopt.solvers._root_cuts as rc
import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_engine_selection(monkeypatch):
    monkeypatch.delenv("DISCOPT_NLPBB_ROOT_CUTS_ENGINE", raising=False)
    pytest.importorskip("highspy")
    assert rc.root_cuts_engine() == "highs"
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_ENGINE", "rust")
    assert rc.root_cuts_engine() == "rust"
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_ENGINE", "rsut")
    with pytest.raises(ValueError, match="DISCOPT_NLPBB_ROOT_CUTS_ENGINE"):
        rc.root_cuts_engine()


def _block_highspy(monkeypatch):
    real = builtins.__import__

    def imp(name, *a, **k):
        if name == "highspy" or name.startswith("highspy."):
            raise ImportError("simulated: highspy unavailable")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", imp)


def _root(seed=7):
    import types

    rng = np.random.default_rng(seed)
    n_int, n_cont, m = 3, 2, 4
    n = n_int + n_cont
    a_mat = np.round(rng.normal(0, 2, size=(m, n)), 1)
    lb = np.zeros(n)
    ub = np.concatenate([rng.integers(1, 6, n_int).astype(float), rng.uniform(1, 8, n_cont)])
    b = a_mat @ ((lb + ub) / 2) + rng.uniform(0.5, 3.0, m)
    c = np.round(rng.normal(0, 1, n), 2)
    return types.SimpleNamespace(
        n=n, lb=lb, ub=ub, is_int=np.array([True] * n_int + [False] * n_cont), A_le=a_mat,
        b_le=b, A_eq=np.zeros((0, n)), b_eq=np.zeros(0), c=c, c0=0.0, c0_negate=False,
        ev=types.SimpleNamespace(evaluate_objective=lambda x, _c=c: float(_c @ x)),
        sense_max=True,
    )  # fmt: skip


def test_no_highspy_install_gets_rust_root_cuts(monkeypatch):
    monkeypatch.delenv("DISCOPT_NLPBB_ROOT_CUTS_ENGINE", raising=False)
    _block_highspy(monkeypatch)
    assert rc.root_cuts_engine() == "rust"
    produced = 0
    for seed in range(12):
        root = _root(seed)
        obj, x, duals, h = rc._solve_lp(root, [], [])
        if x is None:
            continue
        assert isinstance(h, rc._RustRootBasis)
        produced += len(rc.separate_gmi(root, h, x, root.A_le, root.b_le, max_cuts=16))
    assert produced > 0, "no GMI cut from the Rust arm: the test lost its teeth"


def test_rust_gmi_cuts_valid_by_exact_enumeration(monkeypatch):
    # The HiGHS-arm test, run on the Rust arm (its ``_solve_lp`` follows the flag).
    import importlib.util
    from pathlib import Path

    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_ENGINE", "rust")
    pytest.importorskip("scipy.optimize")
    path = Path(__file__).parent / "test_nlpbb_root_cuts_781.py"
    spec = importlib.util.spec_from_file_location("t781_p14", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.test_gmi_cuts_valid_by_exact_enumeration()


def test_the_deadline_reaches_the_rust_simplex(monkeypatch):
    from discopt.solvers import lp_std_engine as E

    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_ENGINE", "rust")
    seen = []
    real = E.RustEngine.solve

    def spy(self, sf, **k):
        seen.append(k.get("time_limit"))
        return real(self, sf, **k)

    monkeypatch.setattr(E.RustEngine, "solve", spy)
    rc._solve_lp(_root(), [], [], time_limit=4.25)
    assert seen == [4.25]
