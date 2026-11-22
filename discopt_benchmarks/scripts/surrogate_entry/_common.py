"""Shared fixtures for the surrogate-embedding entry experiments.

Provenance and acknowledgement
------------------------------
These experiments were prompted by two sources, and the questions they test are
taken from them:

* S. Sanyal, C. Demir, M. Türkay, E. Aydın, "PWLNet: A Framework for Linearized
  Neural Surrogates in Nonlinear Process Optimization", arXiv:2611.25969v1
  (2026), code at https://github.com/SunoySanyal/PWLNet. Source of: big-M
  ReLU constants from interval propagation and their looseness over deep or
  unrolled networks (their Sec. 3.2, 4.3, 5.3.3), and post-training
  piecewise-linear tanh activations embedded as MILPs (their Sec. 4.4-4.5).
* J. Kitchin, "MIP surrogates" (blog post, 2026-10-28),
  https://kitchingroup.cheme.cmu.edu/posts/mip-surrogates/. Source of: the 2-D
  test function used below, and the input-convex network (ICNN) LP route with no
  binaries.

No code is copied from either; the experiments re-measure their claims on
discopt's own formulations. The underlying methods are older and are cited in
each script: LP-based neuron bounds (Tjeng, Xiao, Tedrake 2019; Grimstad &
Andersson 2019), input-convex networks (Amos, Xu, Kolter 2017).

Training uses scikit-learn's MLPRegressor with fixed seeds, so every network in
the panel is reproducible.
"""

from __future__ import annotations

import os
import subprocess
import warnings

import discopt
import numpy as np

# --------------------------------------------------------------------------
# Loaded-code check (CLAUDE.md §8): print what was imported, from where.
# --------------------------------------------------------------------------


def loaded_code_banner() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
            check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - banner only, re-raised below
        raise RuntimeError("cannot resolve git HEAD for the loaded-code banner") from exc
    load = os.getloadavg()
    return (
        f"discopt from {discopt.__file__} @ {sha}; "
        f"load avg {load[0]:.2f}/{load[1]:.2f}/{load[2]:.2f} on {os.cpu_count()} cpus"
    )


# --------------------------------------------------------------------------
# Test functions
# --------------------------------------------------------------------------


def kitchin_f(X: np.ndarray) -> np.ndarray:
    """The 2-D benchmark from Kitchin (2026), generalised to d inputs.

    For d = 2 this is f(x) = 1/4 ||x||^2 + sin(3 x1) + cos(2 x2) + 0.15 x2 on
    [-2, 2]^2, exactly as in the post. For d > 2 the sin/cos pattern alternates
    over coordinates (this extension is ours).
    """
    X = np.atleast_2d(X)
    out = 0.25 * np.sum(X**2, axis=1)
    for i in range(X.shape[1]):
        out = out + (np.sin(3 * X[:, i]) if i % 2 == 0 else np.cos(2 * X[:, i]))
    if X.shape[1] >= 2:
        out = out + 0.15 * X[:, 1]
    return out


def convex_f(X: np.ndarray) -> np.ndarray:
    """A convex, non-smooth target for the ICNN experiment (ours)."""
    X = np.atleast_2d(X)
    return 0.25 * np.sum(X**2, axis=1) + np.abs(X[:, 0] - 0.5) + 0.3 * np.abs(X[:, -1] + 0.7)


BOX = 2.0


def sample(d: int, n: int, seed: int, fn=kitchin_f) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-BOX, BOX, size=(n, d))
    y = fn(X) + 0.02 * rng.standard_normal(n)
    return X, y


def train_mlp(d: int, hidden: tuple[int, ...], activation: str, seed: int, fn=kitchin_f):
    """Train a reproducible sklearn MLP surrogate and return it with its test RMSE."""
    from sklearn.neural_network import MLPRegressor

    X, y = sample(d, 400 * d, seed, fn)
    Xt, yt = sample(d, 2000, seed + 1000, fn)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=activation,
        max_iter=3000,
        random_state=seed,
        tol=1e-6,
        n_iter_no_change=50,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ConvergenceWarning only; RMSE is reported
        mlp.fit(X, y)
    rmse = float(np.sqrt(np.mean((mlp.predict(Xt) - yt) ** 2)))
    return mlp, rmse


def to_network(mlp, d: int):
    """sklearn MLP -> discopt NetworkDefinition on the [-BOX, BOX]^d input box."""
    from discopt.ml import load_sklearn_mlp

    net = load_sklearn_mlp(mlp)
    net.input_bounds = (np.full(d, -BOX), np.full(d, BOX))
    return net


def describe(res) -> dict:
    """The fields of a SolveResult that the experiments compare."""
    return {
        "status": res.status,
        "objective": None if res.objective is None else float(res.objective),
        "bound": None if res.bound is None else float(res.bound),
        "gap_certified": bool(getattr(res, "gap_certified", False)),
        "nodes": int(getattr(res, "node_count", 0) or 0),
        "wall": float(res.wall_time),
    }
