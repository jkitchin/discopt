"""Surrogate-model protocol for active-learning optimization.

A surrogate is anything that, after seeing some ``(X, y)`` data, can
predict the response **with an uncertainty estimate** at new points.
That uncertainty is what acquisition functions like expected
improvement and UCB consume to decide where to run the next experiment.

Design philosophy
-----------------

There are three reasonable ways to hand a surrogate to
:func:`discopt.doe.optimize_round` (or :func:`discopt.doe.model_based_optimize_round`),
in increasing power:

1. **String preset** (zero knowledge required)::

       optimize_round(wb, surrogate="gp")

   Resolves to a sensible default (here, scikit-learn's
   ``GaussianProcessRegressor`` with a Matern(5/2) kernel + white
   noise). The string aliases live in :data:`PRESETS`.

2. **scikit-learn-compatible estimator** (auto-wrapped)::

       from sklearn.gaussian_process import GaussianProcessRegressor
       from sklearn.gaussian_process.kernels import Matern
       optimize_round(wb, surrogate=GaussianProcessRegressor(kernel=Matern(nu=2.5)))

       from pycse.sklearn.lpr import LinearLPR
       optimize_round(wb, surrogate=LinearLPR())

   Anything that quacks like sklearn (``fit``/``predict``) is wrapped
   in :class:`_SklearnUQAdapter`, which probes for UQ in this order:

   * ``predict(X, return_std=True)`` -- standard scikit-learn convention
     (GaussianProcessRegressor, BayesianRidge, ARDRegression, ...).
   * ``predict(X, return_interval=True)`` -- pycse linear local
     prediction regression returns a 95% interval; we convert to a
     pseudo-σ via ``(upper - lower) / (2 * 1.96)``.
   * Bootstrap residual fallback -- for plain regressors with no UQ,
     refit on resamples and take the per-point std of predictions.

3. **Custom object implementing the protocol** -- full escape hatch::

       class MyBespokeBayesian:
           def fit(self, X, y): ...; return self
           def predict(self, X): return mean, std
       optimize_round(wb, surrogate=MyBespokeBayesian())

For mechanistic models (you have ``y = f(d; θ)`` and want to fit ``θ``
between rounds), use :class:`discopt.doe.ParametricSurrogate` with
:func:`discopt.doe.model_based_optimize_round` instead.

You never need to subclass anything; the protocol is structural.

Conventions
-----------

* ``X`` is a 2D numpy array of shape ``(n_samples, n_features)``.
* ``y`` is a 1D numpy array of shape ``(n_samples,)``.
* ``predict(X)`` returns ``(mean, std)``, both 1D arrays of length
  ``n_samples``. ``std`` is non-negative; surrogates with no native
  UQ should not silently return zeros -- use the bootstrap adapter
  or raise.
"""

from __future__ import annotations

from typing import Callable, Protocol, cast, runtime_checkable

import numpy as np


@runtime_checkable
class Surrogate(Protocol):
    """Minimal interface every surrogate must satisfy.

    Implementations are expected to be *re-fittable*: each call to
    :meth:`fit` should reset the model to a fresh state trained on the
    new data, not incrementally update.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate": ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class _SklearnUQAdapter:
    """Adapt a scikit-learn-compatible estimator to the Surrogate protocol.

    The adapter probes for an uncertainty signal in this order:

    1. ``estimator.predict(X, return_std=True)`` -> ``(mean, std)``
    2. ``estimator.predict(X, return_interval=True)`` -> ``(mean, (lo, hi))``;
       converted to ``std = (hi - lo) / (2 * 1.96)`` (assumes 95%).
    3. Residual bootstrap: refit on ``n_bootstrap`` resamples, take the
       per-point std across predictions.

    The probe is done **once at fit time** and cached, so prediction
    is cheap.
    """

    def __init__(self, estimator, *, n_bootstrap: int = 32, random_state: int = 0):
        self._estimator = estimator
        self._n_bootstrap = int(n_bootstrap)
        self._random_state = int(random_state)
        self._mode: str | None = None
        self._bootstrap_models: list | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    @property
    def estimator(self):
        return self._estimator

    @property
    def mode(self) -> str | None:
        """One of ``"return_std"``, ``"return_interval"``, ``"bootstrap"``."""
        return self._mode

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SklearnUQAdapter":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._X_train = X
        self._y_train = y
        self._estimator.fit(X, y)
        self._mode = self._probe_mode(X)
        if self._mode == "bootstrap":
            self._bootstrap_models = self._fit_bootstrap(X, y)
        else:
            self._bootstrap_models = None
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if self._mode is None:
            raise RuntimeError("call fit() before predict()")
        if self._mode == "return_std":
            mean, std = self._estimator.predict(X, return_std=True)
            return np.asarray(mean, dtype=float).ravel(), np.asarray(std, dtype=float).ravel()
        if self._mode == "return_interval":
            out = self._estimator.predict(X, return_interval=True)
            mean, interval = out
            mean = np.asarray(mean, dtype=float).ravel()
            interval = np.asarray(interval, dtype=float)
            lo = interval[..., 0].ravel()
            hi = interval[..., 1].ravel()
            std = np.maximum(hi - lo, 0.0) / (2.0 * 1.959963984540054)
            return mean, std
        # bootstrap
        assert self._bootstrap_models is not None
        preds = np.stack([m.predict(X) for m in self._bootstrap_models], axis=0)
        mean = preds.mean(axis=0).ravel()
        std = preds.std(axis=0, ddof=1).ravel() if preds.shape[0] > 1 else np.zeros_like(mean)
        return mean, std

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _probe_mode(self, X: np.ndarray) -> str:
        sample = X[:1] if len(X) else X
        try:
            out = self._estimator.predict(sample, return_std=True)
        except TypeError:
            pass
        else:
            if isinstance(out, tuple) and len(out) == 2:
                return "return_std"
        try:
            out = self._estimator.predict(sample, return_interval=True)
        except TypeError:
            pass
        else:
            if isinstance(out, tuple) and len(out) == 2:
                interval = np.asarray(out[1])
                if interval.ndim >= 1 and interval.shape[-1] == 2:
                    return "return_interval"
        return "bootstrap"

    def _fit_bootstrap(self, X: np.ndarray, y: np.ndarray) -> list:
        from sklearn.base import clone

        rng = np.random.default_rng(self._random_state)
        n = len(X)
        models = []
        for _ in range(self._n_bootstrap):
            idx = rng.integers(0, n, size=n)
            m = clone(self._estimator)
            m.fit(X[idx], y[idx])
            models.append(m)
        return models


def _gp_preset() -> Surrogate:
    """Default GP: Matern(5/2) + WhiteKernel, normalized output."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=4, alpha=0.0
    )
    return _SklearnUQAdapter(gp)


def _response_surface_preset() -> Surrogate:
    """Default response surface: degree-2 polynomial with BayesianRidge UQ."""
    from sklearn.linear_model import BayesianRidge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        BayesianRidge(),
    )
    return _SklearnUQAdapter(pipe)


PRESETS: dict[str, Callable[[], Surrogate]] = {
    "gp": _gp_preset,
    "response-surface": _response_surface_preset,
}


def coerce_surrogate(obj: object) -> Surrogate:
    """Normalize a user-supplied surrogate spec to the Surrogate protocol.

    Accepts:

    * a string in :data:`PRESETS` (e.g. ``"gp"``);
    * any object with ``fit`` and ``predict`` (sklearn-style) -- wrapped
      in :class:`_SklearnUQAdapter`;
    * an object that already implements :class:`Surrogate`.

    Returns the surrogate ready for ``fit()``. Raises ``TypeError`` for
    anything else.
    """
    if isinstance(obj, str):
        try:
            factory = PRESETS[obj]
        except KeyError as e:
            raise ValueError(
                f"unknown surrogate preset {obj!r}; available: {sorted(PRESETS)}"
            ) from e
        return factory()
    if _matches_surrogate_protocol(obj):
        return cast(Surrogate, obj)  # already returns (mean, std)
    if hasattr(obj, "fit") and hasattr(obj, "predict"):
        return _SklearnUQAdapter(obj)
    raise TypeError(
        f"surrogate {obj!r} must be a string preset, a sklearn-style estimator "
        "(with fit/predict), or implement the Surrogate protocol"
    )


def _matches_surrogate_protocol(obj: object) -> bool:
    """True when predict() is known to return a 2-tuple already.

    We can't tell from the signature alone, so the protocol is treated
    as opt-in via a class attribute ``_is_discopt_surrogate = True``.
    Anything else goes through the sklearn adapter, which is the
    correct path for ``GaussianProcessRegressor`` etc.
    """
    return bool(getattr(obj, "_is_discopt_surrogate", False))


__all__ = [
    "PRESETS",
    "Surrogate",
    "coerce_surrogate",
]
