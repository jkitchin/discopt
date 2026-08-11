"""Native POUNCE NLP solve — route node NLPs through POUNCE's own AD.

The production node-NLP path drives POUNCE (a pure-Rust Ipopt port) through a
Python callback proxy (``_IpoptCallbacks`` → JAX ``NLPEvaluator``): POUNCE asks
for ``f/∇f/g/J/H`` once per IPM iteration and each request crosses the Rust→
Python→JAX boundary. POUNCE already parses ``.nl`` files into its *own*
AD-backed problem (``pounce.read_nl`` → ``NlProblem``), so for a model that
originated from (or can be emitted to) an ``.nl`` we can hand POUNCE the whole
problem once and let it differentiate natively — zero JAX in the node loop.

Why the orderings line up (validated by ``tests/test_native_nlp_equivalence``):

* **Variables** — ``from_nl`` builds ``model._variables`` in ``.nl`` column
  order, and the JAX evaluator's flat ``x`` is that same declaration order
  (``dag_compiler._compute_var_offset``). So for ``.nl``-originated models the
  evaluator order *is* the ``.nl`` order: objective/gradient/Hessian agree to
  ~1e-15 under the identity map. A model emitted via ``to_nl`` is reordered into
  canonical AMPL order (nonlinear-first); for that case we recover the exact
  permutation from the writer and apply it on the way in/out.
* **Constraints** — the ``.nl`` canonicalizes bodies (constant terms moved to
  the bound side, rows reordered) so its constraint *representation* differs
  from the evaluator's. This is benign: POUNCE owns the constraints end-to-end
  from the ``.nl`` (same feasible region), and discopt only consumes the
  solution vector (variable-ordered) and the objective. The node solve overrides
  *variable* bounds only; constraint bounds are static and come from the ``.nl``.

Objective sense: POUNCE always *minimizes* internally — ``base.objective``,
``base.gradient`` and the solved ``obj_val`` are all reported in minimization
sense even when ``base.minimize`` is ``False`` (a maximize ``.nl``). The
evaluator/B&B also work in minimization sense (the evaluator negates a maximize
objective), so the two line up directly with no sign correction. ``base.minimize``
is metadata only.

Every base is *validated* numerically (objective + gradient at a probe point)
before use; on any mismatch (or when POUNCE/AD is unavailable) the caller falls
back to the JAX path, so correctness never depends on an unverified assumption.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("discopt.nlp_native")

try:  # POUNCE is a core dependency, but keep the import defensive.
    import pounce

    _POUNCE_OK = hasattr(pounce, "read_nl") and hasattr(pounce, "solve_nlp_batch")
except Exception:  # pragma: no cover - exercised only when pounce is absent
    pounce = None  # type: ignore[assignment]
    _POUNCE_OK = False


@dataclass
class NativeNlpBase:
    """A POUNCE-native NLP problem aligned to a discopt evaluator's variables.

    ``base`` is the parsed ``NlProblem`` (POUNCE's own AD). ``perm`` maps a
    ``.nl`` column index to the evaluator's flat variable index — ``perm[j]`` is
    the evaluator index of ``.nl`` column ``j`` — so ``x_nl = x_eval[perm]`` and
    ``x_eval[perm] = x_nl``. ``perm is None`` means the identity map (the common
    ``from_nl`` case). No objective-sign correction is needed: POUNCE reports
    everything in minimization sense (see module docstring); ``base.minimize`` is
    metadata only.
    """

    base: "pounce.NlProblem"  # type: ignore[name-defined]
    n: int
    perm: Optional[np.ndarray]  # .nl-column -> evaluator-index; None = identity
    source: str
    _tmpfiles: tuple[str, ...] = ()

    def to_nl_order(self, x_eval: np.ndarray) -> np.ndarray:
        """Permute an evaluator-ordered vector into ``.nl`` column order."""
        x_eval = np.asarray(x_eval, dtype=np.float64)
        return x_eval if self.perm is None else x_eval[self.perm]

    def to_eval_order(self, x_nl: np.ndarray) -> np.ndarray:
        """Permute a ``.nl``-ordered vector back into evaluator order."""
        x_nl = np.asarray(x_nl, dtype=np.float64)
        if self.perm is None:
            return x_nl
        out = np.empty(self.n, dtype=np.float64)
        out[self.perm] = x_nl
        return out

    def variant(self, node_lb, node_ub, x0):
        """Build a per-node ``NlProblem`` with tightened *variable* bounds.

        Shares the parsed DAG / AD tapes with ``base`` (cheap); only the bound /
        start vectors are replaced. Inputs are in evaluator order; they are
        permuted into ``.nl`` order for POUNCE.
        """
        return self.base.variant(
            x_l=self.to_nl_order(node_lb),
            x_u=self.to_nl_order(node_ub),
            x0=self.to_nl_order(x0),
        )

    def cleanup(self) -> None:
        for p in self._tmpfiles:
            try:
                os.unlink(p)
            except OSError:
                pass


def _writer_permutation(model) -> tuple[str, np.ndarray]:
    """Emit ``model`` to ``.nl`` text and return the column→evaluator permutation.

    The AMPL ``.nl`` writer reorders variables into canonical nonlinear-first
    order. We recover ``perm[nl_col] = evaluator_flat_index`` from the writer's
    final flat-variable list and the evaluator's declaration-order offsets so
    bounds/solutions can be mapped both ways exactly.
    """
    from discopt._relax.dag_compiler import _compute_var_offset
    from discopt.export.nl import _NLWriter

    writer = _NLWriter(model)
    text = writer.write()  # populates writer._flat_vars in .nl column order
    flat = writer._flat_vars  # list[(Variable, elem)] in .nl column order
    perm = np.empty(len(flat), dtype=np.intp)
    for nl_col, (var, elem) in enumerate(flat):
        perm[nl_col] = _compute_var_offset(var, model) + elem
    return text, perm


def _validate(base, evaluator, perm: Optional[np.ndarray]) -> bool:
    """Confirm the native problem matches the evaluator (objective + gradient).

    Probes a feasible-ish interior point; checks the objective and the full
    gradient (the gradient pins variable ordering AND sign). Both sides are in
    minimization sense, so no sign correction is applied. Returns ``True`` only
    on a tight match.
    """
    n = evaluator.n_variables
    if base.n != n:
        logger.debug("native base rejected: n mismatch (%d vs %d)", base.n, n)
        return False
    lo, hi = evaluator.variable_bounds
    lo = np.where(np.isfinite(lo), lo, -1.0)
    hi = np.where(np.isfinite(hi), hi, 1.0)
    # A deterministic interior probe (not the exact midpoint, which can sit on a
    # symmetry / stationary point that hides an ordering bug).
    x_eval = lo + 0.37 * (hi - lo)

    def nl(v):
        return v if perm is None else np.asarray(v)[perm]

    def to_eval(v_nl):
        if perm is None:
            return np.asarray(v_nl)
        out = np.empty(n, dtype=np.float64)
        out[perm] = np.asarray(v_nl)
        return out

    x_nl = nl(x_eval)
    try:
        o_native = float(base.objective(x_nl))
        g_native = to_eval(base.gradient(x_nl))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("native base rejected: eval failed (%s)", exc)
        return False
    o_eval = float(evaluator.evaluate_objective(x_eval))
    g_eval = np.asarray(evaluator.evaluate_gradient(x_eval), dtype=np.float64)

    o_ok = abs(o_native - o_eval) <= 1e-6 * (1.0 + abs(o_eval))
    g_ok = np.allclose(g_native, g_eval, rtol=1e-6, atol=1e-6)
    if not (o_ok and g_ok):
        logger.debug(
            "native base rejected: obj/grad mismatch (do=%.2e, dg=%.2e)",
            abs(o_native - o_eval),
            float(np.max(np.abs(g_native - g_eval))) if n else 0.0,
        )
    return bool(o_ok and g_ok)


def build_native_base(evaluator, *, validate: bool = True) -> Optional[NativeNlpBase]:
    """Construct a POUNCE-native problem aligned to ``evaluator``, or ``None``.

    Prefers the model's source ``.nl`` (identity variable order). Falls back to
    emitting the model via ``to_nl`` (canonical reorder → recovered permutation).
    Returns ``None`` when POUNCE is unavailable, the model cannot be expressed as
    an ``.nl``, or numeric validation fails — the caller then uses the JAX path.
    """
    if not _POUNCE_OK:
        return None
    model = getattr(evaluator, "_model", None)
    if model is None:
        return None

    base = None
    perm: Optional[np.ndarray] = None
    source = ""

    # Preferred: the original .nl this model was loaded from (identity order).
    src = getattr(model, "_source_nl_path", None)
    if src and os.path.exists(src):
        try:
            base = pounce.read_nl(src)
            perm = None
            source = f"from_nl:{src}"
        except Exception as exc:
            logger.debug("read_nl(%s) failed: %s", src, exc)
            base = None

    # Fallback: emit the in-memory model (Python-API / reformulated) to .nl.
    # read_nl parses the file into owned AD tapes, so the temp file is deleted
    # immediately — the base does not depend on it surviving.
    if base is None:
        try:
            text, perm = _writer_permutation(model)
            fd, path = tempfile.mkstemp(suffix=".nl", prefix="discopt_native_")
            try:
                with os.fdopen(fd, "w") as fh:
                    fh.write(text)
                base = pounce.read_nl(path)
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            source = "to_nl"
        except Exception as exc:
            logger.debug("to_nl native base failed: %s", exc)
            return None

    if validate and not _validate(base, evaluator, perm):
        return None

    return NativeNlpBase(
        base=base,
        n=evaluator.n_variables,
        perm=perm,
        source=source,
    )


# Cross-thread capability of POUNCE's ``NlProblem`` (issue #932).
#
# The base is cached on the *model* and handed to whichever thread solves a node,
# so discopt relies on one ``NlProblem`` being usable — and droppable — from more
# than one thread. POUNCE guarantees that as of pounce#477 ("Let one NlProblem
# serve a whole worker pool"), which removed the pyo3 ``unsendable`` marker after
# establishing that ``NlTnlp`` is ``Send`` and that every ``&self`` method
# evaluates under the GIL.
#
# Against an *older* POUNCE the marker is still present and the guarantee does not
# hold. That failure is not survivable in place: pyo3 raises ``PanicException``,
# which derives from ``BaseException`` and so slips past every ``except Exception``
# in the solver — a cross-thread access would abort the run rather than surface as
# an error, and the *drop* path trips even for code that never used an instance
# cross-thread (whichever thread runs the cyclic GC inherits the object). Both are
# the failures #932 reported.
#
# POUNCE's version string cannot express the requirement (pounce#477 landed inside
# an unreleased ``0.9.0``), so the capability is *measured* once per process, on
# the first base built, and the native path is disabled if it does not hold. The
# probe deliberately performs the access pyo3 would reject, on a real second
# thread, and catches ``BaseException`` there — the panic is the observation, and
# it is reported, never swallowed.
_CROSS_THREAD_OK: Optional[bool] = None
_CROSS_THREAD_LOCK = threading.Lock()


def _probe_cross_thread_use(nb: NativeNlpBase) -> bool:
    """Touch ``nb``'s ``NlProblem`` from a second thread; True if POUNCE allows it."""
    outcome: dict = {}

    def worker():
        outcome["tid"] = threading.get_ident()
        try:
            outcome["value"] = nb.base.n
        except BaseException as exc:  # noqa: BLE001 - a pyo3 PanicException is the signal
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=worker, name="discopt-native-base-probe")
    t.start()
    t.join()

    if outcome.get("tid") == threading.get_ident():  # pragma: no cover - defensive
        raise RuntimeError("cross-thread probe ran on the calling thread; it proves nothing")
    return "error" not in outcome


def _cross_thread_use_supported(nb: NativeNlpBase) -> bool:
    """Establish the cross-thread guarantee once per process (see the note above)."""
    global _CROSS_THREAD_OK
    with _CROSS_THREAD_LOCK:
        if _CROSS_THREAD_OK is None:
            _CROSS_THREAD_OK = _probe_cross_thread_use(nb)
            if not _CROSS_THREAD_OK:
                logger.warning(
                    "POUNCE's NlProblem cannot be used off its creating thread, so the "
                    "native NLP path is disabled for this process (falling back to the "
                    "JAX callback bridge). This POUNCE predates pounce#477; upgrade to a "
                    "build that drops the pyo3 `unsendable` marker. See discopt#932."
                )
    return _CROSS_THREAD_OK


def get_native_base(evaluator) -> Optional[NativeNlpBase]:
    """Return a cached native base for ``evaluator``'s model, building once.

    Cached on the model and invalidated by the same structural fingerprint the
    evaluator cache uses, so a reformulated/edited model rebuilds its base.
    ``False`` is cached for a model that cannot use the native path, so we do not
    repeatedly re-attempt (and re-emit) a failing model.

    The cache is shared across threads, which POUNCE supports as of pounce#477;
    the first base built in a process verifies that (see ``_CROSS_THREAD_OK``) and
    the native path is disabled outright against a POUNCE that does not.
    """
    model = getattr(evaluator, "_model", None)
    if model is None:
        return None
    from discopt.solver import _evaluator_fingerprint

    fp = _evaluator_fingerprint(model)
    cached = getattr(model, "_native_nlp_base_cache", None)
    if cached is not None:
        base, cached_fp = cached
        if cached_fp == fp:
            return base or None  # may be False (sentinel for "unavailable")

    nb = build_native_base(evaluator)
    if nb is not None and not _cross_thread_use_supported(nb):
        nb = None
    model._native_nlp_base_cache = (nb if nb is not None else False, fp)
    return nb


def solve_node_native(nb: NativeNlpBase, x0, node_lb, node_ub, options) -> "object":
    """Solve one node NLP natively; return an ``NLPResult`` (minimization sense)."""
    from discopt.solvers import NLPResult, SolveStatus
    from discopt.solvers.nlp_ipopt import _IPOPT_STATUS_MAP

    x0 = np.asarray(x0, dtype=np.float64)
    node_lb = np.asarray(node_lb, dtype=np.float64)
    node_ub = np.asarray(node_ub, dtype=np.float64)
    warm = np.clip(x0, node_lb, node_ub)

    opts = {"print_level": 0}
    for k in ("max_iter", "tol", "acceptable_tol"):
        if options.get(k) is not None:
            opts[k] = options[k]
    caller_limit = options.get("max_wall_time", 30.0)
    if not caller_limit or caller_limit <= 0:
        caller_limit = 30.0
    opts["max_wall_time"] = min(30.0, caller_limit)

    node = nb.variant(node_lb, node_ub, warm)
    results = pounce.solve_nlp_batch([node], x0s=[nb.to_nl_order(warm)], options=opts)
    x_nl, info = results[0]
    status = _IPOPT_STATUS_MAP.get(info.get("status", -100), SolveStatus.ERROR)
    x_eval = nb.to_eval_order(x_nl)
    obj = info.get("obj_val")
    return NLPResult(
        status=status,
        x=x_eval,
        objective=float(obj) if obj is not None else None,
        multipliers=None,
        iterations=int(info.get("iter_count", 0)),
        wall_time=0.0,
    )
