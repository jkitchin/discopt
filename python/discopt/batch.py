"""Batch solving: many small independent models, optionally in parallel (#1246).

The motivating workload is a decomposition, not a sweep: discopt-calphad prices
one phase per global solve, and a phase diagram or a parameter fit multiplies
that by the number of (T, x) conditions — tens to thousands of small, mutually
independent global solves. Before this module every such caller had to write its
own process pool, and each one had to rediscover the two facts below.

**Why models are shipped as text, not pickled.** A :class:`~discopt.modeling.core.Model`
that has been solved is not picklable: the solve leaves a live Rust ``PyModelRepr``
and module references on the model and on its variables, so ``pickle.dumps``
raises ``TypeError: cannot pickle 'module' object`` (the same obstacle CLAUDE.md
records for ``copy.deepcopy``). The native round-trippable format (#1240,
:func:`discopt.modeling.dumps` / :func:`discopt.modeling.loads`) has no such
problem, so a worker receives the model's serialized text and rebuilds it.

**Why results come back with their model detached.** :class:`SolveResult` carries
a back-reference to its model for ``result.value(var)`` and ``result.sensitivity``,
which would drag the unpicklable model back into the pipe. The worker clears it
and this module re-attaches the caller's *own* model object on the way out, so a
returned result behaves like one from a local solve.

What a parallel batch does NOT do is make each solve deterministic: a wall-clock
``time_limit`` buys less work when N solves share the machine, so a budget-limited
model can legitimately return a different bound under ``workers=8`` than under
``workers=1``. Models that terminate on work (the gap, ``max_nodes``) are
unaffected — those are the ones the equivalence test uses.
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Any, Iterable, Optional, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import Model, SolveResult

logger = logging.getLogger(__name__)

__all__ = ["solve_batch"]

#: Solve keyword arguments that cannot cross a process boundary. Callbacks close
#: over the caller's objects, so a worker could not run them even if the pickle
#: succeeded; refusing names the fix instead of failing inside the pool.
#:
#: ``warm_start`` is here for the reason this module's docstring gives for a
#: solved ``Model``: a ``SolveResult`` returned by ``Model.solve()`` carries a
#: live back-reference to that model (``result._model``), whose solve left a Rust
#: ``PyModelRepr`` behind, so pickling it raises ``TypeError: cannot pickle
#: 'module' object`` from inside ``concurrent.futures`` — aborting the WHOLE batch
#: with no mention of ``warm_start`` and no guidance (#1316).
_UNSENDABLE_KWARGS = frozenset(
    {"lazy_constraints", "incumbent_callback", "node_callback", "debug", "tuning", "warm_start"}
)

#: Result fields that may hold backend objects with no pickle support. Dropped
#: one at a time, worst-diagnostic-value first, only if the full result refuses
#: to pickle — a returned result missing a diagnostic beats a lost solve.
_DROPPABLE_RESULT_FIELDS = (
    "infeasibility_certificate",
    "mip_nlp_trace",
    "validation_report",
    "mpec_report",
    "solver_stats",
    "_sensitivity",
    "_explanation",
)


def _error_result(message: str) -> "SolveResult":
    """A failed-solve placeholder carrying the reason (see ``SolveResult.error``)."""
    from discopt.modeling.core import SolveResult

    return SolveResult(status="error", error=message)


def _solve_serialized(payload: tuple) -> tuple:
    """Worker entry point: rebuild a model from text, solve it, return the result.

    Runs in a spawned process, so it must be importable at module level and must
    not rely on anything the parent set up in memory.
    """
    text, kwargs, index = payload
    try:
        import discopt.modeling as dm

        model = dm.loads(text)
        result = model.solve(**kwargs)
        assert not isinstance(result, Iterator), "solve_batch refuses stream=True"
        result._model = None  # the model cannot cross the pipe; the parent re-attaches
        blob, dropped = _pickle_result(result)
        return ("ok", blob, dropped, index)
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover - propagate control flow
        raise
    except BaseException as exc:  # noqa: BLE001
        # Deliberately broader than ``Exception``: a Rust ``PanicException`` from
        # the native core derives from ``BaseException`` (CLAUDE.md), and one
        # model's panic must not take the rest of the batch with it. The message
        # is reported, never swallowed.
        return ("error", f"{type(exc).__name__}: {exc}", (), index)


def _pickle_result(result) -> tuple[bytes, tuple]:
    """Pickle ``result``, dropping unpicklable diagnostic fields if it refuses.

    Each refusal is logged with its type and message (#864): a result that comes
    back missing a diagnostic must say why it is missing, or the worker's
    reduction reads as the solver never having produced it.
    """
    try:
        return pickle.dumps(result), ()
    except Exception as exc:  # noqa: BLE001 - reduce and retry, having said why
        logger.debug(
            "solve_batch worker: the full result will not pickle (%s: %s); dropping "
            "diagnostic fields one at a time",
            type(exc).__name__,
            exc,
        )
    dropped: list[str] = []
    for field in _DROPPABLE_RESULT_FIELDS:
        if getattr(result, field, None) is None:
            continue
        setattr(result, field, None)
        dropped.append(field)
        try:
            return pickle.dumps(result), tuple(dropped)
        except Exception as exc:  # noqa: BLE001 - keep dropping, having said why
            logger.debug(
                "solve_batch worker: still unpicklable after dropping %s (%s: %s)",
                field,
                type(exc).__name__,
                exc,
            )
    # Out of things to drop: let the exception name the offender at the caller.
    return pickle.dumps(result), tuple(dropped)


def _solve_locally(model: "Model", kwargs: dict) -> "SolveResult":
    """Solve in this process, capturing a failure as an error result."""
    try:
        return cast("SolveResult", model.solve(**kwargs))
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover
        raise
    except BaseException as exc:  # noqa: BLE001 - see ``_solve_serialized``
        logger.warning("solve_batch: model %r failed: %s", model.name, exc)
        return _error_result(f"{type(exc).__name__}: {exc}")


def _run_pool(payloads: list, workers: int, ctx) -> list:
    """Run the worker calls in a process pool, returning results in input order."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(_solve_serialized, payloads))


def solve_batch(
    models: Iterable["Model"],
    *,
    workers: int = 1,
    start_method: str = "spawn",
    **solve_kwargs: Any,
) -> list["SolveResult"]:
    r"""Solve many independent models and return their results in input order.

    Parameters
    ----------
    models : iterable of Model
        The models to solve. They must be independent: nothing is shared between
        them, and with ``workers > 1`` each is solved on its own deserialized
        copy, so a solve cannot see another's state.
    workers : int, default 1
        Number of worker processes. ``1`` solves in this process, in order, with
        no serialization at all — byte-for-byte what calling ``m.solve()`` on
        each model would do, including the bound tightening a solve writes back
        onto the model. With ``workers > 1`` the solves run in a spawned process
        pool and the *caller's* model objects are left untouched — except for a
        batch of a single model, which is solved in this process whatever
        ``workers`` says, because spawning a pool to run one solve costs more
        than the solve.
    start_method : str, default "spawn"
        Multiprocessing start method. ``"spawn"`` is the default because the
        native core and its solver threads are not fork-safe; change it only if
        you know the whole stack in your process is. Under ``"spawn"`` each
        worker re-imports the calling module, so a *script* calling this with
        ``workers > 1`` must guard its entry point with
        ``if __name__ == "__main__":`` — the standard multiprocessing rule. A
        worker that dies because of a missing guard is reported as such rather
        than as a bare ``BrokenProcessPool``.
    \*\*solve_kwargs
        Passed to :meth:`Model.solve` for every model. Callback and debug
        arguments are refused (they cannot cross a process boundary); pass them
        to individual ``solve()`` calls instead.

    Returns
    -------
    list of SolveResult
        One result per input model, in the same order. A model whose solve
        raised — including one that could not be serialized for a worker —
        yields ``SolveResult(status="error", error=<reason>)`` rather than
        aborting the batch.

    Notes
    -----
    A wall-clock ``time_limit`` is not comparable across worker counts: N solves
    sharing a machine each get less work done per second, so a budget-limited
    model can return a weaker bound under ``workers=8`` than under ``workers=1``.
    Models that terminate on work — the gap, ``max_nodes`` — return identical
    results either way.

    Examples
    --------
    >>> import discopt.modeling as dm
    >>> models = [make_pricing_model(phase) for phase in phases]   # doctest: +SKIP
    >>> results = dm.solve_batch(models, workers=8, time_limit=10)  # doctest: +SKIP
    >>> [r.bound for r in results]                                  # doctest: +SKIP
    """
    from discopt.modeling.core import Model

    model_list = list(models)
    for idx, m in enumerate(model_list):
        if not isinstance(m, Model):
            raise TypeError(f"solve_batch: models[{idx}] is a {type(m).__name__}, expected a Model")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError(f"solve_batch: workers must be a positive int, got {workers!r}")
    if solve_kwargs.get("stream"):
        raise ValueError(
            "solve_batch does not support stream=True: a batch returns one result per model, "
            "not an interleaving of update streams. Call Model.solve(stream=True) per model."
        )

    if workers == 1 or len(model_list) <= 1:
        return [_solve_locally(m, solve_kwargs) for m in model_list]

    unsendable = sorted(
        k for k in _UNSENDABLE_KWARGS.intersection(solve_kwargs) if solve_kwargs[k] is not None
    )
    if unsendable:
        detail = (
            " A warm_start SolveResult holds a live reference to the model it came from, "
            "whose solve left a Rust handle behind, so it cannot be pickled."
            if "warm_start" in unsendable
            else ""
        )
        raise ValueError(
            f"solve_batch(workers={workers}) cannot forward {unsendable} to a worker process: "
            "callbacks and debug handles close over objects that only exist in this process."
            f"{detail} Drop them, or use workers=1."
        )

    import discopt.modeling as dm

    results: list[Optional["SolveResult"]] = [None] * len(model_list)
    payloads: list[tuple] = []
    for idx, m in enumerate(model_list):
        try:
            payloads.append((dm.dumps(m), dict(solve_kwargs), idx))
        except Exception as exc:  # noqa: BLE001 - one model's problem, not the batch's
            logger.warning("solve_batch: model %r could not be serialized: %s", m.name, exc)
            results[idx] = _error_result(
                f"model could not be serialized for a worker process: {type(exc).__name__}: {exc}"
            )

    if payloads:
        ctx = multiprocessing.get_context(start_method)
        try:
            completed = _run_pool(payloads, workers, ctx)
        except BrokenProcessPool as exc:
            # Overwhelmingly this is the missing ``if __name__ == "__main__":``
            # guard: under ``spawn`` every worker re-imports the parent's
            # ``__main__``, so an unguarded script re-runs its own ``solve_batch``
            # inside each child. The pool's own message says none of that.
            raise RuntimeError(
                "solve_batch: a worker process died. If you are calling this from a script, "
                'its entry point must be guarded with `if __name__ == "__main__":` — under the '
                '"spawn" start method each worker re-imports the calling module, and an '
                "unguarded script re-runs the batch inside every worker. Otherwise re-run with "
                f"workers=1 to see the underlying failure. ({exc})"
            ) from exc
        for kind, blob, dropped, idx in completed:
            if kind == "ok":
                result = pickle.loads(blob)
                if dropped:
                    logger.info(
                        "solve_batch: model %r returned without %s (unpicklable "
                        "diagnostics dropped in the worker)",
                        model_list[idx].name,
                        ", ".join(dropped),
                    )
                # Re-attach the caller's model so ``result.value(var)`` and
                # ``result.sensitivity`` work as they do after a local solve.
                result._model = model_list[idx]
                results[idx] = result
            else:
                logger.warning("solve_batch: model %r failed: %s", model_list[idx].name, blob)
                results[idx] = _error_result(str(blob))

    missing = [i for i, r in enumerate(results) if r is None]
    if missing:  # pragma: no cover - the pool returns one result per payload
        raise RuntimeError(f"solve_batch: no result came back for models {missing}")
    return [r for r in results if r is not None]
