"""Communication layer — run subproblem solves on a chosen backend (design §13).

One coordinator algorithm, many execution backends. The :class:`CommunicationLayer`
trait abstracts *where* the per-block work runs so the decomposition algorithm is
written once; a backend is a detail chosen at solve time. Phase 6 ships the two
shared-memory backends — a deterministic sequential reference and a thread pool —
mirroring the repo's bit-reproducible Rayon discipline
(``design/rayon-parallelization.md``): the reduce order is fixed regardless of
worker timing, so results never depend on the schedule. Ray / MPI / GPU backends
(design §13.1) slot in behind the same trait later.

Every backend's ``map`` returns results **in input order** — the determinism
guarantee the coordinator relies on (design §13.4).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Protocol, TypeVar, runtime_checkable

_T = TypeVar("_T")
_R = TypeVar("_R")


@runtime_checkable
class CommunicationLayer(Protocol):
    """Maps a function over work items on some execution backend."""

    name: str

    def map(self, items: list[_T], fn: Callable[[_T], _R]) -> list[_R]:
        """Apply *fn* to each item, returning results in input order."""
        ...


class SequentialComm:
    """Run tasks one at a time, in order. The deterministic reference backend."""

    name = "sequential"

    def map(self, items: list[_T], fn: Callable[[_T], _R]) -> list[_R]:
        """Apply *fn* to each item sequentially."""
        return [fn(x) for x in items]


class ThreadPoolComm:
    """Run tasks concurrently on a thread pool; results stay in input order.

    Suited to coarse per-block solves that release the GIL or call out to a
    native solver. ``max_workers=None`` lets the pool size itself.
    """

    name = "threads"

    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers

    def map(self, items: list[_T], fn: Callable[[_T], _R]) -> list[_R]:
        """Apply *fn* concurrently; ``ThreadPoolExecutor.map`` preserves order."""
        items = list(items)
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(fn, items))


def select_backend(backend: "str | CommunicationLayer") -> CommunicationLayer:
    """Resolve a backend name (or pass through a :class:`CommunicationLayer`).

    Names: ``"sequential"``, ``"threads"``. Unknown names raise ``ValueError``.
    """
    if isinstance(backend, str):
        if backend == "sequential":
            return SequentialComm()
        if backend in ("threads", "threadpool", "thread_pool"):
            return ThreadPoolComm()
        raise ValueError(
            f"unknown backend {backend!r}; expected 'sequential' or 'threads', "
            "or a CommunicationLayer instance"
        )
    return backend


__all__ = [
    "CommunicationLayer",
    "SequentialComm",
    "ThreadPoolComm",
    "select_backend",
]
