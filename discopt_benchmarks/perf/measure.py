"""Per-solve performance measurement (perf plan Stage 0).

Captures the cost-center metrics the plan flagged as missing — **XLA compile
count/seconds** and **time-to-first-incumbent** — alongside the built-in
``jax_time / rust_time / python_time / node_count`` split, so a regression can be
attributed to a cost center without a manual profiling session.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")


@dataclass
class PerfRecord:
    """One panel instance's measured perf + correctness fields."""

    instance: str
    status: str
    objective: float | None
    bound: float | None
    wall_time: float
    node_count: int
    jax_time: float
    rust_time: float
    python_time: float
    subnlp_calls: int
    xla_compile_count: int
    xla_compile_seconds: float
    time_to_first_incumbent: float | None

    @property
    def nodes_per_second(self) -> float | None:
        return self.node_count / self.wall_time if self.wall_time > 0 and self.node_count else None

    @property
    def compiles_per_node(self) -> float | None:
        return self.xla_compile_count / self.node_count if self.node_count else None

    def to_json(self) -> dict:
        return asdict(self)


class _CompileCounter(logging.Handler):
    """Counts XLA compilations and sums their reported compile seconds by parsing
    JAX's ``jax_log_compiles`` records. Attached to the root logger so each record
    is seen exactly once."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.seconds = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg.startswith("Compiling jit("):
            self.count += 1
        elif msg.startswith("Finished XLA compilation of") and " in " in msg:
            # "...in 0.0036 sec"
            with contextlib.suppress(ValueError, IndexError):
                self.seconds += float(msg.rsplit(" in ", 1)[1].split(" sec", 1)[0])


@contextlib.contextmanager
def count_xla_compiles():
    """Context manager yielding a counter with ``.count`` (XLA compilations) and
    ``.seconds`` (summed compile time) over the enclosed block.

    Deterministic for a fixed model/build, unlike wall time — this is the metric
    the perf gate keys on (per node), since it directly reflects the
    recompilation / evaluator-rebuild cost the plan targets.
    """
    import jax

    handler = _CompileCounter()
    root = logging.getLogger()
    prev_level = root.level
    prev_flag = bool(getattr(jax.config, "jax_log_compiles", False))
    jax.config.update("jax_log_compiles", True)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)
        jax.config.update("jax_log_compiles", prev_flag)


def measure_solve(nl_path: str, time_limit: float, gap_tolerance: float = 1e-4) -> PerfRecord:
    """Solve a ``.nl`` instance and return a :class:`PerfRecord`.

    Records time-to-first-incumbent via the solver's ``incumbent_callback`` and
    XLA compile count/seconds via :func:`count_xla_compiles`. The solve itself is
    unmodified — these are pure observers, so the measured run is the production
    run.
    """
    import discopt.modeling as dm

    model = dm.from_nl(nl_path)
    instance = os.path.splitext(os.path.basename(nl_path))[0]

    first_inc: dict[str, float] = {}
    t0 = time.perf_counter()

    def _on_incumbent(ctx, _model, _solution):  # noqa: ANN001
        first_inc.setdefault("t", time.perf_counter() - t0)
        return True

    with count_xla_compiles() as compiles:
        r = model.solve(
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            incumbent_callback=_on_incumbent,
        )
    wall = time.perf_counter() - t0

    return PerfRecord(
        instance=instance,
        status=str(r.status),
        objective=None if r.objective is None else float(r.objective),
        bound=None if r.bound is None else float(r.bound),
        wall_time=wall,
        node_count=int(r.node_count or 0),
        jax_time=float(r.jax_time or 0.0),
        rust_time=float(r.rust_time or 0.0),
        python_time=float(r.python_time or 0.0),
        subnlp_calls=int(getattr(r, "subnlp_calls", 0) or 0),
        xla_compile_count=compiles.count,
        xla_compile_seconds=round(compiles.seconds, 3),
        time_to_first_incumbent=first_inc.get("t"),
    )
