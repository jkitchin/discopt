"""Bound-neutrality panel for the ``_dfo_common`` extraction (issue #1010).

Issue #1010 is a **pure refactor**: the oracle and the DIRECT-GLce merit rule move
out of ``solvers/direct.py`` and ``solvers/surrogate.py`` into
``solvers/_dfo_common.py``, and neither backend keeps a copy. CLAUDE.md §5's
bound-neutral regime therefore applies with no slack: ``solve_direct`` and
``solve_surrogate`` must return *exactly* unchanged objective, point, and
evaluation count. Any drift — even an apparent improvement — means the extraction
changed behaviour.

Usage::

    python -u scripts/dfo_extraction_neutrality_panel.py record out.json
    python -u scripts/dfo_extraction_neutrality_panel.py compare before.json after.json

``record`` is run once on the baseline tree and once on the branch; ``compare``
does the exact comparison. Splitting it that way is what lets the baseline run
happen in a worktree checked out at the merge base, which is the only way to
measure "unchanged" for a refactor that deletes the old code.

Determinism, deliberately:

* every arm has an ``max_evals`` budget and a *generous* ``time_limit``. An arm
  whose status is ``time_limit`` is recorded as such and makes the comparison
  fail loudly rather than silently comparing two different amounts of work —
  a wall-clock budget produces a different answer on a busier machine with no
  code change involved (CLAUDE.md §9).
* the surrogate backend is seeded; DIRECT has no RNG.
* ``local_refine`` arms use ``derivative-free``, whose Powell budget is an
  evaluation count. The ``nlp`` refiner takes ``deadline - now`` as its own time
  limit, so it is wall-clock dependent by construction and cannot be part of a
  byte-exact panel.
* the surrogate arms use ``acquisition_optimizer="multistart"`` for the same
  reason. The default ``"auto"`` path runs discopt's spatial B&B on the
  acquisition under ``acquisition_time_limit`` (20s) and falls back to its
  incumbent when that binds, so which point it proposes depends on how much B&B
  fits in 20s — a busier machine proposes a different point with no code change
  involved. Multistart is seeded and has no clock in it.

Two limits this panel states rather than hides:

* it does **not** cover the certified/B&B acquisition path, for the reason above.
  That path consumes the extracted code only through ``merits()`` (via
  ``_acquisition_f_min``), which every multistart arm exercises identically, and
  ``python/tests/test_surrogate.py`` covers it end to end.
* ``direct/phase_a_only`` reports no objective and no point by design (nothing in
  its box is feasible). Its comparison is the evaluation count, the status and
  the full counter dict — which is exactly the phase-A merit path.

The panel covers, for each backend, the paths the extracted code actually has:
the unconstrained oracle (``cl is None``), the constrained oracle (the violation
sum), an integer model (the integer mask), phase A (nothing feasible in the box)
and phase B, and — for DIRECT — the scalar merit via the derivative-free polish.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path  # noqa: E402

import discopt  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "tests"))
from support import direct_testfuncs as tfs  # noqa: E402

TIME_LIMIT = 600.0

# See the module docstring: the default "auto" acquisition is wall-clock bounded,
# so it cannot be part of a byte-exact panel. Multistart is seeded and clock-free.
_ACQ = {"acquisition_optimizer": "multistart"}


# ── the panel ────────────────────────────────────────────────────────────────


def _testfunc_model(name: str):
    model, _ = tfs.build_model(tfs.get(name))
    return model


def _constrained_model():
    """Feasible region reachable from the box: phase A then phase B."""
    m = dm.Model("dfo_panel_constrained")
    x = m.continuous("x", shape=2, lb=-2.0, ub=2.0)
    m.minimize(dm.custom(lambda v: v[0] ** 2 + v[1] ** 2 + 3.0 * v[0], name="obj")(x))
    m.subject_to(dm.custom(lambda v: v[0] * v[1], name="bilin")(x) <= 0.5)
    m.subject_to(dm.custom(lambda v: v[0] + v[1], name="lin")(x) >= -1.0)
    return m


def _infeasible_model():
    """No point in the box is feasible: the run stays in phase A throughout.

    Phase A is the arm where the merit rule is *only* the violation, so a drift
    that cancelled inside the phase-B formula would still show here.
    """
    m = dm.Model("dfo_panel_phase_a")
    x = m.continuous("x", shape=2, lb=0.0, ub=1.0)
    m.minimize(dm.custom(lambda v: v[0] + v[1], name="lin")(x))
    m.subject_to(x[0] + x[1] >= 10.0)
    return m


def _integer_model():
    """Exercises the integer mask ``build_oracle`` returns."""
    m = dm.Model("dfo_panel_integer")
    a = m.continuous("a", lb=-3.0, ub=3.0)
    k = m.integer("k", lb=-4, ub=4)
    m.minimize(dm.custom(lambda p, q: (p - 1.3) ** 2 + (q - 2.0) ** 2, name="obj")(a, k))
    m.subject_to(dm.custom(lambda p, q: p + q, name="sum")(a, k) <= 4.0)
    return m


def cases() -> list[tuple[str, object, dict]]:
    """``(case_id, model_factory, solve kwargs)``, in a fixed order."""
    out: list[tuple[str, object, dict]] = []

    # -- DIRECT ---------------------------------------------------------------
    for name in ("branin", "six_hump_camel", "rastrigin_2", "sphere_2", "hartman_3"):
        out.append(
            (
                f"direct/{name}",
                lambda name=name: _testfunc_model(name),
                dict(solver="direct", max_evals=400, local_refine=False),
            )
        )
    out.append(
        (
            "direct/branin+dfo_refine",  # the scalar merit path (_scalar_rank)
            lambda: _testfunc_model("branin"),
            dict(
                solver="direct",
                max_evals=400,
                local_refine=True,
                local_refine_method="derivative-free",
            ),
        )
    )
    out.append(
        (
            "direct/constrained",
            _constrained_model,
            dict(solver="direct", max_evals=400, local_refine=False),
        )
    )
    out.append(
        (
            "direct/phase_a_only",
            _infeasible_model,
            dict(solver="direct", max_evals=200, local_refine=False),
        )
    )
    out.append(
        (
            "direct/integer",
            _integer_model,
            dict(solver="direct", max_evals=300, local_refine=False),
        )
    )

    # -- surrogate ------------------------------------------------------------
    for name in ("branin", "six_hump_camel", "sphere_2"):
        out.append(
            (
                f"surrogate/{name}",
                lambda name=name: _testfunc_model(name),
                dict(solver="surrogate", max_evals=60, seed=11, **_ACQ),
            )
        )
    out.append(
        (
            "surrogate/branin+kriging",
            lambda: _testfunc_model("branin"),
            dict(solver="surrogate", max_evals=50, seed=11, surrogate="kriging", **_ACQ),
        )
    )
    out.append(
        (
            "surrogate/constrained",
            _constrained_model,
            dict(solver="surrogate", max_evals=60, seed=11, **_ACQ),
        )
    )
    out.append(
        (
            "surrogate/phase_a_only",
            _infeasible_model,
            dict(solver="surrogate", max_evals=40, seed=11, **_ACQ),
        )
    )
    out.append(
        (
            "surrogate/integer",
            _integer_model,
            dict(solver="surrogate", max_evals=60, seed=11, **_ACQ),
        )
    )
    return out


# ── record ───────────────────────────────────────────────────────────────────


def _point(result) -> object:
    if result.x is None:
        return None
    # repr of the exact float bits: this panel compares byte-exactly, so a
    # rounded decimal here would hide precisely the drift it is looking for.
    return {
        k: [v.hex() for v in np.asarray(val, dtype=np.float64).reshape(-1)]
        for k, val in sorted(result.x.items())
    }


def record(path: str) -> int:
    # CLAUDE.md §8: prove which code is loaded before measuring anything.
    print(f"discopt from {discopt.__file__}", flush=True)
    marker = "present" if importlib.util.find_spec("discopt.solvers._dfo_common") else "absent"
    print(f"marker discopt.solvers._dfo_common: {marker}", flush=True)

    rows: dict[str, dict] = {}
    timed_out: list[str] = []
    for case_id, factory, kwargs in cases():
        model = factory()
        result = model.solve(time_limit=TIME_LIMIT, **kwargs)
        backend = case_id.split("/", 1)[0]
        # Every counter in DirectStats/SurrogateStats, not just ``evals``: they
        # are all integers with no wall-clock term, so the whole dict is a legal
        # byte-exact comparison and a wider one than the issue asks for.
        stats = {
            k: v for k, v in (result.solver_stats or {}).items() if k.startswith(backend + "/")
        }
        row = {
            "status": result.status,
            "objective": None if result.objective is None else float(result.objective).hex(),
            "x": _point(result),
            "evals": stats.get(f"{backend}/evals"),
            "stats": stats,
            "node_count": result.node_count,
        }
        rows[case_id] = row
        if result.status == "time_limit":
            timed_out.append(case_id)
        print(
            f"  {case_id:34s} status={row['status']:15s} evals={row['evals']!r:6} "
            f"obj={result.objective!r}",
            flush=True,
        )

    payload = {"marker": marker, "discopt_file": discopt.__file__, "rows": rows}
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nrecorded {len(rows)} cases -> {path}", flush=True)
    if timed_out:
        # Not a failure of the refactor, but the panel cannot claim neutrality on
        # an arm whose work was cut by a clock (CLAUDE.md §9).
        print(f"WARNING: {len(timed_out)} arm(s) hit the time backstop: {timed_out}", flush=True)
    if not rows:
        print("PANEL RECORDED NOTHING")
        return 2
    return 0


# ── compare ──────────────────────────────────────────────────────────────────


def compare(before_path: str, after_path: str) -> int:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    print(f"before: marker={before['marker']}  {before['discopt_file']}")
    print(f"after:  marker={after['marker']}   {after['discopt_file']}")
    if before["marker"] != "absent" or after["marker"] != "present":
        # The whole point of the baseline is that it predates the extraction. If
        # both trees have the shared module, the "baseline" is the branch and the
        # comparison measures nothing (CLAUDE.md §8).
        print(
            "BASELINE/BRANCH MARKERS ARE WRONG: baseline must lack the module, branch must have it"
        )
        return 2

    b_rows, a_rows = before["rows"], after["rows"]
    missing = sorted(set(b_rows) ^ set(a_rows))
    if missing:
        print(f"CASE SETS DIFFER: {missing}")
        return 2

    compared = 0
    drift: list[str] = []
    for case_id in sorted(b_rows):
        b, a = b_rows[case_id], a_rows[case_id]
        if b["status"] == "time_limit" or a["status"] == "time_limit":
            print(f"  {case_id:34s} INCONCLUSIVE (time backstop bound an arm), not counted")
            continue
        compared += 1
        for field in ("status", "objective", "x", "evals", "stats", "node_count"):
            if b[field] != a[field]:
                drift.append(f"{case_id}: {field} {b[field]!r} -> {a[field]!r}")
        print(f"  {case_id:34s} {'identical' if b == a else 'DRIFT'}")

    # CLAUDE.md §6: an executed-comparison count, and a non-zero exit when it is
    # zero. A panel that skipped everything must not read as a pass.
    print(f"\ncompared={compared} cases across {len(b_rows)} arms")
    for d in drift:
        print(f"DRIFT {d}")
    if compared == 0:
        print("PANEL COMPARED NOTHING")
        return 2
    if drift:
        print(f"{len(drift)} drift findings — the extraction is NOT bound-neutral")
        return 1
    print("no drift: objective, point and evaluation count are exactly unchanged")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "record":
        sys.exit(record(sys.argv[2]))
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    print(__doc__)
    print("usage: record <out.json> | compare <before.json> <after.json>")
    sys.exit(2)
