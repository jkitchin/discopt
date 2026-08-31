"""Regression (#1134): a baseline regeneration may not shrink the panel silently.

`cert-baseline.jsonl` is the §0.2.5 bound-neutrality reference. `gen_cert_baseline`
writes only the *deterministic certifying* subset, and three of that filter's four
rejection reasons — not-optimal, near-budget, and (via a wall-clock exit) node
drift — are properties of the **machine and the budget**, not of the search tree.
So a regeneration on a slower box deletes rows rather than recording a regression,
and the smaller reference that results still reads as green while covering less.

#1134 is what that costs: `clay0303hfsg`, `nvs05` and `tanksize` stopped certifying
inside their 60 s budgets, and regenerating would have dropped all three out of the
panel instead of recording it. The same issue also cost a bisect to answer "when did
this reference last match the tree?", because the committed reference carried no
provenance at all.

Pinned here:

* `coverage_loss` names exactly the instances the previous reference covered that a
  new run does not admit;
* `build_meta` records those, each drop's reason, and the previous row's verdict,
  plus the generating commit and host — the provenance whose absence forced the
  bisect;
* `host_speed_ratio` measures this box against the reference machine using only
  instances whose `node_count` reproduced *exactly* (equal trees ⇒ the wall ratio is
  the machines', not the work's), and abstains below a sample floor rather than
  publishing a timing claim from a handful of sub-second solves (CLAUDE.md §9).

The guard these feed — refusing the overwrite without `--allow-shrink` — is asserted
on the plan (`coverage_loss` non-empty) rather than by running a 56-instance solve.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BENCH_ROOT.parent
for _p in (str(_BENCH_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.check_cert_neutrality import (  # noqa: E402
    _MIN_CALIBRATION_SAMPLES,
    host_speed_ratio,
)
from scripts.gen_cert_baseline import build_meta, coverage_loss  # noqa: E402


def _row(name: str, *, nodes: int = 10, wall: float = 1.0, obj: float = 1.0) -> dict:
    return {
        "instance": name,
        "status": "optimal",
        "objective": obj,
        "node_count": nodes,
        "wall_time": wall,
    }


# --------------------------------------------------------------------------- #
# coverage_loss — the quantity the guard is built on
# --------------------------------------------------------------------------- #


def test_coverage_loss_names_the_instances_the_reference_covered():
    """The #1134 shape: three rows the reference had that the new run will not admit."""
    previous = {n: _row(n) for n in ("clay0303hfsg", "nvs05", "tanksize", "nvs14", "gbd")}
    lost = coverage_loss(previous, ["nvs14", "gbd"])
    assert lost == ["clay0303hfsg", "nvs05", "tanksize"], lost


def test_coverage_loss_is_empty_when_the_panel_only_grows():
    previous = {n: _row(n) for n in ("nvs14", "gbd")}
    assert coverage_loss(previous, ["nvs14", "gbd", "tanksize"]) == []


def test_coverage_loss_ignores_a_reordered_panel():
    previous = {n: _row(n) for n in ("gbd", "nvs14")}
    assert coverage_loss(previous, ["nvs14", "gbd"]) == []


# --------------------------------------------------------------------------- #
# build_meta — the committed drop record + provenance
# --------------------------------------------------------------------------- #


def test_meta_records_each_drop_with_its_reason_and_the_lost_row():
    previous = {n: _row(n, nodes=283, obj=41.5) for n in ("clay0303hfsg", "nvs14")}
    meta = build_meta(
        certifying=["nvs14"],
        dropped=[("clay0303hfsg", "not-optimal(time_limit/time_limit/time_limit)")],
        previous=previous,
        time_limit=60.0,
        attempted=56,
        allow_shrink=False,
    )
    assert meta["coverage_lost"] == ["clay0303hfsg"]
    (drop,) = meta["dropped"]
    assert drop["instance"] == "clay0303hfsg"
    assert "time_limit" in drop["reason"]
    # The record must carry what was LOST, not merely that something was.
    assert drop["in_previous_baseline"] is True
    assert drop["previous"] == {"status": "optimal", "node_count": 283, "objective": 41.5}


def test_meta_marks_a_drop_the_reference_never_covered():
    """A never-admitted instance is a drop but not a coverage loss."""
    meta = build_meta(
        certifying=["nvs14"],
        dropped=[("nvs17", "near-limit(wall 45s/60s)")],
        previous={"nvs14": _row("nvs14")},
        time_limit=60.0,
        attempted=56,
        allow_shrink=False,
    )
    assert meta["coverage_lost"] == []
    (drop,) = meta["dropped"]
    assert drop["in_previous_baseline"] is False
    assert drop["previous"] is None


def test_meta_carries_the_provenance_the_bisect_had_to_recover():
    meta = build_meta(
        certifying=["nvs14"],
        dropped=[],
        previous={},
        time_limit=60.0,
        attempted=56,
        allow_shrink=False,
    )
    # This repo is a git checkout, so the generating commit must be recorded: a null
    # here is the exact gap that turned "is the reference stale?" into a bisect.
    assert isinstance(meta["commit"], str) and len(meta["commit"]) == 40, meta["commit"]
    assert meta["generated_at"]
    assert meta["time_limit"] == 60.0
    assert meta["instances_attempted"] == 56
    assert meta["instances_certifying"] == 1
    assert meta["host"]["cpu_count"]


# --------------------------------------------------------------------------- #
# host_speed_ratio — reading a wall-clock verdict against the reference machine
# --------------------------------------------------------------------------- #


def test_speed_ratio_uses_only_instances_that_reproduced_their_node_count():
    """Equal node counts ⇒ the two runs did the same work, so the wall ratio is the
    machines'. A row whose tree changed carries a work difference and must not
    contaminate the estimate."""
    baseline = {f"i{k}": _row(f"i{k}", nodes=k + 1, wall=1.0) for k in range(6)}
    new = {f"i{k}": _row(f"i{k}", nodes=k + 1, wall=3.0) for k in range(6)}
    # One instance whose tree exploded: 100x the wall, and it must be excluded.
    new["i0"] = _row("i0", nodes=999, wall=100.0)
    ratio, n = host_speed_ratio(new, baseline)
    assert n == 5, n
    assert abs(ratio - 3.0) < 1e-9, ratio


def test_speed_ratio_abstains_below_the_sample_floor():
    """Fewer samples than the floor ⇒ no ratio at all, rather than a timing claim
    published off a handful of solves (CLAUDE.md §9)."""
    n_rows = _MIN_CALIBRATION_SAMPLES - 1
    baseline = {f"i{k}": _row(f"i{k}", nodes=k + 1, wall=1.0) for k in range(n_rows)}
    new = {f"i{k}": _row(f"i{k}", nodes=k + 1, wall=3.0) for k in range(n_rows)}
    ratio, n = host_speed_ratio(new, baseline)
    assert ratio is None
    assert n == n_rows


def test_speed_ratio_drops_sub_noise_baseline_walls():
    """A 0.01 s baseline row is process noise, not throughput."""
    baseline = {f"i{k}": _row(f"i{k}", nodes=k + 1, wall=0.01) for k in range(6)}
    new = {f"i{k}": _row(f"i{k}", nodes=k + 1, wall=3.0) for k in range(6)}
    ratio, n = host_speed_ratio(new, baseline)
    assert ratio is None, ratio
    assert n == 0


def test_speed_ratio_of_the_reference_machine_against_itself_is_one():
    baseline = {f"i{k}": _row(f"i{k}", nodes=k + 1, wall=1.0 + k) for k in range(6)}
    ratio, n = host_speed_ratio(dict(baseline), baseline)
    assert n == 6
    assert abs(ratio - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# The routing note the panel could not see (#1134, Cause 2)
# --------------------------------------------------------------------------- #


def test_solve_result_round_trips_the_routing_note():
    """`algorithm_route` survives `to_dict`/`from_dict`.

    #1134's second cause — `cvxnonsep_nsig30` going 1.12 s → 42.5 s on an
    *identical* 165-node tree because the auto-routed `oa` master burned 39.6 s and
    abstained before the default path solved it — was invisible in a table of nodes
    and seconds, and cost a second bisect to recover something the solver already
    knew and was already reporting on `SolveResult.algorithm_route`.
    """
    from benchmarks.metrics import SolveResult, SolveStatus  # noqa: PLC0415

    note = "mip-nlp/oa: minlp certified convex at the root; did not certify in 39.63s"
    r = SolveResult(
        instance="cvxnonsep_nsig30",
        solver="discopt",
        status=SolveStatus.OPTIMAL,
        algorithm_route=note,
    )
    d = r.to_dict()
    assert d["algorithm_route"] == note
    assert SolveResult.from_dict(d).algorithm_route == note


def test_rows_written_before_the_field_still_load():
    """Backward compatibility, and the reason the field is optional: every row of
    the committed `cert-baseline.jsonl` predates it, and `load`/`from_dict` must
    not start rejecting the reference the neutrality check is built on."""
    from benchmarks.metrics import SolveResult  # noqa: PLC0415
    from scripts.check_cert_neutrality import _CERT_BASELINE  # noqa: PLC0415
    from utils.cert_neutrality import load_baseline  # noqa: PLC0415

    baseline = load_baseline(_CERT_BASELINE)
    assert baseline, "committed cert baseline is empty"
    loaded = 0
    for row in baseline.values():
        assert "algorithm_route" not in row, "test premise: committed rows predate the field"
        assert SolveResult.from_dict(row).algorithm_route is None
        loaded += 1
    assert loaded == len(baseline), (loaded, len(baseline))
