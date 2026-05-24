"""Local web dashboard for browsing discopt benchmark runs.

Discovers runs under ``reports/``, joins per-instance results against MINLPLib
reference data, and serves a single-page UI for live progress, per-instance
drill-down, and side-by-side solver comparison.

Usage:
    python -m discopt_benchmarks.dashboard
    python -m discopt_benchmarks.dashboard --reports-dir reports --port 8765
    python -m discopt_benchmarks.dashboard --open  # also opens a browser tab

Zero extra dependencies: pure stdlib http.server + an embedded vanilla-JS SPA.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_BENCH_ROOT = Path(__file__).resolve().parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from utils.minlplib_data import (  # noqa: E402
    InstanceMeta,
    OUTCOME_INCORRECT,
    OUTCOME_OPTIMAL,
    OUTCOME_FEASIBLE,
    OUTCOME_TIMEOUT,
    OUTCOME_ERROR,
    OUTCOME_INFEASIBLE,
    OUTCOME_UNKNOWN,
    load_instance_data,
)

# ── Run discovery ───────────────────────────────────────────────────────────


def _iter_run_paths(reports_dir: Path) -> list[Path]:
    """A "run" is either: a subdir with results.json/instances/, or a
    top-level *.json results file (old-style runs)."""
    out: list[Path] = []
    if not reports_dir.exists():
        return out
    for p in sorted(reports_dir.iterdir()):
        if p.is_dir():
            if (p / "results.json").exists() or (p / "instances").exists():
                out.append(p)
        elif p.suffix == ".json" and not p.name.startswith("phase") and "history" not in p.name:
            # Top-level run file (smoke_2026..., global_opt_2026...).
            out.append(p)
    return out


def _solver_results_from_results_json(path: Path) -> dict[str, list[dict]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("solver_results") or {}


def _solver_results_from_instances_dir(idir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if not idir.exists():
        return out
    for f in sorted(idir.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        solver = r.get("solver", "discopt")
        out.setdefault(solver, []).append(r)
    return out


def _load_run(path: Path) -> dict[str, Any]:
    """Return {id, suite, source, last_modified, solver_results}."""
    if path.is_dir():
        rid = path.name
        results_json = path / "results.json"
        instances_dir = path / "instances"

        solver_results: dict[str, list[dict]] = {}
        if results_json.exists():
            solver_results = _solver_results_from_results_json(results_json)
            source = "results.json"
            stamp = results_json.stat().st_mtime
        else:
            source = "live"
            stamp = path.stat().st_mtime

        # Always overlay per-instance files if present so live runs work and
        # results.json files get refreshed mid-run.
        live = _solver_results_from_instances_dir(instances_dir)
        for solver, recs in live.items():
            seen = {r["instance"] for r in solver_results.get(solver, [])}
            solver_results.setdefault(solver, [])
            for r in recs:
                if r["instance"] not in seen:
                    solver_results[solver].append(r)
        if live and instances_dir.exists():
            stamp = max(stamp, instances_dir.stat().st_mtime)

        # Suite name: heuristic — strip trailing _demo/_compare/_<timestamp>
        suite = rid.split("_")[0] if "_" in rid else rid
        return {
            "id": rid,
            "suite": suite,
            "source": source,
            "last_modified": stamp,
            "solver_results": solver_results,
        }
    else:
        # Top-level *.json
        data = {}
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        return {
            "id": path.stem,
            "suite": data.get("suite", path.stem.split("_")[0]),
            "source": str(path.name),
            "last_modified": path.stat().st_mtime,
            "solver_results": data.get("solver_results", {}),
        }


# ── Scoring + summary ───────────────────────────────────────────────────────

_STATUS_TO_OUTCOME = {
    "optimal": OUTCOME_OPTIMAL,
    "OPTIMAL": OUTCOME_OPTIMAL,
    "feasible": OUTCOME_FEASIBLE,
    "FEASIBLE": OUTCOME_FEASIBLE,
    "time_limit": OUTCOME_TIMEOUT,
    "TIME_LIMIT": OUTCOME_TIMEOUT,
    "timeout": OUTCOME_TIMEOUT,
    "memory_limit": OUTCOME_TIMEOUT,
    "infeasible": OUTCOME_INFEASIBLE,
    "INFEASIBLE": OUTCOME_INFEASIBLE,
    "error": OUTCOME_ERROR,
    "ERROR": OUTCOME_ERROR,
    "numerical_error": OUTCOME_ERROR,
    "unknown": OUTCOME_UNKNOWN,
}


def _outcome(rec: dict, meta: InstanceMeta | None) -> str:
    status = rec.get("status", "unknown")
    base = _STATUS_TO_OUTCOME.get(status, OUTCOME_UNKNOWN)
    if base == OUTCOME_OPTIMAL and meta is not None:
        ref = meta.known_optimum
        obj = rec.get("objective")
        if ref is not None and obj is not None:
            tol = 1e-4 + 1e-3 * abs(ref)
            if abs(obj - ref) > tol:
                return OUTCOME_INCORRECT
    return base


def _sgm_time(times: list[float], shift: float = 1.0) -> float:
    """Shifted geometric mean — the standard MINLP benchmarking metric."""
    if not times:
        return float("nan")
    log_sum = 0.0
    n = 0
    for t in times:
        if t is None or math.isnan(t):
            continue
        log_sum += math.log(max(t, 0.0) + shift)
        n += 1
    if n == 0:
        return float("nan")
    return math.exp(log_sum / n) - shift


def _summarize_solver(records: list[dict], index: dict[str, InstanceMeta]) -> dict:
    by_outcome: dict[str, int] = {}
    by_cell: dict[str, dict] = {}
    times_solved: list[float] = []
    incorrect: list[str] = []
    for r in records:
        name = r["instance"]
        meta = index.get(name)
        out = _outcome(r, meta)
        by_outcome[out] = by_outcome.get(out, 0) + 1
        if out == OUTCOME_INCORRECT:
            incorrect.append(name)
        cat = meta.category_bucket if meta else "unknown"
        size = meta.size_bucket if meta else "?"
        cell_key = f"{cat} {size}"
        cell = by_cell.setdefault(cell_key, {"category": cat, "size": size, "total": 0})
        cell["total"] += 1
        cell[out] = cell.get(out, 0) + 1
        if out in (OUTCOME_OPTIMAL, OUTCOME_FEASIBLE) and r.get("wall_time") is not None:
            times_solved.append(float(r["wall_time"]))

    return {
        "total": len(records),
        "by_outcome": by_outcome,
        "by_cell": sorted(by_cell.values(), key=lambda c: (c["category"], c["size"])),
        "sgm_time": _sgm_time(times_solved),
        "median_time": statistics.median(times_solved) if times_solved else None,
        "incorrect": incorrect,
    }


# ── HTTP handler ────────────────────────────────────────────────────────────


class _Dashboard(BaseHTTPRequestHandler):
    server_version = "discopt-dashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter
        return

    # ----- helpers
    def _json(self, payload: Any, code: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str) -> None:
        data = body.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _err(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code=code)

    # ----- routing
    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        qs = urllib.parse.parse_qs(url.query)
        ctx: _Ctx = self.server.ctx  # type: ignore[attr-defined]

        try:
            if not parts:
                return self._html(_INDEX_HTML)

            if parts == ["api", "runs"]:
                return self._json(self._list_runs(ctx))

            if len(parts) == 3 and parts[:2] == ["api", "run"]:
                # GET /api/run/<run_id>
                run_id = urllib.parse.unquote(parts[2])
                run = self._get_run(ctx, run_id)
                if run is None:
                    return self._err(404, f"unknown run {run_id!r}")
                return self._json(self._summarize_run(ctx, run))

            if (len(parts) == 4 and parts[:2] == ["api", "run"]
                    and parts[3] == "instances"):
                run_id = urllib.parse.unquote(parts[2])
                run = self._get_run(ctx, run_id)
                if run is None:
                    return self._err(404, f"unknown run {run_id!r}")
                return self._json(self._instances_table(ctx, run))

            if (len(parts) == 4 and parts[:2] == ["api", "run"]
                    and parts[3] == "compare"):
                run_id = urllib.parse.unquote(parts[2])
                run = self._get_run(ctx, run_id)
                if run is None:
                    return self._err(404, f"unknown run {run_id!r}")
                solvers_q = qs.get("solvers", [""])[0]
                solvers = [s for s in solvers_q.split(",") if s] or list(run["solver_results"])
                return self._json(self._compare(ctx, run, solvers))

            if parts == ["api", "history"]:
                suite = qs.get("suite", [""])[0]
                return self._json(self._history(ctx, suite))

            return self._err(404, f"no route for {url.path!r}")
        except Exception as exc:  # noqa: BLE001
            return self._err(500, f"{type(exc).__name__}: {exc}")

    # ----- API logic
    @staticmethod
    def _list_runs(ctx: "_Ctx") -> list[dict]:
        runs = []
        for p in _iter_run_paths(ctx.reports_dir):
            r = _load_run(p)
            solvers = list(r["solver_results"])
            total_records = sum(len(v) for v in r["solver_results"].values())
            runs.append({
                "id": r["id"],
                "suite": r["suite"],
                "source": r["source"],
                "last_modified": r["last_modified"],
                "solvers": solvers,
                "total_records": total_records,
            })
        runs.sort(key=lambda x: x["last_modified"], reverse=True)
        return runs

    @staticmethod
    def _get_run(ctx: "_Ctx", run_id: str) -> dict | None:
        for p in _iter_run_paths(ctx.reports_dir):
            if p.is_dir() and p.name == run_id:
                return _load_run(p)
            if p.is_file() and p.stem == run_id:
                return _load_run(p)
        return None

    @staticmethod
    def _summarize_run(ctx: "_Ctx", run: dict) -> dict:
        per_solver: dict[str, dict] = {}
        for solver, recs in run["solver_results"].items():
            per_solver[solver] = _summarize_solver(recs, ctx.index)
        return {
            "id": run["id"],
            "suite": run["suite"],
            "source": run["source"],
            "last_modified": run["last_modified"],
            "solvers": per_solver,
        }

    @staticmethod
    def _instances_table(ctx: "_Ctx", run: dict) -> dict:
        rows: list[dict] = []
        for solver, recs in run["solver_results"].items():
            for r in recs:
                meta = ctx.index.get(r["instance"])
                rows.append({
                    "instance": r["instance"],
                    "solver": solver,
                    "category": meta.category_bucket if meta else None,
                    "size": meta.size_bucket if meta else None,
                    "n_vars": meta.n_vars if meta else None,
                    "n_constraints": meta.n_constraints if meta else None,
                    "status": r.get("status"),
                    "outcome": _outcome(r, meta),
                    "objective": r.get("objective"),
                    "known_optimum": meta.known_optimum if meta else None,
                    "abs_gap_vs_ref": (
                        abs(r["objective"] - meta.known_optimum)
                        if meta and meta.known_optimum is not None
                        and r.get("objective") is not None
                        else None
                    ),
                    "wall_time": r.get("wall_time"),
                    "node_count": r.get("node_count"),
                })
        return {"rows": rows}

    @staticmethod
    def _compare(ctx: "_Ctx", run: dict, solvers: list[str]) -> dict:
        by_instance: dict[str, dict] = {}
        for solver in solvers:
            for r in run["solver_results"].get(solver, []):
                name = r["instance"]
                meta = ctx.index.get(name)
                cell = by_instance.setdefault(name, {
                    "instance": name,
                    "category": meta.category_bucket if meta else None,
                    "size": meta.size_bucket if meta else None,
                    "known_optimum": meta.known_optimum if meta else None,
                    "by_solver": {},
                })
                cell["by_solver"][solver] = {
                    "outcome": _outcome(r, meta),
                    "objective": r.get("objective"),
                    "wall_time": r.get("wall_time"),
                    "node_count": r.get("node_count"),
                }
        # Speedup vs first solver in list (the reference column).
        ref_solver = solvers[0] if solvers else None
        for cell in by_instance.values():
            ref = cell["by_solver"].get(ref_solver, {}) if ref_solver else {}
            ref_t = ref.get("wall_time")
            for solver, payload in cell["by_solver"].items():
                t = payload.get("wall_time")
                if (ref_t is not None and t is not None
                        and ref_t > 1e-9 and t > 1e-9
                        and solver != ref_solver):
                    payload["ratio_vs_ref"] = t / ref_t
        return {
            "solvers": solvers,
            "ref_solver": ref_solver,
            "rows": sorted(by_instance.values(), key=lambda c: c["instance"]),
        }

    @staticmethod
    def _history(ctx: "_Ctx", suite: str) -> dict:
        path = ctx.reports_dir / "history" / f"{suite}_history.jsonl"
        out: list[dict] = []
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                # MINLPLib's "Infinity"/"NaN" tokens are valid in Python's
                # json.loads, so we tolerate them.
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return {"suite": suite, "entries": out}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    raise TypeError(f"{type(obj).__name__} not serializable")


class _Ctx:
    def __init__(self, reports_dir: Path, index: dict[str, InstanceMeta]) -> None:
        self.reports_dir = reports_dir
        self.index = index


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, ctx: _Ctx) -> None:  # noqa: ANN001
        super().__init__(addr, handler)
        self.ctx = ctx


# ── Embedded SPA ────────────────────────────────────────────────────────────

_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>discopt benchmark explorer</title>
<style>
  :root {
    --bg: #0f1115; --panel: #161a22; --line: #262b36; --fg: #d8dee9;
    --muted: #7b8494; --accent: #5fa8d3; --good: #79c08c; --bad: #e06c75;
    --warn: #d19a66; --neutral: #5c6370;
  }
  body { margin: 0; font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
         color: var(--fg); background: var(--bg); }
  header { padding: 10px 16px; border-bottom: 1px solid var(--line);
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  header .right { margin-left: auto; display: flex; gap: 12px; align-items: center;
                  color: var(--muted); font-size: 12px; }
  #app { display: grid; grid-template-columns: 280px 1fr; height: calc(100vh - 41px); }
  #sidebar { border-right: 1px solid var(--line); overflow-y: auto;
             background: var(--panel); }
  #sidebar h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
                color: var(--muted); padding: 12px 14px 6px; margin: 0; }
  .run { padding: 8px 14px; border-bottom: 1px solid var(--line); cursor: pointer; }
  .run:hover { background: #1c2230; }
  .run.active { background: #1e2a3a; border-left: 3px solid var(--accent);
                padding-left: 11px; }
  .run .name { font-weight: 600; }
  .run .meta { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .live { color: var(--good); }
  main { padding: 14px 18px; overflow-y: auto; }
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--line);
          margin-bottom: 12px; }
  .tab { padding: 6px 12px; cursor: pointer; border-bottom: 2px solid transparent;
         color: var(--muted); }
  .tab.active { color: var(--fg); border-bottom-color: var(--accent); }
  .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          margin: 12px 0; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
          padding: 10px 12px; }
  .card .label { color: var(--muted); font-size: 11px; text-transform: uppercase;
                 letter-spacing: 0.06em; }
  .card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .card .sub { color: var(--muted); font-size: 11px; margin-top: 2px; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 8px; }
  th, td { border-bottom: 1px solid var(--line); padding: 4px 8px; text-align: left;
           vertical-align: top; }
  th { color: var(--muted); font-weight: 500; cursor: pointer; user-select: none; }
  th.sort-asc::after  { content: " ▲"; color: var(--accent); }
  th.sort-desc::after { content: " ▼"; color: var(--accent); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 1px 6px; border-radius: 9px;
          font-size: 11px; font-weight: 600; }
  .pill.optimal_proven { background: #234836; color: var(--good); }
  .pill.feasible_only  { background: #3a3320; color: var(--warn); }
  .pill.timeout        { background: #2c333d; color: var(--muted); }
  .pill.error          { background: #4a2424; color: var(--bad); }
  .pill.infeasible     { background: #2c333d; color: var(--muted); }
  .pill.incorrect      { background: #5a1e1e; color: #ffb4b4; }
  .pill.unknown        { background: #2c333d; color: var(--muted); }
  .solver-tag { display: inline-block; padding: 1px 6px; border-radius: 9px;
                background: #1e2a3a; color: var(--accent); font-size: 11px;
                margin-right: 4px; }
  .muted { color: var(--muted); }
  .controls { display: flex; gap: 10px; align-items: center; margin: 8px 0; }
  input[type=search] { background: #0b0e13; color: var(--fg);
    border: 1px solid var(--line); border-radius: 4px; padding: 4px 8px;
    font-size: 12px; min-width: 200px; }
  label { display: inline-flex; gap: 4px; align-items: center; color: var(--muted);
          font-size: 12px; }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
  pre { background: #0b0e13; border: 1px solid var(--line); border-radius: 4px;
        padding: 10px; overflow-x: auto; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>discopt · benchmark explorer</h1>
  <span class="muted" id="run-count"></span>
  <div class="right">
    <label><input type="checkbox" id="auto" checked> auto-refresh (3s)</label>
    <span id="status"></span>
  </div>
</header>
<div id="app">
  <aside id="sidebar"><h2>Runs</h2><div id="runs"></div></aside>
  <main id="main"><div class="empty">Select a run from the left.</div></main>
</div>

<script>
const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => [...root.querySelectorAll(sel)];

const state = {
  runs: [],
  selectedId: null,
  tab: "summary",
  data: null,         // /api/run/<id>
  rows: null,         // /api/run/<id>/instances
  compare: null,      // /api/run/<id>/compare
  history: null,      // /api/history
  sortKey: null,
  sortDir: 1,
  filter: "",
  selectedSolvers: null,
};

function fmt(v, d=2) {
  if (v == null || Number.isNaN(v)) return "—";
  if (typeof v === "number") return v.toFixed(d);
  return v;
}
function fmtTime(t) {
  if (!t) return "—";
  const d = new Date(t * 1000);
  const now = Date.now() / 1000;
  const dt = now - t;
  if (dt < 60) return Math.floor(dt) + "s ago";
  if (dt < 3600) return Math.floor(dt/60) + "m ago";
  if (dt < 86400) return Math.floor(dt/3600) + "h ago";
  return d.toLocaleString();
}
function pill(outcome) {
  return `<span class="pill ${outcome}">${(outcome||"").replace("_"," ")}</span>`;
}

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

async function loadRuns() {
  state.runs = await api("/api/runs");
  $("#run-count").textContent = `${state.runs.length} runs`;
  $("#runs").innerHTML = state.runs.map(r => `
    <div class="run ${r.id===state.selectedId?"active":""}" data-id="${r.id}">
      <div class="name">${r.id}</div>
      <div class="meta">
        ${r.solvers.join(", ") || "no solvers"} ·
        ${r.total_records} rec ·
        <span class="${r.source==="live"?"live":""}">${fmtTime(r.last_modified)}</span>
      </div>
    </div>`).join("");
  $$(".run").forEach(el => el.onclick = () => selectRun(el.dataset.id));
}

async function selectRun(id) {
  state.selectedId = id;
  state.selectedSolvers = null;
  await refresh();
  await loadRuns();   // re-highlight active
}

async function refresh() {
  if (!state.selectedId) return;
  $("#status").textContent = "loading…";
  try {
    state.data = await api(`/api/run/${encodeURIComponent(state.selectedId)}`);
    if (state.tab === "instances")
      state.rows = await api(`/api/run/${encodeURIComponent(state.selectedId)}/instances`);
    if (state.tab === "compare") {
      const ss = (state.selectedSolvers || Object.keys(state.data.solvers)).join(",");
      state.compare = await api(
        `/api/run/${encodeURIComponent(state.selectedId)}/compare?solvers=${ss}`);
    }
    if (state.tab === "history")
      state.history = await api(`/api/history?suite=${encodeURIComponent(state.data.suite)}`);
    render();
    $("#status").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (e) {
    $("#status").textContent = "ERR " + e.message;
  }
}

function render() {
  const d = state.data;
  if (!d) { $("#main").innerHTML = `<div class="empty">Select a run.</div>`; return; }
  const tabs = ["summary","instances","compare","history"];
  $("#main").innerHTML = `
    <h2 style="margin:0 0 4px 0; font-size:18px;">${d.id}</h2>
    <div class="muted">
      suite: <b>${d.suite}</b> · solvers: ${Object.keys(d.solvers).map(s=>`<span class="solver-tag">${s}</span>`).join("")}
      · source: ${d.source} · ${fmtTime(d.last_modified)}
    </div>
    <div class="tabs">
      ${tabs.map(t => `<div class="tab ${t===state.tab?"active":""}" data-tab="${t}">${t}</div>`).join("")}
    </div>
    <div id="tab-body"></div>
  `;
  $$(".tab").forEach(t => t.onclick = () => { state.tab = t.dataset.tab; refresh(); });
  renderTab();
}

function renderTab() {
  const body = $("#tab-body");
  if (state.tab === "summary") return renderSummary(body);
  if (state.tab === "instances") return renderInstances(body);
  if (state.tab === "compare")  return renderCompare(body);
  if (state.tab === "history")  return renderHistory(body);
}

function renderSummary(root) {
  const solvers = Object.entries(state.data.solvers);
  if (!solvers.length) { root.innerHTML = `<div class="empty">no solvers</div>`; return; }
  root.innerHTML = solvers.map(([name, s]) => {
    const o = s.by_outcome;
    const card = (lbl, val, sub="", cls="") =>
      `<div class="card"><div class="label">${lbl}</div>
        <div class="value ${cls}">${val}</div>
        ${sub?`<div class="sub">${sub}</div>`:""}</div>`;
    const incList = s.incorrect.length
      ? `<div style="margin-top:8px; color: var(--bad);">
           ⚠ incorrect: ${s.incorrect.join(", ")}</div>`
      : "";
    const cellRows = s.by_cell.map(c => `
      <tr><td>${c.category}</td><td>${c.size}</td><td class="num">${c.total}</td>
        <td class="num">${c.optimal_proven||0}</td>
        <td class="num">${c.feasible_only||0}</td>
        <td class="num">${c.timeout||0}</td>
        <td class="num">${c.error||0}</td>
        <td class="num" style="color:${c.incorrect?'var(--bad)':'inherit'}">
          ${c.incorrect||0}</td></tr>`).join("");
    return `
      <h3 style="margin:18px 0 4px 0;">Solver: <span class="solver-tag">${name}</span></h3>
      <div class="grid">
        ${card("instances", s.total)}
        ${card("optimal", o.optimal_proven||0, "ref-validated", "good")}
        ${card("feasible", o.feasible_only||0)}
        ${card("timeout", o.timeout||0)}
        ${card("error", o.error||0)}
        ${card("incorrect", o.incorrect||0, "must be 0",
               (o.incorrect||0)>0?"bad":"")}
        ${card("SGM time (s)", fmt(s.sgm_time, 2))}
        ${card("median time (s)", fmt(s.median_time, 2))}
      </div>
      ${incList}
      <table><thead><tr>
        <th>class</th><th>size</th><th>#inst</th><th>#opt</th><th>#feas</th>
        <th>#timeout</th><th>#error</th><th>#incorrect</th>
      </tr></thead><tbody>${cellRows}</tbody></table>
    `;
  }).join("");
  for (const el of $$(".value.good")) el.style.color = "var(--good)";
  for (const el of $$(".value.bad"))  el.style.color = "var(--bad)";
}

function renderInstances(root) {
  if (!state.rows) { root.innerHTML = "<div class='empty'>loading…</div>"; return; }
  root.innerHTML = `
    <div class="controls">
      <input type="search" id="flt" placeholder="filter by instance/solver/category"
        value="${state.filter}">
      <span class="muted">${state.rows.rows.length} rows</span>
    </div>
    <table><thead><tr id="hdr"></tr></thead><tbody id="tb"></tbody></table>
  `;
  const cols = [
    {k:"instance", lbl:"instance"},
    {k:"solver", lbl:"solver"},
    {k:"category", lbl:"class"},
    {k:"size", lbl:"size"},
    {k:"n_vars", lbl:"#vars", num:true},
    {k:"outcome", lbl:"outcome"},
    {k:"wall_time", lbl:"time (s)", num:true, f:v=>fmt(v,2)},
    {k:"node_count", lbl:"nodes", num:true, f:v=>fmt(v,0)},
    {k:"objective", lbl:"objective", num:true, f:v=>fmt(v,6)},
    {k:"known_optimum", lbl:"ref", num:true, f:v=>fmt(v,6)},
    {k:"abs_gap_vs_ref", lbl:"|gap|", num:true, f:v=>fmt(v,2)},
  ];
  $("#hdr").innerHTML = cols.map(c => {
    let cls = c.num?"num":"";
    if (state.sortKey===c.k) cls += state.sortDir>0?" sort-asc":" sort-desc";
    return `<th class="${cls}" data-k="${c.k}">${c.lbl}</th>`;
  }).join("");
  $$("#hdr th").forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    if (state.sortKey === k) state.sortDir *= -1;
    else { state.sortKey = k; state.sortDir = 1; }
    renderInstances(root);
  });
  $("#flt").oninput = e => { state.filter = e.target.value.toLowerCase();
    renderInstances(root); };

  let rows = state.rows.rows.slice();
  if (state.filter) {
    rows = rows.filter(r =>
      `${r.instance} ${r.solver} ${r.category} ${r.outcome}`
        .toLowerCase().includes(state.filter));
  }
  if (state.sortKey) {
    rows.sort((a,b) => {
      const av = a[state.sortKey], bv = b[state.sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return av > bv ? state.sortDir : av < bv ? -state.sortDir : 0;
    });
  }
  $("#tb").innerHTML = rows.map(r =>
    `<tr>${cols.map(c => {
      let v = r[c.k];
      if (c.k === "outcome") v = pill(v);
      else if (c.f) v = c.f(v);
      else if (v == null) v = "—";
      return `<td class="${c.num?"num":""}">${v}</td>`;
    }).join("")}</tr>`).join("");
}

function renderCompare(root) {
  const c = state.compare;
  if (!c) { root.innerHTML = "<div class='empty'>loading…</div>"; return; }
  const allSolvers = Object.keys(state.data.solvers);
  if (allSolvers.length < 2) {
    root.innerHTML = `<div class="empty">need ≥2 solvers in this run to compare<br>
      <span class="muted">re-run with --solvers a,b,...</span></div>`;
    return;
  }
  const picks = allSolvers.map(s => `
    <label><input type="checkbox" data-s="${s}"
      ${(state.selectedSolvers||allSolvers).includes(s)?"checked":""}>${s}</label>`).join("");
  root.innerHTML = `
    <div class="controls">solvers: ${picks}
      <span class="muted">first solver = reference for ratio</span></div>
    <table><thead><tr>
      <th>instance</th><th>class</th><th>size</th>
      ${c.solvers.map(s => `<th colspan="3">${s}</th>`).join("")}
    </tr><tr>
      <th></th><th></th><th></th>
      ${c.solvers.map(() => `<th>outcome</th><th class="num">time</th><th class="num">vs ref</th>`).join("")}
    </tr></thead><tbody>${
      c.rows.map(r => `<tr>
        <td>${r.instance}</td><td>${r.category||"—"}</td><td>${r.size||"—"}</td>
        ${c.solvers.map(s => {
          const x = r.by_solver[s] || {};
          return `<td>${x.outcome?pill(x.outcome):"—"}</td>
                  <td class="num">${fmt(x.wall_time,2)}</td>
                  <td class="num">${s===c.ref_solver?"<span class='muted'>ref</span>":fmt(x.ratio_vs_ref,2)}</td>`;
        }).join("")}
      </tr>`).join("")}</tbody></table>`;
  $$("input[data-s]").forEach(cb => cb.onchange = () => {
    state.selectedSolvers = $$("input[data-s]:checked").map(x => x.dataset.s);
    refresh();
  });
}

function renderHistory(root) {
  const h = state.history;
  if (!h) { root.innerHTML = "<div class='empty'>loading…</div>"; return; }
  if (!h.entries.length) {
    root.innerHTML = `<div class="empty">no history for suite "${h.suite}"</div>`; return;
  }
  root.innerHTML = `<table><thead><tr>
    <th>timestamp</th><th>git</th><th class="num">total</th><th class="num">solved</th>
    <th class="num">mean time</th><th class="num">median gap</th>
  </tr></thead><tbody>${
    h.entries.slice().reverse().map(e => `<tr>
      <td>${e.timestamp||"—"}</td>
      <td><code>${(e.git_sha||"—").slice(0,7)}</code></td>
      <td class="num">${e.total_instances}</td>
      <td class="num">${e.solved_count}</td>
      <td class="num">${fmt(e.mean_time, 2)}</td>
      <td class="num">${fmt(e.median_gap, 4)}</td>
    </tr>`).join("")
  }</tbody></table>`;
}

setInterval(async () => {
  if ($("#auto").checked) {
    await loadRuns();
    if (state.selectedId) await refresh();
  }
}, 3000);

(async () => { await loadRuns(); })();
</script>
</body></html>
"""


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports-dir", type=Path,
                   default=_BENCH_ROOT.parent / "reports",
                   help="Directory with benchmark run outputs (default: ./reports)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--minlplib-cache", type=Path, default=None,
                   help="MINLPLib cache dir (default: ~/.cache/discopt/minlplib)")
    p.add_argument("--minlplib-version", default="current")
    p.add_argument("--open", dest="open_browser", action="store_true",
                   help="Open a browser tab after starting")
    args = p.parse_args()

    from scripts.fetch_minlplib import get_cache_dir, get_instancedata_path
    cache = args.minlplib_cache or get_cache_dir()
    csv = get_instancedata_path(cache, args.minlplib_version)
    if csv.exists():
        print(f"[dashboard] loading MINLPLib reference: {csv}")
        index = load_instance_data(csv)
        print(f"[dashboard] {len(index)} reference instances")
    else:
        print(f"[dashboard] WARN: {csv} missing — reference-check disabled")
        index = {}

    ctx = _Ctx(args.reports_dir.resolve(), index)
    srv = _Server((args.host, args.port), _Dashboard, ctx)
    url = f"http://{args.host}:{args.port}/"
    print(f"[dashboard] reports_dir={ctx.reports_dir}")
    print(f"[dashboard] serving on {url}  (Ctrl-C to stop)")

    if args.open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] shutting down")
        srv.shutdown()


if __name__ == "__main__":
    main()
