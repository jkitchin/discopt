"""
Self-contained HTML dashboard generator for discopt benchmarks.

Generates an interactive HTML report with Plotly.js charts (loaded
from CDN) and a sortable results table using vanilla JavaScript.
No external Python dependencies beyond numpy (already required).
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from benchmarks.metrics import (
    BenchmarkResults,
    SolveResult,
    performance_profile,
    solved_count,
    speedup_table,
)

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# Solver color palette (up to 8 solvers)
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text))


def _fmt_time(t: float) -> str:
    """Format a wall time for display."""
    if not math.isfinite(t):
        return "&mdash;"
    if t < 0.01:
        return f"{t * 1000:.2f}ms"
    if t < 100:
        return f"{t:.3f}s"
    return f"{t:.1f}s"


def _fmt_obj(obj: float | None) -> str:
    """Format an objective value for display."""
    if obj is None:
        return "&mdash;"
    if abs(obj) < 1e-8:
        return "0"
    if abs(obj) >= 1e6 or (abs(obj) < 1e-3 and obj != 0):
        return f"{obj:.6e}"
    return f"{obj:.6f}"


def _solver_color(idx: int) -> str:
    """Get color for solver index."""
    return COLORS[idx % len(COLORS)]


def _build_header_html(
    benchmark: BenchmarkResults,
    category: str,
    level: str,
) -> str:
    """Build the page header section."""
    solvers = benchmark.get_solvers()
    instances = benchmark.get_instances()
    return f"""
<div class="header">
  <h1>discopt Benchmark Dashboard</h1>
  <div class="meta">
    <span class="tag">Suite: <strong>{_esc(benchmark.suite)}</strong></span>
    <span class="tag">Category: <strong>{_esc(category)}</strong></span>
    <span class="tag">Level: <strong>{_esc(level)}</strong></span>
    <span class="tag">Solvers: <strong>{len(solvers)}</strong></span>
    <span class="tag">Instances: <strong>{len(instances)}</strong></span>
    <span class="tag">Timestamp: <strong>{_esc(benchmark.timestamp)}</strong></span>
  </div>
</div>
"""


def _build_solved_bar_chart(benchmark: BenchmarkResults) -> str:
    """Build a Plotly bar chart of solved counts per solver."""
    solvers = benchmark.get_solvers()
    counts = [solved_count(benchmark.get_results(s)) for s in solvers]
    colors = [_solver_color(i) for i in range(len(solvers))]

    trace = {
        "x": solvers,
        "y": counts,
        "type": "bar",
        "marker": {"color": colors},
        "text": counts,
        "textposition": "outside",
    }
    layout = {
        "title": {"text": "Solved Instances per Solver"},
        "yaxis": {
            "title": "Solved count",
            "rangemode": "tozero",
        },
        "margin": {"t": 50, "b": 60, "l": 60, "r": 30},
        "height": 400,
    }
    return _plotly_div("solved-bar", trace, layout)


def _build_scatter_plots(benchmark: BenchmarkResults) -> str:
    """Build log-log scatter plots of solve times for each solver pair."""
    solvers = benchmark.get_solvers()
    if len(solvers) < 2:
        return ""

    sections = []
    sections.append("<h2>Pairwise Solve Time Comparison</h2>")

    pair_idx = 0
    for i in range(len(solvers)):
        for j in range(i + 1, len(solvers)):
            sa, sb = solvers[i], solvers[j]
            times_a = {r.instance: r.wall_time for r in benchmark.get_results(sa)}
            times_b = {r.instance: r.wall_time for r in benchmark.get_results(sb)}
            common = sorted(set(times_a.keys()) & set(times_b.keys()))
            if not common:
                continue

            xs = [max(times_b[inst], 1e-6) for inst in common]
            ys = [max(times_a[inst], 1e-6) for inst in common]
            labels = list(common)

            lo = min(min(xs), min(ys)) * 0.5
            hi = max(max(xs), max(ys)) * 2.0

            scatter_trace = {
                "x": xs,
                "y": ys,
                "mode": "markers",
                "type": "scatter",
                "text": labels,
                "hovertemplate": (
                    "%{text}<br>"
                    f"{_esc(sb)}: " + "%{x:.4f}s<br>"
                    f"{_esc(sa)}: " + "%{y:.4f}s"
                    "<extra></extra>"
                ),
                "marker": {
                    "size": 8,
                    "color": _solver_color(i),
                    "opacity": 0.7,
                },
                "name": "instances",
            }
            identity_trace = {
                "x": [lo, hi],
                "y": [lo, hi],
                "mode": "lines",
                "type": "scatter",
                "line": {
                    "color": "gray",
                    "dash": "dash",
                    "width": 1,
                },
                "name": "y = x",
                "showlegend": False,
            }
            layout = {
                "title": {"text": f"{_esc(sa)} vs {_esc(sb)} (log-log)"},
                "xaxis": {
                    "title": f"{_esc(sb)} time (s)",
                    "type": "log",
                },
                "yaxis": {
                    "title": f"{_esc(sa)} time (s)",
                    "type": "log",
                },
                "margin": {"t": 50, "b": 60, "l": 70, "r": 30},
                "height": 450,
                "showlegend": False,
            }
            div_id = f"scatter-{pair_idx}"
            sections.append(
                _plotly_div(
                    div_id,
                    [scatter_trace, identity_trace],
                    layout,
                )
            )
            pair_idx += 1

    return "\n".join(sections)


def _build_performance_profile_chart(
    benchmark: BenchmarkResults,
) -> str:
    """Build Dolan-More performance profile chart."""
    solvers = benchmark.get_solvers()
    if len(solvers) < 2:
        return ""

    profiles = performance_profile(benchmark, tau_max=100.0, tau_steps=500)

    traces = []
    for idx, solver in enumerate(solvers):
        if solver not in profiles:
            continue
        tau_arr, frac_arr = profiles[solver]
        traces.append(
            {
                "x": tau_arr.tolist(),
                "y": frac_arr.tolist(),
                "mode": "lines",
                "type": "scatter",
                "name": solver,
                "line": {
                    "color": _solver_color(idx),
                    "width": 2,
                },
            }
        )

    layout = {
        "title": {"text": "Performance Profile (Dolan-Mor&eacute;)"},
        "xaxis": {
            "title": "&tau; (time ratio to best solver)",
            "type": "log",
        },
        "yaxis": {
            "title": "Fraction of problems solved",
            "range": [0, 1.05],
        },
        "margin": {"t": 50, "b": 60, "l": 70, "r": 30},
        "height": 450,
        "legend": {"x": 0.7, "y": 0.1},
    }
    return "<h2>Performance Profile</h2>\n" + _plotly_div("perf-profile", traces, layout)


def _build_results_table(benchmark: BenchmarkResults) -> str:
    """Build a sortable HTML table of all results."""
    solvers = benchmark.get_solvers()
    instances = benchmark.get_instances()

    # Build a lookup: (solver, instance) -> SolveResult
    lookup: dict[tuple[str, str], SolveResult] = {}
    for solver in solvers:
        for r in benchmark.get_results(solver):
            lookup[(solver, r.instance)] = r

    # Table header
    header_cells = ['<th onclick="sortTable(0)">Instance</th>']
    col = 1
    for s in solvers:
        header_cells.append(f'<th onclick="sortTable({col})">{_esc(s)} Status</th>')
        col += 1
        header_cells.append(f'<th onclick="sortTable({col})">{_esc(s)} Obj</th>')
        col += 1
        header_cells.append(f'<th onclick="sortTable({col})">{_esc(s)} Time</th>')
        col += 1
        header_cells.append(f'<th onclick="sortTable({col})">{_esc(s)} Iters</th>')
        col += 1
        header_cells.append(f'<th onclick="sortTable({col})">{_esc(s)} Nodes</th>')
        col += 1

    rows = []
    for inst in instances:
        cells = [f"<td>{_esc(inst)}</td>"]
        for s in solvers:
            r = lookup.get((s, inst))
            if r is None:
                cells.extend(["<td>&mdash;</td>"] * 5)
            else:
                status_cls = "solved" if r.is_solved else "unsolved"
                cells.append(f'<td class="{status_cls}">{_esc(r.status.value)}</td>')
                cells.append(f'<td class="num">{_fmt_obj(r.objective)}</td>')
                cells.append(f'<td class="num">{_fmt_time(r.wall_time)}</td>')
                cells.append(f'<td class="num">{r.iterations}</td>')
                cells.append(f'<td class="num">{r.node_count}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
<h2>Results Table <span class="hint">(click headers to sort)</span></h2>
<div class="table-wrap">
<table id="results-table">
  <thead><tr>{"".join(header_cells)}</tr></thead>
  <tbody>
    {"".join(rows)}
  </tbody>
</table>
</div>
"""


def _build_speedup_heatmap(benchmark: BenchmarkResults) -> str:
    """Build a Plotly heatmap of pairwise speedup ratios."""
    solvers = benchmark.get_solvers()
    if len(solvers) < 2:
        return ""

    table = speedup_table(benchmark, shift=1.0)

    z = []
    annotations = []
    for i, sa in enumerate(solvers):
        row = []
        for j, sb in enumerate(solvers):
            val = table.get((sa, sb), float("nan"))
            if math.isfinite(val):
                row.append(round(val, 3))
                annotations.append(
                    {
                        "x": j,
                        "y": i,
                        "text": f"{val:.2f}",
                        "showarrow": False,
                        "font": {"color": "white" if val > 1.5 else "black"},
                    }
                )
            else:
                row.append(None)
        z.append(row)

    trace = {
        "z": z,
        "x": solvers,
        "y": solvers,
        "type": "heatmap",
        "colorscale": [
            [0.0, "#2ca02c"],
            [0.5, "#f7f7f7"],
            [1.0, "#d62728"],
        ],
        "zmid": 1.0,
        "colorbar": {"title": "SGM ratio"},
    }
    layout = {
        "title": {"text": ("Pairwise Speedup (SGM ratio: row / col, &lt;1 = row faster)")},
        "annotations": annotations,
        "margin": {"t": 60, "b": 80, "l": 100, "r": 30},
        "height": 100 + 60 * len(solvers),
        "xaxis": {"side": "bottom"},
    }
    return "<h2>Speedup Heatmap</h2>\n" + _plotly_div("speedup-heatmap", trace, layout)


def _plotly_div(
    div_id: str,
    traces: dict | list,
    layout: dict,
) -> str:
    """Generate a <div> + <script> for a Plotly chart."""
    if isinstance(traces, dict):
        traces = [traces]
    traces_json = json.dumps(traces, default=str)
    layout_json = json.dumps(layout)
    return f"""
<div id="{div_id}" class="chart"></div>
<script>
Plotly.newPlot(
  "{div_id}",
  {traces_json},
  {layout_json},
  {{responsive: true}}
);
</script>
"""


def _css() -> str:
    """Return the dashboard CSS."""
    return """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont,
      "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #f8f9fa;
    color: #212529;
    padding: 24px;
  }
  .container {
    max-width: 1400px;
    margin: 0 auto;
  }
  .header {
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 24px;
  }
  .header h1 {
    font-size: 1.6rem;
    margin-bottom: 12px;
  }
  .meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }
  .tag {
    background: #e9ecef;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.85rem;
  }
  h2 {
    font-size: 1.2rem;
    margin: 28px 0 12px 0;
  }
  .hint {
    font-size: 0.8rem;
    font-weight: normal;
    color: #6c757d;
  }
  .chart {
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
  }
  .table-wrap {
    overflow-x: auto;
    margin-bottom: 24px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    background: #fff;
    font-size: 0.82rem;
  }
  th, td {
    padding: 6px 10px;
    border: 1px solid #dee2e6;
    white-space: nowrap;
  }
  th {
    background: #343a40;
    color: #fff;
    cursor: pointer;
    user-select: none;
    position: sticky;
    top: 0;
    z-index: 1;
  }
  th:hover { background: #495057; }
  td.num { font-family: "SF Mono", Menlo, monospace; text-align: right; }
  td.solved { color: #2ca02c; font-weight: 600; }
  td.unsolved { color: #d62728; }
  tr:nth-child(even) { background: #f8f9fa; }
  tr:hover { background: #e2e6ea; }
</style>
"""


def _sort_js() -> str:
    """Return vanilla JS for sortable table."""
    return """
<script>
var sortDir = {};
function sortTable(colIdx) {
  var table = document.getElementById("results-table");
  var tbody = table.tBodies[0];
  var rows = Array.from(tbody.rows);
  var dir = sortDir[colIdx] === "asc" ? "desc" : "asc";
  sortDir[colIdx] = dir;
  rows.sort(function(a, b) {
    var cellA = a.cells[colIdx].textContent.trim();
    var cellB = b.cells[colIdx].textContent.trim();
    var numA = parseFloat(cellA.replace(/[^\\d.eE\\-+]/g, ""));
    var numB = parseFloat(cellB.replace(/[^\\d.eE\\-+]/g, ""));
    var valA = isNaN(numA) ? cellA.toLowerCase() : numA;
    var valB = isNaN(numB) ? cellB.toLowerCase() : numB;
    if (valA < valB) return dir === "asc" ? -1 : 1;
    if (valA > valB) return dir === "asc" ? 1 : -1;
    return 0;
  });
  rows.forEach(function(row) { tbody.appendChild(row); });
}
</script>
"""


def generate_html_dashboard(
    benchmark: BenchmarkResults,
    category: str,
    level: str = "smoke",
    known_optima: dict[str, float] | None = None,
    output_path: Path | None = None,
) -> str:
    """
    Generate a self-contained HTML dashboard from benchmark results.

    Parameters
    ----------
    benchmark : BenchmarkResults
        Collected benchmark data with solver results.
    category : str
        Problem category label (e.g. "MINLP", "NLP", "LP").
    level : str
        Benchmark level (e.g. "smoke", "phase1", "global_opt").
    known_optima : dict, optional
        Mapping of instance name to known optimal objective.
    output_path : Path, optional
        If provided, write the HTML to this file path.

    Returns
    -------
    str
        Complete HTML string for the dashboard.
    """
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>discopt Benchmark Dashboard &mdash; {_esc(benchmark.suite)}</title>",
        f'<script src="{PLOTLY_CDN}"></script>',
        _css(),
        "</head>",
        "<body>",
        '<div class="container">',
        _build_header_html(benchmark, category, level),
        _build_solved_bar_chart(benchmark),
        _build_scatter_plots(benchmark),
        _build_performance_profile_chart(benchmark),
        _build_results_table(benchmark),
        _build_speedup_heatmap(benchmark),
        "</div>",
        _sort_js(),
        "</body>",
        "</html>",
    ]

    html_str = "\n".join(parts)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_str, encoding="utf-8")

    return html_str
