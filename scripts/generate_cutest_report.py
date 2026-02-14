#!/usr/bin/env python
"""
Generate an executed Jupyter notebook report from CUTEst benchmark JSON.

Usage:
    python scripts/generate_cutest_report.py results/cutest_<TS>.json results/cutest_<TS>.ipynb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def _md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source.strip())


def _code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip())


def build_notebook(json_path: str) -> nbformat.NotebookNode:
    """Build a notebook that analyzes CUTEst benchmark results."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    cells = nb.cells

    # --- Header ---
    cells.append(
        _md(f"""\
# CUTEst Benchmark Report

**Results file:** `{json_path}`

Comparing **Ipopt** vs **Ripopt** on CUTEst nonlinear optimization problems.
""")
    )

    # --- Load data ---
    cells.append(
        _code(f"""\
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (10, 5)

data = json.loads(Path({json_path!r}).read_text())
df = pd.DataFrame(data["results"])

print(f"Benchmark: {{data['label']}}")
print(f"Timestamp: {{data['timestamp']}}")
print(f"Problems:  {{data['n_problems']}}")
print(f"Max n:     {{data.get('max_n', '?')}}")
""")
    )

    # --- Summary table ---
    cells.append(
        _md("""\
## Summary
""")
    )

    cells.append(
        _code("""\
both_solved = df[(df.ipopt_status == "optimal") & (df.ripopt_status == "optimal")]
ipopt_only = df[(df.ipopt_status == "optimal") & (df.ripopt_status != "optimal")]
ripopt_only = df[(df.ripopt_status == "optimal") & (df.ipopt_status != "optimal")]
both_failed = df[(df.ipopt_status != "optimal") & (df.ripopt_status != "optimal")]
load_err = df[(df.ipopt_status == "LOAD_ERR") | (df.ripopt_status == "LOAD_ERR")]

summary = pd.DataFrame({
    "Category": [
        "Both solved", "Ipopt only", "Ripopt only",
        "Both failed", "Load errors", "Total",
    ],
    "Count": [
        len(both_solved), len(ipopt_only), len(ripopt_only),
        len(both_failed), len(load_err), len(df),
    ],
})
summary["Pct"] = (100 * summary["Count"] / len(df)).round(1).astype(str) + "%"
summary.style.hide(axis="index")
""")
    )

    # --- Status breakdown ---
    cells.append(
        _md("""\
## Status Breakdown
""")
    )

    cells.append(
        _code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, solver, col in zip(axes, ["Ipopt", "Ripopt"], ["ipopt_status", "ripopt_status"]):
    counts = df[col].value_counts().sort_index()
    colors = []
    for s in counts.index:
        if s == "optimal":
            colors.append("#2ecc71")
        elif s in ("acceptable", "solved"):
            colors.append("#3498db")
        elif s in ("iteration_limit", "time_limit"):
            colors.append("#f39c12")
        elif s in ("ERROR", "LOAD_ERR", "error"):
            colors.append("#e74c3c")
        else:
            colors.append("#95a5a6")
    counts.plot.barh(ax=ax, color=colors)
    ax.set_title(f"{solver} Status")
    ax.set_xlabel("Count")
    for i, v in enumerate(counts):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)

plt.tight_layout()
plt.show()
""")
    )

    # --- Runtime distributions ---
    cells.append(
        _md("""\
## Runtime Distributions
""")
    )

    cells.append(
        _code("""\
# Filter to problems where both solvers returned a finite time
mask = (
    df.ipopt_time.notna() & df.ripopt_time.notna()
    & (df.ipopt_time < 1e6) & (df.ripopt_time < 1e6)
    & (df.ipopt_time > 0) & (df.ripopt_time > 0)
)
tdf = df[mask].copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Histogram of log10(time)
ax = axes[0]
bins = np.linspace(-4, 2, 50)
ax.hist(np.log10(tdf.ipopt_time), bins=bins, alpha=0.6, label="Ipopt", color="#3498db")
ax.hist(np.log10(tdf.ripopt_time), bins=bins, alpha=0.6, label="Ripopt", color="#e67e22")
ax.set_xlabel("log10(time / s)")
ax.set_ylabel("Count")
ax.set_title("Runtime Distribution")
ax.legend()

# Scatter: ipopt vs ripopt time
ax = axes[1]
lo = min(tdf.ipopt_time.min(), tdf.ripopt_time.min()) * 0.5
hi = max(tdf.ipopt_time.max(), tdf.ripopt_time.max()) * 2
ax.scatter(tdf.ipopt_time, tdf.ripopt_time, s=12, alpha=0.5, color="#2c3e50")
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Ipopt time (s)")
ax.set_ylabel("Ripopt time (s)")
ax.set_title("Runtime Scatter (log-log)")
ax.legend()

plt.tight_layout()
plt.show()
""")
    )

    # --- Performance profile ---
    cells.append(
        _md("""\
## Performance Profile (Dolan-More)

Fraction of problems solved within factor $\\tau$ of the best solver time.
""")
    )

    cells.append(
        _code("""\
# Only consider problems both solved optimally
cs = df[(df.ipopt_status == "optimal") & (df.ripopt_status == "optimal")].copy()

if len(cs) > 0:
    best = np.minimum(cs.ipopt_time.values, cs.ripopt_time.values)
    best = np.maximum(best, 1e-10)  # avoid div-by-zero

    ratio_ipopt = cs.ipopt_time.values / best
    ratio_ripopt = cs.ripopt_time.values / best

    taus = np.sort(np.unique(np.concatenate([ratio_ipopt, ratio_ripopt, [1.0]])))
    taus = taus[taus <= 100]  # cap at 100x
    taus = np.concatenate([[1.0], taus, [100.0]])

    def profile(ratios, taus):
        return np.array([np.mean(ratios <= t) for t in taus])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(taus, profile(ratio_ipopt, taus), where="post", label="Ipopt", lw=2)
    ax.step(taus, profile(ratio_ripopt, taus), where="post", label="Ripopt", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Performance ratio τ")
    ax.set_ylabel("Fraction of problems solved")
    ax.set_title(f"Performance Profile (n={len(cs)} commonly-solved)")
    ax.legend()
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No commonly-solved problems for performance profile.")
""")
    )

    # --- Objective comparison ---
    cells.append(
        _md("""\
## Objective Comparison
""")
    )

    cells.append(
        _code("""\
cs = df[
    (df.ipopt_status == "optimal") & (df.ripopt_status == "optimal")
    & df.ipopt_obj.notna() & df.ripopt_obj.notna()
].copy()

if len(cs) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: ipopt_obj vs ripopt_obj
    ax = axes[0]
    ax.scatter(cs.ipopt_obj, cs.ripopt_obj, s=12, alpha=0.5, color="#8e44ad")
    lo = min(cs.ipopt_obj.min(), cs.ripopt_obj.min())
    hi = max(cs.ipopt_obj.max(), cs.ripopt_obj.max())
    margin = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", lw=0.8)
    ax.set_xlabel("Ipopt objective")
    ax.set_ylabel("Ripopt objective")
    ax.set_title("Objective Values")

    # Relative difference histogram
    ax = axes[1]
    denom = np.maximum(np.abs(cs.ipopt_obj.values), np.abs(cs.ripopt_obj.values))
    denom = np.maximum(denom, 1e-10)
    rel_diff = np.abs(cs.ipopt_obj.values - cs.ripopt_obj.values) / denom
    rel_diff_log = np.log10(np.maximum(rel_diff, 1e-16))

    ax.hist(rel_diff_log, bins=40, color="#2980b9", edgecolor="white")
    ax.axvline(-4, color="red", ls="--", lw=1, label="1e-4 threshold")
    ax.set_xlabel("log10(|obj_diff| / max(|obj|))")
    ax.set_ylabel("Count")
    ax.set_title("Relative Objective Difference")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Count disagreements
    disagree_mask = rel_diff > 1e-3
    n_agree = (~disagree_mask).sum()
    print(f"Objective agreement (rel < 1e-3): {n_agree}/{len(cs)} "
          f"({100*n_agree/len(cs):.1f}%)")
    if disagree_mask.sum() > 0:
        print(f"Disagreements: {disagree_mask.sum()}")
else:
    print("No commonly-solved problems for objective comparison.")
""")
    )

    # --- Problem size analysis ---
    cells.append(
        _md("""\
## Problem Size Analysis
""")
    )

    cells.append(
        _code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scatter: n vs time, colored by solver
ax = axes[0]
valid_i = df[df.ipopt_time.notna() & (df.ipopt_time < 1e6) & (df.ipopt_time > 0)]
valid_r = df[df.ripopt_time.notna() & (df.ripopt_time < 1e6) & (df.ripopt_time > 0)]
ax.scatter(valid_i.n, valid_i.ipopt_time, s=10, alpha=0.4, label="Ipopt", color="#3498db")
ax.scatter(valid_r.n, valid_r.ripopt_time, s=10, alpha=0.4, label="Ripopt", color="#e67e22")
ax.set_xlabel("Number of variables (n)")
ax.set_ylabel("Time (s)")
ax.set_yscale("log")
ax.set_title("Solve Time vs Problem Size")
ax.legend()

# Failure rate by size bucket
ax = axes[1]
bins = [0, 5, 10, 20, 50, 100, float("inf")]
labels = ["1-5", "6-10", "11-20", "21-50", "51-100", "100+"]
df["size_bin"] = pd.cut(df.n, bins=bins, labels=labels, right=True)

fail_data = []
for label_val in labels:
    subset = df[df.size_bin == label_val]
    if len(subset) == 0:
        continue
    ipopt_fail = (subset.ipopt_status != "optimal").sum() / len(subset)
    ripopt_fail = (subset.ripopt_status != "optimal").sum() / len(subset)
    fail_data.append({"bin": label_val, "Ipopt": ipopt_fail, "Ripopt": ripopt_fail})

if fail_data:
    fdf = pd.DataFrame(fail_data).set_index("bin")
    fdf.plot.bar(ax=ax, color=["#3498db", "#e67e22"])
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel("Failure rate")
    ax.set_title("Failure Rate by Size")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
""")
    )

    # --- Timing statistics ---
    cells.append(
        _md("""\
## Timing Statistics (Commonly-Solved)
""")
    )

    cells.append(
        _code("""\
import math

cs = df[(df.ipopt_status == "optimal") & (df.ripopt_status == "optimal")].copy()

if len(cs) > 0:
    def sgm(times, shift=1.0):
        log_sum = sum(math.log(t + shift) for t in times)
        return math.exp(log_sum / len(times)) - shift

    ipopt_t = cs.ipopt_time.values
    ripopt_t = cs.ripopt_time.values

    stats = pd.DataFrame({
        "Metric": ["Mean", "Median", "SGM (shift=1)", "Min", "Max", "Faster count"],
        "Ipopt": [
            f"{ipopt_t.mean():.4f}s",
            f"{np.median(ipopt_t):.4f}s",
            f"{sgm(ipopt_t):.4f}s",
            f"{ipopt_t.min():.4f}s",
            f"{ipopt_t.max():.4f}s",
            f"{(ipopt_t < ripopt_t).sum()}",
        ],
        "Ripopt": [
            f"{ripopt_t.mean():.4f}s",
            f"{np.median(ripopt_t):.4f}s",
            f"{sgm(ripopt_t):.4f}s",
            f"{ripopt_t.min():.4f}s",
            f"{ripopt_t.max():.4f}s",
            f"{(ripopt_t < ipopt_t).sum()}",
        ],
    })

    sgm_ratio = sgm(ripopt_t) / max(sgm(ipopt_t), 1e-10)
    print(f"SGM ratio (Ripopt/Ipopt): {sgm_ratio:.3f}x")
    print(f"n = {len(cs)} commonly-solved problems\\n")
    stats.style.hide(axis="index")
else:
    print("No commonly-solved problems.")
""")
    )

    # --- Detailed tables ---
    cells.append(
        _md("""\
## Detailed Tables

### Problems Solved by Ipopt Only
""")
    )

    cells.append(
        _code("""\
ipopt_only = df[(df.ipopt_status == "optimal") & (df.ripopt_status != "optimal")]
if len(ipopt_only) > 0:
    display_cols = ["name", "n", "m", "ripopt_status", "ipopt_time", "ipopt_obj"]
    print(f"{len(ipopt_only)} problems solved by Ipopt only:")
    ipopt_only[display_cols].sort_values("name").head(40).style.hide(axis="index")
else:
    print("None — Ripopt solves everything Ipopt does!")
""")
    )

    cells.append(
        _md("""\
### Problems Solved by Ripopt Only
""")
    )

    cells.append(
        _code("""\
ripopt_only = df[(df.ripopt_status == "optimal") & (df.ipopt_status != "optimal")]
if len(ripopt_only) > 0:
    display_cols = ["name", "n", "m", "ipopt_status", "ripopt_time", "ripopt_obj"]
    print(f"{len(ripopt_only)} problems solved by Ripopt only:")
    ripopt_only[display_cols].sort_values("name").head(40).style.hide(axis="index")
else:
    print("None — Ipopt solves everything Ripopt does!")
""")
    )

    cells.append(
        _md("""\
### Largest Objective Disagreements
""")
    )

    cells.append(
        _code("""\
cs = df[
    (df.ipopt_status == "optimal") & (df.ripopt_status == "optimal")
    & df.ipopt_obj.notna() & df.ripopt_obj.notna()
].copy()

if len(cs) > 0:
    denom = np.maximum(np.abs(cs.ipopt_obj.values), np.abs(cs.ripopt_obj.values))
    denom = np.maximum(denom, 1e-10)
    cs = cs.copy()
    cs["rel_diff"] = np.abs(cs.ipopt_obj.values - cs.ripopt_obj.values) / denom
    cs["abs_diff"] = np.abs(cs.ipopt_obj.values - cs.ripopt_obj.values)

    top = cs.nlargest(20, "rel_diff")
    cols = ["name", "n", "m", "ipopt_obj", "ripopt_obj", "abs_diff", "rel_diff"]
    top[cols].style.hide(axis="index").format({
        "ipopt_obj": "{:.6e}", "ripopt_obj": "{:.6e}",
        "abs_diff": "{:.2e}", "rel_diff": "{:.2e}",
    })
else:
    print("No commonly-solved problems.")
""")
    )

    return nb


def main():
    parser = argparse.ArgumentParser(description="Generate CUTEst report notebook")
    parser.add_argument("json_path", help="Path to CUTEst results JSON")
    parser.add_argument("output_path", help="Path for output .ipynb")
    parser.add_argument(
        "--no-execute", action="store_true", help="Skip execution (just build the notebook)"
    )
    args = parser.parse_args()

    json_path = str(Path(args.json_path).resolve())
    output_path = Path(args.output_path)

    if not Path(json_path).exists():
        print(f"ERROR: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Building notebook from {json_path}...")
    nb = build_notebook(json_path)

    if not args.no_execute:
        print("Executing notebook...")
        ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": str(output_path.parent)}})
        print("Execution complete.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(output_path))
    print(f"Notebook saved to {output_path}")


if __name__ == "__main__":
    main()
