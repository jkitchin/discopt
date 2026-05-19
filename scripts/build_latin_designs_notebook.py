"""Generate docs/notebooks/latin-designs.ipynb."""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(
    r"""# Latin-Square Designs and ANOVA

Classical *Latin-square* designs let you test whether several factors
matter, **after removing two known nuisance sources of variation**,
using a remarkably small number of runs. They are the natural
generalization of randomized complete-block designs to more than one
blocking variable {cite:p}`Fisher1935,Cochran1957`.

## When to use a Latin-style design

You have one experimental factor whose effect you want to estimate
(call it the *treatment*) and two or more nuisance factors you want
to *block out*: time of day, operator, batch of raw material,
position in a plate, …

A $k \times k$ Latin square uses $k^2$ runs to study one treatment
across two blocking factors at $k$ levels each. A Graeco-Latin
square adds a second treatment at the same total cost. A
hyper-Graeco-Latin square adds a third. All four designs assume
**no interactions** between the factors — they are additive-effects
designs. If you suspect interactions matter, use a factorial design
instead.

`discopt.doe` provides:

| Function                       | Factors | Total runs   |
|--------------------------------|--------:|-------------:|
| `latin_square_design(k=1, …)`  | 1       | $k$          |
| `latin_square_design(k=2, …)`  | 2       | $k^2$ (RCB)  |
| `latin_square(k)`              | 3       | $k^2$        |
| `graeco_latin_square(k)`       | 4       | $k^2$        |
| `hyper_graeco_latin_square(k)` | 5       | $k^2$        |

`anova_report` then produces an additive-effects F-table from the
completed runs — Type-I sums of squares, F-statistics, p-values, with
support for two-way interactions if you want to test the no-interaction
assumption explicitly.

## Plan

1. Build a 4×4 Latin square design programmatically; inspect the
   randomization and balance.
2. Synthesize a response with known true effects and one blocking
   nuisance, then recover the effects via `anova_report`.
3. Show how to add a Graeco-Latin square (4 factors at the same
   cost).
4. End with a quick CLI round-trip showing the same workflow from
   `discopt doe new latin-square` through `discopt doe anova`.
"""
)

code(
    """import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import matplotlib.pyplot as plt

from discopt.doe import (
    AnovaTable,
    anova_report,
    graeco_latin_square,
    hyper_graeco_latin_square,
    latin_square,
    latin_square_design,
)

rng = np.random.default_rng(0)
"""
)

md(
    """## 1. A 4×4 Latin square

A Latin square is a $k \\times k$ matrix of $k$ symbols such that
**every symbol appears exactly once in each row and once in each
column**. With $k = 4$, the unique 4×4 standard square is::

    A B C D
    B A D C
    C D A B
    D C B A

`latin_square(k)` returns a randomized version. The rows correspond
to one nuisance factor, the columns to a second, and the symbol
inside each cell to the treatment of interest."""
)

code(
    """from discopt.doe.latin import _build_squares

square = latin_square(4, seed=1)
print("Latin square (randomized):")
print(np.array(square))
print()
print("Each row contains each symbol once:")
print({i: sorted(set(row)) for i, row in enumerate(square)})
"""
)

md(
    """### Materializing the run sheet

For data analysis we want one row per experiment, with explicit
columns for the row block, column block, and treatment. The helper
`latin_square_design` does that — it returns balanced row dicts
with optional replicates."""
)

code(
    """design = latin_square_design(
    factors={
        "row":       ["morning", "noon", "afternoon", "evening"],
        "column":    ["bench-1", "bench-2", "bench-3", "bench-4"],
        "treatment": ["A", "B", "C", "D"],
    },
    replicates=1,
    seed=2,
)
print(f"{len(design.rows)} runs from a 4x4 Latin square")
print()
for r in design.rows[:6]:
    print(r)
print("...")
"""
)

md(
    """Balance check: every treatment appears the same number of times in
each block. That's exactly what makes ANOVA on a Latin square
clean — the main effects are orthogonal."""
)

code(
    """from collections import Counter
print("treatment counts by row block:")
for block in ["morning", "noon", "afternoon", "evening"]:
    cnt = Counter(r["treatment"] for r in design.rows if r["row"] == block)
    print(f"  {block:10s}: {dict(cnt)}")
"""
)

md(
    """## 2. Synthesize responses and recover the effects

We simulate a response with

* a treatment effect: A=0, B=2, C=5, D=3,
* a row-block nuisance: morning=0, noon=1, afternoon=-1, evening=2,
* a column-block nuisance: 0, 0.5, -0.5, 1,
* Gaussian noise with σ=0.5.

If the design works, `anova_report` should declare the treatment
factor significant and let the blocks soak up their nuisance
variance."""
)

code(
    """true_treatment = {"A": 0.0, "B": 2.0, "C": 5.0, "D": 3.0}
true_row        = {"morning": 0.0, "noon": 1.0, "afternoon": -1.0, "evening": 2.0}
true_column     = {"bench-1": 0.0, "bench-2": 0.5, "bench-3": -0.5, "bench-4": 1.0}

rng = np.random.default_rng(3)
rows_with_response = []
for r in design.rows:
    y = (
        true_treatment[r["treatment"]]
        + true_row[r["row"]]
        + true_column[r["column"]]
        + 0.5 * rng.normal()
    )
    rows_with_response.append({**r, "y": float(y)})

# Look at the first few synthetic measurements
for r in rows_with_response[:5]:
    print(r)
"""
)

md(
    """### Running ANOVA

`anova_report(rows, response, factors=None)` auto-detects the factor
columns (everything except the response and bookkeeping columns like
`replicate`, `run_order`, `_*`). It returns an `AnovaTable` whose
`summary()` method prints the familiar F-table."""
)

code(
    """table: AnovaTable = anova_report(
    rows_with_response,
    response="y",
    factors=["row", "column", "treatment"],
)
print(table.summary())
"""
)

md(
    """**Reading the table:**

* `treatment` is highly significant (large F, tiny p-value) — the
  design correctly recovered the synthetic effect.
* The two blocks (`row`, `column`) absorbed their nuisance variance;
  if either one's effect had been small you would see a high p-value
  there, and you could justify dropping the block from the model.
* Residual MS is close to $\\sigma^2 = 0.25$, the variance we used
  for the noise.

You can also pull out the per-effect dataclasses for programmatic use::

    [e.source for e in table.rows]
"""
)

code(
    """print("Per-effect summary:")
for e in table.rows:
    f_str = f"{e.f:7.2f}" if e.f is not None else "    --"
    p_str = f"{e.p:.4e}" if e.p is not None else "    --"
    print(f"  {e.source:14s}  SS={e.ss:7.3f}  df={e.df:2d}  F={f_str}  p={p_str}")
"""
)

md(
    """### Testing the no-interaction assumption

A Latin square cannot estimate interactions in general (the design
doesn't have enough degrees of freedom). But if you have
*replicates* — independent repeats of the entire square — you can
add a `replicate` column and request specific 2-way interactions via
the `interactions=` argument."""
)

code(
    """design_rep = latin_square_design(
    factors={
        "row":       ["morning", "noon", "afternoon", "evening"],
        "column":    ["bench-1", "bench-2", "bench-3", "bench-4"],
        "treatment": ["A", "B", "C", "D"],
    },
    replicates=2,
    seed=4,
)
rng = np.random.default_rng(5)
rows_rep = []
for r in design_rep.rows:
    y = (
        true_treatment[r["treatment"]]
        + true_row[r["row"]]
        + true_column[r["column"]]
        + 0.5 * rng.normal()
    )
    rows_rep.append({**r, "y": float(y)})

table_int = anova_report(
    rows_rep, response="y",
    factors=["row", "column", "treatment"],
    interactions=[("row", "treatment")],
)
print(table_int.summary())
"""
)

md(
    """The `row × treatment` interaction is non-significant — exactly
what we'd expect, because the synthetic data was generated from an
additive model. If you saw a significant interaction here, the
no-interaction assumption that underlies the Latin square would be
violated and you should consider switching to a factorial design.
"""
)

md(
    """## 3. Graeco-Latin squares: four factors at the same cost

A Graeco-Latin square is two mutually-orthogonal Latin squares
superimposed. The second symbol set adds a fourth factor without
any extra runs — orthogonality means the two treatments are still
balanced against each other and against both blocks.

`graeco_latin_square(k)` returns a list of two MOLS. The convenience
wrapper `latin_square_design` accepts four factor columns and routes
to it automatically."""
)

code(
    """gl_design = latin_square_design(
    factors={
        "row":         ["t1", "t2", "t3", "t4"],
        "column":      ["b1", "b2", "b3", "b4"],
        "treatment_A": ["A1", "A2", "A3", "A4"],
        "treatment_B": ["B1", "B2", "B3", "B4"],
    },
    seed=6,
)
print(f"{len(gl_design.rows)} runs — still 16, now with 4 factors")
print()
for r in gl_design.rows[:6]:
    print(r)
"""
)

md(
    """For $k=2$ no MOLS exist; for $k=6$ Euler showed none exist either
(the famous "36 officers" non-result). The generator raises
`ValueError` in those cases — try `graeco_latin_square(6)` and you'll
see why this design doesn't scale to arbitrary $k$.

A `hyper_graeco_latin_square(k)` adds a third orthogonal symbol set,
giving five factors at $k^2$ runs. This works for $k \\in \\{4, 5,
7, 8, 9, ...\\}$ — any prime power $\\geq 4$. The generator
returns three MOLS; combine them with `latin_square_design` for an
additive-effects design across one treatment + two blocks + three
"extra" factors.
"""
)

code(
    """# Hyper-Graeco-Latin for k=5 -> 25 runs covering 5 factors
hg = hyper_graeco_latin_square(5)
print(f"hyper-Graeco-Latin k=5: {len(hg)} mutually-orthogonal squares")
print(f"each square is {len(hg[0])}x{len(hg[0])}")
"""
)

md(
    """## 4. CLI round trip

The same workflow is exposed from the command line. The CLI writes
to an `.xlsx` workbook so a wet-lab user can fill in the response
column by hand and re-run `discopt doe anova` later.

```bash
discopt doe new latin-square \\
    --level "row:morning,noon,afternoon,evening" \\
    --level "column:bench-1,bench-2,bench-3,bench-4" \\
    --level "treatment:A,B,C,D" \\
    --replicates 1 \\
    --out latin.xlsx

# … fill in the y column in latin.xlsx …

discopt doe anova latin.xlsx
# or, with interactions:
discopt doe anova latin.xlsx --interaction "row:treatment"
```

The same `--level NAME:v1,v2,…` syntax works for `graeco-latin`
(four factors) and `hyper-graeco-latin` (five factors).

## Summary

* A $k \\times k$ Latin square gives you one treatment + two blocks
  at $k^2$ runs.
* Graeco-Latin → 3 nuisance factors / 1 treatment, hyper-Graeco-Latin
  → 4 nuisance / 1 treatment, **same** $k^2$ runs.
* All four are *additive-effects* designs — the analysis assumes no
  interactions. Test that assumption with replicates +
  `interactions=` if you're unsure.
* `anova_report` is the same function for any balanced design;
  it auto-detects factor columns and handles unbalanced data with a
  warning.

## References

* {cite:t}`Fisher1935` — the original treatment of Latin squares as
  blocked designs.
* {cite:t}`Cochran1957` — comprehensive coverage of Latin and
  Graeco-Latin analysis.
"""
)

nb["cells"] = cells

dest = Path("docs/notebooks/latin-designs.ipynb")
dest.parent.mkdir(parents=True, exist_ok=True)
with dest.open("w") as f:
    nbf.write(nb, f)
print(f"wrote {dest} with {len(cells)} cells")
