# LR-0 — Log-space root-LP entry experiment (GO/KILL)

**Task:** LR-0 of `docs/dev/lever-a-root-tightness-plan.md` (§4). Gates the whole
Lever-A campaign. **Date:** 2026-07-11. **Regime:** entry experiment, no
production code (probe only). **Env:** `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`,
`PYTHONPATH=<worktree>/python`, prebuilt `_rust.so` (crates hash
`f919f41e…` matches this worktree).

## Verdict (headline)

**MIXED — H-LOG is confirmed on nvs09 but only *with* H-UNI, and is INERT on
nvs05 and tanksize.** By the letter of the §4 gate this is **not a clean GO**
(only 1 of 3 targets reaches a root certificate) and **not a clean KILL** (nvs09
*is* fully closed). The honest read:

- **H-LOG alone is not the lever for the class.** On the two multi-variable
  targets (nvs05, tanksize) the log-space monomial transform closes **0 %** of
  the root gap — byte-identical to recursive McCormick — because their gap is
  **not** driven by positive-monomial looseness (see per-instance root-cause).
- **H-LOG + H-UNI certifies nvs09 at the root** (bound −43.1343 vs opt −43.134,
  gap 3.4e-4 < tol 4.4e-3). But nvs09's win is **carried by H-UNI (LR-2), not
  H-LOG**: H-LOG-only leaves −51.14 (8.01 loose); the exact 1-D univariate
  envelope of the `(ln(x−2))²+(ln(10−x))²` composites is what closes it.

**Recommendation:** treat LR-0 as a **conditional KILL of H-LOG-as-the-campaign-
lever** and a **GO for LR-2 (H-UNI) scoped to nvs09's univariate-composite
class**. Do **not** build LR-1 (the log-monomial compiler pass) as specified: on
this measured target set it is inert where the gap lives. Re-scope per §Re-scope
below before spending the 2–4 day LR-1 budget.

## Hypothesis (H-LOG, plan §3)

For a monomial `t = ∏ xᵢ^{aᵢ}` with every `xᵢ.lb > 0`, the log-space relaxation
(`zᵢ = ln xᵢ` exact concave envelope; `s = Σ aᵢ zᵢ` exact linear; `t = exp(s)`
exact convex envelope) is tight enough that the root LP certifies nvs05, nvs09,
tanksize (BARON-shaped, §1.1). Variant (b) for nvs09 adds H-UNI: exact 1-D convex
envelope of the per-variable composites.

## Method (probe, `docs/dev/lr0_probe/`)

- **`nl_parse.py`** — a from-scratch recursive-descent `.nl` prefix-opcode parser
  → typed DAG. **Validated against the Rust `_nl_repr` oracle** (`evaluate_objective`
  / `evaluate_constraint`) to machine precision on all three instances before any
  relaxation is trusted: objective err **0.0** on all three; max constraint err
  **0.0** (nvs05, nvs09) / **7.3e-12** (tanksize). This eliminates hand-
  transcription risk.
- **`lr0_envelopes.py`** — rigorous univariate envelope rows: `ln` concave
  (tangents = overestimators, secant = underestimator), `exp`/`x²` convex
  (tangents = underestimators, secant = overestimator), plus an **exact 1-D
  convex-underenvelope** builder (lower convex hull of a fine sample, shifted
  down by the *measured* max hull-vs-function gap so every facet is a **proven**
  underestimator; slack on nvs09 = **2.2e-10**).
- **`lr0_relax.py`** — general sound DAG→LP relaxer. Positive-monomial detection
  applies H-LOG (toggle `use_log_monomial`); bilinear→McCormick, sqrt→secant,
  powers/ln/exp→curvature envelopes, affine→exact. **Strict positivity is a hard
  precondition** (factor `lb ≤ 1e-9` ⇒ term is *not* log-transformed; no
  epsilon-shift, per plan §3 / §0.1). Interval arithmetic gives rigorous node
  enclosures. Genuinely-unbounded lifted directions stay **free** (never a huge
  finite sentinel). FBBT (`_nl_repr.fbbt`) runs first (plan sequencing).
- **`lr0_general.py`** — builds the root LP (min objective LP-var s.t. relaxed
  constraint rows), solves with HiGHS via `scipy.linprog`. A constraint whose
  sound relaxation is numerically explosive (lifted magnitude > 1e10, from the
  `1e15`-scaled welded-beam stress rows) is **dropped** — which only *enlarges*
  the feasible region, so the LP bound stays a valid lower bound.
- **`lr0_nvs09.py`** — dedicated nvs09 builder for variant (a) H-LOG-only vs (b)
  H-LOG+H-UNI, with `(prod xᵢ)^0.2 = exp(0.2 Σ ln xᵢ)`.
- **Rigor / feasible-point sampling:** every reported bound was checked
  `bound ≤ optimum` and, for nvs09, `bound ≤ min true objective over 20 000
  random integer points` (variant a −51.14 ✓, variant b −43.1343 ✓). No relaxation
  row cuts a sampled feasible point.

## Per-instance results

| instance | opt | discopt root (plan) | McCormick (H-LOG OFF) | **H-LOG ON** | H-LOG+H-UNI | % root-gap closed by H-LOG | root cert? |
|---|---|---|---|---|---|---|---|
| **nvs09** | −43.134 | (—) | — | **−51.144** (variant a) | **−43.1343** (variant b) | H-LOG-only loose by 8.01; **+H-UNI ⇒ certificate** | **(b) yes**, (a) no |
| **nvs05** | 5.4709 | 0.674 | 0.67402 | **0.67402** | n/a | **0.0 %** | no |
| **tanksize** | 1.2686 | 0.847 | 0.83824 | **0.83824** | n/a | **~0 %** (−2.1 % vs probe McCormick; noise) | no |

Root-certificate tolerances (`1e-4·(1+|opt|)`): nvs09 4.4e-3, nvs05 6.5e-4,
tanksize 2.3e-4. All probe bounds are **sound** (≤ optimum; nvs09 ≤ min sampled
feasible objective).

### nvs09 (a)-vs-(b) — sizes H-UNI's necessity

| variant | root LP bound | gap-to-opt |
|---|---|---|
| (a) H-LOG only | −51.1443 | 8.010 |
| (b) H-LOG + H-UNI | −43.1343 | **0.00034** (root certificate) |

The product term `(∏xᵢ)^0.2` handled by H-LOG is *not* the residual — H-LOG
handles it well. The 8.01 of looseness in (a) is entirely the per-variable
`gᵢ(x)=(ln(x−2))²+(ln(10−x))²` composites, whose term-by-term composition lets
each square independently approach 0. The **exact 1-D convex envelope of `gᵢ`**
(H-UNI) is what closes nvs09. Measured exact 1-D min of `gᵢ` on [3,9] = 3.6667
(the plan estimated 3.79; the numeric here is tighter). Note `gᵢ` is **not
globally convex** on [3,9] (min g'' = −0.097 near the edges), so H-UNI's "exact
convex envelope" is the genuine lower convex hull, not the function — still a
proven underestimator.

## Root-cause of the two KILLs (why H-LOG is inert where the gap lives)

**nvs05 — the gap is division-constraint-driven, not monomial-driven.**
The objective `1.10471·x0²·x1 + 0.04811·x2·x3·(14+x1)` is a posynomial whose
**unconstrained box minimum is exactly 0.6740** — which is precisely discopt's
0.674 root bound and my probe's bound. The optimum 5.4709 is lifted entirely by
the welded-beam **stress constraints C2/C6, which are *ratios/divisions*** (`o3`
nodes) over wide boxes: C2's lifted range is `[−2.9e12, −4.4e-5]`, **identical
under H-LOG ON and OFF** (log-space applies to positive monomials, not to
division). The objective monomials H-LOG *does* tighten are already at their
box-min bound, so H-LOG moves the root bound by **0.0**. The lever for nvs05 is a
tight relaxation of `a/(x·y·…)` ratio constraints over wide boxes (OBBT to shrink
the box, or a fractional/signomial-in-log treatment of the *division*), not the
objective monomial transform.

**tanksize — 23 of 47 variables have `lb = 0`, disqualifying most bilinears
from H-LOG by the (correctly enforced) strict-positivity precondition.**
FBBT does not lift those lower bounds off zero. Of the 84 bilinear/product terms,
only a minority are `x·y` with both factors strictly positive; only **16**
log-space lifted vars are created, and the root bound is byte-identical to
McCormick (0.838). Honoring the plan's hard "no epsilon-shift" rule (§3, §0.1)
means H-LOG **cannot** engage on the zero-lb bilinears that dominate tanksize.
The lever is a strong bilinear/RLT treatment that works at `lb = 0` (McCormick is
already exact-at-corner there; the residual is elsewhere — likely the sqrt terms
and the wide-box bilinear envelopes), not log-space.

## Falsification statement (for plan §7 + gap-closing-plan §6)

> **F16 — H-LOG (log-space positive-monomial relaxation) is NOT the root-tightness
> lever for the nvs05/tanksize class, and is not sufficient for nvs09.** On the
> measured LR-0 probe (sound log/exp/ln envelopes, strict-positivity enforced,
> validated `.nl` transcription, feasible-point-sampled): H-LOG closes **0 %** of
> the root gap on **nvs05** (its gap is the box-min of a posynomial objective;
> the 4.8 to the optimum is carried by *division* stress constraints C2/C6 whose
> lifted range is identical under H-LOG ON/OFF) and **~0 %** on **tanksize**
> (23/47 vars have `lb=0`, so the strict-positivity precondition — no
> epsilon-shift — excludes the dominant bilinears; only 16 log vars form and the
> bound is byte-identical to McCormick). On **nvs09** H-LOG-only leaves −51.14
> (8.01 loose); the root certificate (−43.1343) is produced by **H-UNI** (exact
> 1-D univariate-composite envelope), not H-LOG. Therefore LR-1 (the log-monomial
> compiler pass) as specified is inert where the gap lives on this target set;
> the genuine levers are per-class: (nvs09) H-UNI/LR-2; (nvs05) wide-box
> division/ratio relaxation + OBBT; (tanksize) zero-lb-tolerant bilinear/RLT +
> sqrt envelope strength.

## Re-scope (what to build instead of LR-1-as-written)

1. **LR-2 (H-UNI) is GO, scoped to univariate composites** — nvs09's certificate
   is entirely H-UNI's. Build the exact 1-D convex/concave envelope for maximal
   single-variable subtrees (rigorous curvature via interval arithmetic, refuse
   past a subdivision budget — plan §LR-2). Default-OFF, regime-2 gates. This is
   the one banked win from LR-0.
2. **nvs05 → a division/ratio lever, not a monomial lever.** The gap is `a/(x·y)`
   over `[0.01,200]²`-wide boxes. Path: aggressive root+node OBBT to shrink the
   box feeding the ratio envelope, and/or a log-space treatment of the *division*
   (`ln(a/(x·y)) = ln a − ln x − ln y`, valid since numerator/denominator are
   positive) — i.e. H-LOG on the **denominator monomial inside a quotient**, a
   different detection target than "objective monomial." Entry-experiment this
   before building.
3. **tanksize → a zero-lb-tolerant bilinear/sqrt lever.** H-LOG is structurally
   excluded. Measure whether the residual 0.42 is the sqrt terms or the wide-box
   bilinears and target that.

## Reproduce

```bash
cd <worktree> && export PYTHONPATH=$PWD/python:$PWD/docs/dev/lr0_probe
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
# nvs09 (both variants + feasible-point sampling):
python3 docs/dev/lr0_probe/lr0_nvs09.py
# nvs05 / tanksize (H-LOG ON vs OFF):
python3 docs/dev/lr0_probe/lr0_general.py nvs05    --opt 5.4709 --root 0.674
python3 docs/dev/lr0_probe/lr0_general.py tanksize --opt 1.2686 --root 0.847
```

(The probe symlinks `python/tests/data/minlplib_nl` for the `.nl` files; run from
the worktree root so the relative path resolves, or edit the path constant.)
