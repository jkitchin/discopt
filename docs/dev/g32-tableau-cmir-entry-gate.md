# #1917 G3.2 — tableau c-MIR separator: pre-registered entry gate

Registered **before** any implementation (CLAUDE.md §4). It is a gate, not a report.

## Measured premise

On the first LOA master of `med_term_purchasing` (n=498, 216 binaries, 924 rows;
LP root −8358.999245, optimum −6797.5397), the Rust MILP root bound is the bare
LP under every cut profile. Root GMI cuts leave the objective unchanged (the LP
is dual-degenerate), and 60 s of B&B reaches bound −7698 without proving
optimality. We built `ref/HiGHS` with each separator env-gated and ran it at 1
thread, 60 s:

| HiGHS separators     | root bound | nodes   | wall   |
|----------------------|-----------:|--------:|-------:|
| all off              | −8358.999  | 438,498 | 60 s (unsolved, bound −7507.7) |
| tableau only         | −7061.0    | 1,433   | 0.73 s |
| path only            | −7512.6    | 2,010   | 0.69 s |
| default (all)        | −6908.9    | 65      | 0.45 s |

## Hypothesis

HiGHS's tableau separator pipeline alone closes the gap. The pipeline is
`HighsTableauSeparator.cpp`, then `HighsTransformedLp.cpp` (bound substitution,
simple and variable bounds), then `HighsCutGeneration.cpp` (c-MIR with
complementation and lifting heuristics, plus integral-support strengthening).
A faithful port of it closes the gap on the same master, where our GMI does not.
Probing implications are added only if the gate shows they are needed.

## Gate (on the fixture, prototype in Python or Rust)

1. **Pass:** the root bound after the root cut loop is **≥ −7100**.
2. **Soundness (non-negotiable):** every cut must keep the known optimal point
   feasible, and it must also keep feasible points sampled from the MILP
   (incumbents from fixed-binary LP solves) feasible.
3. **Kill:** a faithful port of the tableau path that stays **above −7800**
   (i.e. bound < −7800) is killed. The kill is reported together with its delta
   against HiGHS's cut-by-cut log.

## After the gate passes

1. Put it in the Rust root cut loop behind a default-OFF `DISCOPT_*` flag, with
   a row in `flag-retirement-audit.md`.
2. Add `cargo test -p discopt-core` unit tests for cut validity.
3. Run a differential panel (LOA master replay set, in-repo MILP corpus, MTP)
   that is cert-clean with no valid point cut, and report the node and wall
   ratios.
4. Graduate or retire the flag per CLAUDE.md §5.
