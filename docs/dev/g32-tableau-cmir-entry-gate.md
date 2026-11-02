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

## Result (2026-11-01): the premise is falsified, the kill criterion fires

**Retraction (§11).** The "tableau only" row above ran with HiGHS **presolve on**.
It does not measure the tableau separator alone, and the hypothesis built on it
("the tableau pipeline alone closes the gap") is wrong. The same patched HiGHS
binary, `threads=1`, `G32_OFF=path,modk,implbound,clique` (only the tableau
separator on), gives:

| model fed to HiGHS (presolve off unless noted)          | root after cuts | nodes | wall   |
|---------------------------------------------------------|----------------:|------:|-------:|
| raw fixture, HiGHS presolve **on** (the row above)      | −7061.0         | 1,433 | 0.73 s |
| raw fixture                                             | **−8357.4**     | 4,159 | 3.63 s |
| HiGHS's own presolved model (`write_presolved_model_file`) | −7233.2      | 1,355 | 0.62 s |
| raw + FBBT only                                         | −8281.7         | 2,964 | 2.26 s |
| raw + FBBT + OBBT, no coefficient tightening            | −8224.8         | 2,782 | 2.09 s |
| raw + FBBT + big-M coefficient tightening               | −7166.3         | 1,099 | 1.12 s |
| raw + FBBT + coef tightening + OBBT + coef again        | −7125.0         |   550 | 0.82 s |

Without presolve, HiGHS's own tableau separator stays at −8357.4. That is above
−7800, so a *faithful* port of the tableau path alone on this master is killed
by gate 3. It cannot do better than the code it ports. The lever is **big-M
coefficient tightening**. It removes 624 coefficients ≥ 1000 in HiGHS presolve
(max |a| 10100 → 171.9). For example, `x261 ≤ 1000·y76` becomes
`x261 ≤ 1.59·y76`. Coefficient tightening does not move the LP root (−8358.999
in every row). It changes which cuts the tableau rows yield. FBBT or OBBT bound
changes without it are worth almost nothing (−8282, −8225).

Our own separators gain less from the same reformulation. On the same tightened
models, under default kwargs, with the 60 s bound at 1 thread:

| model                        | Rust root | Rust 60 s bound      |
|------------------------------|----------:|---------------------:|
| raw                          | −8359.0   | −7763.6 (unsolved)   |
| FBBT + coef                  | −8219.3   | −7372.1 (unsolved)   |
| FBBT + coef + OBBT + coef    | −8057.1   | −7149.9 (unsolved)   |

So both pieces are needed. Coefficient tightening comes first (presolve), and
HiGHS-quality tableau c-MIR comes on top of it. Neither suffices alone. The
known optimum (−6797.5397) is feasible in both tightened models, with max row
violation 1.2e-10. Scripts are `g32/coef_entry.py` (with `ABL=` for the
ablation), `g32/rtight.py` and `g32/feas_chk.py` in the session scratchpad.
