# G3.2 probes (#2136)

These are the measurement scripts behind `docs/dev/g32-tableau-cmir-entry-gate.md`, which
should be read first. They were written as throwaway session probes and are kept here so the
work can resume on any machine. That is why they are terse. Paths are portable: the repo root
comes from this file's location (override with `$DISCOPT_ROOT`), and captured inputs live in
`./captures/` (override with `$G32_CAPTURES`). `captures/` is git-ignored; regenerate it as
shown below.

Every probe that loads the extension asserts `discopt.__file__` is under the repo root and
checks a marker string in `_rust.abi3.so` (CLAUDE.md §8). Build first, from the repo root:
`maturin develop --release`, or build a wheel and copy its `_rust.abi3.so` into
`python/discopt/`. You also need `gdplib` from source (see
`discopt_benchmarks/benchmarks/gdplib_runner.py`), `pyomo` and `highspy`.

## Inputs

| file | what | made by |
|---|---|---|
| `captures/<method>__<instance>.pkl` | exact arguments of the **first `solve_milp_csc_py` call** of a GDPlib solve, i.e. the Rust-MILP input in standard form (structural columns, then one logical per row) | `capture_all.sh` → `caprust.py` |
| `captures/inrepo__*.pkl` | the same, for 7 in-repo MILP fixtures (i1183, clay0203 master, the HiGHS presolve-cycle case, issue1229, fac2/rsyn0830m/tls2 OA masters) | `cap_inrepo.py` (run from `captures/`) |
| `captures/mtp_raw.pkl` | the first LOA master of `med_term_purchasing` in `milp_highs` row form (`c, A_ub, b_ub, A_eq, b_eq, bounds, integrality`) | `capture_master.py med_term_purchasing loa captures/mtp_raw.pkl` |
| `*.mps` | MTP master after the coef-entry stages, for HiGHS CLI ablations | `coef_entry.py captures/mtp_raw.pkl out.mps`, with `ABL=fbbt_only|fbbt_coef|obbt_nocoef` |

```bash
cd discopt_benchmarks/scripts/g32
DISCOPT_LP_MILP_BACKEND=rust bash capture_all.sh        # ~36 pkls, <=150 s each
(cd captures && python ../cap_inrepo.py)
python capture_master.py med_term_purchasing loa captures/mtp_raw.pkl
```

**Check (2026-11-04, branch `fix/1917-g32-decide` merged with main at `d032897d`).**
Recapturing `loa__med_term_purchasing` with `caprust.py` gave a pickle identical to the
2026-11-02 original in every key except `time_limit_s`. `prof2.py` on it reproduced the gate
doc's 200/10/select row exactly: coef ON / c-MIR OFF −8099.15; ON/ON −8119.32;
OFF/OFF −8359.00.

## Probes

Both flags these probes were built to decide, `DISCOPT_RUST_MILP_COEF_TIGHTEN` and
`DISCOPT_RUST_MILP_TABLEAU_CMIR`, were **retired by #2136** (gate doc, section "#2136
decisions"). The probes that only toggled them (`prof2.py`, `grid2x2.py`, `rustgrid.py`,
`replay_step1.py`, `replay_step2.py`, `nsentry.py`, `ct_seed.py`, `mk_rustcoef.py`) were
deleted with them; they are in git history at `3eba9bf`. What remains is reusable:

| script | measures | gate-doc section |
|---|---|---|
| `replay.py --env NAME --fire-ctr CTR [--profile captured\|strong\|big] [--base ENV=V] T out.jsonl pkl...` | any Rust-MILP flag OFF vs ON on the captures, interleaved, judged by `oracle.judge` against the best verified point (arms + HiGHS); the ON arm's firing counter is recorded per row | #2136 differentials |
| `oracle.py` | the verdict rules (`python/tests/test_2136_g32_oracle.py`) | #2136 oracle |
| `cutgrid.py out.jsonl pkl...` | root gap closed per cut-loop budget (GMI + cover) over every capture | step 5 cut-loop budget |
| `hugechk.py pkl...` | columns at the `±9.999e19` default box per input | NS default-box |
| `coef_entry.py`, `rtight.py`, `feas_chk.py`, `mps2pkl.py` | step 1 entry experiment on the HiGHS side (FBBT + coefficient tightening of the MTP root LP in Python) | step 1 entry |
| `tab_cmir_entry2.py <pkl> <cmir\|cmir_novub> <rounds>`, `proto_abl.py` | the Python tableau c-MIR prototype and ablations; every cut is checked to keep the HiGHS-optimal point; env `CAP` = cuts per round | step 2 entry |

## The step-2 replay oracle defect (fixed 2026-11-04, #2136)

The last pre-fix step-2 replay (T=30, 36 inputs) flagged one row as BAD:

| `inrepo__clay0203_master` (captured before `2df0b0a`) | status | objective | bound |
|---|---|---|---|
| HiGHS reference | Optimal | 54489.415 | 54489.415 |
| Rust, both flags OFF | **optimal** | **41907.460**, point verified (viol 1.4e-14) | −752903.98 |
| Rust, coef ON | feasible | 55668.585 | −inf |

Both halves of the diagnosis are now resolved:

1. **The Rust `optimal` was a real certificate defect, not a probe artefact.** The capture
   predates `2df0b0a`, so 6 of its 30 columns carried the default box (±9.999e19) as
   finite bounds. With the box re-closed on today's capture, the driver still returned
   `optimal` 41907.46 / bound −752904 on the strong profile (true optimum 41420.27). The
   node LP reported ≈ −752904 for a vertex whose `cᵀx` is 41907; every gap test ran
   against the LP value, and only the returned objective was recomputed (#1296). The
   driver now withdraws the certificate whenever the returned objective and the tree's
   incumbent value disagree beyond `1e-6·max(1,|obj|)` (counter
   `MilpIncumbentMisvaluedDecert`; `test_milp_default_box.py::
   test_driver_withdraws_a_misvalued_incumbent_certificate`, which fails before). On a
   capture made after `2df0b0a` the trigger is gone: Rust and HiGHS both return 41420.27.
2. **The oracle** is `oracle.py`: the primal reference is the best *independently verified*
   point across both arms and HiGHS; an arm's `optimal` must have its bound within gap of
   its objective and must not be beaten by a verified point; HiGHS disagreements are
   `notes`, not BAD rows. `python/tests/test_2136_g32_oracle.py` pins the rules, including
   the row above.
3. **`replay_step2.py` did not measure step 2.** It toggled `COEF_TIGHTEN`, and its
   `cmir_cuts=True` is #2107's aggregated c-MIR, not the tableau separator. The #2136
   panels ran through `replay.py`, which toggles the named flag and records its firing
   counter on each arm (step 2: `DISCOPT_RUST_MILP_TABLEAU_CMIR` with coefficient
   tightening held ON in both arms).

## Counter names

#2107 (main) added an aggregated c-MIR separator with phase `SepCmir` and counter
`SepCmirCuts`. This branch's tableau separator uses phase `SepTabCmir` and the counters
`SepCmirCalls`, `SepCmirRows`, `SepCmirCutsEmitted`, `SepCmirVb*` and `SepCmirRefused*`,
which only `lp/tableau_cmir.rs` increments. The names are distinct but easy to confuse; a
rename to `SepTabCmir*` would be cleaner.
