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

| script | measures | gate-doc section |
|---|---|---|
| `prof2.py <coef 0/1> <root_cuts> <rounds> <select 0/1> <cmir 0/1>` | Rust root bound on the MTP replay at `max_nodes=2`, with the cut counters | step 2 grid |
| `grid2x2.py` | the coef × c-MIR 2×2 grid on MTP; asserts every c-MIR arm separated | step 2 grid |
| `rustgrid.py` | the Rust cut-loop budget grid (GMI only) on MTP | "the loop is the lever" |
| `replay_step1.py T out.jsonl pkl...` | step 1 differential: `COEF_TIGHTEN=0` vs `=1`, interleaved, against a HiGHS reference on the same standard form, with an independent feasibility check of the returned point | step 1 panel |
| `replay_step2.py T out.jsonl pkl...` | the same with the strong root profile + `cmir_cuts=True` | step 2 differential (**see the defect below**) |
| `nsentry.py out.jsonl pkl...` | NS default-box entry experiment (killed, 0/36 at default) | NS default-box |
| `hugechk.py pkl...` | columns at the `±9.999e19` default box per input | NS default-box |
| `ct_seed.py` | coef tightening with a seeded incumbent | step 1 |
| `coef_entry.py`, `mk_rustcoef.py`, `rtight.py`, `feas_chk.py`, `mps2pkl.py` | step 1 entry experiment: FBBT + coef tightening on the MTP root LP, and HiGHS-side checks | step 1 entry |
| `tab_cmir_entry2.py <pkl> <cmir|cmir_novub> <rounds>`, `proto_abl.py` | the Python tableau c-MIR prototype and ablations; every cut is checked to keep the HiGHS-optimal point; env `CAP` = cuts per round | step 2 entry |

## Open defect: the step-2 replay oracle

The last step-2 replay, `replay_step2.py` at T=30 over 36 inputs, flagged one row as BAD:

| `inrepo__clay0203_master` | status | objective | bound |
|---|---|---|---|
| HiGHS reference | Optimal | 54489.415 | 54489.415 |
| Rust, both flags OFF | **optimal** | **41907.460**, point verified (viol 1.4e-14) | −752903.98 |
| Rust, coef ON | feasible | 55668.585 | −inf |

- **The HiGHS reference is wrong here.** Rust's OFF arm returns an independently verified
  point *below* HiGHS's "optimal" value. HiGHS's false optimum on the clay0203 master is
  #2069.
- **The flag is not implicated.** The row fails on the OFF arm.
- **Before any §5 panel, the replay oracle needs to:**
  1. stop trusting HiGHS's status as ground truth; use the best *verified* point across
     arms and HiGHS as the primal reference;
  2. flag a Rust `optimal` whose `bound` is not within gap of its `obj`. The OFF arm above
     reports `optimal` with bound −752904 against obj 41907. That is either a status
     mapping issue in the probe or a real certificate problem, and it needs a look on its
     own.

## Counter names

#2107 (main) added an aggregated c-MIR separator with phase `SepCmir` and counter
`SepCmirCuts`. This branch's tableau separator uses phase `SepTabCmir` and the counters
`SepCmirCalls`, `SepCmirRows`, `SepCmirCutsEmitted`, `SepCmirVb*` and `SepCmirRefused*`,
which only `lp/tableau_cmir.rs` increments. The names are distinct but easy to confuse; a
rename to `SepTabCmir*` would be cleaner.
