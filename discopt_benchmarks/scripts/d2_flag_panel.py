"""D2: a generic §5 graduation panel for one native-MILP flag, ON vs OFF, bars pre-registered
here before any run (CLAUDE.md section 5; the plan's Phase 4 standing rule, plus the two
bars the P4.4 panel lacked).

``--flag NAME`` (or a comma-separated bundle, toggled together) is set to
``--on-value`` (default ``1``) in the ON arm and REMOVED in the
OFF arm, or set to ``--off-value`` when given (a default-ON flag's OFF arm is ``0``:
removing it would run ON twice); ``--fired REGEX`` names the driver profile counters (or
``solver_stats`` keys) that prove the mechanism ran
(summed per solve; ON must be > 0, OFF must be 0). Population and arms as
``p44_gmi_repair_panel.py``: MIPLIB-38 + the M panel, Rust MILP route, fresh subprocess per
solve, ``DISCOPT_PROFILE=1`` in both arms, arm order alternating, ``--reps`` 3,
``--time-limit`` 20 s.

Pre-registered bars:

BAR 1, cert-clean (every solve): certified objective = reference within 1e-4 (1+|opt|); no
  bound past it by > 1e-6 (1+|opt|); no incumbent better than it; every point max_viol <= 1e-6.
BAR 1b (gap plan P0, CLAUDE.md §5 "no gap_certified=True instance regresses"): no instance
  certified in EVERY OFF rep and in NO ON rep. A summed 2a let ``DISCOPT_MILP_CUT_LP``
  graduate while ``enlight8`` went 3/3 -> 0/3. Partial losses (every OFF rep certified,
  some ON reps not) are reported as ``cert_wobble``, not a bar on 3 reps.
BIMODAL rows: an instance certified in some but not all OFF reps is a coin flip, and so is
  one in ``KNOWN_BIMODAL`` (bimodal across panels, with the record). Such rows are excluded
  from 2a's sums and from 2d, and reported separately; bar 1b still applies to them.
BAR 2, net-positive:
  (2a) certified ON >= certified OFF (summed over instances x reps, bimodal-OFF rows
       excluded);
  (2b) nodes over the instances certified in every solve of both arms at least 15 % lower ON,
       OR certified ON >= OFF + 2 with that node total not higher ON;
  (2c) dual-bound losses <= 10 % of the open population (median over reps; a loss is > 0.01
       relative AND beyond OFF's own rep spread);
  (2d) NO incumbent lost: no instance where OFF holds an incumbent in more reps than ON
       (the P4.4 lesson);
  (2e) median primal gap over the open population not higher ON (reported, and a bar).
Graduate only if BAR 1 and all of BAR 2 pass.

PERMUTATION REPS (``--permute``; 2026-11-26, #2350). A deterministic solve has no rep-to-rep
spread, so ``--reps`` repeats one coin flip. For an instance whose node count is decided by
its row/column ORDER (measured: ``bell5`` takes 1,365 to 16,553 nodes OFF across six
orderings), the order is the randomness, so the permutations are the reps: rep 0 solves the
original file and rep ``r >= 1`` solves a copy with rows and columns permuted by seed ``r``
(``permuted_copy``), the SAME copy in both arms. The bars are unchanged; they read the reps as
always (1b: certified in every OFF rep and no ON rep; 2e: the per-row median over reps).
Which rows get permutation reps is fixed before the run: every row 1b flags in EITHER
direction (lost or gained in every rep) on the single-rep panel, plus EVERY row of its 2e
population (the open rows), so no row is picked by how it moved.
``--rescore BASE --merge PERM[,PERM2...]`` replaces BASE's rows for the names in the PERM files
and re-evaluates. As first applied (#2350) every row of the 2e population gets the reps, so 2e
is scored on per-row medians over 7 orderings, with no row selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
PANEL_PY = HERE.parent / "lp_milp_panel.py"
sys.path.insert(0, str(HERE.parent))

import lp_milp_panel as P  # noqa: E402, N812
import p42_rens_panel as R  # noqa: E402, N812  (population, gaps, cert checks)

FLAG = ""
ON_VALUE = "1"
OFF_VALUE: str | None = None
_CTR = re.compile(r"$^")
_CTR_NAME = re.compile(r"$^")


def permuted_copy(path: str, seed: int, out_dir: Path) -> str:
    """Write ``path`` with its rows and columns permuted by ``seed`` (``seed == 0``: ``path``).

    The model is read and written by HiGHS (``highspy``): bounds, costs, integrality, sense,
    offset and names move with their row/column, so the copy is the same problem in a
    different order. The copy is re-read and its shape asserted, so a writer that dropped
    anything fails here rather than in a solve."""
    if seed == 0:
        return path
    import numpy as np
    import scipy.sparse as sp
    from discopt._optional_highs import require_highspy

    highspy = require_highspy("d2_flag_panel --permute")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(path).name.split('.')[0]}_p{seed}.mps"
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if h.readModel(path) != highspy.HighsStatus.kOk:
        raise RuntimeError(f"highspy could not read {path}")
    lp = h.getLp()
    m, n = lp.num_row_, lp.num_col_
    a_mat = sp.csc_matrix(
        (lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_), shape=(m, n)
    )
    rng = np.random.default_rng(seed)
    rp, cp = rng.permutation(m), rng.permutation(n)
    b = a_mat[rp][:, cp].tocsc()
    b.sort_indices()
    new = highspy.HighsLp()
    new.num_col_, new.num_row_ = n, m
    new.sense_, new.offset_ = lp.sense_, lp.offset_
    new.col_cost_ = np.asarray(lp.col_cost_)[cp]
    new.col_lower_ = np.asarray(lp.col_lower_)[cp]
    new.col_upper_ = np.asarray(lp.col_upper_)[cp]
    new.row_lower_ = np.asarray(lp.row_lower_)[rp]
    new.row_upper_ = np.asarray(lp.row_upper_)[rp]
    if len(lp.integrality_):
        new.integrality_ = [lp.integrality_[j] for j in cp]
    if len(lp.col_names_) == n:
        new.col_names_ = [lp.col_names_[j] for j in cp]
    if len(lp.row_names_) == m:
        new.row_names_ = [lp.row_names_[i] for i in rp]
    new.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    new.a_matrix_.num_col_, new.a_matrix_.num_row_ = n, m
    new.a_matrix_.start_, new.a_matrix_.index_ = b.indptr, b.indices
    new.a_matrix_.value_ = b.data
    h2 = highspy.Highs()
    h2.setOptionValue("output_flag", False)
    if h2.passModel(new) != highspy.HighsStatus.kOk or h2.writeModel(str(out)) not in (
        highspy.HighsStatus.kOk,
        highspy.HighsStatus.kWarning,
    ):
        raise RuntimeError(f"highspy could not write the seed-{seed} copy of {path}")
    h3 = highspy.Highs()
    h3.setOptionValue("output_flag", False)
    h3.readModel(str(out))
    back = h3.getLp()
    got = (back.num_row_, back.num_col_, len(back.a_matrix_.value_))
    want = (m, n, a_mat.nnz)
    if got != want:
        raise RuntimeError(f"{out}: re-read shape {got} != {want}")
    if len(lp.integrality_) and sum(map(int, back.integrality_)) != sum(map(int, lp.integrality_)):
        raise RuntimeError(f"{out}: integrality count changed on re-read")
    return str(out)


def solve(path: str, on: bool, a) -> dict:
    flags = FLAG.split(",")  # a comma-separated bundle toggles together
    env = {k: v for k, v in os.environ.items() if k not in (*flags, "DISCOPT_LP_STD_ENGINE")}
    env.update(PYTHONUNBUFFERED="1", DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PROFILE="1")
    if on:
        for f in flags:
            env[f] = ON_VALUE
    elif OFF_VALUE is not None:
        for f in flags:
            env[f] = OFF_VALUE
    cmd = [sys.executable, str(PANEL_PY), "worker", path, "R", str(a.time_limit), str(a.gap),
           str(a.max_nodes)]  # fmt: skip
    load = os.getloadavg()[0]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3 * a.time_limit + 180)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"{path} on={on}: rc={p.returncode} {p.stderr[-1500:]}")
    res = json.loads(lines[0][len("RESULT ") :])
    ctr: dict[str, int] = {}
    for ln in p.stderr.splitlines():
        mm = _CTR.match(ln)
        if mm:
            ctr[mm.group(1)] = ctr.get(mm.group(1), 0) + int(mm.group(2))
    # A Python-side mechanism reports through solver_stats instead of a profile counter.
    for k, v in (res.get("solver_stats") or {}).items():
        if _CTR_NAME.fullmatch(k) and v:
            ctr[k] = ctr.get(k, 0) + 1
    res["fired"] = ctr
    res["load"] = load
    return res


#: Instances recorded as bimodal across panels (gap plan P0). Evidence per entry.
KNOWN_BIMODAL = {
    # OFF certified-rep counts across 16 MIPLIB-38+M panels on the same code paths
    # (docs/dev/data/highs-parity/d2-review/history.out): 0/0 ... 0/1 0/2 3/3 3/2 2/1 1/3
    # 2/0 0/3 2/3 3/2. It alone decided the INCR_PRICING x2 and TOL_ROBUST verdicts.
    "p0548": "history.out: OFF certification ranges 0/3..3/3 across panels",
}


def cert_problems(res: dict, opt: float, maximize: bool) -> list[str]:
    res = {**res, "rens": {}}  # R.cert_problems' RENS check is inert without counters
    return R.cert_problems(res, opt, maximize, "off")


def evaluate(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["name"], []).append(r)
    problems = [(r["name"], r["rep"], arm, p) for r in rows for arm, ps in r["problems"].items()
                for p in ps]  # fmt: skip
    cert = {nm: {arm: sum(bool(r[arm]["gap_certified"]) for r in rs) for arm in ("off", "on")}
            for nm, rs in by.items()}  # fmt: skip
    nrep = {nm: len(rs) for nm, rs in by.items()}
    bimodal = sorted(nm for nm in by if 0 < cert[nm]["off"] < nrep[nm] or nm in KNOWN_BIMODAL)
    lost_all = sorted(nm for nm in by if cert[nm]["off"] == nrep[nm] and cert[nm]["on"] == 0)
    gained_all = sorted(nm for nm in by if cert[nm]["on"] == nrep[nm] and cert[nm]["off"] == 0)
    wobble = sorted(
        nm for nm in by if cert[nm]["off"] == nrep[nm] and 0 < cert[nm]["on"] < nrep[nm]
    )
    stable = [nm for nm in by if nm not in bimodal]
    solved = {arm: sum(cert[nm][arm] for nm in stable) for arm in ("off", "on")}
    solved_all = {arm: sum(bool(r[arm]["gap_certified"]) for r in rows) for arm in ("off", "on")}
    both = [nm for nm, rs in by.items()
            if all(r[arm]["gap_certified"] for r in rs for arm in ("off", "on"))]  # fmt: skip
    nodes = {arm: sum(statistics.median([r[arm]["nodes"] for r in by[nm]]) for nm in both)
             for arm in ("off", "on")}  # fmt: skip
    pop = [nm for nm in by if nm not in both]
    dual_losses, detail = 0, {}
    for nm in pop:
        g = {arm: [R.gaps(r[arm], r["opt"], r["maximize"])[1] for r in by[nm]]
             for arm in ("off", "on")}  # fmt: skip
        med = {arm: statistics.median(g[arm]) for arm in g}
        spread = R._spread(g["off"])
        loss = R._worse_by(med["on"], med["off"], max(0.01, spread))
        dual_losses += loss
        pg = {arm: statistics.median([R.gaps(r[arm], r["opt"], r["maximize"])[0]
                                      for r in by[nm]]) for arm in ("off", "on")}  # fmt: skip
        inc = {arm: sum(r[arm]["objective"] is not None for r in by[nm]) for arm in ("off", "on")}
        detail[nm] = {"dual_off": med["off"], "dual_on": med["on"], "off_spread": spread,
                      "dual_loss": loss, "primal_off": pg["off"], "primal_on": pg["on"],
                      "incumbents_off": inc["off"], "incumbents_on": inc["on"]}  # fmt: skip
    lost_inc = [nm for nm in stable
                if sum(r["off"]["objective"] is not None for r in by[nm])
                > sum(r["on"]["objective"] is not None for r in by[nm])]  # fmt: skip
    med_primal = {arm: (statistics.median([detail[nm][f"primal_{arm}"] for nm in pop])
                        if pop else 0.0) for arm in ("off", "on")}  # fmt: skip
    cap = math.floor(0.10 * len(pop))
    nodes_ok = nodes["off"] > 0 and nodes["on"] <= 0.85 * nodes["off"]
    plus2 = solved["on"] >= solved["off"] + 2 and nodes["on"] <= nodes["off"]
    bar2 = {
        "2a_solved_on_ge_off": solved["on"] >= solved["off"],
        "2b_nodes_minus15_or_solved_plus2": nodes_ok or plus2,
        "2c_dual_losses_le_10pct": dual_losses <= cap,
        "2d_no_incumbent_lost": not lost_inc,
        "2e_median_primal_not_higher": med_primal["on"] <= med_primal["off"],
    }
    fired = {arm: sum(sum(r[arm]["fired"].values()) for r in rows) for arm in ("off", "on")}
    return {
        "bar1_problems": problems,
        "bar1_pass": not problems,
        "bar1b_cert_lost_every_rep": lost_all,
        "bar1b_pass": not lost_all,
        "bar1b_cert_gained_every_rep": gained_all,  # reported; picks --permute rows
        "cert_wobble": {nm: cert[nm] for nm in wobble},
        "bimodal_off": {nm: cert[nm] for nm in bimodal},
        "solved_including_bimodal": solved_all,
        "bar1_checks": 2 * len(rows),
        "solved": solved,
        "both_certified": len(both),
        "nodes_both_certified": nodes,
        "population_open": len(pop),
        "dual_losses": dual_losses,
        "dual_loss_cap": cap,
        "incumbent_lost_on": lost_inc,
        "median_primal_gap": med_primal,
        "bar2": bar2,
        "bar2_pass": all(bar2.values()),
        "graduate": not problems and not lost_all and all(bar2.values()),
        "fired": fired,
        "per_instance": detail,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", default="")
    ap.add_argument("--on-value", default="1")
    ap.add_argument(
        "--off-value",
        default=None,
        help="set the flag to this in the OFF arm instead of removing it (default-ON flags)",
    )
    ap.add_argument("--fired", default="", help="regex over profile counter names")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--gap", type=float, default=P.GAP)
    ap.add_argument("--max-nodes", type=int, default=100_000_000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--instances", default="")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="solve with deterministic=True in both arms (bound the search with --max-nodes)",
    )
    ap.add_argument(
        "--rescore",
        default="",
        help="re-evaluate the rows of an existing panel file (.json or .json.gz) and exit",
    )
    ap.add_argument(
        "--permute",
        action="store_true",
        help="rep r >= 1 solves a row/column permutation of the file with seed r (both arms)",
    )
    ap.add_argument(
        "--merge",
        default="",
        help="with --rescore: replace the rescored rows for the names in these files (comma list)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep the (instance index, rep) rows already in --out and run the rest",
    )
    a = ap.parse_args(argv)
    if a.rescore:
        import gzip

        src = Path(a.rescore)
        if src.suffix == ".gz":
            with gzip.open(src, "rt") as fh:
                data = json.loads(fh.read())
        else:
            data = json.loads(src.read_text())
        rows = data["rows"]
        if a.merge:
            extra = [r for f in a.merge.split(",") for r in json.loads(Path(f).read_text())["rows"]]
            names = {r["name"] for r in extra}
            rows = [r for r in rows if r["name"] not in names] + extra
            print("merged permutation reps for", sorted(names), flush=True)
        new = evaluate(rows)
        old = data.get("verdict") or {}
        keys = ("graduate", "bar1_pass", "bar1b_cert_lost_every_rep",
                "bar1b_cert_gained_every_rep", "bimodal_off", "cert_wobble", "solved",
                "solved_including_bimodal", "nodes_both_certified", "dual_losses",
                "median_primal_gap", "bar2")  # fmt: skip
        print(json.dumps({"file": str(src), "stored_graduate": old.get("graduate"),
                          **{k: new[k] for k in keys}}, indent=1))  # fmt: skip
        return 0
    if not a.flag or not a.fired or not a.out:
        ap.error("--flag, --fired and --out are required unless --rescore is given")
    if a.deterministic:
        os.environ["DISCOPT_PANEL_DETERMINISTIC"] = "1"
    global FLAG, ON_VALUE, OFF_VALUE, _CTR, _CTR_NAME
    FLAG, ON_VALUE, OFF_VALUE = a.flag, a.on_value, a.off_value
    _CTR = re.compile(rf"^\s+({a.fired})\s+(\d+)$")
    _CTR_NAME = re.compile(a.fired)

    P._assert_discopt_under_test()
    hdr = P._header()
    pop = R.population()
    if a.instances:
        keep = set(a.instances.split(","))
        pop = [x for x in pop if x["name"] in keep]
    print("header", json.dumps(hdr), "instances", len(pop), flush=True)

    def one(job):
        k, mem, rep = job
        order = (False, True) if (k + rep) % 2 == 0 else (True, False)
        path = str(REPO / mem["path"])
        if a.permute:
            path = permuted_copy(path, rep, Path(a.out).parent / "perm_copies")
        got = {("on" if on else "off"): solve(path, on, a) for on in order}
        return k, mem, rep, got

    jobs = [(k, mem, rep) for rep in range(a.reps) for k, mem in enumerate(pop)]
    rows, executed = [], 0
    if a.resume and Path(a.out).exists():
        # Keyed by population INDEX, not name: the population has duplicate names.
        rows = [r for r in json.loads(Path(a.out).read_text())["rows"] if "k" in r]
        done = {(r["k"], r["rep"]) for r in rows}
        jobs = [j for j in jobs if (j[0], j[2]) not in done]
        print(f"resume: kept {len(rows)} rows, {len(jobs)} jobs left", flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for k, mem, rep, got in ex.map(one, jobs):
            executed += 2
            opt, maxim = mem["oracle_objective"], mem["maximize"]
            row = {"name": mem["name"], "k": k, "panel": mem.get("panel"), "rep": rep, "opt": opt,
                   "maximize": maxim, **got}  # fmt: skip
            row["problems"] = {arm: cert_problems(got[arm], opt, maxim) for arm in got}
            rows.append(row)
            o, n = got["off"], got["on"]
            print(f"[{len(rows)}/{len(jobs)}] {mem['name']:22s} r{rep} "
                  f"OFF {o['status']}/{o['gap_certified']} bnd={o['bound']} n={o['nodes']} "
                  f"| ON {n['status']}/{n['gap_certified']} bnd={n['bound']} n={n['nodes']} "
                  f"fired={n['fired']} load={o['load']:.1f}", flush=True)  # fmt: skip
            Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "rows": rows},
                                              indent=1) + "\n")  # fmt: skip
    verdict = evaluate(rows)
    Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "rows": rows,
                                       "verdict": verdict}, indent=1) + "\n")  # fmt: skip
    v = {k: verdict[k] for k in verdict if k != "per_instance"}
    print(json.dumps(v, indent=1), flush=True)
    print(f"d2 flag panel {FLAG}: executed={executed} fired={verdict['fired']}", flush=True)
    ok = executed and verdict["fired"]["on"] > 0 and verdict["fired"]["off"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
