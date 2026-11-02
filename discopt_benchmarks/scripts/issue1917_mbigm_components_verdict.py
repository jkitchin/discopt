#!/usr/bin/env python3
"""#1917: apply the pre-registered §5 rule for ``DISCOPT_MBIGM_LP_COMPONENTS``.

The rule is ``docs/dev/data/1917-flag-panel/DISCOPT_MBIGM_LP_COMPONENTS.prereg.md``.
It was written and committed before any panel data existed, and this script was
committed with it. It reads three row sets:

* the lowering panel L (``issue1917_mbigm_components_lowering.py --phase lower``);
* the soundness audit A (the same script, ``--phase audit``);
* the solve panel S (``issue1917_gdp_flag_panel.py --phase solve``).

It then prints V / C / N / K and the outcome.

Exit codes: 0 graduate, 3 retire, 1 the flag stays OFF on a C violation,
2 the run is void (V).

    python discopt_benchmarks/scripts/issue1917_mbigm_components_verdict.py \\
        --lower <L>.jsonl --audit <A>.jsonl --solve <S>.jsonl --guard <V7>.json \\
        [--tree <worktree>]
    python discopt_benchmarks/scripts/issue1917_mbigm_components_verdict.py --self-test
"""

# ruff: noqa: N803, N806, C408 -- L / A / S name the pre-registration's three panels
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

FLAG = "DISCOPT_MBIGM_LP_COMPONENTS"
N_INST = 24
REPS = 2
ARMS = ("off", "on")
#: Lowerings shorter than this in both arms are below timing resolution on a
#: loaded machine; they count for neither N1 nor N2.
MIN_LOWER_S = 1.0
FASTER, SLOWER = 0.8, 1.25
#: The solve panel's per-row cap (time limit + grace); K3 caps elapsed here.
SOLVE_CAP_S = 720.0


def _rows(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _in_tree(path, tree: Path) -> bool:
    return bool(path) and str(path).startswith(str(tree))


def _gp():
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO / "discopt_benchmarks"))
    import issue1917_gdp_flag_panel as gp

    return gp


def collect(lower: list[dict], audit: list[dict], solve: list[dict], tree: Path, oracle, crosses):
    """Normalise the three row sets. ``oracle`` maps instance -> oracle record and
    ``crosses(row, oracle_rec)`` is ``issue1917_gdp_flag_panel._bound_crosses``;
    both are parameters so the self-test can run without the GDPlib oracle."""
    v: list[str] = []
    c: list[str] = []
    L: dict = {}
    for r in lower:
        if r.get("phase") != "lower" or r.get("flag") != FLAG:
            continue
        L.setdefault(r["instance"], {"off": {}, "on": {}})[r["arm"]][r["rep"]] = r
        if r["status"] == "lowered" and not _in_tree(r.get("discopt"), tree):
            v.append(f"V1 L {r['instance']} {r['arm']} rep{r['rep']}: {r.get('discopt')}")
        if r["status"] == "error":
            if r["arm"] == "off":
                v.append(f"V5 L {r['instance']} off rep{r['rep']} errored: {r.get('error')}")
            else:
                c.append(f"C6 L {r['instance']} on rep{r['rep']} errored: {r.get('error')}")
    A: dict = {}
    for r in audit:
        if r.get("phase") != "audit" or r.get("flag") != FLAG:
            continue
        A[r["instance"]] = r
        if r["status"] == "error":
            c.append(f"C6 A {r['instance']} errored: {r.get('error')}")
        elif r["status"] != "lowered":
            v.append(f"V4 A {r['instance']} {r['status']}")
        else:
            if not _in_tree(r.get("discopt"), tree):
                v.append(f"V1 A {r['instance']}: {r.get('discopt')}")
            for bad in r["audit"]["violations"]:
                c.append(f"C5 A {r['instance']}: sub-LP {bad[0]} > full LP {bad[1]}")
    S: dict = {}
    feas = 0
    for r in solve:
        if r.get("phase") != "solve" or r.get("flag") != FLAG:
            continue
        S.setdefault(r["instance"], {"off": {}, "on": {}})[r["arm"]][r["rep"]] = r
        no_path = r.get("status") == "killed" or "error" in r  # neither records one
        if not no_path and not _in_tree(r.get("discopt"), tree):
            v.append(f"V1 S {r['instance']} {r['arm']} rep{r['rep']}: {r.get('discopt')}")
        if r["arm"] == "on" and r.get("in_pyomo") is not None:
            feas += 1
        if r.get("verdict") == "incorrect":
            c.append(f"C1 S {r['instance']} {r['arm']} rep{r['rep']}: incorrect")
        if crosses(r, oracle.get(r["instance"])):
            c.append(f"C2 S {r['instance']} {r['arm']} rep{r['rep']}: bound {r.get('bound')}")

    n_l = sum(len(x) for a in L.values() for x in a.values())
    n_s = sum(len(x) for a in S.values() for x in a.values())
    if n_l != N_INST * 2 * REPS:
        v.append(f"V2 L rows {n_l} != {N_INST * 2 * REPS}")
    if n_s != N_INST * 2 * REPS:
        v.append(f"V2 S rows {n_s} != {N_INST * 2 * REPS}")
    if len(A) != N_INST:
        v.append(f"V2 A rows {len(A)} != {N_INST}")
    for name, panel in (("L", L), ("S", S)):
        gaps = [k for k, a in panel.items() if any(len(a[x]) != REPS for x in ARMS)]
        if gaps:
            v.append(f"V2 {name}: {len(gaps)} instances missing a rep: {gaps[:3]}")
    if feas == 0:
        v.append("V3 no ON solve incumbent was Pyomo-checked")
    checked = sum(r["audit"]["checked"] for r in A.values() if r["status"] == "lowered")
    if checked == 0:
        v.append("V4 the audit re-solved no sub-LP")
    for inst, a in L.items():
        for arm in ARMS:
            ms = [a[arm][r].get("m") for r in a[arm] if a[arm][r]["status"] == "lowered"]
            if len(ms) == REPS and ms[0] != ms[1]:
                v.append(f"V6 L {inst} {arm}: M's differ between reps (nondeterministic)")
    return L, A, S, v, c, checked


def _lower_s(rec: dict, timeout_value: bool = True):
    if rec["status"] == "lowered":
        return rec["lower_s"]
    if rec["status"] == "killed" and timeout_value:
        return rec["timeout"]
    return None


def judge(L, A, S, c: list[str]) -> tuple[list[str], list[str], dict]:
    """Return (C violations, N failures, report) given validated panels."""
    c = list(c)
    rep: dict = {}
    # C3 / C4 from S
    for inst, a in S.items():
        for r in range(REPS):
            if a["off"][r].get("verdict") == "solved" and a["on"][r].get("verdict") != "solved":
                c.append(f"C3 S {inst} rep{r}: solved OFF, not ON")
        if all("error" in a["on"][r] for r in range(REPS)) and not any(
            "error" in a["off"][r] for r in range(REPS)
        ):
            c.append(f"C4 S {inst}: ON errors in both reps, OFF in neither")
    # C7: the lowering's structure (number of LP M's) must not change.
    m_cmp = {"equal": 0, "smaller": 0, "larger": 0}
    m_by_inst = {}
    for inst, a in L.items():
        for r in range(REPS):
            off, on = a["off"][r], a["on"][r]
            if off["status"] == on["status"] == "lowered" and off["n_m"] != on["n_m"]:
                c.append(f"C7 L {inst} rep{r}: n_m OFF {off['n_m']} != ON {on['n_m']}")
        r0 = [(a["off"][r], a["on"][r]) for r in range(REPS)]
        pair = next(((o, n) for o, n in r0 if o["status"] == n["status"] == "lowered"), None)
        if pair is not None and pair[0]["n_m"] == pair[1]["n_m"]:
            cnt = {"equal": 0, "smaller": 0, "larger": 0}
            for mo, mn in zip(pair[0]["m"], pair[1]["m"], strict=True):
                fo, fn = float.fromhex(mo), float.fromhex(mn)
                cnt["equal" if fo == fn else ("smaller" if fn < fo else "larger")] += 1
            m_by_inst[inst] = cnt
            for k in cnt:
                m_cmp[k] += cnt[k]
    rep["m_cmp"] = m_cmp

    n: list[str] = []
    # N1 / N2: lowering wall, both reps
    faster, slower = [], []
    for inst, a in L.items():
        fast_all = slow_all = True
        for r in range(REPS):
            off, on = a["off"][r], a["on"][r]
            t_off, t_on = _lower_s(off), _lower_s(on)
            if on["status"] != "lowered" or t_off is None:
                fast_all = False
            elif off["status"] == "killed":
                pass  # ON lowered inside the timeout, OFF did not: faster
            elif max(t_off, t_on) < MIN_LOWER_S or t_on > FASTER * t_off:
                fast_all = False
            if off["status"] == "lowered" and on["status"] == "killed":
                pass  # OFF lowered, ON did not: slower
            elif (
                t_off is None
                or t_on is None
                or max(t_off, t_on) < MIN_LOWER_S
                or t_on < SLOWER * t_off
            ):
                slow_all = False
        if fast_all:
            faster.append(inst)
        if slow_all:
            slower.append(inst)
    rep["faster"], rep["slower"] = faster, slower
    if len(faster) < 2:
        n.append(f"N1 lowering faster on {len(faster)} instances (< 2): {faster}")
    if slower:
        n.append(f"N2 lowering slower on {slower}")
    # N3: M quality
    worse_m = [k for k, x in m_by_inst.items() if x["larger"] > x["smaller"]]
    if worse_m:
        n.append(f"N3 more M's larger than smaller ON on {worse_m}")

    # N4 / N5 from S
    def inc(x):
        return "error" not in x and isinstance(x.get("objective"), (int, float))

    lost = [
        k
        for k, a in S.items()
        if all(inc(a["off"][r]) for r in range(REPS))
        and not any(inc(a["on"][r]) for r in range(REPS))
    ]
    solved = {
        arm: [
            k
            for k, a in S.items()
            if all(a[arm][r].get("verdict") == "solved" for r in range(REPS))
        ]
        for arm in ARMS
    }
    rep["lost"], rep["solved"] = lost, {k: len(x) for k, x in solved.items()}
    if lost:
        n.append(f"N4 incumbents lost on {lost}")
    if len(solved["on"]) < len(solved["off"]):
        n.append(f"N4 solved in both reps ON {len(solved['on'])} < OFF {len(solved['off'])}")
    ratios = []
    for k in set(solved["on"]) & set(solved["off"]):
        med = {
            arm: statistics.median(S[k][arm][r].get("node_count") or 0 for r in range(REPS))
            for arm in ARMS
        }
        ratios.append((med["on"] + 1) / (med["off"] + 1))
    gm = math.exp(statistics.fmean(math.log(x) for x in ratios)) if ratios else None
    rep["node_gm"], rep["node_cells"] = gm, len(ratios)
    if gm is None or gm > 1.05:
        n.append(f"N5 node geomean ratio {gm} over {len(ratios)} cells (> 1.05 or none)")

    # K3: solve elapsed SGM, per rep
    k3 = []
    for r in range(REPS):
        sgm = {}
        for arm in ARMS:
            ws = [min(float(a[arm][r]["elapsed"]), SOLVE_CAP_S) for a in S.values()]
            sgm[arm] = math.exp(statistics.fmean(math.log(w + 10) for w in ws)) - 10
        k3.append((sgm["off"], sgm["on"], sgm["on"] / sgm["off"]))
    rep["k3"] = k3
    return c, n, rep


def decide(v, c, n, rep) -> int:
    if v:
        print("VOID:\n  " + "\n  ".join(v))
        return 2
    print(f"M's ON vs OFF (first rep both arms lowered): {rep['m_cmp']}")
    print(f"lowering faster (both reps): {rep['faster']}")
    print(f"lowering slower (both reps): {rep['slower']}")
    print(f"solved in both reps: {rep['solved']}; incumbents lost: {rep['lost']}")
    print(f"node geomean ON/OFF over {rep['node_cells']} cells: {rep['node_gm']}")
    for r, (off, on, ratio) in enumerate(rep["k3"]):
        print(f"rep{r} SGM solve elapsed OFF {off:.2f} s, ON {on:.2f} s, ratio {ratio:.3f}")
    if c:
        print("NOT CERT-CLEAN (K1: stays OFF, root-cause):\n  " + "\n  ".join(c))
        return 1
    if all(x[2] > SLOWER for x in rep["k3"]):
        print("K3: solve SGM ratio > 1.25 in both reps -> RETIRE")
        return 3
    if n:
        print("cert-clean, NOT net-positive -> RETIRE:\n  " + "\n  ".join(n))
        return 3
    print("cert-clean and net-positive -> GRADUATE")
    return 0


def guard_failures(guard: dict | None, tree: Path) -> list[str]:
    """V7: the corpus guard (``issue1917_mbigm_components_corpus_guard.py``)."""
    if guard is None:
        return ["V7 corpus guard missing"]
    v = [f"V7 {f}" for f in guard.get("failures", [])]
    if not guard.get("executed") or not guard.get("parsed"):
        v.append("V7 corpus guard executed nothing")
    if not (guard.get("control") or {}).get("calls"):
        v.append("V7 positive control recorded no _compute_big_m_lp call")
    if any(s.get("calls") for s in guard.get("corpus_solves", [])) or not guard.get(
        "corpus_solves"
    ):
        v.append("V7 corpus solves missing or reached _compute_big_m_lp")
    if guard.get("with_gdp"):
        v.append(f"V7 corpus files with GDP: {guard['with_gdp']}")
    if not _in_tree(guard.get("discopt"), tree):
        v.append(f"V7/V1 guard ran another tree: {guard.get('discopt')}")
    return v


def run(lower, audit, solve, tree, oracle, crosses, guard=None) -> int:
    L, A, S, v, c, checked = collect(lower, audit, solve, tree, oracle, crosses)
    v = v + guard_failures(guard, tree)
    print(f"== {FLAG}: L {len(L)} instances, A {len(A)} ({checked} sub-LPs re-solved), S {len(S)}")
    if v:
        return decide(v, c, [], {})
    c, n, rep = judge(L, A, S, c)
    return decide(v, c, n, rep)


# ---------------------------------------------------------------- self-test


def _synthetic(tree: str):
    """A clean panel on which the flag graduates: two instances lower 5x faster,
    one OFF lowering is killed, everything else is identical."""
    m = [float(x).hex() for x in (1.0, 2.0, 3.0)]
    lower, audit, solve = [], [], []
    for i in range(N_INST):
        inst = f"i{i}"
        audit.append(
            dict(
                phase="audit",
                flag=FLAG,
                instance=inst,
                arm="on",
                rep=0,
                status="lowered",
                discopt=tree,
                audit={"checked": 3, "skipped_cap": 0, "violations": []},
            )
        )
        for r in range(REPS):
            for arm in ARMS:
                t = 10.0 if arm == "off" else (2.0 if i < 2 else 10.0)
                row = dict(
                    phase="lower",
                    flag=FLAG,
                    instance=inst,
                    arm=arm,
                    rep=r,
                    status="lowered",
                    lower_s=t,
                    n_m=3,
                    m=list(m),
                    discopt=tree,
                    timeout=900.0,
                )
                if i == 2 and arm == "off":
                    row = dict(row, status="killed")
                    del row["lower_s"], row["m"], row["n_m"], row["discopt"]
                lower.append(row)
                solve.append(
                    dict(
                        phase="solve",
                        flag=FLAG,
                        instance=inst,
                        arm=arm,
                        rep=r,
                        verdict="solved",
                        objective=1.0,
                        bound=1.0,
                        node_count=7,
                        elapsed=20.0,
                        discopt=tree,
                        in_pyomo={},
                    )
                )
    return lower, audit, solve


def self_test() -> int:
    tree = "/T/python/discopt/__init__.py"
    T = Path("/T")
    never = lambda row, o: False  # noqa: E731
    cases = []

    def good_guard():
        return {
            "discopt": tree,
            "failures": [],
            "parsed": 74,
            "constraints": 9000,
            "with_gdp": [],
            "corpus_solves": [{"instance": "a.nl", "calls": 0}],
            "control": {"instance": "pyomo.eight_process", "calls": 9},
            "executed": 76,
        }

    def case(name, mutate, want, guard_mutate=None):
        lo, au, so = _synthetic(tree)
        mutate(lo, au, so)
        g = good_guard()
        if guard_mutate is not None:
            g = guard_mutate(g)
        cases.append((name, run(lo, au, so, T, {}, never, g), want))

    case("clean -> graduate", lambda lo, au, so: None, 0)

    def drop(lo, au, so):
        lo.pop()

    case("missing L row -> void", drop, 2)

    def foreign(lo, au, so):
        so[0]["discopt"] = "/main/python/discopt/__init__.py"

    case("row from another tree -> void", foreign, 2)

    def unsound(lo, au, so):
        au[5]["audit"]["violations"] = [[1.0, 0.5]]

    case("audit violation -> stays OFF", unsound, 1)

    def incorrect(lo, au, so):
        so[1]["verdict"] = "incorrect"

    case("incorrect solve -> stays OFF", incorrect, 1)

    def lost_cert(lo, au, so):
        for s in so:
            if s["instance"] == "i7" and s["arm"] == "on" and s["rep"] == 0:
                s["verdict"] = "time_limit"

    case("certification lost ON -> stays OFF", lost_cert, 1)

    def n_m(lo, au, so):
        for x in lo:
            if x["instance"] == "i9" and x["arm"] == "on":
                x["n_m"], x["m"] = 2, x["m"][:2]

    case("lowering structure changed -> stays OFF", n_m, 1)

    def one_fast(lo, au, so):
        m = [float(v).hex() for v in (1.0, 2.0, 3.0)]
        for x in lo:
            if x["instance"] == "i1" and x["arm"] == "on":
                x["lower_s"] = 10.0  # i1 no longer faster
            if x["instance"] == "i2" and x["arm"] == "off":  # i2 no longer killed
                x.update(status="lowered", lower_s=10.0, n_m=3, m=list(m), discopt=tree)

    case("faster on one instance only -> retire", one_fast, 3)

    def flip(lo, au, so):
        for x in lo:
            if x["instance"] == "i1" and x["arm"] == "on" and x["rep"] == 1:
                x["lower_s"] = 10.0

    case("a rep-1 flip is not a gain (i0 + killed i2 still 2) -> graduate", flip, 0)

    def slow(lo, au, so):
        for x in lo:
            if x["instance"] == "i5" and x["arm"] == "on":
                x["lower_s"] = 13.0

    case("lowering 1.3x slower in both reps -> retire", slow, 3)

    def larger_m(lo, au, so):
        for x in lo:
            if x["instance"] == "i6" and x["arm"] == "on":
                x["m"] = [float(v).hex() for v in (1.0, 2.0, 4.0)]

    case("an M larger ON with none smaller -> retire", larger_m, 3)

    def k3(lo, au, so):
        for s in so:
            if s["arm"] == "on":
                s["elapsed"] = 40.0

    case("solve SGM ratio > 1.25 -> retire", k3, 3)

    def nop(lo, au, so):
        pass

    case("V7 guard missing -> void", nop, 2, lambda g: None)
    case(
        "V7 corpus solve reached the LP -> void",
        nop,
        2,
        lambda g: dict(g, corpus_solves=[{"instance": "a.nl", "calls": 3}]),
    )
    case(
        "V7 positive control saw no call -> void",
        nop,
        2,
        lambda g: dict(g, control={"instance": "x", "calls": 0}),
    )

    bad = [(nm, got, want) for nm, got, want in cases if got != want]
    print(f"\nself-test: {len(cases)} cases executed, {len(bad)} wrong")
    for nm, got, want in bad:
        print(f"  WRONG {nm}: exit {got}, expected {want}")
    if not cases:
        return 2
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lower", type=Path)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--solve", type=Path)
    ap.add_argument("--guard", type=Path, help="V7 corpus-guard JSON")
    ap.add_argument("--tree", type=Path, default=REPO, help="worktree the rows must come from")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not (args.lower and args.audit and args.solve and args.guard):
        ap.error("--lower, --audit, --solve and --guard are required")
    gp = _gp()
    return run(
        _rows(args.lower),
        _rows(args.audit),
        _rows(args.solve),
        args.tree,
        gp.gb_oracle(),
        gp._bound_crosses,
        json.loads(args.guard.read_text()),
    )


if __name__ == "__main__":
    sys.exit(main())
