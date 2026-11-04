"""Capture the HiGHS LPs of roles R3 / R4 / R6 as fixtures (highs-parity P0.5).

``docs/dev/highs-parity-plan.md`` P0.5. The hooks wrap, at runtime only (no solver code is
edited), the calls that hand an LP to HiGHS on the MINLP path:

* **R3** root cutting planes -- ``_root_cuts._solve_lp`` (the root LP) and
  ``_root_cuts.separate_gmi`` (the GMI cuts HiGHS's basis yields at that LP);
* **R4** convex node bound -- ``convex_node_bound._lp_dual`` (the slack-form node LP whose
  row dual feeds ``certify_lp_dual``);
* **R6** verified LPs on the MINLP path -- ``lp_milp_highs.solve_lp_std`` called from
  ``_linear_node_lp_rescue`` (#1611) or ``_verified_free_column_bound`` (#1869), and
  ``lp_milp_highs.solve_lp`` called from ``_linear_node_verified_empty`` (#1537). Calls
  from the pure-LP route (``_solve_lp_highs``, role R1) are not captured.

Every captured LP is stored in one row form -- ``min cᵀx + obj_const`` s.t.
``row_lo <= A x <= row_hi``, ``col_lo <= x <= col_hi`` -- with HiGHS's answer
(``objective`` of the row form, ``x``, row dual, status) and, for R3, the GMI cuts.

Subcommands:

``corpus RAW``   solve every in-repo corpus ``.nl`` (one subprocess each, hooks on) and
                 pickle each captured LP into ``RAW``.
``tests RAW``    run the #1537 / #1611 / #1869 regression files under pytest with the
                 hooks on (``-p capture_highs_parity_fixtures``).
``select RAW OUT``  pick fixtures (all R6, >= 50 R4, >= 20 R3 with cuts, spread over
                 instances), re-solve each with plain HiGHS (presolve off) and keep only
                 those whose objective it reproduces to 1e-9 relative, write
                 ``OUT/<role>__<name>.npz`` and ``OUT/manifest.json``; refuses > 5 MB.

Discipline (CLAUDE.md §6-§10): counters printed, non-zero exit when nothing was captured or
kept; hook exceptions propagate (a hook that cannot record raises into the solve rather
than hiding a gap in the capture).
"""

# ruff: noqa: N803, N806  (matrix names: A, A_ub, A_eq)
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "python" / "tests" / "data"
NL_DIRS = (DATA / "minlplib_nl", DATA / "minlplib")
R6_CALLERS = {
    "_linear_node_lp_rescue": "1611_rescue",
    "_verified_free_column_bound": "1869_free_column",
    "_linear_node_verified_empty": "1537_arbitration",
}
REGRESSION_TESTS = (  # the files that reach the R6 call sites (grep for their names)
    "python/tests/test_1537_invariance.py",
    "python/tests/test_1537_translation_rescaling_invariance.py",
    "python/tests/test_issue_1611_remainder.py",
    "python/tests/test_issue_1869_free_column_lp_cert.py",
    "python/tests/test_issue_1869_unbounded_branching.py",
    "python/tests/test_1966_prolog_root_lp_exactly_unbounded.py",
    "python/tests/test_928_round_cut_short_floor.py",
)
MAX_BYTES = 5 * 1024 * 1024
_HIGHS_INF = 1e20


# ----------------------------------------------------------------------------- row form
def _rows_from_std(sf) -> dict:
    """A ``StdForm`` (``A x = b``) as row-form arrays."""
    return {
        "c": np.asarray(sf.c, float),
        "obj_const": float(sf.obj_const),
        "A": sp.csr_matrix(sf.A),
        "row_lo": np.asarray(sf.b, float),
        "row_hi": np.asarray(sf.b, float),
        "col_lo": np.asarray(sf.xl, float),
        "col_hi": np.asarray(sf.xu, float),
    }


def _rows_from_ub_eq(c, A_ub, b_ub, A_eq, b_eq, lb, ub) -> dict:
    n = len(c)
    blocks, lo, hi = [], [], []
    if A_ub is not None and np.asarray(b_ub).size:
        blocks.append(sp.csr_matrix(A_ub))
        lo.append(np.full(len(b_ub), -np.inf))
        hi.append(np.asarray(b_ub, float))
    if A_eq is not None and np.asarray(b_eq).size:
        blocks.append(sp.csr_matrix(A_eq))
        lo.append(np.asarray(b_eq, float))
        hi.append(np.asarray(b_eq, float))
    A = sp.vstack(blocks).tocsr() if blocks else sp.csr_matrix((0, n))
    return {
        "c": np.asarray(c, float),
        "obj_const": 0.0,
        "A": A,
        "row_lo": np.concatenate(lo) if lo else np.zeros(0),
        "row_hi": np.concatenate(hi) if hi else np.zeros(0),
        "col_lo": np.asarray(lb, float),
        "col_hi": np.asarray(ub, float),
    }


def highs_resolve(rf: dict, time_limit: float = 60.0) -> tuple[str, float | None]:
    """Plain HiGHS (presolve off) on a row form: ``(status, objective incl. obj_const)``."""
    import highspy

    def inf(a):
        a = np.array(a, float, copy=True)
        a[a >= _HIGHS_INF] = highspy.kHighsInf
        a[a <= -_HIGHS_INF] = -highspy.kHighsInf
        return a

    A = sp.csc_matrix(rf["A"])
    n, m = len(rf["c"]), A.shape[0]
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = np.asarray(rf["c"], float)
    lp.offset_ = float(rf["obj_const"])
    lp.col_lower_, lp.col_upper_ = inf(rf["col_lo"]), inf(rf["col_hi"])
    lp.row_lower_, lp.row_upper_ = inf(rf["row_lo"]), inf(rf["row_hi"])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data
    h = highspy.Highs()
    for k, v in (("output_flag", False), ("presolve", "off"), ("time_limit", time_limit)):
        h.setOptionValue(k, v)
    if h.passModel(lp) == highspy.HighsStatus.kError:
        return "refused", None
    h.run()
    st = h.modelStatusToString(h.getModelStatus())
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return st, None
    return st, float(h.getInfo().objective_function_value)


# ----------------------------------------------------------------------------- hooks
_SEQ = [0]
_LAST_R3: dict = {}


def _dump(out: Path, role: str, rec: dict) -> None:
    _SEQ[0] += 1
    rec["role"] = role
    rec["instance"] = os.environ.get("HP_INSTANCE") or os.environ.get("PYTEST_CURRENT_TEST", "")
    with open(out / f"{role}__{os.getpid()}_{_SEQ[0]:06d}.pkl", "wb") as f:
        pickle.dump(rec, f)


def install_hooks(out: Path) -> None:
    import discopt.solvers._root_cuts as rc
    import discopt.solvers.convex_node_bound as cnb
    import discopt.solvers.lp_milp_highs as lmh

    out.mkdir(parents=True, exist_ok=True)
    real_solve_lp_root, real_gmi = rc._solve_lp, rc.separate_gmi
    real_lp_dual = cnb._lp_dual
    real_std, real_lp = lmh.solve_lp_std, lmh.solve_lp

    def solve_lp_root(root, cuts_a, cuts_b, time_limit=None):
        res = real_solve_lp_root(root, cuts_a, cuts_b, time_limit)
        obj, x, _duals, h = res
        if h is None:
            return res
        A_le = np.vstack([root.A_le] + ([np.array(cuts_a)] if cuts_a else []))
        b_le = np.concatenate([root.b_le] + ([np.array(cuts_b)] if cuts_b else []))
        c_min = -root.c if root.sense_max else root.c
        rf = _rows_from_ub_eq(c_min, A_le, b_le, root.A_eq, root.b_eq, root.lb, root.ub)
        _LAST_R3.clear()
        _LAST_R3.update(rf=rf, x=np.asarray(x, float), is_int=np.asarray(root.is_int, bool),
                        objective=float(h.getInfo().objective_function_value),
                        n_prior_cuts=len(cuts_a or []))  # fmt: skip
        return res

    def separate_gmi(root, h, x, a_all, b_all, *a, **k):
        cuts = real_gmi(root, h, x, a_all, b_all, *a, **k)
        if _LAST_R3:
            rec = dict(_LAST_R3)
            rec["gmi_cuts"] = [(np.asarray(ca, float), float(cb)) for ca, cb in (cuts or [])]
            _dump(out, "R3", rec)
            _LAST_R3.clear()
        return cuts

    def lp_dual(sf, J, xl, xu, s_lo, s_hi, grad):
        y, val, why = real_lp_dual(sf, J, xl, xu, s_lo, s_hi, grad)
        if y is not None:
            # ``_lp_dual``'s value is HiGHS's objective WITHOUT the slack form's
            # ``obj_const`` (measured: 100/100 sampled R4 LPs differ by exactly it);
            # the fixture's ``objective`` is the row form's, constant included.
            rec = {
                "rf": _rows_from_std(sf),
                "y": np.asarray(y, float),
                "status": "Optimal",
                "lp_dual_value": float(val),
                "objective": float(val) + float(sf.obj_const),
            }
            _dump(out, "R4", rec)
        return y, val, why

    def solve_lp_std(sf, *, time_limit=None):
        res = real_std(sf, time_limit=time_limit)
        caller = sys._getframe(1).f_code.co_name
        if caller in R6_CALLERS:
            _dump(out, "R6", {"rf": _rows_from_std(sf), "caller": R6_CALLERS[caller],
                              "status": res.status, "objective": res.objective,
                              "x": res.x, "y": res.row_dual})  # fmt: skip
        return res

    def solve_lp(*a, **k):
        res = real_lp(*a, **k)
        caller = sys._getframe(1).f_code.co_name
        if caller in R6_CALLERS:
            names = ("c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds")
            kw = dict(zip(names, a, strict=False)) | k
            bnds = kw.get("bounds") or [(None, None)] * len(kw["c"])
            lb = [-np.inf if lo is None else lo for lo, _ in bnds]
            ub = [np.inf if hi is None else hi for _, hi in bnds]
            rf = _rows_from_ub_eq(kw["c"], kw.get("A_ub"), kw.get("b_ub"), kw.get("A_eq"),
                                  kw.get("b_eq"), lb, ub)  # fmt: skip
            _dump(out, "R6", {"rf": rf, "caller": R6_CALLERS[caller],
                              "status": str(res.status), "objective": res.objective,
                              "x": res.x, "y": None})  # fmt: skip
        return res

    rc._solve_lp, rc.separate_gmi = solve_lp_root, separate_gmi
    cnb._lp_dual = lp_dual
    lmh.solve_lp_std, lmh.solve_lp = solve_lp_std, solve_lp


def pytest_configure(config):  # the ``tests`` subcommand loads this module as a plugin
    install_hooks(Path(os.environ["HP_CAPTURE_DIR"]))


# ----------------------------------------------------------------------------- corpus
def _corpus() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for d in NL_DIRS:
        for p in sorted(d.glob("*.nl")):
            paths.setdefault(p.stem, p)
    return paths


def cmd_corpus(a) -> int:
    out = Path(a.raw).resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = _corpus()
    names = sorted(paths) if not a.instances else a.instances.split(",")
    for i, name in enumerate(names, 1):
        before = len(glob.glob(str(out / "*.pkl")))
        env = dict(os.environ, HP_INSTANCE=name + a.label, PYTHONUNBUFFERED="1")
        env.update(kv.split("=", 1) for kv in a.env)
        t = time.perf_counter()
        try:
            p = subprocess.run(
                [sys.executable, str(HERE), "_solve", str(paths[name]), str(out),
                 str(a.time_limit)],
                env=env, capture_output=True, text=True, timeout=a.time_limit * 3 + 60,
            )  # fmt: skip
            rc, tail = p.returncode, (p.stdout.strip().splitlines() or [""])[-1]
        except subprocess.TimeoutExpired:
            rc, tail = "timeout", ""
        got = len(glob.glob(str(out / "*.pkl"))) - before
        print(f"[corpus {i}/{len(names)}] {name:24s} rc={rc} lps={got} "
              f"wall={time.perf_counter() - t:.1f}s {tail[:120]}", flush=True)  # fmt: skip
    total = len(glob.glob(str(out / "*.pkl")))
    print(f"corpus: instances={len(names)} captured={total}", flush=True)
    return 0 if total else 1


def _solve_one(path: str, out: str, time_limit: float) -> None:
    install_hooks(Path(out))
    from discopt.modeling.core import from_nl

    r = from_nl(path).solve(time_limit=time_limit)
    print(f"status={r.status} certified={r.gap_certified}", flush=True)


def cmd_tests(a) -> int:
    out = Path(a.raw).resolve()
    out.mkdir(parents=True, exist_ok=True)
    before = len(glob.glob(str(out / "*.pkl")))
    env = dict(os.environ, HP_CAPTURE_DIR=str(out), PYTHONPATH=str(HERE.parent))
    tests = [t for t in REGRESSION_TESTS if (REPO / t).exists()]
    print(f"tests: {tests} (missing: {sorted(set(REGRESSION_TESTS) - set(tests))})", flush=True)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o",
                        "addopts=", "-p", "capture_highs_parity_fixtures", *tests],
                       cwd=REPO, env=env)  # fmt: skip
    got = len(glob.glob(str(out / "*.pkl"))) - before
    print(f"tests: pytest rc={p.returncode} captured={got}", flush=True)
    return 0 if got else 1


# ----------------------------------------------------------------------------- select
def _npz_payload(rec: dict) -> dict:
    rf = rec["rf"]
    A = sp.csr_matrix(rf["A"])
    pay = {
        "c": rf["c"], "obj_const": np.float64(rf["obj_const"]),
        "A_data": A.data, "A_indices": A.indices.astype(np.int32),
        "A_indptr": A.indptr.astype(np.int32), "A_shape": np.array(A.shape, np.int64),
        "row_lo": rf["row_lo"], "row_hi": rf["row_hi"],
        "col_lo": rf["col_lo"], "col_hi": rf["col_hi"],
        "objective": np.float64(np.nan if rec["objective"] is None else rec["objective"]),
        "expected_status": np.array(expected_status(rec) or ""),
    }  # fmt: skip
    for k in ("x", "y"):
        if rec.get(k) is not None:
            pay[k] = np.asarray(rec[k], float)
    if rec["role"] == "R3":
        pay["is_int"] = rec["is_int"]
        cuts = rec["gmi_cuts"]
        pay["gmi_a"] = np.array([c for c, _ in cuts], float).reshape(len(cuts), len(rf["c"]))
        pay["gmi_b"] = np.array([b for _, b in cuts], float)
    return pay


def load_fixture(path) -> dict:
    """A fixture's row form + answers (the loader the P0.5 test uses)."""
    d = np.load(path)
    rf = {
        "c": d["c"], "obj_const": float(d["obj_const"]),
        "A": sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]),
                           shape=tuple(int(s) for s in d["A_shape"])),
        "row_lo": d["row_lo"], "row_hi": d["row_hi"], "col_lo": d["col_lo"], "col_hi": d["col_hi"],
    }  # fmt: skip
    extra = {k: d[k] for k in d.files if k not in ("c", "obj_const", "A_data", "A_indices",
                                                   "A_indptr", "A_shape", "row_lo", "row_hi",
                                                   "col_lo", "col_hi")}  # fmt: skip
    return {"rf": rf, **extra}


def expected_status(rec: dict) -> str | None:
    """The HiGHS status a verdict-only record (no objective) must reproduce, else None."""
    if rec.get("objective") is not None:
        return None
    st = str(rec.get("status", "")).lower()
    if "infeasible" in st:
        return "Infeasible"
    if "unbounded" in st:
        return "Unbounded"
    raise ValueError(f"record with no objective and status {rec.get('status')!r}")


def _dedupe(xs: list) -> list:
    """Drop repeats of the same LP and answer (tests re-capture identical arbitrations)."""
    import hashlib

    seen, out = set(), []
    for x in xs:
        rf = x["rf"]
        A = sp.csr_matrix(rf["A"])
        h = hashlib.sha256()
        for arr in (rf["c"], A.data, A.indices, A.indptr, rf["row_lo"], rf["row_hi"],
                    rf["col_lo"], rf["col_hi"]):  # fmt: skip
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(repr((x.get("caller"), str(x.get("status")), x.get("objective"))).encode())
        if h.hexdigest() not in seen:
            seen.add(h.hexdigest())
            out.append(x)
    return out


def reproduces(rec_obj: float, obj: float | None) -> bool:
    return obj is not None and abs(obj - rec_obj) <= 1e-9 * max(1.0, abs(rec_obj))


def cmd_select(a) -> int:
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for f in sorted(glob.glob(str(raw / "*.pkl"))):
        with open(f, "rb") as fh:
            rec = pickle.load(fh)
        rec["src"] = os.path.basename(f)
        if rec["role"] == "R4" and "lp_dual_value" not in rec:
            # Captured before the hook added the constant (see ``lp_dual`` above).
            rec["lp_dual_value"] = rec["objective"]
            rec["objective"] = rec["objective"] + float(rec["rf"]["obj_const"])
        recs.append(rec)
    by_role = {r: [x for x in recs if x["role"] == r] for r in ("R3", "R4", "R6")}
    print("captured:", {k: len(v) for k, v in by_role.items()}, flush=True)
    picked = {
        "R6": _dedupe(by_role["R6"]),
        "R4": _spread(by_role["R4"], a.r4),
        "R3": _spread([x for x in by_role["R3"] if x["gmi_cuts"]], a.r3),
    }
    manifest, kept, dropped, size, hard_entries = [], {}, [], 0, []
    for role, xs in picked.items():
        for i, rec in enumerate(xs):
            st, obj = highs_resolve(rec["rf"])
            inst = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in rec["instance"])[:60]
            name = f"{role}__{inst}__{i:03d}"
            want = expected_status(rec)
            ok = st == want if want else reproduces(rec["objective"], obj)
            if not ok:
                # Plain HiGHS cannot reproduce what the verified route returned: these
                # are the hard LPs (huge bounds), kept apart under hard/ with HiGHS's
                # own answer so a port is judged on them too.
                dropped.append((role, rec["instance"], st, rec["objective"], obj))
                hard_entries.append({"file": f"hard/{name}.npz", "role": role,
                                     "instance": rec["instance"], "caller": rec.get("caller"),
                                     "plain_highs_status": st, "route_objective": rec["objective"],
                                     "plain_highs_objective": obj, "n": int(len(rec["rf"]["c"])),
                                     "m": int(rec["rf"]["A"].shape[0])})  # fmt: skip
                hard = out / "hard"
                hard.mkdir(exist_ok=True)
                np.savez_compressed(hard / f"{name}.npz", **_npz_payload(rec),
                                    plain_highs_status=np.array(st))  # fmt: skip
                size += (hard / f"{name}.npz").stat().st_size
                continue
            np.savez_compressed(out / f"{name}.npz", **_npz_payload(rec))
            size += (out / f"{name}.npz").stat().st_size
            kept[role] = kept.get(role, 0) + 1
            manifest.append({"file": f"{name}.npz", "role": role, "instance": rec["instance"],
                             "caller": rec.get("caller"), "objective": rec["objective"],
                             "n": int(len(rec["rf"]["c"])), "m": int(rec["rf"]["A"].shape[0]),
                             "gmi_cuts": len(rec.get("gmi_cuts") or [])})  # fmt: skip
    doc = {"fixtures": manifest, "hard": hard_entries}
    (out / "manifest.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"select: kept={kept} dropped={len(dropped)} bytes={size}", flush=True)
    for d in dropped:
        print("  dropped", d, flush=True)
    if size > MAX_BYTES:
        raise SystemExit(f"fixtures are {size} bytes > {MAX_BYTES}")
    return 0 if kept else 1


def _spread(xs: list, k: int) -> list:
    """Up to ``k`` records, round-robin over instances (so one instance cannot dominate)."""
    by: dict[str, list] = {}
    for x in xs:
        by.setdefault(x["instance"], []).append(x)
    out, i = [], 0
    while len(out) < k and any(i < len(v) for v in by.values()):
        for v in by.values():
            if i < len(v) and len(out) < k:
                out.append(v[i])
        i += 1
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("raw")
    c.add_argument("--time-limit", type=float, default=20.0)
    c.add_argument("--instances", default="")
    c.add_argument("--env", action="append", default=[], help="K=V set in every solve")
    c.add_argument("--label", default="", help="suffix on the recorded instance name")
    t = sub.add_parser("tests")
    t.add_argument("raw")
    s = sub.add_parser("select")
    s.add_argument("raw")
    s.add_argument("out")
    s.add_argument("--r4", type=int, default=60)
    s.add_argument("--r3", type=int, default=30)
    a = p.parse_args(argv)
    return {"corpus": cmd_corpus, "tests": cmd_tests, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_solve":
        _solve_one(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    else:
        sys.exit(main())
