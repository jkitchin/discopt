"""Master baselines for highs-parity P0.4 (R2a / R2b / R2c): HiGHS vs the Rust simplex.

``docs/dev/highs-parity-plan.md`` P0.4. Every master MILP here is replayed through both
engines with identical arguments (the captured ``solve_milp`` keyword arguments):
``highs`` = ``milp_highs.solve_milp`` (what the convex-route and decomposition masters
use today), ``simplex`` = ``milp_simplex.solve_milp`` (the in-house Rust driver).

Subcommands:

``fixtures DIR``      write the checked-in master fixtures (``python/tests/data``: the
                      ``oa_masters`` set, the #2069 / #2107 clay and rsyn masters) as
                      captured-keyword pickles into ``DIR``.
``capture-decomp DIR``  run the decomposition test files under pytest with a plugin that
                      records every ``get_decomposition_master_solver`` master (R2b).
``capture-gdp DIR``   capture every LOA master of the ``gdp_parity`` panel (R2c), via
                      ``gdp_master_backend_entry.py capture`` (one process per instance).
``replay DIR... --out F``  replay each master on both engines, interleaved with rotated
                      order, ``--rounds`` repeats. Each engine's returned point is checked
                      by this script (rows, bounds, integrality, densely); every verified
                      incumbent is held against every engine's bound. Writes ``F`` after
                      every master.
``route --out F``     the §76.4 route panel end to end, single arm (the shipped default
                      HiGHS master): every in-repo ``.nl`` the convex router diverts, rows
                      scaled by ``10**U(-s, s)`` for each span, plus the #1658 portfolio
                      family. Incumbents verified with ``verify_point``; bounds checked
                      against ``minlplib.solu`` and the verified incumbent.
``decomp-e2e --out F``  run the decomposition test files with ``DISCOPT_DECOMP_MASTER`` set
                      to each engine; per-test outcome per engine.

Discipline (CLAUDE.md §6-§10): every subcommand prints an executed counter and exits
non-zero when it is 0; exceptions propagate (a replay engine that raises is recorded *and*
fails the run); progress is per item, unbuffered.
"""

# ruff: noqa: N803, N806, N812  (matrix names follow the solve_milp keywords: A_ub, A_eq)
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "python" / "tests" / "data"
ENGINES = ("highs", "simplex")
GAP = 1e-4
FEAS_TOL = 1e-6
INT_TOL = 1e-5

#: The decomposition test files whose solves build a master through
#: ``get_decomposition_master_solver`` (R2b).
DECOMP_TESTS = (
    "python/tests/test_benders.py",
    "python/tests/test_benders_soundness.py",
    "python/tests/test_gbd.py",
    "python/tests/test_946_gbd_degenerate_multiplier.py",
    "python/tests/test_986_decomposition_master_engine.py",
    "python/tests/test_1611_benders_purge_certificate.py",
    "python/tests/test_1611_lagrangian.py",
    "python/tests/test_1657_benders_array_sum_linearity.py",
    "python/tests/test_decomposition_solve_equivalence.py",
    "python/tests/test_decomposition_adversarial.py",
)


def _assert_discopt_under_test():
    import discopt

    want = str(REPO / "python")
    if not discopt.__file__.startswith(want):
        raise RuntimeError(f"loaded discopt from {discopt.__file__}, expected under {want}")


def _header() -> dict:
    import highspy

    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {"commit": commit, "load": os.getloadavg(), "highs": highspy.Highs().version()}


# ----------------------------------------------------------------------------- fixtures
def _csr(d, prefix, ind="indices", ptr="indptr"):
    if f"{prefix}_data" not in d.files:
        return None
    shape = tuple(int(s) for s in d[f"{prefix}_shape"])
    return sp.csr_matrix(
        (d[f"{prefix}_data"], d[f"{prefix}_{ind}"], d[f"{prefix}_{ptr}"]), shape=shape
    )


def _bounds(lb, ub):
    return [
        (None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi))
        for lo, hi in zip(lb, ub, strict=True)
    ]


def _ranged_to_ub_eq(A, lo, hi):
    """``lo <= A x <= hi`` as ``A_ub x <= b_ub`` and ``A_eq x == b_eq`` (exact row copies)."""
    A = sp.csr_matrix(A)
    eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    up = np.isfinite(hi) & ~eq
    dn = np.isfinite(lo) & ~eq
    A_ub = sp.vstack([A[np.flatnonzero(up)], -A[np.flatnonzero(dn)]]).tocsr()
    b_ub = np.concatenate([hi[up], -lo[dn]])
    A_eq = A[np.flatnonzero(eq)] if eq.any() else None
    return A_ub, b_ub, A_eq, (lo[eq] if eq.any() else None)


def fixture_kw(path: Path) -> dict:
    """The ``solve_milp`` keyword arguments of one checked-in master fixture."""
    if path.suffix == ".gz":  # an .mps.gz master (#2069)
        sys.path.insert(0, str(REPO / "discopt_benchmarks" / "scripts"))
        import lp_milp_loader as L

        inst = L.read_instance(str(path))
        if inst.maximize or inst.offset:
            raise ValueError(f"{path.name}: expected a minimisation with no offset")
        A_ub, b_ub, A_eq, b_eq = _ranged_to_ub_eq(inst.A, inst.row_lo, inst.row_hi)
        return {
            "c": inst.c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": _bounds(inst.col_lo, inst.col_hi),
            "integrality": inst.is_int.astype(int),
        }
    d = np.load(path)
    if "row_lo" in d.files:  # issue2107 layout: CSR + row ranges
        n = d["c"].shape[0]
        A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=(d["row_lo"].shape[0], n))
        A_ub, b_ub, A_eq, b_eq = _ranged_to_ub_eq(A, d["row_lo"], d["row_hi"])
        kw = {
            "c": d["c"],
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": _bounds(d["lb"], d["ub"]),
            "integrality": d["integrality"],
        }
        if "x_seed" in d.files:
            kw["mip_start"] = d["x_seed"]
        return kw
    if "integ" in d.files:  # milp_default_box layout
        A_ub = sp.csr_matrix(
            (d["A_ub_data"], d["A_ub_ind"], d["A_ub_ptr"]),
            shape=tuple(int(s) for s in d["A_ub_shape"]),
        )
        A_eq = d["A_eq"]
        A_eq = None if A_eq.size == 0 else sp.csr_matrix(A_eq)
        return {
            "c": d["c"],
            "A_ub": A_ub,
            "b_ub": d["b_ub"],
            "A_eq": A_eq,
            "b_eq": None if A_eq is None else d["b_eq"],
            "bounds": _bounds(d["lb"], d["ub"]),
            "integrality": d["integ"],
        }
    kw = {"c": d["c"], "bounds": _bounds(d["lb"], d["ub"]), "integrality": d["integrality"]}
    for tag, rhs in (("A_ub", "b_ub"), ("A_eq", "b_eq")):
        m = _csr(d, tag)
        if m is not None:
            kw[tag], kw[rhs] = m, d[rhs]
    if "gap_tolerance" in d.files:
        kw["gap_tolerance"] = float(d["gap_tolerance"])
    if "mip_start" in d.files:
        kw["mip_start"] = d["mip_start"]
    return kw


FIXTURES = (
    "oa_masters/tls2_master0.npz",
    "oa_masters/rsyn0830m_master0.npz",
    "oa_masters/fac2_master0.npz",
    "oa_masters/m6_final_master_2115.npz",
    "issue2069/clay0203_master.mps.gz",
    "milp_default_box_clay0203_master.npz",
    "issue2107/clay0204hfsg_master.npz",
    "issue2107/rsyn0820m_master.npz",
)


def cmd_fixtures(a) -> int:
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for rel in FIXTURES:
        kw = fixture_kw(DATA / rel)
        name = "fixture__" + Path(rel).name.split(".")[0]
        with open(out / f"{name}.pkl", "wb") as f:
            pickle.dump({"kw": kw, "source": f"python/tests/data/{rel}"}, f)
        m_ub = 0 if kw.get("A_ub") is None else kw["A_ub"].shape[0]
        m_eq = 0 if kw.get("A_eq") is None else kw["A_eq"].shape[0]
        print(
            f"fixture {name}: n={len(kw['c'])} ub={m_ub} eq={m_eq} "
            f"int={int(np.sum(kw['integrality']))}",
            flush=True,
        )
        n += 1
    print(f"fixtures: written={n}", flush=True)
    return 0 if n else 1


# ----------------------------------------------------------------------------- capture
def cmd_capture_decomp(a) -> int:
    out = Path(a.dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MASTER_CAPTURE_DIR=str(out), PYTHONPATH=str(HERE.parent))
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "_master_capture_plugin",
        "-o",
        "addopts=",
        *DECOMP_TESTS,
    ]
    p = subprocess.run(cmd, cwd=REPO, env=env)
    n = len(glob.glob(str(out / "decomp__*.pkl")))
    print(f"capture-decomp: pytest rc={p.returncode} masters={n}", flush=True)
    return 0 if n else 1


def cmd_capture_gdp(a) -> int:
    script = HERE.parent / "gdp_master_backend_entry.py"
    names = subprocess.run(
        [sys.executable, str(script), "names"], capture_output=True, text=True, check=True
    ).stdout.split()
    if not a.all:
        # The GDPlib instances need gdplib installed from source (discopt_benchmarks
        # pyproject, issue #823); the vendored Pyomo GDPopt models need nothing.
        try:
            import gdplib  # noqa: F401
        except ImportError:
            skipped = [n for n in names if not n.startswith("pyomo.")]
            names = [n for n in names if n.startswith("pyomo.")]
            print(
                f"capture-gdp: gdplib not installed; skipping {len(skipped)}: {skipped}", flush=True
            )
    out = Path(a.dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names, 1):
        p = subprocess.run(
            [sys.executable, "-u", str(script), "capture", name, str(out)],
            capture_output=True,
            text=True,
            timeout=a.timeout,
        )
        tail = (p.stdout.strip().splitlines() or [""])[-1]
        print(f"[gdp {i}/{len(names)}] {name}: rc={p.returncode} {tail}", flush=True)
        if p.returncode != 0:
            print(p.stderr[-1500:], flush=True)
    n = len(glob.glob(str(out / "*.pkl")))
    print(f"capture-gdp: instances={len(names)} masters={n}", flush=True)
    return 0 if n else 1


# ----------------------------------------------------------------------------- replay
def check_point(kw: dict, x) -> tuple[bool, float | None, float]:
    """``(verified, objective, max violation)`` of ``x`` for the master ``kw``, densely."""
    if x is None:
        return False, None, math.inf
    x = np.asarray(x, float).ravel()
    c = np.asarray(kw["c"], float)
    if x.shape[0] != c.shape[0] or not np.all(np.isfinite(x)):
        return False, None, math.inf
    viol = 0.0
    for j, (lo, hi) in enumerate(kw.get("bounds") or []):
        if lo is not None:
            viol = max(viol, (lo - x[j]) / (1.0 + abs(lo)))
        if hi is not None:
            viol = max(viol, (x[j] - hi) / (1.0 + abs(hi)))
    for key, rhs, eq in (("A_ub", "b_ub", False), ("A_eq", "b_eq", True)):
        A = kw.get(key)
        if A is None:
            continue
        A = sp.csr_matrix(A)
        b = np.asarray(kw[rhs], float)
        for i in range(A.shape[0]):
            s, e = A.indptr[i], A.indptr[i + 1]
            act = float(np.dot(A.data[s:e], x[A.indices[s:e]]))
            r = act - b[i]
            viol = max(viol, (abs(r) if eq else r) / (1.0 + abs(b[i])))
    integ = np.asarray(kw.get("integrality", np.zeros_like(c)), bool)
    int_viol = float(np.max(np.abs(x[integ] - np.round(x[integ])), initial=0.0))
    ok = viol <= FEAS_TOL and int_viol <= INT_TOL
    return ok, float(c @ x), max(viol, int_viol)


def _engine(name):
    from discopt.solvers.lp_backend import get_milp_solver

    return get_milp_solver(backend=name)


def _kw_for(engine: str, kw: dict, time_limit: float) -> dict:
    out = dict(kw)
    out["time_limit"] = time_limit
    out.setdefault("gap_tolerance", GAP)
    if engine != "highs":
        out.pop("confirm_bound_from", None)  # a milp_highs-only keyword
    return out


def cross_check(rec: dict, gap: float) -> list[str]:
    """Contradictions between the engines on one master (min sense).

    * a verified incumbent strictly below some engine's bound (that bound is false, or
      the incumbent is wrong -- either way the certificates contradict);
    * both engines ``optimal`` with objectives farther apart than the gap allows;
    * an engine claiming ``infeasible`` (or ``unbounded``) while another returned a point
      this script verified feasible (resp. a finite bound).
    """
    why = []
    incs = [(e, rec[e]["obj"]) for e in ENGINES if rec[e].get("verified")]
    for e in ENGINES:
        st = rec[e].get("status", "").lower()
        if "infeasible" in st:
            why += [f"{e} infeasible, {ei} verified a feasible point ({inc})" for ei, inc in incs]
        if "unbounded" in st:
            why += [
                f"{e} unbounded, {f} bound {rec[f]['bound']}"
                for f in ENGINES
                if rec[f].get("bound") is not None
            ]
    bounds = [(e, rec[e]["bound"]) for e in ENGINES if rec[e].get("bound") is not None]
    for ei, inc in incs:
        for eb, b in bounds:
            if b > inc + 1e-6 * max(1.0, abs(inc)):
                why.append(f"{eb} bound {b} above {ei}'s verified incumbent {inc}")
    h, s = rec["highs"], rec["simplex"]
    if "optimal" in h["status"].lower() and "optimal" in s["status"].lower():
        a, b = h["obj"], s["obj"]
        if a is None or b is None or abs(a - b) > gap * max(abs(a), abs(b)) + 1e-6 * (1 + abs(a)):
            why.append(f"both optimal, objectives {a} vs {b}")
    return why


def cmd_replay(a) -> int:
    _assert_discopt_under_test()
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "*.pkl")))
    hdr = _header()
    print("header", json.dumps(hdr), "masters", len(files), flush=True)
    engines = {e: _engine(e) for e in ENGINES}
    rows, compared, contradictions, raised = [], 0, 0, 0
    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            kw = dict(pickle.load(fh)["kw"])
        kw.pop("time_limit", None)
        gap = float(kw.get("gap_tolerance", GAP))
        rec = {
            "gap": gap,
            "master": os.path.basename(f)[:-4],
            "n": int(len(kw["c"])),
            "int": int(np.sum(kw.get("integrality", 0))),
        }
        walls = {e: [] for e in ENGINES}
        for rnd in range(a.rounds):
            order = ENGINES if (i + rnd) % 2 == 0 else ENGINES[::-1]
            for e in order:
                t = time.perf_counter()
                try:
                    r = engines[e](**_kw_for(e, kw, a.time_limit))
                except Exception as ex:  # recorded AND fails the run (CLAUDE.md §7)
                    rec[e] = {"status": f"raised {type(ex).__name__}: {ex}"[:300]}
                    raised += 1
                    continue
                walls[e].append(time.perf_counter() - t)
                ok, obj, viol = check_point(kw, r.x)
                rec[e] = {
                    "status": str(r.status),
                    "obj": obj if ok else (None if r.objective is None else float(r.objective)),
                    "reported_obj": None if r.objective is None else float(r.objective),
                    "bound": None
                    if r.bound is None or not np.isfinite(r.bound)
                    else float(r.bound),
                    "nodes": int(r.node_count),
                    "verified": ok,
                    "max_viol": viol,
                }
        for e in ENGINES:
            if walls[e]:
                rec[e]["wall_med"] = statistics.median(walls[e])
                rec[e]["walls"] = walls[e]
        if all("bound" in rec.get(e, {}) for e in ENGINES):
            why = cross_check(rec, gap)
            compared += len(ENGINES)
            contradictions += bool(why)
            rec["contradictions"] = why
        rows.append(rec)
        brief = {
            e: (
                rec[e]["status"],
                rec[e].get("obj"),
                rec[e].get("bound"),
                rec[e].get("nodes"),
                round(rec[e].get("wall_med", float("nan")), 3),
            )
            for e in ENGINES
        }
        print(f"[replay {i + 1}/{len(files)}] {rec['master']:40s} {brief}", flush=True)
        for w in rec.get("contradictions", []):
            print(f"  !!! CONTRADICTION {w}", flush=True)
        Path(a.out).write_text(
            json.dumps(
                {
                    "header": hdr,
                    "args": vars(a),
                    "rows": rows,
                    "compared": compared,
                    "contradictions": contradictions,
                    "raised": raised,
                },
                indent=1,
                default=str,
            )
            + "\n"
        )
    print(
        f"replay: masters={len(rows)} compared={compared} contradictions={contradictions} "
        f"raised={raised}",
        flush=True,
    )
    return 0 if compared and not contradictions and not raised else 1


def cmd_recheck(a) -> int:
    """Re-apply :func:`cross_check` to a finished ``replay`` JSON (no solve)."""
    run = json.loads(Path(a.json).read_text())
    compared = contradictions = 0
    for rec in run["rows"]:
        if not all("bound" in rec.get(e, {}) for e in ENGINES):
            continue
        gap = float(rec.get("gap", GAP))
        why = cross_check(rec, gap)
        compared += len(ENGINES)
        contradictions += bool(why)
        for w in why:
            print(f"CONTRADICTION {rec['master']}: {w}", flush=True)
    print(
        f"recheck: masters={len(run['rows'])} compared={compared} contradictions={contradictions}",
        flush=True,
    )
    return 0 if compared and not contradictions else 1


# ----------------------------------------------------------------------------- route
SOLU_URL = "https://www.minlplib.org/minlplib.solu"
SOLU = REPO / "ref" / "minlplib" / "minlplib.solu"
NL_DIRS = (DATA / "minlplib_nl", DATA / "minlplib")


def _solu() -> dict[str, float]:
    if not SOLU.exists():
        SOLU.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOLU_URL, timeout=120) as r:  # noqa: S310
            SOLU.write_bytes(r.read())
    out: dict[str, float] = {}
    for line in SOLU.read_text().splitlines():
        p = line.split()
        if p and p[0] == "=opt=":
            out[p[1]] = float(p[2])
    return out


def _route_models(spans):
    """``(label, builder)`` for the §76.4 panel: routed in-repo .nl x spans + portfolios."""
    sys.path.insert(0, str(REPO / "python" / "tests"))
    import discopt.solver as solver
    from _invariance import _rebuild
    from discopt.modeling.core import Constraint, from_nl

    paths = {}
    for d in NL_DIRS:
        for p in sorted(d.glob("*.nl")):
            paths.setdefault(p.stem, p)
    routed = [n for n, p in paths.items() if solver._convex_minlp_auto_route(from_nl(str(p)))[0]]

    def build(path, span):
        base = from_nl(str(path))
        if span == 0:
            return base
        new = _rebuild(base, lambda v: np.zeros(v.lb.shape), 1.0, f"{path.stem}_pr{span:g}")
        rng = np.random.default_rng(zlib.crc32(path.name.encode()))
        new._constraints = [
            Constraint(
                body=float(10.0 ** rng.uniform(-span, span)) * c.body,
                sense=c.sense,
                rhs=0.0,
                name=c.name,
            )
            for c in new._constraints
        ]
        return new

    out = [(f"{n}@{s:g}", n, (lambda p=paths[n], s=s: build(p, s))) for n in routed for s in spans]
    import test_1658_pounce_tol_and_oa_master as t1658

    for n in (12, 16, 20, 24):
        for seed in (21, 22):
            for persp in (False, True):
                lab = f"portfolio_n{n}_s{seed}_{'persp' if persp else 'bigm'}"
                out.append(
                    (
                        lab,
                        None,
                        (
                            lambda n=n, seed=seed, persp=persp: t1658._portfolio(
                                n=n, seed=seed, perspective=persp
                            )
                        ),
                    )
                )
    return routed, out


def cmd_route(a) -> int:
    from discopt.modeling.core import ObjectiveSense
    from discopt.validation.feasibility import verify_point

    _assert_discopt_under_test()
    hdr = _header()
    spans = [float(s) for s in a.spans.split(",")]
    solu = _solu()
    routed, models = _route_models(spans)
    print(
        "header",
        json.dumps(hdr),
        f"routed {len(routed)}: {routed}",
        f"models {len(models)}",
        flush=True,
    )
    rows, checks, viol, errors, fired = [], 0, [], [], 0
    for i, (lab, inst, build) in enumerate(models, 1):
        m = build()
        sense = 1.0 if m._objective.sense == ObjectiveSense.MINIMIZE else -1.0
        t = time.perf_counter()
        try:
            r = m.solve(time_limit=a.time_limit)
        except Exception as e:  # recorded, counted, fails the run
            errors.append(f"{lab}: {type(e).__name__}: {e}"[:300])
            print(f"[route {i}/{len(models)}] {lab:34s} ERROR {errors[-1]}", flush=True)
            continue
        w = time.perf_counter() - t
        ok = None
        if r.x is not None:
            flat = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
            ok = bool(verify_point(m, flat).ok)
        did = "master=highs" in (r.algorithm_route or "")
        fired += did
        refs = [("solu", solu[inst])] if inst in solu else []
        if ok and r.objective is not None:
            refs.append(("incumbent", float(r.objective)))
        if r.bound is not None:
            for lab_ref, ref in refs:
                checks += 1
                if sense * r.bound > sense * ref + 1e-6 * max(1.0, abs(ref)):
                    viol.append(f"{lab} bound {r.bound} past {lab_ref} {ref}")
        if r.gap_certified and ok is False:
            viol.append(f"{lab} certified an unverified point")
        rec = {
            "model": lab,
            "status": str(r.status),
            "cert": bool(r.gap_certified),
            "obj": r.objective,
            "bound": r.bound,
            "wall": w,
            "feas_ok": ok,
            "fired": did,
            "route": r.algorithm_route,
        }
        rows.append(rec)
        print(
            f"[route {i}/{len(models)}] {lab:34s} {rec['status']:12s} cert={rec['cert']!s:5s} "
            f"wall={w:6.2f} feas={ok} fired={did}",
            flush=True,
        )
        Path(a.out).write_text(
            json.dumps(
                {
                    "header": hdr,
                    "args": vars(a),
                    "rows": rows,
                    "checks": checks,
                    "violations": viol,
                    "errors": errors,
                },
                indent=1,
                default=str,
            )
            + "\n"
        )
    cert = sum(r["cert"] for r in rows)
    print(
        f"route: models={len(models)} certified={cert} wall={sum(r['wall'] for r in rows):.1f}s "
        f"fired={fired} checks={checks} violations={len(viol)} errors={len(errors)}",
        flush=True,
    )
    for v in viol + errors:
        print("FLAG", v, flush=True)
    return 0 if checks and fired and not viol and not errors else 1


# ----------------------------------------------------------------------------- decomp e2e
def cmd_decomp_e2e(a) -> int:
    import xml.etree.ElementTree as ET

    results, executed = {}, 0
    for e in ENGINES:
        junit = Path(a.out).with_suffix(f".{e}.xml")
        env = dict(os.environ, DISCOPT_DECOMP_MASTER=e)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            f"--junitxml={junit}",
            *DECOMP_TESTS,
        ]
        p = subprocess.run(cmd, cwd=REPO, env=env)
        tree = ET.parse(junit)
        for tc in tree.iter("testcase"):
            nid = f"{tc.get('classname')}::{tc.get('name')}"
            kids = {k.tag for k in tc}
            outcome = (
                "failed"
                if kids & {"failure", "error"}
                else "skipped"
                if "skipped" in kids
                else "passed"
            )
            results.setdefault(nid, {})[e] = outcome
            executed += outcome != "skipped"
        print(f"decomp-e2e: {e}: pytest rc={p.returncode}", flush=True)
    differ = {k: v for k, v in results.items() if len(set(v.values())) > 1}
    Path(a.out).write_text(json.dumps({"results": results, "differ": differ}, indent=1) + "\n")
    for k, v in differ.items():
        print("DIFFER", k, v, flush=True)
    print(f"decomp-e2e: tests={len(results)} executed={executed} differ={len(differ)}", flush=True)
    return 0 if executed else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("fixtures", "capture-decomp"):
        s = sub.add_parser(name)
        s.add_argument("dir")
    g = sub.add_parser("capture-gdp")
    g.add_argument("dir")
    g.add_argument("--timeout", type=float, default=150.0)
    g.add_argument("--all", action="store_true", help="fail rather than skip without gdplib")
    r = sub.add_parser("replay")
    r.add_argument("dirs", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--time-limit", type=float, default=60.0)
    ro = sub.add_parser("route")
    ro.add_argument("--out", required=True)
    ro.add_argument("--spans", default="0,3,6")
    ro.add_argument("--time-limit", type=float, default=30.0)
    rc = sub.add_parser("recheck")
    rc.add_argument("json")
    d = sub.add_parser("decomp-e2e")
    d.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {
        "fixtures": cmd_fixtures,
        "capture-decomp": cmd_capture_decomp,
        "capture-gdp": cmd_capture_gdp,
        "replay": cmd_replay,
        "recheck": cmd_recheck,
        "route": cmd_route,
        "decomp-e2e": cmd_decomp_e2e,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
