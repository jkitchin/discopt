#!/usr/bin/env python
"""PYPROF-1: Python-side residual-overhead profiler for a single .nl instance.

Complements ``profile_instance.py`` (which buckets wall into solver *phases*).
This script instead buckets the profiled wall into two top-level categories and
runs a **call-count census** to find the single biggest *removable Python
overhead* sink on the small "correct-but-slower-than-BARON-by-seconds" panel
(m3, nvs13, nvs08, ex1224, fac2, nvs06, plus a trivial fixed-tax probe).

Two top-level buckets
---------------------
GENUINE COMPUTE (not removable in Python):
  - POUNCE ``solve_nlp`` / IPM (``pounce`` C ext, incl. JAX callbacks)
  - the Rust LP/simplex (``discopt._rust.solve_lp*``)
  - Rust FBBT / presolve / other ``discopt._rust`` builtins
  - JAX-compiled evaluator kernels (pjit execute) and XLA compile

REMOVABLE PYTHON OVERHEAD (orchestration you could cache/marshal away):
  - term/structure (re)classification (``term_classifier``, ``classify_*``,
    ``convexity``, ``monotonicity``, ``problem_classifier``)
  - McCormick / relaxation DAG (re)build & walk (``mccormick``,
    ``mccormick_lp``, ``milp_relaxation`` build paths)
  - certificate re-derivation around LP calls (Neumaier-Shcherbina / Farkas)
  - marshaling to/from Rust (``np.asarray`` churn, tuple pack/unpack at PyO3)
  - core.py DAG-node hot props (``size``/``__hash__``/``__eq__``) and ``abs``

Everything else profiled is left "unclassified" (reported, not forced into a
bucket) so the removable fraction is a *lower-bound-honest* number.

The script never modifies solver behaviour. It attaches no node_callback by
default (``--trace`` to enable a bound trace). Run each instance one at a time
on a quiet machine.

Usage
-----
  python python_residual_profile.py INST.nl --mode clean   --json out.json
  python python_residual_profile.py INST.nl --mode cprofile --json out.json \
      --pstats out.pstats
  python python_residual_profile.py INST.nl --mode sampled --sample-hz 100 \
      --sample-out out.folded --json out.json
  python python_residual_profile.py INST.nl --mode clean --second-solve  # tax
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import threading
import time


# ---------------------------------------------------------------------------
# In-process sampler (identical to profile_instance.py's; py-spy needs root on
# macOS). Native Rust/C/JAX time is attributed to the calling Python frame.
# ---------------------------------------------------------------------------
class StackSampler:
    def __init__(self, hz: float = 100.0):
        self.interval = 1.0 / hz
        self.counts: dict[str, int] = {}
        self.n_samples = 0
        self._stop = threading.Event()
        self._target = threading.current_thread().ident
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            f = sys._current_frames().get(self._target)
            stack = []
            while f is not None:
                code = f.f_code
                parts = code.co_filename.replace("\\", "/").split("/")
                stack.append(f"{'/'.join(parts[-2:])}:{code.co_name}")
                f = f.f_back
            if stack:
                key = ";".join(reversed(stack))
                self.counts[key] = self.counts.get(key, 0) + 1
                self.n_samples += 1
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def dump(self, path: str):
        with open(path, "w") as fh:
            for k, v in sorted(self.counts.items(), key=lambda kv: -kv[1]):
                fh.write(f"{k} {v}\n")

    def leaf_shares(self) -> dict[str, int]:
        """Aggregate by innermost (leaf) frame — 'who is actually on-CPU'."""
        leaf: dict[str, int] = {}
        for k, v in self.counts.items():
            frame = k.rsplit(";", 1)[-1]
            leaf[frame] = leaf.get(frame, 0) + v
        return leaf


# ---------------------------------------------------------------------------
# Top-level attribution. Each predicate takes the pstats key parts
# (filename, funcname). "~" filename means a C/builtin function; pstats encodes
# builtins as ('~', 0, '<qualname>').
# ---------------------------------------------------------------------------
def _is_pounce(f, n):
    return f == "~" and "pounce" in n


def _is_rust_lp(f, n):
    return f == "~" and "discopt._rust" in n and "solve_lp" in n


def _is_rust_other(f, n):
    return f == "~" and "discopt._rust" in n and "solve_lp" not in n


def _is_jax_execute(f, n):
    # compiled-kernel dispatch/execute + XLA compile (genuine device compute)
    return ("xla" in f or "pxla" in f or "dispatch" in f) and (
        "execute" in n or "compile" in n or "backend_compile" in n
    )


GENUINE = [
    ("pounce_nlp", _is_pounce),
    ("rust_lp", _is_rust_lp),
    ("rust_other", _is_rust_other),
    ("jax_execute_compile", _is_jax_execute),
]

# Removable-Python buckets. Matched on substrings of the *filename* (path tail)
# and/or the function name. Order matters only for reporting.
REMOVABLE = [
    (
        "term_structure_classify",
        lambda f, n: any(
            s in f
            for s in (
                "term_classifier",
                "problem_classifier",
                "monotonicity",
                "convexity/",
            )
        ),
    ),
    (
        "mccormick_relax_build_walk",
        lambda f, n: any(s in f for s in ("mccormick.py", "mccormick_lp.py", "milp_relaxation.py")),
    ),
    (
        "certificate_rederivation",
        lambda f, n: ("neumaier" in n.lower())
        or ("shcherbina" in n.lower())
        or ("farkas" in n.lower())
        or ("safe_bound" in n.lower())
        or ("_certificate" in n.lower()),
    ),
    (
        "marshaling_np_asarray",
        lambda f, n: f == "~" and n in ("numpy.asarray", "numpy.array", "numpy.ascontiguousarray"),
    ),
    (
        "core_dag_hotprops",
        lambda f, n: "modeling/core.py" in f
        and n in ("size", "__hash__", "__eq__", "shape", "__init__"),
    ),
    (
        "builtin_abs",
        lambda f, n: f == "~" and n in ("<built-in function abs>", "abs"),
    ),
]

# Individual functions to census (ncalls) regardless of bucket — the 07-05
# report flagged 32.2M size / 21M abs on hda; check for the analog here.
CENSUS = [
    ("core.size", lambda f, n: "modeling/core.py" in f and n == "size"),
    ("core.__hash__", lambda f, n: "modeling/core.py" in f and n == "__hash__"),
    ("core.__eq__", lambda f, n: "modeling/core.py" in f and n == "__eq__"),
    ("np.asarray", lambda f, n: f == "~" and n == "numpy.asarray"),
    ("np.array", lambda f, n: f == "~" and n == "numpy.array"),
    ("builtin.abs", lambda f, n: f == "~" and "abs" in n and f == "~"),
    ("builtin.len", lambda f, n: f == "~" and n == "<built-in method builtins.len>"),
    ("builtin.isinstance", lambda f, n: f == "~" and "isinstance" in n),
    ("rust.solve_lp*", _is_rust_lp),
    ("rust.other", _is_rust_other),
    ("pounce.*", _is_pounce),
    (
        "build_milp_relaxation",
        lambda f, n: n == "build_milp_relaxation",
    ),
    ("solve_at_node", lambda f, n: n == "solve_at_node"),
    (
        "classify_nonlinear_terms",
        lambda f, n: n == "classify_nonlinear_terms",
    ),
    (
        "distribute_products",
        lambda f, n: n == "distribute_products",
    ),
    (
        "term_visit_walks",
        lambda f, n: "term_classifier" in f and n in ("visit", "_visit"),
    ),
]


def _extract(prof: cProfile.Profile) -> dict:
    st = pstats.Stats(prof)
    total = st.total_tt
    genuine: dict[str, dict] = {}
    removable: dict[str, dict] = {}
    census: dict[str, dict] = {}
    genuine_tot = 0.0
    removable_tot = 0.0

    for (fname, _lineno, funcname), (_cc, nc, tt, ct, _callers) in st.stats.items():
        # tottime is additive & non-double-counting across functions → use it
        # for the removable/genuine fractions (cumtime nests and would double).
        for label, pred in GENUINE:
            try:
                hit = pred(fname, funcname)
            except Exception:
                hit = False
            if hit:
                b = genuine.setdefault(label, {"ncalls": 0, "tottime": 0.0, "cumtime": 0.0})
                b["ncalls"] += nc
                b["tottime"] += tt
                b["cumtime"] = max(b["cumtime"], ct)
                genuine_tot += tt
        for label, pred in REMOVABLE:
            try:
                hit = pred(fname, funcname)
            except Exception:
                hit = False
            if hit:
                b = removable.setdefault(label, {"ncalls": 0, "tottime": 0.0, "cumtime": 0.0})
                b["ncalls"] += nc
                b["tottime"] += tt
                b["cumtime"] = max(b["cumtime"], ct)
                removable_tot += tt
        for label, pred in CENSUS:
            try:
                hit = pred(fname, funcname)
            except Exception:
                hit = False
            if hit:
                c = census.setdefault(label, {"ncalls": 0, "tottime": 0.0, "cumtime": 0.0})
                c["ncalls"] += nc
                c["tottime"] += tt
                c["cumtime"] = max(c["cumtime"], ct)

    return {
        "_total_profiled_s": total,
        "genuine_compute": genuine,
        "genuine_tottime_s": genuine_tot,
        "genuine_frac": (genuine_tot / total) if total else None,
        "removable_python": removable,
        "removable_tottime_s": removable_tot,
        "removable_frac": (removable_tot / total) if total else None,
        "unclassified_tottime_s": total - genuine_tot - removable_tot,
        "unclassified_frac": ((total - genuine_tot - removable_tot) / total) if total else None,
        "census_ncalls": census,
    }


def _f(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return v if v == v and abs(v) != float("inf") else str(v)


def run(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    from discopt.modeling.core import from_nl

    t_import = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = from_nl(args.instance)
    t_parse = time.perf_counter() - t0

    kwargs: dict = {"time_limit": args.time_limit, "gap_tolerance": 1e-4}

    prof = None
    sampler = None
    if args.mode == "cprofile":
        prof = cProfile.Profile()
        prof.enable()
    if args.mode == "sampled" or args.sample_hz:
        sampler = StackSampler(hz=args.sample_hz or 100.0)
        sampler.start()

    t0 = time.perf_counter()
    result = model.solve(**kwargs)
    t_solve = time.perf_counter() - t0

    if sampler is not None:
        sampler.stop()
        if args.sample_out:
            sampler.dump(args.sample_out)
    if prof is not None:
        prof.disable()

    rec: dict = {
        "instance": args.instance,
        "mode": args.mode,
        "time_limit": args.time_limit,
        "import_s": t_import,
        "parse_s": t_parse,
        "solve_s": t_solve,
        "status": str(result.status),
        "objective": _f(result.objective),
        "bound": _f(result.bound),
        "gap": _f(result.gap),
        "node_count": result.node_count,
        "root_time_s": _f(result.root_time),
        "root_bound": _f(result.root_bound),
        "solver_stats": result.solver_stats,
    }

    if args.second_solve:
        t0 = time.perf_counter()
        r2 = model.solve(time_limit=args.time_limit, gap_tolerance=1e-4)
        rec["second_solve_s"] = time.perf_counter() - t0
        rec["second_solve_nodes"] = r2.node_count

    if prof is not None:
        if args.pstats:
            prof.dump_stats(args.pstats)
        rec["attribution"] = _extract(prof)

    if sampler is not None:
        leaf = sampler.leaf_shares()
        top = sorted(leaf.items(), key=lambda kv: -kv[1])[:30]
        rec["sampler_n"] = sampler.n_samples
        rec["sampler_leaf_top"] = [{"frame": k, "count": v} for k, v in top]

    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instance")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--mode", choices=["clean", "cprofile", "sampled"], default="clean")
    ap.add_argument("--pstats", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--second-solve", action="store_true")
    ap.add_argument("--sample-hz", type=float, default=None)
    ap.add_argument("--sample-out", default=None)
    args = ap.parse_args()

    rec = run(args)
    text = json.dumps(rec, indent=1, default=str)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text)
    summary = {k: v for k, v in rec.items() if k not in ("sampler_leaf_top",)}
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
