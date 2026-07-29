"""Card 2b **entry experiment** — does Rust ``in_tree_presolve`` subsume the
per-node Python Jacobian FBBT?

Consolidation plan Card 2b. ``solver._tighten_node_bounds_with_status`` (an
O(m·n²) pure-Python Jacobian-sampled linear-row FBBT + the structural nonlinear
tightening) runs at **every node** on both Python B&B paths immediately before the
Rust ``in_tree_presolve`` kernel, which computes its inferences from the exact DAG.
The hypothesis is that the Rust pass subsumes it. Nothing is deleted until this
experiment rules.

What is measured, exactly
-------------------------

Today each node's box ``B0`` becomes ``Rust(Python(B0))``. After the proposed
deletion it becomes ``Rust(B0)``. So the question that decides the card is not
"does Python tighten anything" (it does) but:

    is ``Rust(Python(B0))`` ever **strictly tighter** than ``Rust(B0)``?

That is the counterfactual this probe computes, per node, on the real node streams
of real corpus instances. ``in_tree_presolve`` takes ``&self`` on the Rust side —
it is a pure function of (repr, box, depth, incumbent) with no interior mutation —
so evaluating the counterfactual arm on the same repr is side-effect free and
cannot perturb the search it is observing.

Two ways a Python-only inference can show up, both counted:

* **bound**: some coordinate of ``Rust(Python(B0))`` is strictly inside
  ``Rust(B0)`` beyond a 1e-12 relative tolerance;
* **fathom**: Python proved the node infeasible (the node loop then ``continue``s
  and Rust never sees it) where ``Rust(B0)`` does not. This arm is evaluated
  without an incumbent cutoff — the wrapper has no access to the loop's cutoff at
  that point — which makes it *conservative*: the real Rust pass gets a cutoff and
  can only fathom more.

Kill criterion (from the card): **> 0.5 % of compared nodes showing a Python-only
inference ⇒ do NOT delete.** Below that, the deletion proceeds to its Regime-C
panel.

Instrumentation discipline
--------------------------

Per CLAUDE.md §6 the probe prints the number of comparisons it actually executed —
nodes compared and individual bound comparisons — and **exits non-zero when zero
comparisons ran**. There are no bare ``except`` blocks around the comparison
(§7): a failure inside the probe crashes the child rather than degrading it into a
silent no-op. The loaded ``discopt`` and ``_rust`` paths are printed per child (§8),
and per-instance progress is flushed (§10).

Usage::

    python -u discopt_benchmarks/scripts/card2b_fbbt_subsumption_entry.py
    python -u discopt_benchmarks/scripts/card2b_fbbt_subsumption_entry.py --budget 20
    python -u discopt_benchmarks/scripts/card2b_fbbt_subsumption_entry.py --subset ex1264,tls2

Internal child mode: ``--solve <instance> <budget>``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

#: Relative tolerance for "strictly tighter" (the card's 1e-12).
_REL = 1e-12

#: Instances drawn from ``reports/panel_baseline_f154dcff.json`` by the rule
#: "node_count > 0 and not convex_fast_path", covering BOTH Python B&B loops —
#: the spatial loop (``nlp_bb=False``) and the NLP-B&B loop (``nlp_bb=True``) —
#: ranked by node count so the node streams are long enough to be informative.
#: Chosen from the baseline, not by hand, so the slice is reproducible.
DEFAULT_INSTANCES = [
    # spatial loop (nlp_bb=False)
    "ex1264",
    "ex1263",
    "ex1266",
    "ex1265",
    "st_testgr3",
    "nvs02",
    "nvs12",
    "util",
    "syn05hfsg",
    "prob02",
    "nvs14",
    "st_e04",
    "st_e36",
    "nvs05",
    "tanksize",
    "st_e05",
    "trig",
    "nvs22",
    # NLP-B&B loop (nlp_bb=True)
    "tls2",
    "clay0303hfsg",
    "cvxnonsep_nsig30",
    "flay03m",
    "cvxnonsep_psig30",
    "m3",
    "fac2",
]

_CHILD_SLACK = 180.0


# --------------------------------------------------------------------------- #
# Child                                                                        #
# --------------------------------------------------------------------------- #
def _run_child(instance: str, budget: float) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import discopt  # noqa: PLC0415
    import discopt.solver as _solver  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from discopt._rust import PyModelRepr  # noqa: PLC0415
    from discopt.modeling.core import from_nl  # noqa: PLC0415

    from scripts.panel_baseline import instance_path  # noqa: PLC0415

    S: dict = {  # noqa: N806 — counter bag, deliberately shouty
        "py_calls": 0,
        "py_infeasible": 0,
        "itp_calls": 0,
        "itp_ran": 0,
        "compared_nodes": 0,
        "bounds_compared": 0,
        "unmatched_boxes": 0,
        "py_only_nodes": 0,
        "py_only_bounds": 0,
        "py_only_fathoms": 0,
        "py_fathom_checked": 0,
        "rust_only_nodes": 0,
        "rust_only_bounds": 0,
        "py_tightened_pre_rust_nodes": 0,
        # Attribution (which half of the Python pass carries the inference).
        "attributed_nodes": 0,
        "nl_only_nodes": 0,
        "jac_only_nodes": 0,
        "both_nodes": 0,
    }
    box_of: dict[bytes, tuple] = {}
    last_repr: list = []
    examples: list[str] = []

    def _key(lb, ub) -> bytes:
        a = np.asarray(lb, dtype=np.float64).tobytes()
        b = np.asarray(ub, dtype=np.float64).tobytes()
        return a + b

    _orig_py = _solver._tighten_node_bounds_with_status
    _orig_itp = PyModelRepr.in_tree_presolve

    def _py_wrap(evaluator, node_lb, node_ub, cl_list, cu_list, max_rounds=3):
        b0_lb = np.asarray(node_lb, dtype=np.float64).copy()
        b0_ub = np.asarray(node_ub, dtype=np.float64).copy()
        t_lb, t_ub, inf = _orig_py(evaluator, node_lb, node_ub, cl_list, cu_list, max_rounds)
        S["py_calls"] += 1
        if inf:
            S["py_infeasible"] += 1
            # Python fathomed: the node loop ``continue``s, so Rust never runs on
            # this node at all. Evaluate the counterfactual here (no cutoff — the
            # wrapper cannot see the loop's incumbent, which makes this arm
            # conservative in Rust's favour).
            if last_repr:
                rp = last_repr[-1]
                if len(b0_lb) == rp.n_var_blocks:
                    S["py_fathom_checked"] += 1
                    d0 = _orig_itp(rp, b0_lb, b0_ub, node_depth=0, depth_stride=1)
                    if d0["ran"] and not d0["infeasible"]:
                        S["py_only_fathoms"] += 1
                        S["py_only_nodes"] += 1
                        if len(examples) < 12:
                            examples.append("fathom: Python proved node empty, Rust(B0) did not")
        else:
            if np.any(t_lb > b0_lb + 1e-12) or np.any(t_ub < b0_ub - 1e-12):
                S["py_tightened_pre_rust_nodes"] += 1
            # ATTRIBUTION ARM. ``_tighten_node_bounds_with_status`` is two
            # mechanisms in one: the 17-rule structural/interval nonlinear pass
            # (``_apply_nonlinear_tightening_with_status``) and the Jacobian
            # linear-row FBBT loop wrapped around it. Recording the nonlinear-only
            # box lets the counterfactual below say WHICH half the Rust kernel is
            # missing, which is what the card asks the kill branch to file.
            nl_lb, nl_ub, _nl_inf = _solver._apply_nonlinear_tightening_with_status(
                evaluator._model, b0_lb.copy(), b0_ub.copy()
            )
            box_of[_key(t_lb, t_ub)] = (b0_lb, b0_ub, nl_lb, nl_ub)
        return t_lb, t_ub, inf

    def _itp_wrap(self, node_lb, node_ub, **kw):
        last_repr.append(self)
        del last_repr[:-1]
        d_p = _orig_itp(self, node_lb, node_ub, **kw)
        S["itp_calls"] += 1
        if not d_p["ran"]:
            return d_p
        S["itp_ran"] += 1
        b0 = box_of.pop(_key(node_lb, node_ub), None)
        if b0 is None:
            # No Python pass preceded this call (e.g. cl_list empty, or a path
            # that does not run the Python FBBT). Counted, never silently dropped.
            S["unmatched_boxes"] += 1
            return d_p
        d_0 = _orig_itp(self, b0[0], b0[1], **kw)
        d_nl = _orig_itp(self, b0[2], b0[3], **kw)
        S["compared_nodes"] += 1

        if d_p["infeasible"] != d_0["infeasible"]:
            if d_p["infeasible"]:
                S["py_only_fathoms"] += 1
                S["py_only_nodes"] += 1
                if len(examples) < 12:
                    examples.append("fathom: Rust(Python(B0)) empty, Rust(B0) not")
            else:
                S["rust_only_nodes"] += 1
            return d_p
        if d_p["infeasible"]:
            return d_p

        lb_p = np.asarray(d_p["lb"], dtype=np.float64)
        ub_p = np.asarray(d_p["ub"], dtype=np.float64)
        lb_0 = np.asarray(d_0["lb"], dtype=np.float64)
        ub_0 = np.asarray(d_0["ub"], dtype=np.float64)
        S["bounds_compared"] += 2 * lb_p.size

        tol_lb = _REL * np.maximum(1.0, np.abs(lb_0))
        tol_ub = _REL * np.maximum(1.0, np.abs(ub_0))
        strict_lb = lb_p > lb_0 + tol_lb
        strict_ub = ub_p < ub_0 - tol_ub
        k = int(strict_lb.sum() + strict_ub.sum())
        if k:
            S["py_only_nodes"] += 1
            S["py_only_bounds"] += k
            # Attribute: is the nonlinear-only arm already as tight as the full
            # Python pass, only tighter than Rust(B0), or neither?
            lb_n = np.asarray(d_nl["lb"], dtype=np.float64)
            ub_n = np.asarray(d_nl["ub"], dtype=np.float64)
            nl_beats_rust = bool((lb_n > lb_0 + tol_lb).any() or (ub_n < ub_0 - tol_ub).any())
            jac_beats_nl = bool(
                (lb_p > lb_n + _REL * np.maximum(1.0, np.abs(lb_n))).any()
                or (ub_p < ub_n - _REL * np.maximum(1.0, np.abs(ub_n))).any()
            )
            S["attributed_nodes"] += 1
            if nl_beats_rust and jac_beats_nl:
                S["both_nodes"] += 1
            elif nl_beats_rust:
                S["nl_only_nodes"] += 1
            elif jac_beats_nl:
                S["jac_only_nodes"] += 1
            if len(examples) < 12:
                j = int(np.flatnonzero(strict_lb | strict_ub)[0])
                examples.append(
                    f"bound: var[{j}] Rust(Py(B0))=[{lb_p[j]:.12g},{ub_p[j]:.12g}] "
                    f"vs Rust(B0)=[{lb_0[j]:.12g},{ub_0[j]:.12g}]"
                )
        # The reverse direction is informational: Rust from the *raw* box can be
        # tighter than from the Python-tightened one only through path-dependence,
        # which is worth knowing about.
        rk = int((lb_0 > lb_p + tol_lb).sum() + (ub_0 < ub_p - tol_ub).sum())
        if rk:
            S["rust_only_nodes"] += 1
            S["rust_only_bounds"] += rk
        return d_p

    _solver._tighten_node_bounds_with_status = _py_wrap
    PyModelRepr.in_tree_presolve = _itp_wrap

    out: dict = {
        "instance": instance,
        "discopt_file": discopt.__file__,
        "rust_file": sys.modules["discopt._rust"].__file__,
        "budget": float(budget),
    }
    try:
        model = from_nl(str(instance_path(instance)))
        t0 = time.perf_counter()
        r = model.solve(time_limit=budget)
        out["wall"] = time.perf_counter() - t0
        out["status"] = str(r.status)
        out["node_count"] = int(r.node_count)
        out["objective"] = None if r.objective is None else float(r.objective)
    except Exception as exc:
        out["status"] = "errored"
        out["error"] = repr(exc)
    out["stats"] = S
    out["examples"] = examples
    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Parent                                                                       #
# --------------------------------------------------------------------------- #
def _solve_one(instance: str, budget: float) -> dict:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--solve", instance, str(budget)]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=budget + _CHILD_SLACK, env=env
        )
    except subprocess.TimeoutExpired:
        return {"instance": instance, "status": "child_timeout", "stats": {}}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "status": "child_crashed",
        "stats": {},
        "stderr_tail": proc.stderr[-1500:],
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 3 and argv[0] == "--solve":
        return _run_child(argv[1], float(argv[2]))

    p = argparse.ArgumentParser(description="Card 2b entry experiment.")
    p.add_argument("--budget", type=float, default=20.0)
    p.add_argument("--subset", help="comma-separated instance names")
    p.add_argument("--out", default=str(_REPO_ROOT / "reports" / "card2b_fbbt_subsumption.json"))
    args = p.parse_args(argv)

    instances = (
        [s.strip() for s in args.subset.split(",") if s.strip()]
        if args.subset
        else list(DEFAULT_INSTANCES)
    )
    print(
        f"Card 2b entry experiment: {len(instances)} instance(s), {args.budget:.0f}s budget, "
        f"defaults + probe. load={os.getloadavg()[0]:.2f}",
        flush=True,
    )
    tot: dict = {}
    rows = []
    for i, inst in enumerate(instances, 1):
        row = _solve_one(inst, args.budget)
        rows.append(row)
        st = row.get("stats") or {}
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + int(v)
        print(
            f"  [{i:2d}/{len(instances)}] {inst:20s} {str(row.get('status')):12s} "
            f"nodes={str(row.get('node_count', '-')):>7s} "
            f"py_calls={st.get('py_calls', 0):6d} compared={st.get('compared_nodes', 0):6d} "
            f"py_only_nodes={st.get('py_only_nodes', 0):5d} "
            f"py_only_bounds={st.get('py_only_bounds', 0):6d} "
            f"py_only_fathoms={st.get('py_only_fathoms', 0):4d}",
            flush=True,
        )
        if row.get("status") in ("child_crashed",):
            print(f"      stderr: {row.get('stderr_tail', '')[-400:]}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"totals": tot, "rows": rows}, indent=1) + "\n")

    compared = tot.get("compared_nodes", 0)
    fathom_checked = tot.get("py_fathom_checked", 0)
    denom = compared + fathom_checked
    py_only = tot.get("py_only_nodes", 0)
    print("\n" + "=" * 78, flush=True)
    print(f"totals: {json.dumps(tot, sort_keys=True)}", flush=True)
    print(
        f"executed comparisons: {compared} node counterfactuals "
        f"+ {fathom_checked} fathom counterfactuals "
        f"= {denom} decided nodes; {tot.get('bounds_compared', 0)} individual bound comparisons",
        flush=True,
    )
    print(
        f"attribution over {tot.get('attributed_nodes', 0)} Python-only bound nodes: "
        f"nonlinear-rules-only={tot.get('nl_only_nodes', 0)}  "
        f"jacobian-FBBT-only={tot.get('jac_only_nodes', 0)}  "
        f"both={tot.get('both_nodes', 0)}",
        flush=True,
    )
    print(
        f"nodes where the Python pass tightened SOMETHING before Rust ran: "
        f"{tot.get('py_tightened_pre_rust_nodes', 0)} "
        f"(this is NOT the criterion — Rust may re-derive all of it)",
        flush=True,
    )
    ex = [e for r in rows for e in (r.get("examples") or [])][:12]
    if ex:
        print("examples of Python-only inferences:", flush=True)
        for e in ex:
            print(f"  - {e}", flush=True)
    if denom == 0:
        print(
            "\nFAIL: ZERO node counterfactuals executed. The probe measured nothing "
            "and a 'no Python-only inference' reading here would be a no-op reported "
            "as a pass (CLAUDE.md §6).",
            flush=True,
        )
        return 3
    frac = py_only / denom
    print(
        f"\nPython-only inference nodes: {py_only}/{denom} = {100 * frac:.4f}% "
        f"(kill criterion: > 0.5%)",
        flush=True,
    )
    print(
        f"VERDICT: {'DO NOT DELETE' if frac > 0.005 else 'DELETION IS SAFE TO PROCEED'}", flush=True
    )
    print("=" * 78, flush=True)
    return 1 if frac > 0.005 else 0


if __name__ == "__main__":
    raise SystemExit(main())
