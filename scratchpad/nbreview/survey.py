#!/usr/bin/env python3
"""Static survey of notebooks: cell counts, never-executed cells, stale-claim smells."""
import json, re, sys, os

REMAINING = """advanced_features amp_global_minlp benchmark_dashboard benchmarks_by_class bound_tightening
callbacks conflict_analysis convex_fast_path convexity_detection cstr_fit_doe_optimize cutest_interface
decision_focused_learning decomposition_advisor differentiable_milp differentiable_pounce_layer export_formats
implicit_function_node infeasibility_iis interactive_debugger ipm_vs_ipopt llm_integration minlp_examples
model_derivatives model_representations model_serialization modeling_guide neural_dae nlp_bb nn_embedding
presolve primal_heuristics problem_classes pyomo_solver qp_solver quickstart sensitivity_analysis
sets_and_indexing solver_internals suspect_head_to_head symbolic_envelopes tutorial_estimation
tutorial_lagrangian tutorial_pounce_sipopt tutorial_solver_cyipopt tutorial_solver_pounce
tutorial_solver_selection warm_start""".split()

rows = []
for name in REMAINING:
    p = f"docs/notebooks/{name}.ipynb"
    nb = json.load(open(p))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    nonempty = [c for c in code if "".join(c["source"]).strip()]
    never = [i for i, c in enumerate(code)
             if "".join(c["source"]).strip() and c.get("execution_count") is None and not c.get("outputs")]
    errs = [i for i, c in enumerate(code)
            if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    src_all = "\n".join("".join(c["source"]) for c in code)
    out_all = "\n".join(
        "".join("".join(o.get("text", [])) for o in c.get("outputs", []) if o.get("output_type") == "stream")
        for c in code)
    smells = []
    if re.search(r'os\.environ\[\s*["\']JAX_', src_all): smells.append("jaxenv")
    if re.search(r"\bimport jax\b|from jax\b|\bjnp\.", src_all): smells.append("USES_JAX")
    if "ansi" not in "" and "\x1b[" in out_all: smells.append("ansi")
    if re.search(r"timing-bucket-unknown", out_all): smells.append("bucket-unknown")
    if re.search(r"INFO .*pounce|POUNCE INFO", out_all): smells.append("pounce-info")
    if "cyipopt" in src_all or "ipopt" in src_all.lower(): smells.append("ipopt")
    if re.search(r"if .*exists\(|os\.path\.exists|\.is_file\(\)", src_all): smells.append("exists-guard")
    nbytes = os.path.getsize(p)
    rows.append((name, len(nonempty), len(never), len(errs), nbytes // 1024, ",".join(smells)))

print(f"{'notebook':32s} {'cells':>5} {'never':>5} {'err':>4} {'KB':>5}  smells")
for r in sorted(rows, key=lambda r: -r[2]):
    print(f"{r[0]:32s} {r[1]:5d} {r[2]:5d} {r[3]:4d} {r[4]:5d}  {r[5]}")
print(f"\n[surveyed {len(rows)} notebooks]")
assert len(rows) == 47, len(rows)
