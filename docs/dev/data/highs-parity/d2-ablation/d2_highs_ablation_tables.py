#!/usr/bin/env python
"""Render the ablation JSONL files as the markdown tables of d2_highs_ablation.md."""
import json, sys
rows = {}
for f in ['d2_highs_ablation_phase1.jsonl', 'd2_highs_ablation_phase2.jsonl', 'd2_highs_ablation_phase3.jsonl']:
    for ln in open(f):
        r = json.loads(ln); rows[(r['name'], r['arm'])] = r
b40 = {}
for ln in open('d2_highs_ablation_base40.jsonl'):
    r = json.loads(ln); b40[(r['name'], r['arm'])] = r
pan = {r['name']: r for r in json.load(open('/home/user/discopt/docs/dev/data/highs-parity/p6.6/miplib.json'))['rows']}
for r in json.load(open('/home/user/discopt/docs/dev/data/highs-parity/p6.6/lmi.json'))['rows']:
    if r['panel'] == 'M' and r['name'] not in pan: pan[r['name']] = r
lpb = json.load(open('d2_lp_bounds.json'))
names = sorted({n for n, _ in rows}, key=str.lower)
out = []
P = out.append
def cell(r, bn):
    if r is None: return '-'
    k = r['nodes'] / max(1, bn)
    if r['certified']:
        s = f"{r['nodes']}"
        if k >= 10: s = f"**{r['nodes']} ({k:.0f}x)**"
        return s
    dg = '-' if r['dual_gap'] is None else f"{100*r['dual_gap']:.1f}"
    pg = '-' if r['primal_gap'] is None else f"{100*r['primal_gap']:.1f}"
    return f"**FAIL** {r['nodes']} d{dg}/p{pg}"
loo = ['base_tdef', 'base', 'presolve_off', 'restart_off', 'symmetry_off', 'heur_all_off', 'node_cuts_off', 'cuts_throttle', 'sb_off', 'probing_off']
P("### Table 1 — leave-one-out (threads=1 except `base_tdef`; 20 s; nodes to certify, or FAIL with dual/primal gap % at the limit; bold = >= 10x the `base` nodes)\n")
P("| instance | " + " | ".join(loo) + " |")
P("|" + "---|" * (len(loo) + 1))
for n in names:
    bn = rows[(n, 'base')]['nodes']
    P(f"| {n} | " + " | ".join(cell(rows.get((n, a)), bn) for a in loo) + " |")
P("")
loi = ['all_off', 'keep_cuts', 'keep_presolve', 'keep_heur', 'keep_sb', 'keep_cuts_heur', 'keep_cuts_presolve', 'keep_cuts_sb']
P("### Table 2 — leave-one-in: plain HiGHS B&B (`all_off` = presolve/restart/symmetry off, heuristics off, cuts throttled, no node cuts, no strong branching) plus ONE component (or two); same cell format, x = vs `base`\n")
P("| instance | Rust p6.6 nodes (pg/dg %) | base | " + " | ".join(loi) + " |")
P("|" + "---|" * (len(loi) + 3))
cert = {a: 0 for a in loi + ['base']}
for n in names:
    R = pan[n]['R']; o = pan[n]['oracle']
    pg = '-' if R['objective'] is None else f"{abs(R['objective']-o)/max(1,abs(o))*100:.1f}"
    dg = f"{abs(R['bound']-o)/max(1,abs(o))*100:.1f}"
    bn = rows[(n, 'base')]['nodes']
    for a in loi + ['base']:
        cert[a] += rows[(n, a)]['certified']
    P(f"| {n} | {R['nodes']} ({pg}/{dg}) | {cell(rows[(n,'base')], bn)} | " + " | ".join(cell(rows.get((n, a)), bn) for a in loi) + " |")
P(f"| **certified / 25** | 0 | {cert['base']} | " + " | ".join(str(cert[a]) for a in loi) + " |")
P("")
ro = ['root_only', 'root_only+cuts_throttle', 'root_only+presolve_off', 'root_only+restart_off', 'root_only+heur_all_off']
P("### Table 3 — root-only decomposition (`mip_max_nodes=1`): dual gap % | primal gap % after the root (restarts in parentheses); first column = pure LP relaxation gap (integrality dropped, `d2_lp_bounds.json`)\n")
P("| instance | LP gap % | " + " | ".join(a.replace('root_only', 'root') for a in ro) + " |")
P("|" + "---|" * (len(ro) + 2))
for n in names:
    cells = []
    for a in ro:
        r = rows.get((n, a))
        if r is None: cells.append('-'); continue
        dg = '-' if r['dual_gap'] is None else f"{100*r['dual_gap']:.2f}"
        pg = '-' if r['primal_gap'] is None else f"{100*r['primal_gap']:.2f}"
        cells.append(f"{dg} \\| {pg} ({r['log']['restarts']})")
    P(f"| {n} | {100*lpb[n]['lp_gap']:.2f} | " + " | ".join(cells) + " |")
P("")
P("### Table 4 — sub-switches (phase 3) on the instances where the family mattered: nodes (x base), FAIL = not certified at 20 s\n")
groups = {'heuristics': ['heur_all_off', 'fj_off', 'rins_off', 'rens_off', 'rootredcost_off', 'heur_effort_0'],
          'presolve rules (`presolve_rule_off` bits)': ['presolve_off', 'probing_off', 'parallel_rc_off', 'aggregator_off', 'doubleton_off', 'sparsify_off', 'dualfix_off', 'presolve_reduce_only'],
          'cuts': ['cuts_throttle', 'node_cuts_off', 'pool_soft_1', 'lp_age_0']}
for g, arms in groups.items():
    insts = sorted({n for n, a in rows if a == arms[-1]}, key=str.lower)
    P(f"**{g}**\n")
    P("| instance | base | " + " | ".join(arms) + " |")
    P("|" + "---|" * (len(arms) + 2))
    for n in insts:
        bn = rows[(n, 'base')]['nodes']
        P(f"| {n} | {bn} | " + " | ".join(cell(rows.get((n, a)), bn) for a in arms) + " |")
    P("")
P("### Table 5 — 40 s re-runs of the slow baselines (threads=1, host load ~2): status, nodes, wall\n")
P("| instance | base @40 s | all_off @40 s |")
P("|---|---|---|")
for n in sorted({n for n, _ in b40}, key=str.lower):
    c = []
    for a in ['base', 'all_off']:
        r = b40[(n, a)]
        c.append(f"{'certified' if r['certified'] else 'FAIL d' + f'{100*r['dual_gap']:.1f}' + '/p' + ('-' if r['primal_gap'] is None else f'{100*r['primal_gap']:.1f}')} {r['nodes']} n {r['wall']:.1f} s")
    P(f"| {n} | {c[0]} | {c[1]} |")
P("")
print("\n".join(out))
print(f"<!-- tables rendered from {len(rows)} + {len(b40)} rows -->")
