"""Summarize the #727 Track 1.2 DISCOPT_OBBT_ITERATE graduation panel.

Reads panel_cluster.json + panel_easy.json and emits a combined compare_summary.txt
with the two-bar (cert-clean / net-positive) assessment. Distinguishes
FLAG-ATTRIBUTABLE soundness problems (ON differs from OFF) from pre-existing
both-arm artifacts (identical OFF and ON — not caused by the flag).

Usage: python issue727_obbt_iterate_compare.py cluster.json easy.json out.txt
"""

from __future__ import annotations

import json
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def flag_attributable(rec):
    """True iff a soundness problem is present in ON but the corresponding OFF
    measurement is clean for that same check (i.e. the flag introduced it)."""
    off, on = rec["off"], rec["on"]
    opt = rec["oracle"]
    tol = 1e-4 * (1 + abs(opt)) if opt is not None else 1e-4
    reasons = []
    # false bound in ON but not OFF
    if opt is not None and on["bound"] is not None and on["bound"] > opt + tol:
        if not (off["bound"] is not None and off["bound"] > opt + tol):
            reasons.append(f"ON false bound {on['bound']:.6g} > oracle {opt:.6g}")
    # ON obj beats optimum but OFF does not
    if opt is not None and on["obj"] is not None and on["obj"] < opt - tol:
        if not (off["obj"] is not None and off["obj"] < opt - tol):
            reasons.append(f"ON obj {on['obj']:.6g} beats oracle {opt:.6g}")
    # ON incumbent infeasible but OFF feasible
    if not on["incumbent_feasible"] and off["incumbent_feasible"]:
        reasons.append("ON incumbent infeasible (OFF feasible)")
    # cert lost
    if off["gap_certified"] and not on["gap_certified"]:
        reasons.append("ON lost certification")
    return reasons


def main():
    cluster, easy, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    lines = []
    W = lines.append

    def section(title, data, is_cluster):
        rows = data["rows"]
        W(f"\n{'='*100}\n{title}  (n={len(rows)}, TL={data['time_limit_s']}s)\n{'='*100}")
        W(
            f"{'instance':22s} {'oracle':>11s} | {'OFF bnd':>11s} {'ON bnd':>11s} | "
            f"{'OFFn':>6s} {'ONn':>6s} | {'OFFw':>5s} {'ONw':>5s} | {'OFFcert':>7s} {'ONcert':>6s} | delta"
        )
        n_gain = n_loss = n_tight = n_loose = 0
        flag_sound = []
        both_arm_notes = []
        for r in rows:
            off, on = r["off"], r["on"]
            fo = lambda x: f"{x:.4g}" if x is not None else "None"  # noqa: E731
            fr = flag_attributable(r)
            if fr:
                flag_sound.append((r["instance"], fr))
            if r["problems"] and not fr:
                both_arm_notes.append((r["instance"], r["problems"]))
            n_gain += r["cert_delta"] == "ON gained cert"
            n_loss += r["cert_delta"] == "ON LOST cert"
            n_tight += r["bound_delta"].startswith("ON tighter")
            n_loose += r["bound_delta"].startswith("ON LOOSER")
            delta = r["bound_delta"]
            if r["cert_delta"] != "same":
                delta = r["cert_delta"] + "; " + delta
            W(
                f"{r['instance']:22s} {fo(r['oracle']):>11s} | {fo(off['bound']):>11s} "
                f"{fo(on['bound']):>11s} | {off['nodes']:>6d} {on['nodes']:>6d} | "
                f"{off['wall']:>5.1f} {on['wall']:>5.1f} | {str(off['gap_certified']):>7s} "
                f"{str(on['gap_certified']):>6s} | {delta}"
            )
        W(
            f"\n  cert gained ON: {n_gain} | cert lost ON: {n_loss} | "
            f"bound tighter ON: {n_tight} | bound looser ON: {n_loose}"
        )
        if flag_sound:
            W("  !! FLAG-ATTRIBUTABLE soundness problems:")
            for name, rs in flag_sound:
                W(f"     {name}: {'; '.join(rs)}")
        else:
            W("  FLAG-ATTRIBUTABLE soundness problems: NONE")
        if both_arm_notes:
            W("  (pre-existing both-arm notes — identical OFF/ON, NOT flag-caused:)")
            for name, ps in both_arm_notes:
                W(f"     {name}: {'; '.join(ps)}")
        return n_gain, n_loss, n_tight, n_loose, len(flag_sound)

    W("#727 Track 1.2 — DISCOPT_OBBT_ITERATE (#720) §5 graduation panel")
    W("Two bars: (1) cert-clean [flag introduces no false/looser bound, no cert loss]")
    W("          (2) net-positive [ON gains cert or tightens bound broadly, regresses nothing]")

    cg = section("CLUSTER (#727 QCQP wide-box targets)", cluster, True)
    eg = section("EASY / BROAD in-repo certifying corpus (regression guard)", easy, False)

    tot_gain = cg[0] + eg[0]
    tot_loss = cg[1] + eg[1]
    tot_tight = cg[2] + eg[2]
    tot_loose = cg[3] + eg[3]
    tot_flag_sound = cg[4] + eg[4]

    W(f"\n{'='*100}\nVERDICT\n{'='*100}")
    W(f"  cluster cert gained ON: {cg[0]}   easy cert gained ON: {eg[0]}")
    W(f"  cluster cert lost ON:   {cg[1]}   easy cert lost ON:   {eg[1]}")
    W(f"  bound tighter ON (total): {tot_tight}   bound looser ON (total): {tot_loose}")
    W(f"  FLAG-ATTRIBUTABLE soundness problems (total): {tot_flag_sound}")
    cert_clean = tot_flag_sound == 0 and tot_loose == 0
    net_positive = (tot_gain > 0 or tot_tight > 0) and tot_loss == 0 and tot_loose == 0
    # Graduation additionally requires cluster benefit that MATTERS: a certification
    # gain on the target cluster, not merely a tighter-but-still-open bound.
    material_cluster_win = cg[0] > 0
    W(f"\n  BAR 1 cert-clean:   {'PASS' if cert_clean else 'FAIL'}")
    W(f"  BAR 2 net-positive: {'PASS' if net_positive else 'FAIL'}")
    W(f"  (cluster certifications gained by ON: {cg[0]} — material closure of a #727 target?)")
    verdict = "GRADUATE" if (cert_clean and net_positive and material_cluster_win) else "HOLD"
    W(f"\n  VERDICT: {verdict}")
    text = "\n".join(lines)
    print(text)
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
