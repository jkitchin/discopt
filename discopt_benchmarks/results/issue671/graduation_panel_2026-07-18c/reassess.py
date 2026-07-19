import json, math, sys, importlib.util
# Load the (fixed) harness module to reuse assess()
spec = importlib.util.spec_from_file_location(
    "gp", "discopt_benchmarks/scripts/issue671_rowfilter_graduation_panel.py")
gp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gp)

DATA = "discopt_benchmarks/results/issue671/graduation_panel_2026-07-18c/panel_data.json"
d = json.load(open(DATA))
rows = d["rows"]
n_fail = n_fired = 0
new_rows = []
for rec in rows:
    if "off" not in rec:   # error/missing instance
        new_rows.append(rec); continue
    problems, notes, prim, inert, both = gp.assess(rec)
    rec = dict(rec, problems=problems, notes=notes, signal=prim, inert=inert)
    new_rows.append(rec)
    n_fail += bool(problems)
    n_fired += (not inert)
    if problems:
        print(f"CERT-FAIL {rec['instance']}: {problems}")

# headline hda (mirror main())
hda = next((r for r in new_rows if r.get("instance")=="hda" and "off" in r), None)
onb=hda["on"]["bound"]; offb=hda["off"]["bound"]; opt=hda["oracle"]
hda_ok = onb is not None and math.isfinite(onb) and onb>=-7e4 and (opt is None or onb<=opt+1e-2)
cert_clean = n_fail==0
soft = [r["instance"] for r in new_rows if "on" in r and r.get("signal")=="ON LOOSER bound" and r["instance"]!="hda"]
net_positive = hda_ok and cert_clean
verdict = "GRADUATE" if cert_clean and net_positive else "HOLD"
summary = {"n_instances": len([r for r in new_rows if "off" in r]),
           "n_cert_fail": n_fail, "n_filter_fired": n_fired,
           "cert_clean": cert_clean, "hda_tight_and_sound": hda_ok,
           "hda_detail": f"hda ON bound {onb} (OFF {offb}, opt {opt}) tight&sound={hda_ok}",
           "soft_looser_partial_bounds": soft, "net_positive": net_positive, "verdict": verdict}
print("\nRE-ASSESSED SUMMARY:", json.dumps(summary, indent=2))
print("\nVERDICT:", verdict)
out = "discopt_benchmarks/results/issue671/graduation_panel_2026-07-18c/panel_data_reassessed.json"
json.dump({"hda_tl": d["hda_tl"], "corpus_tl": d["corpus_tl"], "summary": summary,
           "reassessment_note": "assess() corrected: drop=0 node_count/obj differences are non-determinism NOTES, not HARD bound-neutral violations (tls2 flaps 353<->421 under constant flag OFF). No re-solve; same measured 18c arm data.",
           "rows": new_rows}, open(out,"w"), indent=2)
print("wrote", out)
