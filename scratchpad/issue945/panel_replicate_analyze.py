"""Re-analysis of panel_replicate.json with a criterion that actually works.

RETRACTION (CLAUDE.md §11). ``panel_replicate.py``'s own verdict column reported
"arm effect" for all six instances and is WRONG. Its test was "do the arms'
[min,max] intervals overlap", with no tolerance — so two arms whose objective
ranges are the single points 262.6473953 and 262.6473958 (a 2e-9 RELATIVE
difference, and exactly the last-digit drift this change is expected to produce)
came out as an arm effect. A criterion that fires on every instance distinguishes
nothing. Do not read that column; read this file.

The replacement asks two separate questions, which the broken version conflated:

  1. Did a STATUS difference reproduce? That is the categorical question, and it
     is the one the corpus panel actually flagged for tspn12.
  2. Is the between-arm shift larger than BOTH the within-arm spread and a 1e-6
     relative floor? Below that floor a difference is real but not a finding —
     the change moves the NLP point, so the last digits are expected to move.

Bounds are compared in the instance's own sense: syn05hfsg MAXIMIZES, so a LOWER
upper bound is tighter there.

Reads the saved JSON; no re-solving. Prints an executed-comparison count.
"""

from __future__ import annotations

import json
import sys

REL_FLOOR = 1e-6
# Fallback only. The sense decides which direction "tighter" is, so it is read from
# the run itself when recorded; a hardcoded set is the same hole that made the
# corpus panel misread syn05hfsg, and it silently rots as TARGETS changes.
_FALLBACK_MAXIMIZE = {"syn05hfsg"}


def _is_max(name, runs):
    for r in runs:
        if "maximize" in r:
            return bool(r["maximize"]), "recorded"
    return name in _FALLBACK_MAXIMIZE, "FALLBACK (sense not recorded in this run)"


def _vals(runs, key):
    return [r[key] for r in runs if r.get(key) is not None]


def _span(v):
    return (min(v), max(v)) if v else None


def main(path: str) -> int:
    d = json.load(open(path))
    res = d["results"]
    comparisons = 0
    findings = []

    print(f"reps={d['reps']}  time_limit={d['time_limit']}s\n")
    for name, arms in res.items():
        pre, post = arms["pre"], arms["post"]
        mx, src = _is_max(name, pre + post)
        st_pre = sorted({str(r.get("status")) for r in pre})
        st_post = sorted({str(r.get("status")) for r in post})
        print(f"{name}  ({'max' if mx else 'min'}, sense {src})")
        print(f"   status   pre={','.join(st_pre):22s} post={','.join(st_post)}")

        for key in ("objective", "bound"):
            a, b = _vals(pre, key), _vals(post, key)
            comparisons += 1
            sa, sb = _span(a), _span(b)
            if not a or not b:
                print(f"   {key:9s} pre={sa} post={sb}   (absent in one arm)")
                continue
            # Within-arm spread vs between-arm shift.
            spread = max(sa[1] - sa[0], sb[1] - sb[0])
            mid_a, mid_b = sorted(a)[len(a) // 2], sorted(b)[len(b) // 2]
            shift = mid_b - mid_a
            scale = max(1.0, abs(mid_a), abs(mid_b))
            if abs(shift) <= max(spread, REL_FLOOR * scale):
                verdict = "within noise"
            else:
                if key == "bound":
                    tighter = (shift < 0) if mx else (shift > 0)
                    verdict = f"REPRODUCIBLE: post bound {'TIGHTER' if tighter else 'LOOSER'}"
                else:
                    verdict = "REPRODUCIBLE shift"
                findings.append((name, key, mid_a, mid_b, shift / scale))
            print(
                f"   {key:9s} pre={sa[0]:.10g}..{sa[1]:.10g}  post={sb[0]:.10g}..{sb[1]:.10g}  "
                f"shift={shift:+.4g} ({shift / scale:+.2e} rel)  spread={spread:.4g}  -> {verdict}"
            )

        comparisons += 1
        if st_pre != st_post:
            findings.append((name, "status", ",".join(st_pre), ",".join(st_post), None))
            print("   -> STATUS DIFFERS reproducibly")
        print()

    print(f"EXECUTED_COMPARISONS={comparisons}  reproducible_findings={len(findings)}")
    for f in findings:
        print(f"   {f[0]:11s} {f[1]:9s} {f[2]} -> {f[3]}" + (f"  ({f[4]:+.2e} rel)" if f[4] else ""))
    if comparisons == 0:
        print("ANALYSIS COMPARED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/issue945/panel_replicate.json"))
