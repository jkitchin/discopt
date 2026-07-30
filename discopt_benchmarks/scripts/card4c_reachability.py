"""Card 4c entry experiment — can the Regime-N panel corpus reach the three loops?

Hypothesis (implicit in the card): the three stray B&B loops
(`lp_spatial_bb.solve_lp_spatial_bb`, `gp.solve_gp_minlp`,
`signomial_global.solve_signomial_global`) are exercised by the 119-instance
Regime-N panel, so a `panel_baseline.py --check` PASS is meaningful evidence that
a port of them is bound-neutral.

Kill criterion: any loop whose *class* is absent from the corpus (its classifier
declines on all 119 instances) cannot be covered by the panel at all — even with
its flag forced ON. For such a loop a panel PASS is VACUOUS (CLAUDE.md §6) and the
port must be verified by dedicated tests instead, or reported as unverifiable.

No exception is swallowed (CLAUDE.md §7): a classifier that raises is counted and
its traceback recorded, never silently treated as "declined".
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO = Path("/home/user/discopt")
sys.path.insert(0, str(REPO / "python"))

import discopt  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

# --- CLAUDE.md §8: verify WHICH code was loaded, not just that it imported.
assert discopt.__file__ == str(REPO / "python" / "discopt" / "__init__.py"), discopt.__file__
import discopt._rust as _rust  # noqa: E402

assert "_rust" in Path(_rust.__file__).name, _rust.__file__
print(f"[loaded] discopt={discopt.__file__}", flush=True)
print(f"[loaded] _rust={_rust.__file__}", flush=True)

from discopt._jax.convexity.signomial_global import classify_signomial_global  # noqa: E402
from discopt.gp import classify_gp, classify_gp_minlp  # noqa: E402

CORPUS_DIRS = (
    REPO / "python" / "tests" / "data" / "minlplib_nl",
    REPO / "python" / "tests" / "data" / "minlplib",
)

paths: dict[str, Path] = {}
for d in CORPUS_DIRS:
    if d.is_dir():
        for p in sorted(d.glob("*.nl")):
            paths.setdefault(p.stem, p)

print(f"[corpus] {len(paths)} distinct instances from {[str(d.name) for d in CORPUS_DIRS]}")

# Executed-assertion counters (CLAUDE.md §6). A run that classifies nothing must
# exit non-zero rather than print a clean zero.
n_examined = 0
n_load_error = 0
gp_minlp_hits: list[str] = []
sgo_hits: list[str] = []
gp_hits: list[str] = []
classify_errors: dict[str, str] = {}

for stem, path in sorted(paths.items()):
    try:
        model = from_nl(str(path))
    except Exception:
        n_load_error += 1
        classify_errors[f"{stem}:load"] = traceback.format_exc(limit=2)
        print(f"  {stem:24s} LOAD-ERROR", flush=True)
        continue

    n_examined += 1
    marks = []
    for label, fn, bucket in (
        ("gp_minlp", classify_gp_minlp, gp_minlp_hits),
        ("sgo", classify_signomial_global, sgo_hits),
        ("gp", classify_gp, gp_hits),
    ):
        try:
            if fn(model) is not None:
                bucket.append(stem)
                marks.append(label)
        except Exception:
            # NOT swallowed: recorded and surfaced in the summary.
            classify_errors[f"{stem}:{label}"] = traceback.format_exc(limit=3)
            marks.append(f"{label}=ERROR")
    print(f"  {stem:24s} {' '.join(marks) if marks else '-'}", flush=True)

print()
print("=" * 72)
print("CARD 4c ENTRY EXPERIMENT — corpus reachability of the three stray loops")
print("=" * 72)
print(f"instances examined (executed classifications): {n_examined}")
print(f"classification calls executed:                 {n_examined * 3}")
print(f"load errors:                                   {n_load_error}")
print(f"classifier errors:                             {len(classify_errors)}")
print()
print(f"classify_gp_minlp accepts       : {len(gp_minlp_hits):3d}  {gp_minlp_hits}")
print(f"classify_signomial_global accepts: {len(sgo_hits):3d}  {sgo_hits}")
print(f"classify_gp accepts (context)   : {len(gp_hits):3d}  {gp_hits}")
if classify_errors:
    print("\n--- classifier errors (NOT swallowed) ---")
    for k, v in classify_errors.items():
        print(f"[{k}]\n{v}")

out = REPO / "reports" / "card4c_reachability.json"
out.write_text(
    json.dumps(
        {
            "instances_examined": n_examined,
            "classification_calls": n_examined * 3,
            "load_errors": n_load_error,
            "gp_minlp_accepts": gp_minlp_hits,
            "signomial_global_accepts": sgo_hits,
            "gp_accepts": gp_hits,
            "classifier_errors": sorted(classify_errors),
        },
        indent=2,
    )
)
print(f"\nwrote {out}")

if n_examined == 0:
    print("FAIL: zero instances examined — the probe measured nothing.")
    sys.exit(1)
sys.exit(0)
