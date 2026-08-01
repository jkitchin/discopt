"""#917 entry experiment, step 1: which in-repo corpus instances take the #844 reserve?

Prints the scope predicate for every ``.nl`` in ``python/tests/data/minlplib_nl``
under the DEFAULT gate (``mixed=False``, matching the shipped default of
``DISCOPT_LP_SPATIAL_MIXED_FALLBACK``), i.e. exactly the models whose
``Model.solve`` deducts 35% of the caller's ``time_limit``.

Per CLAUDE.md §6 this ends with an executed-comparison count and exits non-zero
if the probe evaluated nothing.
"""

import sys
from pathlib import Path

from discopt._jax.lp_spatial_bb import _is_in_scope
from discopt.modeling.core import from_nl

CORPUS = Path(__file__).resolve().parents[1] / "python/tests/data/minlplib_nl"

evaluated = 0
in_scope = []
failed = []

for nl in sorted(CORPUS.glob("*.nl")):
    try:
        m = from_nl(str(nl))
    except Exception as exc:  # a parse failure is data, not something to swallow
        failed.append((nl.stem, f"{type(exc).__name__}: {exc}"))
        continue
    # Mirror the solve-site gate: scope AND at least one constraint row.
    scoped = bool(m._constraints) and _is_in_scope(m, mixed=False)
    evaluated += 1
    print(f"{nl.stem:24s} in_scope={scoped}", flush=True)
    if scoped:
        in_scope.append(nl.stem)

print()
print(f"in-scope instances ({len(in_scope)}): {' '.join(in_scope)}")
if failed:
    print(f"parse failures ({len(failed)}):")
    for name, err in failed:
        print(f"  {name}: {err}")
print(f"EXECUTED_COMPARISONS={evaluated}")
if evaluated == 0:
    print("PROBE FIRED NOTHING", file=sys.stderr)
    sys.exit(1)
