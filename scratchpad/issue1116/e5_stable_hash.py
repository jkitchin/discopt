"""#1116 E5: is the run-to-run drift caused by ``Variable.__hash__ = id(self)``?

Usage: python -u e5_stable_hash.py <instance-stem> <max_nodes> <reps>

Hypothesis
----------
``Variable.__hash__`` returns ``id(self)`` (``modeling/core.py:680``). Every
``set``/``dict`` keyed by a ``Variable`` therefore iterates in an order that is a
function of the ALLOCATOR, not of the model: it differs between two
``from_nl`` calls in the same process and between processes. Any float
accumulation or order-sensitive selection downstream of such an iteration then
differs in its last bits, while every *structural* count (rows separated, LP
solves) stays identical -- exactly the signature #1116 reports.

Kill criterion
--------------
Replace the hash with a value that is a pure function of the model
(``self._index``, the variable's position in the flat vector, assigned at
construction) and re-run the 3-rep probe. If the bound is STILL not
bit-reproducible, address-dependent hash order is NOT the cause and this
direction is dead.

Collisions are safe here: ``Constraint.__bool__`` raises, so if two variables
from different models ever collided the probe would crash loudly rather than
report a quiet wrong answer (CLAUDE.md §3/§7).

Prints per-rep progress (§10), asserts the module it loaded (§8), and counts both
patched-hash calls and comparisons, exiting non-zero if either is zero (§6).
"""

import json
import sys

import discopt
from discopt.modeling.core import Variable, from_nl

print(f"discopt.__file__={discopt.__file__}", flush=True)

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem, max_nodes, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

_hash_calls = [0]
_orig_hash = Variable.__hash__


def _stable_hash(self):
    _hash_calls[0] += 1
    return self._index


Variable.__hash__ = _stable_hash
assert Variable.__hash__ is not _orig_hash, "patch did not take"

rows = []
for rep in range(reps):
    model = from_nl(NL.format(stem))
    r = model.solve(max_nodes=max_nodes)
    row = {
        "rep": rep,
        "nodes": int(r.node_count or 0),
        "bound": repr(float(r.bound)) if r.bound is not None else None,
        "objective": repr(float(r.objective)) if r.objective is not None else None,
        "status": r.status,
        "hash_calls": _hash_calls[0],
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

comparisons = 0
for key in ("nodes", "bound", "objective", "status"):
    distinct = sorted({repr(x[key]) for x in rows})
    comparisons += len(rows) - 1
    print(
        f"{key:10s} {'STABLE' if len(distinct) == 1 else 'VARIES'} "
        f"distinct={len(distinct)} {distinct}",
        flush=True,
    )
print(f"comparisons={comparisons} patched_hash_calls={_hash_calls[0]}", flush=True)
if comparisons == 0 or _hash_calls[0] == 0:
    print("PROBE FIRED NOTHING", flush=True)
    sys.exit(2)
