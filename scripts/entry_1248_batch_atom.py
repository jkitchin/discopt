"""Does a registered atom keep its envelope inside a ``solve_batch`` WORKER?

The end-to-end arm for #1248 A × #1246. ``dm.solve_batch(workers > 1)`` ships
every model to its worker as serialized text and the worker rebuilds it with
``dm.loads``; before the atom tag was written into the document, the worker
relaxed the registered composite term by term and nothing said so.

This is a **script**, not a pytest test, and deliberately. Under ``spawn`` each
worker re-imports the parent's ``__main__``; run under pytest that is pytest, not
the test module, so a module-level ``register_function`` in a test file is *not*
present in the worker and the arm would measure the unregistered path either way.
Only a standalone ``__main__`` puts the registration where the worker will find
it. ``test_1248_atom_survives_serialization.py`` runs this file in a subprocess.

Why ``node_count`` and not ``use_count``: the counter lives in the worker process
and cannot be read from the parent. These models terminate on WORK (the gap), not
on wall clock, so the node count is deterministic and comparable across worker
counts -- ``batch.py``: "Models that terminate on work -- the gap, ``max_nodes``
-- return identical results either way." A worker that lost the atom needs the
term-by-term node count instead.

Measured on this file's family (L0=3.0, L1=1.5, RT=1.0), 4 models:

    without the document tag:   workers=1 -> 15 nodes,  workers=2 -> 23 nodes
    with it:                    workers=1 -> 15 nodes,  workers=2 -> 15 nodes

15 and 23 are the registered and primitive counts this parameterisation is
recorded with in ``test_1248_register_function.py``, so the two arms land exactly
on the known pair rather than merely differing.

Prints an executed-comparison count and exits non-zero if it is zero, or if any
pair disagrees (CLAUDE.md §6).
"""

from __future__ import annotations

import sys

import discopt.modeling as dm
from discopt.operators import get_registered, register_function

ATOM = "entry_1248_batch_rk"


def _rk(x):
    """The Redlich-Kister binary the CALPHAD plugin (#1249) prices phases with."""
    return x * (1 - x) * (3.0 + 1.5 * (2 * x - 1)) + 1.0 * (dm.xlogx(x) + dm.xlogx(1 - x))


# Module level on purpose: this is the registration the spawned worker inherits
# by re-importing __main__, and the placement is itself part of what is tested.
register_function(ATOM, _rk, replace=True)


def _build(i: int) -> dm.Model:
    m = dm.Model(f"batch_atom{i}")
    x = m.continuous("x", lb=1e-9, ub=1 - 1e-9)
    m.minimize(get_registered(ATOM)(x))
    return m


def main() -> int:
    n = 4
    seq = dm.solve_batch([_build(i) for i in range(n)], workers=1)
    par = dm.solve_batch([_build(i) for i in range(n)], workers=2)

    compared = 0
    mismatched: list[tuple[int, int, int]] = []
    for i, (a, b) in enumerate(zip(seq, par)):
        compared += 1
        print(
            f"model {i}: workers=1 nodes={a.node_count} obj={a.objective!r} "
            f"| workers=2 nodes={b.node_count} obj={b.objective!r}"
        )
        if a.node_count != b.node_count:
            mismatched.append((i, a.node_count, b.node_count))

    print(f"\nEXECUTED COMPARISONS: {compared}")
    if compared == 0:
        print("PROBE COMPARED NOTHING", file=sys.stderr)
        return 1
    if mismatched:
        print(
            f"NODE COUNTS DIFFER -- the worker did not restore the registered atom: {mismatched}",
            file=sys.stderr,
        )
        return 1
    print("OK: identical node counts -- the worker restored the registered atom.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
