#!/usr/bin/env python
"""P0 REPRODUCTION: a stale builder variable box certifies a FALSE `infeasible`.

Found while auditing the modeling layer as a foundation for a flowsheet /
process-modeling layer (issue #1215 discussion). This is a **correctness** defect,
not a performance one: the solver returns ``status="infeasible"`` with
``gap_certified=True`` on a model whose true answer is ``optimal``.

## The defect

Variable bounds are pushed into the Rust builder **once, at declaration**
(``core.py:_register_variable``, and again for every existing variable when the
builder is lazily created in ``core.py:_get_builder``). In builder mode
``model_to_repr`` clones ``b.inner.variables`` and never re-reads ``lb``/``ub``
from the Python ``Variable`` objects — the pure-expression arm *does* re-read
them. Root presolve then runs FBBT on that baked box and can declare the root
infeasible.

Mutating ``var.lb`` / ``var.ub`` after construction is **not** misuse: it is the
only fixing route and it is what in-tree code already does --
``estimate.py:317-319`` (fixing design variables), ``_relax/presolve_pipeline.py``
``:408-409`` and ``:503-504``, ``_relax/node_reduce.py:165-166``,
``_relax/root_reduce.py:125-126`` -- plus discopt-doe's ``compute_fim`` general
path.

## Preconditions (established by bisection, one variable at a time)

1. The Rust builder is **active**, and
2. it carries **zero linear blocks**, and
3. a variable's bounds were mutated after the builder registered them.

Condition 2 is why this hides. A builder *with* linear blocks is safe by
accident: the solve path calls ``_materialize_builder_linear_rows``
(``core.py:5702``), which clears the blocks and rebuilds the builder — thereby
re-reading the *current* bounds. That method early-returns when there are no
blocks (``core.py:5731-5732``), so a blockless builder keeps the stale box.

A non-scalar variable also masks it, via the C-40 ``_aligned`` guard at
``solver.py:15211``.

**Public route to a blockless builder:** ``Model.add_linear_objective`` (and
``add_quadratic_objective``) call ``_get_builder()`` and record
``_builder_linear_objective`` — never ``_builder_linear_blocks``. No private API
is needed to reach the bug.

## What this script does

Runs the arms below; each is compared against a reference model built directly at
the final bounds. Exits non-zero when any arm disagrees with its reference, so
this doubles as a regression check once the defect is fixed.

Measured on `main` @ c052e85:

    route=none                  builder=False blocks=0 -> optimal    (ref optimal)
    route=add_linear_objective  builder=True  blocks=0 -> INFEASIBLE (ref optimal)
                                                          gap_certified=True

§6: prints an executed-assertion count and exits non-zero if it is zero.
§8: prints the module file it loaded.
"""

from __future__ import annotations

import sys

import discopt.modeling as dm
import discopt.modeling.core as core
import numpy as np


def build(lo: float, hi: float, route: str):
    """The smallest model that exhibits it: three scalars, one nonlinear body."""
    m = dm.Model("stale_box")
    u = m.continuous("u", lb=lo, ub=hi)
    z = m.continuous("z", lb=0.0, ub=10.0)
    v = m.continuous("v", lb=0.0, ub=10.0)

    if route == "add_linear_objective":
        # PUBLIC route to a blockless builder: registers every variable's bounds
        # into the Rust builder and records no linear block.
        w = m.continuous("w", shape=(2,), lb=0.0, ub=1.0)
        m.add_linear_objective(np.array([1.0, 1.0]), w, constant=0.0, sense="minimize")
    elif route == "fast_family":
        # Contrast: a fast linear family DOES record a block, and the solve path's
        # _materialize_builder_linear_rows then re-reads bounds -- masking the bug.
        s = m.set("S", list(range(4)))
        m.constraint(s, lambda i: z <= 9.0 + i, name="zcap")

    m.subject_to(z == u * u, name="sq")
    m.subject_to(z <= 1.0, name="cap_z")
    m.subject_to(v == dm.sin(u) + z, name="aux")

    if route == "add_linear_objective":
        m.subject_to(v >= 0.0, name="pad")  # objective already set on the builder
    else:
        m.minimize(z - u + 0.1 * v * v)
    return m, u


def arm(route: str, *, declared=(3.0, 3.0), final=(0.0, 10.0)) -> bool:
    """Return True when this arm agrees with its reference."""
    m, u = build(declared[0], declared[1], route)
    active = getattr(m, "_builder", None) is not None
    blocks = len(getattr(m, "_builder_linear_blocks", []) or [])

    # The estimate.py:317-319 idiom: mutate bounds after construction.
    u.lb = np.broadcast_to(np.asarray(float(final[0])), u.shape)
    u.ub = np.broadcast_to(np.asarray(float(final[1])), u.shape)
    got = m.solve(time_limit=60)

    ref, _ = build(final[0], final[1], route)
    want = ref.solve(time_limit=60)

    same = got.status == want.status and (
        (got.objective is None) == (want.objective is None)
        and (got.objective is None or abs(got.objective - want.objective) <= 1e-4)
    )
    tag = ""
    if not same:
        kind = (
            "FALSE INFEASIBLE"
            if got.status == "infeasible" and want.status != "infeasible"
            else "WRONG OBJECTIVE"
        )
        tag = f"   <<< {kind}" + (" (CERTIFIED)" if got.gap_certified else "")
    print(
        f"route={route:22s} builder={active!s:5s} blocks={blocks} -> "
        f"got={got.status:11s} want={want.status:11s}{tag}"
    )
    print(f"{'':30s} got obj={got.objective!r}  want obj={want.objective!r}")
    return same


def main() -> int:
    print(f"# discopt loaded from: {core.__file__}")
    executed = 0
    bad = []
    for route in ("none", "fast_family", "add_linear_objective"):
        executed += 1
        if not arm(route):
            bad.append(route)
    print(f"\n# executed assertions: {executed}")
    if executed == 0:
        print("FAIL: verified nothing")
        return 1
    if bad:
        print(f"DEFECT PRESENT on route(s): {bad}")
        return 1
    print("No disagreement: the stale-box defect appears fixed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
