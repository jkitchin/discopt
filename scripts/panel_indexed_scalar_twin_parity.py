"""Twin parity for ``DISCOPT_INTEGER_PRODUCT_INDEXED``: array model vs scalar twin.

The flag's purpose is that the solver should not care whether a model was written
with an array variable or with scalar ones. This checks that end to end under the
default solve (#1940 lift race on): for each pair, the product witness, the lifted
model (row count, flat bounds), certification, the certified objective, the final
route and the race's sequence of arms must match, and so must node counts wherever
the outcome is clock-free (the lift slice certified).

Usage: ``python -u scripts/panel_indexed_scalar_twin_parity.py``. Exits non-zero on
any failure or when nothing was checked (CLAUDE.md §6).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ["DISCOPT_INTEGER_PRODUCT_INDEXED"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402
import panel_native_kernel_vector_vars as V  # noqa: E402
from discopt._relax import integer_product_reform as ipr  # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt.transformations import get as get_transformation  # noqa: E402

assert ipr._indexed_refs_enabled(), "flag not on"

CHECKS = 0
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)
        print("  FAIL", msg, flush=True)


def int_bilinear_rows(n, seed):
    """Integer products in constraint rows: the class where the lift wins."""

    def build(arr):
        rng = np.random.default_rng(seed)
        m = dm.Model(f"ib_{n}")
        x = V._vars(m, arr, "x", n, 0, 6)
        for r in range(2):
            m.subject_to(
                sum(float(rng.uniform(0.5, 2)) * x[i] * x[(i + 1 + r) % n] for i in range(n))
                <= float(4 * n)
            )
        m.minimize(-sum(float(rng.uniform(0.5, 2)) * x[i] for i in range(n)))
        return m

    return build


def mixed_rows(n, seed):
    """Integer x continuous products in rows."""

    def build(arr):
        rng = np.random.default_rng(seed)
        m = dm.Model(f"mx_{n}")
        x = V._vars(m, arr, "x", n, 0, 5)
        y = V._vars(m, arr, "y", n, 0.0, 5.0, integer=False)
        for i in range(n):
            m.subject_to(x[i] * y[i] >= float(rng.uniform(1.0, 4.0)))
        m.subject_to(sum(x[i] * x[(i + 1) % n] for i in range(n)) <= float(n * 5))
        cost = [(float(rng.uniform(0.5, 3)), float(rng.uniform(0.5, 3))) for _ in range(n)]
        m.minimize(sum(cx * x[i] + cy * y[i] for i, (cx, cy) in enumerate(cost)))
        return m

    return build


CASES = dict(V.TWINS)
CASES.update({f"ib_n{n}": int_bilinear_rows(n, 11) for n in (4, 6, 8)})
CASES.update({f"mx_n{n}": mixed_rows(n, 12) for n in (3, 4, 5)})

_real_solve_model = S.solve_model
_arms: list = []


def _spy(*a, **k):
    _arms.append(S._IPX_RACE_ARM.get())
    return _real_solve_model(*a, **k)


S.solve_model = _spy


def run(m):
    _arms.clear()
    t = time.perf_counter()
    r = m.solve(time_limit=30)
    return r, time.perf_counter() - t, list(_arms)


def _route(r):
    return (r.algorithm_route or "").split(":")[0]


def _certified(r):
    return r.status == "optimal" and r.gap_certified


def main():
    lift = get_transformation("integer.bilinear")
    for name, build in CASES.items():
        a, s = build(True), build(False)
        wit_a = ipr.has_nonconvex_integer_bilinear(a)
        wit_s = ipr.has_nonconvex_integer_bilinear(s)
        check(wit_a == wit_s, f"{name}: witness array={wit_a} scalar={wit_s}")
        if wit_a:
            la_m, ls_m = lift.apply(a), lift.apply(s)
            ra_n, rs_n = len(la_m._constraints), len(ls_m._constraints)
            check(ra_n == rs_n, f"{name}: lifted rows {ra_n} vs {rs_n}")
            la, ua = flat_variable_bounds(la_m)
            ls, us = flat_variable_bounds(ls_m)
            check(
                np.array_equal(la, ls) and np.array_equal(ua, us),
                f"{name}: lifted flat bounds differ",
            )
        res = {arr: run(build(arr)) for arr in (True, False)}
        (ra, wa, arms_a), (rs, ws, arms_s) = res[True], res[False]
        ca, cs = _certified(ra), _certified(rs)
        check(ca == cs, f"{name}: certified array={ca} scalar={cs}")
        if ca and cs:
            check(
                abs(ra.objective - rs.objective) <= 1e-6 * max(1, abs(rs.objective)),
                f"{name}: obj {ra.objective} vs {rs.objective}",
            )
        check(_route(ra) == _route(rs), f"{name}: route {_route(ra)} vs {_route(rs)}")
        check(arms_a == arms_s, f"{name}: race arms {arms_a} vs {arms_s}")
        # A lift slice that certified is clock-free: its tree must be the twin's.
        if ca and cs and arms_a == arms_s and arms_a[1:] == ["lift"]:
            check(
                ra.node_count == rs.node_count,
                f"{name}: nodes {ra.node_count} vs {rs.node_count}",
            )
        print(
            f"{name:10s} witness={wit_a!s:5} "
            f"array {wa:6.2f}s {ra.status:9s} n={ra.node_count:6d} [{_route(ra)}] | "
            f"scalar {ws:6.2f}s {rs.status:9s} n={rs.node_count:6d} [{_route(rs)}] "
            f"arms={arms_a[1:]}",
            flush=True,
        )
    print("checks", CHECKS, "failures", len(FAILS))
    sys.exit(1 if FAILS or CHECKS == 0 else 0)


if __name__ == "__main__":
    main()
