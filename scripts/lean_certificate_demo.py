#!/usr/bin/env python
"""End-to-end demo of the Lean-checkable feasibility certificate (Phase 0).

Solves a small NLP and a small MILP, emits a Tier-1 feasibility certificate for
each, and runs the Python reference checker (the executable twin of the Lean
``checkFeasible``) to (a) accept the genuine certificate and (b) reject three
tamper classes. Writes the NLP certificate to ``lean/examples/`` so it can also be
checked with the Lean binary::

    cd lean && lake exe check examples/qp_feasibility.json   # -> FEASIBLE

Run: ``python scripts/lean_certificate_demo.py``
"""

from __future__ import annotations

import copy
from pathlib import Path

import discopt.modeling as dm
from discopt.certificate import (
    build_feasibility_certificate,
    check_certificate,
    write_certificate,
)

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "lean" / "examples"


def _nlp() -> tuple[dm.Model, object]:
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(x + y <= 5, name="c1")
    m.subject_to(x * y >= 3, name="c2")
    m.minimize((x - 2) ** 2 + (y - 1) ** 2)
    return m, m.solve()


def _milp() -> tuple[dm.Model, object]:
    m = dm.Model()
    a = m.integer("a", lb=0, ub=10)
    b = m.integer("b", lb=0, ub=10)
    m.subject_to(a + b <= 7, name="cap")
    m.subject_to(2 * a + b <= 10, name="res")
    m.maximize(3 * a + 2 * b)
    return m, m.solve()


def _report(label: str, model: dm.Model, result: object) -> dict:
    print(f"\n=== {label} ===")
    print(
        f"  solve: status={result.status}  objective={result.objective:.6g}  "
        f"gap_certified={result.gap_certified}"
    )
    cert = build_feasibility_certificate(model, result)
    ok, reason = check_certificate(cert)
    print(f"  check (genuine)      : {'ACCEPT' if ok else 'REJECT'}  -- {reason}")
    assert ok, "genuine certificate must be accepted"

    # Tamper 1: inflate the claimed objective value.
    t = copy.deepcopy(cert)
    t["certificate"]["incumbent"]["objectiveValue"] = [10**6, 1]
    ok1, r1 = check_certificate(t)
    print(f"  check (bad objective): {'ACCEPT' if ok1 else 'REJECT'}  -- {r1}")
    assert not ok1

    # Tamper 2: move the incumbent's first two columns to 0 (breaks a constraint).
    t = copy.deepcopy(cert)
    t["certificate"]["incumbent"]["x"][0] = [0, 1]
    t["certificate"]["incumbent"]["x"][1] = [0, 1]
    ok2, r2 = check_certificate(t)
    print(f"  check (moved point)  : {'ACCEPT' if ok2 else 'REJECT'}  -- {r2}")
    assert not ok2

    # Tamper 3: nudge the first column off its bound / integrality by +5.
    t = copy.deepcopy(cert)
    num, den = t["certificate"]["incumbent"]["x"][0]
    t["certificate"]["incumbent"]["x"][0] = [num + 5 * den, den]
    ok3, r3 = check_certificate(t)
    print(f"  check (out of box)   : {'ACCEPT' if ok3 else 'REJECT'}  -- {r3}")
    assert not ok3
    return cert


def main() -> None:
    nlp_model, nlp_result = _nlp()
    nlp_cert = _report("NLP  (x,y) continuous, bilinear constraint", nlp_model, nlp_result)

    milp_model, milp_result = _milp()
    _report("MILP (a,b) integer, linear", milp_model, milp_result)

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    out = EXAMPLES / "qp_feasibility.json"
    write_certificate(nlp_cert, out)
    print(f"\nWrote example certificate -> {out.relative_to(REPO)}")
    print("Check it with Lean:  cd lean && lake exe check examples/qp_feasibility.json")


if __name__ == "__main__":
    main()
