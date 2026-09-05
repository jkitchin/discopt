"""#1182 entry experiment, part 3: is there a CLASS big-M/hull cannot lower at all?

E1/E2 measured speed and certification and found the Theorem-1 lowering behind on
both. That only kills the *performance* motive. A separate, non-performance motive
would justify the lowering anyway: big-M lowering needs a finite M, and discopt
refuses (``_unbounded_big_m_error``) when a disjunct row has no finite bound in
the relevant direction; hull needs finite disjunct bounds for its perspective
variables. Theorem 1 needs neither -- its lifted weights are bounded in [0, 1] by
construction.

So: on a GDP whose disjunct rows are unbounded, does big-M refuse where the
simplex lowering solves? That would be a capability gap, not a speed claim, and
it is the strongest remaining case for building this.

Prints the executed-assertion count and exits non-zero if nothing was measured.
"""

from __future__ import annotations

import sys

from discopt import Model
from simplex_proto import reformulate_simplex
from source_check import predicate_report


def build_unbounded_gdp() -> Model:
    """``x`` has no finite bounds, so ``x <= 1`` has no finite big-M."""
    m = Model("unbounded_disjunct")
    x = m.continuous("x")  # default +/-9.999e19 sentinel == unbounded
    z = m.continuous("z", lb=0.0, ub=10.0)
    m.minimize((x - 5.0) * (x - 5.0) + z)
    m.subject_to(z >= x - 100.0)
    m.either_or([[x <= 1.0], [x >= 3.0]], name="gap")
    return m


def build_bounded_control() -> Model:
    """Same model with a finite box, as the control: big-M must succeed here."""
    m = Model("bounded_disjunct")
    x = m.continuous("x", lb=-50.0, ub=50.0)
    z = m.continuous("z", lb=0.0, ub=10.0)
    m.minimize((x - 5.0) * (x - 5.0) + z)
    m.subject_to(z >= x - 100.0)
    m.either_or([[x <= 1.0], [x >= 3.0]], name="gap")
    return m


def build_log_unbounded_gdp() -> Model:
    """Both classical lowerings refuse: no finite big-M *and* ``g(0) = log(0)``.

    ``x`` is bounded below but not above, so ``log(x) <= 0`` has no finite big-M;
    and the Furman-Sawaya-Grossmann perspective needs ``g(0)``, which is ``-inf``
    here, so hull refuses too (:class:`HullPerspectiveOriginError`). If the
    Theorem-1 lowering solves this it covers a class neither classical lowering
    reaches -- a capability argument that survives E1/E2's speed falsification.
    """
    from discopt.modeling import log

    m = Model("log_unbounded")
    x = m.continuous("x", lb=0.1)  # unbounded above
    m.minimize((x - 5.0) * (x - 5.0))
    m.either_or([[log(x) <= 0.0], [x >= 3.0]], name="loggap")
    return m


def build_both_refuse_gdp() -> Model:
    """The decisive case: BOTH classical lowerings refuse.

    ``1.0 / x`` on a box straddling 0 has an unbounded interval enclosure (so
    big-M cannot bound the row) and is undefined at the origin (so the
    Furman-Sawaya-Grossmann perspective cannot be formed). This is the reduced
    form of the 18 ``stranded_gas`` disjunct rows E5 finds in GDPlib, where the
    row is ``log`` of a capacity sum whose box includes 0.

    True optimum 0 at ``x = 5``, which satisfies BOTH disjuncts (1/5 <= 1 and
    5 >= 3) -- so it is also an overlap fixture.
    """
    m = Model("both_refuse")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize((x - 5.0) * (x - 5.0))
    m.either_or([[1.0 / x <= 1.0], [x >= 3.0]], name="recip")
    return m


def attempt(builder, arm, time_limit=30.0):
    from discopt._relax.gdp_reformulate import reformulate_gdp

    model = builder()
    try:
        if arm == "simplex":
            target, _rec, _counts = reformulate_simplex(model)
            res = target.solve(time_limit=time_limit)
        else:
            reformulate_gdp(builder(), method=arm)  # the lowering itself may refuse
            res = builder().solve(time_limit=time_limit, gdp_method=arm)
    except Exception as exc:
        return ("refused", f"{type(exc).__name__}: {str(exc)[:140]}", None, None, None)
    src = predicate_report(builder(), res.x) if res.x else None
    return (
        str(res.status),
        "",
        res.objective,
        res.bound,
        None if src is None else src.max_disjunction_violation,
    )


def main() -> int:
    assertions = 0
    for label, builder in (
        ("unbounded", build_unbounded_gdp),
        ("bounded-control", build_bounded_control),
        ("log-unbounded", build_log_unbounded_gdp),
        ("both-refuse", build_both_refuse_gdp),
    ):
        for arm in ("big-m", "hull", "simplex"):
            status, note, obj, bound, viol = attempt(builder, arm)
            assertions += 1
            print(
                f"[{label}/{arm}] status={status} obj={obj} bound={bound} "
                f"src_viol={viol} {note}",
                flush=True,
            )
    print(f"\nexecuted assertions: {assertions}")
    if assertions == 0:
        print("FAIL: the probe measured nothing (CLAUDE.md section 6)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
