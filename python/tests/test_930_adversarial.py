"""Adversarial suite for #930: can the re-admitted root-LP-probe bound be WRONG?

``test_930_root_probe_bound.py`` proves the fix does what it was built to do —
the bound the probe proved is reported instead of discarded. This file attacks
it. #930 re-admits a value into the *dual bound*, which is the solver's
certificate, so the failure that matters is not "unhelpful" but "invalid": a
bound past the true optimum is a false certificate (CLAUDE.md §1).

Four attack surfaces, chosen because each is a hole the existing #930 tests do
not cover:

1. **The effective-infinity sentinel.** ``np.isfinite`` is not a finiteness test
   in this codebase: discopt's infinity is the sentinel ``1e20``, and
   ``np.isfinite(1e20)`` is ``True``. Admitting a sentinel-magnitude value as a
   lower bound asserts "the optimum is at least 1e20". This is the same
   confusion as #15, where the simplex's mishandling of 1e20-magnitude bounds
   produced a bogus "optimal" *above* the true optimum and certified a
   suboptimal incumbent (see ``_relax/mccormick_lp.py`` §"issue #15"). Defense in
   depth: no end-to-end path is known that hands the gate a sentinel today, but
   the gate is the last checkpoint before a number becomes a certificate.

2. **MAXIMIZE.** The #930 panel was 17/17 minimize, so the sense conversion had
   *zero* coverage. ``_admissible_probe_bound`` returns a **min-space** value (a
   lower bound on ``-objective``); ``solve_model`` negates it at the consumption
   site (``_rr_signed = -_rr if _is_maximize``). Drop that negation and the
   solver publishes ``-1.0`` as an upper bound for a problem whose optimum is
   ``+1.0`` — a false certificate. #301 was exactly a maximize false-optimal.

3. **Box aliasing.** The gate's whole soundness argument is that the probe box
   *equals* the root box. If the two arrays were ever the same object — the
   probe box captured by reference and then tightened in place, or the root
   snapshot aliasing the live box — ``np.array_equal`` would compare an array
   against itself and admit a bound proved over a strict subset of the root box.
   The gate cannot detect this; only the capture site's ``.copy()`` prevents it.

4. **Oracle validity on instances that actually admit.** The gate declining
   everywhere would make every other assertion vacuous, so the corpus test
   counts admissions and fails when there are none (§6).

Surfaces 2–4 are asserted on real instances rather than mocks, because a mocked
probe tuple cannot show that the *capture site* copies or that the *consumption
site* negates. The instances were chosen by sweeping the 66-instance vendored
corpus at ``time_limit=3`` and recording where the gate fires; see ``_PANEL``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.solver as solver_mod  # noqa: E402
from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: E402
from discopt.solver import _admissible_probe_bound  # noqa: E402

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


# ── surface 1: the effective-infinity sentinel ──────────────────────────────


def test_sentinel_magnitude_bound_is_declined():
    """A 1e20-magnitude "bound" is a sentinel, not a bound, and must be refused.

    ``float('inf')`` was already declined by the ``np.isfinite`` check. That
    check is *not sufficient* here: in the Rust LP layer and in ``solve_model``
    itself (``_INF = 1e19  # bound sentinel: ±1e20 ...``) infinity is
    represented by the finite float ``1e20``, so a sentinel arrives past
    ``np.isfinite`` untouched. Before the guard this admitted ``+1e20``,
    ``-1e20`` and ``1e21`` unchanged.

    Refusing ``-1e20`` matters as much as refusing ``+1e20``: ``-1e20`` is
    vacuous as a lower bound but would still be *merged* by ``max`` and, on a
    MAXIMIZE problem, negated into ``+1e20`` — a garbage upper bound reported as
    a certificate.
    """
    lb = np.array([0.0, -1.0, 2.0])
    ub = np.array([1.0, 3.0, 5.0])
    checks = 0

    def gate(value):
        return _admissible_probe_bound((value, lb.copy(), ub.copy()), lb, ub)

    for sentinel in (1e20, -1e20, 1e21, -1e21, 1e19, -1e19, 5e20):
        assert gate(sentinel) is None, (
            f"{sentinel!r} was admitted as a proved bound; it is the "
            "effective-infinity sentinel (or beyond it), not a bound, and "
            "publishing it is a false certificate"
        )
        checks += 1

    # The guard must be a magnitude cut-off, not a blanket rejection of large
    # numbers: real objectives do reach 1e5-1e6 in this corpus (heatexch_gen2's
    # probe bound is 5.6e5), so a guard set too low would silently discard the
    # very bounds #930 exists to keep.
    for ordinary in (0.0, -7.5, 1e6, -5.6e5, 1e18, -9.9e18):
        assert gate(ordinary) == pytest.approx(ordinary), (
            f"{ordinary!r} is an ordinary objective value and was declined; the "
            "sentinel guard must cut off at the sentinel, not at large-but-real "
            "bounds"
        )
        checks += 1

    assert checks == 13, f"PROBE NEVER FIRED: only {checks} sentinel checks executed"


# ── surfaces 2-4: real instances, with the gate under a recording spy ────────


class _Spy:
    """Records every value the gate admits, plus whether the boxes aliased.

    Wraps rather than replaces the real gate, so the decisions under test are
    the production ones. No ``try``/``except`` anywhere (§7): if the gate raises,
    the test must see the traceback, not a silently empty record.
    """

    def __init__(self):
        self.admitted: list[float] = []
        self.aliased: list[bool] = []

    def install(self, monkeypatch):
        real = solver_mod._admissible_probe_bound

        def spy(probe, root_lb, root_ub):
            value = real(probe, root_lb, root_ub)
            if value is not None:
                self.admitted.append(float(value))
                self.aliased.append(
                    bool(np.shares_memory(probe[1], root_lb) or np.shares_memory(probe[2], root_ub))
                )
            return value

        monkeypatch.setattr(solver_mod, "_admissible_probe_bound", spy)
        return self


# (instance, time_limit, sense, oracle, oracle_kind)
#
# Chosen by sweeping all 66 vendored ``minlplib_nl`` instances at
# ``time_limit=3`` under the spy above and keeping the ones where the gate
# actually admits — 3/66 (``heatexch_gen1``, ``heatexch_gen2``, ``nvs05``), plus
# the two MAXIMIZE instances in the corpus that admit at ``time_limit=5``
# (``bchoco06``, ``bchoco07``). The gate is rare by design: it only has anything
# to say when the search ends with a bound-less tree.
#
# Oracles are from ``minlplib.solu``. ``=opt=`` is the proved global optimum, so
# ``bound <= optimum`` is the literal soundness invariant. ``=best=`` is the best
# known *feasible point*: a feasible point's objective is an upper bound on the
# minimum, so ``bound <= best`` is still a rigorous soundness test, just a weaker
# one. ``bchoco06``/``bchoco07`` have neither — MINLPLib records no solution for
# them — so they carry the sign contract instead (see the test).
_PANEL = [
    ("nvs05", 3.0, "min", 5.4709341080, "=opt="),
    ("heatexch_gen1", 3.0, "min", 154895.9330000000, "=best="),
    ("heatexch_gen2", 3.0, "min", 635838.8464000000, "=best="),
    ("bchoco06", 5.0, "max", None, None),
    ("bchoco07", 5.0, "max", None, None),
]


@pytest.mark.slow
def test_admitted_bounds_never_cross_the_oracle(monkeypatch):
    """Every value the gate admits must be a valid bound on the true optimum.

    This is the kill criterion for #930: one admitted value past the oracle
    sinks the change, because the box-equality gate would then be admitting
    bounds it has no right to. Two independent assertions per instance — the
    value at the moment of admission (tests the gate directly, independent of
    what the ``max`` merge later did with it) and the finally reported bound
    (catches anything the merge broke downstream).

    Both are checked in the instance's own sense. ``_admissible_probe_bound``
    returns min-space, so a MAXIMIZE instance's admitted value is negated first
    — the same conversion ``solve_model`` applies.
    """
    spy = _Spy().install(monkeypatch)
    checks = 0
    fired = 0
    violations: list[str] = []

    for name, time_limit, sense, oracle, kind in _PANEL:
        start = len(spy.admitted)
        model = from_nl(str(_NL_DIR / f"{name}.nl"))
        declared = "max" if model._objective.sense == ObjectiveSense.MAXIMIZE else "min"
        assert declared == sense, (
            f"{name}: panel says {sense} but the model declares {declared}; the "
            "oracle comparison below would be inverted"
        )
        checks += 1

        result = model.solve(time_limit=time_limit)
        admitted = spy.admitted[start:]
        if admitted:
            fired += 1

        if oracle is None:
            continue

        tol = 1e-4 * max(1.0, abs(oracle))
        for value in admitted:
            # min-space -> the instance's own sense, exactly as solve_model does
            signed = -value if sense == "max" else value
            crossed = signed < oracle - tol if sense == "max" else signed > oracle + tol
            if crossed:
                violations.append(
                    f"{name}: ADMITTED {signed!r} crosses the {kind} oracle {oracle!r} ({sense})"
                )
            checks += 1
        if result.bound is not None:
            crossed = result.bound < oracle - tol if sense == "max" else result.bound > oracle + tol
            if crossed:
                violations.append(
                    f"{name}: REPORTED {result.bound!r} crosses the {kind} oracle "
                    f"{oracle!r} ({sense})"
                )
            checks += 1

    assert not violations, "invalid dual bound(s):\n  " + "\n  ".join(violations)
    assert fired > 0, (
        "PROBE NEVER FIRED: the #930 gate admitted nothing on any panel instance, "
        "so every soundness assertion above was vacuous and this run proves nothing"
    )
    assert checks >= len(_PANEL), f"PROBE NEVER FIRED: only {checks} oracle checks executed"


@pytest.mark.slow
def test_maximize_bound_is_the_negated_min_space_value(monkeypatch):
    """The MAXIMIZE sense conversion, which the #930 panel never exercised.

    ``_admissible_probe_bound`` is sense-agnostic and returns a lower bound on
    ``-objective``; ``solve_model`` turns it into an upper bound on ``objective``
    with ``_rr_signed = -_rr if _is_maximize``. If that negation is ever dropped
    or double-applied, the number that reaches the certificate has the wrong
    sign. On these two instances the probe is the binding bound source, so the
    reported bound is exactly the negation and the contract is pinned to the
    digit: a dropped negation reports ``-1.0000000000001252`` as the upper bound
    for a problem whose bound is ``+1.0000000000001252``.

    ``bchoco06``/``bchoco07`` have no MINLPLib solution entry, so no oracle
    check is possible here — this is the sign contract, and the oracle-side
    soundness for maximize would need an instance MINLPLib has solved. The
    certificate invariant (``bound >= incumbent`` for a maximize) is asserted
    whenever an incumbent exists.
    """
    spy = _Spy().install(monkeypatch)
    checks = 0

    for name, time_limit, sense, _oracle, _kind in _PANEL:
        if sense != "max":
            continue
        start = len(spy.admitted)
        model = from_nl(str(_NL_DIR / f"{name}.nl"))
        assert model._objective.sense == ObjectiveSense.MAXIMIZE
        result = model.solve(time_limit=time_limit)
        admitted = spy.admitted[start:]
        if not admitted:
            continue

        # The merge keeps the tightest upper bound, i.e. the smallest value in
        # max-space, i.e. the largest in min-space.
        expected = -max(admitted)
        assert result.bound is not None, (
            f"{name}: the gate admitted {admitted!r} but no dual bound was reported"
        )
        assert result.bound == pytest.approx(expected, rel=1e-9), (
            f"{name}: reported bound {result.bound!r} is not the negation of the "
            f"admitted min-space value (expected {expected!r}); a MAXIMIZE dual "
            "bound with the wrong sign is a false certificate"
        )
        checks += 1

        if result.objective is not None:
            assert result.bound >= result.objective - 1e-6 * max(1.0, abs(result.objective)), (
                f"{name}: dual bound {result.bound!r} is below its own incumbent "
                f"{result.objective!r} on a MAXIMIZE problem"
            )
            checks += 1

    assert checks > 0, (
        "PROBE NEVER FIRED: no MAXIMIZE panel instance admitted a probe bound, so "
        "the sense conversion was never exercised and this run proves nothing"
    )


@pytest.mark.slow
def test_probe_box_is_snapshotted_not_aliased(monkeypatch):
    """The probe box and the root box must be independent arrays.

    The gate's entire soundness argument is ``np.array_equal(probe_box,
    root_box)``. That argument evaporates if the two are the same object: an
    array always equals itself, so a bound proved over the FBBT/OBBT-tightened
    box (a strict subset of the root box under ``DISCOPT_ROOT_LP_PROBE_TIGHT``,
    default ON) would be admitted as a global bound. The gate cannot see this;
    it is prevented only by the ``.copy()`` at the capture site in
    ``solve_model``. Nothing else in the suite would catch its removal — the
    other tests would keep passing, and more bounds would be "admitted".
    """
    spy = _Spy().install(monkeypatch)

    for name, time_limit, _sense, _oracle, _kind in _PANEL:
        model = from_nl(str(_NL_DIR / f"{name}.nl"))
        model.solve(time_limit=time_limit)
        if spy.admitted:
            break

    assert spy.admitted, (
        "PROBE NEVER FIRED: no panel instance admitted a probe bound, so no "
        "box pair was inspected and this run proves nothing"
    )
    assert not any(spy.aliased), (
        "the probe box shares memory with the root box: box equality is then "
        "trivially true and a bound proved over the tightened sub-box would be "
        f"admitted as a global bound (aliased on {sum(spy.aliased)} of "
        f"{len(spy.aliased)} admissions)"
    )
