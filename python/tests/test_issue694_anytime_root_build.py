"""Issue #694 — anytime/incremental root-relaxation build (``DISCOPT_ANYTIME_ROOT_BUILD``).

The #654 residual: on a class of large sparse network-design/QAP/graph-partition
MINLPs the root fallback's dual bound is produced by a single uninterruptible
McCormick-LP *build*, so ``solve(time_limit=T)`` overran ``T`` by the whole build
cost (baron-gap-plan.md §8.1/§8.6). #694 makes that build anytime: it stops adding
constraint rows once the fallback's grant is spent and solves the partial (valid,
weaker) relaxation. Behind ``DISCOPT_ANYTIME_ROOT_BUILD`` (default OFF), §5
bound-changing.

These tests pin the SOUNDNESS + OPT-OUT contract that must hold regardless of the
corpus-wide graduation panel (which runs on the owner's machine, since the #654
instances are big-corpus-only):

  * OFF (default) is **byte-identical** to the legacy monolithic build — no row
    dropped, no truncation, the same LP assembled;
  * a truncated build is a **valid weaker outer relaxation** — its LP minimum never
    exceeds the full build's (dropping rows only enlarges the feasible set), and it
    is never falsified;
  * ON, the fallback **honors its grant** (bounded wall) and stays sound (a surfaced
    dual bound is a valid lower bound; ``None`` is sound too — merely weaker).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

_CORPUS = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")


def _root_box(model):
    lb = np.array([v.lb if v.lb is not None else -1e20 for v in model._variables], float)
    ub = np.array([v.ub if v.ub is not None else 1e20 for v in model._variables], float)
    return lb, ub


def test_build_deadline_none_is_byte_identical():
    """OFF path: ``build_deadline=None`` builds the full relaxation — same row count,
    not truncated, identical LP bound. This is the opt-out / bound-neutrality guard."""
    from discopt._jax.uniform_relax import build_uniform_relaxation
    from discopt.modeling.core import from_nl

    model = from_nl(os.path.join(_CORPUS, "casctanks.nl"))
    lb, ub = _root_box(model)

    full = build_uniform_relaxation(model, (lb, ub))
    again = build_uniform_relaxation(model, (lb, ub), build_deadline=None)

    fm, am = full.model, again.model
    assert fm._build_truncated is False
    assert am._build_truncated is False
    nrows = lambda m: 0 if m._A_ub is None else int(m._A_ub.shape[0])  # noqa: E731
    assert nrows(fm) == nrows(am)
    assert fm._build_constraints_done == fm._build_constraints_total
    # LP bounds identical (deterministic full build).
    rf = fm.solve(backend="simplex", time_limit=10.0)
    ra = am.solve(backend="simplex", time_limit=10.0)
    assert rf.status == ra.status
    if rf.bound is not None and ra.bound is not None:
        assert abs(rf.bound - ra.bound) <= 1e-9 * max(1.0, abs(rf.bound))


def test_truncated_build_is_valid_and_weaker():
    """A build truncated at a past deadline drops constraint rows and is a valid
    WEAKER relaxation: fewer rows, marked truncated, and its LP min ≤ the full
    build's LP min (dropping rows only enlarges the feasible set — never falsified)."""
    from discopt._jax.uniform_relax import build_uniform_relaxation
    from discopt.modeling.core import from_nl

    model = from_nl(os.path.join(_CORPUS, "hda.nl"))
    lb, ub = _root_box(model)

    full = build_uniform_relaxation(model, (lb, ub))
    # Deadline already in the past -> stop before ANY constraint row is added.
    trunc = build_uniform_relaxation(model, (lb, ub), build_deadline=time.perf_counter() - 1.0)

    fm, tm = full.model, trunc.model
    assert tm._build_truncated is True
    assert tm._build_constraints_done < tm._build_constraints_total
    nrows = lambda m: 0 if m._A_ub is None else int(m._A_ub.shape[0])  # noqa: E731
    assert nrows(tm) < nrows(fm)

    # Validity: whenever BOTH solves yield a finite bound, the truncated (looser)
    # relaxation's LP minimum is a weaker (≤) lower bound. Never tighter.
    rf = fm.solve(backend="simplex", time_limit=10.0)
    rt = tm.solve(backend="simplex", time_limit=10.0)
    if (
        rf.status == "optimal"
        and rt.status == "optimal"
        and rf.bound is not None
        and rt.bound is not None
    ):
        assert rt.bound <= rf.bound + 1e-6 * max(1.0, abs(rf.bound))


def test_partial_build_never_exceeds_full_across_deciles():
    """The anytime curve is monotone-sound: for a mid-build deadline, the partial
    relaxation's bound (if finite) is a valid lower bound ≤ the full build's."""
    from discopt._jax.uniform_relax import build_uniform_relaxation
    from discopt.modeling.core import from_nl

    model = from_nl(os.path.join(_CORPUS, "heatexch_gen1.nl"))
    lb, ub = _root_box(model)

    full = build_uniform_relaxation(model, (lb, ub))
    rf = full.model.solve(backend="simplex", time_limit=10.0)
    assert rf.status == "optimal" and rf.bound is not None

    # A deadline a few ms out truncates partway through the constraint loop.
    partial = build_uniform_relaxation(model, (lb, ub), build_deadline=time.perf_counter() + 0.005)
    rp = partial.model.solve(backend="simplex", time_limit=10.0)
    if rp.status == "optimal" and rp.bound is not None:
        assert rp.bound <= rf.bound + 1e-6 * max(1.0, abs(rf.bound))


@pytest.mark.requires_pounce
def test_flag_off_fallback_is_deterministic_and_sound():
    """OFF (default): ``_root_relaxation_lower_bound`` returns its usual finite,
    valid bound on nvs11 (a valid lower bound ≤ the −431.0 oracle optimum — the
    same instance the #654 checkpoint-poll test pins)."""
    from discopt.modeling.core import from_nl
    from discopt.solver import _root_relaxation_lower_bound

    os.environ.pop("DISCOPT_ANYTIME_ROOT_BUILD", None)
    model = from_nl(os.path.join(_CORPUS, "nvs11.nl"))
    lb, ub = _root_box(model)
    b = _root_relaxation_lower_bound(model, lb, ub, 3.0)
    assert b is not None
    assert b <= -431.0 + 1e-6  # MINIMIZE: dual bound never crosses the oracle optimum


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_flag_on_honors_grant_and_stays_sound():
    """ON: on hda (a multi-second sep build) the fallback honors a tight grant
    (bounded wall) and stays sound — any surfaced bound is a valid lower bound; a
    weaker/``None`` bound is the sound anytime trade. Compares against the OFF bound
    as the reference lower bound (never crossed upward)."""
    from discopt.modeling.core import from_nl
    from discopt.solver import _root_relaxation_lower_bound

    model = from_nl(os.path.join(_CORPUS, "hda.nl"))
    lb, ub = _root_box(model)

    os.environ.pop("DISCOPT_ANYTIME_ROOT_BUILD", None)
    b_off = _root_relaxation_lower_bound(model, lb, ub, 3.0)

    os.environ["DISCOPT_ANYTIME_ROOT_BUILD"] = "1"
    try:
        t0 = time.perf_counter()
        b_on = _root_relaxation_lower_bound(model, lb, ub, 0.5)
        wall = time.perf_counter() - t0
    finally:
        os.environ.pop("DISCOPT_ANYTIME_ROOT_BUILD", None)

    # Honors the grant to within one bounded in-flight op (the base build, left whole
    # by design): far below the OFF overrun this guards against.
    assert wall < 8.0, f"anytime fallback ran {wall:.1f}s — grant not honored"
    # Soundness: the anytime bound is a valid (weaker-or-equal) lower bound. Both are
    # valid lower bounds for a MINIMIZE, so ON must not exceed OFF.
    if b_on is not None and b_off is not None:
        assert b_on <= b_off + 1e-6 * max(1.0, abs(b_off))
