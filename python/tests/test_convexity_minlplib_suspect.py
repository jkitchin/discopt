"""MINLPLib SUSPECT-failure convexity regression coverage (issue #40).

A label-driven soundness sweep of discopt's convexity detector over a curated
set of MINLPLib instances drawn from the SUSPECT regression list
(cog-imperial/suspect#11, the follow-up corpus referenced from #38 / #40).

Ground truth comes from MINLPLib's own ``instancedata.csv`` ``convex`` column
(authoritative, conservative: an instance is labelled ``convex`` only when
provably so), pinned directly into this module so the test is self-contained and
needs no CSV at run time. Each instance falls into one of three pinned buckets:

Every verdict is taken with ``use_certificate=True`` — the exact setting the
solver uses to gate ``convex_fast_path`` (see ``solver.py`` and ``amp.py``), so
these assertions characterise the real fast-path gate, not a weaker structural-
only path. Each instance falls into one of three pinned buckets:

* ``NEGATIVE`` — MINLPLib labels it non-convex (or it is an unlabelled
  SUSPECT-regression negative control: genuine pooling / power-law / signomial
  non-convexities such as ``super*`` and ``bchoco*``). The **soundness gate**:
  ``classify_model`` must NOT promote it to convex. This is the non-negotiable
  invariant — a flip here is either a detector bug or an unjustified recogniser.
  Because ``convex_fast_path`` is gated on exactly this model-level verdict,
  asserting it at classification level is equivalent to (and far cheaper than)
  asserting ``result.convex_fast_path is False`` after a full solve.

* ``CONVEX_PROVEN`` — MINLPLib labels it convex and discopt *proves* it (via
  structural recognisers and/or the sound interval-Hessian certificate). The
  **completeness regression guard**: discopt must keep proving these. ``alan``
  is a representative certificate win: a PSD quadratic written in un-expanded
  bilinear form ``Σ xᵢ·(linear)`` that the DCP recognisers miss but the
  constant-Hessian Gershgorin bound proves. Four were previously-pinned gaps now
  closed by dedicated recognisers (#40):

    - ``tls2`` — geometric-mean concavity ``√(xᵢxⱼ)`` (sqrt of a bilinear),
      closed by the geo-mean branch of ``classify_sqrt_pattern``
      (``√(∏ baseᵢ^pᵢ) = ∏ baseᵢ^{pᵢ/2}`` is concave for affine nonneg bases
      with ``Σ pᵢ/2 ≤ 1``).
    - ``clay0303hfsg`` — a perspective reformulation ``(g/L)²·L = g²/L`` whose
      algebraic arrangement the quad-over-affine recogniser missed, closed by
      ``classify_perspective_product``.
    - ``cvxnonsep_nsig30`` / ``cvxnonsep_psig30`` — convex non-separable
      *signomial monomials* ``c·∏ xᵢ^aᵢ``. These are convex in the **original**
      variable space (a stronger PSD certificate would also reach them, but the
      dense non-diagonally-dominant Hessian defeats Gershgorin); the
      signomial-monomial sign-pattern recogniser proves them directly: nsig30 is
      a concave generalized geometric mean (all ``aᵢ ≥ 0``, ``Σaᵢ = 0.971 ≤ 1``)
      negated in a ``≤`` constraint, psig30 a convex monomial (all ``aᵢ ≤ 0``).
      The general GP / log-convex subsystem (for *posynomials* convex only in
      ``y = log x`` space, and signomial programs) remains out of scope — its
      own companion issue.

* ``CONVEX_ABSTAIN`` — MINLPLib labels it convex but discopt still abstains even
  with the certificate. Pinned so the set cannot silently grow; if discopt
  *improves* and proves one, the test fails loudly asking for the instance to be
  promoted to ``CONVEX_PROVEN``. Currently empty — every characterised gap in
  this corpus has been closed.

Most instances are vendored under ``data/minlplib_nl/`` and run in the PR-fast
job. The negative controls that classify too slowly for the PR-fast per-test
budget (``casctanks`` / ``heatexch_gen3`` at ~2 min, ``hda`` at ~90 s, and the
mid-weight ``contvar`` / ``bchoco07`` / ``bchoco08`` at ~5-10 s) live in
``NEGATIVE_SLOW`` and are marked ``slow`` (deselected on the PR-fast / coverage
path, exercised in full / slow runs, so their non-convex soundness coverage is
preserved). A few large negative controls (``super1``..``super3t``)
classify in tens of seconds and are left in the local MINLPLib cache only; they
are exercised when present (``--instances super1,super2,super3,super3t`` via
``fetch_minlplib``) and skipped otherwise.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt._jax.convexity import classify_model

_DATA_DIR = Path(__file__).parent / "data" / "minlplib_nl"

# Pinned MINLPLib convexity ground truth (from instancedata.csv `convex` column;
# `None` = no MINLPLib label, an unlabelled SUSPECT-regression negative control).
# Bucket assignment is verified against the live detector below.

# Non-convex / negative controls: the soundness gate. classify_model must return
# is_convex == False for every one of these.
NEGATIVE = (
    # heat-exchanger / scheduling / supply-chain instances (MINLPLib convex=False)
    "4stufen",
    "beuster",
    "ex1224",
    "gkocis",
    "st_e29",
    "tanksize",
    "tspn05",
    "tspn08",
    "tspn10",
    "tspn12",
    "heatexch_gen1",
    "heatexch_gen2",
    "oaer",
    # batch chocolate scheduling (power-law sizing + big-M); unlabelled in the
    # current MINLPLib snapshot, genuine non-convexities. bchoco06 classifies
    # fast; the larger 07/08 DAGs are in NEGATIVE_SLOW.
    "bchoco06",
)

# Slow vendored negative controls: same soundness gate as ``NEGATIVE`` but each
# classifies too slowly for the PR-fast per-test budget — from ~5 s (dense DAGs)
# up to ~2 minutes (``casctanks`` / ``heatexch_gen3``). Marked
# ``@pytest.mark.slow`` so the PR-fast and coverage jobs deselect them; still
# vendored and exercised in full / slow runs, preserving their non-convex
# soundness coverage.
NEGATIVE_SLOW = (
    "casctanks",
    "heatexch_gen3",
    # hda: signomial equilibrium constraints; its convex structure is in
    # log-space, explicitly out of scope for #40 — discopt soundly stays
    # non-convex here. Classifies in ~90 s (large signomial DAG), so it lives
    # on the slow path rather than the 120 s PR-fast budget.
    "hda",
    # contvar (~10 s) and the larger batch-chocolate DAG bchoco08/07 (~5-6 s):
    # correct non-convex verdicts, but too slow for the PR-fast budget.
    "contvar",
    "bchoco07",
    "bchoco08",
)

# Large negative controls kept in the local cache only (not vendored: each
# classifies in ~15-30 s). Skipped when the cache is absent.
NEGATIVE_CACHE_ONLY = (
    "super1",
    "super2",
    "super3",
    "super3t",
)

# MINLPLib convex == True and discopt proves it: completeness regression guard.
CONVEX_PROVEN = (
    "gbd",
    "fac2",
    "flay02m",
    "flay03m",
    "m3",
    "st_test1",
    "st_testgr3",
    "cvxnonsep_psig40r",
    # certificate win: PSD quadratic in un-expanded bilinear form
    "alan",
    # geometric-mean concavity sqrt(x_i x_j) — closed by the geo-mean
    # branch of classify_sqrt_pattern (#40 gap closure).
    "tls2",
    # perspective product ((g/L)^2)*L = g^2/L — closed by
    # classify_perspective_product (#40 gap closure).
    "clay0303hfsg",
    # convex non-separable signomial monomials, proven in the original
    # variable space by the signomial-monomial sign-pattern recogniser in
    # classify_product_pattern (#40 Tier-1 gap closure): nsig30 is a
    # concave geometric mean (all exponents >= 0, sum 0.971 <= 1) negated
    # in a <= constraint; psig30 is a convex monomial (all exponents <= 0).
    "cvxnonsep_nsig30",
    "cvxnonsep_psig30",
)

# MINLPLib convex == True but discopt still abstains even with the certificate:
# pinned, characterised completeness gaps (see module docstring). If one starts
# proving convex, promote it to CONVEX_PROVEN.
#
# Currently EMPTY — every characterised gap in the SUSPECT-regression corpus has
# been closed (geo-mean / perspective / signomial-monomial recognisers, #40).
# Retained as the pin point for any future MINLPLib-convex instance discopt
# cannot yet prove; the bucket-level guard below tolerates an empty set.
CONVEX_ABSTAIN: tuple[str, ...] = ()


def _cache_nl_dir() -> Path:
    """Local MINLPLib cache nl/ directory (mirrors fetch_minlplib)."""
    env = os.environ.get("DISCOPT_MINLPLIB_CACHE")
    base = Path(env) if env else Path.home() / ".cache" / "discopt" / "minlplib"
    return base / "current" / "nl"


def _resolve(name: str) -> Path | None:
    """Locate an instance .nl: vendored data dir first, then the local cache."""
    vendored = _DATA_DIR / f"{name}.nl"
    if vendored.exists():
        return vendored
    cached = _cache_nl_dir() / f"{name}.nl"
    if cached.exists():
        return cached
    return None


def _classify(name: str) -> tuple[bool, list[bool]]:
    """Ingest the instance and run the model-level convexity verdict.

    Uses ``use_certificate=True`` to mirror the solver's ``convex_fast_path``
    gate (``solver.py`` / ``amp.py``), so the assertions reflect the real path.
    """
    path = _resolve(name)
    assert path is not None, f"instance {name!r} not found in {_DATA_DIR} or cache"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = dm.from_nl(str(path))
        return classify_model(model, use_certificate=True)


@pytest.mark.parametrize("name", NEGATIVE)
def test_minlplib_negative_control_stays_nonconvex(name: str) -> None:
    """SOUNDNESS: a non-convex instance must never be promoted to convex.

    This is the non-negotiable #40 invariant. ``convex_fast_path`` keys off
    exactly this model-level verdict, so ``is_convex is False`` here is the
    cheap, equivalent guarantee that the convex fast path is not taken.
    """
    is_convex, _mask = _classify(name)
    assert is_convex is False, (
        f"SOUNDNESS REGRESSION: {name} is non-convex (MINLPLib/SUSPECT regression "
        f"corpus) but classify_model promoted it to convex. A convex verdict must "
        f"be backed by a structural recogniser with a justified precondition."
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", NEGATIVE_SLOW)
def test_minlplib_slow_negative_control_stays_nonconvex(name: str) -> None:
    """SOUNDNESS on the slow vendored negative controls (~2 min each).

    Same non-negotiable invariant as ``test_minlplib_negative_control_stays_nonconvex``,
    but marked ``slow`` so the PR-fast / coverage jobs (120 s per-test timeout)
    deselect them. Exercised in full / slow runs.
    """
    is_convex, _mask = _classify(name)
    assert is_convex is False, (
        f"SOUNDNESS REGRESSION: {name} is non-convex (MINLPLib/SUSPECT regression "
        f"corpus) but classify_model promoted it to convex."
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", NEGATIVE_CACHE_ONLY)
def test_minlplib_large_negative_control_stays_nonconvex(name: str) -> None:
    """SOUNDNESS on the large cache-only negative controls (super*).

    Skipped unless the instance is present in the local MINLPLib cache.
    """
    if _resolve(name) is None:
        pytest.skip(
            f"{name} not cached; fetch with "
            f"`python -m discopt_benchmarks.scripts.fetch_minlplib --instances {name}`"
        )
    is_convex, _mask = _classify(name)
    assert is_convex is False, f"SOUNDNESS REGRESSION: {name} promoted to convex"


@pytest.mark.parametrize("name", CONVEX_PROVEN)
def test_minlplib_convex_instance_is_proven(name: str) -> None:
    """COMPLETENESS guard: discopt must keep proving these MINLPLib-convex cases."""
    is_convex, mask = _classify(name)
    assert is_convex is True, (
        f"COMPLETENESS REGRESSION: {name} is convex (MINLPLib) and was previously "
        f"proven convex, but classify_model now returns is_convex={is_convex} "
        f"(mask {sum(mask)}/{len(mask)})."
    )


def test_minlplib_convex_abstain_set_is_pinned() -> None:
    """Pin the known completeness gaps (MINLPLib-convex, discopt abstains).

    Currently a no-op: ``CONVEX_ABSTAIN`` is empty (all characterised gaps
    closed). When a future gap is pinned here, a failure means discopt now
    *proves* an instance it used to abstain on — a welcome improvement; move
    that instance to ``CONVEX_PROVEN`` to re-tighten the guard.
    """
    for name in CONVEX_ABSTAIN:
        is_convex, _mask = _classify(name)
        assert is_convex is False, (
            f"DETECTOR IMPROVED: {name} is now proven convex. Promote it from "
            f"CONVEX_ABSTAIN to CONVEX_PROVEN so the completeness guard stays tight."
        )


def test_pinned_buckets_are_disjoint_and_present() -> None:
    """Sanity: buckets don't overlap and every vendored instance is loadable."""
    buckets = {
        "NEGATIVE": set(NEGATIVE),
        "NEGATIVE_SLOW": set(NEGATIVE_SLOW),
        "NEGATIVE_CACHE_ONLY": set(NEGATIVE_CACHE_ONLY),
        "CONVEX_PROVEN": set(CONVEX_PROVEN),
        "CONVEX_ABSTAIN": set(CONVEX_ABSTAIN),
    }
    names = (
        list(NEGATIVE)
        + list(NEGATIVE_SLOW)
        + list(NEGATIVE_CACHE_ONLY)
        + list(CONVEX_PROVEN)
        + list(CONVEX_ABSTAIN)
    )
    assert len(names) == len(set(names)), "duplicate instance across buckets"
    # Every vendored (non-cache-only) instance must be present so CI can run it.
    for name in list(NEGATIVE) + list(NEGATIVE_SLOW) + list(CONVEX_PROVEN) + list(CONVEX_ABSTAIN):
        assert (_DATA_DIR / f"{name}.nl").exists(), f"{name}.nl not vendored"
    assert buckets  # keep the structure referenced
