"""Head-to-head parity: discopt's monotonicity detector vs. SUSPECT.

This is the monotonicity counterpart to ``test_convexity_suspect_parity.py``.
SUSPECT (``cog-suspect``) reports a per-expression ``Monotonicity`` verdict
(Ceccon, Siirola, Misener, 2020) on the *raw* body -- increasing / decreasing /
constant / unknown. discopt now exposes a comparable verdict via
:func:`discopt._jax.monotonicity.classify_monotonicity`. Both run over a single
shared corpus and are compared item-by-item.

A key asymmetry, established when this suite was built, shapes what is asserted:

* **discopt's monotonicity is rigorous.** A ``NONDECREASING`` / ``NONINCREASING``
  verdict is an interval-gradient *proof* (``∇f`` enclosed in the nonneg / nonpos
  orthant over the whole box). Every such verdict is validated here directly
  against a dense numeric sampling of the real body -- independent of SUSPECT.

* **SUSPECT's monotonicity is heuristic and occasionally unsound.** Its interval
  cosine ignores interior critical points, so e.g. it declares ``sin`` monotone
  on ``[-3, 3]`` where it is not. discopt correctly abstains there. So a
  *disagreement* is not evidence against discopt; discopt's verdicts stand on
  their own validated soundness, and SUSPECT is the side that errs.

Assertions
----------
1. **discopt monotonicity is sound** -- every discopt monotone verdict matches a
   coordinate-wise numeric sampling of the body (the shippable guarantee).
2. **No proven-direction contradictions** -- discopt and SUSPECT never prove
   opposite directions (nondecreasing vs nonincreasing) on the same body.
   Because discopt's side is independently validated sound, any such conflict
   would be a SUSPECT defect; the set is pinned empty.
3. **Detector-gap sets are pinned** -- instances where one side proves a
   direction the other leaves UNKNOWN are pinned so a regression (or a newly
   closed gap) is caught and triaged rather than silently drifting.

Mechanics mirror the convexity suite: SUSPECT cannot run in this environment,
so its verdicts are recorded once into
``scripts/suspect_oracle/suspect_verdicts.json`` (see that directory's README).
This test imports only the neutral corpus and the discopt renderer.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

_ORACLE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "suspect_oracle"
_GOLDEN = _ORACLE_DIR / "suspect_verdicts.json"

if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))

pytestmark = pytest.mark.skipif(
    not _GOLDEN.exists(),
    reason=(
        "SUSPECT golden verdicts not found; regenerate via "
        "scripts/suspect_oracle/run_suspect.py (see its README)."
    ),
)

# Instances where SUSPECT proves a monotone direction discopt leaves UNKNOWN.
# Pinned so a newly discovered gap fails loudly. These are sound *for SUSPECT to
# report* only in the first two cases; the third is a SUSPECT defect that
# discopt rightly avoids:
#   - sqrt_concave : sqrt is nondecreasing, but its derivative is unbounded at
#     x=0 (the box touches 0), so discopt's interval-gradient enclosure abstains.
#   - cubic        : x^3 is nondecreasing (3x^2 >= 0), but squaring outward-rounds
#     the exact 0 minimum to a sub-ULP negative, so the gradient enclosure admits
#     a negative sliver and discopt abstains. A sound (if conservative) miss.
#   - sine         : SUSPECT declares sin(x) on [-3, 3] *nonincreasing* via its
#     unsound interval cosine; sin is NOT monotone there. discopt correctly
#     abstains. This entry documents a SUSPECT unsoundness, not a discopt gap.
KNOWN_SUSPECT_STRONGER: frozenset[str] = frozenset(
    {
        "sqrt_concave::objective",
        "cubic::objective",
        "sine::objective",
    }
)

# Instances where discopt proves a direction SUSPECT leaves UNKNOWN. Pinned so a
# recognizer regression is caught. Currently none.
EXPECTED_DISCOPT_STRONGER: frozenset[str] = frozenset()

_DIRECTIONS = {
    "nondecreasing": "nondecreasing",
    "nonincreasing": "nonincreasing",
    "constant": "constant",
    "unknown": "unknown",
}


def _token(verdict) -> str:
    """discopt ``Monotonicity`` enum -> short lowercase token."""
    return verdict.value


def _compatible(a: str, b: str) -> bool:
    """``constant`` is both nondecreasing and nonincreasing, so it never
    conflicts with a proven direction (mirrors affine ⊆ convex∩concave)."""
    if a == b:
        return True
    return "constant" in (a, b) and a != "unknown" and b != "unknown"


def _compare(discopt_token: str, suspect_token: str) -> dict:
    if _compatible(discopt_token, suspect_token):
        category = "agree"
    elif discopt_token != "unknown" and suspect_token == "unknown":
        category = "discopt_stronger"
    elif suspect_token != "unknown" and discopt_token == "unknown":
        category = "suspect_stronger"
    else:
        # Both proven and incompatible -> one says nondecreasing, other
        # nonincreasing on the same raw body.
        category = "contradiction"
    return {"discopt": discopt_token, "suspect": suspect_token, "category": category}


def _classify_all() -> dict[str, dict]:
    """discopt vs golden monotonicity verdicts, keyed ``<instance>::<item>``."""
    from corpus import INSTANCES
    from discopt._jax.monotonicity import classify_monotonicity
    from render_discopt import build_discopt_items

    golden = json.loads(_GOLDEN.read_text())["verdicts"]
    results: dict[str, dict] = {}

    for inst in INSTANCES:
        name = inst["name"]
        model, items = build_discopt_items(inst)
        g = golden[name]
        for item in items:
            key = item["key"]
            gd = g["objective"] if key == "objective" else g["constraints"][key]
            if gd is None:
                continue
            discopt_token = _token(classify_monotonicity(item["body"], model))
            suspect_token = _DIRECTIONS.get(gd["monotonicity"], "unknown")
            results[f"{name}::{key}"] = _compare(discopt_token, suspect_token)

    return results


# --- Independent numeric validation of discopt's verdicts --------------------


def _sampled_directions(ast, vbounds: dict, n: int = 9) -> dict:
    """Coordinate-wise monotonicity of ``ast`` sampled over its box.

    Returns ``{"nondecreasing": bool, "nonincreasing": bool, "constant": bool}``
    where each flag is whether the *sampled* body is consistent with that
    property: holding every other variable fixed on a grid, sweep one variable
    and check the value never decreases / never increases / never changes.
    """
    from corpus import eval_ast

    names = list(vbounds)
    grids = {nm: np.linspace(lb, ub, n) for nm, (lb, ub) in vbounds.items()}
    nondec = noninc = const = True
    for vi in names:
        others = [o for o in names if o != vi]
        other_combos = product(*[grids[o] for o in others]) if others else [()]
        for combo in other_combos:
            base = dict(zip(others, combo))
            seq = [eval_ast(ast, {**base, vi: float(t)}) for t in grids[vi]]
            for a, b in zip(seq, seq[1:]):
                tol = 1e-7 * (1.0 + max(abs(a), abs(b)))
                if b < a - tol:
                    nondec = False
                if b > a + tol:
                    noninc = False
                if abs(b - a) > tol:
                    const = False
    return {"nondecreasing": nondec, "nonincreasing": noninc, "constant": const}


@pytest.fixture(scope="module")
def parity() -> dict[str, dict]:
    return _classify_all()


def test_discopt_monotonicity_is_sound() -> None:
    """Every discopt monotone/constant verdict matches a dense numeric sampling.

    This validates discopt's detector directly against the real body, with no
    reference to SUSPECT -- the core soundness guarantee for the new API.
    """
    from corpus import INSTANCES, item_asts
    from discopt._jax.monotonicity import classify_monotonicity
    from render_discopt import build_discopt_items

    violations = []
    for inst in INSTANCES:
        asts = item_asts(inst)
        model, items = build_discopt_items(inst)
        for item in items:
            key = item["key"]
            token = _token(classify_monotonicity(item["body"], model))
            if token == "unknown":
                continue  # abstentions need no witness
            sampled = _sampled_directions(asts[key], inst["vars"])
            if not sampled[token]:
                violations.append(
                    f"{inst['name']}::{key} discopt says {token} but sampling "
                    f"contradicts it (sampled={sampled})"
                )
    assert not violations, "discopt produced an UNSOUND monotonicity verdict:\n" + "\n".join(
        violations
    )


def test_no_monotonicity_contradictions(parity: dict[str, dict]) -> None:
    """discopt and SUSPECT never prove opposite directions on the same body.

    discopt's verdicts are independently validated sound (see the sampling
    test), so a contradiction here would be a SUSPECT defect. The set is empty.
    """
    contradictions = {k: v for k, v in parity.items() if v["category"] == "contradiction"}
    assert not contradictions, (
        "discopt and SUSPECT give opposite monotone directions on identical raw "
        f"bodies: {contradictions}"
    )


def test_monotonicity_gaps_pinned(parity: dict[str, dict]) -> None:
    """Detector-gap sets (either side stronger) must match the pinned sets."""
    suspect_stronger = {k for k, v in parity.items() if v["category"] == "suspect_stronger"}
    new_gaps = suspect_stronger - KNOWN_SUSPECT_STRONGER
    assert not new_gaps, (
        "SUSPECT proves a monotone direction discopt leaves UNKNOWN on new "
        f"instances (gap to triage): {sorted(new_gaps)}"
    )
    stale = KNOWN_SUSPECT_STRONGER - suspect_stronger
    assert not stale, (
        "These instances are no longer SUSPECT-stronger; drop them from "
        f"KNOWN_SUSPECT_STRONGER: {sorted(stale)}"
    )

    discopt_stronger = {k for k, v in parity.items() if v["category"] == "discopt_stronger"}
    regressed = EXPECTED_DISCOPT_STRONGER - discopt_stronger
    assert not regressed, (
        "discopt no longer proves these monotone directions SUSPECT misses -- "
        f"a regression: {sorted(regressed)}"
    )
    unexpected = discopt_stronger - EXPECTED_DISCOPT_STRONGER
    assert not unexpected, (
        "discopt newly proves a direction SUSPECT misses; if intended, add to "
        f"EXPECTED_DISCOPT_STRONGER: {sorted(unexpected)}"
    )


def test_corpus_fully_compared(parity: dict[str, dict]) -> None:
    """Every golden item is exercised (guards against silent corpus drift)."""
    golden = json.loads(_GOLDEN.read_text())["verdicts"]
    expected_items = 0
    for _name, g in golden.items():
        if g.get("objective") is not None:
            expected_items += 1
        expected_items += len(g.get("constraints", {}))
    assert len(parity) == expected_items, (
        f"compared {len(parity)} items but golden has {expected_items}; "
        "corpus and golden file are out of sync -- regenerate the golden."
    )
