"""Head-to-head cross-check: discopt's interval bounds vs. SUSPECT's FBBT.

SUSPECT propagates interval bounds for every expression; discopt has its own
interval evaluator (:func:`discopt._jax.convexity.interval_eval.evaluate_interval`).
This suite cross-checks the two enclosures of the *same raw body* over one
shared corpus.

What the two actually compute differs, and that difference is the point of the
cross-check rather than a flaw in it:

* **discopt** -- a sound *forward* natural-range enclosure: propagate the
  variable box through the expression. It does not use a constraint's RHS.
* **SUSPECT** -- the result of full FBBT: forward propagation *intersected with
  backward propagation from the constraint RHS*. So on a constraint body
  SUSPECT is typically tighter (it knows ``x^2 + y^2 <= 1``), while on
  quadratics / powers / ``tan`` discopt is tighter (SUSPECT often leaves those
  unbounded above).

Established facts this suite encodes:

* discopt's forward enclosure is **sound on every item** -- it contains the
  body's dense numeric sampling. This is asserted directly and is the shippable
  guarantee for the bounds API.
* The two enclosures are **never disjoint**. Two enclosures of the same body
  must overlap; a disjoint pair would prove one tool unsound. (SUSPECT's
  interval trig *is* occasionally unsound -- it under-encloses ``sin`` on a wide
  box -- but always to a *subset* of discopt's correct enclosure, so overlap
  still holds.)
* discopt abstains to an unbounded endpoint on a small, pinned set (the
  sqrt-of-sum-of-squares bodies, whose inner square outward-rounds its exact 0
  minimum just below 0 and poisons the sqrt domain, and the inverse-trig atoms
  the interval evaluator does not model).

Mechanics mirror the convexity / monotonicity suites: SUSPECT's bounds are
recorded once into ``scripts/suspect_oracle/suspect_verdicts.json`` (see that
directory's README); this test imports only the neutral corpus and the discopt
renderer.
"""

from __future__ import annotations

import json
import math
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

# Items where discopt's forward interval has an unbounded endpoint and so cannot
# be compared for tightness. Pinned so the set cannot silently grow.
#   - euclidean_norm / norm_le : sqrt(x^2 + y^2); the inner square's exact 0
#     minimum outward-rounds just below 0, poisoning sqrt's domain to a -inf
#     lower endpoint (sound, but loose).
#   - asin / acos / atan       : the interval evaluator has no enclosure for the
#     inverse-trig atoms and soundly returns (-inf, +inf).
DISCOPT_UNBOUNDED: frozenset[str] = frozenset(
    {
        "euclidean_norm::objective",
        "norm_le::soc",
        "asin_convex_branch::objective",
        "acos_concave_branch::objective",
        "atan_concave_branch::objective",
    }
)

_TOL = 1e-6


def _lo(x) -> float:
    return -math.inf if x is None else float(x)


def _hi(x) -> float:
    return math.inf if x is None else float(x)


def _discopt_interval(body, model) -> tuple[float, float]:
    from discopt._jax.convexity.interval_eval import evaluate_interval

    iv = evaluate_interval(body, model)
    return float(np.asarray(iv.lo).ravel()[0]), float(np.asarray(iv.hi).ravel()[0])


def _sampled_range(ast, vbounds: dict, n: int = 11) -> tuple[float, float]:
    """Min / max of the raw body sampled on a dense grid over its box."""
    from corpus import eval_ast

    names = list(vbounds)
    grids = [np.linspace(lb, ub, n) for lb, ub in (vbounds[nm] for nm in names)]
    mn, mx = math.inf, -math.inf
    for combo in product(*grids):
        env = {nm: float(t) for nm, t in zip(names, combo)}
        y = eval_ast(ast, env)
        mn = min(mn, y)
        mx = max(mx, y)
    return mn, mx


def _iter_items():
    """Yield ``(item_key, body, ast, vbounds, suspect_bounds)`` for every item."""
    from corpus import INSTANCES, item_asts
    from render_discopt import build_discopt_items

    golden = json.loads(_GOLDEN.read_text())["verdicts"]
    for inst in INSTANCES:
        name = inst["name"]
        asts = item_asts(inst)
        model, items = build_discopt_items(inst)
        g = golden[name]
        for item in items:
            key = item["key"]
            gd = g["objective"] if key == "objective" else g["constraints"][key]
            if gd is None:
                continue
            yield (
                f"{name}::{key}",
                item["body"],
                model,
                asts[key],
                inst["vars"],
                gd["bounds"],
            )


def test_discopt_bounds_are_sound() -> None:
    """discopt's forward enclosure contains the body's dense numeric sampling."""
    violations = []
    for ikey, body, model, ast, vbounds, _sb in _iter_items():
        dlo, dhi = _discopt_interval(body, model)
        smn, smx = _sampled_range(ast, vbounds)
        if dlo > smn + _TOL or dhi < smx - _TOL:
            violations.append(
                f"{ikey}: discopt enclosure [{dlo:.6g}, {dhi:.6g}] does not "
                f"contain sampled range [{smn:.6g}, {smx:.6g}]"
            )
    assert not violations, "discopt produced an UNSOUND interval enclosure:\n" + "\n".join(
        violations
    )


def test_no_disjoint_enclosures() -> None:
    """discopt's and SUSPECT's enclosures of the same body always overlap.

    Two enclosures of one quantity can never be disjoint; a disjoint pair would
    prove one tool unsound.
    """
    disjoint = []
    for ikey, body, model, _ast, _vbounds, sb in _iter_items():
        dlo, dhi = _discopt_interval(body, model)
        slo, shi = _lo(sb["lower"]), _hi(sb["upper"])
        if shi < dlo - _TOL or slo > dhi + _TOL:
            disjoint.append(
                f"{ikey}: discopt [{dlo:.6g}, {dhi:.6g}] disjoint from "
                f"SUSPECT [{slo:.6g}, {shi:.6g}]"
            )
    assert not disjoint, "discopt and SUSPECT give disjoint enclosures:\n" + "\n".join(disjoint)


def test_discopt_unbounded_set_pinned() -> None:
    """Exactly the pinned items have an unbounded discopt endpoint."""
    unbounded = set()
    for ikey, body, model, _ast, _vbounds, _sb in _iter_items():
        dlo, dhi = _discopt_interval(body, model)
        if math.isinf(dlo) or math.isinf(dhi):
            unbounded.add(ikey)
    new = unbounded - DISCOPT_UNBOUNDED
    assert not new, (
        "discopt's interval evaluator newly abstains to an unbounded endpoint "
        f"on: {sorted(new)} -- tighten or add to DISCOPT_UNBOUNDED with reason."
    )
    stale = DISCOPT_UNBOUNDED - unbounded
    assert not stale, (
        "These items are no longer unbounded in discopt; drop them from "
        f"DISCOPT_UNBOUNDED: {sorted(stale)}"
    )


def test_corpus_fully_compared() -> None:
    """Every golden item is exercised (guards against silent corpus drift)."""
    golden = json.loads(_GOLDEN.read_text())["verdicts"]
    expected_items = 0
    for _name, g in golden.items():
        if g.get("objective") is not None:
            expected_items += 1
        expected_items += len(g.get("constraints", {}))
    compared = sum(1 for _ in _iter_items())
    assert compared == expected_items, (
        f"compared {compared} items but golden has {expected_items}; "
        "corpus and golden file are out of sync -- regenerate the golden."
    )
