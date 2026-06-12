"""Record SUSPECT's verdicts over the shared corpus into a golden JSON.

Runs only inside the isolated SUSPECT environment. SUSPECT (cog-suspect 2.1.3)
is unmaintained and incompatible with discopt's own numpy / pyomo, so its
verdicts are captured once, here, and committed as ``suspect_verdicts.json``.
The in-repo parity tests (``python/tests/test_convexity_suspect_parity.py``,
``test_monotonicity_suspect_parity.py``, ``test_fbbt_bounds_suspect_parity.py``)
compare discopt against that recorded file and never import SUSPECT.

Three verdict axes are recorded per objective / constraint:

* ``convexity``    -- the sense-/``<=``-normalised body curvature, from the
  high-level ``detect_special_structure``. For ``x^2 <= 4`` it returns
  ``Convex``; for ``x^2 >= 4``, ``Concave``. Persisted with the constraint
  sense so the discopt side can normalise identically.
* ``monotonicity`` -- the *raw-body* increasing / decreasing / constant verdict.
* ``bounds``       -- the *raw-body* FBBT interval enclosure (``lower``/``upper``,
  ``null`` for an unbounded endpoint).

The latter two come from SUSPECT's lower-level ``perform_fbbt`` +
``propagate_special_structure`` pass, which keys by the raw ``obj.expr`` /
``con.body`` (no sense normalisation), so the discopt side classifies the
identical raw body.

Usage (see README.md for the exact environment recipe)::

    python run_suspect.py            # writes suspect_verdicts.json next to this file
    python run_suspect.py --check    # print verdicts, do not write
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# --- Legacy shims: cog-suspect 2.1.3 predates numpy 1.24 alias removals. ------
# These must run *before* `import suspect`, so the SUSPECT / corpus / pyomo
# imports below are deliberately placed after the shim (hence the noqa: E402).
import numpy as _np

for _name, _ty in (
    ("float", float),
    ("int", int),
    ("bool", bool),
    ("object", object),
    ("complex", complex),
):
    if not hasattr(_np, _name):
        setattr(_np, _name, _ty)

warnings.filterwarnings("ignore")

import pyomo.environ as pe  # noqa: E402
from corpus import INSTANCES  # noqa: E402,I001
from render_pyomo import build_pyomo  # noqa: E402
from suspect import detect_special_structure  # noqa: E402
from suspect.fbbt import perform_fbbt  # noqa: E402
from suspect.propagation import propagate_special_structure  # noqa: E402

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "suspect_verdicts.json"


def _convexity_str(value) -> str:
    """Normalise a SUSPECT ``Convexity`` enum to a short lowercase token."""
    name = str(value).split(".")[-1].lower()  # "Convexity.Convex" -> "convex"
    if name in ("convex", "concave", "linear", "unknown"):
        return name
    return "unknown"


def _monotonicity_str(value) -> str:
    """Normalise a SUSPECT ``Monotonicity`` enum to a short lowercase token."""
    name = str(value).split(".")[-1].lower()  # "Monotonicity.Nondecreasing" -> ...
    if name in ("nondecreasing", "nonincreasing", "constant", "unknown"):
        return name
    return "unknown"


def _bound(value):
    """A SUSPECT interval endpoint as a JSON-safe float, or ``None`` for an
    unbounded (``±inf``) endpoint. SUSPECT already uses ``None`` for unbounded
    endpoints; finite floats pass straight through."""
    if value is None:
        return None
    fval = float(value)
    return None if fval != fval or abs(fval) == float("inf") else fval


def _raw_mono_bounds(instance: dict) -> dict:
    """SUSPECT's *raw-body* monotonicity + FBBT interval bounds per item.

    Returns ``{item_key: {"monotonicity": token, "bounds": {"lower", "upper"}}}``
    keyed by ``"objective"`` and each constraint name. Uses SUSPECT's
    lower-level ``perform_fbbt`` + ``propagate_special_structure`` (which key by
    the raw ``obj.expr`` / ``con.body``), so the values are *not* sense- or
    ``<=``-normalised -- the discopt side classifies the same raw body.
    """
    pm = build_pyomo(instance)
    bounds = perform_fbbt(pm)
    mono, _cvx = propagate_special_structure(pm, bounds)

    out: dict = {}
    for obj in pm.component_data_objects(pe.Objective, active=True):
        b = bounds.get(obj.expr)
        out["objective"] = {
            "monotonicity": _monotonicity_str(mono[obj.expr]),
            "bounds": {"lower": _bound(b.lower_bound), "upper": _bound(b.upper_bound)},
        }
    for con in pm.component_data_objects(pe.Constraint, active=True):
        b = bounds.get(con.body)
        out[con.name] = {
            "monotonicity": _monotonicity_str(mono[con.body]),
            "bounds": {"lower": _bound(b.lower_bound), "upper": _bound(b.upper_bound)},
        }
    return out


def run_instance(instance: dict) -> dict:
    """Return SUSPECT's verdicts for one instance, or an error marker.

    Captures, per objective / constraint:
    * ``convexity`` -- the sense-/``<=``-normalised body curvature (from the
      high-level ``detect_special_structure``); and
    * ``monotonicity`` + ``bounds`` -- the *raw-body* monotonicity verdict and
      FBBT interval enclosure (from the lower-level FBBT + propagation pass).
    """
    try:
        model = build_pyomo(instance)
        info = detect_special_structure(model)
        raw = _raw_mono_bounds(instance)
    except Exception as exc:  # SUSPECT FBBT can raise (e.g. EmptyIntervalError)
        return {"error": f"{type(exc).__name__}: {exc}"}

    out: dict = {"objective": None, "constraints": {}}

    if info.objectives:
        # Single objective named "objective" in our renderer.
        for _oname, odata in info.objectives.items():
            ro = raw.get("objective", {})
            out["objective"] = {
                "sense": odata.get("sense"),
                "convexity": _convexity_str(odata["convexity"]),
                "monotonicity": ro.get("monotonicity", "unknown"),
                "bounds": ro.get("bounds", {"lower": None, "upper": None}),
            }
            break

    sense_by_name = {c["name"]: c["op"] for c in instance["constraints"]}
    for cname, cdata in info.constraints.items():
        rc = raw.get(cname, {})
        out["constraints"][cname] = {
            "op": sense_by_name.get(cname),
            "convexity": _convexity_str(cdata["convexity"]),
            "monotonicity": rc.get("monotonicity", "unknown"),
            "bounds": rc.get("bounds", {"lower": None, "upper": None}),
        }
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="print verdicts without writing the JSON file"
    )
    args = parser.parse_args(argv)

    verdicts = {inst["name"]: run_instance(inst) for inst in INSTANCES}

    errors = {k: v["error"] for k, v in verdicts.items() if "error" in v}
    payload = {
        "_meta": {
            "tool": "cog-suspect",
            "tool_version": "2.1.3",
            "pyomo": "6.1.2",
            "n_instances": len(verdicts),
            "n_errors": len(errors),
        },
        "verdicts": verdicts,
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.check:
        print(text)
    else:
        _OUT.write_text(text + "\n")
        print(f"wrote {_OUT} ({len(verdicts)} instances, {len(errors)} SUSPECT errors)")
    if errors:
        print("SUSPECT errored on:", ", ".join(sorted(errors)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
