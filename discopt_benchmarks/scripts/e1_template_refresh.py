"""E1 template-refresh parity harness (docs/dev/scip-parity-kernel-plan.md §3 E1).

For every vendored instance, at the root box AND >=3 perturbed child boxes, build the
uniform relaxation the normal way and reproduce every box-parametric envelope row via
the independent numpy refresh (``e1_refresh``). Compare coefficient-by-coefficient and
report the max ULP deviation per family; ``<=1 ulp`` is the bar. Box-invariant families
(linear passthrough, exact links) are verified to carry ZERO box dependence.

Mechanism: monkeypatch the three box-parametric emitters
(``_emit_mccormick``, ``_emit_1d``, ``_emit_secant_only``) to capture the static
template + FBBT node bounds they receive and the rows they emit, then refresh from the
same inputs and diff in place — no cross-box row matching needed. Every other emitted
row is a box-invariant family; its coefficients are checked for box-independence by
skeleton-keyed comparison across boxes.

Usage:  python discopt_benchmarks/scripts/e1_template_refresh.py
"""

from __future__ import annotations

import struct
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import discopt.modeling as dm
import numpy as np
from discopt._relax import uniform_relax as ur
from discopt._relax.milp_relaxation import _flat_variable_types
from discopt._relax.model_utils import flat_variable_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_refresh as ref  # noqa: E402

NL_DIR = Path(__file__).resolve().parents[2] / "python" / "tests" / "data" / "minlplib_nl"

_BIG = 1e19


# --------------------------------------------------------------------------- #
# ULP distance (monotone-ordering of IEEE-754 doubles)
# --------------------------------------------------------------------------- #
def _ordered(x: float) -> int:
    b = struct.unpack("<q", struct.pack("<d", float(x)))[0]
    return b if b >= 0 else 0x8000000000000000 - b


def ulp_diff(a: float, b: float) -> int:
    if a == b:
        return 0
    return abs(_ordered(a) - _ordered(b))


def _row_finite_ok(coeffs: dict, rhs: float) -> bool:
    """Mirror _Builder.add_row's drop rule (non-finite payload -> row skipped)."""
    import math

    if not math.isfinite(rhs) or abs(rhs) >= _BIG:
        return False
    return all(math.isfinite(c) and abs(c) < _BIG for c in coeffs.values())


def _norm(coeffs: dict) -> dict:
    return {int(j): float(c) for j, c in coeffs.items() if c != 0.0}


# --------------------------------------------------------------------------- #
# Per-run capture state
# --------------------------------------------------------------------------- #
class Capture:
    def __init__(self) -> None:
        # family -> list of (expected_rows, refreshed_rows)
        self.param: dict[str, list[tuple[list, list]]] = defaultdict(list)
        # box-invariant rows emitted this build (coeffs, rhs), skeleton-keyed
        self.invariant: list[tuple[dict, float]] = []
        self._depth = 0  # >0 while inside a wrapped box-parametric emitter


def _install(cap: Capture):
    """Monkeypatch the emitters + add_row; return an uninstall callable."""
    orig_mcc = ur._emit_mccormick
    orig_1d = ur._emit_1d
    orig_sec = ur._emit_secant_only
    orig_add = ur._Builder.add_row

    def add_row(self, coeffs, rhs):
        # A box-parametric emitter's rows are captured by its own wrapper below;
        # everything emitted OUTSIDE one is box-invariant EXCEPT the inline
        # objective separable-floor cut (build_uniform_relaxation, box-dependent
        # RHS via the `sep_lb` local), which is box-parametric — refresh it here.
        if cap._depth == 0 and _row_finite_ok(coeffs, rhs):
            caller = sys._getframe(1)
            loc = caller.f_locals
            if (
                caller.f_code.co_name == "build_uniform_relaxation"
                and loc.get("sep_lb") is not None
                and "obj_lin" in loc
            ):
                obj_lin = loc["obj_lin"]
                expected = [(_norm(coeffs), float(rhs))]
                refreshed = [
                    (_norm(c), float(r))
                    for c, r in ref.refresh_obj_floor(
                        dict(obj_lin.coeffs), float(obj_lin.const), float(loc["sep_lb"])
                    )
                    if _row_finite_ok(c, r)
                ]
                cap.param["objective_floor"].append((expected, refreshed))
            else:
                cap.invariant.append((_norm(coeffs), float(rhs)))
        return orig_add(self, coeffs, rhs)

    def emit_mccormick(ctx, w, la, ba, lb_, bb):
        a_lo, a_hi = ba
        b_lo, b_hi = bb
        tmpl = (w, dict(la.coeffs), float(la.const), dict(lb_.coeffs), float(lb_.const))
        before = len(ctx.rows)
        cap._depth += 1
        try:
            orig_mcc(ctx, w, la, ba, lb_, bb)
        finally:
            cap._depth -= 1
        expected = [(_norm(c), float(r)) for c, r in ctx.rows[before:]]
        refreshed = [
            (_norm(c), float(r))
            for c, r in ref.refresh_mccormick(
                tmpl[0], tmpl[1], tmpl[2], tmpl[3], tmpl[4], a_lo, a_hi, b_lo, b_hi
            )
            if _row_finite_ok(c, r)
        ]
        cap.param["mccormick_bilinear"].append((expected, refreshed))

    def emit_1d(ctx, w, lt, lo, hi, f, fp, curv):
        tmpl = (w, dict(lt.coeffs), float(lt.const))
        before = len(ctx.rows)
        cap._depth += 1
        try:
            out = orig_1d(ctx, w, lt, lo, hi, f, fp, curv)
        finally:
            cap._depth -= 1
        expected = [(_norm(c), float(r)) for c, r in ctx.rows[before:]]
        refreshed = [
            (_norm(c), float(r))
            for c, r in ref.refresh_univariate(tmpl[0], tmpl[1], tmpl[2], lo, hi, f, fp, curv)
            if _row_finite_ok(c, r)
        ]
        # Attribute secant vs tangent for a finer census: the first row is the
        # secant, the rest tangents (matches _emit_1d's emission order).
        cap.param["univariate_secant_tangent"].append((expected, refreshed))
        return out

    def emit_secant_only(ctx, w, lt, lo, hi, f, sign):
        tmpl = (w, dict(lt.coeffs), float(lt.const))
        before = len(ctx.rows)
        cap._depth += 1
        try:
            out = orig_sec(ctx, w, lt, lo, hi, f, sign)
        finally:
            cap._depth -= 1
        expected = [(_norm(c), float(r)) for c, r in ctx.rows[before:]]
        refreshed = [
            (_norm(c), float(r))
            for c, r in ref.refresh_secant_only(tmpl[0], tmpl[1], tmpl[2], lo, hi, f, sign)
            if _row_finite_ok(c, r)
        ]
        cap.param["secant_only"].append((expected, refreshed))
        return out

    ur._emit_mccormick = emit_mccormick
    ur._emit_1d = emit_1d
    ur._emit_secant_only = emit_secant_only
    ur._Builder.add_row = add_row

    def uninstall():
        ur._emit_mccormick = orig_mcc
        ur._emit_1d = orig_1d
        ur._emit_secant_only = orig_sec
        ur._Builder.add_row = orig_add

    return uninstall


# --------------------------------------------------------------------------- #
# Box construction: root + perturbed children
# --------------------------------------------------------------------------- #
def _child_boxes(lb, ub, vtypes, n_shrink=3, seed=0):
    """Root-excluded perturbed boxes: n_shrink random inward shrinks + 1 int-pin."""
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    boxes: list[tuple[str, np.ndarray, np.ndarray]] = []
    for s in range(n_shrink):
        rng = np.random.default_rng(seed * 100 + s)
        nlb, nub = lb.copy(), ub.copy()
        for i in range(len(lb)):
            w = ub[i] - lb[i]
            if not np.isfinite(w) or w <= 0:
                continue
            fl, fu = rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5)
            nlb[i] = lb[i] + fl * w
            nub[i] = ub[i] - fu * w
            if nub[i] < nlb[i]:
                nlb[i] = nub[i] = 0.5 * (lb[i] + ub[i])
        boxes.append((f"shrink{s}", nlb, nub))
    # One box with an integer/binary variable pinned to a lattice point.
    int_idx = [
        i
        for i, t in enumerate(vtypes)
        if str(t).split(".")[-1].lower() in ("integer", "binary")
        and np.isfinite(lb[i])
        and np.isfinite(ub[i])
        and ub[i] > lb[i]
    ]
    if int_idx:
        i = int_idx[0]
        nlb, nub = lb.copy(), ub.copy()
        v = float(np.floor(0.5 * (lb[i] + ub[i])))
        v = min(max(v, lb[i]), ub[i])
        nlb[i] = nub[i] = v
        boxes.append((f"intpin[x{i}={v:g}]", nlb, nub))
    return boxes


def _build_capture(model, box):
    cap = Capture()
    uninstall = _install(cap)
    try:
        ur.build_uniform_relaxation(model, (np.asarray(box[0]), np.asarray(box[1])))
    finally:
        uninstall()
    return cap


# --------------------------------------------------------------------------- #
# Parity evaluation
# --------------------------------------------------------------------------- #
def _cmp_param(cap: Capture, agg: dict):
    """Accumulate per-family (comparisons, max_ulp, count_mismatch)."""
    for fam, groups in cap.param.items():
        st = agg.setdefault(fam, {"cmp": 0, "max_ulp": 0, "count_mismatch": 0, "groups": 0})
        for expected, refreshed in groups:
            st["groups"] += 1
            if len(expected) != len(refreshed):
                st["count_mismatch"] += 1
                continue
            for (ec, er), (rc, rr) in zip(expected, refreshed, strict=True):
                cols = set(ec) | set(rc)
                st["cmp"] += 1
                d = ulp_diff(er, rr)
                for j in cols:
                    d = max(d, ulp_diff(ec.get(j, 0.0), rc.get(j, 0.0)))
                st["max_ulp"] = max(st["max_ulp"], d)


def main() -> int:
    stems = sorted(p.stem for p in NL_DIR.glob("*.nl"))
    print(f"E1 template-refresh parity over {len(stems)} vendored .nl instances")
    print("Boxes per instance: root + 3 random shrink + 1 integer-pin (when available)\n")

    agg: dict[str, dict] = {}
    inv_stats = {"instances": 0, "coeff_box_dependent": 0, "churn_only": 0}
    n_boxes = 0
    n_inst = 0
    failed: list[str] = []

    for stem in stems:
        try:
            model = dm.from_nl(str(NL_DIR / f"{stem}.nl"))
            lb, ub = flat_variable_bounds(model)
            vtypes = _flat_variable_types(model)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{stem}({type(exc).__name__})")
            continue
        boxes = [("root", np.asarray(lb, float), np.asarray(ub, float))]
        # Deterministic per-instance seed (builtin hash is per-process salted).
        boxes += _child_boxes(lb, ub, vtypes, seed=zlib.crc32(stem.encode()) & 0xFFFF)

        inv_by_box: list[Counter] = []
        try:
            for _name, blb, bub in boxes:
                cap = _build_capture(model, (blb, bub))
                _cmp_param(cap, agg)
                inv_by_box.append(Counter((tuple(sorted(c.items())), r) for c, r in cap.invariant))
                n_boxes += 1
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{stem}(build:{type(exc).__name__}:{exc})")
            continue
        n_inst += 1

        # Box-invariance of the invariant families. Compare each child box's
        # invariant-row multiset against root. A row present in root but not the
        # child (or vice-versa) is either (a) analyze-phase lift churn — a whole
        # row added/removed, its column skeleton on only ONE side of the diff — or
        # (b) a genuine coefficient box-dependence — the SAME column skeleton on
        # BOTH sides with differing (coeffs, rhs). Only (b) fails templatability.
        # The +/- pair of a churned exact link lands wholly on one side, so it is
        # correctly read as churn, not a coeff change.
        inv_stats["instances"] += 1
        root_c = inv_by_box[0]
        box_dep = False
        churn = False
        for bc in inv_by_box[1:]:
            only_root = root_c - bc
            only_box = bc - root_c
            if not only_root and not only_box:
                continue
            rs = {tuple(j for j, _ in items) for (items, _r), _n in only_root.items()}
            bs = {tuple(j for j, _ in items) for (items, _r), _n in only_box.items()}
            if rs & bs:
                box_dep = True
            else:
                churn = True
        if box_dep:
            inv_stats["coeff_box_dependent"] += 1
        elif churn:
            inv_stats["churn_only"] += 1

    # ----- report ---------------------------------------------------------- #
    print(f"Instances parity-checked: {n_inst}/{len(stems)}   boxes built: {n_boxes}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print()
    total_cmp = sum(s["cmp"] for s in agg.values())
    if total_cmp == 0:
        print("FATAL: 0 coefficient comparisons executed", file=sys.stderr)
        return 2
    print(
        f"{'box-parametric family':<28}{'groups':>9}{'coef cmps':>11}{'cnt-mism':>10}{'max ulp':>9}"
    )
    print("-" * 67)
    overall_max = 0
    overall_mism = 0
    for fam, s in sorted(agg.items()):
        print(f"{fam:<28}{s['groups']:>9}{s['cmp']:>11}{s['count_mismatch']:>10}{s['max_ulp']:>9}")
        overall_max = max(overall_max, s["max_ulp"])
        overall_mism += s["count_mismatch"]
    print("-" * 67)
    print(f"{'TOTAL':<28}{'':>9}{total_cmp:>11}{overall_mism:>10}{overall_max:>9}")
    print(
        f"\nBox-invariant families over {inv_stats['instances']} instances: "
        f"coeff box-dependent = {inv_stats['coeff_box_dependent']}, "
        f"lift-churn only (coeffs invariant) = {inv_stats['churn_only']}"
    )

    ok = overall_max <= 1 and overall_mism == 0 and inv_stats["coeff_box_dependent"] == 0
    print("\nParity bar: max ulp <= 1, no count mismatch, no box-dependent invariant coeff.")
    box_dep = inv_stats["coeff_box_dependent"]
    print(
        f"Result: {'PASS' if ok else 'FAIL'} (max ulp={overall_max}, "
        f"count-mismatch={overall_mism}, invariant-coeff-box-dep={box_dep})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
