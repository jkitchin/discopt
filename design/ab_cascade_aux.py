"""#208 A/B panel: OBBT aux-cascade OFF vs ON over the vendored corpus.

The reverse-FBBT aux cascade (``obbt_tighten_root(cascade_aux=True)``) is sound by
construction but ships default-OFF because it was measured *not* net-positive on
the integer-heavy corpus. This harness drives it through the *real* solve path via
the ``DISCOPT_OBBT_CASCADE_AUX`` env gate (default ``0``) and reports the flag ON
vs OFF differential the issue's acceptance criteria ask for:

  * **net-positive?** total node_count / wall, and per-instance node deltas;
  * **cert-clean?** ON vs the trusted OFF path (CLAUDE.md §5 bound-changing
    regime) — no false-infeasible, no objective divergence on certified
    instances, no dual bound above OFF's incumbent, and cert gains/regressions.

It changes nothing on the default path (the gate defaults off). Pure measurement.

Usage::

    python design/ab_cascade_aux.py                 # full corpus, 8s budget/instance
    AB_TL=15 AB_LIMIT=20 python design/ab_cascade_aux.py
"""

from __future__ import annotations

import glob
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "python", "tests", "data", "minlplib_nl")
TIME_LIMIT = float(os.environ.get("AB_TL", "8"))
GAP = 1e-4
ATOL, RTOL = 1e-4, 1e-3


def solve_one(path: str, cascade: bool) -> dict:
    """Solve one instance with the aux cascade forced on/off (fresh model)."""
    os.environ["DISCOPT_OBBT_CASCADE_AUX"] = "1" if cascade else "0"
    m = dm.from_nl(path)
    t = time.perf_counter()
    try:
        r = m.solve(time_limit=TIME_LIMIT, gap_tolerance=GAP)
    except Exception as e:  # noqa: BLE001 — report, don't crash the panel
        return {"err": f"{type(e).__name__}: {e}"}
    return {
        "status": r.status,
        "obj": None if r.objective is None else float(r.objective),
        "bound": None if r.bound is None else float(r.bound),
        "nodes": int(r.node_count) if r.node_count is not None else None,
        "wall": time.perf_counter() - t,
        "cert": bool(getattr(r, "gap_certified", False)),
    }


def differential_bad(off: dict, on: dict) -> str | None:
    """Differential soundness of ON against the trusted OFF path (CLAUDE.md §5)."""
    if off.get("err") or on.get("err"):
        return None  # solver error reported separately, not a soundness verdict

    def tol(v: float) -> float:
        return ATOL + RTOL * abs(v)

    if on["status"] == "infeasible" and off.get("obj") is not None:
        return f"ON FALSE-INFEASIBLE (OFF obj={off['obj']:.6g})"
    if off.get("cert") and on.get("cert") and off["obj"] is not None and on["obj"] is not None:
        if abs(on["obj"] - off["obj"]) > tol(off["obj"]):
            return f"OBJ DIVERGENCE on={on['obj']:.6g} off={off['obj']:.6g}"
    if on.get("bound") is not None and off.get("obj") is not None:
        if on["bound"] > off["obj"] + tol(off["obj"]):
            return f"ON BOUND>OFF-INCUMBENT bound={on['bound']:.6g} inc={off['obj']:.6g}"
    if on.get("obj") is not None and off.get("bound") is not None:
        if on["obj"] < off["bound"] - tol(off["bound"]):
            return f"ON OBJ<OFF-BOUND obj={on['obj']:.6g} bound={off['bound']:.6g}"
    return None


def main() -> None:
    files = sorted(glob.glob(os.path.join(CORPUS, "*.nl")))
    lim = int(os.environ.get("AB_LIMIT", "0"))
    if lim:
        files = files[:lim]
    hdr = f"{'instance':24s}  {'nodes off/on':>14s} {'wall off/on':>15s}  cert  note"
    print(hdr)
    print("-" * len(hdr))
    n_changed = n_viol = n_err = n_cert_gain = n_cert_reg = 0
    tot_off = tot_on = twall_off = twall_on = 0.0
    viols: list[str] = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        off, on = solve_one(path, False), solve_one(path, True)
        note = ""
        if off.get("err") or on.get("err"):
            n_err += 1
            note = f"err off={off.get('err', '')} on={on.get('err', '')}"
        v = differential_bad(off, on)
        if v:
            n_viol += 1
            viols.append(f"{name}: {v}")
            note = (note + " !!VIOLATION").strip()
        co, cn = bool(off.get("cert")), bool(on.get("cert"))
        if co and not cn:
            n_cert_reg += 1
            note = (note + " CERT-REGRESSION").strip()
        elif cn and not co:
            n_cert_gain += 1
            note = (note + " CERT-GAIN").strip()
        no, nn = off.get("nodes"), on.get("nodes")
        wo, wn = off.get("wall"), on.get("wall")
        if no is not None and nn is not None:
            tot_off += no
            tot_on += nn
            if no != nn:
                n_changed += 1
                note = (note + " CASCADE-CHANGED").strip()
        if wo and wn:
            twall_off += wo
            twall_on += wn
        print(
            f"{name:24s}  {f'{no}/{nn}':>14s} "
            f"{(f'{wo:.2f}/{wn:.2f}' if wo and wn else '-'):>15s}  "
            f"{(str(co)[0] + str(cn)[0]):>4s}  {note}"
        )
    print("-" * len(hdr))
    print(f"instances: {len(files)}   solver errors: {n_err}   cascade changed nodes: {n_changed}")
    print(
        f"total nodes  OFF={tot_off:.0f}  ON={tot_on:.0f}  "
        f"({tot_on - tot_off:+.0f}, {100 * (tot_on - tot_off) / max(1, tot_off):+.1f}%)"
    )
    wpct = 100 * (twall_on - twall_off) / max(1e-9, twall_off)
    print(
        f"total wall   OFF={twall_off:.1f}s  ON={twall_on:.1f}s  "
        f"({twall_on - twall_off:+.1f}s, {wpct:+.1f}%)"
    )
    print(f"cert gains: {n_cert_gain}   cert regressions: {n_cert_reg}")
    print(f"DIFFERENTIAL SOUNDNESS VIOLATIONS (ON vs OFF): {n_viol}")
    for line in viols:
        print("   ", line)


if __name__ == "__main__":
    main()
