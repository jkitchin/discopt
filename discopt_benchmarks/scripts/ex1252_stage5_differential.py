"""#732 Stage-5 graduation differential: reform-firing subset, ON vs OFF at 60s.

Re-runs the 9-instance ON-vs-OFF differential the Stage-5 panel used, after the
blocker-a (nvs09 reform-build blowup guard) and blocker-b (budget-aware adoption)
fixes. Judged WITHIN this container (which is slower than the committed baseline),
per the plan: compare ON vs OFF arms of the same run, not against committed walls.

ON arm = the full flag stack: integer-multilinear reform + coupling RLT +
disjunctive config bound + narrow-box branch (rider). OFF arm = all unset.

Gates:
  * cert-clean (hard): no dual bound above its reference optimum; ON never loses a
    certificate OFF held; bound <= incumbent internally.
  * net-positive: ON dual/incumbent broadly >= OFF, no regressions.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "python"))

import discopt.modeling as dm  # noqa: E402

_DATA = _REPO / "python" / "tests" / "data"
_FLAGS = [
    "DISCOPT_INTEGER_MULTILINEAR_REFORM",
    "DISCOPT_MULTILINEAR_COUPLING_RLT",
    "DISCOPT_DISJUNCTIVE_CONFIG_BOUND",
    "DISCOPT_NARROW_BOX_BRANCH",
]

# (name, subdir, reference optimum or None if oracle not in-repo)
_INSTANCES = [
    ("nvs01", "minlplib_nl", 12.46966882156821),
    ("nvs05", "minlplib_nl", 5.470934108225147),
    ("nvs09", "minlplib_nl", -43.134336918035316),
    ("nvs16", "minlplib", 0.703125),
    ("nvs22", "minlplib", 6.05822),
    ("st_e36", "minlplib_nl", -246.00000000000003),
    ("st_e40", "minlplib", None),
    ("ex1252", "minlplib", 128893.741),
    ("ex1252a", "minlplib", None),
]

_TL = 60
_ABS = 1e-4  # absolute soundness slack


def _solve(path: str, arm: str):
    for k in _FLAGS:
        os.environ.pop(k, None)
    if arm == "ON":
        for k in _FLAGS:
            os.environ[k] = "1"
    try:
        m = dm.from_nl(path)
        t0 = time.perf_counter()
        r = m.solve(time_limit=_TL)
        dt = time.perf_counter() - t0
    finally:
        for k in _FLAGS:
            os.environ.pop(k, None)
    return {
        "wall": dt,
        "status": str(getattr(r, "status", None)),
        "obj": getattr(r, "objective", None),
        "bound": getattr(r, "bound", None),
        "nodes": getattr(r, "node_count", None),
        "cert": getattr(r, "gap_certified", None),
    }


def _fmt(x, w=10, p=2):
    if x is None:
        return " " * (w - 4) + "None"
    return f"{x:{w}.{p}f}"


def main() -> int:
    print(f"#732 Stage-5 differential (ON vs OFF, {_TL}s, this container)\n")
    hdr = (
        f"{'instance':10s} {'arm':3s} {'status':12s} {'dual':>11s} "
        f"{'incumbent':>13s} {'nodes':>6s} {'cert':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    violations: list[str] = []
    net: list[str] = []
    for name, sub, opt in _INSTANCES:
        path = str(_DATA / sub / f"{name}.nl")
        if not Path(path).exists():
            print(f"{name:10s} MISSING")
            continue
        off = _solve(path, "OFF")
        on = _solve(path, "ON")
        for arm, res in (("OFF", off), ("ON", on)):
            print(
                f"{name:10s} {arm:3s} {res['status'][:12]:12s} "
                f"{_fmt(res['bound'], 11, 2)} {_fmt(res['obj'], 13, 4)} "
                f"{str(res['nodes']):>6s} {str(res['cert']):>5s}"
            )
        # --- soundness (hard) ---
        for arm, res in (("OFF", off), ("ON", on)):
            b = res["bound"]
            if b is None:
                continue
            # minimize: dual bound must not exceed the optimum...
            if opt is not None and b > opt + _ABS + 1e-4 * abs(opt):
                violations.append(f"{name} {arm}: dual {b:.4f} > optimum {opt:.4f}")
            # ...nor its own incumbent.
            o = res["obj"]
            if o is not None and b > o + _ABS + 1e-4 * abs(o):
                violations.append(f"{name} {arm}: dual {b:.4f} > incumbent {o:.4f}")
        # --- cert regression (hard) ---
        if off.get("cert") and not on.get("cert"):
            violations.append(f"{name}: ON lost the certificate OFF held")
        # --- net-positive signal ---
        ob, nb = off["bound"], on["bound"]
        if ob is not None and nb is not None:
            if nb > ob + 1e-6:
                net.append(f"{name}: dual {ob:.3g} -> {nb:.3g} (ON better)")
            elif nb < ob - 1e-6:
                net.append(f"{name}: dual {ob:.3g} -> {nb:.3g} (ON WORSE)")
        oo, no = off["obj"], on["obj"]
        if oo is not None and no is not None and no < oo - 1e-6:
            net.append(f"{name}: incumbent {oo:.6g} -> {no:.6g} (ON better)")
        print("-" * len(hdr))

    print("\n=== SOUNDNESS (hard gate) ===")
    if violations:
        for v in violations:
            print("  VIOLATION:", v)
    else:
        print("  clean: no dual above optimum/incumbent, no certificate regression")
    print("\n=== NET-POSITIVE signal ===")
    if net:
        for n in net:
            print("  ", n)
    else:
        print("  (no ON/OFF dual or incumbent differences)")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
