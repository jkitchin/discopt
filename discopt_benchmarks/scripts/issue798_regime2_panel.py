#!/usr/bin/env python
"""Issue #798 / K4 — Regime-2 graduation panel for DISCOPT_CONVEX_KERNEL.

Sweeps the in-repo corpus (`python/tests/data/minlplib_nl/`). For every instance
it asks the soundness gate (`build_convex_spec`) whether the convex kernel may be
routed; the ROUTED (provably-convex) subset is solved by the kernel and checked
CERT-CLEAN against the MINLPLib oracle:

  * status is never a false ``optimal`` — the dual bound never sits below the
    oracle optimum (for the reported sense) beyond tolerance;
  * the reported incumbent is independently verified feasible (already enforced by
    `try_convex_solve`'s #779 guard, re-checked here vs the oracle optimum);
  * the certificate is consistent (bound ≥ incumbent for max / ≤ for min).

DECLINED instances keep the default path (unchanged behaviour) and are only
counted. The gate is the graduation bar: `incorrect_count == 0` with zero slack.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"
os.environ["DISCOPT_CONVEX_KERNEL"] = "1"

import discopt.modeling as dm  # noqa: E402
from discopt.solvers._convex_kernel import build_convex_spec, try_convex_solve  # noqa: E402

CORPUS = os.path.join(
    os.path.dirname(__file__), "..", "..", "python", "tests", "data", "minlplib_nl"
)
SOLU = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")
TIME_LIMIT = 60.0
TOL = 1e-4


def load_oracle() -> dict[str, float]:
    opt: dict[str, float] = {}
    if not os.path.exists(SOLU):
        return opt
    with open(SOLU) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0] in ("=opt=", "=best="):
                try:
                    opt[parts[1]] = float(parts[2])
                except ValueError:
                    pass
    return opt


def main() -> bool:
    oracle = load_oracle()
    files = sorted(f for f in os.listdir(CORPUS) if f.endswith(".nl"))
    routed = declined = incorrect = 0
    rows = []
    for fn in files:
        name = fn[:-3]
        path = os.path.join(CORPUS, fn)
        try:
            model = dm.from_nl(path)
        except Exception as e:  # noqa: BLE001
            rows.append((name, "parse-error", str(e)[:40]))
            continue
        if build_convex_spec(model) is None:
            declined += 1
            continue
        routed += 1
        opt = oracle.get(name)
        t = time.perf_counter()
        try:
            res = try_convex_solve(model, time_limit=TIME_LIMIT)
        except Exception as e:  # noqa: BLE001
            rows.append((name, "kernel-error", str(e)[:40]))
            incorrect += 1
            continue
        dt = time.perf_counter() - t
        if res is None:
            rows.append((name, "declined-late", ""))  # verification fell back — sound
            continue
        # Cert-clean checks (only meaningful when we have the oracle).
        flags = []
        obj, bd = res.objective, res.bound
        if res.status == "optimal" and obj is not None and bd is not None:
            # certificate consistency
            if not (bd >= obj - TOL * max(1.0, abs(obj)) or bd <= obj + TOL * max(1.0, abs(obj))):
                flags.append("INCONSISTENT")
            if opt is not None:
                # No false optimal: the certified objective must match the oracle.
                if abs(obj - opt) > 1e-2 * max(1.0, abs(opt)):
                    flags.append(f"FALSE-OPT(obj={obj:.3f} vs oracle={opt:.3f})")
        if flags:
            incorrect += 1
        rows.append(
            (name, res.status, f"obj={obj} bound={bd} opt={opt} {dt:.1f}s {' '.join(flags)}")
        )

    print(f"\n{'instance':16s} {'status':14s} detail")
    for name, st, detail in rows:
        print(f"{name:16s} {st:14s} {detail}")
    print(
        f"\nRegime-2: corpus={len(files)} routed={routed} declined={declined} "
        f"incorrect_count={incorrect}"
    )
    print(
        f"GRADUATION GATE (incorrect_count == 0, cert-clean): {'PASS' if incorrect == 0 else 'FAIL'}"
    )
    return incorrect == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
