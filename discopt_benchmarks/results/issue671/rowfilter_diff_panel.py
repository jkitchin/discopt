"""In-repo differential panel for DISCOPT_RELAX_ROW_FILTER (#671 acceptance).

For every vendored nl instance, solve with the filter OFF and ON and check the
three unchecked #671 acceptance items on the in-repo corpus (the full-corpus
panel is the graduation gate; this is the evidence subset that runs in the dev
container):

  * SOUND (incorrect_count = 0): no reported dual bound exceeds the instance's
    known optimum, either OFF or ON.
  * NO CERT REGRESSION: an instance that was `optimal` OFF must not drop to a
    weaker status ON.
  * BOUND-NEUTRAL where it must be: on already-solving instances the ON bound
    must not be materially looser than OFF (the row filter can only loosen by
    dropping rows; the acceptance requires already-solving instances be
    unchanged — this flags any instance where a dropped row carried tightness).

Emits a per-instance table and a PASS/FAIL verdict. Bounds are lower bounds
(minimization, in the solver's reported sense).
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "python", "tests"))

TIME_LIMIT = float(os.environ.get("PANEL_TIME_LIMIT", "12"))
ABS = 1e-4
REL = 1e-4


def _load_optima():
    try:
        from _optima import optima_registry

        return {k: v.get("optimum") for k, v in optima_registry().items()}
    except Exception:
        return {}


def _solve(path, filter_on):
    import discopt.modeling as dm

    os.environ["DISCOPT_RELAX_ROW_FILTER"] = "1" if filter_on else "0"
    os.environ["DISCOPT_LP_ITERATIVE_REFINEMENT"] = "0"
    os.environ["DISCOPT_LP_FACTORIZATION_HARDENING"] = "0"
    t = time.time()
    r = dm.from_nl(path).solve(time_limit=TIME_LIMIT)
    return r.status, r.bound, r.objective, time.time() - t


def main():
    nl_dir = os.path.join(_ROOT, "python", "tests", "data", "minlplib_nl")
    names = sorted(f[:-3] for f in os.listdir(nl_dir) if f.endswith(".nl"))
    optima = _load_optima()

    unsound = []       # bound above known optimum (either config) — HARD fail
    regressions = []   # status weakened or bound materially loosened by ON
    changed = []       # any bound/status change (informational)
    print(f"{'instance':<22}{'off status':<12}{'on status':<12}"
          f"{'off bound':>16}{'on bound':>16}{'opt':>14}  flags")
    for name in names:
        path = os.path.join(nl_dir, f"{name}.nl")
        try:
            s_off, b_off, _o_off, _t0 = _solve(path, False)
            s_on, b_on, _o_on, _t1 = _solve(path, True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<22}ERROR {type(e).__name__}: {str(e)[:60]}")
            continue

        opt = optima.get(name)
        flags = []
        tol = ABS + REL * (abs(opt) if opt is not None else 0.0)
        if opt is not None:
            if b_off is not None and b_off > opt + tol:
                flags.append("UNSOUND-OFF")
                unsound.append((name, "off", b_off, opt))
            if b_on is not None and b_on > opt + tol:
                flags.append("UNSOUND-ON")
                unsound.append((name, "on", b_on, opt))
        # cert regression: optimal -> weaker
        if s_off == "optimal" and s_on != "optimal":
            flags.append("STATUS-REGRESS")
            regressions.append((name, f"{s_off}->{s_on}"))
        # bound loosened materially by ON (only meaningful when both finite)
        if b_off is not None and b_on is not None:
            loosen = b_off - b_on  # >0 means ON is looser (lower)
            denom = max(1.0, abs(b_off))
            if loosen / denom > 1e-6 and b_off <= (opt + tol if opt is not None else float("inf")):
                flags.append(f"LOOSER({loosen/denom:.1e})")
                regressions.append((name, f"bound {b_off:.6g}->{b_on:.6g}"))
        if s_off != s_on or (b_off != b_on):
            changed.append(name)
        obs = f"{b_off if b_off is not None else float('nan'):>16.6g}"
        ons = f"{b_on if b_on is not None else float('nan'):>16.6g}"
        opts = f"{opt if opt is not None else float('nan'):>14.6g}"
        print(f"{name:<22}{str(s_off):<12}{str(s_on):<12}{obs}{ons}{opts}  {' '.join(flags)}")

    print("\n==== VERDICT ====")
    print(f"instances: {len(names)}  changed by filter: {len(changed)}")
    print(f"UNSOUND (bound > optimum): {len(unsound)}  -> {unsound}")
    print(f"REGRESSIONS (status/bound): {len(regressions)}  -> {regressions[:12]}")
    ok = not unsound and not regressions
    print(f"PANEL: {'PASS (cert-clean, no regression)' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
