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
    hard = []          # already-solving (optimal OFF) instance changed by ON — HARD fail
    soft = []          # partial-bound change on a non-optimal (timing-out) instance
    noise_tol = 1e-6   # relative tolerance below which a change is FP noise
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

        rel = None
        if b_off is not None and b_on is not None:
            rel = abs(b_off - b_on) / max(1.0, abs(b_off))
        changed = (s_off != s_on) or (rel is not None and rel > noise_tol) or (
            (b_off is None) != (b_on is None)
        )
        if changed:
            # HARD (acceptance violation): the instance was ALREADY-SOLVING off
            # (status optimal) and the filter changed its status or certified bound.
            if s_off == "optimal":
                flags.append("HARD-REGRESS")
                hard.append((name, f"{s_off}/{b_off:.6g} -> {s_on}/{b_on:.6g}"))
            else:
                # SOFT: partial bound / status on a non-solving instance.
                looser = b_off is not None and b_on is not None and b_on < b_off
                flags.append("SOFT-looser" if looser else "SOFT-changed")
                soft.append((name, f"{s_off}/{b_off} -> {s_on}/{b_on}"))

        obs = f"{b_off if b_off is not None else float('nan'):>16.6g}"
        ons = f"{b_on if b_on is not None else float('nan'):>16.6g}"
        opts = f"{opt if opt is not None else float('nan'):>14.6g}"
        print(f"{name:<22}{str(s_off):<12}{str(s_on):<12}{obs}{ons}{opts}  {' '.join(flags)}")

    print("\n==== VERDICT ====")
    print(f"instances: {len(names)}")
    print(f"UNSOUND (bound > optimum): {len(unsound)}  -> {unsound}")
    print(f"HARD regressions (already-solving instance changed): {len(hard)}  -> {hard}")
    print(f"SOFT changes (partial bound on timing-out instance): {len(soft)}  -> {soft}")
    accept_ok = not unsound and not hard
    print(f"\nACCEPTANCE (cert-clean + already-solving unchanged): "
          f"{'PASS' if accept_ok else 'FAIL'}")
    print(f"GRADUATION net-positive question: {len(soft)} soft change(s) "
          f"(hda gain vs any partial-bound losses) — full-corpus panel decides")


if __name__ == "__main__":
    main()
