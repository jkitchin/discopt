"""#1153: incumbent-monotonicity panel over the in-repo MINLPLib corpus.

For each instance, solve at an increasing ladder of ``time_limit`` values and
report every step where the incumbent got WORSE (or was lost) as the budget grew.

Prints per-instance progress (CLAUDE.md §10) and an executed-comparison count
(CLAUDE.md §6); exits non-zero if zero comparisons ran.
"""
import glob, json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt
from discopt.modeling.core import from_nl

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
MARKER = os.environ.get("I1153_MARKER")
if MARKER:
    import discopt.solver as _s
    has = MARKER in open(_s.__file__).read()
    print(f"# marker {MARKER!r} present: {has}", flush=True)
    if os.environ.get("I1153_MARKER_EXPECT", "1") == "1":
        assert has, "marker absent: wrong tree loaded"
    else:
        assert not has, "marker present on the baseline arm"

SENSE_MIN = True


def solve_one(path, tl):
    m = from_nl(path)
    t = time.perf_counter()
    try:
        r = m.solve(time_limit=tl, gap_tolerance=1e-4)
    except Exception as exc:  # a crash is data, not something to swallow silently
        return dict(tl=tl, error=repr(exc), wall=time.perf_counter() - t)
    return dict(
        tl=tl,
        obj=None if r.objective is None else float(r.objective),
        bound=None if r.bound is None else float(r.bound),
        status=str(r.status),
        nodes=int(getattr(r, "node_count", 0) or 0),
        cert=bool(getattr(r, "gap_certified", False)),
        wall=time.perf_counter() - t,
    )


def main(argv):
    corpus = argv[1]
    ladder = [float(x) for x in argv[2].split(",")]
    out = argv[3]
    only = set(argv[4].split(",")) if len(argv) > 4 and argv[4] else None
    files = sorted(glob.glob(os.path.join(corpus, "*.nl")))
    if only:
        files = [f for f in files if os.path.basename(f)[:-3] in only]
    results, n_cmp, n_viol = {}, 0, 0
    for f in files:
        name = os.path.basename(f)[:-3]
        rows = []
        for tl in ladder:
            row = solve_one(f, tl)
            rows.append(row)
            print(f"{name:26s} tl={tl:6.1f} obj={row.get('obj')!r} "
                  f"bound={row.get('bound')!r} nodes={row.get('nodes')} status={row.get('status')} "
                  f"wall={row['wall']:.2f} {row.get('error','')}", flush=True)
        results[name] = rows
        for a, b in zip(rows, rows[1:]):
            if "error" in a or "error" in b:
                continue
            n_cmp += 1
            ao, bo = a["obj"], b["obj"]
            worse = (ao is not None and bo is None) or (
                ao is not None and bo is not None
                and bo > ao + 1e-6 * max(1.0, abs(ao))
            )
            if worse:
                n_viol += 1
                print(f"!! VIOLATION {name}: tl={a['tl']} obj={ao} -> tl={b['tl']} obj={bo}",
                      flush=True)
        with open(out, "w") as fh:
            json.dump(results, fh, indent=1)
    print(f"# instances={len(files)} executed comparisons={n_cmp} violations={n_viol}",
          flush=True)
    return 1 if n_cmp == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
