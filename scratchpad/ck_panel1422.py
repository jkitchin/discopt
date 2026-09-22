"""#1422 graduation panel: DISCOPT_CONVEX_KERNEL_KEEP_BOUND ON vs OFF.

CLAUDE.md §5 bound-changing gate. Both arms run INTERLEAVED per instance over the
39 convex-kernel-eligible MINLPLib-snapshot instances (selected by the kernel's own
``build_convex_spec``, never by name), each at the same budget.

Two bars:
  (1) cert-clean -- no bound above its reference optimum, no `gap_certified=True`
      regressing to uncertified, no status change, objective drift within tol.
  (2) net-positive -- measurably helpful: bounds gained/tightened, broadly.

§6: prints an executed-comparison count and exits non-zero when it is zero.
§7: nothing is swallowed -- a solve that raises kills the run.
§8: asserts the loaded module path AND a marker unique to this change.
"""
import json, os, pathlib, sys, time

import discopt.modeling.core as core
import discopt.solvers._convex_kernel as ck
from discopt.modeling.core import from_nl, objective_sense_sign

print("core   __file__:", core.__file__, flush=True)
print("kernel __file__:", ck.__file__, flush=True)
assert hasattr(ck, "keep_declined_bound_enabled"), "MARKER ABSENT -- wrong tree loaded"
assert hasattr(ck, "last_declined_bound"), "MARKER ABSENT -- wrong tree loaded"

BASE = pathlib.Path.home() / "Dropbox/projects/discopt-minlp-benchmark"
NL = BASE / "minlplib/nl"
TL = float(os.environ.get("TL", "8"))

# --- oracle -------------------------------------------------------------- #
SOLU = {}
for line in (BASE / "minlplib.solu").read_text().splitlines():
    f = line.split()
    if len(f) >= 3 and f[0] in ("=opt=", "=best="):
        SOLU[f[1]] = (f[0], float(f[2]))

names = json.load(open(sys.argv[1]))["eligible"]
assert len(names) > 10, f"eligible set too small: {len(names)}"
print(f"{len(names)} eligible instances, budget={TL}s, oracle entries={len(SOLU)}\n", flush=True)


def run(nm, on):
    os.environ["DISCOPT_CONVEX_KERNEL_KEEP_BOUND"] = "1" if on else "0"
    m = from_nl(str(NL / f"{nm}.nl"))
    s = objective_sense_sign(m)
    t0 = time.perf_counter()
    r = m.solve(time_limit=TL)
    return {
        "sign": s,
        "status": r.status,
        "objective": None if r.objective is None else float(r.objective),
        "bound": None if r.bound is None else float(r.bound),
        "bound_valid": bool(r.bound_valid),
        "bound_source": r.bound_source,
        "gap_certified": bool(r.gap_certified),
        "wall": time.perf_counter() - t0,
    }


rows, n = [], 0
for i, nm in enumerate(names, 1):
    # OFF first, then ON, per instance -- interleaved, so any drift in machine load
    # lands on both arms rather than on whichever ran later (§9).
    off = run(nm, False)
    on = run(nm, True)
    n += 1
    rows.append({"name": nm, "off": off, "on": on, "solu": SOLU.get(nm)})
    print(
        f"[{i}/{len(names)}] {nm:26s} OFF b={off['bound']} ({off['status']})  "
        f"ON b={on['bound']} ({on['status']})  solu={SOLU.get(nm)}",
        flush=True,
    )

pathlib.Path(sys.argv[2]).write_text(json.dumps(rows, indent=1))

# --- bar 1: cert-clean ---------------------------------------------------- #
TOL = 1e-6
cmp_count = 0
viol_oracle, viol_cert, viol_status, viol_obj, viol_looser = [], [], [], [], []
for r in rows:
    on, off, s = r["on"], r["off"], r["on"]["sign"]
    if on["status"] != off["status"]:
        viol_status.append((r["name"], off["status"], on["status"]))
    if off["gap_certified"] and not on["gap_certified"]:
        viol_cert.append(r["name"])
    if off["objective"] is not None and on["objective"] is not None:
        cmp_count += 1
        d = abs(on["objective"] - off["objective"])
        if d > 1e-6 * max(1.0, abs(off["objective"])):
            viol_obj.append((r["name"], off["objective"], on["objective"]))
    if on["bound"] is not None and r["solu"] is not None:
        cmp_count += 1
        opt = r["solu"][1]
        # s*bound is a LOWER bound in minimization space; s*opt the true optimum.
        if s * on["bound"] > s * opt + 1e-6 * max(1.0, abs(opt)):
            viol_oracle.append((r["name"], on["bound"], opt, r["solu"][0]))
    if on["bound"] is not None and off["bound"] is not None:
        cmp_count += 1
        if s * on["bound"] < s * off["bound"] - 1e-9:
            viol_looser.append((r["name"], off["bound"], on["bound"]))

# --- bar 2: net-positive -------------------------------------------------- #
gained = [r["name"] for r in rows if r["off"]["bound"] is None and r["on"]["bound"] is not None]
tighter = [
    r["name"]
    for r in rows
    if r["off"]["bound"] is not None
    and r["on"]["bound"] is not None
    and r["on"]["sign"] * r["on"]["bound"] > r["on"]["sign"] * r["off"]["bound"] + 1e-9
]
lost = [r["name"] for r in rows if r["off"]["bound"] is not None and r["on"]["bound"] is None]
w_off = sum(r["off"]["wall"] for r in rows)
w_on = sum(r["on"]["wall"] for r in rows)

print(f"\n=== BAR 1: cert-clean ({cmp_count} executed comparisons) ===")
print(f"bounds ABOVE their reference optimum : {len(viol_oracle)}  {viol_oracle}")
print(f"certification regressions            : {len(viol_cert)}  {viol_cert}")
print(f"status changes                       : {len(viol_status)}  {viol_status}")
print(f"objective drift beyond tol           : {len(viol_obj)}  {viol_obj}")
print(f"bounds made LOOSER by the flag       : {len(viol_looser)}  {viol_looser}")
print("\n=== BAR 2: net-positive ===")
print(f"bound GAINED where OFF had none : {len(gained)}/{n}  {gained}")
print(f"bound TIGHTENED                 : {len(tighter)}/{n}  {tighter}")
print(f"bound LOST                      : {len(lost)}/{n}  {lost}")
print(f"total wall OFF={w_off:.1f}s  ON={w_on:.1f}s  delta={100*(w_on-w_off)/w_off:+.1f}%")

clean = not (viol_oracle or viol_cert or viol_status or viol_obj or viol_looser)
print(f"\nCERT-CLEAN={clean}  NET-POSITIVE={len(gained)+len(tighter) > 0 and not lost}")
print(f"EXECUTED_COMPARISONS={cmp_count}")
sys.exit(0 if cmp_count > 0 else 1)
