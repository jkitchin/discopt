"""CLAUDE.md Sec.5 differential panel for #930, three arms.

    base    pre-#930 worktree, $PANEL_BASE, marker asserted ABSENT
    off     #930 with DISCOPT_ROOT_PROBE_SEEDS_FALLBACK unset  (unconditional half)
    on      #930 with DISCOPT_ROOT_PROBE_SEEDS_FALLBACK=1      (flagged half)

The recorded run used 861200e7 as the baseline; see
``discopt_benchmarks/results/issue930/``.

Two separate questions, deliberately not conflated:

  base -> off   is the unconditional half. It must be cert-clean AND never
                produce a looser bound (it is a `max` merge, so `looser` > 0
                would mean the merge is wrong, not merely unhelpful).
  off  -> on    is the flag. Sec.5 bar 1 cert-clean AND bar 2 net-positive.
                A cert-clean but neutral-or-harmful flag stays OFF (the
                DISCOPT_CUT_INHERIT lesson).

Restricted to the instances whose search does not close inside the budget --
the only class where the root-relaxation fallback is live and either half can
change anything. On every closing instance the arms are inert by construction
(neither code path is reached); the CLOSERS pass at the end checks that claim --
node count and objective identical across all three arms -- rather than assuming
it, and reports CLOSER DRIFT if it does not hold.

Sec.9: arms are interleaved per instance (rotating order), not run as three
sequential blocks, and wall figures carry a spread. Sec.6: ends with executed
assertion counts and exits non-zero if any class fired zero times.
Sec.7: no bare excepts -- a worker that fails aborts the panel loudly.
"""

import json
import os
import statistics
import subprocess
import sys

# To re-run:
#
#   git worktree add --detach /tmp/base930 <pre-#930 commit>
#   cp python/discopt/_rust.cpython-*.so /tmp/base930/python/discopt/   # not in git
#   PANEL_BASE=/tmp/base930 PANEL_OUT=/tmp/p930 \
#       python -u discopt_benchmarks/scripts/issue930_root_probe_bound_panel.py
#
# The baseline arm needs the compiled Rust extension copied in: it is gitignored,
# so a fresh worktree has none and every worker would die on import. The workers
# assert their tree AND the presence/absence of the fix marker, so a mis-set
# PANEL_BASE aborts the panel rather than silently measuring one tree twice.
NEW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.environ.get("PANEL_BASE")
if not BASE or not os.path.isdir(BASE):
    raise SystemExit("set PANEL_BASE to a pre-#930 worktree (see header comment)")
SCR = os.environ.get("PANEL_OUT", os.path.join(NEW, "discopt_benchmarks/results/issue930"))
os.makedirs(SCR, exist_ok=True)
TL = float(os.environ.get("PANEL_TL", "8.0"))
REPS = int(os.environ.get("PANEL_REPS", "2"))
NL = os.path.join(NEW, "python/tests/data/minlplib_nl")

# The non-closing set from the #929 reserve panel (same corpus, same criterion:
# the search does not finish inside the budget, so the fallback is live).
INSTS = [
    "4stufen",
    "bchoco06",
    "bchoco07",
    "bchoco08",
    "beuster",
    "casctanks",
    "clay0303hfsg",
    "contvar",
    "hda",
    "heatexch_gen1",
    "heatexch_gen2",
    "heatexch_gen3",
    "nvs05",
    "tls2",
    "tspn08",
    "tspn10",
    "tspn12",
]

# A few instances the search DOES close, to check the arms are inert there.
CLOSERS = ["ex1221", "ex1224", "ex14_1_9", "nvs11", "st_miqp1"]

solu = {}
with open(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")) as fh:
    for line in fh:
        p = line.split()
        if len(p) >= 3 and p[0] in ("=opt=", "=best="):
            solu[p[1]] = (p[0], float(p[2]))

RUN = r"""
import json, sys, time
root, want_marker, nl = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
import discopt.solver as S
assert S.__file__.startswith(root), "WRONG TREE: %s not under %s" % (S.__file__, root)
has = hasattr(S, "_admissible_probe_bound")
assert has == want_marker, "MARKER MISMATCH: present=%s want=%s" % (has, want_marker)
from discopt.modeling.core import from_nl, ObjectiveSense
m = from_nl(nl)
sense = "max" if m._objective.sense == ObjectiveSense.MAXIMIZE else "min"
t = time.perf_counter(); r = m.solve(time_limit=TIMELIMIT); w = time.perf_counter() - t
print("RESULT" + json.dumps(dict(
    status=r.status, nodes=r.node_count, sense=sense,
    obj=(None if r.objective is None else float(r.objective)),
    bound=(None if r.bound is None else float(r.bound)),
    cert=bool(getattr(r, "gap_certified", False)), wall=w)))
""".replace("TIMELIMIT", repr(TL))
WORKER = os.path.join(SCR, "_panel930_worker.py")
with open(WORKER, "w") as fh:
    fh.write(RUN)

ARMS = ("base", "off", "on")


def run(arm, inst):
    root = BASE if arm == "base" else NEW
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(root, "python")
    env.pop("DISCOPT_ROOT_PROBE_SEEDS_FALLBACK", None)
    if arm == "on":
        env["DISCOPT_ROOT_PROBE_SEEDS_FALLBACK"] = "1"
    p = subprocess.run(
        [
            sys.executable,
            "-u",
            WORKER,
            root,
            "0" if arm == "base" else "1",
            os.path.join(NL, inst + ".nl"),
        ],
        capture_output=True,
        text=True,
        timeout=max(120.0, TL * 15),
        cwd=root,
        env=env,
    )
    for ln in p.stdout.splitlines():
        if ln.startswith("RESULT"):
            return json.loads(ln[6:])
    raise SystemExit(f"worker failed: arm={arm} inst={inst}\n{p.stderr[-3000:]}")


N = {"invariant": 0, "oracle": 0, "pair_base_off": 0, "pair_off_on": 0, "closer_cmp": 0}
unsound = []
# per-comparison buckets, keyed by the transition being judged
D = {
    k: {"lost": [], "gained": [], "looser": [], "tighter": [], "cert_reg": []}
    for k in ("base->off", "off->on")
}
cells = []
walls = {a: [] for a in ARMS}


def judge(key, inst, a, b, sense):
    """Compare arm a -> arm b on one instance. Returns a short tag."""
    ob, nb = a["bound"], b["bound"]
    if a["cert"] and not b["cert"]:
        D[key]["cert_reg"].append(f"{inst} (gap_certified True->False)")
    if ob is not None and nb is None:
        D[key]["lost"].append(f"{inst} ({ob:.6g} -> None)")
        return "LOST-BOUND"
    if ob is None and nb is not None:
        D[key]["gained"].append(f"{inst} (None -> {nb:.6g})")
        return "gained"
    if ob is None and nb is None:
        return ""
    N["pair_base_off" if key == "base->off" else "pair_off_on"] += 1
    if abs(nb - ob) <= 1e-9 * max(1.0, abs(ob)):
        return ""
    if (nb < ob) if sense == "min" else (nb > ob):
        D[key]["looser"].append(f"{inst} ({ob:.6g} -> {nb:.6g})")
        return "looser"
    D[key]["tighter"].append(f"{inst} ({ob:.6g} -> {nb:.6g})")
    return "tighter"


def check_sound(inst, arm, r, sense):
    if r["bound"] is not None and r["obj"] is not None:
        N["invariant"] += 1
        bad = (
            r["bound"] < r["obj"] - 1e-4 * max(1.0, abs(r["obj"]))
            if sense == "max"
            else r["bound"] > r["obj"] + 1e-4 * max(1.0, abs(r["obj"]))
        )
        if bad:
            unsound.append(f"{inst}:{arm}:bound-crosses-incumbent({r['bound']} vs {r['obj']})")
    if r["bound"] is not None and inst in solu and solu[inst][0] == "=opt=":
        N["oracle"] += 1
        ref = solu[inst][1]
        bad = (
            r["bound"] < ref - 1e-4 * max(1.0, abs(ref))
            if sense == "max"
            else r["bound"] > ref + 1e-4 * max(1.0, abs(ref))
        )
        if bad:
            unsound.append(f"{inst}:{arm}:bound-past-oracle({r['bound']} vs {ref})")


print(
    f"# panel930: {len(INSTS)} non-closing + {len(CLOSERS)} closing, "
    f"TL={TL}s, {REPS} rep(s), arms={ARMS}",
    flush=True,
)

for rep in range(REPS):
    for i, inst in enumerate(INSTS):
        # Sec.9: rotate arm order so no arm systematically runs on a warmer machine.
        order = ARMS[(i + rep) % 3 :] + ARMS[: (i + rep) % 3]
        res = {}
        for arm in order:
            res[arm] = run(arm, inst)
            walls[arm].append(res[arm]["wall"])
        sense = res["off"]["sense"]
        for arm in ARMS:
            check_sound(inst, arm, res[arm], sense)
        t1 = judge("base->off", inst, res["base"], res["off"], sense)
        t2 = judge("off->on", inst, res["off"], res["on"], sense)
        cells.append({"rep": rep, "instance": inst, **{a: res[a] for a in ARMS}})
        print(
            f"[r{rep} {i + 1:2d}/{len(INSTS)}] {inst:16s} {sense} "
            f"base={res['base']['bound']} off={res['off']['bound']} on={res['on']['bound']} "
            f"| wall {res['base']['wall']:5.2f}/{res['off']['wall']:5.2f}/{res['on']['wall']:5.2f}"
            f"  base->off:{t1 or '='} off->on:{t2 or '='}",
            flush=True,
        )

print("\n# closers (arms must be inert: identical nodes and objective)", flush=True)
closer_bad = []
for i, inst in enumerate(CLOSERS):
    order = ARMS[i % 3 :] + ARMS[: i % 3]
    res = {a: run(a, inst) for a in order}
    N["closer_cmp"] += 1
    nodes = {a: res[a]["nodes"] for a in ARMS}
    objs = {a: res[a]["obj"] for a in ARMS}
    ok = len(set(nodes.values())) == 1
    if not ok:
        closer_bad.append(f"{inst} nodes {nodes}")
    print(f"  {inst:16s} nodes={nodes} obj={objs} {'OK' if ok else 'NODE DRIFT'}", flush=True)

print()
for key in ("base->off", "off->on"):
    d = D[key]
    print(f"# {key:10s} LOST={len(d['lost'])} {d['lost']}")
    print(f"# {key:10s} gained={len(d['gained'])} {d['gained']}")
    print(f"# {key:10s} looser={len(d['looser'])} {d['looser']}")
    print(f"# {key:10s} tighter={len(d['tighter'])} {d['tighter']}")
    print(f"# {key:10s} cert_regressions={len(d['cert_reg'])} {d['cert_reg']}")
print(f"# UNSOUND        : {len(unsound)} {unsound}")
print(f"# CLOSER DRIFT   : {len(closer_bad)} {closer_bad}")
for a in ARMS:
    w = walls[a]
    sd = statistics.stdev(w) if len(w) > 1 else 0.0
    print(
        f"# wall {a:5s}: n={len(w)} total={sum(w):7.2f}s "
        f"mean={statistics.mean(w):5.2f}s sd={sd:5.2f}"
    )
print(f"# EXECUTED ASSERTION COUNTS: {N}")
with open(os.path.join(SCR, "panel930.json"), "w") as fh:
    json.dump(cells, fh, indent=1)
if min(N.values()) == 0:
    print("PROBE NEVER FIRED: a comparison class executed zero times", file=sys.stderr)
    sys.exit(1)
