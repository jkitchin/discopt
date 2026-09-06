"""Is clay0303hfsg self-consistent within ONE arm under deterministic=True?

If the OLD arm alone returns two different objectives across identical repeats,
the divergence seen in the A/B is a pre-existing property of the instance, not
an effect of the #1180 marshaling change. Prints an executed-run count.
"""
import sys, json
sys.path.insert(0, "discopt_benchmarks/scripts")
import issue1180_callback_ab as AB

arm = sys.argv[1]
reps = int(sys.argv[2])
arms = AB.Arms()
arms.install(arm)
arms.verify(arm)
nl = "python/tests/data/minlplib_nl/clay0303hfsg.nl"
AB.solve_once(nl, 120.0, True, 20)  # discarded warm-up
seen = []
for i in range(reps):
    r = AB.solve_once(nl, 120.0, True, 20)
    seen.append((r["nodes"], r["objective"], r["bound"]))
    print(f"{arm} rep{i}: {r['wall_s']:.2f}s nodes={r['nodes']} obj={r['objective']!r} "
          f"bound={r['bound']!r}", flush=True)
print(f"\narm={arm} runs={len(seen)} distinct outcomes={len(set(seen))}")
assert len(seen) == reps, "run loop did not execute"
