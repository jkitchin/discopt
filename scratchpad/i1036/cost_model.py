"""Re-measure the module docstring's cost-model claim under the new design rule."""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("JAX_ENABLE_X64","1")
sys.path.insert(0, "python/tests")
import discopt.solvers.surrogate as S
from support import direct_testfuncs as tfs
print("loaded:", S.__file__)
tf = tfs.get("branin")
n_design = S._default_design_size(tf.n)
marks = []
m, _ = tfs.build_model(tf)
t0 = time.perf_counter()
S.solve_surrogate(m, max_evals=30, time_limit=3600.0, seed=0,
                  on_evaluation=lambda k, v: marks.append((k, time.perf_counter() - t0)))
assert marks, "on_evaluation never fired"
design_wall = next(t for k, t in marks if k == n_design)
total = marks[-1][1]
after = [b - a for (_, a), (_, b) in zip(marks[n_design - 1:], marks[n_design:])]
print(f"design size {n_design}; design wall {design_wall:.2f}s; total {total:.1f}s; "
      f"{len(after)} post-design evaluations, mean {sum(after)/len(after):.1f}s each")
print(f"executed marks: {len(marks)}")
sys.exit(0 if marks else 1)
