import functools, sys, json, pounce
from discopt._tape_nlp_evaluator import TapeNLPEvaluator
use_cb = sys.argv[1] == "1"
counts = {"s": 0, "cb": 0, "objs": []}
ops = pounce.Problem.solve
def ws(self, *a, **k):
    counts["s"] += 1
    t = type(self.problem_obj).__name__
    if t not in counts["objs"]:
        counts["objs"].append(t)
    return ops(self, *a, **k)
pounce.Problem.solve = ws
for name in ("evaluate_objective", "evaluate_gradient", "evaluate_constraints",
             "evaluate_jacobian_values", "evaluate_hessian_values"):
    f = getattr(TapeNLPEvaluator, name)
    def mk(f):
        @functools.wraps(f)
        def w(self, *a, **k):
            counts["cb"] += 1
            return f(self, *a, **k)
        return w
    setattr(TapeNLPEvaluator, name, mk(f))
from discopt.modeling.core import from_nl
m = from_nl(sys.argv[2])
kw = dict(time_limit=20.0, gap_tolerance=1e-4)
if use_cb:
    kw["node_callback"] = lambda a, b: None
r = m.solve(**kw)
print("RESULT " + json.dumps({"node_callback": use_cb, "nodes": int(r.node_count),
                              "objective": None if r.objective is None else float(r.objective),
                              "bound": None if r.bound is None else float(r.bound),
                              "status": str(r.status), **counts}))
