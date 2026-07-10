"""Exp A: does the Gauss-Newton detector fire on DAEBuilder.least_squares + L2 reg?"""

import time

from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.solvers.nlp_pounce import solve_nlp
from hybrid_common import build_nn_model

for gn in (False, True):
    m, dae, x0, _ = build_nn_model()
    t0 = time.perf_counter()
    ev = NLPEvaluator(m, gauss_newton=gn)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = solve_nlp(ev, x0, options={"max_iter": 3000, "tol": 1e-8})
    t_solve = time.perf_counter() - t0
    print(
        f"gauss_newton={gn!s:5}  detector_fired={ev.is_gauss_newton!s:5}  "
        f"build={t_build:6.2f}s  solve={t_solve:6.2f}s  iters={res.iterations:3d}  "
        f"status={res.status.name}  obj={res.objective:.6e}"
    )
