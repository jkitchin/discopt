from pyscipopt import Model, quicksum
for feastol in (1e-6, 1e-9):
    for M in (1e6, 1e8, 1e10, 1e12):
        m = Model(); m.hideOutput(); m.readProblem("python/tests/data/minlplib_nl/prolog.nl")
        xs = {v.name: v for v in m.getVars()}
        m.addCons(quicksum(xs[f"x{j}"] for j in (7, 8)) >= M / 4356)  # same scale, x7+x8 only
        m.setParam("numerics/feastol", feastol); m.setParam("limits/time", 60)
        m.optimize()
        pb = m.getPrimalbound() if m.getNSols() else None
        print(f"feastol {feastol:g} x7+x8 >= {M/4356:.4g}: {m.getStatus()} dual {m.getDualbound():.6g} "
              f"primal {pb if pb is None else f'{pb:.6g}'} | 0.0349*T = {0.0349*M/4356:.4g}", flush=True)
