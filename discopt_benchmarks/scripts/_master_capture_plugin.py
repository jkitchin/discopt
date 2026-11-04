"""pytest plugin for ``master_parity_panel.py capture-decomp`` (highs-parity P0.4).

Wraps :func:`discopt.solvers.lp_backend.get_decomposition_master_solver` so every master
MILP a decomposition solve builds is pickled (its ``solve_milp`` keyword arguments) into
``$MASTER_CAPTURE_DIR`` together with the engine's answer. Every caller imports the selector
inside the function, so patching the module attribute reaches all of them. The wrapped
solver's result is returned unchanged: capture never alters a solve.
"""

from __future__ import annotations

import os
import pickle
import time

_COUNT = [0]


def pytest_configure(config):
    out = os.environ["MASTER_CAPTURE_DIR"]  # KeyError on purpose: no silent no-op
    import discopt.solvers.lp_backend as lpb

    real = lpb.get_decomposition_master_solver

    def get_decomposition_master_solver():
        solve, name = real()

        def wrapped(*args, **kw):
            # Callers pass ``c`` positionally; record it under its keyword so a replay
            # can call either engine with keywords only.
            if len(args) > 1:
                raise TypeError(f"capture expects at most c positionally, got {len(args)} args")
            t = time.perf_counter()
            r = solve(*args, **kw)
            if args:
                kw = {"c": args[0], **kw}
            rec = {
                "kw": kw,
                "engine": name,
                "wall": time.perf_counter() - t,
                "status": str(r.status),
                "objective": r.objective,
                "bound": r.bound,
                "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
            }
            _COUNT[0] += 1
            with open(os.path.join(out, f"decomp__{os.getpid()}_{_COUNT[0]:05d}.pkl"), "wb") as f:
                pickle.dump(rec, f)
            return r

        return wrapped, name

    lpb.get_decomposition_master_solver = get_decomposition_master_solver
