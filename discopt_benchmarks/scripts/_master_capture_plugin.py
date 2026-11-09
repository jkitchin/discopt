"""pytest plugin for ``master_parity_panel.py capture-decomp`` (highs-parity P0.4).

Wraps :func:`discopt.solvers.lp_backend.get_decomposition_master_solver` so every master
MILP a decomposition solve builds is pickled (its ``solve_milp`` keyword arguments) into
``$MASTER_CAPTURE_DIR`` together with the engine's answer. Every caller imports the selector
inside the function, so patching the module attribute reaches all of them. The wrapped
solver's result is returned unchanged, but pickling each master costs wall time, so a
solve with a wall-clock budget runs fewer master iterations under capture. Measured
(P0.4, 3042 masters): ``test_1611_lagrangian.py::
test_requested_lagrangian_is_not_replaced_by_bnb_fallback`` passes alone and fails under
capture with a weaker, still valid, bound (323.67 vs the asserted 331.6 +/- 1; ``<= Z_IP``
holds). The captured masters are real masters either way; the per-test outcomes of a
capture run are not a measurement of the solver (``decomp-e2e`` runs without it).
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
