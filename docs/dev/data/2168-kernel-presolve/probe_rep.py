"""#2168 probe: representative tie-break. Runs kernel_arm.py's ``pre`` arm with the
tie-break in ``_Problem._elim_cost`` flipped (eliminate the LOWER index, keep the
higher one, as SCIP's presolve does on cstr), by patching the cost before the run.

  DISCOPT_PROFILE=1 PYTHONHASHSEED=0 python -u probe_rep.py MAX_NODES"""
import runpy, sys
from discopt._relax import spatial_presolve as SP

_orig = SP._Problem._elim_cost
n_calls = [0]


def _flipped(self, p, q, a, b):
    c = _orig(self, p, q, a, b)
    if c is None:
        return None
    n_calls[0] += 1
    return c[:-1] + (p,)


SP._Problem._elim_cost = _flipped
sys.argv = ["kernel_arm.py", "pre", sys.argv[1], "seeded"]
try:
    runpy.run_path("kernel_arm.py", run_name="__main__")
finally:
    print(f"patched cost calls: {n_calls[0]}")
    assert n_calls[0] > 0
