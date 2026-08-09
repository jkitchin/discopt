"""Is the GDPopt failure caused by bound_relax_factor=0, or by constr_viol_tol,
or pre-existing? Interleaved arms in one process (CLAUDE.md §9).
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt.modeling as dm
import discopt.solvers as S

_orig = S.pounce_option_defaults
ARMS = {
    "POUNCE defaults":  {"print_level": 0, "constr_viol_tol": 1e-4, "bound_relax_factor": 1e-8},
    "cvt only":         {"print_level": 0, "constr_viol_tol": 1e-8, "bound_relax_factor": 1e-8},
    "brf only":         {"print_level": 0, "constr_viol_tol": 1e-4, "bound_relax_factor": 0.0},
    "SHIPPED (both)":   {"print_level": 0, "constr_viol_tol": 1e-8, "bound_relax_factor": 0.0},
}

CHECKS = 0
def run(arm):
    S.pounce_option_defaults = lambda: dict(ARMS[arm])
    # the backends import the symbol directly, so patch those bindings too
    import discopt.solvers.lp_pounce as LPP, discopt.solvers.qp_pounce as QPP
    LPP.pounce_option_defaults = S.pounce_option_defaults
    QPP.pounce_option_defaults = S.pounce_option_defaults
    m = dm.Model("loa_simple")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 3], [x >= 7]], name="choice")
    m.minimize(x)
    r = m.solve(time_limit=30, gdp_method="loa")
    return r.status, r.objective, r.bound

for rep in range(2):
    for arm in (list(ARMS) if rep % 2 == 0 else list(ARMS)[::-1]):
        st, obj, bd = run(arm)
        CHECKS += 1
        print(f"rep{rep} {arm:18s} status={st:10s} obj={obj!r} bound={bd!r}", flush=True)

S.pounce_option_defaults = _orig
print(f"CHECKS_EXECUTED={CHECKS}")
sys.exit(0 if CHECKS else 1)
