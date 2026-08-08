"""pytest plugin: select the POUNCE option arm via DISCOPT_940_ARM.

Lets the SAME test files run under POUNCE's own defaults vs the shipped
baseline, in-process, so a CI failure can be attributed rather than guessed at.
"""
import os

ARMS = {
    "pounce_defaults": {"print_level": 0, "constr_viol_tol": 1e-4, "bound_relax_factor": 1e-8},
    "cvt_only":        {"print_level": 0, "constr_viol_tol": 1e-8, "bound_relax_factor": 1e-8},
    "brf_only":        {"print_level": 0, "constr_viol_tol": 1e-4, "bound_relax_factor": 0.0},
    "shipped":         None,   # leave the real defaults in place
}


def pytest_configure(config):
    arm = os.environ.get("DISCOPT_940_ARM", "shipped")
    if arm not in ARMS:
        raise SystemExit(f"unknown DISCOPT_940_ARM={arm!r}; expected {sorted(ARMS)}")
    override = ARMS[arm]
    print(f"\n[#940 arm] {arm} -> {override if override else 'module defaults'}")
    if override is None:
        return
    import discopt.solvers as S
    import discopt.solvers.lp_pounce as LPP
    import discopt.solvers.qp_pounce as QPP

    fn = lambda: dict(override)  # noqa: E731
    S.pounce_option_defaults = fn
    LPP.pounce_option_defaults = fn
    QPP.pounce_option_defaults = fn
