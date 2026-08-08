"""pytest plugin: select the #940 change dimensions via DISCOPT_940_ARM.

Three independent dimensions shipped in #943, so an attribution must be able to
switch each off separately:
  * constr_viol_tol / bound_relax_factor  (pounce_option_defaults)
  * ray certification of UNBOUNDED        (_settle_ambiguous_unbounded / _certify_unbounded_ray)
"""
import os

POUNCE_DEFAULTS = {"print_level": 0, "constr_viol_tol": 1e-4, "bound_relax_factor": 1e-8}

ARMS = {
    # everything off -> pre-#943 behaviour
    "baseline":      {"opts": POUNCE_DEFAULTS, "ray": False},
    # option dimensions only
    "opts_only":     {"opts": None,            "ray": False},
    # ray certification only
    "ray_only":      {"opts": POUNCE_DEFAULTS, "ray": True},
    # what ships
    "shipped":       {"opts": None,            "ray": True},
    # option dimensions split, ray left ON (measured innocent)
    "cvt_only":      {"opts": {"print_level": 0, "constr_viol_tol": 1e-8,
                               "bound_relax_factor": 1e-8}, "ray": True},
    "brf_only":      {"opts": {"print_level": 0, "constr_viol_tol": 1e-4,
                               "bound_relax_factor": 0.0}, "ray": True},
}


def pytest_configure(config):
    arm = os.environ.get("DISCOPT_940_ARM", "shipped")
    if arm not in ARMS:
        raise SystemExit(f"unknown DISCOPT_940_ARM={arm!r}; expected {sorted(ARMS)}")
    spec = ARMS[arm]
    print(f"\n[#940 arm] {arm} -> opts={'module' if spec['opts'] is None else 'pounce'} "
          f"ray_certification={spec['ray']}")

    import discopt.solvers as S
    import discopt.solvers.lp_pounce as LPP
    import discopt.solvers.qp_pounce as QPP

    if spec["opts"] is not None:
        fn = lambda: dict(spec["opts"])  # noqa: E731
        S.pounce_option_defaults = fn
        LPP.pounce_option_defaults = fn
        QPP.pounce_option_defaults = fn

    if not spec["ray"]:
        # restore the pre-#943 inference: keep whatever UNBOUNDED the status map gave
        LPP._settle_ambiguous_unbounded = lambda result, c, A, cl, cu, lb, ub, opts: result
        QPP._certify_unbounded_ray = lambda *a, **k: True
