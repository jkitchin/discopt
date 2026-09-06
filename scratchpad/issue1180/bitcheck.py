"""#1180 build: the tape evaluator must be BIT-identical after the marshaling change.

Compares every tape entry point against a reference that reproduces the OLD
list-building code path, on 5 points per instance across the corpus. Anything
other than exact equality (nan-aware) fails; the executed-comparison count is
printed and a zero count exits non-zero.
"""
import os, sys, numpy as np

ASSERTS = {"n": 0}

def eq(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True))

def old_x(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]

def main():
    from discopt.modeling.core import from_nl
    from discopt._tape_nlp_evaluator import try_build
    nl_dir = "python/tests/data/minlplib_nl"
    names = sorted(f[:-3] for f in os.listdir(nl_dir) if f.endswith(".nl"))
    bad = []
    n_inst = 0
    for name in names:
        m = from_nl(os.path.join(nl_dir, f"{name}.nl"))
        ev = try_build(m)
        if ev is None:
            continue
        n_inst += 1
        p = ev._problem
        lb, ub = ev.variable_bounds
        lo = np.clip(lb, -100.0, 100.0); hi = np.clip(ub, -100.0, 100.0)
        rng = np.random.RandomState(7)
        for k in range(5):
            x = lo + (hi - lo) * (0.5 if k == 0 else rng.uniform(size=lo.shape[0]))
            xo, xn = old_x(x), ev._x(x)
            lam = rng.uniform(size=max(ev.n_constraints, 0))
            checks = [
                ("objective", p.objective(xo), p.objective(xn)),
                ("gradient", p.gradient(xo), p.gradient(xn)),
            ]
            if ev.n_constraints:
                checks += [
                    ("constraints", p.constraints(xo), p.constraints(xn)),
                    ("jacobian", p.jacobian(xo), p.jacobian(xn)),
                ]
            checks.append((
                "hessian",
                p.hessian(xo, lam=[float(v) for v in lam], obj_factor=1.0),
                p.hessian(xn, lam=np.asarray(lam, dtype=np.float64), obj_factor=1.0),
            ))
            for label, a, b in checks:
                ASSERTS["n"] += 1
                if not eq(a, b):
                    bad.append((name, k, label))
        print(f"  {name}: ok ({ASSERTS['n']} comparisons so far)", flush=True)
    print(f"\ninstances with a tape: {n_inst}")
    print(f"executed comparisons : {ASSERTS['n']}")
    print(f"mismatches           : {len(bad)}")
    for b in bad[:20]:
        print("   MISMATCH", b)
    if ASSERTS["n"] == 0:
        print("PROBE COMPARED NOTHING", file=sys.stderr)
        return 1
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
