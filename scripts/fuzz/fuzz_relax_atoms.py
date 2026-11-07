"""Soundness fuzzer for the uniform relaxation on 1-D atoms over adversarial boxes.

For each (expression f, box [lo, hi], sense) build the uniform relaxation of
    min f(x)  (or max f(x))  s.t. x in [lo, hi]
solve its LP, and look for a WITNESS: a sampled x in the box with f(x) < bound
(min) or f(x) > bound (max). A witness proves the bound invalid; no tolerance
judgement is involved beyond f's own evaluation error, which we guard with mpmath.

Usage: python scripts/fuzz/fuzz_relax_atoms.py

Prints an executed-check count and exits non-zero if zero checks ran
(CLAUDE.md section 6).
"""

import math
import sys

import discopt.modeling as dm
import mpmath
import numpy as np
from discopt._relax.uniform_relax import build_uniform_relaxation

mpmath.mp.dps = 50

# (name, discopt builder, mpmath evaluator, domain predicate)
ATOMS = []


def atom(name, fdm, fmp, dom=lambda lo, hi: True):
    ATOMS.append((name, fdm, fmp, dom))


for p in (2, 3, 4, 5, 6, 7):
    atom(f"x**{p}", lambda x, p=p: x**p, lambda t, p=p: t**p)
    atom(f"(1-x)**{p}", lambda x, p=p: (1 - x) ** p, lambda t, p=p: (1 - t) ** p)
    atom(
        f"(0.1*x+0.3)**{p}",
        lambda x, p=p: (0.1 * x + 0.3) ** p,
        lambda t, p=p: (t / 10 + mpmath.mpf(3) / 10) ** p,
    )
for p in (-1, -2, -3):
    atom(f"x**{p}", lambda x, p=p: x**p, lambda t, p=p: t**p, lambda lo, hi: lo > 0 or hi < 0)
for p in (0.5, 1.5, 2.5, 0.3, 1 - 2**-17, 1 + 2**-17, 3 - 2**-20, 3 + 2**-20):
    atom(
        f"x**{p!r}",
        lambda x, p=p: x**p,
        lambda t, p=p: mpmath.power(t, mpmath.mpf(p)),
        lambda lo, hi: lo >= 0,
    )
atom("exp", dm.exp, mpmath.exp)
atom("exp(-x)", lambda x: dm.exp(-x), lambda t: mpmath.exp(-t))
atom("log", dm.log, mpmath.log, lambda lo, hi: lo > 0)
atom("log1p", dm.log1p, mpmath.log1p, lambda lo, hi: lo > -1)
atom("log2", dm.log2, lambda t: mpmath.log(t, 2), lambda lo, hi: lo > 0)
atom("log10", dm.log10, mpmath.log10, lambda lo, hi: lo > 0)
atom("sqrt", dm.sqrt, mpmath.sqrt, lambda lo, hi: lo >= 0)
atom("sin", dm.sin, mpmath.sin)
atom("cos", dm.cos, mpmath.cos)
atom(
    "tan",
    dm.tan,
    mpmath.tan,
    lambda lo, hi: hi - lo < 3 and math.floor(lo / math.pi + 0.5) == math.floor(hi / math.pi + 0.5),
)
atom("tanh", dm.tanh, mpmath.tanh)
atom("atan", dm.atan, mpmath.atan)
atom("asin", dm.asin, mpmath.asin, lambda lo, hi: -1 <= lo and hi <= 1)
atom("acos", dm.acos, mpmath.acos, lambda lo, hi: -1 <= lo and hi <= 1)
atom("sinh", dm.sinh, mpmath.sinh)
atom("cosh", dm.cosh, mpmath.cosh)
atom("asinh", dm.asinh, mpmath.asinh)
atom("acosh", dm.acosh, mpmath.acosh, lambda lo, hi: lo >= 1)
atom("atanh", dm.atanh, mpmath.atanh, lambda lo, hi: -1 < lo and hi < 1)
atom("sigmoid", dm.sigmoid, lambda t: 1 / (1 + mpmath.exp(-t)))
atom("softplus", dm.softplus, lambda t: mpmath.log1p(mpmath.exp(t)))
atom("erf", dm.erf, mpmath.erf)
atom("abs", dm.abs, abs)
atom(
    "xlogx",
    dm.xlogx,
    lambda t: t * mpmath.log(t) if t > 0 else mpmath.mpf(0),
    lambda lo, hi: lo >= 0,
)
atom("x*exp(x)", lambda x: x * dm.exp(x), lambda t: t * mpmath.exp(t))
atom("x/(1+x)", lambda x: x / (1 + x), lambda t: t / (1 + t), lambda lo, hi: lo > -1)
atom("exp(-x**2)", lambda x: dm.exp(-(x**2)), lambda t: mpmath.exp(-(t**2)))
atom("log(1+x**2)", lambda x: dm.log(1 + x**2), lambda t: mpmath.log(1 + t**2))
atom("sin(x)**2", lambda x: dm.sin(x) ** 2, lambda t: mpmath.sin(t) ** 2)
atom("x**3-x", lambda x: x**3 - x, lambda t: t**3 - t)
atom("1e6*sin(x)", lambda x: 1e6 * dm.sin(x), lambda t: 10**6 * mpmath.sin(t))

E = 2.0**-52
BOXES = [
    (-1.0, 1.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (-E, 1.0),
    (-1.0, E),
    (-1e-300, 1.0),
    (-1.0, 1e-300),
    (-2.2e-16, 1.0),
    (1e-300, 1.0),
    (1e-12, 1.0),
    (0.0, 1e-8),
    (-1e-8, 1e-8),
    (0.5, 0.5 + 1e-12),
    (-3.0, 7.0),
    (-100.0, 3.0),
    (1.0, 2.0),
    (-1e6, 1e6),
    (1e8, 1e8 + 1),
    (-0.999999, 0.999999),
    (-1.0, 1.0 - E),
    (0.999, 1.0),
    (1.0, 1.0 + 1e-9),
    (2.0, 50.0),
    (-50.0, -2.0),
    (-700, 700),
    (6.283185307179586 - 1e-9, 6.283185307179586 + 1e-9),
    (1.5, 1.6),
    (-0.3, 1e-15),
]

NSAMP = 4001


def samples(lo, hi):
    xs = np.linspace(lo, hi, NSAMP)
    extra = [lo, hi, 0.0, -1e-300, 1e-300, math.nextafter(lo, hi), math.nextafter(hi, lo)]
    xs = np.concatenate([xs, [t for t in extra if lo <= t <= hi]])
    return xs


def main():
    print(f"discopt loaded from {dm.__file__}", flush=True)  # CLAUDE.md section 8
    checks = 0
    witnesses = []
    errors = []
    for name, fdm, fmp, dom in ATOMS:
        for lo, hi in BOXES:
            if not dom(lo, hi):
                continue
            # Skip boxes whose f range overflows double.
            try:
                vals = []
                for t in samples(lo, hi)[:: max(1, NSAMP // 400)]:
                    vals.append(float(fmp(mpmath.mpf(t))))
                if not all(math.isfinite(v) and abs(v) < 1e200 for v in vals):
                    continue
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            for sense in ("min", "max"):
                m = dm.Model("fz")
                x = m.continuous("x", lb=lo, ub=hi)
                e = fdm(x)
                (m.minimize if sense == "min" else m.maximize)(e)
                try:
                    rel = build_uniform_relaxation(m)
                    res = rel.model.solve(backend="simplex")
                except Exception as exc:  # record, never swallow silently
                    errors.append((name, lo, hi, sense, repr(exc)[:200]))
                    continue
                if res.bound is None or res.status != "optimal":
                    continue
                b = float(res.bound)
                # rel bound is in the MINIMIZE-native sense; maximize is negated
                if sense == "max":
                    b = -b if rel.obj_sense_sign < 0 else b
                checks += 1
                worst = None
                for t in samples(lo, hi):
                    v = fmp(mpmath.mpf(float(t)))
                    v = float(v.real) if isinstance(v, mpmath.mpc) else float(v)
                    viol = (b - v) if sense == "min" else (v - b)
                    if viol > 1e-9 * max(1.0, abs(v)) and (worst is None or viol > worst[0]):
                        worst = (viol, float(t), v)
                if worst:
                    witnesses.append((name, lo, hi, sense, b, *worst))
                    print(
                        f"WITNESS {name:22s} box=[{lo!r}, {hi!r}] {sense} bound={b!r} "
                        f"x={worst[1]!r} f={worst[2]!r} viol={worst[0]:.3g}",
                        flush=True,
                    )
    print(
        f"\nchecks executed: {checks}; witnesses: {len(witnesses)}; "
        f"build/solve errors: {len(errors)}"
    )
    for err in errors[:30]:
        print("ERROR", err)
    if checks == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
