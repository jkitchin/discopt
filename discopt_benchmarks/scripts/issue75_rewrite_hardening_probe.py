"""Entry experiment (CLAUDE.md §4) for hardening the tape's algebraic rewrites.

Ten operators have no tape opcode and are lowered by rewrite. This measures, for
every one of them, over a domain that DELIBERATELY includes the overflow and
boundary regions, how each backend behaves on:

    * value non-finite (nan/inf) where the true value is finite
    * gradient non-finite where the true gradient is finite
    * relative drift at points where BOTH are finite

Run before the fix to establish the baseline, and after to show the delta.

Design notes:
  * NaN is never collapsed into "agreement" (an earlier version of this probe
    scored nan-vs-nan as reldiff 0.0 and hid exactly the failures it was built to
    find -- §6/§7: the instrument must not launder its own blind spot).
  * The reference for "the true value" is JAX, because `_jax/dag_compiler.py` is
    the documented authority the tape must reproduce. Where JAX is ALSO
    non-finite that is reported separately rather than scored as a tape defect.
  * No try/except around the evaluations (§7).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import math
import sys

import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
from discopt._nl_expr_compiler import compile_to_nl_expr
from discopt.modeling.core import Constant, FunctionCall

# --- unary rewrites: (dag build, jax authority, sample points) ------------------
_BIG = [1e-320, 1e-300, 1e-17, 1e-13, 1e-8, 1e-3, 0.5, 1.0, 2.0, 40.0, 300.0, 700.0, 745.0, 1e300]
_SIGNED = [-x for x in reversed(_BIG)] + [0.0] + _BIG

UNARY = {
    "log1p": (dm.log1p, jnp.log1p, [x for x in _SIGNED if x > -1.0]),
    "log2": (dm.log2, jnp.log2, _BIG),
    "sigmoid": (dm.sigmoid, jax.nn.sigmoid, _SIGNED),
    "softplus": (dm.softplus, lambda x: jnp.logaddexp(x, 0.0), _SIGNED),
    "abs": (abs, jnp.abs, _SIGNED),
    "sign": (dm.sign, jnp.sign, _SIGNED),
    "entropy": (
        lambda x: FunctionCall("entropy", x),
        lambda x: x * jnp.log(jnp.maximum(x, 1e-300)),
        [x for x in _SIGNED if x >= 0.0],
    ),
    # controls -- native opcodes, no rewrite. Must stay clean in every column.
    "exp": (dm.exp, jnp.exp, _SIGNED),
    "log": (dm.log, jnp.log, _BIG),
    "sqrt": (dm.sqrt, jnp.sqrt, _BIG),
}

BINARY = {
    "centropy": (
        lambda x, y: FunctionCall("centropy", x, y),
        lambda x, y: x * jnp.log(jnp.maximum(x, 1e-300) / y),
        [
            (a, b)
            for a in [0.0, 1e-300, 1e-8, 0.5, 2.0, 1e300]
            for b in [1e-300, 1e-8, 0.5, 2.0, 1e300]
        ],
    ),
    "signpower": (
        lambda x, y: FunctionCall("signpower", x, Constant(3.0)),
        lambda x, y: jnp.sign(x) * jnp.abs(x) ** 3.0,
        [(a, 3.0) for a in _SIGNED],
    ),
}


def finite(v):
    return math.isfinite(v)


def reldiff(t, j):
    if t == j:
        return 0.0
    return abs(t - j) / max(abs(t), abs(j), 1e-300)


rows = []
compared = 0

for name, (build, jfun, pts) in UNARY.items():
    m = dm.Model("p")
    x = m.continuous("x", lb=-1e309, ub=1e309)
    tape = compile_to_nl_expr(build(x), m)
    # `jfun=jfun` binds the CURRENT iteration's function; a bare closure over the
    # loop variable (ruff B023) would silently compare every operator against the
    # last one if this lambda ever outlived its iteration.
    jg = jax.grad(lambda v, jfun=jfun: jnp.asarray(jfun(v[0]), dtype=jnp.float64).sum())

    bad_v = bad_g = jax_bad = 0
    worst = 0.0
    worst_at = None
    for p in pts:
        jv = float(np.asarray(jfun(jnp.float64(p))))
        jgv = float(np.asarray(jg(np.array([float(p)])))[0])
        tv = float(tape.eval([float(p)]))
        tgv = float(np.asarray(tape.gradient([float(p)]), dtype=float)[0])
        compared += 2

        if not (finite(jv) and finite(jgv)):
            jax_bad += 1
            continue
        if not finite(tv):
            bad_v += 1
        if not finite(tgv):
            bad_g += 1
        if finite(tv) and finite(tgv):
            d = max(reldiff(tv, jv), reldiff(tgv, jgv))
            if d > worst:
                worst, worst_at = d, p
    rows.append((name, len(pts), bad_v, bad_g, jax_bad, worst, worst_at))

for name, (build, jfun, pts) in BINARY.items():
    m = dm.Model("p")
    x = m.continuous("x", lb=-1e309, ub=1e309)
    y = m.continuous("y", lb=-1e309, ub=1e309)
    tape = compile_to_nl_expr(build(x, y), m)
    jg = jax.grad(lambda v, jfun=jfun: jnp.asarray(jfun(v[0], v[1]), dtype=jnp.float64).sum())

    bad_v = bad_g = jax_bad = 0
    worst = 0.0
    worst_at = None
    for a, b in pts:
        arr = np.array([float(a), float(b)])
        jv = float(np.asarray(jfun(jnp.float64(a), jnp.float64(b))))
        jgv = np.asarray(jg(arr), dtype=float)
        tv = float(tape.eval([float(a), float(b)]))
        tgv = np.asarray(tape.gradient([float(a), float(b)]), dtype=float)
        compared += 2

        if not (finite(jv) and np.all(np.isfinite(jgv))):
            jax_bad += 1
            continue
        if not finite(tv):
            bad_v += 1
        if not np.all(np.isfinite(tgv)):
            bad_g += 1
        if finite(tv) and np.all(np.isfinite(tgv)):
            d = max(reldiff(tv, jv), max(reldiff(float(tgv[i]), float(jgv[i])) for i in range(2)))
            if d > worst:
                worst, worst_at = d, (a, b)
    rows.append((name, len(pts), bad_v, bad_g, jax_bad, worst, worst_at))

hdr = (
    f"{'operator':10s} {'pts':>4s} {'nonfin f':>9s} {'nonfin g':>9s} "
    f"{'jax n/f':>8s} {'worst rel':>11s}  worst@"
)
print(hdr)
print("-" * len(hdr))
defects = 0
for name, n, bv, bg, jb, w, wa in rows:
    flag = "  <== DEFECT" if (bv or bg or w > 1e-12) else ""
    defects += bool(bv or bg or w > 1e-12)
    print(f"{name:10s} {n:4d} {bv:9d} {bg:9d} {jb:8d} {w:11.3e}  {wa}{flag}")

print(f"\ncomparisons executed: {compared}")
print(f"operators with a defect: {defects}")
if compared == 0:
    print("PROBE ASSERTED NOTHING", file=sys.stderr)
    sys.exit(1)
