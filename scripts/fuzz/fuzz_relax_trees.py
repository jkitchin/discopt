"""Random expression-tree soundness fuzzer for the uniform relaxation (2 variables).

Generate random f(x, y) as a discopt expression and a parallel numpy/mpmath evaluator,
pick an adversarial box, build the uniform relaxation of min f (and max f), solve the
LP, and search a grid for a WITNESS point with f < bound (min) / f > bound (max).
Float witnesses are re-confirmed with mpmath at 50 digits before being reported.

Usage: python scripts/fuzz/fuzz_relax_trees.py SEED N
Prints the executed-check count; exits 2 if zero checks ran.
"""

import random
import sys
import warnings

import discopt.modeling as dm
import mpmath
import numpy as np
from discopt._relax.uniform_relax import build_uniform_relaxation

warnings.filterwarnings("ignore")
mpmath.mp.dps = 50

E = 2.0**-52
EDGE_VALUES = [0.0, -E, E, -2.2e-16, 1e-300, -1e-300, 1.0, -1.0, 1e-8, 0.5, 3.0, -3.0, 1e6, 1e-3]


class Node:
    def __init__(self, op, kids=(), val=None):
        self.op, self.kids, self.val = op, kids, val

    def __repr__(self):
        if self.op == "var":
            return self.val
        if self.op == "const":
            return repr(self.val)
        if self.op in "+-*/":
            return f"({self.kids[0]!r} {self.op} {self.kids[1]!r})"
        if self.op == "pow":
            return f"({self.kids[0]!r})**{self.val}"
        return f"{self.op}({self.kids[0]!r})"


UN = ["exp", "log", "sqrt", "sin", "cos", "abs", "tanh", "atan"]
POWS = [2, 3, 4, 5, -1, -2, 0.5, 1.5, 3]


def gen(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        r = rng.random()
        if r < 0.75:
            return Node("var", val=rng.choice("xy"))
        return Node("const", val=rng.choice([1.0, 2.0, 0.5, -1.0, 3.0, 1e-3, 1e3, 10.0]))
    r = rng.random()
    if r < 0.45:
        return Node(rng.choice("+-*/"), (gen(rng, depth - 1), gen(rng, depth - 1)))
    if r < 0.7:
        return Node("pow", (gen(rng, depth - 1),), rng.choice(POWS))
    return Node(rng.choice(UN), (gen(rng, depth - 1),))


def to_dm(n, env):
    if n.op == "var":
        return env[n.val]
    if n.op == "const":
        return n.val
    a = [to_dm(k, env) for k in n.kids]
    if n.op == "+":
        return a[0] + a[1]
    if n.op == "-":
        return a[0] - a[1]
    if n.op == "*":
        return a[0] * a[1]
    if n.op == "/":
        return a[0] / a[1]
    if n.op == "pow":
        return a[0] ** n.val
    return getattr(dm, n.op)(a[0])


NP = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "abs": np.abs,
    "tanh": np.tanh,
    "atan": np.arctan,
}
MP = {
    "exp": mpmath.exp,
    "log": mpmath.log,
    "sqrt": mpmath.sqrt,
    "sin": mpmath.sin,
    "cos": mpmath.cos,
    "abs": abs,
    "tanh": mpmath.tanh,
    "atan": mpmath.atan,
}


def ev(n, env, lib):
    if n.op == "var":
        return env[n.val]
    if n.op == "const":
        return n.val if lib is NP else mpmath.mpf(n.val)
    a = [ev(k, env, lib) for k in n.kids]
    if n.op == "+":
        return a[0] + a[1]
    if n.op == "-":
        return a[0] - a[1]
    if n.op == "*":
        return a[0] * a[1]
    if n.op == "/":
        return a[0] / a[1]
    if n.op == "pow":
        p = n.val
        if lib is NP:
            return np.power(a[0], float(p)) if not float(p).is_integer() else a[0] ** int(p)
        return mpmath.power(a[0], mpmath.mpf(p)) if not float(p).is_integer() else a[0] ** int(p)
    return lib[n.op](a[0])


def rand_box(rng):
    kind = rng.random()
    if kind < 0.5:
        a, b = sorted(rng.sample(EDGE_VALUES, 2))
    elif kind < 0.8:
        a = rng.choice(EDGE_VALUES)
        b = a + rng.choice([1e-9, 1e-3, 1.0, 10.0])
    else:
        a, b = sorted([rng.uniform(-5, 5), rng.uniform(-5, 5)])
    if not a < b:
        b = a + 1.0
    return a, b


def main():
    seed, n = int(sys.argv[1]), int(sys.argv[2])
    print(f"discopt loaded from {dm.__file__}", flush=True)  # CLAUDE.md section 8
    rng = random.Random(seed)
    checks = witnesses = errors = 0
    G = 161
    for it in range(n):
        tree = gen(rng, rng.choice([2, 3, 3, 4]))
        bx, by = rand_box(rng), rand_box(rng)
        xs = np.concatenate([np.linspace(*bx, G), bx, [0.0] if bx[0] <= 0 <= bx[1] else []])
        ys = np.concatenate([np.linspace(*by, G), by, [0.0] if by[0] <= 0 <= by[1] else []])
        X, Y = np.meshgrid(xs, ys)
        try:
            with np.errstate(all="ignore"):
                F = ev(tree, {"x": X, "y": Y}, NP)
        except (ArithmeticError, ValueError):
            continue  # f undefined somewhere on the grid: skip, the domain is ambiguous
        F = np.broadcast_to(np.asarray(F, dtype=float), X.shape)
        # Only well-defined, moderate-range functions: no ambiguity about the domain.
        if not np.all(np.isfinite(F)) or np.max(np.abs(F)) > 1e12:
            continue
        if "x" not in repr(tree) and "y" not in repr(tree):
            continue
        for sense in ("min", "max"):
            m = dm.Model("fz")
            env = {
                "x": m.continuous("x", lb=bx[0], ub=bx[1]),
                "y": m.continuous("y", lb=by[0], ub=by[1]),
            }
            try:
                e = to_dm(tree, env)
                (m.minimize if sense == "min" else m.maximize)(e)
                rel = build_uniform_relaxation(m)
                res = rel.model.solve(backend="simplex")
            except Exception as exc:
                errors += 1
                print(f"ERROR it={it} {sense} {tree!r} box={bx},{by}: {exc!r:.200}", flush=True)
                continue
            if res.bound is None or res.status != "optimal":
                continue
            b = float(res.bound)
            if sense == "max" and rel.obj_sense_sign < 0:
                b = -b
            checks += 1
            V = (b - F) if sense == "min" else (F - b)
            k = int(np.argmax(V))
            if V.flat[k] <= 1e-7 * max(1.0, abs(F.flat[k])):
                continue
            xw, yw = float(X.flat[k]), float(Y.flat[k])
            fm = ev(tree, {"x": mpmath.mpf(xw), "y": mpmath.mpf(yw)}, MP)
            fm = float(fm.real) if isinstance(fm, mpmath.mpc) else float(fm)
            viol = (b - fm) if sense == "min" else (fm - b)
            if viol > 1e-7 * max(1.0, abs(fm)):
                witnesses += 1
                print(
                    f"WITNESS it={it} {sense} f={tree!r} box x={bx} y={by} bound={b!r} "
                    f"at x={xw!r} y={yw!r} f={fm!r} viol={viol:.3g}",
                    flush=True,
                )
    print(f"\nseed={seed} checks executed: {checks}; witnesses: {witnesses}; errors: {errors}")
    if checks == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
