"""Neutral instance corpus for the discopt vs. SUSPECT convexity head-to-head.

This module is the **single source of truth** for the shared instance set. It
is deliberately dependency-free (standard library only) so it can be imported
*both* by the discopt-side parity test (modern numpy / pyomo-free environment)
*and* by the SUSPECT oracle runner (isolated py3.10 + pyomo 6.1.2 environment).
Neither tool's model objects appear here; each side renders the same neutral
AST into its own representation, which is what makes the comparison a genuine
head-to-head on identical mathematics rather than two hand-written models that
might silently differ.

Expression AST
--------------
An expression is one of:

* ``int`` / ``float``                      -- a numeric constant
* ``("var", name)``                        -- a decision variable
* ``("+", a, b, ...)``                      -- n-ary sum
* ``("-", a, b)``                           -- binary difference
* ``("*", a, b, ...)``                      -- n-ary product
* ``("/", a, b)``                           -- division
* ``("neg", a)``                            -- unary negation
* ``("pow", a, p)``                         -- ``a ** p`` (``p`` numeric constant)
* ``("exp", a)`` / ``("log", a)`` / ``("log2", a)`` / ``("sqrt", a)``
* ``("sin", a)`` / ``("cos", a)`` / ``("tan", a)`` / ``("abs", a)``
* ``("asin", a)`` / ``("acos", a)`` / ``("atan", a)``

Instance schema
---------------
Each instance is a dict::

    {
        "name": str,
        "vars": {name: (lb, ub), ...},           # finite bounds (SUSPECT FBBT needs them)
        "objective": {"sense": "min"|"max", "expr": AST} | None,
        "constraints": [{"name": str, "expr": AST, "op": "<="|">="|"==", "rhs": float}, ...],
        "note": str,                              # human description of the intended structure
    }

Every comparable *item* (the objective, each constraint) is reduced on both
sides to the curvature of its ``<=``-normalised body, so the verdicts line up
directly with SUSPECT's per-constraint / per-objective ``Convexity`` enum.

Bounds are chosen finite and domain-safe (strictly positive where ``log`` /
``sqrt`` / division by a variable appear) so SUSPECT's interval FBBT does not
raise ``EmptyIntervalError`` on degenerate boxes.
"""

from __future__ import annotations

import math

# --- AST constructor helpers (readability only; the AST is plain tuples) -----


def var(name: str):
    return ("var", name)


def add(*args):
    return ("+", *args)


def sub(a, b):
    return ("-", a, b)


def mul(*args):
    return ("*", *args)


def div(a, b):
    return ("/", a, b)


def neg(a):
    return ("neg", a)


def power(a, p):
    return ("pow", a, p)


def exp(a):
    return ("exp", a)


def log(a):
    return ("log", a)


def sqrt(a):
    return ("sqrt", a)


def sin(a):
    return ("sin", a)


def cos(a):
    return ("cos", a)


def tan(a):
    return ("tan", a)


def asin(a):
    return ("asin", a)


def acos(a):
    return ("acos", a)


def atan(a):
    return ("atan", a)


def log2(a):
    return ("log2", a)


def absval(a):
    return ("abs", a)


# Atoms SUSPECT has a convexity rule for but that NO shared instance can reach.
# Empty by construction: discopt's expression layer is a generic FunctionCall
# node whose JAX backend already maps every SUSPECT atom (asin->jnp.arcsin,
# etc.), so discopt can represent anything SUSPECT can. (asin/acos/atan have no
# convenience wrapper in discopt.modeling, but build fine via FunctionCall and
# are exercised below -- they are detector gaps, not expressivity gaps.)
UNBUILDABLE_SUSPECT_ATOMS: tuple[str, ...] = ()

# The reverse gap: atoms discopt expresses but SUSPECT 2.1.3 cannot -- it raises
# "ValueError: Unknown function type" for these (no rule in its registry), so no
# shared instance can be built. (``log2`` IS buildable: it is rendered as a
# positively-scaled native ``log``, so SUSPECT sees an ordinary concave ``log``
# rather than a dedicated atom.)
SUSPECT_UNSUPPORTED_ATOMS = ("tanh", "sinh", "cosh", "log10")


# --- Curated instances -------------------------------------------------------
#
# "expected" records the ground-truth curvature of the item's <=-normalised
# body: "convex", "concave", "affine", or "indefinite" (provably neither).
# It is documentation / sanity-check fodder, not used to score either tool
# against the other.

INSTANCES: list[dict] = [
    # ----- Convex objectives: cone primitives discopt recognises -----
    {
        "name": "psd_quadratic",
        "note": "PSD quadratic form x^2 + y^2 (convex).",
        "vars": {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        "objective": {"sense": "min", "expr": add(power(var("x"), 2), power(var("y"), 2))},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "shifted_sum_of_squares",
        "note": "(x-1)^2 + (2y+3)^2 -- convex, affine inner.",
        "vars": {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        "objective": {
            "sense": "min",
            "expr": add(power(sub(var("x"), 1), 2), power(add(mul(2, var("y")), 3), 2)),
        },
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "euclidean_norm",
        "note": "sqrt(x^2 + y^2) -- 2-norm, convex via sqrt-of-PSD-quadratic.",
        "vars": {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        "objective": {
            "sense": "min",
            "expr": sqrt(add(power(var("x"), 2), power(var("y"), 2))),
        },
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "exp_univariate",
        "note": "exp(x) -- convex.",
        "vars": {"x": (-3.0, 3.0)},
        "objective": {"sense": "min", "expr": exp(var("x"))},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "sum_exp",
        "note": "exp(x) + exp(2y) -- sum of convex exponentials.",
        "vars": {"x": (-3.0, 3.0), "y": (-3.0, 3.0)},
        "objective": {
            "sense": "min",
            "expr": add(exp(var("x")), exp(mul(2, var("y")))),
        },
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "neg_entropy",
        "note": "x*log(x) -- convex on x>0 (negative entropy).",
        "vars": {"x": (0.05, 10.0)},
        "objective": {"sense": "min", "expr": mul(var("x"), log(var("x")))},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "reciprocal",
        "note": "1/x -- convex on x>0.",
        "vars": {"x": (0.05, 10.0)},
        "objective": {"sense": "min", "expr": div(1.0, var("x"))},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "quad_over_affine",
        "note": "x^2 / y -- convex on y>0 (quadratic-over-affine).",
        "vars": {"x": (-5.0, 5.0), "y": (0.1, 10.0)},
        "objective": {"sense": "min", "expr": div(power(var("x"), 2), var("y"))},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "exp_perspective",
        "note": "y*exp(x/y) -- perspective of exp, convex on y>0.",
        "vars": {"x": (-3.0, 3.0), "y": (0.1, 5.0)},
        "objective": {
            "sense": "min",
            "expr": mul(var("y"), exp(div(var("x"), var("y")))),
        },
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "softplus",
        "note": "log(exp(x) + exp(y)) -- log-sum-exp, convex.",
        "vars": {"x": (-3.0, 3.0), "y": (-3.0, 3.0)},
        "objective": {
            "sense": "min",
            "expr": log(add(exp(var("x")), exp(var("y")))),
        },
        "constraints": [],
        "expected": "convex",
    },
    # ----- Concave objectives (convex problems under max) -----
    {
        "name": "log_concave",
        "note": "max log(x) -- concave body, convex problem.",
        "vars": {"x": (0.05, 10.0)},
        "objective": {"sense": "max", "expr": log(var("x"))},
        "constraints": [],
        "expected": "concave",
    },
    {
        "name": "sqrt_concave",
        "note": "max sqrt(x) -- concave body.",
        "vars": {"x": (0.0, 10.0)},
        "objective": {"sense": "max", "expr": sqrt(var("x"))},
        "constraints": [],
        "expected": "concave",
    },
    {
        "name": "geo_mean_2d",
        "note": "max sqrt(x*y) -- weighted geometric mean, concave on the positive orthant.",
        "vars": {"x": (0.05, 10.0), "y": (0.05, 10.0)},
        "objective": {"sense": "max", "expr": sqrt(mul(var("x"), var("y")))},
        "constraints": [],
        "expected": "concave",
    },
    # ----- Affine -----
    {
        "name": "affine_objective",
        "note": "3x - 2y + 1 -- affine.",
        "vars": {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        "objective": {
            "sense": "min",
            "expr": add(mul(3, var("x")), mul(-2, var("y")), 1),
        },
        "constraints": [],
        "expected": "affine",
    },
    # ----- Nonconvex / indefinite objectives (foils) -----
    {
        "name": "bilinear",
        "note": "x*y -- indefinite (saddle), neither convex nor concave.",
        "vars": {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        "objective": {"sense": "min", "expr": mul(var("x"), var("y"))},
        "constraints": [],
        "expected": "indefinite",
    },
    {
        "name": "indefinite_quadratic",
        "note": "x^2 - y^2 -- indefinite quadratic.",
        "vars": {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        "objective": {"sense": "min", "expr": sub(power(var("x"), 2), power(var("y"), 2))},
        "constraints": [],
        "expected": "indefinite",
    },
    {
        "name": "sine",
        "note": "sin(x) -- nonconvex (oscillatory) on a wide box.",
        "vars": {"x": (-3.0, 3.0)},
        "objective": {"sense": "min", "expr": sin(var("x"))},
        "constraints": [],
        "expected": "indefinite",
    },
    {
        "name": "cubic",
        "note": "x^3 -- neither convex nor concave on a box straddling 0.",
        "vars": {"x": (-3.0, 3.0)},
        "objective": {"sense": "min", "expr": power(var("x"), 3)},
        "constraints": [],
        "expected": "indefinite",
    },
    # ----- Constraint sense-normalisation (convex feasible sets) -----
    {
        "name": "ball_le",
        "note": "x^2 + y^2 <= 1 -- convex feasible set (convex body, <=).",
        "vars": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)},
        "objective": None,
        "constraints": [
            {
                "name": "ball",
                "expr": add(power(var("x"), 2), power(var("y"), 2)),
                "op": "<=",
                "rhs": 1.0,
            }
        ],
        "expected": "convex",
    },
    {
        "name": "log_lower_bound",
        "note": "log(x) >= 0 -- concave body with >=, convex feasible set.",
        "vars": {"x": (0.05, 10.0)},
        "objective": None,
        "constraints": [{"name": "logc", "expr": log(var("x")), "op": ">=", "rhs": 0.0}],
        "expected": "concave",  # <=-normalised body (-log x) is convex; raw body concave
    },
    {
        "name": "quad_over_affine_epigraph",
        "note": "x^2/y <= t style: x^2 - y*t <= 0 is hard; use x^2/y <= 5 (nlp_cvx_108 shape).",
        "vars": {"x": (-5.0, 5.0), "y": (0.1, 10.0)},
        "objective": None,
        "constraints": [
            {"name": "qoa", "expr": div(power(var("x"), 2), var("y")), "op": "<=", "rhs": 5.0}
        ],
        "expected": "convex",
    },
    {
        "name": "norm_le",
        "note": "sqrt(x^2+y^2) <= 3 -- second-order cone, convex set.",
        "vars": {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        "objective": None,
        "constraints": [
            {
                "name": "soc",
                "expr": sqrt(add(power(var("x"), 2), power(var("y"), 2))),
                "op": "<=",
                "rhs": 3.0,
            }
        ],
        "expected": "convex",
    },
    # ----- Nonconvex feasible sets (foils, sense matters) -----
    {
        "name": "ball_complement_ge",
        "note": "x^2 + y^2 >= 1 -- convex body with >=, NONconvex feasible set.",
        "vars": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)},
        "objective": None,
        "constraints": [
            {
                "name": "shell",
                "expr": add(power(var("x"), 2), power(var("y"), 2)),
                "op": ">=",
                "rhs": 1.0,
            }
        ],
        "expected": "convex",  # raw body convex; the >= makes the SET nonconvex (SUSPECT: Concave)
    },
    {
        "name": "log_upper_bound",
        "note": "log(x) <= 1 -- concave body with <=, NONconvex feasible set.",
        "vars": {"x": (0.05, 10.0)},
        "objective": None,
        "constraints": [{"name": "logu", "expr": log(var("x")), "op": "<=", "rhs": 1.0}],
        "expected": "concave",
    },
    {
        "name": "bilinear_constraint",
        "note": "x*y <= 1 -- indefinite body, nonconvex.",
        "vars": {"x": (0.1, 5.0), "y": (0.1, 5.0)},
        "objective": None,
        "constraints": [{"name": "bil", "expr": mul(var("x"), var("y")), "op": "<=", "rhs": 1.0}],
        "expected": "indefinite",
    },
    # ----- Mixed objective + constraints (whole-model convex) -----
    {
        "name": "convex_qp_model",
        "note": "min x^2+y^2 s.t. affine + a convex ball constraint -- fully convex model.",
        "vars": {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        "objective": {"sense": "min", "expr": add(power(var("x"), 2), power(var("y"), 2))},
        "constraints": [
            {"name": "aff", "expr": add(var("x"), var("y")), "op": "<=", "rhs": 4.0},
            {
                "name": "ball",
                "expr": add(power(sub(var("x"), 1), 2), power(var("y"), 2)),
                "op": "<=",
                "rhs": 9.0,
            },
        ],
        "expected": "convex",
    },
    {
        "name": "nonconvex_mixed_model",
        "note": "convex objective but a bilinear equality -- whole model nonconvex.",
        "vars": {"x": (0.1, 5.0), "y": (0.1, 5.0)},
        "objective": {"sense": "min", "expr": add(power(var("x"), 2), power(var("y"), 2))},
        "constraints": [
            {"name": "bil_eq", "expr": mul(var("x"), var("y")), "op": "==", "rhs": 1.0},
        ],
        "expected": "indefinite",
    },
    # ----- Atom coverage: SUSPECT convexity-rule atoms not in the core set -----
    {
        "name": "abs_value",
        "note": "|x| -- convex (SUSPECT AbsRule).",
        "vars": {"x": (-5.0, 5.0)},
        "objective": {"sense": "min", "expr": absval(var("x"))},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "tan_convex_branch",
        "note": "tan(x) on (0, pi/2) -- convex & increasing (bound-dependent, SUSPECT TanRule).",
        "vars": {"x": (0.1, 1.4)},
        "objective": {"sense": "min", "expr": tan(var("x"))},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "log2_concave",
        "note": "max log2(x) -- concave (base-2 log).",
        "vars": {"x": (0.05, 10.0)},
        "objective": {"sense": "max", "expr": log2(var("x"))},
        "constraints": [],
        "expected": "concave",
    },
    # ----- Power-exponent matrix (SUSPECT PowerRule: sign/parity/bound-aware) -----
    {
        "name": "quartic_even_power",
        "note": "x^4 -- convex (even integer power) on a box straddling 0.",
        "vars": {"x": (-3.0, 3.0)},
        "objective": {"sense": "min", "expr": power(var("x"), 4)},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "inverse_square_power",
        "note": "x^-2 -- convex on x>0 (negative even power).",
        "vars": {"x": (0.1, 5.0)},
        "objective": {"sense": "min", "expr": power(var("x"), -2)},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "pow_three_halves",
        "note": "x^1.5 -- convex on x>=0 (fractional power > 1).",
        "vars": {"x": (0.05, 5.0)},
        "objective": {"sense": "min", "expr": power(var("x"), 1.5)},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "pow_half_concave",
        "note": "max x^0.5 -- concave on x>=0 (fractional power in (0,1); sqrt as a power atom).",
        "vars": {"x": (0.05, 10.0)},
        "objective": {"sense": "max", "expr": power(var("x"), 0.5)},
        "constraints": [],
        "expected": "concave",
    },
    {
        "name": "cubic_positive_branch",
        "note": "x^3 on x>=0 -- convex (bound-restricted odd power; bound-aware PowerRule).",
        "vars": {"x": (0.1, 3.0)},
        "objective": {"sense": "min", "expr": power(var("x"), 3)},
        "constraints": [],
        "expected": "convex",
    },
    # ----- Bound-restricted trig convexity -----
    {
        "name": "sin_convex_branch",
        "note": "sin(x) on [pi, 2pi] -- convex (sin'' = -sin >= 0 there; bound-aware SinRule).",
        "vars": {"x": (3.3, 6.0)},
        "objective": {"sense": "min", "expr": sin(var("x"))},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "cos_concave_branch",
        "note": "max cos(x) on [-pi/2, pi/2] -- cos concave there (bound-aware CosRule).",
        "vars": {"x": (-1.4, 1.4)},
        "objective": {"sense": "max", "expr": cos(var("x"))},
        "constraints": [],
        "expected": "concave",
    },
    # ----- Inverse trig (SUSPECT Asin/Acos/AtanRule; discopt backend supports
    #       them via FunctionCall but has no curvature rule -> detector gaps) -----
    {
        "name": "asin_convex_branch",
        "note": "asin(x) on (0,1) -- convex & increasing there (bound-aware AsinRule).",
        "vars": {"x": (0.1, 0.9)},
        "objective": {"sense": "min", "expr": asin(var("x"))},
        "constraints": [],
        "expected": "convex",
    },
    {
        "name": "acos_concave_branch",
        "note": "max acos(x) on (0,1) -- acos concave there (bound-aware AcosRule).",
        "vars": {"x": (0.1, 0.9)},
        "objective": {"sense": "max", "expr": acos(var("x"))},
        "constraints": [],
        "expected": "concave",
    },
    {
        "name": "atan_concave_branch",
        "note": "max atan(x) on x>0 -- atan concave on the positive axis (bound-aware AtanRule).",
        "vars": {"x": (0.1, 3.0)},
        "objective": {"sense": "max", "expr": atan(var("x"))},
        "constraints": [],
        "expected": "concave",
    },
]


def instance_by_name(name: str) -> dict:
    for inst in INSTANCES:
        if inst["name"] == name:
            return inst
    raise KeyError(name)


# --- Numeric evaluation of the neutral AST ----------------------------------
#
# Evaluates an expression AST at a concrete point ``env`` (``{var_name: float}``)
# using only the standard library. This is the ground truth both detector-side
# verdicts are validated against: the monotonicity and FBBT-bounds parity tests
# *sample* the real body over its box with this evaluator and check that
# discopt's symbolic verdict (monotone direction / interval enclosure) is
# consistent with the sampled values. It deliberately shares no code with
# either renderer, so a bug in a renderer cannot mask a wrong verdict.


def eval_ast(node, env: dict) -> float:
    """Evaluate a neutral AST ``node`` at the point ``env`` (``{name: float}``)."""
    if isinstance(node, (int, float)):
        return float(node)
    head = node[0]
    if head == "var":
        return float(env[node[1]])
    args = [eval_ast(a, env) for a in node[1:]]
    if head == "+":
        return math.fsum(args)
    if head == "-":
        return args[0] - args[1]
    if head == "*":
        acc = 1.0
        for a in args:
            acc *= a
        return acc
    if head == "/":
        return args[0] / args[1]
    if head == "neg":
        return -args[0]
    if head == "pow":
        return float(args[0]) ** node[2]
    if head == "exp":
        return math.exp(args[0])
    if head == "log":
        return math.log(args[0])
    if head == "sqrt":
        return math.sqrt(args[0])
    if head == "sin":
        return math.sin(args[0])
    if head == "cos":
        return math.cos(args[0])
    if head == "tan":
        return math.tan(args[0])
    if head == "asin":
        return math.asin(args[0])
    if head == "acos":
        return math.acos(args[0])
    if head == "atan":
        return math.atan(args[0])
    if head == "log2":
        return math.log2(args[0])
    if head == "abs":
        return abs(args[0])
    raise ValueError(f"unknown AST head: {head!r}")


def item_asts(instance: dict) -> dict:
    """Map each comparable item key to its raw body AST.

    Keys are ``"objective"`` (when present) and each constraint name -- the
    same keys the discopt renderer and the SUSPECT golden use, so a test can
    line up a verdict with the AST it should be validated against.
    """
    out: dict = {}
    obj = instance.get("objective")
    if obj is not None:
        out["objective"] = obj["expr"]
    for con in instance["constraints"]:
        out[con["name"]] = con["expr"]
    return out
