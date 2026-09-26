# ruff: noqa: E402  -- imports follow the §8 load assertion and the counter install
"""#1479 bound-neutral panel: run ONE case against ONE tree, print one JSON line.

Usage: PYTHONPATH=<tree>/python python case.py <tree> <case>

Asserts the loaded discopt is the one under <tree> and records whether the
tree carries the registry dispatch (marker); in a tree that does, every
Transformation.apply call is counted by name so the panel proves the rerouted
call sites actually ran (CLAUDE.md §6/§8). Driven by drive.py.
"""

import json
import pathlib
import sys
import warnings

tree, case = sys.argv[1], sys.argv[2]
warnings.simplefilter("ignore")
import discopt

assert discopt.__file__.startswith(tree + "/"), (discopt.__file__, tree)  # CLAUDE.md §8
src = pathlib.Path(tree, "python/discopt/solver.py").read_text()
marker = '_get_transformation("gdp")' in src
calls = {}
if marker:
    import discopt.transformations as dt

    orig = dt.Transformation.apply

    def counted(self, model, **o):
        calls[self.name] = calls.get(self.name, 0) + 1
        return orig(self, model, **o)

    dt.Transformation.apply = counted
import discopt.modeling as dm
from discopt import mpec
from discopt.modeling.core import (
    from_nl,
)


def gdp_lin():
    m = dm.Model("gl")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x + y)
    m.either_or([[x >= 4, y >= 1], [y >= 6, x >= 1]])
    return m


def gdp_nl():
    m = dm.Model("gn")
    x = m.continuous("x", lb=-4, ub=4)
    y = m.continuous("y", lb=-4, ub=4)
    m.minimize((x - 3) ** 2 + (y - 3) ** 2)
    m.either_or([[x**2 + y**2 <= 1], [(x - 2) ** 2 + (y + 1) ** 2 <= 1]])
    return m


def gdp_ind():
    m = dm.Model("gi")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    m.minimize(-x + 3 * z)
    m.if_then(z, [x <= 8]) if hasattr(m, "if_then") else None
    m.subject_to(x <= 4 + 6 * z)
    return m


def intbil():
    m = dm.Model("ib")
    a = m.integer("a", lb=0, ub=5)
    b = m.continuous("b", lb=0, ub=3)
    m.minimize(-a * b + a)
    m.subject_to(a * b <= 7)
    return m


def intbil2():
    m = dm.Model("ib2")
    a = m.integer("a", lb=0, ub=6)
    c = m.integer("c", lb=0, ub=6)
    m.minimize(-(a * c) + 2 * a + c)
    m.subject_to(a * c <= 10)
    m.subject_to(a + c >= 3)
    return m


def intml():
    m = dm.Model("im")
    a = m.integer("a", lb=0, ub=4)
    c = m.integer("c", lb=0, ub=3)
    b = m.continuous("b", lb=0, ub=2)
    m.minimize(-a * c * b + a + c)
    m.subject_to(a * c * b <= 5)
    return m


def binml():
    m = dm.Model("bm")
    z = m.binary("z", shape=(4,))
    m.minimize(-z[0] * z[1] * z[2] + z[0] - 2 * z[1] * z[3] + z[2])
    m.subject_to(z[0] + z[1] >= 1)
    return m


def mpcc(method):
    m = dm.Model("mp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    m.complementarity(x, y, method=method)
    return m


def res(r):
    return {
        k: (None if getattr(r, k, None) is None else repr(getattr(r, k)))
        for k in ("status", "objective", "bound", "node_count", "gap_certified")
    }


kind, _, arg = case.partition(":")
TL = 30.0
if kind == "nl":
    out = res(from_nl(arg).solve(time_limit=TL))
elif kind == "gdp":
    name, _, route = arg.partition("/")
    m = {"lin": gdp_lin, "nl": gdp_nl, "ind": gdp_ind}[name]()
    kw = {
        "default": {},
        "hull": {"gdp_method": "hull"},
        "bigm": {"gdp_method": "big-m"},
        "mbigm": {"gdp_method": "mbigm"},
        "auto": {"gdp_method": "auto"},
        "amp": {"solver": "amp"},
        "mipnlp": {"solver": "mip-nlp"},
        "oa": {"gdp_method": "oa"},
        "loa": {"gdp_method": "loa"},
        "nlpbb": {"nlp_bb": True},
        "validate": {"validate": True},
    }[route]
    out = res(m.solve(time_limit=TL, **kw))
elif kind == "prod":
    name, _, route = arg.partition("/")
    m = {"ib": intbil, "ib2": intbil2, "im": intml, "bm": binml}[name]()
    kw = {"default": {}, "amp": {"solver": "amp"}}[route]
    out = res(m.solve(time_limit=TL, **kw))
elif kind == "mpcc":
    out = res(mpcc(arg).solve(time_limit=TL))
elif kind == "solve_mpec":
    m = dm.Model("sm")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    pair = mpec.complementarity(x, y)
    out = res(mpec.solve_mpec(m, [pair], method=arg))
elif kind == "mcp":
    m = dm.Model("mc")
    z = m.continuous("z", lb=0, ub=10)
    w = m.continuous("w", lb=0, ub=10)
    m.minimize((z - 2) ** 2 + (w - 1) ** 2)
    m.mcp(w - z + 0.5, z, lb=0.0, ub=float("inf"))
    out = res(m.solve(time_limit=TL))
elif kind == "bilevel":
    from discopt.bilevel import BilevelProblem

    m = dm.Model("bard")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - 4 * y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[x + y >= 3, y <= 2 * x],
        lower_sense="min",
        multiplier_ub=50.0,
    )
    bl.formulate(method="kkt", mpec_method=arg)
    out = res(m.solve(time_limit=TL))
else:
    raise SystemExit(f"unknown case {case}")
print(json.dumps({"case": case, "tree": tree, "marker": marker, "calls": calls, **out}), flush=True)
