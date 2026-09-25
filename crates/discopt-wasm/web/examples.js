// The examples offered by the dropdown: one model per problem class.
//
// Every block here was executed against discopt before it shipped
// (`python/tests/test_wasm_examples.py` keeps doing so in CI -- it lives under
// `python/tests` rather than next to this file because that is the only tree
// pytest collects). An example that does not solve is worse than none, because
// the dropdown's default is the page's first impression.
//
// The taxonomy is the one in `docs/notebooks/problem_classes.ipynb` on purpose:
// the same six models appear in both, so the page and the notebook cannot drift
// into disagreeing about what discopt does with each class.

export const EXAMPLES = [
  {
    id: "LP",
    label: "LP \u2014 feed blending",
    title: "Linear program \u2014 feed blending",
    code: "# LINEAR PROGRAM -- a blending problem.\n# Cheapest mix of three feeds that meets a protein and a fibre floor.\n# Continuous variables, linear objective, linear constraints.\nimport discopt.modeling as dm\n\nm = dm.Model(\"feed blend\")\n\n# Tonnes of each feed. Bounded above by what the supplier can deliver.\ncorn = m.continuous(\"corn\", lb=0, ub=40)\nsoy = m.continuous(\"soy\", lb=0, ub=30)\nbran = m.continuous(\"bran\", lb=0, ub=25)\n\ncost = 210 * corn + 340 * soy + 155 * bran\nm.minimize(cost)\n\nm.subject_to(corn + soy + bran == 50, name=\"tonnage\")\nm.subject_to(0.09 * corn + 0.44 * soy + 0.15 * bran >= 0.20 * 50, name=\"protein\")\nm.subject_to(0.02 * corn + 0.06 * soy + 0.11 * bran >= 0.05 * 50, name=\"fibre\")\n\nres = m.solve()\n\nprint(res.status)\nprint(f\"cost = ${res.objective:,.2f} for 50 t\")\nfor name, v in [(\"corn\", corn), (\"soy\", soy), (\"bran\", bran)]:\n    print(f\"  {name:5s} {res.value(v):6.2f} t\")\n",
  },
  {
    id: "MILP",
    label: "MILP \u2014 facility location",
    title: "Mixed-integer linear \u2014 facility location",
    code: "# MIXED-INTEGER LINEAR PROGRAM -- facility location.\n# Open a subset of warehouses, then ship from the open ones only.\n# Binary open/closed decisions + continuous shipment volumes.\nimport discopt.modeling as dm\n\nsites = [\"Pittsburgh\", \"Fresno\", \"Tulsa\"]\nfixed = [190.0, 240.0, 155.0]      # $k to open\ncap = [70.0, 90.0, 55.0]           # kt/yr\nship = [[4.0, 9.0, 6.0],           # $/t, site -> region\n        [8.0, 3.0, 7.0],\n        [5.0, 8.0, 4.0]]\ndemand = [40.0, 55.0, 35.0]        # kt/yr per region\n\nm = dm.Model(\"facility location\")\n\nopen_ = m.binary(\"open\", shape=(3,))\nx = m.continuous(\"x\", shape=(3, 3), lb=0, ub=90)\n\nm.minimize(\n    dm.sum([fixed[i] * open_[i] for i in range(3)])\n    + dm.sum([ship[i][j] * x[i, j] for i in range(3) for j in range(3)])\n)\n\nfor j in range(3):\n    m.subject_to(dm.sum([x[i, j] for i in range(3)]) >= demand[j], name=f\"demand{j}\")\nfor i in range(3):\n    # The big-M that links the two kinds of variable: ship nothing from a site\n    # that is not open, and never more than its capacity.\n    m.subject_to(dm.sum([x[i, j] for j in range(3)]) <= cap[i] * open_[i], name=f\"cap{i}\")\n\nres = m.solve()\n\n# `value()` takes the variable and returns its whole array -- index the result,\n# not the variable.\nopened, flow = res.value(open_), res.value(x)\n\nprint(res.status, f\"in {res.node_count} nodes\")\nprint(f\"total cost = ${res.objective:,.1f}k\")\nfor i, site in enumerate(sites):\n    if opened[i] > 0.5:\n        flows = \", \".join(f\"r{j}={flow[i, j]:.1f}\" for j in range(3))\n        print(f\"  OPEN  {site:12s} {flows}\")\n    else:\n        print(f\"  --    {site}\")\n",
  },
  {
    id: "QP",
    label: "QP \u2014 portfolio variance",
    title: "Quadratic \u2014 portfolio variance",
    code: "# QUADRATIC PROGRAM -- Markowitz portfolio selection.\n# Minimize variance at a required return. Convex quadratic objective,\n# linear constraints, continuous variables.\nimport discopt.modeling as dm\n\nret = [0.07, 0.11, 0.05, 0.14]           # expected annual return\n# Covariance matrix (symmetric, positive definite).\ncov = [[0.0400, 0.0120, 0.0020, 0.0180],\n       [0.0120, 0.0900, 0.0030, 0.0400],\n       [0.0020, 0.0030, 0.0100, 0.0040],\n       [0.0180, 0.0400, 0.0040, 0.1600]]\n\nm = dm.Model(\"portfolio\")\n\nw = m.continuous(\"w\", shape=(4,), lb=0, ub=1)    # long-only weights\n\n# Portfolio variance w' C w.\nm.minimize(dm.sum([cov[i][j] * w[i] * w[j] for i in range(4) for j in range(4)]))\n\nm.subject_to(dm.sum([w[i] for i in range(4)]) == 1, name=\"budget\")\nm.subject_to(dm.sum([ret[i] * w[i] for i in range(4)]) >= 0.09, name=\"return\")\n\nres = m.solve()\n\nweights = res.value(w)   # the whole array; `value()` takes the variable\n\nprint(res.status)\nprint(f\"variance = {res.objective:.5f}   (sd = {res.objective ** 0.5:.3%})\")\nfor i in range(4):\n    print(f\"  asset {i}  {weights[i]:6.2%}  (return {ret[i]:.0%})\")\n",
  },
  {
    id: "MIQP",
    label: "MIQP \u2014 cardinality-constrained portfolio",
    title: "Mixed-integer quadratic \u2014 cardinality-constrained portfolio",
    code: "# MIXED-INTEGER QUADRATIC PROGRAM -- the portfolio again, with a\n# cardinality limit: hold at most 2 of the 4 assets, and any asset held\n# must be a meaningful position. Convex QP objective + binary indicators.\nimport discopt.modeling as dm\n\nret = [0.07, 0.11, 0.05, 0.14]\ncov = [[0.0400, 0.0120, 0.0020, 0.0180],\n       [0.0120, 0.0900, 0.0030, 0.0400],\n       [0.0020, 0.0030, 0.0100, 0.0040],\n       [0.0180, 0.0400, 0.0040, 0.1600]]\n\nm = dm.Model(\"portfolio with cardinality\")\n\nw = m.continuous(\"w\", shape=(4,), lb=0, ub=1)\nhold = m.binary(\"hold\", shape=(4,))\n\nm.minimize(dm.sum([cov[i][j] * w[i] * w[j] for i in range(4) for j in range(4)]))\n\nm.subject_to(dm.sum([w[i] for i in range(4)]) == 1, name=\"budget\")\nm.subject_to(dm.sum([ret[i] * w[i] for i in range(4)]) >= 0.09, name=\"return\")\nm.subject_to(dm.sum([hold[i] for i in range(4)]) <= 2, name=\"cardinality\")\nfor i in range(4):\n    # Semi-continuous: either nothing, or a position of at least 10%.\n    m.subject_to(w[i] <= hold[i], name=f\"on{i}\")\n    m.subject_to(w[i] >= 0.10 * hold[i], name=f\"floor{i}\")\n\nres = m.solve()\n\nweights, held = res.value(w), res.value(hold)\n\nprint(res.status, f\"in {res.node_count} nodes\")\nprint(f\"variance = {res.objective:.5f}   (sd = {res.objective ** 0.5:.3%})\")\nfor i in range(4):\n    if held[i] > 0.5:\n        print(f\"  asset {i}  {weights[i]:6.2%}\")\nprint(\"compare the QP example: fewer holdings costs you diversification.\")\n",
  },
  {
    id: "NLP",
    label: "NLP \u2014 pressure vessel",
    title: "Nonlinear \u2014 pressure vessel",
    code: "# NONLINEAR PROGRAM -- a pressure vessel, continuous and nonconvex.\n# Minimize the surface area of a cylinder with hemispherical ends at a\n# fixed volume. Smooth nonlinear objective and constraint.\nimport discopt.modeling as dm\n\nm = dm.Model(\"pressure vessel\")\n\nr = m.continuous(\"r\", lb=0.2, ub=3.0)     # radius, m\nL = m.continuous(\"L\", lb=0.0, ub=20.0)    # length of the cylindrical section, m\n\npi = 3.141592653589793\n\n# Cylinder wall + two hemispherical caps.\narea = 2 * pi * r * L + 4 * pi * r * r\nm.minimize(area)\n\n# Fixed internal volume of 8 m^3 -- the nonconvex equality that makes this\n# a genuine NLP rather than a convex program.\nm.subject_to(pi * r * r * L + (4.0 / 3.0) * pi * r * r * r == 8.0, name=\"volume\")\n\nres = m.solve()\n\nprint(res.status)\nprint(f\"area   = {res.objective:.4f} m^2\")\nprint(f\"r      = {res.value(r):.4f} m\")\nprint(f\"L      = {res.value(L):.4f} m\")\nprint()\nprint(\"The optimum drives L -> 0: a sphere is the minimum-area shape for a\")\nprint(\"given volume, and a sphere of 8 m^3 has r = (3*8/(4*pi))**(1/3) =\",\n      f\"{(3 * 8 / (4 * pi)) ** (1 / 3):.4f} m\")\n",
  },
  {
    id: "MINLP",
    label: "MINLP \u2014 reactor network",
    title: "Mixed-integer nonlinear \u2014 reactor network",
    code: "# MIXED-INTEGER NONLINEAR PROGRAM -- the class discopt is built for.\n# A small process network: pick which reactors to build (binary) and how\n# hard to run them (continuous), where conversion is nonlinear in load.\n#\n# discopt solves this to GLOBAL optimality with spatial branch and bound,\n# and reports a certified bound -- not just a local solution.\nimport discopt.modeling as dm\n\nm = dm.Model(\"reactor network\")\n\nbuild = m.binary(\"build\", shape=(3,))\nload = m.continuous(\"load\", shape=(3,), lb=0.0, ub=1.0)\n\nfixed = [55.0, 38.0, 72.0]       # $k capital to build\nrate = [90.0, 65.0, 120.0]       # t/yr at full load\n# Operating cost grows faster than throughput: running one unit hard is\n# more expensive than sharing the load, which is what makes this nonconvex.\nopcost = [40.0, 33.0, 46.0]\n\nm.minimize(\n    dm.sum([fixed[i] * build[i] for i in range(3)])\n    + dm.sum([opcost[i] * load[i] * load[i] for i in range(3)])\n)\n\n# Conversion falls off as a unit is pushed: rate * load * (1 - 0.25*load).\nproduction = dm.sum([rate[i] * load[i] * (1 - 0.25 * load[i]) for i in range(3)])\nm.subject_to(production >= 150.0, name=\"demand\")\n\nfor i in range(3):\n    m.subject_to(load[i] <= build[i], name=f\"link{i}\")\n\nres = m.solve()\n\nprint(res.status, f\"in {res.node_count} nodes\")\nprint(f\"cost  = ${res.objective:,.2f}k\")\nprint(f\"bound = ${res.bound:,.2f}k     gap = {res.gap:.2e}\"\n      f\"   certified: {res.gap_certified}\")\nprint()\nbuilt, loads = res.value(build), res.value(load)\nfor i in range(3):\n    if built[i] > 0.5:\n        ld = loads[i]\n        print(f\"  BUILD unit {i}  load {ld:5.1%}  ->  {rate[i] * ld * (1 - 0.25 * ld):6.2f} t/yr\")\n    else:\n        print(f\"  --    unit {i}\")\n",
  },
];
