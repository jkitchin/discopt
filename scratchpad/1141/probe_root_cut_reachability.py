"""Can `DISCOPT_ROOT_CUT_DEADLINE` be exercised on the in-repo corpus at all? (#1141)

The flag bounds `generate_root_cuts`, which `solver.py` runs only when the model is
convexity-CERTIFIED (`_model_is_convex`) on a top-level solve, and which returns
immediately when the model has no integer variables. Both gates are budget- and
route-independent, so they can be evaluated without solving anything.

A panel over instances that cannot reach the stage would measure nothing while
printing "0 violations" (CLAUDE.md §6), so this establishes the population first.
Prints an executed-check count.
"""
import pathlib, sys
import numpy as np
from discopt.modeling.core import from_nl, VarType
from discopt.solvers.oa import _classify_model_convexity_for_probe  # noqa: F401  (may not exist)
