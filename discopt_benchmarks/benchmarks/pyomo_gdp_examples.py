"""Pyomo's ``examples/gdp`` models as benchmark instances (#1775 G0.5).

The models are vendored verbatim under ``discopt_benchmarks/data/pyomo_gdp_examples``
(Pyomo tag 6.10.0; see its ``PROVENANCE.md``). They are not a Python package and do
not use one entry-point convention: some expose ``build_model()``, some a differently
named builder, some an ``AbstractModel`` that needs a ``.dat`` file, some a dataset
dictionary that selects among several instances. :data:`CATALOG` names, for every
panel instance, the file, the builder and its arguments, so the instance set is
explicit rather than discovered.

Instances are named ``pyomo.<name>``: GDPlib has its own ``jobshop``,
``batch_processing`` and ``disease_model``, and those are different models.

Two examples are deliberately absent, each for a stated reason:

* ``jobshop-nodisjuncts.py`` has no disjunction, so it is not a GDP instance;
* ``eight_process/eight_proc_verbose_model.py`` does not build under Pyomo 6.10:
  it adds two disjunct ``indicator_var`` Booleans algebraically
  (``TypeError: unsupported operand type(s) for +: 'AutoLinkedBooleanVar' and
  'AutoLinkedBooleanVar'``). ``eight_proc_model.py`` is the same flowsheet.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

#: The vendored directory.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "pyomo_gdp_examples"

PREFIX = "pyomo."


@dataclass(frozen=True)
class Entry:
    """How to build one instance: ``module`` (a path under :data:`DATA_DIR`),
    ``builder`` (an attribute of it), its keyword arguments, and, for an
    ``AbstractModel``, the ``.dat`` file it is instantiated with. A ``builder`` of
    ``None`` with a ``dat`` means the module defines a module-level ``model``."""

    module: str
    builder: str | None
    kwargs: dict[str, Any] = field(default_factory=dict)
    dat: str | None = None
    #: ``kwargs`` values given as ``(dict attribute, key)`` are looked up in the module.
    lookup: dict[str, tuple[str, str]] = field(default_factory=dict)


def _layout(module: str, builder: str, table: str, arg: str, keys: list[str], stem: str):
    return {f"{stem}_{k.lower()}": Entry(module, builder, lookup={arg: (table, k)}) for k in keys}


CATALOG: dict[str, Entry] = {
    "simple1": Entry("simple1.py", "build_model"),
    "simple2": Entry("simple2.py", "build_model"),
    "simple3": Entry("simple3.py", "build_model"),
    "jobshop_small": Entry("jobshop.py", "build_model", dat="jobshop-small.dat"),
    "jobshop": Entry("jobshop.py", "build_model", dat="jobshop.dat"),
    "batch_processing": Entry("batchProcessing.py", "build_model", dat="batchProcessing1.dat"),
    # stickies.build_model() instantiates its own .dat and returns the instance.
    "stickies": Entry("stickies.py", "build_model"),
    "med_term_purchasing": Entry(
        "medTermPurchasing_Literal.py", "build_model", dat="medTermPurchasing_Literal_Chull.dat"
    ),
    "disease_model": Entry("disease_model.py", "build_model"),
    "eight_process": Entry("eight_process/eight_proc_model.py", "build_eight_process_flowsheet"),
    "eight_process_logical": Entry(
        "eight_process/eight_proc_logical.py", "build_eight_process_flowsheet"
    ),
    "nine_process": Entry("nine_process/small_process.py", "build_model"),
    "nine_process_nonexclusive": Entry("nine_process/small_process.py", "build_nonexclusive_model"),
    **_layout(
        "circles/circles.py",
        "build_model",
        "circles_model_examples",
        "data",
        ["Circles2D3", "Circles2D3_modified", "Circles3D4"],
        "circles",
    ),
    **_layout(
        "farm_layout/farm_layout.py",
        "build_model",
        "farm_layout_model_examples",
        "params",
        ["FLay02", "FLay03", "FLay04", "FLay05", "FLay06", "FLay03_alt_1", "FLay03_alt_2"],
        "farm_layout",
    ),
    **_layout(
        "constrained_layout/cons_layout_model.py",
        "build_constrained_layout_model",
        "constrained_layout_model_examples",
        "params",
        ["CLay0203", "CLay0204", "CLay0205", "CLay0303", "CLay0304", "CLay0305"],
        "constrained_layout",
    ),
    "ex1_lee": Entry("small_lit/ex1_Lee.py", "build_model"),
    "nonconvex_hen": Entry("small_lit/nonconvex_HEN.py", "build_gdp_model"),
    "ex_633_trespalacios": Entry("small_lit/ex_633_trespalacios.py", "build_simple_nonconvex_gdp"),
    "basic_step": Entry("small_lit/basic_step.py", "build_gdp_model"),
    "contracts_problem": Entry("small_lit/contracts_problem.py", "build_model"),
    "strip_packing_concrete": Entry(
        "strip_packing/strip_packing_concrete.py", "build_rect_strip_packing_model"
    ),
    "strip_packing_8rect": Entry(
        "strip_packing/strip_packing_8rect.py", "build_rect_strip_packing_model"
    ),
    "strip_packing_4": Entry("strip_packing/stripPacking.py", None, dat="stripPacking_4BigM.dat"),
    "strip_packing_12": Entry(
        "strip_packing/stripPacking.py", None, dat="stripPacking_12Chull.dat"
    ),
    "two_rxn_lee": Entry("two_rxn_lee/two_rxn_model.py", "build_model"),
}

#: Pyomo's own expected values, from ``pyomo/contrib/gdpopt/tests/test_gdpopt.py``
#: (spec §8). **Cross-checks, not oracles**: they are what Pyomo's tests assert of
#: GDPopt runs, not values a global solver certified with a feasibility-verified
#: point. Pyomo's other strip-packing value, 13, is asserted for
#: ``strip_packing_concrete`` with two logical constraints the test adds, so it is not
#: a property of any catalog instance and is not listed.
PYOMO_EXPECTED: dict[str, tuple[float, float]] = {
    # name: (value, absolute tolerance), as the Pyomo test asserts them
    "eight_process": (68.0, 1e-2),
    "strip_packing_concrete": (11.0, 1e-2),
    "constrained_layout_clay0203": (41573.0, 200.0),
    "ex_633_trespalacios": (4.46, 5e-3),
    "nonconvex_hen": (1.14385e5, 5e2),
}


def names() -> list[str]:
    """Every catalog instance, as panel names (``pyomo.<name>``)."""
    return [PREFIX + k for k in CATALOG]


def _module(relpath: str):
    """Load a vendored file by path, once. The examples are not a package; each
    gets a private module name so two files with the same stem cannot collide."""
    path = DATA_DIR / relpath
    if not path.is_file():
        raise FileNotFoundError(f"vendored Pyomo example missing: {path}")
    mod_name = "_pyomo_gdp_example_" + relpath.replace("/", "_").replace(".", "_").replace("-", "_")
    mod = sys.modules.get(mod_name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[mod_name]
            raise
    return mod


def builder(name: str) -> Callable[[], Any]:
    """A zero-argument builder for panel instance ``name`` (with or without the
    ``pyomo.`` prefix). It returns a fresh concrete Pyomo model on every call."""
    key = name[len(PREFIX) :] if name.startswith(PREFIX) else name
    entry = CATALOG[key]

    def build() -> Any:
        mod = _module(entry.module)
        kwargs = dict(entry.kwargs)
        for arg, (table, k) in entry.lookup.items():
            kwargs[arg] = getattr(mod, table)[k]
        m = mod.model if entry.builder is None else getattr(mod, entry.builder)(**kwargs)
        if entry.dat is not None:
            m = m.create_instance(str(DATA_DIR / Path(entry.module).parent / entry.dat))
        return m

    build.__name__ = f"build_{key}"
    return build


__all__ = ["CATALOG", "DATA_DIR", "Entry", "PREFIX", "PYOMO_EXPECTED", "builder", "names"]
