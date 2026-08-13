"""Load a GDPlib model as a discopt ``Model`` (bigm reformulation, .nl round-trip).

Same pipeline the benchmark runner uses (``discopt.pyomo._writer.write_nl`` ->
``discopt.modeling.from_nl``), factored out so a probe can hold the discopt model
directly instead of going through ``SolverFactory``.
"""

from __future__ import annotations

import os
import tempfile


def load_gdplib(name: str, method: str = "bigm"):
    """Build GDPlib model *name*, apply ``gdp.<method>``, return a discopt Model."""
    import discopt.modeling as dm
    from benchmarks.gdplib_runner import discover_models
    from discopt.pyomo import _writer
    from pyomo.core import TransformationFactory

    specs = {s.name: s for s in discover_models()}
    if name not in specs:
        raise KeyError(f"no gdplib model {name!r}; have {sorted(specs)}")
    pyomo_model = specs[name].builder()
    TransformationFactory(f"gdp.{method}").apply_to(pyomo_model)

    workdir = tempfile.mkdtemp(prefix="issue1004_")
    nl_path = os.path.join(workdir, f"{name}.nl")
    cols, rows, _elim = _writer.write_nl(pyomo_model, nl_path)
    if not cols:
        raise RuntimeError(f"{name}: zero-variable model")
    return dm.from_nl(nl_path)
