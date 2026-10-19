# Provenance

Verbatim copy of `examples/gdp/` from the Pyomo repository
(https://github.com/Pyomo/pyomo), vendored for the `gdp_parity` benchmark panel
(discopt #1775, task G0.5).

- Tag: `6.10.0`
- Commit: `2ebd0873e63d60f563fe30ba72548b366e4976f9`
- License: 3-clause BSD; the full text is in `LICENSE.md`, and every `.py` file
  keeps its Pyomo copyright header.

No file is modified. Nothing in this directory is imported by discopt itself; the
benchmark loader `discopt_benchmarks/benchmarks/pyomo_gdp_examples.py` loads the
builders by file path, and the catalog there says which builder and data file make
each panel instance. To refresh, copy `examples/gdp/` from a new tag over this
directory and update the tag and commit above.
