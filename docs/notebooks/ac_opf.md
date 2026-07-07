# AC optimal power flow (`discopt.opf`)

The **optimal power flow (OPF)** problem chooses generator setpoints and bus
voltages to meet electrical demand at minimum generation cost, subject to the
full nonlinear AC power-flow (Kirchhoff) equations. Because the power injections
are quadratic in the bus voltages, AC-OPF is a nonconvex
**quadratically-constrained quadratic program (QCQP)** — the canonical benchmark
for relaxation and global optimization research in power systems.

The rectangular-coordinate AC-OPF builder now lives in the standalone
[discopt-apps](https://github.com/jkitchin/discopt-apps) plugin (#431),
mirroring the course extraction (#430). It is no longer bundled with the core
`discopt` package. The builder is pure modeling code over
`discopt.modeling.core`, so the spatial branch-and-bound solver in core handles
the resulting `Model` as an ordinary QCQP — no core changes are required.

## Installation

```bash
pip install discopt-apps
```

## Usage

```python
from discopt.opf import (
    ACOPF,
    Bus,
    Line,
    admittance_matrix,
    build_ac_opf_rectangular,
    two_bus_example,
)

acopf = two_bus_example()
model = build_ac_opf_rectangular(acopf)
result = model.solve()
```

See the
[discopt-apps documentation](https://github.com/jkitchin/discopt-apps)
for the full AC-OPF formulation and examples.
