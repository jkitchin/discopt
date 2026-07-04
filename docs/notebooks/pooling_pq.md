# The pooling problem and the pq-formulation (`discopt.pooling`)

The **pooling problem** blends raw inputs of known quality through intermediate
*pools* into products with quality specifications, maximizing profit. Because the
blended quality at a pool is a *bilinear* function of the inflows, the problem is a
nonconvex NLP — even tiny instances are hard for global solvers when modeled
naively. The classic benchmark is Haverly's Pooling Problem 1, whose global
optimum is a profit of **400**.

The **pq-formulation** builder (Tawarmalani & Sahinidis) — which replaces raw
pool inflows with *proportion* variables and adds the redundant-but-tightening
Reformulation–Linearization (RLT) cuts that make the McCormick relaxation
provably at least as tight as the textbook *p*-formulation — now lives in the
standalone [discopt-apps](https://github.com/jkitchin/discopt-apps) plugin
(#431), mirroring the course extraction (#430). It is no longer bundled with the
core `discopt` package. The builder is pure modeling code over
`discopt.modeling.core`; no core changes are required.

## Installation

```bash
pip install discopt-apps
```

## Usage

```python
from discopt.pooling import (
    Input,
    Output,
    Pool,
    PoolingProblem,
    build_pq_formulation,
    haverly_hpp1,
)

problem = haverly_hpp1()
model = build_pq_formulation(problem)
result = model.solve()
```

See the
[discopt-apps documentation](https://github.com/jkitchin/discopt-apps)
for the full pq-formulation and examples.
