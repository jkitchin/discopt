"""Domain-specific envelope libraries derived with the symbolic engine.

Each module ships **hand-written JAX** relaxation closures (so the solver hot
path never imports SymPy) whose formulas were *derived* by
:mod:`discopt._jax.symbolic.envelope_deriver` and are *certified* equal to the
symbolic result in the tests. This realizes the design principle: SymPy derives,
we commit the JAX, tests prove equivalence.

Domains:
    * :mod:`~discopt._jax.symbolic.domains.gas` — gas-network terms
      (Weymouth ``f|f|`` pressure drop, signed-power flow).
"""
