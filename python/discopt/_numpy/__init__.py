"""Pure-numpy McCormick relaxation compilation backend.

Compiles a Model's McCormick relaxation to plain numpy callables
(``discopt._numpy.relaxation_compiler``), avoiding JAX trace/compile overhead.
Unsupported ops raise ``NotImplementedError`` and the caller falls back to the
JAX path.

The former scipy/SLSQP relaxation NLP solver (``nlp_solver.py``) and its
dispatch from ``discopt._jax.mccormick_nlp`` were removed with HiGHS in issue
#356; the default relaxation NLP path is POUNCE.
"""
