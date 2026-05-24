"""Pure-numpy McCormick relaxation backend.

A drop-in alternative to the JAX McCormick path. Avoids JAX trace/compile
overhead on small B&B nodes by evaluating relaxations directly in numpy
and solving the convex NLP via scipy.optimize.minimize.

The dispatcher in ``discopt._jax.mccormick_nlp`` routes small instances
here when all expression ops are supported. Unsupported ops raise
``NotImplementedError`` and the caller falls back to the JAX path.
"""
