"""#875 item 3 entry experiment: is ``build_linear_context`` O(rows x n_vars)?

The issue's post-#878 attribution puts ``_classify_model_convexity ->
build_linear_context`` at **15.8 s** of a constant ~17 s overrun on
``watercontamination0202`` (106,711 vars / 107,209 rows). The owner's remaining-work
note says it "densifies one length-n row per constraint and keeps it - the same
pattern #878 replaced elsewhere with ``_any_linear_constraint_form``. Making it
sparse would cut the 15.8 s directly rather than budgeting around it, and unlike a
poll it is measurable."

That instance is not available here (no MINLPLib snapshot, no route to
minlplib.org), so this measures the *mechanism* the same way #878's own entry
experiment did: hold the number of constraints FIXED and vary only ``n_vars``. A
cost that scales with ``n_vars`` at a fixed row count can only come from the dense
per-row array, not from the bodies.

Kill criterion: if the cost is flat in ``n_vars``, the dense row is not the problem
and this is the wrong lever.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
from discopt._jax.convexity.linear_context import build_linear_context  # noqa: E402

N_ROWS = 400  # fixed: only n_vars varies


def build_model(n_vars: int) -> dm.Model:
    """``n_vars`` scalar variables, ``N_ROWS`` two-term linear rows.

    Every row touches exactly 2 variables, so the total nonzero count is 2*N_ROWS
    regardless of ``n_vars`` — any growth with ``n_vars`` is the dense row.
    """
    m = dm.Model(f"lc{n_vars}")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n_vars)]
    for r in range(N_ROWS):
        i = (2 * r) % n_vars
        j = (2 * r + 1) % n_vars
        m.subject_to(xs[i] + 2.0 * xs[j] <= 5.0)
    m.minimize(xs[0])
    return m


def main():
    print(f"{'n_vars':>8s} {'build_s':>10s} {'A_ub type':>14s} {'A_ub bytes':>13s} {'ratio':>7s}")
    prev = None
    for n_vars in (2_000, 8_000, 32_000, 128_000):
        m = build_model(n_vars)
        t0 = time.perf_counter()
        ctx = build_linear_context(m)
        dt = time.perf_counter() - t0
        A = ctx.A_ub
        kind = type(A).__name__
        nbytes = A.data.nbytes if hasattr(A, "data") and hasattr(A.data, "nbytes") else A.nbytes
        ratio = "-" if prev is None else f"{dt / prev:.2f}x"
        prev = dt
        print(f"{n_vars:>8d} {dt:>10.3f} {kind:>14s} {nbytes:>13,d} {ratio:>7s}")
        # Sanity: the rows must still say the same thing whatever the storage.
        dense = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
        assert dense.shape == (N_ROWS, n_vars), dense.shape
        assert int(np.count_nonzero(dense)) == 2 * N_ROWS, int(np.count_nonzero(dense))


if __name__ == "__main__":
    main()
