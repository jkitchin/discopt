"""T1.2 investigation: extract build_milp_relaxation rows per term family and see
how they depend on the box, to derive closed-form patch generators."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import scipy.sparse as sp
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def build(model, lb, ub):
    terms = classify_nonlinear_terms(model)
    relax, varmap = build_milp_relaxation(
        model, terms, DiscretizationState(), bound_override=(lb, ub)
    )
    A = np.asarray(sp.csr_matrix(relax._A_ub).todense(), dtype=np.float64)  # noqa: N806
    b = np.asarray(relax._b_ub, dtype=np.float64).ravel()
    bnds = np.asarray(relax._bounds, dtype=np.float64)
    return A, b, bnds, varmap


def show(name, make_model, boxes, fam_key):
    print(f"\n########## {name} ##########")
    for (lb, ub) in boxes:
        model = make_model()
        A, b, bnds, varmap = build(model, np.array(lb), np.array(ub))  # noqa: N806
        fam = varmap.get(fam_key, {})
        print(f"\n box lb={lb} ub={ub}  ncol={A.shape[1]}  {fam_key}={dict(fam)}")
        # aux columns = beyond the n original vars
        n = len(model._variables)
        print(f"   aux bounds (cols>={n}): {bnds[n:]}")
        # print rows touching each aux col
        for a in range(n, A.shape[1]):
            rows = [k for k in range(A.shape[0]) if abs(A[k, a]) > 1e-9]
            for k in rows:
                supp = {c: A[k, c] for c in range(A.shape[1]) if abs(A[k, c]) > 1e-9}
                print(f"   aux{a} row{k}: {supp}  <= {b[k]:.4f}")


# --- monomial x^3 (odd), x^4 (even) ---
def m_pow(p):
    def make():
        m = dm.Model()
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x**p)
        return m
    return make


# --- trilinear x*y*z ---
def m_tri():
    m = dm.Model()
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    z = m.continuous("z", lb=-10, ub=10)
    m.minimize(x * y * z)
    return m


boxes_pos = [([1.0], [4.0]), ([2.0], [5.0])]
boxes_signed = [([-2.0], [3.0]), ([-4.0], [1.0])]

show("x**3 positive box", m_pow(3), boxes_pos, "monomial")
show("x**3 sign-spanning box", m_pow(3), boxes_signed, "monomial")
show("x**4 positive box", m_pow(4), boxes_pos, "monomial")
show("x**4 sign-spanning box", m_pow(4), boxes_signed, "monomial")
show("trilinear x*y*z", m_tri, [([1.0, 1.0, 1.0], [4.0, 5.0, 6.0])], "trilinear")
