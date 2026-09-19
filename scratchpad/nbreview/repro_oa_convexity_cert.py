#!/usr/bin/env python
"""Repro: OA's rule-based convexity false negative, and the false certificate that
appears when the obvious fix (use_certificate=True) is applied.

Run:  python scratchpad/nbreview/repro_oa_convexity_cert.py
Exits non-zero if either finding fails to reproduce (so it cannot silently no-op).
"""

import discopt._relax.convexity as convpkg
import discopt.modeling as dm
import numpy as np
from discopt._relax.convexity import classify_oa_cut_convexity

n, K = 8, 3
mu = np.array([0.12, 0.10, 0.07, 0.03, 0.15, 0.08, 0.11, 0.06])
L = np.diag([0.10, 0.08, 0.06, 0.04, 0.12, 0.07, 0.09, 0.05])
L[1, 0] = 0.02
L[2, 0] = 0.01
L[3, 1] = 0.01
L[4, 0] = 0.03
L[5, 2] = 0.01
L[6, 1] = 0.02
L[7, 3] = 0.01
Sigma = L @ L.T

TRUE_OPT = 0.0020154286  # brute force over all C(8,3) supports, SLSQP per support


def build():
    m = dm.Model("portfolio")
    z = m.binary("z", shape=(n,))
    w = m.continuous("w", shape=(n,), lb=0.0, ub=0.4)
    m.minimize(
        dm.sum(lambda i: dm.sum(lambda j: Sigma[i, j] * w[i] * w[j], over=range(n)), over=range(n))
    )
    m.subject_to(dm.sum(lambda i: w[i], over=range(n)) == 1.0)
    m.subject_to(dm.sum(lambda i: mu[i] * w[i], over=range(n)) >= 0.09)
    m.subject_to(dm.sum(z) <= K)
    for i in range(n):
        m.subject_to(w[i] >= 0.02 * z[i])
        m.subject_to(w[i] <= 0.40 * z[i])
    return m


print(f"min eigenvalue of Sigma = {np.linalg.eigvalsh(Sigma).min():.6e}  (Sigma = L L^T, so PSD)")

# --- Finding 1: the rule-based classifier reports the convex objective non-convex.
rule = classify_oa_cut_convexity(build()).objective_is_convex
cert = classify_oa_cut_convexity(build(), use_certificate=True).objective_is_convex
print(f"\nobjective_is_convex: rule-based={rule}  certificate={cert}")
assert rule is False and cert is True, "Finding 1 did not reproduce"

# --- Finding 2: forcing the certificate yields a false optimality certificate.
_orig = convpkg.classify_oa_cut_convexity


def _forced(model, *, use_certificate=False):
    return _orig(model, use_certificate=True)


results = {}
for label, patch in (("default (cert=False)", _orig), ("forced  (cert=True)", _forced)):
    convpkg.classify_oa_cut_convexity = patch
    r = build().solve()
    results[label] = r
    print(f"\n{label}:")
    print(f"  status={r.status}  objective={r.objective:.10f}  bound={r.bound:.10f}")
    cert_flag = getattr(r, "gap_certified", None)
    print(f"  gap={r.gap:.3e}  gap_certified={cert_flag}  nodes={r.node_count}")
    print(f"  excess over true optimum = {(r.objective - TRUE_OPT) / TRUE_OPT:+.3e} (relative)")
convpkg.classify_oa_cut_convexity = _orig

bad = results["forced  (cert=True)"]
rel = (bad.objective - TRUE_OPT) / TRUE_OPT
assert rel > 1e-4, f"Finding 2 did not reproduce (rel={rel:.3e})"
assert bad.bound < bad.objective - 1e-9, "expected bound below incumbent"
print(
    f"\nFinding 2 reproduced: incumbent {rel:.3e} above the true optimum "
    f"(> 1e-4 rel tol), reported status=optimal with gap=0 and gap_certified=True,\n"
    f"while its own bound ({bad.bound:.10f}) sits BELOW its incumbent "
    f"({bad.objective:.10f})."
)
print("\nBoth findings reproduced.")
