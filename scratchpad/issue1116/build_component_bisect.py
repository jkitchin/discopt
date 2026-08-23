"""#1116 stage 3: WHICH part of the very first root LP differs between two runs?

Usage: python -u build_component_bisect.py <instance-stem> <max_nodes> [n_calls]

``lp_seam_bisect.py`` established that ``first_differing_INPUT = 0`` — the FIRST
``MilpRelaxationModel.solve`` of the run is already handed a different problem in
two repetitions of the same solve in the same process. So the divergence is in
the relaxation BUILD (or in the presolve/FBBT box it is built over), not in the
simplex and not in the search.

This probe splits that first LP (and the next few) into its components — ``c``,
``A_ub``, ``b_ub``, ``bounds``, ``integrality`` — hashes each separately, and for
any component that differs reports the first differing entry with both values and
their ULP distance. A last-bit difference means order-of-summation; a large or
structural difference (different shape, different nnz) means a different amount
of work was done.

Per-rep progress (§10), module assertion (§8), executed-comparison count with a
non-zero exit when it is zero (§6), no swallowed exceptions (§7).
"""

import hashlib
import sys

import discopt
import numpy as np
import scipy.sparse as sp
from discopt._relax.milp_relaxation import MilpRelaxationModel
from discopt.modeling.core import from_nl

print(f"discopt.__file__={discopt.__file__}", flush=True)

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem = sys.argv[1]
max_nodes = int(sys.argv[2])
N_CALLS = int(sys.argv[3]) if len(sys.argv) > 3 else 5


def _flat(p):
    """Component -> (label, dense 1-D float view) for hashing and diffing."""
    if p is None:
        return "none", np.zeros(0)
    if sp.issparse(p):
        m = sp.csr_matrix(p)
        return (
            f"sparse{m.shape}nnz={m.nnz}",
            np.concatenate(
                [
                    m.indptr.astype(np.float64),
                    m.indices.astype(np.float64),
                    np.asarray(m.data, dtype=np.float64),
                ]
            ),
        )
    a = np.asarray(p, dtype=np.float64).ravel()
    return f"dense{a.shape}", a


def _hash(a: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=10).hexdigest()


COMPONENTS = ("c", "A_ub", "b_ub", "bounds", "integrality")

records: list[list[tuple[str, str, np.ndarray]]] = []
_cur: list[tuple[str, str, np.ndarray]] = []
_orig = MilpRelaxationModel.solve


def _traced(self, *a, **kw):
    if len(_cur) < N_CALLS * len(COMPONENTS):
        raw = (
            self._c,
            self._A_ub,
            self._b_ub,
            np.asarray(self._bounds, dtype=np.float64) if self._bounds else None,
            self._integrality,
        )
        for name, p in zip(COMPONENTS, raw):
            label, flat = _flat(p)
            _cur.append((name, f"{label}|{_hash(flat)}", flat))
    return _orig(self, *a, **kw)


MilpRelaxationModel.solve = _traced

for rep in range(2):
    _cur = []
    model = from_nl(NL.format(stem))
    r = model.solve(max_nodes=max_nodes)
    records.append(_cur)
    print(
        f"rep={rep} captured={len(_cur) // len(COMPONENTS)} calls "
        f"nodes={r.node_count} bound={r.bound!r} objective={r.objective!r}",
        flush=True,
    )

A, B = records
comparisons = 0
for k in range(min(len(A), len(B))):
    name, sig_a, arr_a = A[k]
    _, sig_b, arr_b = B[k]
    call = k // len(COMPONENTS)
    comparisons += 1
    if sig_a == sig_b:
        continue
    print(f"CALL {call} component {name}: DIFFERS  rep0={sig_a}  rep1={sig_b}", flush=True)
    if arr_a.shape != arr_b.shape:
        print(f"    shapes differ: {arr_a.shape} vs {arr_b.shape}", flush=True)
        continue
    idx = np.flatnonzero(arr_a != arr_b)
    print(f"    differing entries: {idx.size} of {arr_a.size}", flush=True)
    for j in idx[:5]:
        x, y = float(arr_a[j]), float(arr_b[j])
        ulps = (
            abs(int(np.asarray(x).view(np.int64)) - int(np.asarray(y).view(np.int64)))
            if np.isfinite(x) and np.isfinite(y) and np.sign(x) == np.sign(y)
            else -1
        )
        print(f"    [{j}] {x!r} vs {y!r}   ulps={ulps}", flush=True)

print(f"comparisons={comparisons}", flush=True)
if comparisons == 0:
    print("PROBE FIRED NOTHING", flush=True)
    sys.exit(2)
