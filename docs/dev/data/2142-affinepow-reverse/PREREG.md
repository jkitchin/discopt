# #2142 Part B entry check — pre-registered 2026-11-05 before any run

Hypothesis (findings.md, "Hull"): cstr hull's gap to SCIP is the missing reverse step on the
20 eps-perspective `AffinePow` terms `w = ((1-1e-8) y + 1e-8)^-1` (kernel propagates them
forward only; SCIP emu with nonlinear propagation off reproduces discopt's 0.0940 @10k).

Instrument (§105 style, `b1_capture.py`, `b2_kernel.py`, `b3_scip.py`, `b4_compare.py`):
capture cstr hull node boxes (pre-propagation box + cutoff) from the deterministic base tree
(flag OFF), every 10th node of the first 1000. On each box compute
* K  = kernel `propagate_spec_py`, `DISCOPT_AFFINE_POW_REVERSE` unset,
* KR = the same with `DISCOPT_AFFINE_POW_REVERSE=1`,
* S  = SCIP 6.3 root propagation (`2142-propagation/scip_prop.py`, BASE_PARAMS),
each at the tree's round cap (15) and at 100000 rounds (converged).

Gap metric. Per box and arm A, excess(A) = sum over all spec columns j with finite S and
root width of clip((width_A,j - width_S,j) / rootwidth_j, 0, 1); a box SCIP proves empty
counts excess(A) = n_cols if A leaves it non-empty, 0 if A also empties it.
closure = (sum_b excess(K) - sum_b excess(KR)) / sum_b excess(K).

**Kill criterion (fixed):** primary closure (converged rounds) < 0.50 -> stop, record,
retire `DISCOPT_AFFINE_POW_REVERSE`. The 15-round closure and the emptiness counts are
reported alongside, not used for the decision.
