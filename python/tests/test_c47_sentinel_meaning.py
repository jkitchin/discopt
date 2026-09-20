"""C-47: the `1e30` sentinel's MEANING must cross the Rust boundary.

`1e30` says both "this region is proven empty" (prune it) and "this node could
not be bounded" (do not prune it), and the two are indistinguishable from the
value alone. The tree-side guard refuses to install the second kind as a node's
lower bound; that is only possible if every Python producer states which kind it
is handing over.

Two failure modes, one per direction:

* an `import_results` call site left unwired defaults its sentinels to "not an
  exclusion", which is sound but **demotes a rigorous emptiness proof** and
  loses a real fathom — measured on `alan`, which went from
  ``optimal cert=True nodes=13`` to ``feasible cert=False nodes=15`` while the
  MIQP path was unwired;
* a producer that flags a *failure* as an exclusion reinstates the original
  false-certificate bug.

`test_every_import_results_call_site_passes_an_exclusion_mask` guards the first
mode structurally (it is the one that actually happened, twice), and the
`_solve_batch_pounce` tests pin the rigor rule on the path that needed a new
parameter to express it at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import discopt.solver as S
import numpy as np
import pytest

_SENTINEL = 1e29


@pytest.mark.correctness
def test_every_import_results_call_site_passes_an_exclusion_mask():
    """No `tree.import_results(...)` may leave the meaning unstated.

    The 6th argument is `sentinel_is_exclusion`. Omitting it is *sound* — the
    binding defaults it to all-False — but it silently demotes that path's
    rigorous emptiness proofs, so an unwired site is a certificate leak rather
    than a correctness one. Structural, because the leak is invisible in any
    single-path test: the MIQP site was missed by reading and only caught by a
    corpus panel.
    """
    src = Path(S.__file__).read_text()
    tree = ast.parse(src)

    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_results"
    ]
    # §6: the probe must prove it fired. A refactor that renames the call would
    # otherwise turn this test into a silent no-op that reports a pass.
    assert len(sites) >= 4, f"expected the four orchestrator call sites, found {len(sites)}"

    unwired = []
    for node in sites:
        has_kw = any(kw.arg == "sentinel_is_exclusion" for kw in node.keywords)
        if len(node.args) < 6 and not has_kw:
            unwired.append(node.lineno)
    assert not unwired, (
        "import_results call sites that do not state whether their sentinels are "
        f"exclusions (solver.py lines {unwired}); see C-47"
    )


def _convex_eval():
    """A genuinely convex NLP: Hessian [[2, 1], [1, 2]] is PSD, constraint linear."""
    pytest.importorskip("pounce")
    import discopt.modeling as dm
    from discopt.solver import _make_evaluator
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    m = dm.Model("c47_convex")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2 + x * y)
    m.subject_to(x + y >= 1)

    ev = _make_evaluator(m)
    cl, cu = _infer_constraint_bounds(ev)
    return ev, list(zip(cl.tolist(), cu.tolist()))


_OPTS = {"max_iter": 200, "max_wall_time": 30.0}
# x + y >= 1 is impossible with both variables pinned near -5.
_EMPTY_LB = [-5.0, -5.0]
_EMPTY_UB = [-4.9, -4.9]
# A box the optimum lies inside, so the node solves cleanly.
_FULL_LB = [-5.0, -5.0]
_FULL_UB = [5.0, 5.0]


def _run(boxes, convex):
    ev, cb = _convex_eval()
    excl = np.zeros(len(boxes), dtype=bool)
    _, rlb, _, _, _ = S._solve_batch_pounce(
        ev,
        [b[0] for b in boxes],
        [b[1] for b in boxes],
        list(range(len(boxes))),
        ev.n_variables,
        cb,
        _OPTS,
        convex=convex,
        excl_out=excl,
    )
    return rlb, excl


@pytest.mark.correctness
def test_batch_pounce_flags_a_convex_infeasible_node_as_an_exclusion():
    """``Infeasible_Problem_Detected`` on a *convex* node is a proof; flag it.

    Restoration converged to a local minimizer of the constraint violation with
    the violation still positive. That measure is convex here, so its local
    minimizer is global and the box really is empty. Note the mapped
    ``SolveStatus`` is ``ERROR``, deliberately (a caller without a convexity
    certificate may not read the code as a verdict) — so the rule has to read
    ``raw_status``, and a version that checked the mapped status would be a dead
    branch that silently passes nothing through.
    """
    rlb, excl = _run([(_EMPTY_LB, _EMPTY_UB)], convex=True)
    assert abs(rlb[0]) >= _SENTINEL, "the empty box must come back sentinelled"
    assert bool(excl[0]), "a convex locally-infeasible verdict must be an exclusion"


@pytest.mark.correctness
def test_batch_pounce_does_not_flag_a_solved_node():
    """A node that solves is not an exclusion, and must not be marked one."""
    rlb, excl = _run([(_FULL_LB, _FULL_UB)], convex=True)
    assert abs(rlb[0]) < _SENTINEL, "the feasible box must come back with an objective"
    assert not bool(excl[0])


@pytest.mark.correctness
def test_batch_pounce_does_not_flag_a_nonconvex_infeasible_node():
    """On a nonconvex node the same verdict is not a proof, so no exclusion.

    Same box, same solver, same raw code — only the claim about the model
    differs. The flag must track the *rigor* of the verdict, not the sentinel
    value and not the solver's opinion.
    """
    _, excl = _run([(_EMPTY_LB, _EMPTY_UB)], convex=False)
    assert not bool(excl[0]), "a nonconvex local infeasibility is not a proof"
