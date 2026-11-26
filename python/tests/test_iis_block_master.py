"""The block-master IIS stack (``discopt._iis_blocks`` and its five flags).

The fixture is a small covering model with the structure the oracle arms on,
built from the rule, not from an instance: ``B`` interchangeable blocks, each
choosing a half-plane ``c . p + d >= 1`` with ``c`` in ``{-1,0,1}^2`` and ``d``
in ``[-4,4]`` that must contain none of the 9 inner points of a 5x5 grid; one
binary per (block, boundary point) that may be 1 only when the block's
half-plane contains the point; and one linking row per boundary point asking
for some block to contain it. The middle point of each side needs its own
half-plane, so 3 blocks are infeasible and 4 are feasible.

Every verdict of the oracle is compared with a reference that shares no code
with it: brute-force enumeration where every integrality is held, and the
engine's ordinary two-route probe otherwise. Each comparing test counts its
comparisons and fails when it made none.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest
import scipy.sparse as sp
from discopt import _iis_blocks as ib
from discopt import _iis_milp as im

INF = im._INF
BIG_M = 8
INNER = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)]
OUTER = [(x, y) for x in range(-2, 3) for y in range(-2, 3) if max(abs(x), abs(y)) == 2]
# A smaller geometry on the same path, for the tests that run a whole filter: a
# five-point inner cross, and the eight boundary points that are neither corners
# nor edge midpoints. Three blocks cannot cover it either (two cannot, four can),
# and the full-stack run arms the same steps (block proofs and points, orbit
# drops, an ordering-row cross-check that agrees) with 509 master decisions where
# the full geometry takes 1614: 2.0 s against 6.5 s on a dev machine, two
# interleaved repetitions each; the full geometry was 15.4 s on CI (#2348).
SMALL_INNER = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
SMALL_OUTER = [p for p in OUTER if 0 not in p and abs(p[0]) != abs(p[1])]
FLAGS = (
    "DISCOPT_IIS_BLOCK_MASTER",
    "DISCOPT_IIS_BLOCK_LEX",
    "DISCOPT_IIS_ORBIT_QX",
    "DISCOPT_IIS_BLOCK_CONFINED",
    "DISCOPT_IIS_IMPLIED_INT",
)


def cover_model(n_blocks: int, inner=None, outer=None):
    """``(A, row_lo, row_hi, col_lb, col_ub, is_int)`` of the fixture, plus the
    column layout ``(core, sat)``: ``core[b] = (c1, c2, d)``, ``sat[b][k]``.
    ``inner``/``outer`` replace the point sets (default ``INNER``/``OUTER``)."""
    INNER_, OUTER_ = (INNER if inner is None else inner), (OUTER if outer is None else outer)
    rows, lo, hi = [], [], []
    core = [(3 * b, 3 * b + 1, 3 * b + 2) for b in range(n_blocks)]
    n = 3 * n_blocks
    sat = []
    for b in range(n_blocks):
        sat.append(list(range(n, n + len(OUTER_))))
        n += len(OUTER_)
    for b in range(n_blocks):
        c1, c2, d = core[b]
        for x, y in INNER_:  # the half-plane contains no inner point
            rows.append({c1: x, c2: y, d: 1})
            lo.append(-INF)
            hi.append(0.0)
        for k, (x, y) in enumerate(OUTER_):  # z = 1 only if it contains the point
            rows.append({c1: x, c2: y, d: 1, sat[b][k]: -BIG_M})
            lo.append(1.0 - BIG_M)
            hi.append(INF)
    for k in range(len(OUTER_)):  # every boundary point is in some half-plane
        rows.append({sat[b][k]: 1 for b in range(n_blocks)})
        lo.append(1.0)
        hi.append(INF)
    ri = [i for i, r in enumerate(rows) for _ in r]
    ci = [j for r in rows for j in r]
    va = [float(v) for r in rows for v in r.values()]
    A = sp.csr_matrix((va, (ri, ci)), shape=(len(rows), n))
    A.eliminate_zeros()
    lb = np.zeros(n)
    ub = np.ones(n)
    for c1, c2, d in core:
        lb[[c1, c2]], ub[[c1, c2]] = -1.0, 1.0
        lb[d], ub[d] = -4.0, 4.0
    return (A, np.array(lo), np.array(hi), lb, ub, np.ones(n, dtype=bool)), (core, sat)


def brute_force_infeasible(n_blocks, rlo, rhi, layout) -> bool:
    """Ground truth with every integrality and bound held: enumerate each
    block's ``(c1, c2, d)`` and give each binary its best value."""
    core, sat = layout
    per_block = 9 + len(OUTER)
    link0 = n_blocks * per_block
    covers = []
    for b in range(n_blocks):
        base = b * per_block
        opts = set()
        for c1, c2, d in itertools.product((-1, 0, 1), (-1, 0, 1), range(-4, 5)):
            if any(
                rhi[base + t] < INF and c1 * x + c2 * y + d > rhi[base + t]
                for t, (x, y) in enumerate(INNER)
            ):
                continue
            z = []
            for k, (x, y) in enumerate(OUTER):
                r = base + 9 + k
                # active row: z <= (c.p + d - (1 - M)) / M, z binary
                z.append(1 if rlo[r] <= -INF or c1 * x + c2 * y + d >= 1 else 0)
            opts.add(tuple(z))
        covers.append(opts)
    need = [k for k in range(len(OUTER)) if rlo[link0 + k] > -INF]
    for pick in itertools.product(*covers):
        if all(any(p[k] for p in pick) for k in need):
            return False
    return True


def probe_arrays(model, rows_off=(), ints_off=(), bounds_off=()):
    A, rlo, rhi, lb, ub, ii = model
    rlo, rhi, lb, ub, ii = rlo.copy(), rhi.copy(), lb.copy(), ub.copy(), ii.copy()
    for r in rows_off:
        rlo[r], rhi[r] = -INF, INF
    for j in ints_off:
        ii[j] = False
    for j in bounds_off:
        lb[j], ub[j] = -INF, INF
    return rlo, rhi, lb, ub, ii


def _stop() -> float:
    """A deadline far above any probe here (a backstop, not a quality bar)."""
    return time.perf_counter() + 60.0


def engine_for(model, engine):
    A, rlo, rhi, lb, ub, ii = model
    groups = {i: np.array([i], dtype=np.int64) for i in range(rlo.size)}
    return im._Engine(A, rlo, rhi, lb, ub, ii, groups, 60.0, True, engine=engine)


@pytest.fixture(scope="module")
def engine_name():
    return im._resolve_iis_engine(None)


def test_structure_is_read_from_the_matrix(engine_name):
    model, (core, sat) = cover_model(3)
    bm = ib.BlockMaster(*model, engine_name)
    assert bm.describe() == {
        "linking_rows": 16,
        "blocks": 3,
        "core_columns": 9,
        "satellites": 48,
        "canonical_blocks": 3,
    }
    assert [sorted(b.cols) for b in bm.blocks] == [sorted(c) for c in core]


def test_models_outside_the_rule_are_refused(engine_name):
    # one row over binaries: a linking row and nothing else
    A = sp.csr_matrix(np.ones((1, 3)))
    with pytest.raises(ib.Unsupported):
        ib.BlockMaster(
            A,
            np.array([4.0]),
            np.array([INF]),
            np.zeros(3),
            np.ones(3),
            np.ones(3, bool),
            engine_name,
        )
    # non-integer data in a block row
    model, _ = cover_model(3)
    A2 = model[0].copy().tolil()
    A2[0, 0] = 0.5
    with pytest.raises(ib.Unsupported):
        ib.BlockMaster(A2.tocsr(), *model[1:], engine_name)
    # a continuous core column
    ii = model[5].copy()
    ii[0] = False
    with pytest.raises(ib.Unsupported):
        ib.BlockMaster(*model[:5], ii, engine_name)


@pytest.mark.parametrize("n_blocks,infeasible", [(2, True), (3, True), (4, False)])
def test_full_model_verdict_and_arming(engine_name, n_blocks, infeasible):
    model, layout = cover_model(n_blocks)
    bm = ib.BlockMaster(*model, engine_name)
    verdict, x = bm.decide(*model[1:], stop=_stop())
    assert verdict is infeasible
    assert brute_force_infeasible(n_blocks, model[1], model[2], layout) is infeasible
    if not infeasible:
        assert x is not None and bm.verify(x, *model[1:])
    # armed only when the master proves the whole model infeasible
    assert engine_for(model, engine_name).arm_blocks() is infeasible


def test_verdicts_match_brute_force_on_row_subsystems(engine_name):
    model, layout = cover_model(3)
    bm = ib.BlockMaster(*model, engine_name)
    rng = np.random.default_rng(0)
    m = model[1].size
    compared = n_inf = n_feas = 0
    for trial in range(60):
        k = int(rng.integers(1, 30)) if trial % 2 else int(rng.integers(1, 6))
        off = rng.choice(m, size=k, replace=False)
        arrs = probe_arrays(model, rows_off=off)
        verdict, x = bm.decide(*arrs, stop=_stop())
        truth = brute_force_infeasible(3, arrs[0], arrs[1], layout)
        assert verdict is truth, (trial, sorted(off.tolist()))
        if verdict is False:
            assert x is not None and bm.verify(x, *arrs)
            n_feas += 1
        else:
            n_inf += 1
        compared += 1
    assert compared == 60 and n_inf > 5 and n_feas > 5, (compared, n_inf, n_feas)


def test_verdicts_match_the_ordinary_probe_with_relaxed_columns(engine_name):
    """Relaxed integralities and dropped bounds (satellites; the unit-coefficient
    core column under the implied-integer rule), against the engine's two-route
    probe on the same set."""
    model, (core, sat) = cover_model(3)
    bm = ib.BlockMaster(*model, engine_name, implied_int=True)
    ref = engine_for(model, engine_name)
    rng = np.random.default_rng(1)
    m, n = model[1].size, model[3].size
    d_cols = [c[2] for c in core]
    relaxable = d_cols + [j for s in sat for j in s]
    compared = unsupported = 0
    for trial in range(40):
        rows_off = set(rng.choice(m, size=int(rng.integers(0, 8)), replace=False).tolist())
        ints_off = set(rng.choice(relaxable, size=int(rng.integers(1, 6)), replace=False).tolist())
        bounds_off = {j for j in ints_off if rng.random() < 0.5}
        arrs = probe_arrays(model, rows_off, ints_off, bounds_off)
        cons = set(range(m)) - rows_off
        ints = set(range(n)) - ints_off
        bnds = {(j, s) for j in range(n) for s in ("lower", "upper") if j not in bounds_off}
        try:
            verdict, x = bm.decide(*arrs, stop=_stop())
        except ib.Unsupported:
            unsupported += 1
            continue
        want, _ = ref.probe(cons, bnds, ints)
        assert want is not None, trial
        assert verdict is want, (trial, sorted(rows_off), sorted(ints_off), sorted(bounds_off))
        if verdict is False:
            assert x is not None and bm.verify(x, *arrs)
        compared += 1
    assert compared >= 20, (compared, unsupported)


def test_implied_integer_rule_is_behind_its_switch(engine_name):
    model, (core, _sat) = cover_model(3)
    d0, c0 = core[0][2], core[0][0]
    arrs = probe_arrays(model, ints_off=[d0])
    # max_verbatim=0: the enumeration alone, no block carried as its own rows
    with pytest.raises(ib.Unsupported):
        ib.BlockMaster(*model, engine_name, max_verbatim=0).decide(*arrs, stop=_stop())
    bm = ib.BlockMaster(*model, engine_name, implied_int=True, max_verbatim=0)
    assert (
        bm.decide(*arrs, stop=_stop())[0] is True
    )  # d has coefficient 1 in every row: implied integer
    # c has coefficients of magnitude 2: not implied, refused even with the rule on
    with pytest.raises(ib.Unsupported):
        bm.decide(*probe_arrays(model, ints_off=[c0]), stop=_stop())


def test_symmetry_is_verified_on_the_probed_subsystem(engine_name):
    model, (core, sat) = cover_model(3)
    bm = ib.BlockMaster(*model, engine_name)
    assert bm.classes(*model[1:]) == [[0, 1, 2]]
    idx, val = bm.lex_rows([[0, 1, 2]])
    assert len(idx) == 2 and all(v.sum() == 0 for v in val)
    # one inner row of block 1 gone: block 1 is no longer interchangeable
    one_row = probe_arrays(model, rows_off=[25 + 4])
    assert bm.classes(*one_row) == [[0, 2]]
    # the same row gone from every block: the symmetry is back
    orbit = probe_arrays(model, rows_off=[4, 25 + 4, 50 + 4])
    assert bm.classes(*orbit) == [[0, 1, 2]]
    # a linking row is fixed by every swap
    assert bm.classes(*probe_arrays(model, rows_off=[75])) == [[0, 1, 2]]
    # a relaxed column or a dropped bound in one block breaks it too
    assert bm.classes(*probe_arrays(model, ints_off=[core[2][2]])) == [[0, 1]]
    assert bm.classes(*probe_arrays(model, bounds_off=[sat[0][3]])) == [[1, 2]]


def test_ordering_rows_agree_with_the_master(engine_name):
    model, layout = cover_model(3)
    bm = ib.BlockMaster(*model, engine_name)
    rng = np.random.default_rng(2)
    compared = 0
    for trial in range(12):
        off = [] if trial == 0 else rng.choice(model[1].size, size=3, replace=False).tolist()
        if trial % 3 == 1:  # whole orbits, so the ordering rows stay in play
            off = [r % 25 + 25 * b for r in off for b in range(3) if r < 75]
        arrs = probe_arrays(model, rows_off=off)
        verdict, _ = bm.decide(*arrs, stop=_stop())
        lv, lx, k = ib.lex_decide(bm, *arrs, stop=time.perf_counter() + 60.0)
        if k == 0:
            assert lv is None
            continue
        assert lv is verdict, (trial, off)
        compared += 1
    assert compared >= 4, compared


def _arrays_arg(model):
    A, rlo, rhi, lb, ub, ii = model
    return (A, rlo, rhi, {i: np.array([i], dtype=np.int64) for i in range(rlo.size)}, lb, ub, ii)


def _independent_confirm(model, res, engine):
    """The answer through an engine with no oracle armed: both presolve routes."""
    eng = engine_for(model, engine)
    assert eng._blk is None
    return eng.confirm_infeasible(set(res.constraints), set(res.bounds), set(res.integrality))


DEFAULT_ON = {"master": True, "lex": False, "orbit": True, "confined": True, "implied_int": True}


def test_the_shipped_defaults_and_their_opt_outs(monkeypatch):
    """Four switches are default ON with ``=0`` as the opt-out; the ordering rows
    (``DISCOPT_IIS_BLOCK_LEX``) and nothing else are an opt-in (``=1``)."""
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    assert im._block_flags() == DEFAULT_ON
    for f in FLAGS:
        monkeypatch.setenv(f, "0")
    assert not any(im._block_flags().values())
    for f in FLAGS:
        monkeypatch.setenv(f, "1")
    assert all(im._block_flags().values())


def test_flags_opted_out_leave_the_engine_untouched(engine_name, monkeypatch):
    for f in FLAGS:
        monkeypatch.setenv(f, "0")
    assert not any(im._block_flags().values())
    model, _ = cover_model(2, SMALL_INNER, SMALL_OUTER)
    res = im.milp_iis_arrays(_arrays_arg(model), iis_engine=engine_name, total_time_limit=120.0)
    assert res.n_block_proofs == res.n_block_points == res.n_orbit_dropped == 0
    assert _independent_confirm(model, res, engine_name) is True


@pytest.mark.parametrize("order", ["ICB", "CIB"])
def test_full_stack_answer_is_confirmed_independently(engine_name, monkeypatch, order):
    for f in FLAGS:
        monkeypatch.setenv(f, "1")
    model, _ = cover_model(3, SMALL_INNER, SMALL_OUTER)
    res = im.milp_iis_arrays(
        _arrays_arg(model), iis_engine=engine_name, order=order, total_time_limit=120.0
    )
    assert res.complete
    assert res.n_block_proofs > 0 and res.n_block_points > 0
    assert res.n_orbit_dropped > 0
    assert res.n_lex_agree >= 1  # the answer's proof was cross-checked
    assert _independent_confirm(model, res, engine_name) is True
    # every certificate is a point of the answer minus its element (arithmetic)
    eng = engine_for(model, engine_name)
    ok = im._check_certificates(
        res.certificates,
        eng,
        set(res.constraints),
        res.bounds,
        set(res.integrality),
        1e-6,
    )
    assert len(ok) == len(res.certificates) > 0


@pytest.mark.parametrize("extra", (None,) + FLAGS[1:])
def test_each_flag_alone_gives_a_confirmed_answer(engine_name, monkeypatch, extra):
    """The two-block fixture: the same code path as three blocks at a fraction of
    the time (five three-block runs took 154 s on CI)."""
    model, _ = cover_model(2, SMALL_INNER, SMALL_OUTER)
    for f in FLAGS:
        monkeypatch.setenv(f, "0")
    monkeypatch.setenv(FLAGS[0], "1")
    if extra:
        monkeypatch.setenv(extra, "1")
    res = im.milp_iis_arrays(_arrays_arg(model), iis_engine=engine_name, total_time_limit=120.0)
    assert res.n_block_proofs > 0
    assert _independent_confirm(model, res, engine_name) is True


def test_feasible_model_is_not_armed_and_still_refused(engine_name, monkeypatch):
    for f in FLAGS:
        monkeypatch.setenv(f, "1")
    model, _ = cover_model(4)
    with pytest.raises(ValueError, match="feasible"):
        im.milp_iis_arrays(_arrays_arg(model), iis_engine=engine_name, total_time_limit=120.0)


def test_a_refused_block_enters_the_master_as_its_own_rows(engine_name):
    """A relaxed ``d`` next to a relaxed satellite of the same row is outside the
    implied-integer rule. The block is then carried verbatim, for at most
    ``max_verbatim`` blocks; every verdict is compared with the ordinary probe."""
    model, (core, sat) = cover_model(3)
    m, n = model[1].size, model[3].size
    ref = engine_for(model, engine_name)
    rng = np.random.default_rng(2)
    compared = n_inf = n_feas = 0
    for trial in range(30):
        nb = 1 + trial % 2  # blocks with relaxed columns
        ints_off: set[int] = set()
        for b in range(nb):
            ints_off.add(core[b][2])
            ints_off.update(
                rng.choice(sat[b], size=int(rng.integers(1, 5)), replace=False).tolist()
            )
            if trial % 3 == 0:
                ints_off.add(core[b][0])
        rows_off = set(rng.choice(m, size=int(rng.integers(0, 6)), replace=False).tolist())
        arrs = probe_arrays(model, rows_off, ints_off)
        with pytest.raises(ib.Unsupported):
            ib.BlockMaster(*model, engine_name, implied_int=True, max_verbatim=0).decide(
                *arrs, stop=_stop()
            )
        bm = ib.BlockMaster(*model, engine_name, implied_int=True)
        verdict, x = bm.decide(*arrs, stop=_stop())
        assert bm.n_verbatim == 1
        bnds = {(j, sd) for j in range(n) for sd in ("lower", "upper")}
        want, _ = ref.probe(set(range(m)) - rows_off, bnds, set(range(n)) - ints_off)
        assert want is not None, trial
        assert verdict is want, (trial, sorted(rows_off), sorted(ints_off))
        if verdict is False:
            assert x is not None and bm.verify(x, *arrs)
            n_feas += 1
        else:
            n_inf += 1
        compared += 1
    assert compared == 30 and n_inf > 0 and n_feas > 0, (compared, n_inf, n_feas)
    # more relaxed blocks than the cap: no verdict from the oracle
    arrs = probe_arrays(model, ints_off=[c[0] for c in core])
    with pytest.raises(ib.Unsupported):
        ib.BlockMaster(*model, engine_name, implied_int=True).decide(*arrs, stop=_stop())


def hole_model(n_holes: int, n_pigeons: int):
    """Pigeonhole: ``x[p, h]`` binary, ``sum_h x[p, h] >= 1`` per pigeon (the
    linking rows) and ``x[p, h] + x[q, h] <= 1`` per hole and pair (the block of
    hole ``h``). No satellites: the core columns lie in the linking rows."""
    col = {(p, h): h * n_pigeons + p for p in range(n_pigeons) for h in range(n_holes)}
    rows, lo, hi = [], [], []
    for h in range(n_holes):
        for p, q in itertools.combinations(range(n_pigeons), 2):
            rows.append({col[p, h]: 1, col[q, h]: 1})
            lo.append(-INF)
            hi.append(1.0)
    for p in range(n_pigeons):
        rows.append({col[p, h]: 1 for h in range(n_holes)})
        lo.append(1.0)
        hi.append(INF)
    n = len(col)
    ri = [i for i, r in enumerate(rows) for _ in r]
    ci = [j for r in rows for j in r]
    A = sp.csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(rows), n))
    return A, np.array(lo), np.array(hi), np.zeros(n), np.ones(n), np.ones(n, dtype=bool)


def test_core_columns_may_lie_in_linking_rows(engine_name):
    model = hole_model(4, 5)
    bm = ib.BlockMaster(*model, engine_name)
    assert bm.describe() == {
        "linking_rows": 5,
        "blocks": 4,
        "core_columns": 20,
        "satellites": 0,
        "canonical_blocks": 4,
    }
    assert bm.decide(*model[1:], stop=_stop())[0] is True
    assert bm.classes(*model[1:]) == [[0, 1, 2, 3]]
    ok = hole_model(4, 4)
    verdict, x = ib.BlockMaster(*ok, engine_name).decide(*ok[1:], stop=_stop())
    assert verdict is False and x is not None
    # row subsystems, relaxed columns and dropped bounds against the ordinary probe
    ref = engine_for(model, engine_name)
    m, n = model[1].size, model[3].size
    rng = np.random.default_rng(3)
    compared = n_inf = n_feas = unsupported = 0
    for trial in range(60):
        rows_off = set(rng.choice(m, size=int(rng.integers(0, 4)), replace=False).tolist())
        ints_off = set(rng.choice(n, size=int(rng.integers(0, 3)), replace=False).tolist())
        bounds_off = {j for j in ints_off if rng.random() < 0.3}
        arrs = probe_arrays(model, rows_off, ints_off, bounds_off)
        try:
            verdict, x = bm.decide(*arrs, stop=_stop())
        except ib.Unsupported:
            unsupported += 1
            continue
        bnds = {(j, sd) for j in range(n) for sd in ("lower", "upper") if j not in bounds_off}
        want, _ = ref.probe(set(range(m)) - rows_off, bnds, set(range(n)) - ints_off)
        assert want is not None, trial
        assert verdict is want, (trial, sorted(rows_off), sorted(ints_off), sorted(bounds_off))
        if verdict is False:
            assert x is not None and bm.verify(x, *arrs)
            n_feas += 1
        else:
            n_inf += 1
        compared += 1
    assert compared >= 40 and n_inf > 3 and n_feas > 3, (compared, n_inf, n_feas, unsupported)


def test_stack_on_a_pigeonhole_model_matches_the_plain_filter(engine_name, monkeypatch):
    """Whole column orbits relaxed are outside the master's rule: the ordering
    rows' infeasibility claims go to the plain probe for their proof, and the
    frozen blocks take the free block's certificates through the block swap.
    The answer is no larger than the plain filter's (it was: 495 against 479 on
    ``hole9`` while a claim without a proof left its elements undecided)."""
    model = hole_model(4, 5)
    for f in FLAGS:
        monkeypatch.setenv(f, "0")
    off = im.milp_iis_arrays(_arrays_arg(model), iis_engine=engine_name, total_time_limit=120.0)
    for f in FLAGS:
        monkeypatch.setenv(f, "1")
    on = im.milp_iis_arrays(_arrays_arg(model), iis_engine=engine_name, total_time_limit=120.0)
    assert on.complete and off.complete
    assert on.n_block_proofs > 0
    assert on.n_lex_unadopted == 0
    assert on.n_confined_carried > 0
    assert _independent_confirm(model, on, engine_name) is True

    def size(r):
        return len(r.constraints) + len(r.bounds) + len(r.integrality)

    assert size(on) <= size(off), (size(on), size(off))


def test_orbit_mates_follow_the_current_subsystem(engine_name):
    model, (core, sat) = cover_model(3)
    eng = engine_for(model, engine_name)
    assert eng.arm_blocks() is True
    m, n = model[1].size, model[3].size
    groups = {i: np.array([i], dtype=np.int64) for i in range(m)}
    cons, ints = set(range(m)), set(range(n))
    bnds = {(j, sd) for j in range(n) for sd in ("lower", "upper")}
    k = ("integrality", core[0][0])
    assert im._orbit_mates(eng, groups, cons, bnds, ints, k) == {
        ("integrality", core[1][0]),
        ("integrality", core[2][0]),
    }
    kb = ("bound", sat[0][3], "upper")
    assert im._orbit_mates(eng, groups, cons, bnds, ints, kb) == {
        ("bound", sat[1][3], "upper"),
        ("bound", sat[2][3], "upper"),
    }
    # one inner row of block 1 gone: block 1 is in no class, so it is nobody's mate
    assert im._orbit_mates(eng, groups, cons - {25 + 4}, bnds, ints, k) == {
        ("integrality", core[2][0])
    }
    assert (
        im._orbit_mates(eng, groups, cons - {25 + 4}, bnds, ints, ("integrality", core[1][0]))
        == set()
    )
