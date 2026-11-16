"""Issue #2223 cause B: gradient noise in OA cut rows decertified the in-house master.

On ``clay0303hfsg`` the convex-MINLP route's first Rust lazy master (304 x 99) holds
six entries below 1e-10 of their line in OA cut rows. #2107 relaxes such an entry
into its row's slack only while the row's total relaxed effect stays within an
absolute 1e-7; these sum to ~2e-7, so three stay, the #1296 gate withdraws the
certificate (``MilpTinyEntryDecert=1``, bound ``-inf``) and the route falls back to
the spatial path (measured: ~35 s, four master calls, every one ``bound=None``).

``DISCOPT_OA_MASTER_RELAX_CUT_NOISE`` (default ON since its #2223 panel; ``=0`` opts
out) marks the cut rows ``relaxable_ub_rows``:
their cap becomes ``1e-6 * max(|b_i|, 1)``. Measured with it: one master call,
``MilpNoiseEntriesRelaxedMasked=3``, no decertification, certified in ~4 s.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

DATA = Path(__file__).parent / "data" / "minlplib_nl"
CLAY_REF = 26669.1095724859


def test_relaxable_std_rows_maps_ub_rows_and_leaves_equalities_unmarked():
    from discopt.solvers.milp_simplex import _relaxable_std_rows

    A_ub = sp.csr_matrix(np.ones((3, 2)))
    b_ub = np.zeros(3)
    out = _relaxable_std_rows(np.array([False, True, True]), A_ub, b_ub, 5)
    assert out == [False, True, True, False, False]
    # Nothing marked: the driver's unmasked path, not an all-false mask.
    assert _relaxable_std_rows(np.zeros(3, dtype=bool), A_ub, b_ub, 5) is None
    assert _relaxable_std_rows(None, A_ub, b_ub, 5) is None
    with pytest.raises(ValueError, match="2 flags for 3 A_ub rows"):
        _relaxable_std_rows(np.array([True, False]), A_ub, b_ub, 5)


def test_relaxable_rows_length_is_checked_by_the_binding():
    from discopt._rust import solve_milp_lazy_csc_py
    from discopt.solvers.milp_simplex import _marshal_std_form

    std = _marshal_std_form(
        np.array([-1.0]), np.array([[1.0]]), np.array([1.0]), None, None, [(0, 1)], [1]
    )
    with pytest.raises(ValueError, match="1 flags for 2 rows"):
        solve_milp_lazy_csc_py(
            std.c,
            std.m + 1,
            std.n + std.m,
            std.col_ptr,
            std.row_idx,
            std.vals,
            std.b,
            std.lb,
            std.ub,
            std.int_cols,
            std.n,
            lambda x: None,
            relaxable_rows=[True],
        )


def test_noise_flag_defaults_on_and_opts_out(monkeypatch):
    from discopt.solvers.oa import _oa_master_relax_cut_noise_enabled

    monkeypatch.delenv("DISCOPT_OA_MASTER_RELAX_CUT_NOISE", raising=False)
    assert _oa_master_relax_cut_noise_enabled()
    monkeypatch.setenv("DISCOPT_OA_MASTER_RELAX_CUT_NOISE", "0")
    assert not _oa_master_relax_cut_noise_enabled()


def test_oa_cut_row_mask_marks_exactly_the_cut_block():
    from types import SimpleNamespace

    from discopt.solvers.oa import _oa_cut_row_mask

    master = SimpleNamespace(A_ub=sp.csr_matrix(np.ones((7, 3))), b_ub=np.zeros(7))
    mask = _oa_cut_row_mask(master, ["<=", "==", ">=", "<="], 3)
    # Three inequality linear rows first, then the three cut rows, then a trailing block.
    assert mask.tolist() == [False, False, False, True, True, True, False]
    with pytest.raises(ValueError, match="exceed"):
        _oa_cut_row_mask(master, ["<="] * 5, 3)


@pytest.mark.slow
@pytest.mark.correctness
def test_clay0303hfsg_certifies_on_the_route(monkeypatch):
    import discopt
    import discopt._rust as rust
    import discopt.solvers.milp_simplex as ms
    from discopt.modeling import from_nl

    monkeypatch.setenv("DISCOPT_OA_MASTER_RELAX_CUT_NOISE", "1")
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    counters: list[dict] = []
    orig = ms.solve_milp_with_lazy_cuts

    def counted(*args, **kwargs):
        rust.profile_reset_py()
        out = orig(*args, **kwargs)
        counters.append(dict(rust.profile_counters_py()))
        return out

    monkeypatch.setattr(ms, "solve_milp_with_lazy_cuts", counted)
    assert os.path.realpath(discopt.__file__).startswith(
        os.path.realpath(Path(__file__).parents[1] / "discopt")
    )
    # 60 s is a backstop over the ~4 s measured, not the quality bar.
    res = from_nl(str(DATA / "clay0303hfsg.nl")).solve(time_limit=60)
    # The probe fired: the route ran the in-house lazy master and the mask relaxed
    # entries the unmasked cap refuses (CLAUDE.md section 6).
    assert counters, "the Rust lazy master never ran"
    assert counters[0]["MilpNoiseEntriesRelaxedMasked"] > 0, counters[0]
    assert counters[0]["MilpTinyEntryDecert"] == 0, counters[0]
    route = str(res.algorithm_route)
    assert "lp_nlp_bb" in route and "fell back" not in route, route
    assert res.status == "optimal" and res.gap_certified, (res.status, route)
    assert res.objective == pytest.approx(CLAY_REF, rel=1e-6)
    assert res.bound is not None and res.bound <= CLAY_REF + 1e-6 * abs(CLAY_REF)
