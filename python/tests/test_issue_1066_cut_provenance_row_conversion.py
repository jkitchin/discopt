"""#1066: the MIP-NLP cut-provenance ledger must not do per-element Python work.

Every generated master cut is recorded with its full coefficient row, and the row
is the FULL master width. Converting it one element at a time made recording the
cuts cost more than generating them: profiled on ``squfl020-150`` (3020 columns)
at the default 60 s limit, the element-wise generator alone accounted for
159,557,152 calls / 11.5 s tottime and ``add_row`` 20.8 s cumulative -- 31% of a
67 s solve.

The conversion itself is unchanged (``float64 -> float``), so this is
bound-neutral by construction (CLAUDE.md §5). These tests pin both halves: that
the values, types and hashes are identical to the element-wise conversion, and
that the per-element Python frame is gone -- the property that a future
"tidy-up" back to a comprehension would silently undo.
"""

import sys

import numpy as np
from discopt.solvers.oa import MIPNLPCutRecord, _float_tuple


def _elementwise(values) -> tuple[float, ...]:
    """The conversion this replaced. The reference for both value and cost."""
    return tuple(float(v) for v in values)


def _genexpr_frames(fn, arg) -> int:
    """Python generator frames entered while ``fn(arg)`` runs.

    ``sys.setprofile`` reports a ``call`` event every time a generator frame is
    entered or resumed, so this counts exactly the per-element Python work the
    fix removes. Counting frames -- not wall time -- keeps the assertion
    deterministic under load (CLAUDE.md §9).
    """
    seen = [0]

    def _hook(frame, event, _arg):
        if event == "call" and frame.f_code.co_name == "<genexpr>":
            seen[0] += 1

    sys.setprofile(_hook)
    try:
        fn(arg)
    finally:
        sys.setprofile(None)
    return seen[0]


def test_conversion_is_indistinguishable_from_the_elementwise_one():
    """Same values, same types, same hashes -- so same dedup keys, same tree."""
    rng = np.random.default_rng(1066)
    cases = [
        np.array([]),
        np.array([0.0]),
        np.array([1.0, 2.5, -3.0]),
        np.array([1e20, -0.0, 3.0, -1e-18]),
        rng.normal(size=257),
        rng.normal(size=(1, 64)),  # a 2-D row: reshape(-1) must flatten it
        [1, 2, 3],  # ints must widen to float, as float(v) did
        np.float64(7.0),  # a 0-d value is a one-element row
    ]
    checks = 0
    for case in cases:
        got = _float_tuple(case)
        want = _elementwise(np.asarray(case, dtype=np.float64).reshape(-1))
        assert got == want, case
        assert all(type(x) is float for x in got), case
        assert hash(got) == hash(want), case
        checks += 1
    assert checks == len(cases), "vacuous: not every shape was converted"


def test_no_python_frame_per_element():
    """The regression guard: the row is converted in C, not one element at a time.

    Before the fix this counted 3021 frames on a 3020-column row -- the shape of
    ``squfl020-150``'s master.
    """
    row = np.arange(3020, dtype=np.float64)
    reference = _genexpr_frames(_elementwise, row)
    assert reference > len(row), (
        "the counter measured nothing -- it must see the element-wise generator "
        f"it is the control for (saw {reference})"
    )
    assert _genexpr_frames(_float_tuple, row) == 0


def test_the_record_still_keys_and_dedups_on_the_row():
    """The ledger's dedup key is the converted row, so it must stay hashable and
    compare equal for two spellings of the same cut."""
    coeffs = np.array([1.5, -2.0, 0.0, 3.25])
    point = np.array([0.5, 0.25, 0.0, 1.0])
    a = MIPNLPCutRecord.from_row(
        "objective", coeffs, 4.0, global_valid=True, supporting_point=point
    )
    b = MIPNLPCutRecord.from_row(
        "objective", list(coeffs), 4.0, global_valid=True, supporting_point=list(point)
    )
    assert a.dedup_key == b.dedup_key
    assert len({a.dedup_key, b.dedup_key}) == 1
    assert a.coefficients == _elementwise(coeffs)
    assert a.supporting_point == _elementwise(point)
    assert a.violation == b.violation
