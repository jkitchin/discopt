"""Tests for the QPLIB reader (issue #830).

The ``.qplib`` format is line-oriented with **no section markers**: which blocks
are present depends on the three-character ``probtype`` code. Reading it with the
wrong conditioning does not raise -- it shifts the token stream and produces a
well-formed but wrong model. Every test here therefore checks parsed structure
against QPLIB's own ``instancedata.csv``, and the decisive one recomputes the
objective at the published reference point.

Fixtures live in ``data/qplib/`` and were chosen to cover the reader's
conditional branches; see the README there. Corpus-wide coverage is in
``test_qplib_corpus.py``.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest
from discopt.interfaces import qplib as qp

DATA = os.path.join(os.path.dirname(__file__), "data", "qplib")

FIXTURES = [
    "QPLIB_0031",
    "QPLIB_2967",
    "QPLIB_3385",
    "QPLIB_3496",
    "QPLIB_3562",
    "QPLIB_3814",
    "QPLIB_3815",
    "QPLIB_3852",
    "QPLIB_3871",
]


@pytest.fixture(scope="module")
def meta():
    with open(os.path.join(DATA, "instancedata.csv"), encoding="utf-8") as fh:
        return {r["name"]: r for r in csv.DictReader(fh)}


@pytest.fixture(scope="module")
def solu():
    return qp.read_solu(os.path.join(DATA, "qplib.solu"))


def _read(name):
    return qp.read_qplib(os.path.join(DATA, "qplib", f"{name}.qplib"))


@pytest.mark.unit
def test_fixtures_cover_every_conditional_branch():
    """The fixture set is only meaningful if it exercises each layout branch.

    Guards against a fixture being dropped and quietly taking a branch's only
    coverage with it.
    """
    obj, var, con = set(), set(), set()
    for name in FIXTURES:
        o, v, c = _read(name).probtype
        obj.add(o)
        var.add(v)
        con.add(c)
    # O=L omits the objective-quadratic block; V=B omits bounds *and* types;
    # V=C omits types only; C=N omits the ncons line itself; C=L omits the
    # constraint-quadratic block.
    assert {"L", "Q"} <= obj
    assert {"B", "C", "I", "M", "G"} <= var
    assert {"N", "L", "Q"} <= con


@pytest.mark.unit
@pytest.mark.parametrize("name", FIXTURES)
def test_structure_matches_instancedata(name, meta):
    """Parsed structure must agree with QPLIB's own metadata."""
    inst = _read(name)
    r = meta[name]
    assert inst.n_vars == int(r["nvars"])
    assert inst.n_cons == int(r["ncons"])
    assert inst.probtype == r["probtype"]
    assert inst.sense.startswith(r["objsense"].lower())
    assert len(inst.obj_quad) == int(r["nobjquadnz"])
    assert inst.n_vars - inst.n_integral == int(r["ncontvars"])
    # QPLIB's binary/integer split comes from the *source model* and is not
    # recorded in the .qplib file; only the total is derivable. See the note on
    # QplibInstance.n_binary.
    assert inst.n_integral == int(r["nbinvars"]) + int(r["nintvars"])


@pytest.mark.unit
@pytest.mark.parametrize("name", FIXTURES)
def test_objective_reproduced_at_reference_point(name, solu):
    """The decisive check: recompute the published objective from the file.

    This is what catches a misread section layout or a wrong quadratic scaling
    convention -- both of which produce a parse that looks structurally fine.
    """
    inst = _read(name)
    x, objvar = qp.read_solution(os.path.join(DATA, "sol", f"{name}.sol"), inst)
    assert objvar is not None, f"{name}: fixture .sol carries no objvar"
    recomputed = inst.evaluate_objective(x)
    assert recomputed == pytest.approx(objvar, rel=1e-6, abs=1e-9)
    # ...and it must agree with the independent qplib.solu oracle.
    assert solu[name] == pytest.approx(objvar, rel=1e-6, abs=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("name", FIXTURES)
def test_reference_point_is_feasible(name):
    """A reference point that violates the parsed model means we parsed it wrong."""
    inst = _read(name)
    x, _ = qp.read_solution(os.path.join(DATA, "sol", f"{name}.sol"), inst)
    assert inst.max_violation(x) <= 1e-4


@pytest.mark.unit
def test_quadratic_scale_is_uniform_one_half():
    """Both quadratic coefficients are 1/2 -- diagonal *and* off-diagonal.

    This convention is not documented upstream; it was fitted by least squares
    over 40 all-continuous corpus instances, coming out at exactly (0.5, 0.5)
    with rank 2 and residual 4.5e-12. The natural reading ("halve the diagonal,
    take off-diagonals at face value") is wrong and yields objectives ~2x too
    large.

    This re-fits the same two coefficients from the fixtures, so a regression to
    the natural reading fails here with a diagnosis rather than as an opaque
    objective mismatch. The fit is linear in the two scales regardless of
    integrality, so every fixture with a quadratic objective contributes.
    """
    rows, rhs = [], []
    for name in FIXTURES:
        inst = _read(name)
        x, objvar = qp.read_solution(os.path.join(DATA, "sol", f"{name}.sol"), inst)
        diag = sum(v * x[i] * x[j] for i, j, v in inst.obj_quad if i == j)
        off = sum(v * x[i] * x[j] for i, j, v in inst.obj_quad if i != j)
        if diag == 0.0 and off == 0.0:
            continue
        rows.append([diag, off])
        rhs.append(objvar - inst.obj_const - float(inst.obj_lin @ x))

    assert len(rows) >= 2, f"fit needs >=2 usable instances, got {len(rows)}"
    A = np.array(rows)
    assert np.linalg.matrix_rank(A) == 2, "fit is degenerate: cannot separate the two scales"
    coef, *_ = np.linalg.lstsq(A, np.array(rhs), rcond=None)
    np.testing.assert_allclose(coef, [0.5, 0.5], rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_binary_variables_are_derived_from_bounds_not_a_type_code():
    """V=B declares no bound or type block; the variables are implicitly [0,1]."""
    inst = _read("QPLIB_3815")  # QBL
    assert inst.variable_type == "B"
    assert inst.n_integral == inst.n_vars
    assert np.all(inst.lb == 0.0)
    assert np.all(inst.ub == 1.0)
    assert inst.n_binary == inst.n_vars


@pytest.mark.unit
def test_solution_index_offset_guard_is_load_bearing():
    """``.sol`` names are GAMS variable numbers counting objvar as 1.

    A wrong offset shifts every value by one variable. The reader guards against
    that by checking the integrality implied by each record's prefix against the
    parsed variable, plus a range check.

    The guard is *corpus-level*, not per-instance: a shift is only detectable
    where it crosses a variable-type boundary or runs off the end, so an
    all-binary instance cannot detect it at all. This test corrupts the offset
    and asserts the guard still fires on part of the fixture set -- enough to
    prove it is load-bearing rather than decorative. The offset itself was
    established over the whole corpus, where offset 0 produced 93 out-of-range
    and 34 type-mismatch records and the adopted offset produced zero of each.
    """
    original = qp._SOL_INDEX_OFFSET
    for delta in (+1, -1):
        fired = 0
        try:
            qp._SOL_INDEX_OFFSET = original + delta
            for name in FIXTURES:
                inst = _read(name)
                try:
                    qp.read_solution(os.path.join(DATA, "sol", f"{name}.sol"), inst)
                except ValueError:
                    fired += 1
        finally:
            qp._SOL_INDEX_OFFSET = original
        assert fired > 0, f"offset guard detected nothing when shifted by {delta:+d}"

    # ...and it must be silent at the correct offset.
    for name in FIXTURES:
        inst = _read(name)
        x, _ = qp.read_solution(os.path.join(DATA, "sol", f"{name}.sol"), inst)
        assert x.shape == (inst.n_vars,)


@pytest.mark.unit
def test_malformed_solution_name_raises(tmp_path):
    inst = _read("QPLIB_3562")
    bad = tmp_path / "bad.sol"
    bad.write_text("objvar 1.0\nnot-a-name 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        qp.read_solution(str(bad), inst)


@pytest.mark.unit
def test_out_of_range_solution_index_raises(tmp_path):
    inst = _read("QPLIB_3562")
    bad = tmp_path / "oob.sol"
    bad.write_text(f"objvar 1.0\ni{inst.n_vars + 99} 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        qp.read_solution(str(bad), inst)


@pytest.mark.unit
@pytest.mark.parametrize("name", FIXTURES)
def test_to_model_builds_a_solvable_model(name):
    """The parsed instance must translate into a discopt Model.

    Only the translation is checked here, not a solve: the objective value at
    the reference point is already pinned by the tests above.
    """
    inst = _read(name)
    m = qp.to_model(inst)
    assert len(m._variables) == inst.n_vars
    assert m.name == name


@pytest.mark.unit
def test_truncated_file_raises_rather_than_parsing_short(tmp_path):
    """A short read must fail loudly.

    The cursor raises on exhaustion instead of returning a sentinel, so a
    truncated file cannot silently yield a smaller, well-formed model.
    """
    src = os.path.join(DATA, "qplib", "QPLIB_3814.qplib")
    with open(src, encoding="utf-8") as fh:
        lines = fh.readlines()
    cut = tmp_path / "truncated.qplib"
    cut.write_text("".join(lines[: len(lines) // 2]), encoding="utf-8")
    with pytest.raises(ValueError):
        qp.read_qplib(str(cut))


@pytest.mark.unit
def test_trailing_garbage_raises(tmp_path):
    """Unconsumed records mean the layout was misread; that must not pass."""
    src = os.path.join(DATA, "qplib", "QPLIB_3814.qplib")
    with open(src, encoding="utf-8") as fh:
        body = fh.read()
    extra = tmp_path / "extra.qplib"
    extra.write_text(body + "\n999 999 999\n", encoding="utf-8")
    with pytest.raises(ValueError):
        qp.read_qplib(str(extra))
