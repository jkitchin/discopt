"""discopt's MPS reader (``discopt._mps``, highs-parity P6) reproduces HiGHS's
``readModel`` -> ``getLp`` exactly, and works with ``highspy`` absent.

The corpus sweep (``docs/dev/data/highs-parity/mps_reader/``) compares 176 files
(MIPLIB panel, HiGHS ``check/instances``, every in-repo fixture); this test pins the
in-repo files and one synthetic case per HiGHS rule the reader reproduces.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

import numpy as np
import pytest
from _mps_differential import differences, highs_arrays
from discopt._mps import read_mps

HERE = os.path.dirname(__file__)
IN_REPO = sorted(
    glob.glob(os.path.join(HERE, "data", "**", "*.mps"), recursive=True)
    + glob.glob(os.path.join(HERE, "data", "**", "*.mps.gz"), recursive=True)
)

# One case per HiGHS rule (HMpsFF.cpp); comments name the rule.
CASES = {
    # INTORG column without bounds -> binary [0, 1]; with UP 5 -> [0, 5]; with LO 2
    # -> [2, inf]; offset is minus the objective RHS; OBJSENSE on the next line.
    "markers_offset_objsense": """NAME t
OBJSENSE
    MAX
ROWS
 N  obj
 L  c1
 G  c2
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    x  obj  1  c1  1
    y  obj  2  c2  1
    z  c1  3
    MARKER                 'MARKER'                 'INTEND'
    w  obj  -1  c1  2
RHS
    rhs  obj  7  c1  10
    rhs  c2  1
BOUNDS
 UP bnd y 5
 LO bnd z 2
ENDATA
""",
    # RANGES on every row type incl. E with R < 0 and R > 0, applied to current
    # bounds; a later duplicate RHS/RANGES value is ignored.
    "ranges": """NAME r
ROWS
 N  obj
 E  e1
 E  e2
 L  l1
 G  g1
COLUMNS
    x  obj  1  e1  1
    x  e2  1  l1  1
    x  g1  1
RHS
    rhs  e1  4  e2  4
    rhs  l1  9  g1  1
    rhs  l1  99
RANGES
    rng  e1  -2  e2  3
    rng  l1  5  g1  6
    rng  l1  50
ENDATA
""",
    # A column name that reappears later is a SECOND column (names cleared); a
    # duplicate (column, row) entry keeps the first value; zeros are dropped; a later
    # N row is dropped; entries below 1e-9 are dropped by assessLp.
    "duplicates_and_dropped": """NAME d
ROWS
 N  obj
 N  free
 E  c1
 E  c2
COLUMNS
    x  obj  1  c1  2
    x  c1  5  c2  0
    x  free  3  c2  1e-10
    y  c1  1
    x  c2  4
RHS
    rhs  c1  1
ENDATA
""",
    # BOUNDS: MI/PL/FR/FX/BV, a bound for an unknown column appends it, a second
    # upper setter is ignored, UP < 0 leaves the lower bound at 0, 1e30 is infinite,
    # D exponents and SIF (no set name) RHS.
    "bounds": """NAME b
ROWS
 N  obj
 L  c1
COLUMNS
    a  obj  1  c1  1
    b  c1  1
    c  c1  1
    d  c1  1
    e  c1  1
    f  c1  1
RHS
    c1  1.5D+1
BOUNDS
 MI bnd a
 UP bnd a 3
 UP bnd a 4
 FR bnd b
 FX bnd c 2.5
 BV bnd d
 UP bnd e -2
 PL bnd f
 UP bnd g 1e30
ENDATA
""",
}


@pytest.mark.parametrize("path", IN_REPO, ids=[os.path.basename(p) for p in IN_REPO])
def test_in_repo_files_match_highs_exactly(path):
    pytest.importorskip("highspy")
    assert differences(read_mps(path), highs_arrays(path)) == []


@pytest.mark.parametrize("case", sorted(CASES))
def test_synthetic_rules_match_highs_exactly(case, tmp_path):
    pytest.importorskip("highspy")
    f = tmp_path / f"{case}.mps"
    f.write_text(CASES[case])
    assert differences(read_mps(str(f)), highs_arrays(str(f))) == []


def test_rules_without_highs(tmp_path):
    """Expected values of the rules above, checked without HiGHS (counted)."""
    checked = 0
    f = tmp_path / "m.mps"
    f.write_text(CASES["markers_offset_objsense"])
    d = read_mps(str(f))
    assert d["maximize"] and d["offset"] == -7.0
    assert list(d["col_lb"]) == [0.0, 0.0, 2.0, 0.0]
    assert list(d["col_ub"]) == [1.0, 5.0, np.inf, np.inf]
    assert list(d["integrality"]) == [1, 1, 1, 0]
    checked += 1
    f.write_text(CASES["ranges"])
    d = read_mps(str(f))
    assert list(d["row_lo"]) == [2.0, 4.0, 4.0, 1.0]
    assert list(d["row_hi"]) == [4.0, 7.0, 9.0, 7.0]
    checked += 1
    f.write_text(CASES["duplicates_and_dropped"])
    d = read_mps(str(f))
    assert d["n"] == 3 and d["col_names"] == []  # x appears twice: two columns
    assert d["A"].toarray().tolist() == [[2.0, 1.0, 0.0], [0.0, 0.0, 4.0]]
    checked += 1
    f.write_text(CASES["bounds"])
    d = read_mps(str(f))
    assert d["n"] == 7  # g appended by BOUNDS
    assert list(d["col_lb"]) == [-np.inf, -np.inf, 2.5, 0.0, 0.0, 0.0, 0.0]
    assert list(d["col_ub"]) == [3.0, np.inf, 2.5, 1.0, -2.0, np.inf, np.inf]
    assert d["row_hi"][0] == 15.0
    checked += 1
    assert checked == 4


def test_fixed_format_and_errors_are_refused(tmp_path):
    """A file HiGHS would hand to its legacy fixed reader is refused, not read
    differently; a file HiGHS rejects is rejected."""
    f = tmp_path / "fixed.mps"
    f.write_text("NAME x\nROWS\n N  obj\n L  row one\nCOLUMNS\nENDATA\n")
    with pytest.raises(NotImplementedError):
        read_mps(str(f))
    f.write_text("NAME x\nROWS\n N  obj\n")  # no ENDATA
    with pytest.raises(ValueError):
        read_mps(str(f))


def test_mipcc_reads_without_highspy():
    """``discopt.mipcc.read_mps`` no longer needs HiGHS (highspy blocked)."""
    path = os.path.join(HERE, "data", "issue2069", "clay0203_master.mps.gz")
    code = (
        "import sys; sys.modules['highspy'] = None\n"
        "from discopt.mipcc import read_mps\n"
        f"d = read_mps({path!r})\n"
        "assert 'highspy' not in sys.modules or sys.modules['highspy'] is None\n"
        "print('READ', d['A'].shape, int(d['is_int'].sum()))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    assert "READ (531, 30)" in p.stdout, p.stdout
