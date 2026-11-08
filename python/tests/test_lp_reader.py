"""discopt's LP-format reader (``discopt._lpfile``, highs-parity P6) reproduces HiGHS
1.15.1's ``readModel`` -> ``getLp`` and ``getModel().hessian_`` exactly, and works
with ``highspy`` absent.

The corpus sweep (``docs/dev/data/highs-parity/lp_reader/``) compares every LP file in
HiGHS's ``check/instances`` and the repo plus the synthetic cases of
``_lp_probe_cases.py``; this test pins the in-repo files, every synthetic case (exact
vs highspy, kError included), expected values of the hairy rules without HiGHS, and
the HiGHS-less loader.
"""

from __future__ import annotations

import glob
import gzip
import os
import subprocess
import sys

import numpy as np
import pytest
from _lp_probe_cases import PROBE_CASES
from _mps_differential import differences, highs_arrays
from discopt._lpfile import read_lp

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
IN_REPO = sorted(
    glob.glob(os.path.join(HERE, "data", "**", "*.lp"), recursive=True)
    + glob.glob(os.path.join(HERE, "data", "**", "*.lp.gz"), recursive=True)
)
QPLIB_0031 = os.path.join(HERE, "data", "qplib", "lp", "QPLIB_0031.lp")


def _write(tmp_path, text: str, name: str = "m.lp") -> str:
    f = tmp_path / name
    f.write_bytes(text.encode("utf-8"))
    return str(f)


def _read_or_none(fn, path):
    """The reading, or ``None`` where the reader rejects the file (``ValueError``)."""
    try:
        return fn(path)
    except UnicodeDecodeError:
        raise  # a ValueError subclass, but never a rejection: no verdict
    except ValueError:
        return None


@pytest.mark.parametrize("path", IN_REPO, ids=[os.path.basename(p) for p in IN_REPO])
def test_in_repo_files_match_highs_exactly(path):
    pytest.importorskip("highspy")
    assert differences(read_lp(path), highs_arrays(path)) == []


@pytest.mark.parametrize(
    "k", range(len(PROBE_CASES)), ids=[f"{k}-{t[:40]}" for k, (t, _) in enumerate(PROBE_CASES)]
)
def test_synthetic_cases_match_highs_exactly(k, tmp_path):
    """Exact agreement, and where HiGHS returns kError ours raises."""
    pytest.importorskip("highspy")
    path = _write(tmp_path, PROBE_CASES[k][1])
    ref = _read_or_none(highs_arrays, path)
    ours = _read_or_none(read_lp, path)
    assert (ref is None) == (ours is None), (ref is None, ours is None)
    if ref is not None:
        assert differences(ours, ref) == []


def test_gz_matches_plain(tmp_path):
    """``.lp.gz`` is decompressed and read identically."""
    plain = read_lp(QPLIB_0031)
    gz = tmp_path / "q.lp.gz"
    with open(QPLIB_0031, "rb") as f, gzip.open(gz, "wb") as g:
        g.write(f.read())
    assert differences(read_lp(str(gz)), plain) == []


def test_rules_without_highs(tmp_path):
    """Expected values of the hairy rules, checked without HiGHS (counted)."""
    checked = 0
    # Cost assigned per term (last wins); offset summed; LHS constant dropped; auto
    # row names over all rows; duplicates summed into the first position.
    d = read_lp(
        _write(
            tmp_path,
            "min\n obj: 2 x + 3 y - x + 5\nst\n c1: x + y - 3 >= 1\n x + 2 y + x <= 4\n"
            "-3 + z = 2\nend\n",
        )
    )
    assert list(d["c"]) == [-1.0, 3.0, 0.0] and d["offset"] == 5.0
    assert d["objective_name"] == "obj"
    assert d["row_names"] == ["c1", "HiGHS_R1", "HiGHS_R2"]
    assert list(d["row_lo"]) == [1.0, -np.inf, 2.0]
    assert d["A"].toarray().tolist() == [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    checked += 1
    # An explicit HiGHS_R name plus a needed automatic one clears all row names.
    d = read_lp(_write(tmp_path, "min\n x\nst\n HiGHS_R7: x >= 2\n x <= 5\nend\n"))
    assert d["row_names"] == []
    checked += 1
    # Sections parse in FIXED order (columns: objective, rows, bounds, gen) whatever
    # the file order; MIN wins over MAX, whose variables are never created.
    d = read_lp(
        _write(tmp_path, "general\n g\nbounds\n b <= 3\nmax\n q\nmin\n o\nst\n c: k >= 1\nend\n")
    )
    assert d["col_names"] == ["o", "k", "b", "g"] and not d["maximize"]
    assert list(d["integrality"]) == [0, 0, 0, 1]
    checked += 1
    # Bounds: last setter wins; bin sets upper 1 only on an infinite upper bound and
    # leaves the lower alone; 1e30 becomes infinite only later (assessLp).
    d = read_lp(
        _write(
            tmp_path,
            "min\n a + b + c + e\nst\n r: a + b + c + e >= 1\nbounds\n 5 <= a\n a free\n"
            " a <= 2\n b >= -3\n c <= 1e30\n e <= -2\nbin\n b c\nend\n",
        )
    )
    assert list(d["col_lb"]) == [-np.inf, -3.0, 0.0, 0.0]
    assert list(d["col_ub"]) == [2.0, 1.0, np.inf, -2.0]
    assert list(d["integrality"]) == [0, 1, 1, 0]
    checked += 1
    # Lexer: strtod prefixes (2e3x = 2000 x, 1d2 = 1 d2), inf/nan prefixes.
    d = read_lp(_write(tmp_path, "min\n 2e3x + 1d2 + infeasible\nst\n c: x >= 1\nend\n"))
    assert d["col_names"] == ["x", "d2", "easible"]
    assert list(d["c"]) == [2000.0, 1.0, np.inf]
    checked += 1
    # Hessian: lower triangle, HiGHS's aliased-gather order, zero diagonals filled.
    d = read_lp(
        _write(tmp_path, "min\n obj: x + [ z*x + y*x + x^2 ]/2\nst\n c: x + y + z >= 2\nend\n")
    )
    h = d["hessian"]
    assert d["has_hessian"] and h.shape == (3, 3)
    assert list(h.indptr) == [0, 3, 4, 5] and list(h.indices) == [0, 2, 1, 1, 2]
    assert list(h.data) == [1.0, 0.5, 0.5, 0.0, 0.0]
    checked += 1
    # A Hessian whose terms cancel is cleared.
    d = read_lp(_write(tmp_path, "min\n obj: [ x*y - y*x ]/2\nst\n c: x + y >= 1\nend\n"))
    assert d["hessian"] is None and not d["has_hessian"]
    checked += 1
    # Tokens before the first keyword are ignored: garbage is an empty model.
    d = read_lp(_write(tmp_path, "this is not an lp file 3 + y\n"))
    assert (d["n"], d["m"]) == (0, 0)
    checked += 1
    assert checked == 8


ERROR_CASES = {
    "no end after content": "min\n x\nst\n c: x >= 1\n",
    "content after end": "min\n x\nst\n c: x >= 1\nend\nfoo\n",
    "repeated st": "min\n x\nst\n c: x >= 2\nst\n d: x <= 5\nend\n",
    "strict <": "min\n x\nst\n c: x < 1\nend\n",
    "=<": "min\n x\nst\n c: x =< 1\nend\n",
    "ranged row": "min\n x\nst\n -5 <= x + y <= 10\nend\n",
    "rhs sum": "min\n x\nst\n c: x <= 10 + 2\nend\n",
    "3 * x": "min\n x\nst\n c: 3 * x >= 2\nend\n",
    "triple sign": "min\n - - - x\nst\n c: x >= 2\nend\n",
    "-[": "min\n obj: - [ x^2 ]/2\nst\n c: x >= 1\nend\n",
    "exponent 3": "min\n obj: [ x^3 ]/2\nst\n c: x >= 1\nend\n",
    "missing /2": "min\n obj: [ x^2 ]\nst\n c: x >= 1\nend\n",
    "/4": "min\n obj: [ x^2 ]/4\nst\n c: x >= 1\nend\n",
    "quadratic row": "min\n x\nst\n c: x + [ x^2 ] >= 2\nend\n",
    "sos": "min\n x\nst\n c: x >= 1\nsos\n s1: S1 :: x:1\nend\n",
    "bad bound": "min\n x\nst\n c: x >= 1\nbounds\n 2 x <= 5\nend\n",
    "x = -inf": "min\n x\nst\n c: x >= 1\nbounds\n x = -inf\nend\n",
    "NaN row bound": "min\n x\nst\n c: x >= nan\nend\n",
    "|a| >= 1e15": "min\n x\nst\n c: 1e15 x >= 1\nend\n",
    "non-variable in general": "min\n x\nst\n c: x >= 1\ngeneral\n 3\nend\n",
    "MIN:": "MIN: x\nst\n c: x >= 2\nend\n",
    "odd comment": "min\n x /* a */ + y\nst\n c: x >= 1\nend\n",
    "NUL": "min\n x\nst\n c: x >= 1\n\x00\nend\n",
}


@pytest.mark.parametrize("case", sorted(ERROR_CASES))
def test_errors_without_highs(case, tmp_path):
    """Files HiGHS rejects are rejected (also pinned against highspy above)."""
    with pytest.raises(ValueError):
        read_lp(_write(tmp_path, ERROR_CASES[case]))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_lp(os.path.join(HERE, "data", "no_such_model.lp"))


def test_reader_and_loader_work_without_highspy(tmp_path):
    """``discopt._lpfile`` and ``lp_milp_loader.read_instance`` need no HiGHS
    (``highspy`` blocked); the loader still refuses a quadratic objective."""
    lin = _write(tmp_path, "max\n 3 x + 2 y\nst\n c: x + y <= 4\nbounds\n x <= 3\ngen\n y\nend\n")
    code = (
        "import sys; sys.modules['highspy'] = None\n"
        f"sys.path.insert(0, {os.path.join(ROOT, 'discopt_benchmarks', 'scripts')!r})\n"
        "from discopt._lpfile import read_lp\n"
        "import lp_milp_loader as L\n"
        f"d = read_lp({QPLIB_0031!r})\n"
        "print('READ', d['n'], d['m'], d['hessian'].nnz)\n"
        f"inst = L.read_instance({lin!r})\n"
        "print('LOADED', inst.n, inst.m, inst.maximize, inst.is_int.tolist())\n"
        "try:\n"
        f"    L.read_instance({QPLIB_0031!r})\n"
        "except ValueError as e:\n"
        "    print('REFUSED', 'quadratic' in str(e))\n"
        "assert sys.modules['highspy'] is None\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    assert "READ 60 32 494" in p.stdout, p.stdout
    assert "LOADED 2 1 True [False, True]" in p.stdout, p.stdout
    assert "REFUSED True" in p.stdout, p.stdout
