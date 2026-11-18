"""discopt's LP-format reader (``discopt._lpfile``, highs-parity P6; #2221) reproduces
HiGHS **1.16.0**'s ``readModel`` -> ``getLp`` and ``getModel().hessian_`` exactly, and
works with ``highspy`` absent.

Why a frozen fixture. HiGHS 1.16.0 rewrote its LP reader (a recursive-descent parser
replacing 1.15's staged pipeline), and discopt ports the 1.16.0 reader. The two read
81 of the 275 synthetic cases of ``_lp_probe_cases.py`` differently (1.16 rejects 37
that 1.15.1 accepted, accepts 27 it rejected, and reads 17 differently), so a live
comparison is only meaningful against highspy >= 1.16.0. highspy is optional and the
supported floor is older, so every case is also compared against HiGHS 1.16.0's own
readings frozen in ``data/lp_reader/highs_1_16_0_readings.json.gz`` (regenerated with
highspy 1.16.0 by ``_regen_lp_reader_fixture.py``). Both comparisons are the same exact
one (``_mps_differential.differences``), kError included; the fixture test runs on
every install, the live test when highspy >= 1.16.0 is installed. The in-repo LP files
are read identically by HiGHS 1.11.0, 1.15.1 and 1.16.0, so they are compared against
whatever highspy is installed.

The corpus sweep (``docs/dev/data/highs-parity/lp_reader/``) covers HiGHS's
``check/instances`` and a fuzzer; this test pins the in-repo files, every synthetic
case, expected values of the hairy rules without HiGHS, and the HiGHS-less loader.
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
from _regen_lp_reader_fixture import HIGHS_VERSION, decode, load_fixture, text_sha256
from discopt._lpfile import read_lp

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
IN_REPO = sorted(
    glob.glob(os.path.join(HERE, "data", "**", "*.lp"), recursive=True)
    + glob.glob(os.path.join(HERE, "data", "**", "*.lp.gz"), recursive=True)
)
QPLIB_0031 = os.path.join(HERE, "data", "qplib", "lp", "QPLIB_0031.lp")
CASE_IDS = [f"{k}-{t[:40]}" for k, (t, _) in enumerate(PROBE_CASES)]


def _version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split(".")[:3])


def _live_highs_reads_like_the_port() -> bool:
    """True when the installed highspy has HiGHS 1.16.0's reader (>= 1.16.0)."""
    try:
        import highspy
    except ImportError:
        return False
    return _version(highspy.Highs().version()) >= _version(HIGHS_VERSION)


LIVE = _live_highs_reads_like_the_port()
needs_live = pytest.mark.skipif(
    not LIVE, reason=f"highspy < {HIGHS_VERSION} or absent; the fixture test covers this"
)
_FIXTURE = load_fixture()


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


def _assert_same(ours, ref):
    assert (ref is None) == (ours is None), ("HiGHS rejects", ref is None, "ours", ours is None)
    if ref is not None:
        assert differences(ours, ref) == []


@pytest.mark.parametrize("path", IN_REPO, ids=[os.path.basename(p) for p in IN_REPO])
def test_in_repo_files_match_highs_exactly(path):
    pytest.importorskip("highspy")
    assert differences(read_lp(path), highs_arrays(path)) == []


def test_fixture_covers_every_case():
    """The fixture is HiGHS 1.16.0's, one entry per probe case, texts unchanged."""
    assert _FIXTURE["highs_version"] == HIGHS_VERSION
    cases = _FIXTURE["cases"]
    assert len(cases) == len(PROBE_CASES)
    stale = [
        k
        for k, (tag, text) in enumerate(PROBE_CASES)
        if (cases[k]["k"], cases[k]["tag"], cases[k]["sha256"]) != (k, tag, text_sha256(text))
    ]
    assert stale == [], f"probe cases changed; run _regen_lp_reader_fixture.py: {stale}"


@pytest.mark.parametrize("k", range(len(PROBE_CASES)), ids=CASE_IDS)
def test_synthetic_cases_match_highs_1_16_fixture(k, tmp_path):
    """Exact agreement with HiGHS 1.16.0's frozen reading; kError <-> ours raises."""
    text = PROBE_CASES[k][1]
    case = _FIXTURE["cases"][k]
    assert case["sha256"] == text_sha256(text), "stale fixture: run _regen_lp_reader_fixture.py"
    _assert_same(_read_or_none(read_lp, _write(tmp_path, text)), decode(case["reading"]))


@needs_live
@pytest.mark.parametrize("k", range(len(PROBE_CASES)), ids=CASE_IDS)
def test_synthetic_cases_match_live_highs_exactly(k, tmp_path):
    """Exact agreement with the installed highspy (>= 1.16.0); kError <-> ours raises."""
    path = _write(tmp_path, PROBE_CASES[k][1])
    _assert_same(_read_or_none(read_lp, path), _read_or_none(highs_arrays, path))


def test_gz_matches_plain(tmp_path):
    """``.lp.gz`` is decompressed and read identically."""
    plain = read_lp(QPLIB_0031)
    gz = tmp_path / "q.lp.gz"
    with open(QPLIB_0031, "rb") as f, gzip.open(gz, "wb") as g:
        g.write(f.read())
    assert differences(read_lp(str(gz)), plain) == []


def test_rules_without_highs(tmp_path):
    """Expected values of the hairy rules, checked without HiGHS (counted).

    Values are HiGHS 1.16.0's; where 1.15.1 read the same text differently the
    comment says so."""
    checked = 0
    # Costs of a repeated variable are SUMMED (1.15.1: the last term won, c[x] = -1);
    # offset summed; LHS constant dropped; auto row names over all rows; duplicate
    # row entries summed into the first position.
    d = read_lp(
        _write(
            tmp_path,
            "min\n obj: 2 x + 3 y - x + 5\nst\n c1: x + y - 3 >= 1\n x + 2 y + x <= 4\n"
            "-3 + z = 2\nend\n",
        )
    )
    assert list(d["c"]) == [1.0, 3.0, 0.0] and d["offset"] == 5.0
    assert d["objective_name"] == "obj"
    assert d["row_names"] == ["c1", "HiGHS_R1", "HiGHS_R2"]
    assert list(d["row_lo"]) == [1.0, -np.inf, 2.0]
    assert d["A"].toarray().tolist() == [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    checked += 1
    # An explicit HiGHS_R name plus a needed automatic one clears all row names.
    d = read_lp(_write(tmp_path, "min\n x\nst\n HiGHS_R7: x >= 2\n x <= 5\nend\n"))
    assert d["row_names"] == []
    checked += 1
    # Sections parse in FILE order and columns are created on first mention (1.15.1
    # parsed in a fixed order -- objective, rows, bounds, general -- giving
    # ["o", "k", "b", "g"]). A second objective section is an error (1.15.1: MIN won
    # over MAX); see ERROR_CASES["two objectives"].
    d = read_lp(_write(tmp_path, "general\n g\nbounds\n b <= 3\nmin\n o\nst\n c: k >= 1\nend\n"))
    assert d["col_names"] == ["g", "b", "o", "k"] and not d["maximize"]
    assert list(d["integrality"]) == [1, 0, 0, 0]
    assert list(d["col_ub"]) == [np.inf, 3.0, np.inf, np.inf]
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
    # Lexer: a number ends where strtod stops (2e3x = 2000 x, 1d2 = 1 d2, 1e400 is
    # +inf); inf/nan are numbers only where a number is expected, so "infeasible" is
    # a variable (1.15.1 lexed it as inf followed by "easible", with cost +inf).
    d = read_lp(_write(tmp_path, "min\n 2e3x + 1d2 + infeasible + 1e400 y\nst\n c: x >= 1\nend\n"))
    assert d["col_names"] == ["x", "d2", "infeasible", "y"]
    assert list(d["c"]) == [2000.0, 1.0, 1.0, np.inf]
    checked += 1
    # Hessian: lower triangle, entries sorted by (column, row) and summed, zero
    # diagonals filled (1.15.1 kept its aliased-gather order, indices [0, 2, 1, 1, 2]).
    d = read_lp(
        _write(tmp_path, "min\n obj: x + [ z*x + y*x + x^2 ]/2\nst\n c: x + y + z >= 2\nend\n")
    )
    h = d["hessian"]
    assert d["has_hessian"] and h.shape == (3, 3)
    assert list(h.indptr) == [0, 3, 4, 5] and list(h.indices) == [0, 1, 2, 1, 2]
    assert list(h.data) == [1.0, 0.5, 0.5, 0.0, 0.0]
    checked += 1
    # A Hessian whose terms cancel is cleared.
    d = read_lp(_write(tmp_path, "min\n obj: [ x*y - y*x ]/2\nst\n c: x + y >= 1\nend\n"))
    assert d["hessian"] is None and not d["has_hessian"]
    checked += 1
    # A file of comments only is an empty model. (1.15.1 also ignored any tokens
    # before the first keyword; 1.16 rejects them, ERROR_CASES["garbage"].)
    d = read_lp(_write(tmp_path, "\\ nothing but a comment\n"))
    assert (d["n"], d["m"]) == (0, 0)
    checked += 1
    assert checked == 8


# Files HiGHS 1.16.0 rejects. The last group were accepted by 1.15.1.
ERROR_CASES = {
    "no end after content": "min\n x\nst\n c: x >= 1\n",
    "content after end": "min\n x\nst\n c: x >= 1\nend\nfoo\n",
    "ranged row": "min\n x\nst\n -5 <= x + y <= 10\nend\n",
    "rhs sum": "min\n x\nst\n c: x <= 10 + 2\nend\n",
    "exponent 3": "min\n obj: [ x^3 ]/2\nst\n c: x >= 1\nend\n",
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
    # Accepted by 1.15.1: MIN won over MAX; tokens before the first keyword were
    # ignored; "subject to" could span two lines.
    "two objectives": "general\n g\nbounds\n b <= 3\nmax\n q\nmin\n o\nst\n c: k >= 1\nend\n",
    "garbage": "this is not an lp file 3 + y\n",
    "subject\\nto": "min\n x\nsubject\nto\n c: x >= 2\nend\n",
}

# Files 1.15.1 rejected that HiGHS 1.16.0 reads (they were ERROR_CASES before #2221).
ACCEPTED_SINCE_1_16 = {
    "repeated st": "min\n x\nst\n c: x >= 2\nst\n d: x <= 5\nend\n",
    "strict <": "min\n x\nst\n c: x < 1\nend\n",
    "=<": "min\n x\nst\n c: x =< 1\nend\n",
    "3 * x": "min\n x\nst\n c: 3 * x >= 2\nend\n",
    "triple sign": "min\n - - - x\nst\n c: x >= 2\nend\n",
    "-[": "min\n obj: - [ x^2 ]/2\nst\n c: x >= 1\nend\n",
    "missing /2 (doubled)": "min\n obj: [ x^2 ]\nst\n c: x >= 1\nend\n",
}


@pytest.mark.parametrize("case", sorted(ERROR_CASES))
def test_errors_without_highs(case, tmp_path):
    """Files HiGHS 1.16.0 rejects are rejected (and live HiGHS agrees, when >= 1.16)."""
    path = _write(tmp_path, ERROR_CASES[case])
    with pytest.raises(ValueError):
        read_lp(path)
    if LIVE:
        with pytest.raises(ValueError):
            highs_arrays(path)


@pytest.mark.parametrize("case", sorted(ACCEPTED_SINCE_1_16))
def test_accepted_since_1_16(case, tmp_path):
    """Files 1.15.1 rejected are read (exactly as live HiGHS reads them, when >= 1.16)."""
    path = _write(tmp_path, ACCEPTED_SINCE_1_16[case])
    d = read_lp(path)
    assert d["n"] >= 1
    if LIVE:
        assert differences(d, highs_arrays(path)) == []


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
