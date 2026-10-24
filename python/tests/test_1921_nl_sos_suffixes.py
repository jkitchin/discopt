"""#1921: ``from_nl`` honours the SOS sets of ``sosno``/``ref`` variable suffixes.

The text parser used to read past every ``S`` (suffix) segment, so the SOS1/SOS2
sets a ``.nl`` file declares that way were silently dropped and ``Model.solve()``
returned an optimum of the relaxation as the optimum of the model (``(1, 1)``
with objective ``-2`` on the SOS1 case below; true optimum ``-1``).
"""

import discopt.modeling as dm
import pytest
from discopt.modeling.core import _SOSConstraint


def _nl(n, obj, suffix_text):
    """A text .nl: min sum(obj[j] x_j) s.t. sum(x) <= n + 1, x in [0, 1]^n."""
    lines = [
        "g3 1 1 0",
        f" {n} 1 1 0 0",
        " 0 0",
        " 0 0",
        " 0 0 0",
        " 0 0 0 1",
        " 0 0 0 0 0",
        f" {n} {n}",
        " 0 0",
        " 0 0 0 0 0",
    ]
    lines.append(suffix_text.rstrip("\n"))
    lines += ["C0", "n0", "O0 0", "n0", "r", f"1 {n + 1}", "b"]
    lines += ["0 0 1"] * n
    lines += [f"k{n - 1}"] + [str(j + 1) for j in range(n - 1)]
    lines += [f"J0 {n}"] + [f"{j} 1" for j in range(n)]
    lines += [f"G0 {n}"] + [f"{j} {c}" for j, c in enumerate(obj)]
    return "\n".join(lines) + "\n"


def _load(tmp_path, text):
    p = tmp_path / "m.nl"
    p.write_text(text)
    return dm.from_nl(str(p))


def _sos(m):
    return [c for c in m._constraints if isinstance(c, _SOSConstraint)]


SOS1 = "S0 2 sosno\n0 1\n1 1\nS0 2 ref\n0 1\n1 2\n"


@pytest.mark.smoke
def test_sos1_suffix_is_declared_and_solved(tmp_path):
    m = _load(tmp_path, _nl(2, [-1, -1], SOS1))
    (sos,) = _sos(m)
    assert sos.sos_type == 1
    assert [v.name for v in sos.variables] == ["x0", "x1"]
    # The raw .nl does not encode the SOS reformulation POUNCE would see.
    assert getattr(m, "_source_nl_path", None) is None

    r = m.solve(time_limit=20)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0, abs=1e-6)
    vals = sorted(float(r.value(v)) for v in m._variables)
    assert vals[0] == pytest.approx(0.0, abs=1e-6)
    assert vals[1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "refs, expected",
    [
        # order x0, x1, x2: x0 and x2 are not adjacent -> at most one of them.
        ((1, 2, 3), -1.0),
        # order x0, x2, x1: x0 and x2 adjacent -> both can be 1.
        ((1, 3, 2), -2.0),
    ],
)
def test_sos2_members_are_ordered_by_ref(tmp_path, refs, expected):
    sfx = "S0 3 sosno\n0 -1\n1 -1\n2 -1\nS4 3 ref\n" + "".join(
        f"{j} {w}\n" for j, w in enumerate(refs)
    )
    m = _load(tmp_path, _nl(3, [-1, 0, -1], sfx))
    (sos,) = _sos(m)
    assert sos.sos_type == 2
    order = sorted(range(3), key=lambda j: refs[j])
    assert [v.name for v in sos.variables] == [f"x{j}" for j in order]
    r = m.solve(time_limit=20)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(expected, abs=1e-6)


def test_file_without_sos_suffixes_is_unchanged(tmp_path):
    # A non-SOS suffix is read and ignored; the native-.nl route stays available.
    m = _load(tmp_path, _nl(2, [-1, -1], "S0 1 priority\n0 5\n"))
    assert _sos(m) == []
    assert m._source_nl_path is not None
    r = m.solve(time_limit=20)
    assert r.objective == pytest.approx(-2.0, abs=1e-6)


def test_sos2_without_ref_weight_is_refused(tmp_path):
    sfx = "S0 2 sosno\n0 -1\n1 -1\nS4 1 ref\n0 1\n"
    with pytest.raises(ValueError, match="no 'ref' weight"):
        _load(tmp_path, _nl(2, [-1, -1], sfx))


def test_sos2_with_tied_ref_weights_is_refused(tmp_path):
    sfx = "S0 2 sosno\n0 -1\n1 -1\nS4 2 ref\n0 1\n1 1\n"
    with pytest.raises(ValueError, match="tied 'ref' weights"):
        _load(tmp_path, _nl(2, [-1, -1], sfx))


def test_uninterpreted_sos_encoding_is_refused(tmp_path):
    # AMPL's piecewise-linear SOS encoding (sos/sosref) is not read: refuse
    # instead of dropping the constraints.
    sfx = "S0 2 sos\n0 2\n1 2\nS4 2 sosref\n0 1\n1 2\n"
    with pytest.raises(ValueError, match="'sos' suffix"):
        _load(tmp_path, _nl(2, [-1, -1], sfx))


def test_sos1_member_without_declared_bound_uses_row_implied_bound():
    # ``waters`` (MINLPLib) declares its SOS members with ub = inf; the rows imply
    # finite bounds, which are a valid big-M for the SOS reformulation.
    m = dm.Model("sos1_implied")
    x0 = m.continuous("x0", lb=0)
    x1 = m.continuous("x1", lb=0)
    m.subject_to(x0 + x1 <= 3)
    m.sos1([x0, x1])
    m.minimize(-x0 - 2 * x1)
    r = m.solve(time_limit=20)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-6.0, abs=1e-6)
    assert float(r.value(x0)) == pytest.approx(0.0, abs=1e-6)


def test_sos1_member_with_no_finite_bound_still_refuses():
    m = dm.Model("sos1_unbounded")
    x0 = m.continuous("x0", lb=0)
    x1 = m.continuous("x1", lb=0)
    m.subject_to(x0 - x1 <= 3)
    m.sos1([x0, x1])
    m.minimize(x0 + x1)
    with pytest.raises(ValueError, match="upper bound is infinite"):
        m.solve(time_limit=20)
