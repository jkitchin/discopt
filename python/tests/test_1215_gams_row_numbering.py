"""Row `k` of a constraint family is `{name}_{k}` in EVERY format (#1215).

`lp.py`, `mps.py` and the Rust builder row naming all number an expanded family
from 0. `gams.py` numbered from 1, which made GAMS the only one of the four
formats where row k of family `c` was not called `c_k` -- so the same model
exported to `.lp` and to `.gms` disagreed on every expanded row's name.

It also meant that vectorising an emitter -- replacing N per-element constraints
named `c_0 … c_{N-1}` with one family named `c` -- silently SHIFTED every GAMS row
name by one while leaving `.nl` and LP byte-identical. That is how the #1215
NN-emitter work shipped a GAMS rename unnoticed: the verification covered `.nl`
and LP, the two formats that could not show it.
"""

from __future__ import annotations

import re

import discopt.modeling as dm
import pytest
from discopt.export import to_gams, to_lp, to_mps
from discopt.modeling import Model

N = 4


def _model():
    m = Model("fam")
    x = m.continuous("x", shape=(N,), lb=0.0, ub=5.0)
    y = m.continuous("y", shape=(N,), lb=0.0, ub=5.0)
    m.subject_to(x + 2.0 * y <= 3.0, name="c")
    m.minimize(dm.sum(x))
    return m


def _gams_eq_names(text):
    return sorted(set(re.findall(r"^(c_\d+)\.\.", text, re.M)))


def test_gams_numbers_family_rows_from_zero():
    assert _gams_eq_names(to_gams(_model())) == [f"c_{k}" for k in range(N)]


@pytest.mark.parametrize("writer", [to_lp, to_mps])
def test_the_other_writers_agree(writer):
    text = writer(_model())
    for k in range(N):
        assert f"c_{k}" in text, f"c_{k} missing"
    assert f"c_{N}" not in text, "numbering ran past the family size"


def test_all_three_formats_use_the_same_row_names():
    """The invariant, stated once: the name set does not depend on the format."""
    names = {}
    for label, writer in (("gams", to_gams), ("lp", to_lp), ("mps", to_mps)):
        text = writer(_model())
        names[label] = {f"c_{k}" for k in range(N) if f"c_{k}" in text}
    assert names["gams"] == names["lp"] == names["mps"] == {f"c_{k}" for k in range(N)}, names


def test_a_single_row_family_keeps_the_bare_name_in_every_format():
    """One row is emitted under the family name, with no index, in all formats."""
    m = Model("one")
    x = m.continuous("x", shape=(1,), lb=0.0, ub=5.0)
    m.subject_to(x <= 3.0, name="solo")
    m.minimize(x[0])
    for writer in (to_gams, to_lp, to_mps):
        text = writer(m)
        assert "solo" in text
        assert "solo_0" not in text, f"{writer.__name__} indexed a single-row family"
