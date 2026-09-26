"""EX-3: the MPS/LP/GAMS writers must not widen a pinned binary back to 0-1.

Each writer has a shorthand for a binary column -- MPS ``BV``, an LP
``Binaries`` listing, a GAMS ``Binary Variables`` declaration -- and every one
of them *means* ``0 <= x <= 1``. All three emitted the shorthand unconditionally
and skipped the bounds section entirely, so a binary pinned by the public
``Variable.fix`` API exported as a free binary: the file is a **relaxation** of
the model discopt was handed, and an external solver answers a different
question without anything raising.

The sharp test is not the bound but the optimum. ``minimize x + y`` with ``x``
pinned at 1 answers 1 in discopt and answered 0 in every exported file. These
tests round-trip MPS and LP through HiGHS -- a core dependency, so this is an
end-to-end check and not a string assertion about a format the reader may be
interpreting differently than the solver does.

``.nl`` was already correct (bound code 4, "fixed"), and is asserted here as a
cross-writer control so a future regression there is caught by the same file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import discopt.modeling as dm
import highspy
import numpy as np
import pytest
from discopt.export import to_gams, to_lp, to_mps, to_nl


def _pinned_model(value: float = 1.0) -> dm.Model:
    """``minimize x + y`` over binaries, with ``x`` pinned at *value*.

    The objective is chosen so that dropping the pin *changes the answer*:
    minimizing pushes ``x`` to 0, which only the pin prevents. Pinning at 1 and
    at 0 are both exercised -- at 0 the pin agrees with the minimizer, so the
    objective cannot detect it and the column bound is what matters.
    """
    m = dm.Model("pinned_binary")
    x = m.binary("x")
    y = m.binary("y")
    m.minimize(x + y)
    m.subject_to(x + y >= 0)
    x.fix(value)
    return m


def _read_with_highs(text: str, suffix: str):
    """Parse an exported model with HiGHS; return (lower, upper, optimum)."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"model{suffix}"
        p.write_text(text)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        status = h.readModel(str(p))
        assert status == highspy.HighsStatus.kOk, f"HiGHS refused the {suffix} export"
        lp = h.getLp()
        names = [h.getColName(i)[1] for i in range(lp.num_col_)]
        lower = dict(zip(names, lp.col_lower_))
        upper = dict(zip(names, lp.col_upper_))
        h.run()
        obj = h.getObjectiveValue()
    return lower, upper, obj


@pytest.mark.parametrize("writer,suffix", [(to_mps, ".mps"), (to_lp, ".lp")])
@pytest.mark.parametrize("pin", [1.0, 0.0])
def test_pinned_binary_survives_roundtrip(writer, suffix, pin):
    """The exported column carries the pin, and the exported optimum matches."""
    lower, upper, obj = _read_with_highs(writer(_pinned_model(pin)), suffix)

    assert lower["x"] == pytest.approx(pin)
    assert upper["x"] == pytest.approx(pin)
    # y is untouched: a writer that fixed every binary would also pass the
    # assertions above, so pin the control too.
    assert lower["y"] == pytest.approx(0.0)
    assert upper["y"] == pytest.approx(1.0)
    # The answer discopt would give. Before the fix this was 0.0 for pin=1.
    assert obj == pytest.approx(pin)


@pytest.mark.parametrize("writer,suffix", [(to_mps, ".mps"), (to_lp, ".lp")])
def test_default_binary_keeps_the_shorthand(writer, suffix):
    """A binary with its declared box is still exported as an ordinary binary.

    The guard must not turn every binary into an explicitly bounded column;
    that would be a different (if harmless) drift, and it would mean the guard
    is not testing what it claims to.
    """
    m = dm.Model("plain_binary")
    x = m.binary("x")
    m.minimize(x)
    m.subject_to(x >= 0)

    text = writer(m)
    lower, upper, _obj = _read_with_highs(text, suffix)
    assert lower["x"] == pytest.approx(0.0)
    assert upper["x"] == pytest.approx(1.0)

    if suffix == ".mps":
        assert " BV BND  x" in text
        assert "FX BND  x" not in text
    else:
        assert "Binaries" in text
        # No explicit bound line for a default binary.
        bounds = text.split("Bounds", 1)[1].split("Generals")[0].split("Binaries")[0]
        assert "x" not in bounds


def test_partially_pinned_binary_array():
    """A ``where=`` mask pins some elements and not others (EX-3, array case).

    One narrowed element disqualifies the whole block from the shorthand, since
    the shorthand is declared per variable and not per element.
    """
    m = dm.Model("masked")
    x = m.binary("x", shape=(3,))
    m.minimize(dm.sum(x))
    m.subject_to(dm.sum(x) >= 0)
    x.fix(np.array([1.0, 0.0, 0.0]), where=[True, False, False])

    lower, upper, obj = _read_with_highs(to_mps(m), ".mps")
    pinned = [n for n in lower if lower[n] == pytest.approx(1.0)]
    assert len(pinned) == 1, f"exactly one element should be pinned, got {pinned}"
    assert obj == pytest.approx(1.0)

    lower, upper, obj = _read_with_highs(to_lp(m), ".lp")
    assert sum(1 for n in lower if lower[n] == pytest.approx(1.0)) == 1
    assert obj == pytest.approx(1.0)


@pytest.mark.parametrize("pin", [1.0, 0.0])
def test_gams_emits_the_pin(pin):
    """GAMS has no round-trip here, so assert the ``.lo``/``.up`` it must emit."""
    gms = to_gams(_pinned_model(pin))
    assert f"x.lo = {pin};" in gms
    assert f"x.up = {pin};" in gms
    # The control binary keeps the implicit box.
    assert "y.lo" not in gms
    assert "y.up" not in gms


def test_gams_default_binary_emits_no_bounds():
    m = dm.Model("plain_binary")
    x = m.binary("x")
    m.minimize(x)
    m.subject_to(x >= 0)
    gms = to_gams(m)
    assert "Binary Variables x;" in gms
    assert "x.lo" not in gms
    assert "x.up" not in gms


def test_nl_writer_control():
    """``.nl`` was already correct; assert it so a regression there is caught.

    Bound code 4 is "fixed at the following value"; code 0 is a two-sided
    range. ``x`` is pinned at 1, ``y`` is free over [0, 1].
    """
    lines = to_nl(_pinned_model(1.0)).splitlines()
    b = next(k for k, ln in enumerate(lines) if ln.startswith("b"))
    assert lines[b + 1] == "4 1.0"
    assert lines[b + 2] == "0 0.0 1.0"
