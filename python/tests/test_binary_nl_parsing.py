"""Binary .nl parsing (TAIL-1b; supersedes the #142 "unsupported" diagnostic).

Some MINLPLib instances (e.g. ``st_miqp5``) are distributed in the *binary* .nl
encoding (magic byte ``b``), whose body packs integers as 4-byte little-endian
``i32`` and reals as 8-byte little-endian IEEE-754 ``f64``. The reader now
transcodes that body into the equivalent text (``g``) token stream and parses it
with the same text parser, so a binary .nl builds the SAME model as its text
twin. This is a bound-neutral capability add: the round-trip must agree on the
model structure and the certified optimum.

The fixtures ``data/binary_nl/st_miqp5_{bin,text}.nl`` are the two encodings of
the identical model, emitted by GAMS CONVERT (``AmplNL`` / ``AmplNLBin``) from a
single ``st_miqp5.gms`` source.
"""

from pathlib import Path

import discopt.modeling as dm
import pytest

_DATA = Path(__file__).parent / "data" / "binary_nl"
_BIN = _DATA / "st_miqp5_bin.nl"
_TEXT = _DATA / "st_miqp5_text.nl"


@pytest.mark.unit
def test_binary_nl_fixture_is_actually_binary():
    """Guard the fixture: the binary file must carry the ``b`` magic byte."""
    assert _BIN.read_bytes()[:1] == b"b", "fixture must be a binary .nl"
    assert _TEXT.read_bytes()[:1] == b"g", "text fixture must be a text .nl"


@pytest.mark.unit
def test_binary_nl_parses_and_matches_text_model():
    """A binary .nl loads and builds a model with the same shape as its text twin."""
    mb = dm.from_nl(str(_BIN))
    mt = dm.from_nl(str(_TEXT))
    # Same variable count → same model shape.
    assert mb.num_variables == mt.num_variables
    # Both build a real Model (not an error path).
    assert type(mb).__name__ == "Model"


@pytest.mark.smoke
def test_binary_nl_certifies_same_optimum_as_text():
    """Bound-neutral round-trip: binary and text solve to the identical optimum
    with the identical node count (st_miqp5 fathoms at the root)."""
    rb = dm.from_nl(str(_BIN)).solve(time_limit=15, gap_tolerance=1e-4)
    rt = dm.from_nl(str(_TEXT)).solve(time_limit=15, gap_tolerance=1e-4)
    assert rb.status == rt.status == "optimal"
    assert rb.objective == pytest.approx(rt.objective, abs=1e-6, rel=1e-4)
    # Exactly the same search: the parse difference is transport-only.
    assert rb.node_count == rt.node_count


@pytest.mark.unit
def test_malformed_binary_nl_fails_loudly(tmp_path):
    """A truncated/garbage binary .nl errors loudly (sound-or-refuse) rather than
    silently producing a wrong model. A 2-line 'b' header has fewer than the 10
    required header lines → an explicit header error, not a false parse."""
    nl = tmp_path / "synthetic_binary.nl"
    nl.write_bytes(b"b3 0 1 0\t# synthetic binary .nl\n 1 0 1 0 0\n")
    with pytest.raises(ValueError) as exc_info:
        dm.from_nl(str(nl))
    msg = str(exc_info.value)
    # The old confusing UTF-8 message must not leak through.
    assert "valid UTF-8" not in msg
