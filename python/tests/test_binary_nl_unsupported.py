"""Regression test for issue #142: binary .nl format diagnostic.

MINLPLib ``portfol_robust050_34`` is distributed in the *binary* .nl encoding
(magic byte ``b``), whose body embeds raw little-endian IEEE-754 doubles. The
old reader did a UTF-8 ``read_to_string`` and only handled the text (``g``)
format, so loading this instance failed with a confusing

    ValueError: .nl parse error: ... stream did not contain valid UTF-8

The reader now detects the format from the leading magic byte and emits an
explicit, accurate diagnostic instead of the UTF-8 noise.
"""

from pathlib import Path

import discopt.modeling as dm
import pytest

_NL = Path(__file__).parent / "data" / "minlplib" / "portfol_robust050_34.nl"


@pytest.mark.unit
def test_binary_nl_explicit_error():
    """Loading a binary .nl raises a clear 'binary not supported' error."""
    assert _NL.exists(), f"missing {_NL}"
    # Confirm the fixture really is the binary format.
    assert _NL.read_bytes().lstrip()[:1] == b"b"

    with pytest.raises(ValueError) as exc_info:
        dm.from_nl(str(_NL))

    msg = str(exc_info.value)
    assert "binary .nl format not supported" in msg
    # The old confusing UTF-8 message must not leak through.
    assert "valid UTF-8" not in msg
