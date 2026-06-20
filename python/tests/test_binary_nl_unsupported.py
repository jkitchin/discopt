"""Regression test for issue #142: binary .nl format diagnostic.

Some MINLPLib instances are distributed in the *binary* .nl encoding (magic byte
``b``), whose body embeds raw little-endian IEEE-754 doubles. The old reader did a
UTF-8 ``read_to_string`` and only handled the text (``g``) format, so loading such
a file failed with a confusing

    ValueError: .nl parse error: ... stream did not contain valid UTF-8

The reader now detects the format from the leading magic byte and emits an
explicit, accurate diagnostic instead of the UTF-8 noise. The parser rejects on
the leading ``b`` of the header (before reading the binary body), so a synthetic
one-line header is sufficient to exercise the diagnostic — no vendored binary
fixture is required.
"""

import discopt.modeling as dm
import pytest


@pytest.mark.unit
def test_binary_nl_explicit_error(tmp_path):
    """Loading a binary .nl raises a clear 'binary not supported' error."""
    nl = tmp_path / "synthetic_binary.nl"
    # A ``b``-prefixed header line is the binary-format marker; the reader
    # rejects it before attempting to decode the (binary) body.
    nl.write_bytes(b"b3 0 1 0\t# synthetic binary .nl\n 1 0 1 0 0\n")

    with pytest.raises(ValueError) as exc_info:
        dm.from_nl(str(nl))

    msg = str(exc_info.value)
    assert "binary .nl format not supported" in msg
    # The old confusing UTF-8 message must not leak through.
    assert "valid UTF-8" not in msg
