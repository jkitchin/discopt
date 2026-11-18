"""HiGHS 1.16.0's readings of the LP-reader probe cases, frozen for ``test_lp_reader.py``.

discopt's LP reader (``discopt._lpfile``) ports HiGHS 1.16.0's reader (#2221), whose
readings differ from every earlier HiGHS on many of the synthetic cases in
``_lp_probe_cases.py`` (1.15.1 misread or rejected 81 of them). The test compares
against live highspy when highspy >= 1.16.0 is installed; the supported floor is
older and highspy is optional, so on any other install it compares against this
fixture instead. The comparison is the same exact one (``_mps_differential``), kError
included: a case HiGHS rejects is stored as ``null``.

Regenerate (only with highspy 1.16.0 itself; the script refuses anything else)::

    python python/tests/_regen_lp_reader_fixture.py

Every case is keyed by its index and tag and carries the SHA-256 of its text, so an
edited or added probe case fails the fixture test until the fixture is regenerated.
Floats are stored with Python's shortest round-tripping ``repr`` (``json`` writes
``Infinity``/``NaN``), so the stored readings are bit-exact.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "data", "lp_reader", "highs_1_16_0_readings.json.gz")
HIGHS_VERSION = "1.16.0"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _floats(a) -> list[float]:
    return [float(v) for v in np.asarray(a, dtype=np.float64)]


def _csc(M) -> dict:
    M = sp.csc_matrix(M)
    return {
        "shape": list(M.shape),
        "indptr": [int(v) for v in M.indptr],
        "indices": [int(v) for v in M.indices],
        "data": _floats(M.data),
    }


def _uncsc(d: dict):
    return sp.csc_matrix(
        (
            np.array(d["data"], dtype=np.float64),
            np.array(d["indices"], dtype=np.int64),
            np.array(d["indptr"], dtype=np.int64),
        ),
        shape=tuple(d["shape"]),
    )


def encode(reading: dict | None) -> dict | None:
    """A ``highs_arrays`` reading as JSON-able data (``None``: HiGHS's kError)."""
    if reading is None:
        return None
    out = {k: int(reading[k]) for k in ("n", "m")}
    out["offset"] = float(reading["offset"])
    out["maximize"] = bool(reading["maximize"])
    for k in ("c", "col_lb", "col_ub", "row_lo", "row_hi"):
        out[k] = _floats(reading[k])
    out["integrality"] = [int(v) for v in reading["integrality"]]
    out["col_names"] = list(reading["col_names"])
    out["row_names"] = list(reading["row_names"])
    # The matrix as stored by HiGHS (column-wise; differences() canonicalises it).
    out["A"] = _csc(reading["A"])
    # The Hessian raw: its entry order is part of the contract.
    h = reading["hessian"]
    out["hessian"] = None if h is None else _csc(h)
    return out


def decode(data: dict | None) -> dict | None:
    """The inverse of :func:`encode`, in ``highs_arrays``'s format."""
    if data is None:
        return None
    out = dict(data)
    for k in ("c", "col_lb", "col_ub", "row_lo", "row_hi"):
        out[k] = np.array(data[k], dtype=np.float64)
    out["integrality"] = np.array(data["integrality"], dtype=np.int64)
    out["A"] = _uncsc(data["A"])
    out["hessian"] = None if data["hessian"] is None else _uncsc(data["hessian"])
    return out


def load_fixture() -> dict:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    import highspy

    sys.path.insert(0, HERE)
    from _lp_probe_cases import PROBE_CASES
    from _mps_differential import highs_arrays

    version = highspy.Highs().version()
    if version != HIGHS_VERSION:
        sys.exit(f"the fixture holds HiGHS {HIGHS_VERSION}'s readings; highspy is {version}")
    cases = []
    n_read = n_error = 0
    with tempfile.TemporaryDirectory(prefix="lpfixture_") as tmp:
        path = os.path.join(tmp, "m.lp")
        for k, (tag, text) in enumerate(PROBE_CASES):
            with open(path, "wb") as f:
                f.write(text.encode("utf-8"))
            try:
                reading = highs_arrays(path)
            except UnicodeDecodeError:
                raise  # no verdict: never store one
            except ValueError:
                reading = None
            n_error += reading is None
            n_read += reading is not None
            cases.append(
                {"k": k, "tag": tag, "sha256": text_sha256(text), "reading": encode(reading)}
            )
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    payload = json.dumps(
        {"highs_version": version, "cases": cases}, separators=(",", ":"), sort_keys=True
    )
    # mtime=0: regenerating identical readings gives an identical file.
    with open(FIXTURE, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(payload.encode("utf-8"))
    print(
        f"HiGHS {version}: {len(cases)} cases ({n_read} read, {n_error} kError) -> "
        f"{FIXTURE} ({os.path.getsize(FIXTURE)} bytes)"
    )
    return 0 if cases else 1


if __name__ == "__main__":
    sys.exit(main())
