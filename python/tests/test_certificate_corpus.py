"""Generalization + CLI tests for feasibility certificates on real ``.nl`` models.

These solve genuine MINLPLib instances from the in-repo corpus and drive the
``discopt`` CLI, so they are marked ``slow``. The point is to show the emitter is
a *general* solution (CLAUDE.md #2): it certifies real multi-variable models, and
it conservatively *refuses* what the Tier-1 exact-rational checker cannot verify
(transcendental functions), rather than emitting something a checker might wrongly
accept.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from discopt.certificate import (
    build_feasibility_certificate,
    check_certificate,
)
from discopt.modeling.core import from_nl

DATA = Path(__file__).parent / "data" / "minlplib_nl"


def _nl(name: str) -> Path:
    p = DATA / f"{name}.nl"
    if not p.exists():
        pytest.skip(f"instance {name}.nl not in corpus")
    return p


@pytest.mark.slow
@pytest.mark.parametrize("name", ["alan", "nvs03"])
def test_real_instance_certifies(name):
    """A genuine rational-arithmetic instance emits a certificate the reference
    checker accepts."""
    m = from_nl(str(_nl(name)))
    r = m.solve(time_limit=20)
    if r.x is None or r.objective is None:
        pytest.skip(f"{name}: solve produced no incumbent (status={r.status})")
    cert = build_feasibility_certificate(m, r)
    ok, reason = check_certificate(cert)
    assert ok, f"{name}: {reason}"
    # The certificate is self-contained and multi-column.
    assert cert["certificate"]["model"]["n_columns"] >= 1


@pytest.mark.slow
def test_transcendental_instance_is_refused():
    """nvs01 contains sqrt; the exact-rational checker must refuse, not accept."""
    m = from_nl(str(_nl("nvs01")))
    r = m.solve(time_limit=20)
    if r.x is None or r.objective is None:
        pytest.skip(f"nvs01: solve produced no incumbent (status={r.status})")
    cert = build_feasibility_certificate(m, r)  # emission succeeds (fn nodes allowed)
    ok, reason = check_certificate(cert)
    assert not ok and "transcendental" in reason.lower()


@pytest.mark.slow
def test_cli_emit_and_cert_check_round_trip(tmp_path, monkeypatch):
    """`discopt solve --emit-certificate` writes a cert `discopt cert-check` accepts."""
    from discopt.cli import main

    nl = _nl("alan")

    def _run(argv):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            main()
        return exc.value.code

    code = _run(
        [
            "discopt",
            "solve",
            str(nl),
            "--emit-certificate",
            "--no-daemon",
            "--out-dir",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert code in (0, 1)  # solve exit reflects status, not certificate
    cert_path = tmp_path / "alan.cert.json"
    assert cert_path.exists(), "certificate file not written"
    assert check_certificate(json.loads(cert_path.read_text()))[0]

    # cert-check accepts the genuine certificate...
    assert _run(["discopt", "cert-check", str(cert_path)]) == 0

    # ...and rejects a tampered one (exit 1).
    bad = json.loads(cert_path.read_text())
    bad["certificate"]["incumbent"]["objectiveValue"] = [10**9, 1]
    bad_path = tmp_path / "alan_bad.cert.json"
    bad_path.write_text(json.dumps(bad))
    assert _run(["discopt", "cert-check", str(bad_path)]) == 1
