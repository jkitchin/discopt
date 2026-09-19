"""Issue #1345: every default-OFF gate over solver math carries an audit verdict.

CLAUDE.md §5's retirement clause is only worth writing if a new flag cannot quietly
skip it. This test enumerates the default-OFF flags from source and asserts each one
is either explicitly out of scope (a numeric knob, or a gate over non-solver
behaviour) or has a row in ``docs/dev/flag-retirement-audit.md``.

Adding a default-OFF gate over solver math therefore fails here until its row exists,
which is the enforcement the clause needs.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
PKG = REPO / "python" / "discopt"
AUDIT = REPO / "docs" / "dev" / "flag-retirement-audit.md"

_READ = re.compile(r'environ\.get\(\s*["\'](DISCOPT_[A-Z0-9_]+)["\']\s*,\s*(["\'][^"\']*["\'])')

#: Numeric knobs: ``0`` is a value (zero offset, zero rounds), not an off-switch.
KNOBS = {"DISCOPT_HEUR_OFFSET", "DISCOPT_ROOT_CUT_ROUNDS"}

#: Booleans that gate NON-solver behaviour — an import strategy, a compilation cache,
#: a subprocess daemon. No §5 panel can apply to them.
NON_SOLVER = {
    "DISCOPT_EAGER_IMPORTS",
    "DISCOPT_DISABLE_JAX_CACHE",
    "DISCOPT_GAMS_NO_DAEMON",
}


def _default_off_flags() -> dict[str, list[str]]:
    """Flags read with a literal ``"0"`` default, mapped to their read sites."""
    found: dict[str, list[str]] = {}
    scanned = 0
    for path in sorted(PKG.rglob("*.py")):
        scanned += 1
        for i, line in enumerate(path.read_text().splitlines(), 1):
            for m in _READ.finditer(line):
                if m.group(2).strip("\"'") == "0":
                    found.setdefault(m.group(1), []).append(f"{path.relative_to(PKG)}:{i}")
    assert scanned > 100, f"only scanned {scanned} files - probe measured nothing"
    assert found, "no default-OFF flags found - the regex stopped matching"
    return found


def test_the_audit_document_exists():
    assert AUDIT.is_file(), f"{AUDIT} is the standing record CLAUDE.md §5 points at"


def test_every_solver_math_gate_has_an_audit_verdict():
    flags = _default_off_flags()
    audit = AUDIT.read_text()

    gates = sorted(set(flags) - KNOBS - NON_SOLVER)
    assert gates, "classification removed every flag - the exclusion sets are wrong"

    missing = [g for g in gates if f"`{g}`" not in audit]
    assert not missing, (
        "default-OFF gate(s) over solver math with no row in "
        f"docs/dev/flag-retirement-audit.md: {missing}. Per CLAUDE.md §5, a new gate "
        "adds its row in the same PR that introduces the flag — graduate, retire, or "
        "keep as a documented opt-out."
    )
    # Sanity floor against the classification silently collapsing. Deliberately well
    # below the current count: RETIREMENT legitimately shrinks this population, and a
    # floor pinned to "today's number" would fail every time the rule works (#1357 took
    # it from 14 to 13 and tripped exactly that).
    assert len(gates) >= 8, f"classification collapsed: only {len(gates)} solver-math gates"


@pytest.mark.parametrize("excluded", sorted(KNOBS | NON_SOLVER))
def test_the_out_of_scope_flags_are_still_read_as_claimed(excluded):
    """Guard the exclusion lists: an excluded flag that vanished should be removed."""
    flags = _default_off_flags()
    assert excluded in flags, (
        f"{excluded} is on an exclusion list in this test but is no longer read with a "
        '"0" default. Drop it from the list rather than leaving a stale exemption.'
    )


def test_the_audit_states_the_three_states_from_claude_md():
    audit = AUDIT.read_text().lower()
    for state in ("graduated", "retire", "documented opt-out"):
        assert state in audit, f"the audit never mentions the '{state}' state"
