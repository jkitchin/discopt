"""Ratchet over the LP/QP Rust-Python boundary (docs/dev/lp-qp-boundary.md).

The contract in that doc's §0 is: *Python turns a Model into a problem, Rust turns
a problem into a certified result, and nothing that enters the certificate is
computed in Python.* §2 of the doc is the audit of where that line is crossed
today.

This module pins the audit. Each test fails when a violation class **grows** --
a new Python implementation of a safe bound, a new equilibration, a new
standard-form round trip -- and fails with an explicit instruction when one
**shrinks**, so the pinned count is updated in the PR that earns it. The counts
only ever go down.

Deliberately source-level: it reads files with ``re`` and never imports
``discopt._rust``, so it runs in an environment where the extension is not built.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PY = _REPO / "python" / "discopt"
_RS = _REPO / "crates"


def _iter_sources(root: Path, suffix: str) -> list[Path]:
    return sorted(p for p in root.rglob(f"*{suffix}") if "/tests/" not in p.as_posix())


def _find(pattern: str, root: Path, suffix: str) -> list[str]:
    """Return ``path:line: text`` for every line matching ``pattern``."""
    rx = re.compile(pattern)
    hits: list[str] = []
    for path in _iter_sources(root, suffix):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - unreadable file is a real failure
            raise
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                rel = path.relative_to(_REPO).as_posix()
                hits.append(f"{rel}:{n}: {line.strip()}")
    return hits


def _ratchet(label: str, hits: list[str], pinned: int, note: str) -> None:
    """Assert ``len(hits) == pinned`` with a directional message."""
    found = len(hits)
    listing = "\n".join(f"    {h}" for h in hits)
    if found > pinned:
        pytest.fail(
            f"{label}: {found} occurrences, pinned at {pinned} -- the LP/QP "
            f"boundary got WORSE.\n{note}\n"
            f"See docs/dev/lp-qp-boundary.md §0 for the contract.\n{listing}"
        )
    if found < pinned:
        pytest.fail(
            f"{label}: {found} occurrences, pinned at {pinned} -- the boundary "
            f"got BETTER. Lower the pinned count in this test and update the "
            f"§2 audit in docs/dev/lp-qp-boundary.md in the same PR.\n{listing}"
        )


# --- V1: the safe dual bound (the certificate) -------------------------------

#: Python evaluators of the Neumaier-Shcherbina bound, plus the dispatcher and the
#: regularized sweep that select/drive them. Excludes ``shor_sdp_safe_dual_bound``
#: (SDP, a different problem class -- see the §2 parenthetical).
_NS_PY = r"^\s*def (_ns_safe_lp_lower_bound|_safe_lp_lower_bound\w*|_refined_safe_bound\w*)\s*\("

#: Rust production evaluators (the ``#[cfg(test)]`` reference copy in primal.rs is
#: not ``pub`` and so does not match).
_NS_RS = r"^\s*pub fn \w*safe_bound\w*\s*\("


@pytest.mark.smoke
def test_v1_python_safe_bound_implementations_do_not_grow():
    """No new Python implementation of the safe dual bound.

    The bound IS the certificate; it must converge on one implementation in Rust
    (docs/dev/lp-qp-boundary.md §3 step 1). Measured 2026-09-12: the two LP-path
    evaluators are both sound (0/132 violations of g <= p*) but disagree on
    111/132 random LPs, by up to 2.4e-3 relative in the ill-conditioned regime --
    so which bound a node gets depends on which path reached it.
    """
    hits = _find(_NS_PY, _PY, ".py")
    _ratchet(
        "V1 Python safe-bound implementations",
        hits,
        pinned=5,
        note=(
            "A safe dual bound computed in Python is a certificate computed on "
            "the producer side of the boundary. Add the evaluation to "
            "crates/discopt-core/src/lp/simplex/refine.rs and call it."
        ),
    )


@pytest.mark.smoke
def test_v1_rust_safe_bound_implementations_do_not_grow():
    """The Rust side converges too: dense + CSC, not one per call site."""
    hits = _find(_NS_RS, _RS, ".rs")
    _ratchet(
        "V1 Rust safe-bound implementations",
        hits,
        pinned=2,
        note=(
            "ns_safe_bound and ns_safe_bound_csc are the dense/CSC pair. A third "
            "means a call site grew its own rigor model -- generalize instead."
        ),
    )


# --- V2: LP scaling ----------------------------------------------------------

_EQUILIBRATE_PY = r"^\s*def \w*equilibrate\w*\s*\("


@pytest.mark.smoke
def test_v2_python_equilibration_does_not_grow():
    """Scaling decides which vertex the simplex lands on, so it decides the bound.

    Two Python implementations plus the Rust one means the same LP is conditioned
    differently depending on which engine will solve it
    (_relax/milp_relaxation.py:749 selects on ``backend != "simplex"``).

    The pinned 3 are the two implementations (``equilibrate_relaxation_lp``,
    ``_equilibrate_rows``) and the one solve wrapper that applies them
    (``_solve_lp_warm_equilibrated``). The wrapper is counted deliberately: a
    narrower detector that excluded it would also miss a new implementation
    spelled as a method.
    """
    hits = _find(_EQUILIBRATE_PY, _PY, ".py")
    _ratchet(
        "V2 Python equilibration definitions",
        hits,
        pinned=3,
        note=(
            "Scaling is solver numerics: it belongs next to the simplex that "
            "consumes it (crates/discopt-core/src/lp/simplex/scaling.rs)."
        ),
    )


# --- V3: the standard-form round trip ----------------------------------------

_DECOMPOSE_CALLS = r"_decompose_eq_slack_form(_sparse)?\("


@pytest.mark.smoke
def test_v3_standard_form_round_trip_does_not_grow():
    """Python builds ``[A|I]`` and then re-derives the row senses back out of it.

    ``_decompose_eq_slack_form`` classifies a row as an inequality by inspecting
    its slack coefficient against 1e-15 -- structural information the producer
    knew exactly and discarded. Every new call site deepens the inversion.
    """
    hits = [h for h in _find(_DECOMPOSE_CALLS, _PY, ".py") if " def " not in h]
    _ratchet(
        "V3 standard-form round-trip call sites",
        hits,
        pinned=12,
        note=(
            "Have the producer carry row senses alongside the matrices instead of "
            "re-deriving them from float comparisons on the slack block."
        ),
    )


# --- V4: Python repairing solver output --------------------------------------

_SNAP_TOL = r"_BOUND_SNAP_TOL"


@pytest.mark.smoke
def test_v4_python_side_solution_repair_does_not_grow():
    """Python must not silently move a point, on the way in or on the way out.

    Two unrelated repairs share the name ``_BOUND_SNAP_TOL`` with tolerances four
    orders of magnitude apart, which is itself the hazard:

    * ``lp_simplex.py:46`` = **1e-3**, applied to the solver's OUTPUT -- a
      returned point up to 1e-3 outside its box is snapped back onto it;
    * ``lp_pounce.py:56`` = **1e-7**, applied to the solver's INPUT -- a
      ``lb > ub`` inversion up to 1e-7 is collapsed to its midpoint before POUNCE
      sees it, because POUNCE rejects an inverted bound that HiGHS would presolve
      away.

    Both guards stay until their Rust equivalents exist (§3 step 4 -- never remove
    a guard and add its replacement in separate PRs), but neither may spread.
    """
    hits = _find(_SNAP_TOL, _PY, ".py")
    _ratchet(
        "V4 Python-side solution repair",
        hits,
        pinned=5,
        note=(
            "A solver that cannot validate its own output has the validation on "
            "the wrong side of the boundary. Port the check into the engine that "
            "produced the point."
        ),
    )


@pytest.mark.smoke
def test_boundary_contract_doc_exists():
    """The ratchet is meaningless without the contract it ratchets against."""
    doc = _REPO / "docs" / "dev" / "lp-qp-boundary.md"
    assert doc.is_file(), f"missing the LP/QP boundary contract at {doc}"
    text = doc.read_text(encoding="utf-8")
    assert "§0 The line (binding)" in text, "the doc lost its binding §0 contract"
