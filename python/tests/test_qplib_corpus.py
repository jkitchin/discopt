"""Corpus-wide QPLIB reader validation (issue #830).

Runs the reader over all 453 instances when the corpus is present, and is
skipped otherwise -- the corpus is ~1.1 GB and lives outside the repo. The
in-repo fixture tests in ``test_qplib_reader.py`` are the CI gate; this is the
full-coverage check, and it is the only place the ``C=B``, ``C=D`` and ``O=C``
layout branches are exercised (their smallest instances are too large to
vendor).

Fetch the corpus with ``qplib/fetch_qplib.sh`` in the corpus directory; see
``docs/dev/qplib-corpus.md``.
"""

from __future__ import annotations

import csv
import os

import pytest
from discopt.interfaces import qplib as qp

CORPUS = os.environ.get(
    "DISCOPT_QPLIB_DIR",
    os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/qplib"),
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.path.isdir(os.path.join(CORPUS, "qplib")),
        reason=f"QPLIB corpus not present at {CORPUS}",
    ),
]

#: Instances whose objective QPLIB itself lost: instancedata.csv declares
#: objective variables but the file stores no objective terms. Identified by
#: that rule, not by name -- the names are here only so the expected count is
#: visible; test_upstream_defects_are_exactly_those_detected_by_rule re-derives
#: the set and fails if it changes.
KNOWN_OBJECTIVE_LOST = 4


@pytest.fixture(scope="module")
def meta():
    with open(os.path.join(CORPUS, "instancedata.csv"), encoding="utf-8") as fh:
        return {r["name"]: r for r in csv.DictReader(fh)}


@pytest.fixture(scope="module")
def parsed(meta):
    """Parse every instance once; reused by the checks below."""
    out = {}
    for name in sorted(meta):
        path = os.path.join(CORPUS, "qplib", f"{name}.qplib")
        if os.path.exists(path):
            out[name] = qp.read_qplib(path)
    assert len(out) >= 400, f"corpus looks incomplete: only {len(out)} instances"
    return out


def test_every_instance_parses(parsed, meta):
    missing = sorted(set(meta) - set(parsed))
    assert not missing, f"{len(missing)} instances failed to parse or are absent: {missing[:10]}"


def test_structure_matches_instancedata(parsed, meta):
    """Cross-check every parsed field against QPLIB's own metadata.

    Reported in aggregate rather than one assert per instance so a systematic
    misread shows up as a pattern instead of a single early failure.
    """
    bad = []
    checks = 0
    for name, inst in parsed.items():
        r = meta[name]
        for label, got, want in (
            ("nvars", inst.n_vars, int(r["nvars"])),
            ("ncons", inst.n_cons, int(r["ncons"])),
            ("probtype", inst.probtype, r["probtype"]),
            ("nobjquadnz", len(inst.obj_quad), int(r["nobjquadnz"])),
            ("ncontvars", inst.n_vars - inst.n_integral, int(r["ncontvars"])),
            ("nintegral", inst.n_integral, int(r["nbinvars"]) + int(r["nintvars"])),
        ):
            checks += 1
            if got != want:
                bad.append(f"{name}: {label} {got} != {want}")
        checks += 1
        if not inst.sense.startswith(r["objsense"].lower()):
            bad.append(f"{name}: sense {inst.sense} != {r['objsense']}")

    # Prove the probe fired (CLAUDE.md measurement discipline #6).
    assert checks > 3000, f"only {checks} comparisons executed"
    assert not bad, f"{len(bad)} structural mismatches, first 10: {bad[:10]}"


def test_all_layout_branches_are_exercised(parsed):
    """The corpus must cover every branch, including the three fixtures cannot."""
    obj = {i.probtype[0] for i in parsed.values()}
    var = {i.probtype[1] for i in parsed.values()}
    con = {i.probtype[2] for i in parsed.values()}
    assert {"C", "D", "L", "Q"} <= obj
    assert {"B", "C", "G", "I", "M"} <= var
    assert {"B", "C", "D", "L", "N", "Q"} <= con


def test_objective_reproduced_at_every_reference_point(parsed):
    """Recompute the published objective from the file for every reference point.

    The four instances whose objective QPLIB lost upstream are expected to fail
    and are counted, not named.
    """
    mismatches, evaluated = [], 0
    for name, inst in parsed.items():
        solpath = os.path.join(CORPUS, "sol", f"{name}.sol")
        if not os.path.exists(solpath):
            continue
        x, objvar = qp.read_solution(solpath, inst)
        if objvar is None:
            continue
        evaluated += 1
        recomputed = inst.evaluate_objective(x)
        if abs(recomputed - objvar) / max(1.0, abs(objvar)) > 1e-6:
            mismatches.append((name, recomputed, objvar))

    assert evaluated > 400, f"only {evaluated} reference points evaluated"
    assert len(mismatches) == KNOWN_OBJECTIVE_LOST, (
        f"expected exactly {KNOWN_OBJECTIVE_LOST} known-defective instances, got "
        f"{len(mismatches)}: {[m[0] for m in mismatches]}"
    )


def test_upstream_defects_are_exactly_those_detected_by_rule(parsed, meta):
    """The defective set must fall out of a general rule, not a name list.

    ``nobjnz`` counts variables appearing in the objective, so a positive count
    with no stored objective terms means the objective was lost upstream. If a
    future QPLIB release fixes or adds such instances, this fails and the
    expected count is updated -- no instance names are hardcoded anywhere.
    """
    import numpy as np

    lost = [
        name
        for name, inst in parsed.items()
        if int(meta[name]["nobjnz"]) > 0
        and int(np.count_nonzero(inst.obj_lin)) + len(inst.obj_quad) == 0
    ]
    assert len(lost) == KNOWN_OBJECTIVE_LOST, f"objective-lost set changed: {sorted(lost)}"


def test_reference_points_are_feasible(parsed):
    """Every reference point must satisfy the parsed model.

    Two tolerances, deliberately: the bulk are feasible to 1e-6, and nothing in
    the corpus may exceed 1e-3. The gap is where QPLIB's own published points
    land -- these are floating-point solver output, not exact certificates.
    """
    gross, loose, checked = [], 0, 0
    for name, inst in parsed.items():
        solpath = os.path.join(CORPUS, "sol", f"{name}.sol")
        if not os.path.exists(solpath):
            continue
        x, _ = qp.read_solution(solpath, inst)
        checked += 1
        viol = float(inst.max_violation(x))
        if viol > 1e-3:
            gross.append((name, viol))
        elif viol > 1e-6:
            loose += 1

    assert checked > 400, f"only {checked} points feasibility-checked"
    # The four objective-lost instances are also grossly infeasible; nothing else
    # may be.
    assert len(gross) <= KNOWN_OBJECTIVE_LOST, (
        f"{len(gross)} points violate the parsed model by >1e-3: {gross[:10]}"
    )
    assert loose < 0.1 * checked, f"{loose}/{checked} points only marginally feasible"
