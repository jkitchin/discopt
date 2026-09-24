"""Issue #1345/#1421: every default-OFF gate over solver math carries an audit verdict.

CLAUDE.md §5's retirement clause is only worth writing if a new flag cannot quietly
skip it. This test enumerates the default-OFF flags from source and asserts each one
is either explicitly out of scope (a knob, a selector, or a gate over non-solver
behaviour) or has a row in ``docs/dev/flag-retirement-audit.md``.

**What #1421 changed, and why the old version of this file was worse than no test.**
The scan was one regex — ``environ.get(F, "0")``. It saw 10 flags. Fifty were
invisible to it, including all 23 ``_env_flag(..., default=False)`` gates, several of
them bound-changing solver math (the ``DISCOPT_RLT`` family, ``SHOR_SDP_ROOT_BOUND``,
``PHASE2_DBBT``). The file argued that cross-checking the scan against the document's
``<!-- live-solver-math-gates: N -->`` marker was "strictly STRONGER" than a numeric
floor because "a regex that silently stops matching some gates now fails here". That
holds for a regex that *stops* matching; this one never matched those forms at all.
Scan and marker were derived from the same too-narrow definition, so they agreed with
each other while the real population was several times larger — two instruments, one
blind spot, mutual confirmation. The scan now lives in ``flag_audit_scan.py``, reads
all five forms, and decides polarity from the gate's own comparison rather than from
its default literal (five live flags default ON behind an empty-string default).

There is deliberately **no exemption table for untriaged gates**. Every default-OFF
solver-math gate the scan finds must have a row in the audit document, and #1421 wrote
the 37 that were missing. Where no measurement exists for a flag, its row says so —
the document's own words, from the first pass: *"panel owed, never run" is itself the
finding*. A row recording an absent panel is a verdict; no row at all is the defect §5
names, and a test-side "pending" list would have re-created exactly the invisibility
this file is being repaired for.

The only exemptions are ``OUT_OF_SCOPE`` (booleans that gate non-solver behaviour) and
``NOT_A_GATE`` (reads that are values, not switches), both small, both guarded by tests
that fail if an entry goes stale or its flag changes polarity.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from flag_audit_scan import (
    BOOLEAN_DEFAULTS,
    OFF_WORDS,
    ON_WORDS,
    polarity_of,
    scan,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
AUDIT = REPO / "docs" / "dev" / "flag-retirement-audit.md"

#: Default-OFF booleans that gate NON-solver behaviour — an import strategy, a
#: compilation cache, a subprocess daemon. No §5 panel can apply to them, so they
#: never owe a row. Classified by how the flag is CONSUMED, per §5's own wording.
OUT_OF_SCOPE = {
    "DISCOPT_EAGER_IMPORTS": "import strategy; cannot change a bound",
    "DISCOPT_DISABLE_JAX_CACHE": "compilation cache; cannot change a bound",
    "DISCOPT_GAMS_NO_DAEMON": "subprocess lifetime for the GAMS link",
}

#: Reads the scan cannot classify as a boolean gate, each with the reason it is not
#: one. These are values, not switches, so "default-OFF" is not a meaningful question
#: about them. Every ``unknown`` from the scan must appear here — an unaccounted
#: ``unknown`` fails ``test_no_read_is_left_unclassified``, which is the guard that
#: stops the blind spot from reopening somewhere new.
NOT_A_GATE = {
    "DISCOPT_DECOMP_STORE": "a filesystem path for the decomposition record store",
    "DISCOPT_LLM_MODEL": "a litellm model identifier",
    "DISCOPT_LLM_TIMEOUT": "a request timeout in seconds; 0 would be a duration, not an off-switch",
    "DISCOPT_PROVENANCE_AUTHOR": "an author name stamped into a provenance block",
    "DISCOPT_LP_SPATIAL_PLUNGE": (
        "a three-state override: unset defers to the caller's `require_incremental`, "
        "so it has no single default polarity. Tracked as a solver-math gate below."
    ),
}


def _gates() -> tuple[dict[str, list], dict[str, str]]:
    """(all sites by flag, default-OFF solver-math gates -> first read site)."""
    sites = scan()
    gates = {}
    for flag, s in sorted(sites.items()):
        if polarity_of(s) != "off" or flag in OUT_OF_SCOPE:
            continue
        gates[flag] = s[0].where
    # LP_SPATIAL_PLUNGE has no single default polarity (see NOT_A_GATE) but is
    # unambiguously a solver-math gate whose docstring says "Opt in with =1", so it is
    # audited as one rather than falling through the classification.
    for flag, s in sites.items():
        if flag == "DISCOPT_LP_SPATIAL_PLUNGE":
            gates[flag] = s[0].where
    return sites, gates


def test_the_audit_document_exists():
    assert AUDIT.is_file(), f"{AUDIT} is the standing record CLAUDE.md §5 points at"


def test_no_read_is_left_unclassified():
    """Every `DISCOPT_*` read lands in a named bucket; nothing falls off the edge.

    This is the guard the pre-#1421 scan lacked. It had no notion of a read it could
    not classify, so 50 flags were not "unclassified" — they were simply absent, and
    absence looks exactly like a clean bill of health.
    """
    sites = scan()
    unknown = {f: s for f, s in sites.items() if polarity_of(s) == "unknown"}
    unaccounted = sorted(set(unknown) - set(NOT_A_GATE))
    assert not unaccounted, (
        f"the scan cannot classify {unaccounted} and this test does not account for "
        "them either. Decide what each one is: if it is a value or a selector add it "
        "to NOT_A_GATE with the reason; if it is a boolean gate, teach "
        "flag_audit_scan.py the form it uses. Do not leave it undecided — an "
        "unclassified read is how #1421 happened."
    )
    mixed = sorted(f for f, s in sites.items() if polarity_of(s) == "mixed")
    assert not mixed, (
        f"{mixed} are read with DIFFERENT default polarity at different call sites. "
        "That is a defect in the flag, not in the scan: the same variable turns a "
        "feature on in one place and off in another."
    )


def test_every_default_off_solver_math_gate_has_an_audit_verdict():
    sites, gates = _gates()
    audit = AUDIT.read_text()
    assert gates, "classification removed every flag — the exclusion sets are wrong"

    # A ROW, not a mention. The first version of this test accepted any occurrence of
    # the flag name in the document, and two gates passed it on prose alone:
    # `DISCOPT_OBBT_ITERATE` (named in a paragraph about a verdict nobody acted on) and
    # `DISCOPT_RLT` (named only as "the whole DISCOPT_RLT family"). Both then turned out
    # to have decisive recorded verdicts. Measuring a mention is the same class of error
    # as measuring a docstring: it counts the flag being *spoken about*, not triaged.
    rows = {line.split("`")[1] for line in audit.splitlines() if line.startswith("| `DISCOPT_")}
    missing = [g for g in sorted(gates) if g not in rows]
    assert not missing, (
        "default-OFF gate(s) over solver math with no ROW in "
        f"docs/dev/flag-retirement-audit.md: {missing}. Per CLAUDE.md §5, a new gate "
        "adds its row in the same PR that introduces the flag — graduate, retire, or "
        "keep as a documented opt-out. `panel owed, never run` is a legitimate row "
        "and the audit says so; having NO row is the defect §5 names. A paragraph "
        "mentioning the flag is NOT a row: it does not state a verdict, and this test "
        "used to accept one."
    )

    # Cross-check the scan against the document's declared population. Unlike the
    # pre-#1421 version this is NOT mutual confirmation of one definition: the scan
    # now reads five forms and derives polarity independently of the default literal,
    # and `test_the_scan_sees_every_read_form` pins that it still does.
    m = re.search(r"<!--\s*live-solver-math-gates:\s*(\d+)\s*-->", audit)
    assert m, (
        "docs/dev/flag-retirement-audit.md has no `<!-- live-solver-math-gates: N -->` "
        "marker. It is the machine-readable count this test cross-checks the source "
        "scan against; restore it rather than removing the check."
    )
    declared = int(m.group(1))
    assert len(gates) == declared, (
        f"the source scan finds {len(gates)} default-OFF solver-math gates but "
        f"docs/dev/flag-retirement-audit.md declares {declared}. Either a gate was "
        f"added or retired without updating the marker, or the scan changed. "
        f"Scan: {sorted(gates)}"
    )


def test_the_exemption_tables_name_real_flags():
    """A stale entry is a silent exemption for a flag that no longer exists."""
    sites = scan()
    for label, table in (
        ("OUT_OF_SCOPE", OUT_OF_SCOPE),
        ("NOT_A_GATE", NOT_A_GATE),
    ):
        stale = sorted(set(table) - set(sites))
        assert not stale, (
            f"{label} names {stale}, which the source scan no longer reads. Drop the "
            "entry rather than leaving a stale exemption behind."
        )


def test_the_scan_sees_every_read_form():
    """The #1421 regression test. Fails against the pre-#1421 single-regex scan.

    Each form below is represented by a flag that form is the ONLY way to reach. The
    old scan saw the first one and none of the rest.
    """
    sites = scan()
    expected = {
        "DISCOPT_G_CONVEX_CUTS": "literal",  # environ.get(F, "0") -- the only old form
        "DISCOPT_RLT": "env_flag",  # _env_flag(F, default=False)
        "DISCOPT_NORM_ATOM": "no_default",  # environ.get(F)
        "DISCOPT_BLOCK_VECTOR_EVAL": "literal",  # via a module-level constant name
        "DISCOPT_NLPBB_ROOT_CUTS": "literal",  # environ.get(F, "") -- default ON
    }
    checked = 0
    for flag, form in expected.items():
        assert flag in sites, (
            f"{flag} is not seen by the scan at all. It is read via the {form!r} form; "
            "the scan has stopped recognising that form."
        )
        assert any(s.form == form for s in sites[flag]), (
            f"{flag} is seen but not as {form!r}: {[s.form for s in sites[flag]]}"
        )
        checked += 1
    assert checked == len(expected), "probe did not run every case"

    forms = {s.form for v in sites.values() for s in v}
    for form in ("literal", "env_flag", "no_default"):
        assert form in forms, f"no read of form {form!r} found — the scan narrowed"


def test_polarity_is_not_read_off_the_default_literal():
    """The trap #1421 names: `""` is the default of both ON and OFF gates.

    A widening that treated every empty default as OFF would misclassify all five of
    the default-ON gates below, and quietly demand audit rows for shipped defaults.
    """
    sites = scan()
    on_with_empty_default = [
        "DISCOPT_CONVEX_STALL_ABSTAIN",
        "DISCOPT_GDP_CONFIG_PRIMAL",
        "DISCOPT_NLPBB_ROOT_CUTS",
        "DISCOPT_QUBO_PRIMAL",
        "DISCOPT_ROOT_BOUND_SEED",
    ]
    checked = 0
    for flag in on_with_empty_default:
        assert flag in sites, f"{flag} vanished from the tree; re-check this list"
        assert polarity_of(sites[flag]) == "on", (
            f"{flag} is a default-ON opt-out (its predicate is negative, or it returns "
            f"True on the empty string) but the scan calls it "
            f"{polarity_of(sites[flag])!r}. Polarity must come from the gate's own "
            "comparison, never from the default literal."
        )
        checked += 1
    off_with_empty_default = ["DISCOPT_NODE_PROBING", "DISCOPT_NARROW_BOX_BRANCH"]
    for flag in off_with_empty_default:
        assert flag in sites, f"{flag} vanished from the tree; re-check this list"
        assert polarity_of(sites[flag]) == "off", (
            f"{flag} has the same empty default as the gates above and IS default-OFF; "
            f"the scan calls it {polarity_of(sites[flag])!r}."
        )
        checked += 1
    assert checked == 7, f"probe made {checked} comparisons, expected 7"


def test_the_word_vocabularies_are_disjoint():
    """`ON_WORDS`/`OFF_WORDS` decide polarity; an overlap would make it arbitrary."""
    assert not (ON_WORDS & OFF_WORDS)
    assert (ON_WORDS | OFF_WORDS) <= BOOLEAN_DEFAULTS


@pytest.mark.parametrize("excluded", sorted(OUT_OF_SCOPE))
def test_the_out_of_scope_flags_are_still_default_off(excluded):
    """Guard the exclusion list: an excluded flag that flipped ON should be removed."""
    sites = scan()
    assert excluded in sites, (
        f"{excluded} is on the OUT_OF_SCOPE list but is no longer read at all. Drop it "
        "from the list rather than leaving a stale exemption."
    )
    assert polarity_of(sites[excluded]) == "off", (
        f"{excluded} is exempted as a default-OFF non-solver switch but now reads "
        f"{polarity_of(sites[excluded])!r}."
    )


def test_the_audit_states_the_three_states_from_claude_md():
    audit = AUDIT.read_text().lower()
    for state in ("graduated", "retire", "documented opt-out"):
        assert state in audit, f"the audit never mentions the '{state}' state"
