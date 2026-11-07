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

**What #2211 changed.** The rebuilt scan still walked ``python/discopt`` only, so every
gate the Rust core reads through ``std::env::var`` was invisible here, and the
document's marker was a Python-only count -- #1421's failure mode one language over.
Ten default-OFF Rust gates over solver math had no row, ``DISCOPT_PRIMAL_EXPAND_RESET``
among them. The scan now reads ``crates/*/src`` as well; the tests at the bottom of this
file pin each Rust consumer shape, on the live tree and on synthetic snippets.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from flag_audit_scan import (
    BOOLEAN_DEFAULTS,
    OFF_WORDS,
    ON_WORDS,
    _rust_polarity,
    _rust_strip,
    polarity_of,
    scan,
    scan_rust,
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
    # #2211: the Rust scan's three diagnostics. Each only arms counters or writes to
    # stderr; none reaches a pivot, a bound or a branching decision.
    "DISCOPT_PROFILE": "arms the Rust profile counters (`profile.rs`); observation only",
    "DISCOPT_LP_DUAL_TRACE": "one stderr trace line per dual pivot; observation only",
    "DISCOPT_T14_DBG": "stderr accept/reject lines in the warm primal; observation only",
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


# ------------------------------------------------------------------ #2211: Rust reads


def test_the_scan_reads_the_rust_core():
    """The #2211 regression test: fails against the Python-only scan.

    `DISCOPT_SIMPLEX_NO_EXPAND` is read only in `lp/simplex/primal.rs`; before #2211
    `scan()` did not return it at all, so it could not owe a row. (Its sibling
    `DISCOPT_PRIMAL_EXPAND_RESET`, the gate #2211 was filed on, was retired by the
    panel that row prompted.)
    """
    sites = scan()
    rust = [s for v in sites.values() for s in v if s.path.startswith("crates/")]
    assert len(rust) >= 30, f"only {len(rust)} Rust read sites -- the Rust scan narrowed"
    assert "DISCOPT_SIMPLEX_NO_EXPAND" in sites, "the Rust scan does not reach primal.rs"
    assert all(s.path.startswith("crates/") for s in sites["DISCOPT_SIMPLEX_NO_EXPAND"])


#: Live Rust flags, one or more per consumer shape, with the polarity read by hand from
#: the parse site. Negated presence tests are the #1421 polarity trap in Rust form: the
#: literal `"0"` appears in all three, and all three are default ON.
RUST_EXPECTED = {
    "DISCOPT_SIMPLEX_NO_EXPAND": ("rust_unwrap_or", "off"),  # .map(..).unwrap_or(false)
    "DISCOPT_NS_MARGIN": ("rust_unwrap_or", "on"),  # .map(..).unwrap_or(true)
    "DISCOPT_PRIMAL_XB_REFRESH": ("rust_unwrap_or", "selector"),  # .parse::<usize>().unwrap_or(0)
    "DISCOPT_TREE_SENTINEL_PRUNE_GUARD": ("rust_presence", "on"),  # !var(F).is_ok_and(=="0")
    "DISCOPT_MILP_NOISE_RELAX": ("rust_presence", "on"),  # !var(F).is_ok_and(in off-words)
    "DISCOPT_OBJ_INTEGRALITY": ("rust_presence", "on"),
    "DISCOPT_MILP_RC_FIX_REFACTOR": ("rust_presence", "off"),  # var(F).is_ok_and(=="1")
    "DISCOPT_DISABLE_CSE": ("rust_presence", "on"),  # var_os(F).is_none()
    "DISCOPT_FBBT_SIGN_DOMAIN": ("rust_match", "on"),  # match var(F) { Err(_) => true, .. }
    "DISCOPT_OBJ_LATTICE_SUBST": ("rust_match", "on"),
    "DISCOPT_LP_DUAL_COST_PERTURB": ("rust_parse_table", "on"),  # "" | "1" | .. => Ok(EPS)
    "DISCOPT_LP_DUAL_STALL_BAIL": ("rust_parse_table", "off"),  # "" | "0" | .. => Ok(0)
    "DISCOPT_LP_KAPPA_RECOVERY": ("rust_parse_table", "off"),  # "" | "0" | .. => Ok(None)
    "DISCOPT_LP_REFAC_INTERVAL": ("rust_parse_table", "selector"),  # is_empty() -> Ok(48)
}


def test_the_rust_scan_classifies_every_shape_on_the_live_tree():
    sites = scan_rust()
    checked = 0
    for flag, (form, pol) in RUST_EXPECTED.items():
        assert flag in sites, f"{flag} is no longer read in crates/; re-pick this example"
        got = {(s.form, s.polarity) for s in sites[flag]}
        assert got == {(form, pol)}, f"{flag}: expected {(form, pol)}, scan says {got}"
        checked += 1
    assert checked == len(RUST_EXPECTED), "probe did not run every case"
    forms = {s.form for v in sites.values() for s in v}
    for form in ("rust_unwrap_or", "rust_presence", "rust_match", "rust_parse_table"):
        assert form in forms, f"no Rust read of form {form!r} found -- the scan narrowed"


#: (snippet, form, polarity). Independent of the live tree, so retiring the last flag
#: of a shape cannot silently take that shape's coverage with it.
RUST_SNIPPETS = [
    (
        'fn f() -> bool { std::env::var("DISCOPT_X").map(|v| v != "0").unwrap_or(true) }',
        "rust_unwrap_or",
        "on",
    ),
    (
        'fn f() -> bool { !std::env::var("DISCOPT_X").ok().map(|v| v == "1").unwrap_or(false) }',
        "rust_unwrap_or",
        "on",
    ),
    ('fn f() -> bool { std::env::var_os("DISCOPT_X").is_some() }', "rust_presence", "off"),
    ('fn f() -> bool { !std::env::var_os("DISCOPT_X").is_some() }', "rust_presence", "on"),
    ('fn f() -> bool { env::var("DISCOPT_X").is_err() }', "rust_presence", "on"),
    (
        'fn f() -> bool { match std::env::var("DISCOPT_X") { Err(_) => false, Ok(v) => true } }',
        "rust_match",
        "off",
    ),
    (
        'fn f() -> bool { match std::env::var("DISCOPT_X") { Err(_) => DEF, Ok(_) => true } }',
        "rust_match",
        "unknown:match-err-arm-not-literal",
    ),
    (
        'fn f() -> bool { let raw = std::env::var("DISCOPT_X").unwrap_or_default(); p(&raw) }\n'
        'fn p(raw: &str) -> bool { match raw { "1" | "on" => true, "" | "0" | "off" => false,'
        " _ => panic!() } }",
        "rust_parse_table",
        "off",
    ),
    (
        'fn f() -> bool { let raw = std::env::var("DISCOPT_X").unwrap_or_default(); p(&raw) }\n'
        'fn p(raw: &str) -> bool { match raw { "" | "0" | "1" => true, _ => false } }',
        "rust_parse_table",
        "unknown:parse-mixed-word-set",
    ),
    (
        'fn f() -> u32 { std::env::var("DISCOPT_X").ok().and_then(|v| v.parse().ok())'
        ".unwrap_or(16) }",
        "rust_unwrap_or",
        "selector",
    ),
    (
        'fn f() -> Opt { std::env::var("DISCOPT_X").ok().map(Opt::from).unwrap_or(Opt::A) }',
        "rust_unwrap_or",
        "unknown:unwrap_or-arg:Opt::A",
    ),
    (
        'fn f() -> bool { std::env::var("DISCOPT_X").ok().as_deref() == Some("1") }',
        "rust_unrecognised",
        "unknown:rust-consumer-shape",
    ),
]


@pytest.mark.parametrize(("src", "form", "pol"), RUST_SNIPPETS)
def test_the_rust_polarity_rules_on_snippets(src, form, pol):
    import re

    text = _rust_strip(src)
    m = re.search(r"(?:\bstd::)?\benv::var(_os)?\(\"DISCOPT_X\"\)", text)
    assert m, "snippet has no read"
    assert _rust_polarity(text, m.start(), m.end(), bool(m.group(1))) == (form, pol)


def test_the_rust_scan_ignores_comments_and_test_code():
    """A doc comment quoting a read, or a test saving and restoring a variable, is not
    a gate. Counting either would be measuring the docstring (CLAUDE.md §6)."""
    src = (
        '/// `std::env::var("DISCOPT_DOC").unwrap_or(false)`\n'
        "fn live() -> bool {\n"
        '    std::env::var("DISCOPT_LIVE").map(|v| v == "1").unwrap_or(false) // var("DISCOPT_C")\n'
        "}\n"
        '/* block: std::env::var("DISCOPT_BLOCK") */\n'
        "#[cfg(test)]\n"
        "mod tests {\n"
        '    fn t() { let prior = std::env::var_os("DISCOPT_TEST"); }\n'
        "}\n"
    )
    text = _rust_strip(src)
    assert text.count("\n") == src.count("\n"), "stripping moved line numbers"
    assert "DISCOPT_LIVE" in text
    for gone in ("DISCOPT_DOC", "DISCOPT_C", "DISCOPT_BLOCK", "DISCOPT_TEST"):
        assert gone not in text, f"{gone} survived stripping"


def test_no_rust_read_is_left_unclassified():
    """Rule 3 for the Rust half, stated on its own so a failure names the language."""
    unknown = {
        f: [s.where for s in v if s.polarity.startswith("unknown")] for f, v in scan_rust().items()
    }
    unknown = {f: w for f, w in unknown.items() if w}
    assert not unknown, (
        f"the Rust scan cannot classify {unknown}. Teach flag_audit_scan._rust_polarity "
        "the consumer shape (and add a snippet to RUST_SNIPPETS), or spell the read in "
        "one of the shapes it knows."
    )
