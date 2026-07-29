"""Phase 1 Card 1a: the flag helper's truth table, the registry, and the grep gate.

Three things are locked here:

1. **The truth table** (:mod:`discopt._env`) — the seven incompatible boolean parse
   idioms the architecture review found (§2.4) collapse to one, and a value outside
   the table raises instead of silently picking an arm.
2. **The registry** — every string-literal flag name passed to a ``discopt._env``
   helper anywhere in ``python/discopt`` has a row in ``FLAG_REGISTRY``, including
   the 12 daemon flags whose names are built by f-string and are invisible to grep.
3. **The grep gate** — no raw ``os.environ.get("DISCOPT_`` / ``os.getenv("DISCOPT_``
   survives outside ``_env.py`` / ``solver_tuning.py``.

Every scanning test prints and asserts an **executed count** (CLAUDE.md §6): a
regex that stops matching would otherwise turn these into silent no-ops that pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from discopt._env import (
    FALSE_VALUES,
    TRUE_VALUES,
    env_bool,
    env_enum,
    env_float,
    env_int,
    env_is_set,
    env_str,
)
from discopt._flag_registry import (
    DAEMON_PREFIXES,
    DAEMON_SUFFIXES,
    FLAG_REGISTRY,
    KINDS,
    solver_tuning_flags,
)

pytestmark = pytest.mark.unit

_PKG = Path(__file__).resolve().parents[1] / "discopt"
#: Modules allowed to touch ``os.environ`` for a ``DISCOPT_`` name directly.
_GREP_GATE_EXEMPT = {"_env.py", "solver_tuning.py"}

#: ``env_bool("DISCOPT_X", ...)`` / ``env_int('DISCOPT_X', ...)`` etc.
_CALL_SITE_RE = re.compile(r"\benv_(?:bool|int|float|str|enum)\(\s*[\"'](DISCOPT_[A-Z0-9_]+)[\"']")
_RAW_READ_RE = re.compile(r"os(?:\.environ\.get|\.getenv)\(\s*[\"']DISCOPT_")


def _python_sources() -> list[Path]:
    return sorted(_PKG.rglob("*.py"))


# --------------------------------------------------------------------------- 1
def test_env_bool_truth_table():
    """1/true/yes/on and 0/false/no/off, case-insensitive, both directions."""
    checked = 0
    for raw in TRUE_VALUES:
        for spelling in (raw, raw.upper(), raw.capitalize(), f"  {raw}  "):
            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("DISCOPT_TRUTH_TABLE_PROBE", spelling)
                assert env_bool("DISCOPT_TRUTH_TABLE_PROBE", False) is True
                assert env_bool("DISCOPT_TRUTH_TABLE_PROBE", True) is True
            checked += 1
    for raw in FALSE_VALUES:
        for spelling in (raw, raw.upper(), raw.capitalize(), f"  {raw}  "):
            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("DISCOPT_TRUTH_TABLE_PROBE", spelling)
                assert env_bool("DISCOPT_TRUTH_TABLE_PROBE", False) is False
                assert env_bool("DISCOPT_TRUTH_TABLE_PROBE", True) is False
            checked += 1
    print(f"executed truth-table assertions: {checked} spellings")
    assert checked == 8 * 4


def test_env_bool_unset_and_blank_take_the_default(monkeypatch):
    monkeypatch.delenv("DISCOPT_TRUTH_TABLE_PROBE", raising=False)
    assert env_bool("DISCOPT_TRUTH_TABLE_PROBE", True) is True
    assert env_bool("DISCOPT_TRUTH_TABLE_PROBE", False) is False
    for blank in ("", "   ", "\t"):
        monkeypatch.setenv("DISCOPT_TRUTH_TABLE_PROBE", blank)
        assert env_bool("DISCOPT_TRUTH_TABLE_PROBE", True) is True
        assert env_bool("DISCOPT_TRUTH_TABLE_PROBE", False) is False
        assert env_is_set("DISCOPT_TRUTH_TABLE_PROBE") is False


@pytest.mark.parametrize("bad", ["2", "ture", "yes please", "-1", "None"])
def test_env_bool_refuses_loudly(monkeypatch, bad):
    """A typo used to silently pick an arm; it must now name the flag (CLAUDE.md §3)."""
    monkeypatch.setenv("DISCOPT_TRUTH_TABLE_PROBE", bad)
    with pytest.raises(ValueError) as exc:
        env_bool("DISCOPT_TRUTH_TABLE_PROBE", False)
    assert "DISCOPT_TRUTH_TABLE_PROBE" in str(exc.value)
    for token in TRUE_VALUES + FALSE_VALUES:
        assert token in str(exc.value)


def test_the_seven_idioms_are_gone(monkeypatch):
    """The four review-cited misreads now resolve the obvious way."""
    cases = [
        # (value, default, expected) — each one was wrong under some old idiom.
        ("false", True, False),  # was True under ``raw != "0"`` (DISCOPT_RLT)
        ("off", False, False),  # was True under ``not in ("0","","false")``
        ("0", True, False),  # presence tests read this as ON (DISCOPT_PROFILE)
        ("on", False, True),  # was False under the ``== "1"`` idiom
        ("yes", False, True),
        ("no", True, False),
    ]
    for value, default, expected in cases:
        monkeypatch.setenv("DISCOPT_TRUTH_TABLE_PROBE", value)
        assert env_bool("DISCOPT_TRUTH_TABLE_PROBE", default) is expected, value
    print(f"executed idiom assertions: {len(cases)}")
    assert len(cases) == 6


def test_env_int_float_str_enum(monkeypatch):
    monkeypatch.delenv("DISCOPT_TRUTH_TABLE_PROBE", raising=False)
    assert env_int("DISCOPT_TRUTH_TABLE_PROBE", 7) == 7
    assert env_float("DISCOPT_TRUTH_TABLE_PROBE", 1.5) == 1.5
    assert env_str("DISCOPT_TRUTH_TABLE_PROBE", "d") == "d"
    assert env_enum("DISCOPT_TRUTH_TABLE_PROBE", "a", ("a", "b")) == "a"

    monkeypatch.setenv("DISCOPT_TRUTH_TABLE_PROBE", "12")
    assert env_int("DISCOPT_TRUTH_TABLE_PROBE", 7) == 12
    assert env_float("DISCOPT_TRUTH_TABLE_PROBE", 1.5) == 12.0
    monkeypatch.setenv("DISCOPT_TRUTH_TABLE_PROBE", "B")
    assert env_enum("DISCOPT_TRUTH_TABLE_PROBE", "a", ("a", "b")) == "b"

    monkeypatch.setenv("DISCOPT_TRUTH_TABLE_PROBE", "nope")
    for fn in (
        lambda: env_int("DISCOPT_TRUTH_TABLE_PROBE", 7),
        lambda: env_float("DISCOPT_TRUTH_TABLE_PROBE", 1.5),
        lambda: env_enum("DISCOPT_TRUTH_TABLE_PROBE", "a", ("a", "b")),
    ):
        with pytest.raises(ValueError, match="DISCOPT_TRUTH_TABLE_PROBE"):
            fn()


# --------------------------------------------------------------------------- 2
def test_every_call_site_name_is_registered():
    """Every literal flag name passed to a ``_env`` helper is documented somewhere.

    ``solver_tuning.py`` is the one file whose reads belong to the *other* half of
    the surface — a ``SolverTuning`` dataclass field — so its names must resolve
    through :func:`solver_tuning_flags` instead of ``FLAG_REGISTRY``. Every other
    file must hit the registry.
    """
    tuning = solver_tuning_flags()
    seen: dict[str, list[str]] = {}
    for path in _python_sources():
        for name in _CALL_SITE_RE.findall(path.read_text()):
            seen.setdefault(name, []).append(str(path.relative_to(_PKG)))
    total = sum(len(v) for v in seen.values())
    print(f"executed call-site comparisons: {total} ({len(seen)} distinct flags)")
    assert seen, "no env_* call sites found — the scan regex is broken, not the code"
    missing = []
    for name, files in seen.items():
        if all(f == "solver_tuning.py" for f in files):
            if name not in tuning:
                missing.append(f"{name} (solver_tuning.py, not a SolverTuning field)")
        elif name not in FLAG_REGISTRY:
            missing.append(f"{name} ({files[0]})")
    assert not missing, "flags read but not in discopt._flag_registry.FLAG_REGISTRY: " + ", ".join(
        sorted(missing)
    )


def test_registry_and_solver_tuning_do_not_overlap():
    """A flag is documented in exactly one of the two halves, never both."""
    overlap = sorted(set(FLAG_REGISTRY) & set(solver_tuning_flags()))
    print(f"executed overlap comparisons: {len(FLAG_REGISTRY)} registry rows")
    assert not overlap, f"documented twice: {overlap}"


def test_daemon_fstring_flags_are_registered():
    """The daemon builds its names by f-string; register them fully expanded."""
    checked = 0
    for prefix in DAEMON_PREFIXES:
        for suffix in DAEMON_SUFFIXES:
            name = f"{prefix}_{suffix}"
            assert name in FLAG_REGISTRY, name
            checked += 1
    print(f"executed daemon-flag comparisons: {checked}")
    assert checked == 12


def test_daemon_suffix_list_matches_the_code():
    """``DAEMON_SUFFIXES`` must track ``_daemon_core.DaemonConfig``'s actual reads."""
    src = (_PKG / "_daemon_core.py").read_text()
    found = set(re.findall(r'f"\{env_prefix\}_([A-Z_]+)"', src))
    print(f"executed suffix comparisons: {len(found)}")
    assert found, "no f-string env names found in _daemon_core.py — regex is stale"
    assert found == set(DAEMON_SUFFIXES), (found, set(DAEMON_SUFFIXES))


def test_registry_rows_are_well_formed():
    for name, spec in FLAG_REGISTRY.items():
        assert spec.name == name
        assert spec.kind in KINDS
        assert spec.side in ("python", "rust")
        assert spec.doc.endswith("."), f"{name}: doc must be a sentence"
        if spec.kind == "graduated":
            assert spec.default is not False, (
                f"{name}: a graduated flag is default-ON (CLAUDE.md §5)"
            )
        if spec.kind == "parked":
            assert spec.default is False, f"{name}: a parked flag is default-OFF"
    print(f"executed registry-row assertions: {len(FLAG_REGISTRY)} rows")
    assert len(FLAG_REGISTRY) >= 60


def test_rust_flags_are_read_by_rust():
    """Each ``side='rust'`` row names a flag the Rust tree actually reads."""
    crates = Path(__file__).resolve().parents[2] / "crates"
    if not crates.is_dir():  # installed-wheel checkout: nothing to scan
        pytest.skip("crates/ not present")
    blob = "\n".join(p.read_text() for p in crates.rglob("*.rs"))
    rust_rows = [s for s in FLAG_REGISTRY.values() if s.side == "rust"]
    print(f"executed rust-flag comparisons: {len(rust_rows)}")
    assert rust_rows
    for spec in rust_rows:
        assert f'"{spec.name}"' in blob, f"{spec.name} is not read anywhere in crates/"


# --------------------------------------------------------------------------- 3
def test_no_raw_discopt_env_reads_outside_the_helper():
    """CI grep-gate (Card 1a exit criterion)."""
    offenders: list[str] = []
    scanned = 0
    for path in _python_sources():
        scanned += 1
        if path.name in _GREP_GATE_EXEMPT and path.parent == _PKG:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if _RAW_READ_RE.search(line):
                offenders.append(f"{path.relative_to(_PKG)}:{lineno}: {line.strip()}")
    print(f"executed grep-gate scans: {scanned} files")
    assert scanned > 100, "source scan found almost nothing — the glob is broken"
    assert not offenders, (
        "raw DISCOPT_ environment reads must go through discopt._env "
        "(env_bool/env_int/env_float/env_str/env_enum):\n" + "\n".join(offenders)
    )


# --------------------------------------------------------------------------- 4
def test_flags_doc_is_current():
    """``docs/reference/flags.md`` is generated; a new flag cannot skip it."""
    import importlib.util

    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "gen_flag_docs.py"
    if not script.exists():  # installed-wheel checkout
        pytest.skip("scripts/gen_flag_docs.py not present")
    spec = importlib.util.spec_from_file_location("_gen_flag_docs", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generated = module.render_markdown()
    assert module.DOC_PATH.exists(), f"{module.DOC_PATH} is missing — run the generator"
    current = module.DOC_PATH.read_text()
    print(f"executed doc comparison: {len(current)} vs {len(generated)} bytes")
    assert current == generated, (
        "docs/reference/flags.md is stale — run `python scripts/gen_flag_docs.py`"
    )
    for name in FLAG_REGISTRY:
        assert f"`{name}`" in current, f"{name} missing from the generated doc"
