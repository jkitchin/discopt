"""The serialization docs say what the code does (plan WP-8, D14).

``docs/dev/serialization-format.md`` holds three tables between ``<!-- generated:
... -->`` markers -- the top-level keys, the ``state`` keys and the permanent
refusals -- and each must equal what this file renders from ``discopt.serialize``.
On a mismatch the assertion prints the table to paste. The user guide
(``docs/notebooks/model_serialization.ipynb``) must name every refused
transformation, and carry the same compatibility statement as the format doc.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import discopt.serialize as ser

REPO = Path(__file__).resolve().parents[2]
FORMAT_DOC = REPO / "docs" / "dev" / "serialization-format.md"
NOTEBOOK = REPO / "docs" / "notebooks" / "model_serialization.ipynb"

_PASS_WHY = "records point back at the source model; some feed a bound"
_PASS_INSTEAD = "save the source model, re-apply the pass after loading"


def _block(text: str, name: str) -> str:
    m = re.search(
        rf"<!-- generated: {re.escape(name)} \(test_serialize_docs_match\.py\) -->\n"
        r"(.*?)<!-- end generated -->",
        text,
        flags=re.S,
    )
    assert m, f"no generated block {name!r} in {FORMAT_DOC.name}"
    return m.group(1)


def _key_table(keys: dict[str, int]) -> str:
    rows = "".join(f"| `{k}` | {v} |\n" for k, v in keys.items())
    return "| key | since minor |\n|---|---|\n" + rows


def _refusal_table() -> str:
    rows = [
        (
            "`dm.custom` (a `CustomCall`)",
            "wraps an arbitrary Python callable; its name would reload into a different model",
            "algebraic expressions, or rebuild it in code",
        )
    ]
    rows += [
        (f"output of `{label}`", _PASS_WHY, _PASS_INSTEAD)
        for label in ser._REFORMULATION_RECORD_PREFIXES.values()
    ]
    rows += [
        (
            "a user `big_m` on a disjunction (#1775 G2.6)",
            "no schema key records it; an older reader would reload computed Ms",
            "save without `big_m`, pass it again after loading",
        ),
        (
            "a `Model.disjunction()` / `Disjunct` never attached",
            "`validate` refuses the model anyway (#1430)",
            "attach it, or drop it",
        ),
        (
            "an unclassified private attribute",
            "dropping it would reload a different model",
            "classify it in `serialize.py`",
        ),
    ]
    body = "".join(f"| {a} | {b} | {c} |\n" for a, b, c in rows)
    return "| what | why | instead |\n|---|---|---|\n" + body


def _refused_transformations() -> set[str]:
    return {
        name for label in ser._REFORMULATION_RECORD_PREFIXES.values() for name in label.split(" / ")
    }


def _notebook_markdown() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def _section(text: str, heading: str) -> str:
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    assert m, f"no section {heading!r}"
    return m.group(1)


def test_format_doc_key_table_matches_code():
    doc = FORMAT_DOC.read_text(encoding="utf-8")
    for name, keys in (("_TOP_LEVEL_KEYS", ser._TOP_LEVEL_KEYS), ("_STATE_KEYS", ser._STATE_KEYS)):
        want = _key_table(keys)
        assert _block(doc, name) == want, f"regenerate {name}:\n{want}"


def test_format_doc_history_covers_every_minor():
    rows = re.findall(r"^\| (\d+) \|", _section(FORMAT_DOC.read_text(), "Schema history"), re.M)
    assert [int(r) for r in rows] == list(range(ser.SCHEMA_MINOR + 1))


def test_docs_refused_list_matches_code():
    """The format doc's refusal table, and the notebook's refusal section, against
    ``_REFORMULATION_RECORD_PREFIXES`` -- and both against the builder sweep's
    REFUSED table, which proves each refusal fires."""
    want = _refusal_table()
    assert _block(FORMAT_DOC.read_text(encoding="utf-8"), "refusals") == want, want

    refused = _section(_notebook_markdown(), "What is refused, and why")
    missing = sorted(t for t in _refused_transformations() if f"`{t}`" not in refused)
    assert not missing, f"the notebook's refusal section does not name {missing}"
    assert "`dm.custom`" in _notebook_markdown()

    sys.path.insert(0, str(Path(__file__).parent))
    import serialize_builder_cases as cases

    swept = {k.split(":", 1)[1] for k in cases.REFUSED if k.startswith("transform:")}
    assert swept == _refused_transformations()


def test_notebook_carries_the_compatibility_statement():
    want = " ".join(_section(FORMAT_DOC.read_text(encoding="utf-8"), "Compatibility").split())
    got = " ".join(_section(_notebook_markdown(), "Versions and compatibility").split())
    first = want.split("`discopt upgrade-model")[0].strip()
    assert first in got, "the notebook's compatibility paragraph differs from the format doc's"
