"""Execute the browser page's example models.

The dropdown on ``crates/discopt-wasm/web/index.html`` offers one model per
problem class, and whichever one is selected first is the page's first
impression. An example that raises is worse than no example at all, and the
failure would only ever surface in a browser, where nobody runs pytest.

So the examples live in JavaScript but are *executed here*, as Python, against
the same solver the wheel is built from. This does not test the wasm build --
it tests that the code shipped in the editor is code that solves.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

EXAMPLES_JS = (
    Path(__file__).resolve().parents[2] / "crates" / "discopt-wasm" / "web" / "examples.js"
)

# The six classes the page promises. Named here rather than derived from the
# file so that deleting an example is a test failure, not a silently shorter
# parametrization -- the failure mode this whole module exists to prevent.
EXPECTED_IDS = ["LP", "MILP", "QP", "MIQP", "NLP", "MINLP"]


def _load_examples() -> list[dict[str, str]]:
    """Parse ``examples.js``'s ``EXAMPLES`` array.

    The array is JSON apart from two ES niceties: bare object keys and trailing
    commas. Both are repaired textually rather than by shelling out to node,
    which CI does not promise. Anything else in the file -- a template literal,
    a computed value, a comment inside the array -- makes ``json.loads`` raise,
    which is the correct outcome: this reader is deliberately narrow, and a
    silent partial parse would be worse than a crash.
    """
    text = EXAMPLES_JS.read_text(encoding="utf-8")
    start = text.index("[", text.index("export const EXAMPLES"))
    end = text.rindex("]")
    body = text[start : end + 1]
    body = re.sub(r"(?m)^(\s*)(id|label|title|code):", r'\1"\2":', body)
    body = re.sub(r",(\s*[}\]])", r"\1", body)
    return json.loads(body)


EXAMPLES = _load_examples()


def test_examples_file_covers_every_problem_class() -> None:
    assert [e["id"] for e in EXAMPLES] == EXPECTED_IDS
    for example in EXAMPLES:
        assert example["label"], example["id"]
        assert example["title"], example["id"]
        assert "import discopt.modeling as dm" in example["code"], example["id"]


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["id"] for e in EXAMPLES])
def test_example_runs(example: dict[str, str], capsys: pytest.CaptureFixture[str]) -> None:
    """Run the example exactly as the browser will: exec the source, top level.

    No ``try``/``except`` -- an exception here is the finding. The exec
    namespace carries ``__name__ == "__main__"`` because that is what
    ``pyodide.runPythonAsync`` gives the script.
    """
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(compile(example["code"], f"<{example['id']} example>", "exec"), namespace)

    # A model that printed nothing solved into a void; the console is the whole
    # output of the page.
    printed = capsys.readouterr().out
    assert printed.strip(), f"{example['id']} printed nothing"
    assert "optimal" in printed.lower(), f"{example['id']} did not report an optimal status"
