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

import importlib
import json
import re
import sys
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


# Neither jaxlib nor highspy publishes an emscripten wheel, and jax hard-pins
# jaxlib, so the page installs discopt with `deps: false` and these three are
# simply absent in the browser.
BROWSER_MISSING = ("highspy", "jax", "jaxlib")


@pytest.fixture
def browser_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this process resemble the browser for the duration of one test.

    Without this the module is worthless for its stated purpose. A developer
    machine and CI both have highspy, jax and jaxlib installed, so an example
    that works here can still die on the page -- and did: the LP and MILP
    examples, which are the first two in the dropdown, both raised

        discopt.solvers.lp_milp_highs.HighsUnavailable:
            the LP/MILP HiGHS route needs highspy>=1.10

    because a pure LP dispatches to the #1229 HiGHS route unconditionally and
    that route has no fallback by design. Import-time checks cannot catch this:
    every use site is a function-local import, so nothing fails until a solve
    actually asks for it.

    ``DISCOPT_LP_MILP_BACKEND=rust`` is what ``web/worker.js`` sets, and it must
    be set here too or the two environments are not the same environment. It is
    discopt's own documented opt-out from the HiGHS route, selecting the
    certified in-house simplex -- not a relaxation of anything.
    """
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    _block_imports(monkeypatch)


def _block_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``BROWSER_MISSING`` unimportable by any route, reversibly.

    A ``sys.meta_path`` finder rather than a ``builtins.__import__`` wrapper.
    The wrapper was the first attempt and it is not equivalent: an ``import x``
    statement compiles to ``__import__``, but ``importlib.import_module`` goes
    straight to the import machinery and sails past it. discopt's lazy sites
    happen to use import statements, so the wrapper did catch the real failure
    -- but "happens to" is not a property to rest a guard on, and a future lazy
    site written with ``importlib`` would silently stop being covered.

    The finder is installed first in ``meta_path`` and the modules are evicted
    from ``sys.modules``, since a cached entry is returned without consulting
    any finder. ``monkeypatch`` restores both.
    """

    class Blocked:
        @staticmethod
        def find_spec(name: str, path: object = None, target: object = None) -> None:
            if name.split(".")[0] in BROWSER_MISSING:
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    for name in list(sys.modules):
        if name.split(".")[0] in BROWSER_MISSING:
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Blocked] + sys.meta_path)


def test_browser_environment_actually_blocks() -> None:
    """The fixture is the whole test; an inert one would report six false passes.

    Both import routes are checked, because the first version of the fixture
    blocked only one of them and this test is what found that out.
    """
    checks = 0
    with pytest.MonkeyPatch.context() as mp:
        _block_imports(mp)
        for blocked in BROWSER_MISSING:
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(blocked)
            checks += 1
            with pytest.raises(ModuleNotFoundError):
                exec(f"import {blocked}", {})
            checks += 1

    # And the block must lift, or every later test in the session inherits it.
    assert importlib.import_module("json") is not None
    assert checks == 2 * len(BROWSER_MISSING), checks


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["id"] for e in EXAMPLES])
def test_example_runs(
    example: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    browser_environment: None,
) -> None:
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
