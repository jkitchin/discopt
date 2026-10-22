"""JAX is the optional ``discopt[jax]`` extra, not a core dependency (#1873).

Three things have to hold for that to be true, and each is pinned here:

1. **The packaging says so.** ``jax``/``jaxlib`` are absent from ``[project]
   dependencies`` and present in the ``jax`` extra.
2. **The default path never reaches JAX.** Every in-repo ``.nl`` instance is solved
   in a fresh interpreter whose import system refuses ``jax`` and *records the
   stack of every attempt* -- so a path that catches the ``ImportError`` and
   quietly degrades is caught too, not only one that crashes. This is the
   corpus-wide form of the per-path probes in ``test_issue_1870_polish_budget.py``.
3. **A JAX-only feature refuses loudly and names the extra.** The evaluator
   fallback (an expression with no tape opcode), ``dm.custom``, and the legacy
   ``_relax`` modules each raise :class:`MissingJaxError` -- a
   ``ModuleNotFoundError`` with ``name="jax"`` -- whose message says how to install
   it, instead of ``No module named 'jax'`` from deep inside a solve.

The JAX-less behaviour is simulated in a subprocess (``_optional_jax._available =
False`` plus an import blocker), so this file runs, and means the same thing, in
both the JAX-installed lanes and the JAX-less one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"

# Prepended to every subprocess: JAX is "not installed", and every attempt to
# import it is recorded with the stack that made it (CLAUDE.md §7: the probe must
# see an attempt that a caller swallows, not only one that propagates).
_BLOCK_JAX = r"""
import sys, traceback
JAX_ATTEMPTS = []
class _NoJax:
    def find_spec(self, name, path=None, target=None):
        root = name.partition(".")[0]
        if root in ("jax", "jaxlib"):
            JAX_ATTEMPTS.append((name, "".join(traceback.format_stack(limit=30))))
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NoJax())
import discopt._optional_jax as _oj
_oj._available = False
"""


def _run_without_jax(body: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "DISCOPT_NLP_EVAL"}
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_JAX + body],
        capture_output=True,
        text=True,
        cwd="/",
        env=env,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- #
# 1. packaging
# --------------------------------------------------------------------------- #


def test_jax_is_an_extra_not_a_core_dependency():
    meta = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    core = [d.split(">")[0].split("=")[0].split("[")[0].strip() for d in meta["dependencies"]]
    assert "jax" not in core and "jaxlib" not in core, core
    extra = [d.split(">")[0].strip() for d in meta["optional-dependencies"]["jax"]]
    assert {"jax", "jaxlib"} <= set(extra), extra


# --------------------------------------------------------------------------- #
# 2. the error type
# --------------------------------------------------------------------------- #


def test_missing_jax_error_is_a_module_not_found_error_naming_the_extra():
    from discopt._optional_jax import MissingJaxError, missing_jax_error

    err = missing_jax_error("the frobnicator")
    assert isinstance(err, MissingJaxError)
    # Existing ``except ImportError`` handlers must see what they saw before.
    assert isinstance(err, ModuleNotFoundError) and err.name == "jax"
    msg = str(err)
    assert "the frobnicator requires JAX" in msg
    assert "pip install 'discopt[jax]'" in msg


def test_require_jax_raises_only_when_jax_is_absent(monkeypatch):
    import discopt._optional_jax as oj

    monkeypatch.setattr(oj, "_available", True)
    oj.require_jax("x")  # no raise
    monkeypatch.setattr(oj, "_available", False)
    with pytest.raises(oj.MissingJaxError, match="x requires JAX"):
        oj.require_jax("x")


def test_a_blocked_jax_counts_as_unavailable(monkeypatch):
    """``sys.modules['jax'] = None`` is how an import is blocked; it is not 'installed'."""
    import discopt._optional_jax as oj

    monkeypatch.setattr(oj, "_available", None)
    monkeypatch.setitem(sys.modules, "jax", None)
    assert oj.jax_available() is False


# --------------------------------------------------------------------------- #
# 3. JAX-only features refuse loudly
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_without_jax_the_evaluator_fallback_names_the_construct_and_the_extra():
    """The measured case from CLAUDE.md: ``dm.norm(X, 2)`` of a 3x3 ``X`` has no tape
    opcode, so its solve used to load 219 ``jax`` modules. Without JAX it must refuse,
    and say which construct needs the fallback, not just that JAX is missing."""
    p = _run_without_jax(
        "import discopt.modeling as dm\n"
        "from discopt._optional_jax import MissingJaxError\n"
        "m = dm.Model()\n"
        "X = m.continuous('X', shape=(3, 3), lb=-2.0, ub=2.0)\n"
        "m.minimize(dm.norm(X, 2) + dm.sum(X))\n"
        "try:\n"
        "    m.solve(time_limit=10)\n"
        "except MissingJaxError as exc:\n"
        "    print('REFUSED', exc)\n"
        "else:\n"
        "    print('SOLVED')\n"
    )
    assert p.returncode == 0, p.stderr[-3000:]
    assert "REFUSED" in p.stdout, p.stdout
    assert "no opcode" in p.stdout and "norm2" in p.stdout, p.stdout
    assert "pip install 'discopt[jax]'" in p.stdout, p.stdout


@pytest.mark.smoke
def test_without_jax_dm_custom_refuses_at_construction():
    p = _run_without_jax(
        "import discopt.modeling as dm\n"
        "from discopt._optional_jax import MissingJaxError\n"
        "try:\n"
        "    dm.custom(lambda v: v)\n"
        "except MissingJaxError as exc:\n"
        "    print('REFUSED', exc)\n"
    )
    assert p.returncode == 0, p.stderr[-3000:]
    assert "REFUSED dm.custom" in p.stdout and "discopt[jax]" in p.stdout, p.stdout


# Every module with a module-scope ``import jax``. Importing one without JAX must
# raise the named error, not a bare ``No module named 'jax'``.
_JAX_MODULES = sorted(
    ".".join(p.relative_to(REPO / "python").with_suffix("").parts)
    for p in (REPO / "python" / "discopt").rglob("*.py")
    if any(
        line.startswith(("import jax", "from jax"))
        for line in p.read_text(encoding="utf-8").splitlines()
    )
)


def test_every_module_scope_jax_importer_refuses_with_the_named_error():
    assert len(_JAX_MODULES) >= 30, _JAX_MODULES  # the scan found the modules
    p = _run_without_jax(
        "import importlib, json\n"
        "from discopt._optional_jax import MissingJaxError\n"
        f"mods = {_JAX_MODULES!r}\n"
        "bad = {}\n"
        "for name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except MissingJaxError as exc:\n"
        "        if 'discopt[jax]' not in str(exc):\n"
        "            bad[name] = str(exc)\n"
        "    except Exception as exc:\n"
        "        bad[name] = f'{type(exc).__name__}: {exc}'\n"
        "    else:\n"
        "        bad[name] = 'imported without JAX'\n"
        "print('RESULT', json.dumps({'checked': len(mods), 'bad': bad}))\n"
    )
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout.split("RESULT", 1)[1])
    assert out["checked"] == len(_JAX_MODULES) > 0
    assert out["bad"] == {}, out["bad"]


# --------------------------------------------------------------------------- #
# 4. the default path never reaches JAX
# --------------------------------------------------------------------------- #

_SOLVE_ALL = r"""
import json, logging, warnings
logging.disable(logging.CRITICAL)
warnings.simplefilter("ignore")
import discopt.modeling as dm
paths = {paths!r}
solved, errors, first_attempt = 0, {{}}, {{}}
for path in paths:
    seen = len(JAX_ATTEMPTS)
    try:
        dm.from_nl(path).solve(time_limit={time_limit})
        solved += 1
    except Exception as exc:
        errors[path] = f"{{type(exc).__name__}}: {{exc}}"[:500]
    if len(JAX_ATTEMPTS) > seen:
        first_attempt[path] = JAX_ATTEMPTS[seen][1][-4000:]
print("RESULT", json.dumps({{"n": len(paths), "solved": solved, "errors": errors,
                             "jax": first_attempt}}))
"""


def _assert_corpus_is_jax_free(paths: list[str], time_limit: float, timeout: float) -> None:
    assert paths, "the corpus is empty -- the guard would measure nothing"
    p = _run_without_jax(
        _SOLVE_ALL.format(paths=paths, time_limit=time_limit), timeout=timeout
    )
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout.split("RESULT", 1)[1])
    assert out["n"] == len(paths)
    # A path that reached for JAX is reported with the stack that did it.
    assert out["jax"] == {}, "\n\n".join(f"== {k}\n{v}" for k, v in out["jax"].items())
    assert out["errors"] == {}, out["errors"]
    assert out["solved"] == len(paths)


@pytest.mark.smoke
def test_a_few_instances_solve_without_ever_reaching_for_jax():
    names = ["nvs13", "chance", "ex1221"]
    paths = [str(NL_DIR / f"{n}.nl") for n in names if (NL_DIR / f"{n}.nl").exists()]
    assert len(paths) == len(names), "a named probe instance moved"
    _assert_corpus_is_jax_free(paths, time_limit=10.0, timeout=600.0)


@pytest.mark.slow
def test_the_whole_in_repo_corpus_solves_without_ever_reaching_for_jax():
    """The ``Done when`` guard of #1873: every in-repo instance, ``time_limit=10``."""
    paths = sorted(str(p) for p in NL_DIR.glob("*.nl"))
    assert len(paths) >= 60, len(paths)
    _assert_corpus_is_jax_free(paths, time_limit=10.0, timeout=3000.0)
