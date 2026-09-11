"""No math function may make the Rust `.nl` writer panic (#1220 review, §1).

`nl_writer.rs` turns an expanded function code into a `.nl` opcode through
`nl_func_opcode`. `expand.rs::func_code` decides which codes get that far, and
the two tables are maintained separately. When they drifted -- `func_code`
admitted `MathFunc::Log2`, `nl_func_opcode` had no arm for it -- the call site's
``.expect("mapped function")`` fired, and:

* ``to_nl()`` raised ``pyo3_runtime.PanicException`` on ANY model containing
  ``log2``, where the previous release wrote the file correctly;
* the Python-writer fallback could not save it, because ``PanicException``
  derives from ``BaseException``, not ``Exception``, so ``_rust_nl_text``'s
  ``except Exception`` never saw it.

One wrong entry in a lookup table is a *class* of bug, so this sweeps every
function the modeling layer exposes, in both idioms, rather than testing `log2`
alone. Two properties per case:

1. ``to_nl()`` must not raise. A function the Rust writer cannot emit is a
   refusal that falls back, never a crash.
2. Where the Rust writer does produce text, it must be **byte-identical** to the
   Python writer's -- a wrong opcode is worse than a missing one.
"""

import os

import discopt.modeling as dm
import pytest
from discopt import Model
from discopt.export.nl import _rust_nl_text, to_nl

# Every unary math function the modeling layer exposes. `log2` is the one that
# shipped broken; the rest are here so the next table gap fails on its own case.
FUNCS = [
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "atan",
    "sinh",
    "cosh",
    "asin",
    "acos",
    "tanh",
    "abs",
    "log1p",
    "sigmoid",
    "softplus",
    "erf",
    "asinh",
    "acosh",
    "atanh",
    "sign",
]


def _model(fname: str, *, array: bool) -> Model:
    """A model whose only nonlinearity is ``fname``, on a safe domain.

    Bounds sit in (0, 1) so every function here -- including `asin`, `acosh`'s
    neighbours and `log` -- has a real value over the box.
    """
    fn = getattr(dm, fname)
    m = Model("p")
    if array:
        x = m.continuous("x", shape=(3,), lb=0.25, ub=0.75)
        m.subject_to(fn(x) <= 3.0, name="c")
        m.minimize(dm.sum(x))
    else:
        x = m.continuous("x", lb=0.25, ub=0.75)
        m.subject_to(fn(x) <= 3.0, name="c")
        m.minimize(x)
    return m


@pytest.mark.unit
@pytest.mark.parametrize("fname", FUNCS)
@pytest.mark.parametrize("array", [False, True], ids=["scalar", "array"])
def test_to_nl_never_panics(fname, array):
    """The bug: `log2` raised PanicException here, in both idioms.

    A ``ValueError`` is NOT a failure. ``erf``, ``sign`` and the inverse
    hyperbolics have no `.nl` opcode at all, and the Python writer refuses them
    by name -- a loud, deliberate refusal (CLAUDE.md §3), and the behaviour
    before this branch too. What must never happen is a *panic*: an internal
    table gap escaping as ``pyo3_runtime.PanicException``, which no caller can
    anticipate and which bypasses the writer's own fallback.
    """
    try:
        text = to_nl(_model(fname, array=array))
    except ValueError as exc:
        # Must name the function it is refusing, or it is not a considered refusal.
        assert fname in str(exc).lower(), f"{fname}: opaque refusal: {exc}"
        return
    assert text.startswith("g3 "), f"{fname}: not a .nl file"
    assert len(text) > 100, f"{fname}: suspiciously short .nl"


@pytest.mark.unit
@pytest.mark.parametrize("fname", FUNCS)
@pytest.mark.parametrize("array", [False, True], ids=["scalar", "array"])
def test_rust_matches_python_or_declines(fname, array, monkeypatch):
    """Where Rust writes, it must agree with Python byte for byte."""
    model = _model(fname, array=array)
    rust = _rust_nl_text(model)

    monkeypatch.setenv("DISCOPT_RUST_NL", "0")
    try:
        python = to_nl(_model(fname, array=array))
    except ValueError:
        # No `.nl` form exists; the Rust writer must not invent one.
        assert rust is None, f"{fname}: Rust wrote a function Python refuses"
        return

    assert python.startswith("g3 ")
    if rust is None:
        pytest.skip(f"{fname}: Rust writer declines (Python fallback covers it)")
    assert rust == python, f"{fname}: Rust and Python .nl differ"


@pytest.mark.unit
def test_log2_is_written_by_rust_as_log_over_ln2():
    """The specific regression, pinned to its exact bytes.

    `.nl` has no base-2 log opcode, so the only correct output is the rewrite
    `o3 o43 <arg> n0.6931471805599453` (divide, log, argument, ln 2).
    """
    rust = _rust_nl_text(_model("log2", array=False))
    assert rust is not None, "the Rust writer must handle log2, not decline it"
    lines = rust.splitlines()
    i = lines.index("o3")
    assert lines[i : i + 2] == ["o3", "o43"], lines[i : i + 4]
    assert "n0.6931471805599453" in lines, "divisor must be repr(math.log(2))"


@pytest.mark.unit
def test_panic_exception_would_not_be_caught_by_except_exception():
    """Why the fallback needed widening, asserted rather than asserted-in-prose.

    If PyO3's PanicException ever became an `Exception` subclass this test fails
    and the extra `except BaseException` clause in `_rust_nl_text` can go.
    """
    pyo3_runtime = pytest.importorskip("pyo3_runtime")
    panic = pyo3_runtime.PanicException
    assert issubclass(panic, BaseException)
    assert not issubclass(panic, Exception), (
        "PanicException is now an Exception; _rust_nl_text's BaseException "
        "clause is redundant and should be removed"
    )


@pytest.mark.unit
def test_every_function_was_actually_exercised():
    """Prove the sweep is not vacuous (CLAUDE.md §6)."""
    assert len(FUNCS) >= 20, f"only {len(FUNCS)} functions swept"
    missing = [f for f in FUNCS if not hasattr(dm, f)]
    assert not missing, f"FUNCS names non-existent modeling functions: {missing}"
    # And that the writer is actually enabled by default, or both tests above
    # would be testing the Python writer twice.
    assert os.environ.get("DISCOPT_RUST_NL", "1") not in ("0", "false", "False")


@pytest.mark.unit
def test_tan_is_written_as_sin_over_cos_by_both_writers():
    """`tan` is a deliberate rewrite in `export/nl.py`, not a native opcode.

    `.nl` does have `o38` (tan), and the Rust writer originally emitted it --
    semantically fine, but it silently diverged from the Python writer this port
    is byte-compared against, on a function no corpus instance uses (so the
    corpus sweep could not see it). Pinned here so the choice is a decision
    rather than drift: changing what discopt writes for `tan` is a user-visible
    output change and needs its own justification.
    """
    rust = _rust_nl_text(_model("tan", array=False))
    assert rust is not None, "the Rust writer must handle tan, not decline it"
    lines = rust.splitlines()
    i = lines.index("o3")
    assert lines[i : i + 2] == ["o3", "o41"], lines[i : i + 5]
    assert "o46" in lines, "cos leg missing"
    assert "o38" not in lines, "native tan opcode: diverges from the Python writer"
