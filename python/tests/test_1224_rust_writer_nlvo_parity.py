"""The Rust `.nl` writer must carry #1224's `nlvo` fix too (#1220 merge).

#1224 corrected header line 4: `nlvo` is ASL's **prefix bound**, not the raw
count of objective-nonlinear variables. Under-declaring it makes ASL truncate
its nonlinear-column prefix and mis-assign every column past the truncation --
a silent wrong answer (#1222: Ipopt reported ``Optimal Solution Found.`` with
7269.45 instead of 8457.69 on ``fuel``).

That fix landed in `export/nl.py` only. This branch adds a **second** writer in
Rust that `to_nl()` prefers, so without the same correction the fast path would
quietly re-emit the old, wrong header -- reintroducing the defect for exactly
the models that take that path, and breaking the byte-parity the port rests on.

The branch is not hypothetical: 5 of the 66 corpus instances in
``python/tests/data/minlplib_nl`` exercise it, some by a wide margin
(``heatexch_gen3`` writes ``nlvo`` 510 where the raw count is 85).

These tests are written against a **constructed** model rather than those
instance names, so they check the rule and not the sample (CLAUDE.md §2), and
they assert the branch actually fires -- a model where the two formulas agree
would pass vacuously (§6).
"""

import discopt.modeling as dm
import pytest
from discopt import Model
from discopt.export.nl import _NLWriter, _rust_nl_text, to_nl


def _objs_only_model() -> Model:
    """A model with BOTH constraint-only and objective-only nonlinear variables.

    That combination is what separates the prefix bound from the raw count: `y`
    is nonlinear in the constraint and `z` is nonlinear only in the objective,
    so the objective-only block starts after `y` and `nlvo` must name its END.
    """
    m = Model("nlvo")
    y = m.continuous("y", lb=0.5, ub=4.0)
    z = m.continuous("z", lb=0.5, ub=4.0)
    w = m.continuous("w", lb=0.5, ub=4.0)
    m.subject_to(dm.exp(y) + w <= 10.0, name="c0")
    m.minimize(dm.log(z) + y + w)
    return m


# `.nl` header lines, 0-based. Line 4 in the format's own numbering -- the
# `nlvc nlvo nlvb` row -- is index 4; index 3 is "network constraints".
_HEADER_NLV_LINE = 4


def _header_line4(text: str) -> tuple[int, int, int]:
    raw = text.splitlines()[_HEADER_NLV_LINE]
    assert "nonlinear vars in cons" in raw, f"not the nlv header row: {raw!r}"
    parts = raw.split("#")[0].split()
    assert len(parts) == 3, f"unexpected header line 4: {raw!r}"
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


@pytest.mark.unit
def test_the_prefix_bound_branch_actually_fires():
    """Guard against a vacuous test: the two formulas must disagree here."""
    writer = _NLWriter(_objs_only_model())
    writer.write()
    nlvc, nlvo, nlvb = _header_line4(writer.write())
    assert nlvc > 0 and nlvb == 0, (nlvc, nlvo, nlvb)
    # Raw count would be |objs-only| + |both| = 1; the prefix bound is nlvc + 1.
    assert nlvo == nlvc + 1, f"expected the prefix bound, got {nlvo}"
    assert nlvo != 1, "raw-count and prefix-bound agree -- this model tests nothing"


@pytest.mark.unit
def test_rust_and_python_agree_on_nlvo():
    """The regression: the Rust writer emitted the raw count."""
    model = _objs_only_model()
    rust = _rust_nl_text(model)
    assert rust is not None, "the Rust writer must handle this model, not decline"
    python = _NLWriter(_objs_only_model()).write()
    assert _header_line4(rust) == _header_line4(python)
    assert rust == python, "Rust and Python .nl differ"


@pytest.mark.unit
def test_to_nl_is_unaffected_by_which_writer_runs(monkeypatch):
    """Whichever path `to_nl` takes, the header must be the same."""
    fast = to_nl(_objs_only_model())
    monkeypatch.setenv("DISCOPT_RUST_NL", "0")
    slow = to_nl(_objs_only_model())
    assert _header_line4(fast) == _header_line4(slow)
    assert fast == slow


@pytest.mark.unit
def test_an_initial_point_takes_the_python_path():
    """The Rust writer emits no `x` section, so it must not silently drop one.

    `to_nl(..., initial_point=...)` has to reach `_NLWriter`, which is the only
    writer that knows how to emit the section (#1224).
    """
    m = _objs_only_model()
    y = m._variables[0]
    text = to_nl(m, initial_point={y: 1.25})
    assert "\nx1\n" in text or text.splitlines().count("x1") == 1, (
        "no x section: the initial point was dropped"
    )
    assert "1.25" in text
