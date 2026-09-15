"""Issue #1237: ``floor``/``ceil`` are refused explicitly, at every doorway.

``floor`` and ``ceil`` have no node in the expression IR, and after the entry
experiment (below) they still do not. What changed is *how* discopt says so.

Before this change there were two distinct wrong behaviours:

1. ``dm.floor`` raised ``AttributeError: module 'discopt.modeling' has no
   attribute 'floor'`` -- indistinguishable from a name someone forgot to
   export, rather than a deliberate exclusion.
2. The GAMS importer **silently accepted** ``ceil(x)`` over a variable. Its
   ``_map_func`` fell through to an unchecked node named ``ceil``, so ``from_gams``
   returned a model carrying a node nothing downstream understands. The failure
   surfaced much later, mid-solve, as an opaque ``ValueError: Unknown function:
   'ceil'`` -- *after* the incumbent-verification snapshot had already failed
   and disabled the false-primal guard for that solve. ``serialize._known_funcs``
   documents exactly this failure mode for the deserialization doorway, where it
   was already fixed; the GAMS doorway was not.

Entry experiment (CLAUDE.md §4), run before implementing
(``scripts/i1237_floor_ceil_corpus_probe.py``):

    Hypothesis: endogenous floor/ceil appear in enough of the MINLPLib corpus to
    justify the IR + FBBT + relaxation + reformulation work.
    Measurement: **6,380 instance files** across three formats -- 6,221 JuMP
    models from MINLPLib.jl (``lanl-ansi/MINLPLib.jl``, which includes the 1,513
    MINLPLib2 instances), 153 ``.nl`` files scanned for opcodes o13 (floor) /
    o14 (ceil), and 6 ``.gms`` files. AMPL/GAMS emit o13/o14 only for a
    NON-literal argument, and JuMP's nonlinear macros accept ``floor``/``ceil``
    over a variable, so an instance using one would appear rather than be dropped.
    Result: **0** endogenous uses, over 884,764 executed checks.
    The probe carries a positive control -- 779,084 ``exp``/``log``/``sqrt`` hits
    across the same files -- and exits non-zero if that control is empty, so the
    zero is a measurement and not a scanner that silently read nothing.

The kill criterion fired, and this file pins the outcome it prescribes: a named,
explanatory refusal, uniform across every doorway. It does NOT pin "floor is
unimplementable" -- ``DiscontinuousIntrinsicError`` says *not implemented*
precisely because ``floor`` over a bounded box is relaxable in principle. If
demand ever shows up, native support is a new feature, and these tests change
with it.
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

import discopt
import discopt.modeling as dm
import discopt.modeling.core as core
import pytest
from discopt.modeling.gams_parser import GamsParseError

#: Every name in the shared registry, with the public shim that should refuse it
#: (``intdiv`` has no natural Python spelling and is reachable only through the
#: ``.nl`` parser, so it has no shim).
_SHIMS = {"floor": "floor", "ceil": "ceil", "round": "round_", "trunc": "trunc"}


def _gms(call: str) -> str:
    """A minimal, complete GAMS model whose single equation contains *call*."""
    return f"""
Variables x, obj;
x.lo = 0.5; x.up = 3.5;
Equations c1, defobj;
c1 ..    {call} =L= 2;
defobj.. obj =E= x;
Model m /all/;
Solve m using minlp minimizing obj;
"""


def _from_gams(call: str):
    with tempfile.NamedTemporaryFile("w", suffix=".gms", delete=False) as fh:
        fh.write(_gms(call))
        path = fh.name
    try:
        return dm.from_gams(path)
    finally:
        Path(path).unlink()


# ─────────────────────────────────────────────────────────────
# 1. the names exist, and explain themselves
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shim", sorted(set(_SHIMS.values())))
@pytest.mark.parametrize("namespace", [discopt, dm])
def test_shim_is_exported_and_refuses(namespace, shim):
    """``dm.floor`` exists. Calling it raises, and says why and what to do."""
    fn = getattr(namespace, shim)  # AttributeError here is the #1237 bug itself
    with pytest.raises(core.DiscontinuousIntrinsicError) as exc:
        fn(1.5)
    msg = str(exc.value)
    # The three things the message must carry to beat a bare AttributeError:
    # what it is, why it is refused rather than approximated, and the way out.
    assert "not implemented" in msg
    assert "false certificate" in msg or "different problem" in msg
    assert "m.integer(" in msg
    assert "issues/1237" in msg


def test_error_is_a_notimplementederror():
    """The kill criterion asked for a *named* ``NotImplementedError``."""
    assert issubclass(core.DiscontinuousIntrinsicError, NotImplementedError)
    assert dm.DiscontinuousIntrinsicError is core.DiscontinuousIntrinsicError
    assert discopt.DiscontinuousIntrinsicError is core.DiscontinuousIntrinsicError


def test_reformulation_recipe_in_the_message_is_correct():
    """The message tells the user to bound z by ``floor(l) .. floor(u)``.

    A recipe in an error message is read as instructions, so check the arithmetic
    it prescribes actually brackets ``floor(x)`` on the stated box -- a wrong
    recipe would send someone to a model that cuts off their optimum.
    """
    lo, hi = 0.5, 3.5
    checked = 0
    for x in (lo, 0.999, 1.0, 1.5, 2.0, 2.9999, 3.0, hi):
        z = math.floor(x)
        # (a) the bounds the recipe puts on z do contain floor(x) everywhere on
        #     the box -- so the reformulated model does not cut off any point.
        assert math.floor(lo) <= z <= math.floor(hi), (x, z)
        # (b) the two constraints the recipe writes do identify z as floor(x):
        #     z <= x < z + 1 has floor(x) as its unique integer solution.
        assert z <= x < z + 1, (x, z)
        checked += 1
    assert checked == 8, "the recipe probe evaluated no points"

    # (c) the eps in `x <= z + 1 - eps` is the strictness relaxation, and its
    #     only cost is that x within eps below an integer is excluded. Pin that
    #     it is a *sliver*, not a systematic exclusion, so the recipe is honest.
    eps = 1e-6
    excluded = [x for x in (1.5, 2.0 - eps / 2, 2.0, 2.5) if not x <= math.floor(x) + 1 - eps]
    assert excluded == [2.0 - eps / 2], excluded


# ─────────────────────────────────────────────────────────────
# 2. the name cannot enter the IR through ANY doorway
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(core._UNREPRESENTABLE_INTRINSICS))
def test_functioncall_refuses_at_construction(name):
    """The single choke point: no node is ever built carrying these names.

    Guarding the constructor is what makes the refusal general -- the GAMS
    parser, ``serialize.loads`` and user code all funnel through it, so none of
    them needs its own copy of the check to be safe.
    """
    m = dm.Model("m")
    x = m.continuous("x", lb=0.0, ub=5.0)
    with pytest.raises(core.DiscontinuousIntrinsicError):
        core.FunctionCall(name, x)


def test_registry_and_nl_parser_refusals_agree():
    """The Python registry must not drift from the ``.nl`` parser's C-5 list.

    ``nl_parser.rs`` refuses o13/o14/o55/o57/o58 -- floor, ceil, intdiv, round,
    trunc. A name refused on one import path and accepted on another is exactly
    the inconsistency this issue is about.
    """
    src = Path(__file__).resolve().parents[2] / "crates/discopt-core/src/nl_parser.rs"
    text = src.read_text()
    refused = set(re.findall(r'UnsupportedOpcode\s*\{\s*name:\s*"([a-z0-9]+)"', text))
    assert refused, "probe found no UnsupportedOpcode arms -- the pattern went stale"
    # atan2/if are refused there for a different reason (C-8, arity), so the
    # discontinuous registry is a subset, not an equality.
    assert set(core._UNREPRESENTABLE_INTRINSICS) <= refused, (
        f"Python registry {sorted(core._UNREPRESENTABLE_INTRINSICS)} is not covered "
        f"by the .nl parser's refusals {sorted(refused)}"
    )


# ─────────────────────────────────────────────────────────────
# 3. the GAMS importer refuses at the boundary, not mid-solve
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "call",
    ["ceil(x)", "floor(x)", "round(x)", "mod(x, 2)", "uniform(0,1)*x", "normal(0,1)*x"],
)
def test_gams_import_refuses_endogenous_use(call):
    """``from_gams`` fails on the file, not three layers later during a solve."""
    with pytest.raises(GamsParseError) as exc:
        _from_gams(call)
    msg = str(exc.value)
    assert "Refusing rather than building a node that fails later" in msg or (
        "not implemented" in msg
    ), msg


@pytest.mark.parametrize(
    ("call", "folded"),
    [
        ("x + ceil(2.3)", 3.0),
        ("x + floor(2.7)", 2.0),
        ("x + round(2.567)", 3.0),
    ],
)
def test_gams_import_still_folds_literal_arguments(call, folded):
    """A literal argument is a number, not a step function -- it must still fold.

    Caught by the probe for this change: the first cut refused ``ceil(2.3)``
    inside an equation body too, because ``_eval_const_expr_with_env`` folds only
    data/assignment sites and equation bodies reach ``_map_func`` directly.
    """
    model = _from_gams(call)
    body = str(model._constraints[0].body)
    assert "ceil" not in body and "floor" not in body and "round" not in body
    assert str(int(folded)) in body


@pytest.mark.parametrize(
    ("call", "expect"),
    [("sqrt(x)", "sqrt"), ("exp(x)", "exp"), ("sign(x)", "sign"), ("min(x, 1)", "min")],
)
def test_gams_import_unaffected_for_supported_intrinsics(call, expect):
    """The refusal is targeted: it must not narrow what already imported.

    (``defobj`` becomes the objective, so the model carries one constraint.)
    """
    model = _from_gams(call)
    assert len(model._constraints) == 1
    assert expect in str(model._constraints[0].body)


# ─────────────────────────────────────────────────────────────
# 4. the scope note: why ``sign`` is exported and these are not
# ─────────────────────────────────────────────────────────────


def test_sign_stays_supported_and_is_not_in_the_registry():
    """#1237's scope note asks that ``sign`` either match or be explained.

    It is explained, and the explanation is checkable: ``sign`` is equally
    discontinuous but already has the pieces floor/ceil lack -- an IR variant, an
    FBBT interval rule and an envelope -- because its range is a bounded
    three-point set needing no auxiliary variable. So it keeps working.
    """
    assert "sign" not in core._UNREPRESENTABLE_INTRINSICS
    m = dm.Model("m")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    assert dm.sign(x).func_name == "sign"

    fbbt = Path(__file__).resolve().parents[2] / "crates/discopt-core/src/presolve/fbbt.rs"
    assert "MathFunc::Sign" in fbbt.read_text(), (
        "sign lost its FBBT interval rule; the rationale in "
        "DiscontinuousIntrinsicError's docstring for treating it differently "
        "from floor/ceil no longer holds"
    )
    assert "Why ``sign`` is exported" in core.DiscontinuousIntrinsicError.__doc__


def test_gams_func_table_partitions_into_mapped_and_refused():
    """Pin the partition the ``_map_func`` comment asserts, so it cannot drift.

    The comment there claims the only ``_GAMS_FUNCS`` names without an IR mapping
    are ceil/floor/round/mod/uniform/normal. A name added to ``_GAMS_FUNCS``
    without a mapping used to become a silently-built ``FunctionCall``; now it
    raises, and this test is what tells you which half it landed in.
    """
    from discopt.modeling.core import Constant
    from discopt.modeling.gams_parser import _ModelBuilder, _Parser

    builder = _ModelBuilder.__new__(_ModelBuilder)
    m = dm.Model("m")
    x = m.continuous("x", lb=0.5, ub=3.5)

    refused, mapped, attempts = set(), set(), 0
    for name in sorted(_Parser._GAMS_FUNCS):
        for argv in ([x], [x, Constant(2.0)]):  # try unary, then binary arity
            attempts += 1
            try:
                builder._map_func(name, argv)
            except GamsParseError:
                continue
            except (IndexError, TypeError):
                continue
            mapped.add(name)
            break
        else:
            refused.add(name)

    assert attempts >= 2 * len(_Parser._GAMS_FUNCS) - len(mapped), "probe short-circuited"
    assert refused == {"ceil", "floor", "round", "mod", "uniform", "normal"}
    assert mapped and not (mapped & refused)


# ─────────────────────────────────────────────────────────────
# 5. the remaining two doorways: the GMO link, and the GAMS writer
# ─────────────────────────────────────────────────────────────


def test_gmo_link_refuses_with_the_shared_message():
    """The GAMS *link* (GMO instruction stream) is a fourth doorway.

    It already refused these, but from its own hand-maintained
    ``_DISCONTINUOUS`` set with its own wording. That set is now derived from the
    core registry, so it cannot drift -- which is how the GAMS *parser* ended up
    accepting the same names this one rejected.
    """
    from discopt.gams.instructions import (
        _DISCONTINUOUS,
        FUNC_NAME,
        GamsTranslationError,
        _apply_func,
    )
    from discopt.modeling import core as _core

    assert set(_core._UNREPRESENTABLE_INTRINSICS) <= _DISCONTINUOUS

    codes = {name: code for code, name in FUNC_NAME.items()}
    checked = 0
    for name in ("ceil", "floor", "round", "trunc"):
        with pytest.raises(GamsTranslationError) as exc:
            _apply_func(codes[name], [1.0])
        assert "not implemented" in str(exc.value)
        assert "m.integer(" in str(exc.value)  # the shared reformulation recipe
        checked += 1
    assert checked == 4


def test_gams_writer_refuses_a_name_it_cannot_spell():
    """The GAMS *writer* passed any unmapped name straight through.

    Measured before the fix: a one-argument node named `mod` was written as
    `mod(y)`, but GAMS's mod takes two arguments -- so the exported file was not
    valid GAMS and the writer reported success anyway. Same class as the parser's
    silent acceptance, in the other direction.

    (Function names are spelled here without a literal call construction on
    purpose: `test_every_produced_operator_name_is_accepted` scans the tree for
    those and would read this docstring as a node the package produces.)
    """
    from discopt.export import gams as gexp

    m = dm.Model("p")
    y = m.continuous("y", lb=0.5, ub=3.5)
    m.minimize(y)
    m.subject_to(core.FunctionCall("entropy", y) <= 2.0)
    with pytest.raises(ValueError, match="Unknown function in GAMS export"):
        gexp.to_gams(m)

    # and a name it CAN spell still exports, so the guard is not over-broad
    m2 = dm.Model("q")
    z = m2.continuous("z", lb=0.5, ub=3.5)
    m2.minimize(z)
    m2.subject_to(dm.sqrt(z) <= 2.0)
    assert "sqrt(z)" in gexp.to_gams(m2)


def test_every_doorway_into_the_ir_refuses_floor():
    """The closing argument for #1237: no route into the IR accepts ``floor``.

    The issue's defect was that *some* doorways refused and others silently
    accepted, so which behaviour you got depended on how the model arrived. This
    enumerates every route a ``floor`` node could take and asserts all of them
    refuse. A new import path that forgets the registry fails here.
    """
    import tempfile

    from discopt.export import gams as gexp
    from discopt.gams.instructions import FUNC_NAME, _apply_func

    refused = 0

    def refuses(fn):
        nonlocal refused
        try:
            fn()
        except Exception:  # each doorway raises its own layer's error type
            refused += 1
            return True
        return False

    m = dm.Model("m")
    x = m.continuous("x", lb=0.5, ub=3.5)

    assert refuses(lambda: discopt.floor(1.5))  # public API
    assert refuses(lambda: core.FunctionCall("floor", x))  # constructor
    assert refuses(lambda: _from_gams("floor(x)"))  # .gms parser
    assert refuses(  # GAMS link / GMO instruction stream
        lambda: _apply_func({v: k for k, v in FUNC_NAME.items()}["floor"], [1.0])
    )

    import discopt.serialize as ser

    assert refuses(  # serialize.loads
        lambda: ser.loads('{"nodes":[{"op":"call","f":"floor","args":[]}]}')
    )

    def _nl():  # .nl parser (Rust, C-5): objective is the bare o13 over v0
        with tempfile.NamedTemporaryFile("w", suffix=".nl", delete=False) as fh:
            fh.write(
                "g3 0 1 0\n 1 1 1 0 1\n 1 1\n 0 0\n 1 0 0\n 0 0 0 1\n 1 1\n"
                " 0 1\n 0 0 0 0 0\nC0\no13\nv0\nb\n3\nr\n1 0\n"
            )
            path = fh.name
        try:
            return dm.from_nl(path)
        finally:
            Path(path).unlink()

    assert refuses(_nl)

    # and the way OUT: the GAMS writer no longer emits a name it cannot spell
    w = dm.Model("w")
    y = w.continuous("y", lb=0.5, ub=3.5)
    w.minimize(y)
    w.subject_to(core.FunctionCall("entropy", y) <= 2.0)
    assert refuses(lambda: gexp.to_gams(w))

    assert refused == 7, f"only {refused} of 7 doorways refused"
