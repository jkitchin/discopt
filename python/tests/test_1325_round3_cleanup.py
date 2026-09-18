"""#1325: three loud-or-cosmetic defects from the round-3 adversarial sweep.

Nothing here was silently wrong; each was a message that misnamed its own cause,
a guard that failed open on a reload, or a comment a later measurement falsified.
"""

import importlib
import sys

import pytest
from discopt.modeling.gams_parser import GamsParseError, parse_gams

# ── 1. discopt.nn installs exactly one alias finder ──────────────────────


@pytest.mark.smoke
def test_reloading_discopt_nn_does_not_stack_alias_finders():
    """#1314's guard used ``isinstance(f, _AliasFinder)``. Re-executing the
    module builds a NEW ``_AliasFinder`` class, so the finder already on
    ``sys.meta_path`` is not an instance of it: a reload took the count 1 -> 2,
    and a ``sys.modules`` purge plus re-import took it to 3."""
    import discopt.nn as nn

    def _count():
        return sum(1 for f in sys.meta_path if getattr(f, "_discopt_nn_alias_finder", False))

    assert _count() == 1

    with pytest.warns(DeprecationWarning):
        importlib.reload(nn)
    assert _count() == 1

    for name in [n for n in list(sys.modules) if n == "discopt.nn" or n.startswith("discopt.nn.")]:
        del sys.modules[name]
    with pytest.warns(DeprecationWarning):
        import discopt.nn as nn2  # noqa: F401
    assert _count() == 1

    # And aliasing still resolves to the same object as discopt.ml.
    import discopt.ml.network as ml_network
    import discopt.nn.network as nn_network

    assert nn_network is ml_network


@pytest.mark.smoke
def test_discopt_nn_imports_importlib_util_itself():
    """``find_spec`` is ``importlib.util.find_spec``; ``import importlib`` does
    not bring the submodule in. It worked only because something else in the
    process had imported it, and the finder's broad ``except`` would have turned
    "the submodule is missing" into "the alias does not exist"."""
    import discopt.nn as nn

    src = open(nn.__file__).read()
    assert "import importlib.util" in src


# ── 2. GAMS min/max arity ────────────────────────────────────────────────


_ONE_ARG_ENDOGENOUS = """
Variables x1, obj;
Equations objdef;
x1.lo = 0; x1.up = 5;
objdef.. obj =e= {fn}(x1);
Model mm /all/;
Solve mm using nlp minimizing obj;
"""

_ONE_ARG_CONSTANT = """
Variables x1, obj;
Equations objdef;
x1.lo = 0; x1.up = 5;
objdef.. obj =e= x1 + {fn}(4);
Model mm /all/;
Solve mm using nlp minimizing obj;
"""


@pytest.mark.smoke
@pytest.mark.parametrize("fn", ["min", "max"])
@pytest.mark.parametrize("src", [_ONE_ARG_ENDOGENOUS, _ONE_ARG_CONSTANT])
def test_one_argument_min_max_is_a_gams_parse_error_in_both_arms(fn, src):
    """The endogenous arm raised a bare numpy ``TypeError`` about ``np.minimum``
    while the constant-folding arm accepted ``min(4)`` and returned 4 -- one
    construct, two unrelated behaviours depending on whether its argument
    happened to fold."""
    with pytest.raises(GamsParseError, match="two or more arguments"):
        parse_gams(src.format(fn=fn))


@pytest.mark.smoke
@pytest.mark.parametrize("fn", ["min", "max"])
def test_two_and_three_argument_min_max_still_parse(fn):
    """No false positive: the n-ary forms #1312 fixed are untouched."""
    for args in ("x1, 2", "x1, 2, 3"):
        m = parse_gams(
            f"""
            Variables x1, obj;
            Equations objdef;
            x1.lo = 0; x1.up = 5;
            objdef.. obj =e= {fn}({args});
            Model mm /all/;
            Solve mm using nlp minimizing obj;
            """
        )
        assert any(v.name == "x1" for v in m._variables)


# ── 3. GAMS smin/smax in an equation body ────────────────────────────────


@pytest.mark.smoke
@pytest.mark.parametrize("fn", ["smin", "smax"])
def test_smin_smax_in_an_equation_body_names_the_unsupported_construct(fn):
    """It used to fail with "Unresolved reference: 'i'" -- naming the index,
    which is not the problem, instead of the construct, which is."""
    src = f"""
    Set i /1*3/;
    Variables x(i), obj;
    Equations objdef;
    objdef.. obj =e= {fn}(i, x(i));
    Model mm /all/;
    Solve mm using nlp minimizing obj;
    """
    with pytest.raises(GamsParseError) as exc:
        parse_gams(src)
    text = str(exc.value)
    assert fn in text
    assert "not supported" in text
    assert "Unresolved reference" not in text


@pytest.mark.smoke
def test_sum_over_an_index_set_still_works():
    """No false positive: the supported indexed operations are untouched."""
    m = parse_gams(
        """
        Set i /1*3/;
        Variables x(i), obj;
        Equations objdef;
        x.lo(i) = 1; x.up(i) = 4;
        objdef.. obj =e= sum(i, x(i));
        Model mm /all/;
        Solve mm using nlp minimizing obj;
        """
    )
    assert m.solve().objective == pytest.approx(3.0, abs=1e-5)


# ── 4. the comment #1309 falsified ───────────────────────────────────────


@pytest.mark.smoke
def test_the_lp_pounce_status_table_no_longer_claims_code_2_is_sound():
    """#1309 measured the opposite: the barrier method raises Ipopt code 2 from
    numerical failure on huge-magnitude-bound problems with no infeasibility
    behind it, which is why every such exit is cross-checked against the elastic
    Phase-1 LP. A comment that still says the old thing is how the next reader
    removes the cross-check."""
    import discopt.solvers.lp_pounce as lp_pounce

    src = open(lp_pounce.__file__).read()
    marker = "_LP_STATUS_MAP = {"
    assert marker in src, "the probe is looking at the wrong file"
    # The comment block the table carries, and only that block: the falsified
    # claim is quoted verbatim elsewhere in the module, in the note that records
    # its falsification.
    block = src[: src.index(marker)].rsplit("\n\n", 1)[-1]
    assert "#1309" in block, f"the table's comment does not mention the falsification:\n{block}"
    assert "is a sound INFEASIBLE;" not in block
    assert "cross-checked" in block
