"""#1666: ``from_gams`` built wrong models from ordinary GAMS and reported them optimal.

The transport LP (optimum 1280) came back 0.0, 552.5 or 0.0 depending on how
labels and declaration text were spelled -- all well-formed GAMS:

* **B** unquoted explanatory text (``Sets i canning plants / ... /``) was read as
  more declared names, so ``capacity`` and ``cases`` became parameters and the
  data landed on the wrong symbol;
* **C** a hyphenated label (``san-diego``) was split at the ``-`` into two
  labels, and the tokenizer dropped what it could not read;
* **D** numeric labels (``1 2``) were never collected as table column headers,
  so the cost table came out empty.

Every expected value here was checked against GAMS 53 itself. For **D** as the
issue writes it, GAMS refuses the table (``$225 Floating entry ignored``: the
short headers ``1`` and ``2`` do not sit over their values), so the right answer
there is a ``GamsParseError``; the same labels laid out under their columns
solve to 1280.

The fix makes the reader follow GAMS's own rules (explanatory text runs to the
end of the line; a label is one unbroken run of characters; table values sit
under their column header) and refuses -- ``GamsParseError`` -- input it cannot
place: an unknown character, a value under no column header, a label outside its
declared domain.
"""

from __future__ import annotations

import textwrap

import pytest
from discopt.modeling.gams_parser import GamsParseError, _Parser, _tokenize, parse_gams

CLEAN = textwrap.dedent("""\
    Sets i / seattle, sandiego /
         j / newyork, chicago /;
    Parameters a(i) / seattle 350, sandiego 600 /
               b(j) / newyork 325, chicago 275 /;
    Table c(i,j)
                newyork  chicago
    seattle       2.5      1.7
    sandiego      2.5      1.8;
    Variables z;
    Positive Variables x(i,j);
    Equations cost, supply(i), demand(j);
    cost.. z =e= sum((i,j), c(i,j)*x(i,j));
    supply(i).. sum(j, x(i,j)) =l= a(i);
    demand(j).. sum(i, x(i,j)) =g= b(j);
    Model t / all /;
    Solve t using lp minimizing z;
    """)

TEXT = (
    CLEAN.replace("Sets i /", "Sets i canning plants /")
    .replace("     j /", "     j markets /")
    .replace("Parameters a(i) /", "Parameters a(i) capacity in cases /")
    .replace("b(j) /", "b(j) demand in cases /")
    .replace("Table c(i,j)", "Table c(i,j) distance in thousand miles")
)
HYPHEN = CLEAN.replace("sandiego", "san-diego").replace("newyork", "new-york")
NUMERIC = CLEAN.replace("newyork", "1").replace("chicago", "2")
NUMERIC_ALIGNED = NUMERIC.replace("            1  2\n", "              1        2\n")

TRANSPORT_OPT = 1280.0


def _parse(src: str) -> _Parser:
    p = _Parser(_tokenize(src))
    p.parse()
    return p


@pytest.mark.parametrize(
    "src",
    [CLEAN, TEXT, HYPHEN, NUMERIC_ALIGNED],
    ids=["clean", "text", "hyphen", "numeric-aligned"],
)
def test_transport_variants_solve_to_the_same_optimum(src):
    r = parse_gams(src).solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(TRANSPORT_OPT, abs=1e-6)


def test_numeric_headers_not_over_their_values_refuse():
    """GAMS: ``$225 Floating entry ignored``. Not 0.0, and not a guess."""
    assert "            1  2\n" in NUMERIC
    with pytest.raises(GamsParseError, match="column label"):
        parse_gams(NUMERIC)


def test_explanatory_text_is_not_declared_names():
    p = _parse(TEXT)
    assert set(p.sets) == {"i", "j"}
    assert p.sets["i"].description == "canning plants"
    assert set(p.parameters) == {"a", "b"}
    assert p.parameters["a"].description == "capacity in cases"
    assert p.tables["c"].description == "distance in thousand miles"


def test_text_on_scalar_variable_equation_and_model_declarations():
    src = textwrap.dedent("""\
        Scalar f freight in dollars per case / 90 /;
        Variables z total transportation cost;
        Equations cost define objective function;
        cost.. z =e= f;
        Model transport the transportation problem / all /;
        Solve transport using lp minimizing z;
        """)
    p = _parse(src)
    assert set(p.scalars) == {"f"} and p.scalars["f"].value == 90.0
    assert set(p.variables) == {"z"}
    assert set(p.equations) == {"cost"}
    assert p.models["transport"].equations == ["all"]
    assert parse_gams(src).solve().objective == pytest.approx(90.0)


def test_hyphenated_labels_stay_whole():
    p = _parse(HYPHEN)
    assert p.sets["i"].elements == ["seattle", "san-diego"]
    assert p.parameters["b"].data == {"new-york": 325.0, "chicago": 275.0}
    assert p.tables["c"].data[("san-diego", "new-york")] == 2.5


def test_set_element_text_is_not_more_elements():
    p = _parse("Set i / a first plant, b 'second plant'\n c /;")
    assert p.sets["i"].elements == ["a", "b", "c"]


def test_numeric_table_headers_and_row_labels():
    src = textwrap.dedent("""\
        Sets i / 1, 2 /  j / 10, 20 /;
        Table t(i,j)
              10    20
        1    1.5   2.5
        2   -3.0   4.0;
        """)
    data = _parse(src).tables["t"].data
    assert data == {("1", "10"): 1.5, ("1", "20"): 2.5, ("2", "10"): -3.0, ("2", "20"): 4.0}


def test_sparse_table_values_sit_under_their_column():
    """Blanks are zero: the value under ``c3`` belongs to ``c3``, not ``c2``."""
    src = textwrap.dedent("""\
        Table t(i,j)
              c1   c2   c3
        r1    1.0        3.0
        r2         5.0   6.0 ;
        """)
    data = _parse(src).tables["t"].data
    assert data == {("r1", "c1"): 1.0, ("r1", "c3"): 3.0, ("r2", "c2"): 5.0, ("r2", "c3"): 6.0}


def test_table_continuation_block():
    src = textwrap.dedent("""\
        Table t(i,j)
              c1   c2
        r1     1    2
        +     c3
        r1     3 ;
        """)
    assert _parse(src).tables["t"].data == {("r1", "c1"): 1.0, ("r1", "c2"): 2.0, ("r1", "c3"): 3.0}


def test_duplicate_row_in_one_table_block_refuses():
    """GAMS: ``$176 A row with the same name has been defined before``."""
    src = textwrap.dedent("""\
        Table t(i,j)
              c1   c2
        r1     1    2
        r1     3    4 ;
        """)
    with pytest.raises(GamsParseError, match="r1"):
        _parse(src)


def test_data_after_a_record_on_the_same_line_refuses():
    """GAMS: ``$334 Illegal data following a data element``."""
    with pytest.raises(GamsParseError, match="after a data record"):
        _parse("Set i / a, b /;\nParameter p(i) / a 1 b 2 /;")


def test_table_value_under_no_column_refuses():
    src = textwrap.dedent("""\
        Table t(i,j)
              c1   c2
        r1                  7 ;
        """)
    with pytest.raises(GamsParseError, match="column"):
        _parse(src)


def test_unknown_character_refuses():
    with pytest.raises(GamsParseError, match="line 2"):
        _tokenize("Scalar s / 3 /;\nScalar t / 4 / # comment;\n")


def test_ontext_block_is_a_comment():
    src = "$ontext\nanything # goes here & there\n$offtext\nScalar s / 3 /;\n"
    p = _parse(src)
    assert set(p.scalars) == {"s"}


def test_label_outside_its_domain_refuses():
    src = CLEAN.replace("sandiego 600", "boston 600")
    with pytest.raises(GamsParseError, match="boston"):
        parse_gams(src)


def test_table_label_outside_its_domain_refuses():
    src = CLEAN.replace("sandiego      2.5", "boston        2.5")
    with pytest.raises(GamsParseError, match="boston"):
        parse_gams(src)


def test_domain_check_is_case_insensitive():
    src = CLEAN.replace("seattle 350", "Seattle 350")
    assert parse_gams(src).solve().objective == pytest.approx(TRANSPORT_OPT, abs=1e-6)


def test_compile_time_variable_is_substituted():
    """MINLPLib's ``Solve m using %NLP% ...``; the tokenizer used to drop the ``%``."""
    src = textwrap.dedent("""\
        Variables z;
        Equations e;
        e.. z =e= 3;
        Model m / all /;
        $if not set NLP $set NLP LP
        $if not set NLP $set NLP MINLP
        Solve m using %NLP% minimizing z;
        """)
    p = _parse(src)
    assert p.solves[0].model_type == "lp"


def test_undefined_compile_time_variable_refuses():
    with pytest.raises(GamsParseError, match="%NLP%"):
        _tokenize("Solve m using %NLP% minimizing z;")
