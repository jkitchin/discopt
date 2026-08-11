"""#863: sparse implied-integer detection must mark EXACTLY the same variables.

``detect_implied_integers`` built an ``np.zeros(n)`` per equality body, retained one
per integer-data row, and re-derived each row's support with ``np.nonzero`` on every
fixpoint round. On ``watercontamination0202`` (106,711 variables / 107,209
constraints) that cost **71.1 s and +31.2 GiB RSS**; after switching to
``_extract_linear_coefficients_sparse`` and a precomputed support it is **0.92 s**.

Speed is not the thing under test here. This function marks continuous variables
INTEGER, so a representation change is soundness-relevant: marking a variable that is
not implied-integer cuts off feasible points — the cardinal correctness violation
(CLAUDE.md §1). What is tested is that the marked *set* is identical, against an
independent reimplementation of the dense predecessor, over the in-repo ``.nl``
corpus (MILP / MIQP / MINLP alike) plus targeted structures.

``_dense_reference_detect`` below is a faithful transcription of the pre-#863 body.
It must not be "simplified" to share code with the implementation — its whole value
is being an independent oracle.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.gdp_reformulate import _is_linear  # noqa: E402
from discopt._relax.implied_integer import (  # noqa: E402
    _INT_TOL,
    _is_int_value,
    detect_implied_integers,
    mark_implied_integers,
)
from discopt._relax.problem_classifier import (  # noqa: E402
    _extract_linear_coefficients,
    _NotLinearError,
)
from discopt.modeling.core import Constraint, VarType  # noqa: E402

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def _dense_reference_detect(model) -> set[tuple[int, int]]:
    """The pre-#863 dense implementation, verbatim, as an independent oracle."""
    n = sum(v.size for v in model._variables)
    flat = [(v, e) for v in model._variables for e in range(v.size)]
    is_int = np.array(
        [flat[i][0].var_type in (VarType.INTEGER, VarType.BINARY) for i in range(n)],
        dtype=bool,
    )

    eq_rows: list[tuple[np.ndarray, float]] = []
    for c in model._constraints:
        if not isinstance(c, Constraint) or c.sense != "==":
            continue
        if not _is_linear(c.body):
            continue
        try:
            a, const = _extract_linear_coefficients(c.body, model, n)
        except _NotLinearError:
            continue
        a = np.asarray(a, dtype=np.float64)
        if not np.all(np.abs(a - np.round(a)) <= _INT_TOL) or not _is_int_value(float(const)):
            continue
        eq_rows.append((a, float(const)))

    marked: set[tuple[int, int]] = set()
    changed = True
    while changed:
        changed = False
        for a, _const in eq_rows:
            nz = np.nonzero(np.abs(a) > _INT_TOL)[0]
            for idx in nz:
                if is_int[idx]:
                    continue
                if abs(abs(a[idx]) - 1.0) > _INT_TOL:
                    continue
                if all(is_int[j] for j in nz if j != idx):
                    var, elem = flat[idx]
                    marked.add((var._index, elem))
                    is_int[idx] = True
                    changed = True
    return marked


# --------------------------------------------------------------------------
# Identity on real instances
# --------------------------------------------------------------------------

_CORPUS = sorted(p.stem for p in _NL_DIR.glob("*.nl"))


@pytest.mark.parametrize("name", _CORPUS)
def test_marked_set_is_identical_to_the_dense_reference_on_the_corpus(name):
    """The soundness gate: same set, on every .nl instance in the repo corpus."""
    model = dm.from_nl(str(_NL_DIR / f"{name}.nl"))
    expected = _dense_reference_detect(model)
    actual = detect_implied_integers(model)
    assert actual == expected, (
        f"{name}: implied-integer set changed. "
        f"newly marked (UNSOUND if wrong): {sorted(actual - expected)}; "
        f"no longer marked: {sorted(expected - actual)}"
    )


def test_the_corpus_actually_exercises_a_nonempty_marking():
    """A parity test over a corpus where nothing is ever marked proves nothing.
    At least one instance must produce a non-empty set."""
    nonempty = []
    for name in _CORPUS:
        model = dm.from_nl(str(_NL_DIR / f"{name}.nl"))
        if detect_implied_integers(model):
            nonempty.append(name)
    assert nonempty, (
        "no corpus instance yields an implied-integer marking, so the parity test "
        "above is vacuous; add an instance with an integer-defining equality"
    )
    print(f"instances with implied integers: {nonempty}")


# --------------------------------------------------------------------------
# Identity on targeted structures the corpus may not cover
# --------------------------------------------------------------------------


def _trim_loss_model():
    """The ``ex126x`` structure from the module docstring: ``x - b0 - 2 b1 - 4 b2 = 0``
    with binaries forces ``x`` integer."""
    m = dm.Model("trimloss")
    x = m.continuous("x", lb=0.0, ub=7.0)
    bs = [m.binary(f"b{i}") for i in range(3)]
    m.minimize(x)
    m.subject_to(x - bs[0] - 2.0 * bs[1] - 4.0 * bs[2] == 0)
    return m


def _chain_model():
    """A chain, so the fixpoint iteration matters: y is integer only after x is."""
    m = dm.Model("chain")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    b = m.binary("b")
    m.minimize(x + y)
    m.subject_to(x - b == 0)
    m.subject_to(y - x == 0)
    return m


def _two_unknowns_model():
    """Must mark NOTHING: a row with two unknowns proves neither of them integer."""
    m = dm.Model("twounknowns")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    b = m.binary("b")
    m.minimize(x + y)
    m.subject_to(x + y - b == 0)
    return m


def _fractional_coefficient_model():
    """Must mark NOTHING: a non-integer coefficient kills the whole row."""
    m = dm.Model("fractional")
    x = m.continuous("x", lb=0.0, ub=5.0)
    b = m.binary("b")
    m.minimize(x)
    m.subject_to(x - 0.5 * b == 0)
    return m


def _coefficient_not_pm_one_model():
    """Must mark NOTHING for x: coefficient 2 admits x = 0.5 when b = 1."""
    m = dm.Model("coeff2")
    x = m.continuous("x", lb=0.0, ub=5.0)
    b = m.binary("b")
    m.minimize(x)
    m.subject_to(2.0 * x - b == 0)
    return m


def _inequality_only_model():
    """Must mark NOTHING: range links are never sufficient."""
    m = dm.Model("ineq")
    x = m.continuous("x", lb=0.0, ub=5.0)
    b = m.binary("b")
    m.minimize(x)
    m.subject_to(x - b <= 0)
    m.subject_to(x - b >= 0)
    return m


def _wide_narrow_model():
    """The #863 shape: many variables, one integer-defining equality among them."""
    m = dm.Model("widenarrow")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=9.0) for i in range(400)]
    b = m.binary("b")
    m.minimize(sum(xs))
    m.subject_to(xs[137] - b == 0)
    m.subject_to(sum(xs) >= 1)
    return m


_BUILDERS = {
    "trim_loss": (_trim_loss_model, 1),
    "chain": (_chain_model, 2),
    "two_unknowns": (_two_unknowns_model, 0),
    "fractional_coefficient": (_fractional_coefficient_model, 0),
    "coefficient_not_pm_one": (_coefficient_not_pm_one_model, 0),
    "inequality_only": (_inequality_only_model, 0),
    "wide_narrow": (_wide_narrow_model, 1),
}


@pytest.mark.parametrize("label", sorted(_BUILDERS))
def test_marked_set_is_identical_on_targeted_structures(label):
    builder, expected_size = _BUILDERS[label]
    model = builder()
    expected = _dense_reference_detect(model)
    actual = detect_implied_integers(model)
    assert actual == expected, (
        f"{label}: newly marked {sorted(actual - expected)}, lost {sorted(expected - actual)}"
    )
    # Pin the absolute answer too, so a bug that breaks BOTH implementations the
    # same way (e.g. a shared extractor regression) is still caught.
    assert len(actual) == expected_size, f"{label}: expected {expected_size} markings, got {actual}"


def test_no_full_width_row_is_ever_materialised(monkeypatch):
    """The #863 mechanism, asserted directly: the detector must never call the DENSE
    coefficient extractor, which opens with ``np.zeros(n)`` per equality body.

    This is the test that fails before the change (the old body called it once per
    equality constraint) and passes after. A memory-threshold assertion was tried
    first and is useless at test scale — the O(n) ``flat``/``is_int`` structures,
    which are unavoidable and unchanged, dominate the peak.
    """
    import discopt._relax.implied_integer as impl

    calls: list[int] = []

    def _forbidden(*a, **k):
        calls.append(1)
        raise AssertionError("detect_implied_integers built a dense full-width row")

    # The name is only reachable if the module still imports the dense wrapper.
    monkeypatch.setattr(impl, "_extract_linear_coefficients", _forbidden, raising=False)
    got = detect_implied_integers(_wide_narrow_model())
    assert not calls, "the dense full-width row extractor is back"
    assert len(got) == 1, "and the detection still works"


def test_row_support_is_computed_once_not_once_per_fixpoint_round(monkeypatch):
    """The other half of the #863 mechanism: the dense body re-derived each row's
    support with ``np.nonzero`` on EVERY fixpoint round, an O(n) scan per row per
    round. ``_chain_model`` needs three rounds to converge, so a per-round rescan is
    observable: the support of each row must be built exactly once."""
    import discopt._relax.problem_classifier as pcmod

    model = _chain_model()
    real = pcmod._extract_linear_coefficients_sparse
    extractions: list[int] = []

    def _counting(*a, **k):
        extractions.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pcmod, "_extract_linear_coefficients_sparse", _counting)
    # implied_integer imported the symbol directly, so patch it there too.
    import discopt._relax.implied_integer as impl

    monkeypatch.setattr(impl, "_extract_linear_coefficients_sparse", _counting)

    marked = detect_implied_integers(model)
    assert len(marked) == 2, "the chain must need the fixpoint iteration"
    # Two equality constraints, extracted once each -- not once per round.
    assert len(extractions) == 2, (
        f"rows were extracted {len(extractions)} times for 2 equalities; the support "
        "must be precomputed, not rebuilt per fixpoint round"
    )


def test_mark_implied_integers_still_promotes_the_variable():
    """End to end: the detection feeds ``mark_implied_integers``, which is what
    actually changes the model. A representation bug here would silently change the
    feasible set."""
    model = _trim_loss_model()
    x = model._variables[0]
    assert x.var_type == VarType.CONTINUOUS
    assert mark_implied_integers(model) == 1
    assert x.var_type == VarType.INTEGER


def test_a_nonlinear_equality_is_still_ignored():
    """``_is_linear`` gates the row; a nonlinear body must contribute nothing."""
    m = dm.Model("nonlin")
    x = m.continuous("x", lb=0.0, ub=5.0)
    b = m.binary("b")
    m.minimize(x)
    m.subject_to(x * x - b == 0)
    assert detect_implied_integers(m) == _dense_reference_detect(m) == set()
