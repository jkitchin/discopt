"""A registered atom keeps its envelope across ``dumps``/``loads`` — #1248 A × #1246.

The atom tag (``operators.ATOM_ATTR``) is a Python attribute on an ordinary
expression object.  That is what makes registration cheap — no new node type, no
new opcode, every other consumer untouched — but it also means the tag does not
survive a round-trip through the ``.dopt`` document unless the document writes it
down.  It did not.

Losing the tag is **sound**: the reader gets the lowering and relaxes it term by
term, which is the looser, always-valid relaxation.  It is also **silent**, and
the path that loses it is exactly ``dm.solve_batch(workers > 1)``, which ships
every model to its worker through ``dumps``/``loads`` (``batch.py`` §"why models
are shipped as text").  So a user who registered a composite and then reached for
the batch API got the term-by-term bound in every worker, with no error, no
warning, and nothing to see but a slower solve.  Measured before the fix on the
Redlich-Kister body below: ``use_count`` went 0 → 1 on the original model and
1 → 1 on the round-tripped one — the envelope was consulted for the first and
never for the second.

Restoring the tag on read is **not** a matter of trusting the name.  The document
carries the body the registration produced *at write time*, while the relaxer
derives its envelope from whatever the registry holds under that name *now*;
after a ``replace=True`` those are different functions and the new envelope would
cut the old body's true points.  That is the false bound ``atom_of``'s identity
check exists to prevent (commit ``4725635b``), and an identity cannot cross a
process.  The reader therefore re-derives it: re-lower the *current* registration
on the decoded argument and tag only if it reproduces the decoded body exactly.
``test_a_replaced_registration_never_envelopes_a_round_tripped_model`` is the arm
that holds that line, and it is the reason this file exists rather than a
one-line "write the name down".
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.operators import atom_of, get_registered, register_function, registry_snapshot

pytestmark = [pytest.mark.smoke]


def _rk_body(x, L0=3.0, L1=0.0, rt=1.0):
    """The Redlich-Kister binary #1248 is motivated by."""
    return x * (1 - x) * (L0 + L1 * (2 * x - 1)) + rt * (dm.xlogx(x) + dm.xlogx(1 - x))


def _model_using(name: str) -> tuple[dm.Model, object]:
    m = dm.Model("rt")
    x = m.continuous("x", lb=1e-9, ub=1 - 1e-9)
    m.minimize(get_registered(name)(x))
    return m, x


def _fires(model: dm.Model, fn) -> bool:
    """Did the relaxation engine actually consult this atom's envelope?

    ``use_count`` is the direct evidence (CLAUDE.md §6): a node count alone
    cannot tell "the envelope fired" from "the search got lucky", and here it
    could not tell "the tag survived" from "the term-by-term bound happened to
    be as good".
    """
    from discopt._relax.uniform_relax import build_uniform_relaxation

    before = fn.use_count
    build_uniform_relaxation(model)
    return fn.use_count > before


def test_a_registered_atom_keeps_its_envelope_across_a_round_trip():
    with registry_snapshot():
        register_function("t1248rt_keep", _rk_body, replace=True)
        fn = get_registered("t1248rt_keep")
        m, _ = _model_using("t1248rt_keep")

        # CONTROL: without this, an arm that never fires for either model would
        # pass the "round trip fires" assertion vacuously.
        assert _fires(m, fn), "the envelope did not fire on the ORIGINAL model"

        m2 = dm.loads(dm.dumps(m))
        assert atom_of(m2._objective.expression) is not None, "tag lost by the round trip"
        assert _fires(m2, fn), "the envelope did not fire on the ROUND-TRIPPED model"


def test_a_replaced_registration_never_envelopes_a_round_tripped_model():
    """The soundness arm: a name is not an identity.

    Dump a model built against one definition, rebind the name to a *different*
    function, then load.  The loaded body is still the old one, so tagging it
    would hand the relaxer the new definition's envelope for the old body — a
    false bound.  The reader must decline to tag, and the model must still solve
    to the old body's true optimum.
    """
    with registry_snapshot():
        register_function("t1248rt_swap", lambda t: dm.exp(t), replace=True)
        m = dm.Model("swap")
        t = m.continuous("t", lb=-1.0, ub=1.0)
        m.minimize(get_registered("t1248rt_swap")(t))
        text = dm.dumps(m)

        # Same name, a function that differs everywhere by +100.
        register_function("t1248rt_swap", lambda s: dm.exp(s) + 100.0, replace=True)
        new = get_registered("t1248rt_swap")
        uses_before = new.use_count

        m2 = dm.loads(text)
        assert atom_of(m2._objective.expression) is None, (
            "a replaced registration was allowed to claim a body it did not build"
        )

        r = m2.solve()
        truth = float(__import__("math").exp(-1.0))
        assert r.bound <= truth + 1e-6, f"false bound: {r.bound} > {truth}"
        assert r.objective == pytest.approx(truth, abs=1e-5)
        assert new.use_count == uses_before, "the replaced envelope was consulted anyway"


def test_a_name_not_registered_in_this_process_loads_term_by_term():
    """A worker that never imported the registration still gets a valid model.

    Sound, and the documented degradation: the document carries the lowering, so
    the model is complete without the registry.
    """
    with registry_snapshot():
        register_function("t1248rt_gone", _rk_body, replace=True)
        m, _ = _model_using("t1248rt_gone")
        text = dm.dumps(m)

    # Outside the snapshot the name is unregistered again.
    assert get_registered("t1248rt_gone") is None
    m2 = dm.loads(text)
    assert atom_of(m2._objective.expression) is None
    r = m2.solve()
    assert r.status == "optimal"
    # The RK body with L0=3, L1=0, rt=1 is symmetric about x=1/2, where it is a
    # local max; the minima sit near the ends.
    assert r.objective < 0.0


def test_the_document_records_the_atom_by_name_and_argument():
    """The written form, pinned: a reader that ignores it still loads the model."""
    import json

    with registry_snapshot():
        register_function("t1248rt_doc", _rk_body, replace=True)
        m, _ = _model_using("t1248rt_doc")
        doc = json.loads(dm.dumps(m))

    tagged = [nd for nd in doc["nodes"] if "atom" in nd]
    assert len(tagged) == 1, f"expected exactly one tagged node, got {len(tagged)}"
    spec = tagged[0]["atom"]
    assert spec["n"] == "t1248rt_doc"
    assert doc["nodes"][spec["arg"]]["op"] == "var", "the argument reference is not the variable"


@pytest.mark.slow
def test_a_solve_batch_worker_restores_the_registered_atom():
    """The end-to-end arm, in a subprocess — see ``scripts/entry_1248_batch_atom.py``.

    It cannot run in-process: under ``spawn`` the worker re-imports the parent's
    ``__main__``, which under pytest is pytest rather than this module, so a
    registration made here would be absent from the worker and the arm would
    measure the unregistered path whether or not the fix is present. The script
    puts the registration in a real ``__main__``.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "entry_1248_batch_atom.py"
    assert script.is_file(), f"missing {script}"
    proc = subprocess.run(
        [_sys.executable, "-u", str(script)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert "EXECUTED COMPARISONS: 4" in proc.stdout, (
        f"the probe compared nothing\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.returncode == 0, (
        f"a solve_batch worker lost the registered atom\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
