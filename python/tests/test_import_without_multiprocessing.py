"""Guard: ``import discopt`` must not require the ``_multiprocessing`` extension.

``discopt.modeling`` re-exports :func:`discopt.batch.solve_batch`, so
``discopt.batch`` is imported by every ``import discopt``. It used to
``import multiprocessing`` at module level, which pulls the ``_multiprocessing``
C extension — a capability that only ``solve_batch(workers>1)`` uses, and that
``workers=1`` returns before ever reaching.

Not every CPython build ships that extension. Pyodide/WebAssembly has no
``_multiprocessing`` at all, so the eager import made the entire package
unimportable in the browser for a feature no serial solve touches. The fix is
to import it inside the two functions that build a pool.

Two guards, because they fail for different reasons:

* :func:`test_import_discopt_does_not_load_multiprocessing` runs in a fresh
  subprocess and checks ``sys.modules`` — it catches a re-introduced top-level
  import anywhere on the ``import discopt`` path, not just in ``batch.py``.
* :func:`test_discopt_imports_and_solves_without_multiprocessing` installs a
  ``sys.meta_path`` finder that raises :exc:`ModuleNotFoundError` for the
  multiprocessing names, reproducing the Pyodide environment rather than
  approximating it, and then solves an LP, a MILP and an MINLP.

The parallel path is exercised too (:func:`test_solve_batch_workers_still_works`):
moving an import into a function is only correct if the function can still make
it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.smoke

# Names Pyodide does not provide. ``multiprocessing`` is pure Python but is
# useless (and imports ``_multiprocessing``) without the extension, so both are
# blocked: a fix that imported the wrapper and failed later would pass a test
# that blocked only the extension.
_BLOCKED = ("multiprocessing", "_multiprocessing")


def _run(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter, returning the completed process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_import_discopt_does_not_load_multiprocessing() -> None:
    proc = _run(
        """
        import sys

        import discopt  # noqa: F401

        leaked = [n for n in ("multiprocessing", "_multiprocessing") if n in sys.modules]
        # Prove the probe fired: if `discopt` somehow imported as an empty
        # namespace this check would pass vacuously (CLAUDE.md, "Measurement &
        # instrumentation discipline" §6).
        assert "discopt.batch" in sys.modules, "discopt.batch was not imported at all"
        assert hasattr(discopt, "solve_batch"), "solve_batch is no longer exported"
        print("CHECKS=2")
        if leaked:
            raise SystemExit(f"import discopt loaded {leaked}")
        """
    )
    assert "CHECKS=2" in proc.stdout, proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_discopt_imports_and_solves_without_multiprocessing() -> None:
    """The Pyodide environment, reproduced: the names simply are not importable."""
    proc = _run(
        f"""
        import sys

        BLOCKED = {_BLOCKED!r}


        class _Blocker:
            def find_module(self, name, path=None):  # legacy hook, kept harmless
                return None

            def find_spec(self, name, path=None, target=None):
                if name in BLOCKED or name.split(".")[0] in BLOCKED:
                    raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
                return None


        for _name in BLOCKED:
            sys.modules.pop(_name, None)
        sys.meta_path.insert(0, _Blocker())

        # The blocker must actually bite, or everything below is vacuous.
        try:
            import multiprocessing  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            raise SystemExit("blocker did not fire -- multiprocessing still importable")

        import discopt.modeling as dm

        checks = 0

        lp = dm.Model("lp")
        x = lp.continuous("x", shape=(3,), lb=0, ub=1)
        lp.minimize(dm.sum([x[i] for i in range(3)]))
        lp.subject_to(dm.sum([x[i] for i in range(3)]) >= 1.5)
        assert lp.solve().status == "optimal"
        checks += 1

        milp = dm.Model("milp")
        y = milp.binary("y", shape=(4,))
        milp.maximize(dm.sum([float(i + 1) * y[i] for i in range(4)]))
        milp.subject_to(dm.sum([y[i] for i in range(4)]) <= 2)
        assert milp.solve().status == "optimal"
        checks += 1

        minlp = dm.Model("minlp")
        a = minlp.continuous("a", lb=-3, ub=3)
        b = minlp.integer("b", lb=-3, ub=3)
        minlp.minimize(-a - b)
        minlp.subject_to(a * a + b * b <= 10)
        assert minlp.solve().status == "optimal"
        checks += 1

        assert checks == 3, checks
        # Still blocked on the way out: nothing re-imported it behind our back.
        assert not [n for n in BLOCKED if n in sys.modules], sys.modules.keys()
        print(f"CHECKS={{checks}}")
        """
    )
    assert "CHECKS=3" in proc.stdout, proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_solve_batch_workers_still_works() -> None:
    """Moving the import into the pool path must not break the pool path."""
    proc = _run(
        """
        import discopt.modeling as dm

        if __name__ == "__main__":
            models = []
            for k in range(2):
                m = dm.Model(f"m{k}")
                x = m.continuous("x", shape=(2,), lb=0, ub=1)
                m.minimize(dm.sum([float(k + 1) * x[i] for i in range(2)]))
                m.subject_to(dm.sum([x[i] for i in range(2)]) >= 0.5)
                models.append(m)

            results = dm.solve_batch(models, workers=2)
            assert len(results) == 2, results
            assert all(r.status == "optimal" for r in results), [r.status for r in results]
            print("CHECKS=2")
        """
    )
    assert "CHECKS=2" in proc.stdout, proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stdout + proc.stderr
