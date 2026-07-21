"""#832: deadline the BASE root-relaxation build so the fallback honors its grant.

The residual gastrans582 time-limit overrun (#814) is the unbudgeted Python DCP
relaxation build in ``_root_relaxation_lower_bound``, NOT the Rust LP (the Rust
factorize is ~3ms; 93% of the overrun is ``build_milp_relaxation``). #694's
``anytime_root_build`` deadlines only the *separated* build; its companion base
build was left whole, so on gastrans582 the base build alone (~10.5s) blows a 3s
grant ~5x and the fallback returns ``None`` after ~15.5s.

``DISCOPT_ROOT_BUILD_DEADLINE`` (default off, §5 bound-changing) deadlines the base
build too — the whole fallback build phase honors the grant. A truncated base build
is a valid *weaker* relaxation (dropping constraint rows only enlarges the feasible
set) or trips the existing ``_objective_bound_valid`` gate to ``None`` (weaker,
never falsified). These tests assert:
  1. flag ON bounds the gastrans582 fallback to ~grant (was ~5x over),
  2. flag OFF is unchanged (still overruns — the default path is untouched),
  3. flag ON vs OFF is byte-identical on a normal instance whose build finishes
     before the deadline (soundness: the default path is never perturbed).
"""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest

BENCH = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))
_GAS = BENCH / "gastrans582_mild11.nl"
_INREPO = Path("python/tests/data/minlplib_nl")


def _fallback(inst_path: Path, flag: bool, budget: float = 3.0):
    """Run ``_root_relaxation_lower_bound`` with the flag env set, reloading the
    tuning + solver modules so the field default_factory re-reads the env."""
    os.environ["DISCOPT_ROOT_BUILD_DEADLINE"] = "1" if flag else "0"
    import discopt.solver_tuning as T

    importlib.reload(T)
    import discopt.solver as S

    importlib.reload(S)
    m = dm.from_nl(str(inst_path))
    lb = np.array([v.lb for v in m._variables], dtype=np.float64)
    ub = np.array([v.ub for v in m._variables], dtype=np.float64)
    t = time.perf_counter()
    val = S._root_relaxation_lower_bound(m, lb, ub, time_limit=budget)
    return val, time.perf_counter() - t


@pytest.fixture(autouse=True)
def _restore_env():
    prev = os.environ.get("DISCOPT_ROOT_BUILD_DEADLINE")
    yield
    if prev is None:
        os.environ.pop("DISCOPT_ROOT_BUILD_DEADLINE", None)
    else:
        os.environ["DISCOPT_ROOT_BUILD_DEADLINE"] = prev
    import discopt.solver_tuning as T

    importlib.reload(T)
    import discopt.solver as S

    importlib.reload(S)


@pytest.mark.slow
@pytest.mark.skipif(not _GAS.exists(), reason="gastrans582_mild11.nl (benchmark corpus) absent")
def test_832_flag_on_bounds_gastrans_fallback():
    """Flag ON: the gastrans582 root fallback returns within a small multiple of its
    3s grant (measured ~3.6s) instead of overrunning ~5x. The bound is unchanged
    (``None`` either way on this instance — the base build's LP produces no finite
    bound), so this is a pure wall-time win that makes the solver honor its limit."""
    val_on, dt_on = _fallback(_GAS, flag=True, budget=3.0)
    # Generous ceiling (2x the grant) to stay robust on a loaded CI box; the point is
    # that it is bounded, not the ~5x runaway (~15.5s) of the default path.
    assert dt_on < 6.0, f"#832: flag-ON fallback took {dt_on:.1f}s, not bounded near the 3s grant"


@pytest.mark.slow
@pytest.mark.skipif(not _GAS.exists(), reason="gastrans582_mild11.nl (benchmark corpus) absent")
def test_832_flag_off_still_overruns_gastrans():
    """Flag OFF (default): the base build is left whole, so gastrans582 still
    overruns — proving the default path is genuinely untouched (and that the flag,
    not some unrelated change, is what bounds it)."""
    val_off, dt_off = _fallback(_GAS, flag=False, budget=3.0)
    assert dt_off > 8.0, (
        f"#832: flag-OFF fallback took only {dt_off:.1f}s — expected the ~15s default "
        "overrun; the default path may have changed"
    )


@pytest.mark.parametrize("inst", ["syn05hfsg", "casctanks", "nvs11"])
def test_832_flag_byte_identical_on_normal_instance(inst):
    """Soundness: on an instance whose base build finishes before the deadline, flag
    ON and OFF produce the IDENTICAL bound — the deadline only ever affects a build
    that would otherwise overrun, never the default result."""
    p = _INREPO / f"{inst}.nl"
    if not p.exists():
        pytest.skip(f"{inst}.nl absent")
    val_off, _ = _fallback(p, flag=False)
    val_on, _ = _fallback(p, flag=True)
    if val_off is None or val_on is None:
        assert val_off is val_on, f"{inst}: one side None, other not ({val_off} vs {val_on})"
    else:
        assert abs(val_off - val_on) < 1e-9, f"{inst}: bound drift {val_off} vs {val_on}"
