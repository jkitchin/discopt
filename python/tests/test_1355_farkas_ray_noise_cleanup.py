"""Issue #1355: a rounding-noise entry in a Farkas ray blocked fathoming of
infeasible OA-master nodes.

On OA masters of the cardinality Markowitz class (the #1352 class), node LPs
ended ``Numerical``: the phase-1 residual was above tolerance and the Farkas ray
did not verify. HiGHS proves every one of those LPs infeasible. The ray was
genuine except for ~1e-17 entries on rows containing the OA epigraph column
``eta`` (``u = inf``). Those entries give ``eta`` an ``a^T y`` interval strictly
above zero, which selects its open side, and the rigorous verifier rejects the
ray as OPEN. Those nodes were never fathomed, so the master hit its node limit
and OA ended ``feasible`` at the time limit.

``DISCOPT_FARKAS_RAY_CLEANUP=1`` re-verifies the ray with its noise entries
(``|y_i| <= 1e-12 * ||y||_inf``) zeroed. Soundness rests on the unchanged
rigorous verifier; the arithmetic unit test is
``farkas_ray_noise_cleanup_rescues_open_column_rejection_1355`` in
``crates/discopt-core/src/lp/simplex/primal.rs``.

The flag is read once per process, so each solve runs in a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import json
    import numpy as np
    import discopt.modeling as dm
    from discopt.solvers.oa import solve_oa

    n, K = 12, 3
    rng = np.random.default_rng(2)
    F = rng.normal(size=(n, 3)) * 0.1
    Sigma = F @ F.T + np.diag(rng.uniform(0.001, 0.02, n))
    mu = rng.uniform(0.02, 0.15, n)
    rmin, ub = float(np.quantile(mu, 0.6)), 0.5
    m = dm.Model("port-12-3-2")
    z = m.binary("z", shape=(n,))
    w = m.continuous("w", shape=(n,), lb=0.0, ub=ub)
    m.minimize(
        dm.sum(
            lambda i: dm.sum(lambda j: Sigma[i, j] * w[i] * w[j], over=range(n)),
            over=range(n),
        )
    )
    m.subject_to(dm.sum(lambda i: w[i], over=range(n)) == 1.0)
    m.subject_to(dm.sum(lambda i: mu[i] * w[i], over=range(n)) >= rmin)
    m.subject_to(dm.sum(z) <= K)
    for i in range(n):
        m.subject_to(w[i] >= 0.02 * z[i])
        m.subject_to(w[i] <= ub * z[i])
    r = solve_oa(m, time_limit=20)
    print(json.dumps(dict(status=str(r.status), objective=r.objective, bound=r.bound,
                          mip_count=r.mip_count)))
    """
)

# The same model solved by OA with HiGHS as the master solver (unaffected by the
# in-house simplex), and matching the in-house master once it is fathomed.
REFERENCE_OPT = 0.0031843318237648


def _run(flag: str | None) -> dict:
    """Run the captured model with the cleanup flag set to ``flag``.

    ``flag=None`` leaves ``DISCOPT_FARKAS_RAY_CLEANUP`` *unset*, which is what
    exercises the shipped default rather than an explicit opt-in — the only way
    to test a graduation (#1360). The variable is popped rather than skipped:
    the test process may itself have been launched with it set.
    """
    env = dict(os.environ)
    if flag is None:
        env.pop("DISCOPT_FARKAS_RAY_CLEANUP", None)
    else:
        env["DISCOPT_FARKAS_RAY_CLEANUP"] = flag
    env["DISCOPT_OA_CONVEXITY_CERTIFICATE"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=240
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def _assert_sound(r: dict) -> None:
    """Certificate soundness, required of *every* arm: a reported bound never
    crosses the reference optimum, and an incumbent matches it."""
    if r["bound"] is not None:
        assert r["bound"] <= REFERENCE_OPT + 1e-9, r
    if r["objective"] is not None:
        assert abs(r["objective"] - REFERENCE_OPT) <= max(1e-6, 1e-4 * REFERENCE_OPT), r


def test_cleanup_lets_oa_master_fathom_and_certify():
    """Before the fix: ``feasible`` at the time limit (60 MILPs by 30 s). After:
    ``optimal`` in 32 MILPs, ~1.2 s."""
    r = _run("1")
    assert r["status"] == "optimal", r
    assert r["mip_count"] < 45, r
    assert r["bound"] <= r["objective"]
    _assert_sound(r)


def test_cleanup_is_the_default_since_1360():
    """Graduation (#1360): with the variable *unset*, the shipped default must
    behave as the opt-in arm — ``optimal``, fathomed well inside the node
    budget, and sound. A regression that flipped the default back to OFF would
    leave this run ``feasible`` at the time limit."""
    r = _run(None)
    assert r["status"] == "optimal", r
    assert r["mip_count"] < 45, r
    assert r["bound"] <= r["objective"]
    _assert_sound(r)


def test_zero_still_opts_out_and_stays_sound():
    """The legacy path stays reachable with ``=0`` (CLAUDE.md §5 keeps the
    opt-out intact). It is *slower* — that is the bug #1355 describes — so this
    asserts only that it still runs and that whatever it reports is sound, not
    that it certifies."""
    r = _run("0")
    assert r["status"] in {"optimal", "feasible"}, r
    _assert_sound(r)
