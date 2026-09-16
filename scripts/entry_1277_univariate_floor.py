"""#1277 entry experiment: what does a MISSING univariate envelope cost?

Background, and a retraction. #1277 asked for a declared-simplex structure hint,
because a CALPHAD compound-energy model's root bound sat 28.8% from the optimum
while the same algebra, hand-lifted into ``w_ij == y_i*z_j`` variables, sat 4.7%
away. The hypothesis was that declaring the simplex would let the engine add the
RLT marginals ``sum_i y_i z_j = z_j``. The control below the original measurement
FALSIFIED it: lift-only reached -10802.39 and lift+marginals -10898.28 against a
truth of -10315.85, so the marginals are neutral-to-harmful and the whole gain
came from the lifting. The lifting helped because it made the objective convex,
which let ``_try_convex_lift`` fire and put outer-approximation tangents on it.

Chasing THAT led to the real defect, which has nothing to do with simplexes:

    ``entropy`` (``dm.xlogx``), ``softplus``, ``sigmoid`` and ``tan`` are
    first-class discopt intrinsics that reach the relaxation engine as ``call``
    nodes with no entry in ``uniform_relax._UNIVARIATE_FN``, so each is relaxed
    by the interval floor -- a box, with no secant and no tangent.

Why it went unnoticed: on a BARE univariate objective the floor is exact (the aux
*is* the objective, so its enclosure is the answer -- 0.00% root gap, printed
below as the "bare" panel). The cost appears only when the atom sits beside a term
the composite-convex lift cannot certify, because the floor decouples ``w`` from
``t`` and the LP may then take every such term's independent minimum at once.

Both arms run here: "floor" pops the four entries back out of the table, so the
comparison is against the code as it was, in one process, interleaved.
"""

from __future__ import annotations

import discopt._relax.uniform_relax as ur
import discopt.modeling as dm
import numpy as np

_NEW = ("entropy", "softplus", "sigmoid", "tan")
R_T = 8.314 * 1000.0


def _bare_cases():
    """Univariate objectives: the floor is already exact, which is the reason the
    gap below was never seen. Kept as the control that explains the blind spot."""
    return [
        ("xlogx", dm.xlogx, lambda t: t * np.log(t), 1e-3, 0.9),
        ("softplus", dm.softplus, lambda t: np.logaddexp(0.0, t), -3.0, 3.0),
        ("sigmoid", dm.sigmoid, lambda t: 1.0 / (1.0 + np.exp(-t)), 0.0, 4.0),
        ("tan", dm.tan, np.tan, 0.05, 1.3),
    ]


def _coupled_cases():
    """``min f(x) + f(y) + c*x*y  s.t.  x + y == s``. The bilinear term blocks the
    composite-convex lift, so each atom stands on its own envelope."""
    return [
        ("entropy", dm.xlogx, lambda t: t * np.log(t), 1e-6, 1.0, 1.0, 40.0),
        ("softplus", dm.softplus, lambda t: np.logaddexp(0.0, t), -3.0, 3.0, 1.0, 9.0),
        ("sigmoid", dm.sigmoid, lambda t: 1.0 / (1.0 + np.exp(-t)), 0.1, 4.0, 2.0, 5.0),
        ("tan", dm.tan, np.tan, 0.05, 1.3, 1.0, 3.0),
    ]


def _cef():
    """The #1249 CALPHAD compound-energy model the issue was opened about: ideal
    entropy of mixing on two sublattices, coupled by the cross-sublattice
    end-member energies. Truth by dense sampling of the 2-D reduced problem."""
    g0 = np.random.default_rng(0).normal(0.0, 8000.0, size=(2, 2))
    m = dm.Model("cef")
    y = [m.continuous(f"y{i}", lb=1e-6, ub=1.0) for i in range(2)]
    z = [m.continuous(f"z{j}", lb=1e-6, ub=1.0) for j in range(2)]
    m.subject_to(y[0] + y[1] == 1.0)
    m.subject_to(z[0] + z[1] == 1.0)
    g = R_T * (dm.xlogx(y[0]) + dm.xlogx(y[1]) + dm.xlogx(z[0]) + dm.xlogx(z[1]))
    for i in range(2):
        for j in range(2):
            g = g + float(g0[i, j]) * y[i] * z[j]
    m.minimize(g)

    a = np.linspace(1e-6, 1 - 1e-6, 2000)
    ga, gb = np.meshgrid(a, a, indexing="ij")
    y0, y1, z0, z1 = ga, 1 - ga, gb, 1 - gb
    val = g0[0, 0] * y0 * z0 + g0[0, 1] * y0 * z1 + g0[1, 0] * y1 * z0 + g0[1, 1] * y1 * z1
    val = val + R_T * (y0 * np.log(y0) + y1 * np.log(y1) + z0 * np.log(z0) + z1 * np.log(z1))
    return m, float(np.min(val))


def _bare_model(fn, lo, hi):
    m = dm.Model("bare")
    x = m.continuous("x", lb=lo, ub=hi)
    m.minimize(fn(x))
    return m


def _coupled_model(fn, lo, hi, s, c):
    m = dm.Model("coupled")
    x = m.continuous("x", lb=lo, ub=hi)
    y = m.continuous("y", lb=lo, ub=hi)
    m.subject_to(x + y == s)
    m.minimize(fn(x) + fn(y) + c * x * y)
    return m


def _coupled_truth(np_f, lo, hi, s, c, n=2_000_001):
    a = np.linspace(max(lo, s - hi), min(hi, s - lo), n)
    b = s - a
    with np.errstate(all="ignore"):
        v = np_f(a) + np_f(b) + c * a * b
    v = v[np.isfinite(v)]
    assert v.size, "the truth grid enclosed no finite point"
    return float(np.min(v))


def _measure(label, model, truth, expect_atoms):
    rel = ur.build_uniform_relaxation(model)
    cov = [t for k, t in rel.coverage.values() if k == "univariate_call"]
    assert len(cov) == expect_atoms, f"{label}: saw {len(cov)} atoms, wanted {expect_atoms}"
    r = model.solve(time_limit=120, max_nodes=1)
    rb = r.root_bound if r.root_bound is not None else r.bound
    assert rb is not None, f"{label}: no root bound to measure"
    scale = max(1.0, abs(truth))
    assert rb <= truth + 1e-6 * scale, f"{label}: FALSE BOUND {rb} > {truth}"
    return rb, (truth - rb) / scale, ("tight" if cov and all(cov) else "floor")


def main() -> int:
    saved = {k: ur._UNIVARIATE_FN[k] for k in _NEW if k in ur._UNIVARIATE_FN}
    assert len(saved) == len(_NEW), (
        "this script measures the #1277 entries against their absence; the table "
        f"is missing {sorted(set(_NEW) - set(saved))}"
    )

    arms = 0
    gaps: dict[tuple[str, str], float] = {}
    panels = [
        (
            "bare",
            [
                (n, _bare_model(f, lo, hi), float(np.min(g(np.linspace(lo, hi, 200_001)))), 1)
                for n, f, g, lo, hi in _bare_cases()
            ],
        ),
        (
            "coupled",
            [
                (n, _coupled_model(f, lo, hi, s, c), _coupled_truth(g, lo, hi, s, c), 2)
                for n, f, g, lo, hi, s, c in _coupled_cases()
            ],
        ),
        ("cef", [("cef", *_cef(), 4)]),
    ]

    for arm in ("floor", "envelope"):
        if arm == "floor":
            for k in _NEW:
                ur._UNIVARIATE_FN.pop(k, None)
            assert not any(k in ur._UNIVARIATE_FN for k in _NEW)
        else:
            ur._UNIVARIATE_FN.update(saved)
            assert all(k in ur._UNIVARIATE_FN for k in _NEW)
        for panel, cases in panels:
            for label, model, truth, natoms in cases:
                key = f"{panel}/{label}"
                rb, gap, cov = _measure(f"{key}:{arm}", model, truth, natoms)
                print(
                    f"{key:>18} {arm:>9} | bound {rb:14.5f}  truth {truth:14.5f}  "
                    f"gap {gap:8.2%}  {cov}",
                    flush=True,
                )
                gaps[(key, arm)] = gap
                arms += 1

    print()
    gains = 0
    for key in sorted({k for k, _ in gaps}):
        a, b = gaps[(key, "floor")], gaps[(key, "envelope")]
        verdict = "GAIN" if b < a - 1e-9 else ("LOSS" if b > a + 1e-9 else "unchanged")
        gains += verdict == "GAIN"
        print(f"{key:>18}: {a:8.2%} -> {b:8.2%}  {verdict}")
        assert verdict != "LOSS", f"{key}: the envelope LOOSENED the bound"

    print()
    print(f"panels with a strictly tighter root bound: {gains}/{len(gaps) // 2}")
    print("PROCEED" if gains else "KILL")
    print(f"EXECUTED_ARMS={arms}")
    return 0 if arms and gains else 1


if __name__ == "__main__":
    raise SystemExit(main())
