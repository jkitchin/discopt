"""``DISCOPT_NATIVE_KERNEL_VECTOR_VARS``: the native spatial kernel on array variables.

``spatial_producer.build_spatial_kernel_spec`` declined any model with an array
variable ("vector variables shift the layout and are a follow-up"), so every model
written with ``shape=`` ran on the Python spatial tree, 3-7x slower than its scalar
twin on the native kernel. The kernel and its consumer already work on flat columns;
only the producer assumed one column per variable object.

The soundness argument is identity, and these tests pin it: an array model and its
scalar twin -- the same model written with scalar variables -- produce byte-identical
kernel specs, so the array path IS the scalar path, and a certified solve explores
the same tree to the same bound.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import spatial_producer as sp

FLAG = "DISCOPT_NATIVE_KERNEL_VECTOR_VARS"


def _vars(m, arr, name, n, lb, ub, integer=False):
    make = m.integer if integer else m.continuous
    if arr:
        X = make(name, shape=(n,), lb=lb, ub=ub)
        return [X[i] for i in range(n)]
    lo = np.broadcast_to(np.asarray(lb, dtype=float), (n,))
    hi = np.broadcast_to(np.asarray(ub, dtype=float), (n,))
    return [make(f"{name}{i}", lb=float(lo[i]), ub=float(hi[i])) for i in range(n)]


def _qp(arr, n=8, seed=0, integer=True):
    """Indefinite QP; per-element bounds differ so a layout slip cannot hide."""
    rng = np.random.default_rng(seed)
    m = dm.Model("qp")
    lo = np.arange(n, dtype=float) % 3
    hi = lo + 2 + (np.arange(n) % 4)
    x = _vars(m, arr, "x", n, lo, hi, integer=integer)
    m.minimize(
        sum(float(rng.normal()) * x[i] * x[j] for i in range(n) for j in range(i + 1, n)) + sum(x)
    )
    m.subject_to(sum(x) >= n)
    return m


def _two_d(arr, k=3, seed=4):
    rng = np.random.default_rng(seed)
    m = dm.Model("twod")
    if arr:
        W = m.integer("w", shape=(k, k), lb=0, ub=4)
        w = [[W[i, j] for j in range(k)] for i in range(k)]
    else:
        w = [[m.integer(f"w{i}_{j}", lb=0, ub=4) for j in range(k)] for i in range(k)]
    for i in range(k):
        m.subject_to(
            sum(w[i][j] * w[j][i] for j in range(k) if j != i) >= float(rng.integers(1, 5))
        )
    m.minimize(sum(float(rng.uniform(0.5, 2)) * w[i][j] for i in range(k) for j in range(k)))
    return m


def _scalar_then_array(arr, n=6, seed=5):
    """Scalars before and after the array: every column offset must shift."""
    rng = np.random.default_rng(seed)
    m = dm.Model("sa")
    z = m.continuous("z", lb=0.0, ub=3.0)
    x = _vars(m, arr, "x", n, 1, 6, integer=True)
    t = m.continuous("t", lb=-5.0, ub=5.0)
    m.subject_to(z * x[0] + t * x[1] >= 2.0)
    m.subject_to(sum(x) <= 3 * n)
    m.minimize(sum(float(rng.normal()) * x[i] * x[(i + 1) % n] for i in range(n)) + z - t)
    return m


TWINS = {"qp": _qp, "two_d": _two_d, "scalar_then_array": _scalar_then_array}


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(TWINS))
def test_array_spec_is_byte_identical_to_the_scalar_twin(name, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    build = TWINS[name]
    a = sp.build_spatial_kernel_spec(build(True))
    s = sp.build_spatial_kernel_spec(build(False))
    # §6: the identity must not pass by comparing two declines.
    assert s is not None, "the scalar twin left the kernel's scope; pick another model"
    assert a is not None, "the array model was declined under the flag"
    assert set(a) == set(s)
    differ = [k for k in s if not np.array_equal(np.asarray(a[k]), np.asarray(s[k]))]
    assert not differ, f"array spec differs from its scalar twin in {differ}"


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(TWINS))
def test_flag_off_keeps_declining_array_models(name, monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    assert sp.build_spatial_kernel_spec(TWINS[name](True)) is None
    # ...and leaves scalar models exactly as they were.
    assert sp.build_spatial_kernel_spec(TWINS[name](False)) is not None


@pytest.mark.smoke
def test_default_is_on_with_an_opt_out(monkeypatch):
    """Graduated on its §5 panel (``scripts/panel_native_kernel_vector_vars.py``);
    ``=0`` keeps the pre-flag decline reachable for A/B."""
    monkeypatch.delenv(FLAG, raising=False)
    assert sp._vector_vars_enabled() is True
    monkeypatch.setenv(FLAG, "0")
    assert sp._vector_vars_enabled() is False


@pytest.mark.smoke
def test_array_solve_walks_the_scalar_twins_tree(monkeypatch):
    """Same spec, same kernel: the certified objective, the dual bound and the node
    count must all match the scalar twin's exactly. A continuous QP, so neither twin
    takes the integer-bilinear lift and both reach the native kernel."""
    monkeypatch.setenv(FLAG, "1")
    a = _qp(True, n=8, seed=1, integer=False).solve(time_limit=60)
    s = _qp(False, n=8, seed=1, integer=False).solve(time_limit=60)
    assert (s.algorithm_route or "").startswith("native-spatial"), s.algorithm_route
    assert (a.algorithm_route or "").startswith("native-spatial"), a.algorithm_route
    assert s.status == "optimal" and s.gap_certified
    assert a.status == "optimal" and a.gap_certified
    assert a.objective == pytest.approx(s.objective, rel=1e-12, abs=1e-12)
    assert a.bound == s.bound
    assert a.node_count == s.node_count
    # The answer comes back in the array's own shape.
    assert np.asarray(a.x["x"]).shape == (8,)


def _nonconvex(arr):
    m = dm.Model("nc")
    x = _vars(m, arr, "x", 2, 0, 4)
    m.minimize(-x[0] * x[1] + x[0])
    m.subject_to(x[0] + x[1] <= 5)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("arr", [False, True], ids=["scalar", "array"])
def test_a_cut_callback_is_never_swallowed_by_the_kernel(arr, monkeypatch):
    """The kernel has no cut hook. Before this, a ``cut_callback`` on a SCALAR model
    on the spatial path reached it and was called 0 times, silently (the #1535
    defect on this route); its test only passed because its model used an array
    variable, which never reached the kernel. The kernel must decline instead."""
    monkeypatch.setenv(FLAG, "1")
    calls = []

    def cut(ctx, model):
        calls.append(ctx.node_id)
        return []

    with pytest.warns(UserWarning, match="cut_callback"):
        r = _nonconvex(arr).solve(cut_callback=cut, time_limit=30)
    assert calls, "the cut_callback was never called"
    assert not (r.algorithm_route or "").startswith("native-spatial"), r.algorithm_route
    assert r.status == "optimal" and r.gap_certified
