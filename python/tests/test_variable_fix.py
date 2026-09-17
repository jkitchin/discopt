"""Scoped variable fixing: ``Variable.fix``/``unfix``, ``fixed``, ``saved_bounds``.

``lb == ub`` is the only fixing route in discopt, and the Rust lowering re-reads
both bounds from the live Python object on every solve (C-41). That combination
makes a *leaked* fix uniquely dangerous: it does not raise, it silently
redefines the problem and still returns ``status="optimal"``.
``test_leaked_fix_silently_changes_the_answer`` pins that failure mode, and the
rest of this file pins the API that exists to prevent it.
"""

import contextlib

import discopt.modeling as dm
import numpy as np
import pytest


def _tiny_lp():
    """max 3x + 2y  s.t. x + y <= 6, 0 <= x,y <= 10  ->  obj 18 at x=6, y=0."""
    m = dm.Model("tiny")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y <= 6.0)
    m.maximize(3 * x + 2 * y)
    return m, x, y


# ── the primitive ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_fix_pins_bounds_and_unfix_restores_them_exactly():
    m = dm.Model("m")
    v = m.continuous("v", shape=(3,), lb=[0.0, 1.0, 2.0], ub=[4.0, 5.0, 6.0])
    lb0, ub0 = np.array(v.lb), np.array(v.ub)

    v.fix([1.0, 2.0, 3.0])
    assert v.is_fixed
    assert v.fix_depth == 1
    np.testing.assert_array_equal(v.lb, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(v.ub, [1.0, 2.0, 3.0])

    v.unfix()
    assert not v.is_fixed
    assert v.fix_depth == 0
    np.testing.assert_array_equal(v.lb, lb0)
    np.testing.assert_array_equal(v.ub, ub0)


@pytest.mark.unit
def test_is_fixed_reports_state_not_provenance():
    """A variable declared with lb == ub is fixed even though fix() never ran."""
    m = dm.Model("m")
    c = m.continuous("c", lb=3.0, ub=3.0)
    assert c.is_fixed
    assert c.fix_depth == 0


@pytest.mark.unit
def test_nested_fixes_unwind_in_lifo_order():
    """A single saved slot would restore the wrong box here."""
    m = dm.Model("m")
    v = m.continuous("v", lb=0.0, ub=10.0)

    v.fix(4.0)
    v.fix(5.0)
    v.fix(6.0)
    assert v.fix_depth == 3
    assert float(v.lb) == 6.0

    v.unfix()
    assert float(v.lb) == 5.0
    v.unfix()
    assert float(v.lb) == 4.0
    v.unfix()
    assert (float(v.lb), float(v.ub)) == (0.0, 10.0)
    assert v.fix_depth == 0


@pytest.mark.unit
def test_unfix_without_a_fix_refuses_loudly():
    m = dm.Model("m")
    v = m.continuous("v", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="no fix"):
        v.unfix()


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-1.0, 11.0])
def test_fix_outside_the_box_refuses(bad):
    """Pinning outside the domain yields a silently infeasible model; refuse."""
    m = dm.Model("m")
    v = m.continuous("v", lb=0.0, ub=10.0)
    with pytest.raises(ValueError, match="outside its declared bounds"):
        v.fix(bad)
    assert v.fix_depth == 0  # a refused fix leaves no frame
    assert (float(v.lb), float(v.ub)) == (0.0, 10.0)


@pytest.mark.unit
def test_nested_refix_is_checked_against_the_declared_domain():
    """An inner fix may override an outer one, but not escape the declared box.

    Checking against the box a fix *replaces* would refuse every nested
    re-fix -- after ``fix(4.0)`` the box is ``[4, 4]`` and any other value is
    "outside" it -- which would defeat the stack entirely.
    """
    m = dm.Model("m")
    v = m.continuous("v", lb=0.0, ub=10.0)
    v.fix(4.0)
    v.fix(9.0)  # inside [0, 10]: allowed
    assert float(v.lb) == 9.0
    with pytest.raises(ValueError, match="outside its declared bounds"):
        v.fix(11.0)  # outside [0, 10]: refused
    assert v.fix_depth == 2  # the refusal left no frame
    v.unfix()
    assert float(v.lb) == 4.0
    v.unfix()
    assert (float(v.lb), float(v.ub)) == (0.0, 10.0)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fix_at_a_non_finite_value_refuses(bad):
    m = dm.Model("m")
    v = m.continuous("v", lb=0.0, ub=10.0)
    with pytest.raises(ValueError, match="must be finite"):
        v.fix(bad)
    assert v.fix_depth == 0


@pytest.mark.unit
def test_partial_fix_pins_only_the_masked_elements():
    """The multi-experiment case: one design block free, the data blocks pinned."""
    m = dm.Model("m")
    F = m.continuous("F", shape=(3,), lb=0.0, ub=10.0)

    F.fix([9.0, 2.0, 3.0], where=[False, True, True])
    np.testing.assert_array_equal(F.lb, [0.0, 2.0, 3.0])
    np.testing.assert_array_equal(F.ub, [10.0, 2.0, 3.0])
    assert not F.is_fixed  # element 0 is still free

    F.unfix()
    np.testing.assert_array_equal(F.lb, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(F.ub, [10.0, 10.0, 10.0])


# ── the context managers ─────────────────────────────────────────────────


@pytest.mark.unit
def test_variable_fixed_restores_on_exception():
    m = dm.Model("m")
    v = m.continuous("v", lb=0.0, ub=10.0)
    with pytest.raises(RuntimeError, match="boom"):
        with v.fixed(1.0):
            assert v.is_fixed
            raise RuntimeError("boom")
    assert not v.is_fixed
    assert v.fix_depth == 0
    assert (float(v.lb), float(v.ub)) == (0.0, 10.0)


@pytest.mark.unit
def test_variable_fixed_unwinds_a_fix_leaked_inside_it():
    m = dm.Model("m")
    v = m.continuous("v", lb=0.0, ub=10.0)
    with v.fixed(2.0):
        v.fix(3.0)  # leaked: never unfixed
    assert v.fix_depth == 0
    assert (float(v.lb), float(v.ub)) == (0.0, 10.0)


@pytest.mark.unit
def test_model_fixed_accepts_keywords_objects_and_names():
    m = dm.Model("m")
    a = m.continuous("a", lb=0.0, ub=10.0)
    b = m.continuous("b", lb=0.0, ub=10.0)
    c = m.continuous("c", lb=0.0, ub=10.0)

    with m.fixed({a: 1.0}, {"b": 2.0}, c=3.0):
        assert (float(a.lb), float(b.lb), float(c.lb)) == (1.0, 2.0, 3.0)
    assert (a.fix_depth, b.fix_depth, c.fix_depth) == (0, 0, 0)
    assert (float(a.ub), float(b.ub), float(c.ub)) == (10.0, 10.0, 10.0)


@pytest.mark.unit
def test_model_fixed_restores_on_exception():
    m = dm.Model("m")
    a = m.continuous("a", lb=0.0, ub=10.0)
    b = m.continuous("b", lb=0.0, ub=10.0)
    with pytest.raises(RuntimeError):
        with m.fixed(a=1.0, b=2.0):
            raise RuntimeError("boom")
    assert (a.fix_depth, b.fix_depth) == (0, 0)
    assert (float(a.ub), float(b.ub)) == (10.0, 10.0)


@pytest.mark.unit
def test_model_fixed_refuses_a_foreign_variable():
    m1, m2 = dm.Model("m1"), dm.Model("m2")
    m1.continuous("x", lb=0.0, ub=1.0)
    foreign = m2.continuous("y", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="belongs to model"):
        with m1.fixed({foreign: 0.5}):
            pass


@pytest.mark.unit
def test_model_fixed_refuses_an_empty_scope():
    m = dm.Model("m")
    m.continuous("x", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="at least one variable"):
        with m.fixed():
            pass


@pytest.mark.unit
def test_model_fixed_refuses_an_unknown_name():
    m = dm.Model("m")
    m.continuous("x", lb=0.0, ub=1.0)
    with pytest.raises(KeyError):
        with m.fixed(nope=1.0):
            pass


@pytest.mark.unit
def test_partial_fixes_compose_through_exit_stack():
    m = dm.Model("m")
    F = m.continuous("F", shape=(2,), lb=0.0, ub=10.0)
    G = m.continuous("G", shape=(2,), lb=0.0, ub=10.0)
    with contextlib.ExitStack() as stack:
        stack.enter_context(F.fixed([1.0, 1.0], where=[True, False]))
        stack.enter_context(G.fixed([2.0, 2.0], where=[False, True]))
        np.testing.assert_array_equal(F.ub, [1.0, 10.0])
        np.testing.assert_array_equal(G.ub, [10.0, 2.0])
    np.testing.assert_array_equal(F.ub, [10.0, 10.0])
    np.testing.assert_array_equal(G.ub, [10.0, 10.0])


# ── saved_bounds: the low-level primitive ────────────────────────────────


@pytest.mark.unit
def test_saved_bounds_restores_arbitrary_boxes_not_just_fixes():
    """The bound-override passes set lb != ub; saved_bounds must cover that."""
    m = dm.Model("m")
    v = m.continuous("v", shape=(2,), lb=0.0, ub=10.0)
    with m.saved_bounds():
        v.lb = np.array([2.0, 3.0])
        v.ub = np.array([4.0, 5.0])
    np.testing.assert_array_equal(v.lb, [0.0, 0.0])
    np.testing.assert_array_equal(v.ub, [10.0, 10.0])


@pytest.mark.unit
def test_saved_bounds_restores_on_exception_and_unwinds_leaked_fixes():
    m = dm.Model("m")
    v = m.continuous("v", lb=0.0, ub=10.0)
    with pytest.raises(RuntimeError):
        with m.saved_bounds():
            v.fix(5.0)
            raise RuntimeError("boom")
    assert v.fix_depth == 0
    assert (float(v.lb), float(v.ub)) == (0.0, 10.0)


@pytest.mark.unit
@pytest.mark.parametrize("copy", [False, True])
def test_saved_bounds_both_snapshot_modes_restore(copy):
    m = dm.Model("m")
    v = m.continuous("v", shape=(2,), lb=[0.0, 1.0], ub=[8.0, 9.0])
    with m.saved_bounds(copy=copy):
        v.lb = np.array([3.0, 3.0])
        v.ub = np.array([4.0, 4.0])
    np.testing.assert_array_equal(v.lb, [0.0, 1.0])
    np.testing.assert_array_equal(v.ub, [8.0, 9.0])


# ── the failure mode this API exists to prevent ──────────────────────────


@pytest.mark.smoke
def test_leaked_fix_silently_changes_the_answer():
    """Regression pin: a leaked fix does NOT raise -- it returns a wrong optimum.

    This is why fixing is scoped. Bounds are re-read live at every lowering, so
    an ``lb == ub`` left behind by an exception redefines the problem and the
    solver reports ``optimal`` for it, with nothing to distinguish it from the
    answer the caller believes they asked for.
    """
    m, x, _y = _tiny_lp()
    good = float(m.solve(skip_convex_check=True).objective)
    assert good == pytest.approx(18.0, abs=1e-6)

    with contextlib.suppress(RuntimeError):  # the un-scoped idiom
        x.lb = x.ub = np.asarray(1.0)
        raise RuntimeError("a task blew up before restoring")

    leaked = m.solve(skip_convex_check=True)
    assert leaked.status in ("optimal", "feasible")  # no error surfaces
    assert float(leaked.objective) != pytest.approx(good, abs=1e-6)


@pytest.mark.smoke
def test_scoped_fix_keeps_the_next_solve_correct():
    """The same sequence through ``fixed()``: the leak cannot happen."""
    m, x, _y = _tiny_lp()
    good = float(m.solve(skip_convex_check=True).objective)

    with contextlib.suppress(RuntimeError):
        with x.fixed(1.0):
            raise RuntimeError("a task blew up before restoring")

    after = m.solve(skip_convex_check=True)
    assert float(after.objective) == pytest.approx(good, abs=1e-6)
    assert x.fix_depth == 0


@pytest.mark.smoke
def test_fit_then_design_reuses_one_model_across_tasks():
    """The workflow this API is for: one model, two objectives, two free sets.

    Both tasks must reproduce their own answer on every cycle -- that is what
    fails if any state leaks across the objective swap or the bound restore.
    """
    V, CA_IN = 2.0, 5.0

    def cstr(Ca, F, k, n):
        return F * (CA_IN - Ca) - V * k * Ca**n

    F_data = np.array([1.0, 2.0, 4.0])
    Ca_data = np.array([2.0430, 2.5290, 3.0672])  # generated at k=0.8, n=1.5
    nexp = len(F_data)

    m = dm.Model("unified")
    k = m.continuous("k", lb=1e-3, ub=10.0)
    n = m.continuous("n", lb=0.5, ub=3.0)
    Ca = m.continuous("Ca", shape=(nexp,), lb=1e-3, ub=CA_IN)
    F = m.continuous("F", shape=(nexp,), lb=0.1, ub=20.0)
    for i in range(nexp):
        m.subject_to(cstr(Ca[i], F[i], k, n) == 0.0, name=f"cstr_{i}")

    fits, designs = [], []
    for _cycle in range(3):
        # regression: k, n free; F held at the measured conditions
        m.minimize(dm.sum([(Ca[i] - Ca_data[i]) ** 2 for i in range(nexp)]))
        with m.fixed({F: F_data}):
            r = m.solve(skip_convex_check=True)
            assert r.status in ("optimal", "feasible")
            fits.append((float(r.value(k)), float(r.value(n))))
            np.testing.assert_allclose(r.value(F), F_data, atol=1e-9)

        # design: k, n held at the fit; block 0's flow rate free
        m.maximize(F[0] * (CA_IN - Ca[0]) - 0.05 * F[0] ** 2)
        with contextlib.ExitStack() as stack:
            stack.enter_context(m.fixed({k: fits[-1][0], n: fits[-1][1]}))
            stack.enter_context(F.fixed(np.r_[0.0, F_data[1:]], where=[False, True, True]))
            r2 = m.solve(skip_convex_check=True)
            assert r2.status in ("optimal", "feasible")
            designs.append(float(r2.value(F)[0]))

        # every scope closed -> the model is back to its declared state
        assert (k.fix_depth, n.fix_depth, F.fix_depth) == (0, 0, 0)

    assert len({round(f[0], 9) for f in fits}) == 1, f"fit drifted across cycles: {fits}"
    assert len({round(d, 9) for d in designs}) == 1, f"design drifted: {designs}"


# ── #1311: fix()/unfix() must not leave the box writable ─────────────────


@pytest.mark.smoke
def test_fix_unfix_cycle_leaves_bounds_read_only():
    """A single fix()/unfix() cycle must restore the SAME read-only invariant
    the original declared bounds carry. Before the fix, ``unfix()`` restored a
    plain writable copy, so an in-place bound write (``v.lb[...] = ...``)
    would silently succeed after even one fix/unfix cycle -- exactly the
    aliasing hole ``Model.saved_bounds(copy=False)`` documents itself as safe
    from (its docstring claims a naive in-place write always raises)."""
    m = dm.Model("m")
    x = m.continuous("x", lb=0.0, ub=10.0)
    assert not x.lb.flags.writeable
    x.fix(3.0)
    x.unfix()
    assert not x.lb.flags.writeable
    assert not x.ub.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        x.lb[()] = 99.0


@pytest.mark.smoke
def test_saved_bounds_restore_survives_a_prior_fix_unfix_cycle():
    """#1311: saved_bounds(copy=False)'s by-reference snapshot must still be
    protected by read-only arrays after the variable has been through a
    fix()/unfix() cycle -- reproducing the exact aliasing corruption a
    prior fix/unfix could otherwise reopen on the hot B&B path."""
    m = dm.Model("m")
    x = m.continuous("x", lb=0.0, ub=10.0)
    x.fix(3.0)
    x.unfix()
    with pytest.raises(ValueError, match="read-only"):
        with m.saved_bounds(copy=False):
            x.lb[()] = 7.5
    # The snapshot itself (and the live box) must be untouched by the refused write.
    assert (float(x.lb), float(x.ub)) == (0.0, 10.0)


# ── #1311: fixed() scopes closed out of LIFO order must not corrupt state ──


@pytest.mark.smoke
def test_variable_fixed_scopes_closed_out_of_order_raise_immediately():
    """Two fixed() scopes on the same variable, held open across a boundary
    (not nested via a single ``with`` block) and closed in the wrong order,
    must be refused AT THE POINT OF DIVERGENCE -- not silently unwind the
    still-open inner scope and let a solve run against the wrong box before
    anyone notices."""
    m = dm.Model("m")
    x = m.continuous("x", lb=0.0, ub=10.0)

    cm1 = x.fixed(1.0)
    cm1.__enter__()
    cm2 = x.fixed(2.0)
    cm2.__enter__()
    assert (float(x.lb), float(x.ub)) == (2.0, 2.0)

    with pytest.raises(RuntimeError, match="closed out of order"):
        cm1.__exit__(None, None, None)
    # The out-of-order exit must not have touched the box: cm2's fix survives.
    assert (float(x.lb), float(x.ub)) == (2.0, 2.0)
    assert x.fix_depth == 2

    # Closing in the correct (LIFO) order from here recovers cleanly.
    cm2.__exit__(None, None, None)
    assert (float(x.lb), float(x.ub)) == (1.0, 1.0)


@pytest.mark.smoke
def test_model_fixed_scopes_closed_out_of_order_raise_immediately():
    """Same hazard as the Variable.fixed() case, for Model.fixed()."""
    m = dm.Model("m")
    x = m.continuous("x", lb=0.0, ub=10.0)

    cm1 = m.fixed(x=1.0)
    cm1.__enter__()
    cm2 = m.fixed(x=2.0)
    cm2.__enter__()

    with pytest.raises(RuntimeError, match="closed out of order"):
        cm1.__exit__(None, None, None)
    assert (float(x.lb), float(x.ub)) == (2.0, 2.0)

    cm2.__exit__(None, None, None)
    assert (float(x.lb), float(x.ub)) == (1.0, 1.0)


@pytest.mark.smoke
def test_properly_nested_fixed_scopes_are_unaffected():
    """The ordinary, correctly-nested case (a single ``with`` block, or
    sequential non-overlapping scopes) must be completely unaffected by the
    #1311 out-of-order detection."""
    m = dm.Model("m")
    x = m.continuous("x", lb=0.0, ub=10.0)

    with x.fixed(1.0):
        with x.fixed(2.0):
            assert (float(x.lb), float(x.ub)) == (2.0, 2.0)
        assert (float(x.lb), float(x.ub)) == (1.0, 1.0)
    assert (float(x.lb), float(x.ub)) == (0.0, 10.0)
    assert x.fix_depth == 0

    with x.fixed(3.0):
        pass
    with x.fixed(4.0):
        pass
    assert x.fix_depth == 0
