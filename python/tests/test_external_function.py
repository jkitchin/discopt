"""``dm.external``: an external function whose derivatives the caller supplies.

The design rests on one falsifiable claim -- that nesting ``jax.custom_jvp``
(one rule for the value using the caller's Jacobian, one for the Jacobian using
the caller's Hessian) makes a non-traceable callable *twice* differentiable in
the forward-over-reverse pattern POUNCE requests. That was the entry experiment,
and :class:`TestNestedJvpDerivatives` is it, kept as a regression test: if JAX
ever stops supporting this, the feature is unsound and these fail first rather
than a solve quietly losing its second derivatives.

Every test that claims external code was *used* asserts a call counter. A probe
that silently calls nothing reports a pass, which is the failure mode CLAUDE.md
§6 exists for: before the counters were here, an earlier version of the
end-to-end test compared a solve against itself.
"""

from __future__ import annotations

from importlib import import_module

import discopt.modeling as dm
import numpy as np
import pytest


def _ext_module():
    """The ``external`` *submodule*, for the tests that reach past the public API.

    Not ``import discopt.modeling.external`` -- that binds the *function*, because
    ``discopt.modeling.__init__`` re-exports ``external`` into the package
    namespace and shadows the submodule of the same name. ``implicit`` has the
    same shape, so this is the house pattern rather than a bug here; the shadowing
    is invisible to users, who want ``dm.external`` to be the function.
    """
    return import_module("discopt.modeling.external")


# --------------------------------------------------------------------------- #
# The external "simulator": plain numpy, and deliberately NOT JAX-traceable.
# float() on an element forces concreteness, so a tracer cannot survive it --
# this is what makes these tests exercise the callback path rather than
# accidentally tracing through.
# --------------------------------------------------------------------------- #

CALLS: dict[str, int] = {"f": 0, "J": 0, "H": 0}


def reset_calls() -> None:
    for k in CALLS:
        CALLS[k] = 0


def ext_f(x):
    """f(x) = [x0^2 * x1 + exp(x1)] -- one residual, two inputs."""
    CALLS["f"] += 1
    a = np.asarray(x, dtype=float)
    return np.asarray([float(a[0]) ** 2 * float(a[1]) + np.exp(float(a[1]))])


def ext_J(x):
    CALLS["J"] += 1
    a = np.asarray(x, dtype=float)
    return np.asarray([[2.0 * a[0] * a[1], a[0] ** 2 + np.exp(a[1])]])


def ext_H(x):
    CALLS["H"] += 1
    a = np.asarray(x, dtype=float)
    return np.asarray([[[2.0 * a[1], 2.0 * a[0]], [2.0 * a[0], np.exp(a[1])]]])


def sym_body(x):
    """The symbolic twin of :func:`ext_f`, in discopt primitives."""
    return x[0] ** 2 * x[1] + dm.exp(x[1])


TARGET = 3.0


@pytest.fixture(autouse=True)
def _clean_counters():
    reset_calls()
    yield


# --------------------------------------------------------------------------- #
# The entry experiment, as a regression test.
# --------------------------------------------------------------------------- #


class TestNestedJvpDerivatives:
    """The nested-``custom_jvp`` composite is twice differentiable and correct."""

    @staticmethod
    def _scalar_fn():
        # Build the composite exactly as ``external()`` does -- same wrappers, same
        # nesting -- so these test the derivative rules the solver will be handed,
        # then scalarize it so jax.grad/jax.hessian apply.
        ext_mod = _ext_module()

        f_call = ext_mod._checked(ext_f, role="fn", label="sim", want=lambda xs: (1,))
        j_call = ext_mod._checked(ext_J, role="jac", label="sim", want=lambda xs: (1,) + xs)
        h_call = ext_mod._checked(ext_H, role="hess", label="sim", want=lambda xs: (1,) + xs + xs)
        composite = ext_mod._build_composite("sim", (1,), f_call, j_call, h_call)
        return lambda x: composite(x)[0]

    def test_value_matches_the_external_code(self):
        import jax.numpy as jnp

        x0 = jnp.array([1.3, 0.7])
        got = self._scalar_fn()(x0)
        assert CALLS["f"] > 0, "the external value function was never called"
        np.testing.assert_allclose(float(got), float(ext_f(np.asarray(x0))[0]), rtol=0, atol=1e-12)

    def test_gradient_is_the_callers_jacobian(self):
        import jax
        import jax.numpy as jnp

        x0 = jnp.array([1.3, 0.7])
        grad = jax.grad(self._scalar_fn())(x0)
        assert CALLS["J"] > 0, "the external Jacobian was never called"
        np.testing.assert_allclose(np.asarray(grad), ext_J(np.asarray(x0))[0], rtol=0, atol=1e-12)

    def test_reverse_mode_agrees_with_forward_mode(self):
        import jax
        import jax.numpy as jnp

        x0 = jnp.array([1.3, 0.7])
        f = self._scalar_fn()
        np.testing.assert_allclose(
            np.asarray(jax.jacrev(f)(x0)), np.asarray(jax.jacfwd(f)(x0)), rtol=0, atol=1e-12
        )

    def test_hessian_is_the_callers_hessian(self):
        """THE kill criterion: without the second ``custom_jvp`` level this raises."""
        import jax
        import jax.numpy as jnp

        x0 = jnp.array([1.3, 0.7])
        hess = jax.hessian(self._scalar_fn())(x0)
        assert CALLS["H"] > 0, "the external Hessian was never called"
        np.testing.assert_allclose(np.asarray(hess), ext_H(np.asarray(x0))[0], rtol=0, atol=1e-12)

    def test_hessian_survives_jit(self):
        import jax
        import jax.numpy as jnp

        x0 = jnp.array([1.3, 0.7])
        hess = jax.jit(jax.hessian(self._scalar_fn()))(x0)
        np.testing.assert_allclose(np.asarray(hess), ext_H(np.asarray(x0))[0], rtol=0, atol=1e-12)

    def test_hessian_vector_product_forward_over_reverse(self):
        """The exact pattern POUNCE requests, not just ``jax.hessian``."""
        import jax
        import jax.numpy as jnp

        x0 = jnp.array([1.3, 0.7])
        direction = jnp.array([1.0, 0.0])
        _, hvp = jax.jvp(jax.grad(self._scalar_fn()), (x0,), (direction,))
        want = ext_H(np.asarray(x0))[0] @ np.asarray(direction)
        np.testing.assert_allclose(np.asarray(hvp), want, rtol=0, atol=1e-12)

    def test_scalar_input_contracts_without_an_outer_product(self):
        """``tensordot`` with no axes is an outer product; a scalar input must not
        silently take that path."""
        import jax
        import jax.numpy as jnp

        ext_mod = _ext_module()

        cube = lambda x: np.asarray(float(x) ** 3)  # noqa: E731
        d1 = lambda x: np.asarray(3.0 * float(x) ** 2)  # noqa: E731
        d2 = lambda x: np.asarray(6.0 * float(x))  # noqa: E731
        composite = ext_mod._build_composite(
            "cube",
            (),
            ext_mod._checked(cube, role="fn", label="cube", want=lambda xs: ()),
            ext_mod._checked(d1, role="jac", label="cube", want=lambda xs: xs),
            ext_mod._checked(d2, role="hess", label="cube", want=lambda xs: xs + xs),
        )
        x0 = jnp.float64(2.0)
        np.testing.assert_allclose(float(jax.grad(composite)(x0)), 12.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(float(jax.hessian(composite)(x0)), 12.0, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# End to end through the solver.
# --------------------------------------------------------------------------- #


def _model(external: bool, *, hess=ext_H, integer: bool = False):
    m = dm.Model("ext" if external else "sym")
    if integer:
        x = m.integer("x", shape=(2,), lb=1, ub=3)
    else:
        x = m.continuous("x", shape=(2,), lb=0.2, ub=3.0)
    m.minimize(dm.sum(x))
    if external:
        block = dm.external(ext_f, jac=ext_J, hess=hess, shape=(1,), name="sim")
        body = block(x)[0]
    else:
        body = sym_body(x)
    m.subject_to(body == TARGET, name="block")
    return m, x


class TestSolveEndToEnd:
    def test_reaches_the_symbolic_twins_optimum(self):
        """An external block and its symbolic twin must land on the same point.

        The twin is solved FIRST and outside any patching, so this is a
        comparison against an independently obtained answer rather than against
        another run of the same path.
        """
        ms, xs = _model(external=False)
        ref = ms.solve()
        ref_x = np.asarray(ref.value(xs), dtype=float)
        assert ref.status == "optimal", f"the symbolic twin did not solve: {ref.status!r}"

        reset_calls()
        me, xe = _model(external=True)
        got = me.solve()
        got_x = np.asarray(got.value(xe), dtype=float)

        assert CALLS["f"] > 0 and CALLS["J"] > 0, "external value/Jacobian never called"
        assert CALLS["H"] > 0, "external Hessian never called -- the NLP path did not use it"
        np.testing.assert_allclose(got.objective, ref.objective, rtol=0, atol=1e-6)
        np.testing.assert_allclose(got_x, ref_x, rtol=0, atol=1e-5)
        # Feasible against the EXTERNAL function, not against the twin.
        assert abs(float(ext_f(got_x)[0]) - TARGET) < 1e-6

    def test_withholds_the_certificate(self):
        """Inherited ``CustomCall`` contract: a local optimum is not a dual bound."""
        m, _ = _model(external=True)
        res = m.solve()
        assert res.status == "feasible", f"expected 'feasible', got {res.status!r}"
        assert res.bound is None, f"an opaque block must not report a dual bound, got {res.bound!r}"
        assert res.gap is None
        assert getattr(res, "gap_certified", False) is False

    def test_vector_valued_block(self):
        """Two residuals from one external call -- a constraint body, not a term."""

        def f2(x):
            CALLS["f"] += 1
            a = np.asarray(x, dtype=float)
            return np.asarray([float(a[0]) + float(a[1]), float(a[0]) - float(a[1])])

        def j2(x):
            CALLS["J"] += 1
            return np.asarray([[1.0, 1.0], [1.0, -1.0]])

        def h2(x):
            CALLS["H"] += 1
            return np.zeros((2, 2, 2))

        block = dm.external(f2, jac=j2, hess=h2, shape=(2,), name="pair")
        m = dm.Model("vec")
        x = m.continuous("x", shape=(2,), lb=-5.0, ub=5.0)
        m.minimize(x[0] ** 2 + x[1] ** 2)
        out = block(x)
        m.subject_to(out[0] == 2.0, name="sum")
        m.subject_to(out[1] == 0.0, name="diff")
        res = m.solve()
        assert CALLS["f"] > 0 and CALLS["J"] > 0
        xv = np.asarray(res.value(x), dtype=float)
        np.testing.assert_allclose(xv, [1.0, 1.0], rtol=0, atol=1e-5)

    def test_integer_variables_are_refused(self):
        """Inherited sound-or-refuse: no valid node relaxation for an opaque block."""
        m, _ = _model(external=True, integer=True)
        with pytest.raises(ValueError, match="integer/binary"):
            m.solve()


class TestMissingHessian:
    def test_nlp_path_refuses_and_names_both_fixes(self):
        m, _ = _model(external=True, hess=None)
        with pytest.raises(ValueError) as excinfo:
            m.solve()
        msg = str(excinfo.value)
        assert "'sim'" in msg, f"the refusal must name the block: {msg}"
        assert "hess=" in msg, f"the refusal must name the hess= fix: {msg}"
        assert "direct" in msg, f"the refusal must name the solver='direct' fix: {msg}"

    def test_direct_still_solves_without_a_hessian(self):
        """The escape hatch the refusal points at has to actually work."""

        def f1(x):
            CALLS["f"] += 1
            a = np.asarray(x, dtype=float)
            return np.asarray((float(a[0]) - 1.5) ** 2 + (float(a[1]) - 0.5) ** 2)

        def j1(x):
            CALLS["J"] += 1
            a = np.asarray(x, dtype=float)
            return np.asarray([2.0 * (a[0] - 1.5), 2.0 * (a[1] - 0.5)])

        block = dm.external(f1, jac=j1, shape=(), name="bowl")
        m = dm.Model("nohess_direct")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
        m.minimize(block(x))
        res = m.solve(solver="direct", max_evals=400)
        assert CALLS["f"] > 0, "DIRECT never evaluated the external function"
        assert CALLS["H"] == 0, "no Hessian was supplied, so none may be called"
        xv = np.asarray(res.value(x), dtype=float)
        np.testing.assert_allclose(xv, [1.5, 0.5], rtol=0, atol=5e-2)

    def test_second_derivative_request_is_actionable(self):
        """Defence in depth: if something differentiates the Jacobian anyway, the
        message must be ours, not JAX's "Pure callbacks do not support JVP"."""
        import jax
        import jax.numpy as jnp

        ext_mod = _ext_module()

        composite = ext_mod._build_composite(
            "sim",
            (1,),
            ext_mod._checked(ext_f, role="fn", label="sim", want=lambda xs: (1,)),
            ext_mod._checked(ext_J, role="jac", label="sim", want=lambda xs: (1,) + xs),
            None,
        )
        with pytest.raises(ValueError, match="supplies no Hessian"):
            jax.hessian(lambda x: composite(x)[0])(jnp.array([1.3, 0.7]))


class TestArgumentValidation:
    def test_jac_is_required_and_points_at_dm_custom(self):
        with pytest.raises(TypeError) as excinfo:
            dm.external(ext_f, shape=(1,))
        msg = str(excinfo.value)
        assert "jac=" in msg
        assert "direct" in msg, "a derivative-free user needs to be told where to go"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"fn": "not callable", "jac": ext_J},
            {"fn": ext_f, "jac": "not callable"},
            {"fn": ext_f, "jac": ext_J, "hess": "not callable"},
        ],
    )
    def test_non_callables_are_rejected(self, kwargs):
        fn = kwargs.pop("fn")
        with pytest.raises(TypeError):
            dm.external(fn, shape=(1,), **kwargs)

    @pytest.mark.parametrize(
        "declared,want",
        [(None, ()), ((), ()), (3, (3,)), ((2,), (2,)), ((2, 3), (2, 3))],
    )
    def test_shape_normalization(self, declared, want):
        assert _ext_module()._as_shape(declared) == want

    def test_spec_is_attached_and_readable(self):
        external_spec = _ext_module().external_spec

        builder = dm.external(ext_f, jac=ext_J, hess=ext_H, shape=(1,), name="sim")
        node = builder(dm.Model("m").continuous("x", shape=(2,), lb=0.0, ub=1.0))
        spec = external_spec(node.fn)
        assert spec is not None
        assert spec.name == "sim"
        assert spec.out_shape == (1,)
        assert spec.has_hessian is True

    def test_plain_dm_custom_is_not_external(self):
        """The solver's Hessian check must not fire on an ordinary dm.custom node."""
        external_spec = _ext_module().external_spec

        import jax.numpy as jnp

        node = dm.custom(lambda x: jnp.sum(x**2))(
            dm.Model("m").continuous("x", shape=(2,), lb=0.0, ub=1.0)
        )
        assert external_spec(node.fn) is None


class TestReturnShapeErrors:
    """A wrong return shape must name the callable, the role and both shapes.

    ``pure_callback`` does reject these -- soundness never depended on this
    wrapper -- but it reports ``INTERNAL: CpuCallback error calling callback``,
    which identifies neither which of three callables was wrong nor what was
    expected. A transposed Jacobian is the mistake this feature invites most.
    """

    @staticmethod
    def _solve_with(**overrides):
        fns = {"fn": ext_f, "jac": ext_J, "hess": ext_H}
        fns.update(overrides)
        block = dm.external(fns["fn"], jac=fns["jac"], hess=fns["hess"], shape=(1,), name="sim")
        m = dm.Model("bad")
        x = m.continuous("x", shape=(2,), lb=0.2, ub=3.0)
        m.minimize(dm.sum(x))
        m.subject_to(block(x)[0] == TARGET, name="block")
        return m.solve()

    def test_transposed_jacobian(self):
        with pytest.raises(Exception) as excinfo:
            self._solve_with(jac=lambda x: np.zeros((2, 1)))
        msg = str(excinfo.value)
        assert "'sim'" in msg and "jac" in msg
        assert "(2, 1)" in msg and "(1, 2)" in msg, msg
        # The FIRST line must be ours. This error propagates on its own, but
        # wrapped: JAX's `INTERNAL: CpuCallback error calling callback` led until
        # `_reraise_external_failure` covered the propagating case too, and a
        # reader who stops after one line learned nothing.
        head = msg.splitlines()[0]
        assert "CpuCallback" not in head, head
        assert "'sim'" in head, head

    def test_wrong_length_value(self):
        with pytest.raises(Exception) as excinfo:
            self._solve_with(fn=lambda x: np.zeros(3))
        msg = str(excinfo.value)
        assert "'sim'" in msg and "fn" in msg
        assert "(3,)" in msg and "(1,)" in msg, msg

    def test_wrong_rank_hessian(self):
        """A bad Hessian must reach the caller, which needs the solver's help.

        POUNCE catches an exception raised inside a Hessian callback, logs it, and
        runs on to a withheld-incumbent ``status="error"`` -- so before
        ``_external_block_failure`` this returned normally after ~40 stderr lines
        and ``m.solve()`` raised nothing at all. This test is that regression.
        """
        with pytest.raises(Exception) as excinfo:
            self._solve_with(hess=lambda x: np.zeros((1, 2)))
        msg = str(excinfo.value)
        assert "'sim'" in msg and "hess" in msg
        assert "(1, 2, 2)" in msg, msg

    def test_none_return_is_rejected_not_turned_into_nan(self):
        """``np.asarray(None, dtype=float)`` is ``array(nan)``, not an error.

        The declared shape here is ``()``, so a shape check cannot catch it: NaN
        has exactly the shape a scalar block expects. Without the explicit
        ``None`` guard a forgotten ``return`` fed NaN to the solver as a value.
        """
        block = dm.external(
            lambda x: None,
            jac=lambda x: np.zeros(2),
            hess=lambda x: np.zeros((2, 2)),
            shape=(),
            name="forgot",
        )
        m = dm.Model("nonereturn")
        x = m.continuous("x", shape=(2,), lb=0.2, ub=3.0)
        m.minimize(block(x))
        with pytest.raises(Exception) as excinfo:
            m.solve()
        msg = str(excinfo.value)
        assert "'forgot'" in msg and "returned None" in msg, msg

    def test_direct_path_leads_with_our_message(self):
        """``solver="direct"`` propagates, but JAX's wrapper took the first line.

        The named message was always in the exception *text*; it was preceded by
        ``JaxRuntimeError: INTERNAL: CpuCallback error calling callback``, which is
        what a reader sees first and says nothing.
        """
        block = dm.external(
            lambda x: np.zeros(3), jac=lambda x: np.zeros((1, 2)), shape=(1,), name="bad"
        )
        m = dm.Model("direct_bad")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
        m.minimize(block(x)[0])
        with pytest.raises(Exception) as excinfo:
            m.solve(solver="direct", max_evals=50)
        head = str(excinfo.value).splitlines()[0]
        assert "'bad'" in head, f"the first line must name the block, got: {head}"
        assert "CpuCallback" not in head, head

    def test_unconvertible_return(self):
        with pytest.raises(Exception) as excinfo:
            self._solve_with(fn=lambda x: "not an array")
        assert "not convertible" in str(excinfo.value)

    def test_lists_and_int_arrays_are_accepted(self):
        """Coercion is the one leniency: external code returns these routinely."""
        res = self._solve_with(
            fn=lambda x: [
                float(np.asarray(x)[0]) ** 2 * float(np.asarray(x)[1])
                + float(np.exp(np.asarray(x)[1]))
            ],
            jac=lambda x: [
                [
                    2.0 * np.asarray(x)[0] * np.asarray(x)[1],
                    np.asarray(x)[0] ** 2 + np.exp(np.asarray(x)[1]),
                ]
            ],
        )
        assert res.status == "feasible"


class TestInheritedRefusals:
    def test_nl_export_refuses(self):
        """An opaque node has no ``.nl`` representation; that must stay a refusal.

        Silently dropping the block would export a *different model* under the
        same name, which an external solver would then solve and report on.
        """
        m, _ = _model(external=True)
        with pytest.raises(Exception) as excinfo:
            m.to_nl()
        assert "CustomCall" in str(excinfo.value) or "custom" in str(excinfo.value).lower()

    def test_amp_refuses_an_opaque_block_instead_of_failing_internally(self):
        """``solver="amp"`` must refuse up front, not die inside its MILP build.

        AMP certifies by linearizing a partitioned relaxation, which an opaque body
        has no algebraic form for. Measured before the refusal: ``AMP: MILP
        build/solve failed at iteration 1: too many indices for array: array is
        0-dimensional``, then ``status="error"`` with ``objective=None`` -- no false
        bound, but no statement of the cause either.

        The message must name a backend that *does* work, otherwise the refusal
        just relocates the dead end.
        """
        m, _ = _model(external=True)
        with pytest.raises(ValueError) as excinfo:
            m.solve(solver="amp")
        msg = str(excinfo.value)
        assert "amp" in msg.lower()
        assert "direct" in msg, "the refusal must name a backend that works"
        # The failure mode this replaced must not be what the user sees.
        assert "0-dimensional" not in msg and "too many indices" not in msg

    def test_amp_refusal_covers_plain_dm_custom_too(self):
        """The guard is keyed on ``CustomCall``, not on ``dm.external`` (§2).

        The AMP dead end was never specific to external blocks -- an ordinary
        ``dm.custom`` body hit the same internal error. Fixing the class rather
        than the instance is only demonstrated if the plain node is covered, so
        this asserts it directly.
        """
        m = dm.Model("plain-custom")
        x = m.continuous("x", shape=(2,), lb=0.2, ub=3.0)
        m.minimize(dm.sum(x))
        blk = dm.custom(lambda v: v[0] * v[1], name="opaque")
        m.subject_to(blk(x) == 1.0, name="block")
        with pytest.raises(ValueError) as excinfo:
            m.solve(solver="amp")
        assert "direct" in str(excinfo.value)
