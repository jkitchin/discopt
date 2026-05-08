"""Tests for ICNN-based learned relaxations.

Validates:
  1. ICNN architecture: output shape, convexity, JIT/vmap, determinism, weight enforcement
  2. Learned relaxation wrappers: soundness, runtime enforcement, fallback
  3. Training pipeline: loss reduction, soundness on held-out data, data coverage,
     save/load roundtrip, gap reduction target
  4. Integration: compiler mode="learned", fallback, batch evaluator, solver flag

All functions are pure JAX and compatible with jax.jit and jax.vmap.
"""

from __future__ import annotations

import tempfile

import jax
import jax.numpy as jnp
import pytest

try:
    import equinox as eqx

    HAS_EQUINOX = True
except ImportError:
    HAS_EQUINOX = False

pytestmark = pytest.mark.skipif(not HAS_EQUINOX, reason="equinox/optax not installed")

if HAS_EQUINOX:
    from discopt._jax.icnn import (
        create_icnn,
        enforce_nonneg,
        icnn_pair_create,
        verify_convexity,
    )
    from discopt._jax.learned_relaxations import (
        LearnedRelaxationRegistry,
        create_learned_relaxation,
        load_pretrained_registry,
        relax_exp_learned,
    )

TOL = 1e-10
N_POINTS = 10_000


# =====================================================================
# ICNN Architecture Tests
# =====================================================================


class TestICNNArchitecture:
    """Tests for the ICNN module."""

    def test_icnn_output_shape(self):
        """ICNN produces scalar output for given input."""
        key = jax.random.PRNGKey(0)
        icnn = create_icnn(key, input_dim=2, hidden_dim=16, n_layers=2)
        x = jnp.array([0.5, 0.3])
        out = icnn(x)
        assert out.shape == (), f"Expected scalar, got shape {out.shape}"

    def test_icnn_output_shape_univariate(self):
        """ICNN with input_dim=2 (features for univariate op) produces scalar."""
        key = jax.random.PRNGKey(1)
        icnn = create_icnn(key, input_dim=2, hidden_dim=32, n_layers=3)
        x = jnp.array([0.5, 1.0])
        out = icnn(x)
        assert out.shape == (), f"Expected scalar, got shape {out.shape}"

    def test_icnn_convexity_1d(self):
        """ICNN output is convex w.r.t. input on random 1D points."""
        key = jax.random.PRNGKey(2)
        icnn = create_icnn(key, input_dim=2, hidden_dim=16, n_layers=3)
        # Sample random 2D feature vectors
        x_samples = jax.random.uniform(jax.random.PRNGKey(100), shape=(200, 2), dtype=jnp.float64)
        assert verify_convexity(icnn, x_samples)

    def test_icnn_convexity_2d(self):
        """ICNN output is convex w.r.t. 4D input (bilinear features)."""
        key = jax.random.PRNGKey(3)
        icnn = create_icnn(key, input_dim=4, hidden_dim=32, n_layers=3)
        x_samples = jax.random.uniform(jax.random.PRNGKey(101), shape=(200, 4), dtype=jnp.float64)
        assert verify_convexity(icnn, x_samples)

    def test_icnn_jit_compatible(self):
        """ICNN works under eqx.filter_jit."""
        key = jax.random.PRNGKey(4)
        icnn = create_icnn(key, input_dim=2, hidden_dim=16, n_layers=2)

        @eqx.filter_jit
        def forward(model, x):
            return model(x)

        x = jnp.array([0.5, 1.0])
        out1 = forward(icnn, x)
        out2 = forward(icnn, x)
        assert jnp.allclose(out1, out2)

    def test_icnn_vmap_compatible(self):
        """ICNN works with jax.vmap for batch inference."""
        key = jax.random.PRNGKey(5)
        icnn = create_icnn(key, input_dim=2, hidden_dim=16, n_layers=2)

        x_batch = jax.random.uniform(jax.random.PRNGKey(50), shape=(32, 2))
        batched_fn = jax.vmap(icnn)
        out = batched_fn(x_batch)
        assert out.shape == (32,), f"Expected (32,), got {out.shape}"

    def test_icnn_deterministic(self):
        """Same key produces identical ICNN outputs."""
        key = jax.random.PRNGKey(6)
        icnn1 = create_icnn(key, input_dim=2, hidden_dim=16, n_layers=2)
        icnn2 = create_icnn(key, input_dim=2, hidden_dim=16, n_layers=2)

        x = jnp.array([0.3, 0.7])
        assert jnp.allclose(icnn1(x), icnn2(x))

    def test_nonneg_weight_enforcement(self):
        """enforce_nonneg clamps hidden weights to >= 0."""
        key = jax.random.PRNGKey(7)
        icnn = create_icnn(key, input_dim=2, hidden_dim=16, n_layers=3)
        clamped = enforce_nonneg(icnn)

        for layer in clamped.hidden_layers:
            assert jnp.all(layer.weight >= 0.0), "Hidden weights should be non-negative"

    def test_icnn_pair_create(self):
        """icnn_pair_create returns two distinct ICNNs."""
        key = jax.random.PRNGKey(8)
        cv_net, cc_net = icnn_pair_create(key, input_dim=2)
        x = jnp.array([0.5, 1.0])
        # They should produce different outputs (different keys)
        assert not jnp.allclose(cv_net(x), cc_net(x))

    def test_icnn_different_keys_different_outputs(self):
        """Different keys produce different outputs."""
        icnn1 = create_icnn(jax.random.PRNGKey(10), input_dim=2)
        icnn2 = create_icnn(jax.random.PRNGKey(11), input_dim=2)
        x = jnp.array([0.5, 1.0])
        assert not jnp.allclose(icnn1(x), icnn2(x))


# =====================================================================
# Learned Relaxation Tests
# =====================================================================


class TestLearnedRelaxation:
    """Tests for per-operation learned relaxation wrappers."""

    @pytest.fixture
    def trained_exp(self):
        """Lightly trained exp relaxation for testing."""
        from discopt._jax.icnn_trainer import train_relaxation

        return train_relaxation("exp", n_epochs=50, batch_size=256, n_boxes=1000, points_per_box=5)

    @pytest.fixture
    def untrained_exp(self):
        """Untrained exp relaxation."""
        return create_learned_relaxation(jax.random.PRNGKey(0), "exp")

    def test_soundness_exp_runtime_enforcement(self, untrained_exp):
        """Runtime enforcement guarantees cv <= exp(x) <= cc even untrained."""
        key = jax.random.PRNGKey(100)
        x = jax.random.uniform(key, (N_POINTS,), minval=-2.0, maxval=2.0)
        lb = jnp.full(N_POINTS, -2.0)
        ub = jnp.full(N_POINTS, 2.0)
        true_val = jnp.exp(x)

        def eval_one(xi, lbi, ubi, fi):
            return untrained_exp(xi, lbi, ubi, fi)

        cv, cc = jax.vmap(eval_one)(x, lb, ub, true_val)
        assert jnp.all(cv <= true_val + TOL), f"cv > f(x): max={jnp.max(cv - true_val)}"
        assert jnp.all(cc >= true_val - TOL), f"cc < f(x): min={jnp.min(cc - true_val)}"

    def test_soundness_exp_trained(self, trained_exp):
        """Trained exp relaxation maintains soundness."""
        key = jax.random.PRNGKey(200)
        x = jax.random.uniform(key, (N_POINTS,), minval=-3.0, maxval=3.0)
        lb = jnp.full(N_POINTS, -3.0)
        ub = jnp.full(N_POINTS, 3.0)
        true_val = jnp.exp(x)

        def eval_one(xi, lbi, ubi, fi):
            return trained_exp(xi, lbi, ubi, fi)

        cv, cc = jax.vmap(eval_one)(x, lb, ub, true_val)
        assert jnp.all(cv <= true_val + TOL), f"cv > f(x): max={jnp.max(cv - true_val)}"
        assert jnp.all(cc >= true_val - TOL), f"cc < f(x): min={jnp.min(cc - true_val)}"

    def test_soundness_log(self):
        """Runtime enforcement for log relaxation."""
        lr = create_learned_relaxation(jax.random.PRNGKey(1), "log")
        key = jax.random.PRNGKey(101)
        x = jax.random.uniform(key, (N_POINTS,), minval=0.1, maxval=5.0)
        lb = jnp.full(N_POINTS, 0.1)
        ub = jnp.full(N_POINTS, 5.0)
        true_val = jnp.log(x)

        cv, cc = jax.vmap(lambda xi, lbi, ubi, fi: lr(xi, lbi, ubi, fi))(x, lb, ub, true_val)
        assert jnp.all(cv <= true_val + TOL)
        assert jnp.all(cc >= true_val - TOL)

    def test_soundness_sqrt(self):
        """Runtime enforcement for sqrt relaxation."""
        lr = create_learned_relaxation(jax.random.PRNGKey(2), "sqrt")
        key = jax.random.PRNGKey(102)
        x = jax.random.uniform(key, (N_POINTS,), minval=0.01, maxval=10.0)
        lb = jnp.full(N_POINTS, 0.01)
        ub = jnp.full(N_POINTS, 10.0)
        true_val = jnp.sqrt(x)

        cv, cc = jax.vmap(lambda xi, lbi, ubi, fi: lr(xi, lbi, ubi, fi))(x, lb, ub, true_val)
        assert jnp.all(cv <= true_val + TOL)
        assert jnp.all(cc >= true_val - TOL)

    def test_soundness_sin(self):
        """Runtime enforcement for sin relaxation."""
        lr = create_learned_relaxation(jax.random.PRNGKey(3), "sin")
        key = jax.random.PRNGKey(103)
        x = jax.random.uniform(key, (N_POINTS,), minval=-3.0, maxval=3.0)
        lb = jnp.full(N_POINTS, -3.0)
        ub = jnp.full(N_POINTS, 3.0)
        true_val = jnp.sin(x)

        cv, cc = jax.vmap(lambda xi, lbi, ubi, fi: lr(xi, lbi, ubi, fi))(x, lb, ub, true_val)
        assert jnp.all(cv <= true_val + TOL)
        assert jnp.all(cc >= true_val - TOL)

    def test_soundness_bilinear(self):
        """Runtime enforcement for bilinear relaxation."""
        lr = create_learned_relaxation(jax.random.PRNGKey(4), "bilinear")
        key = jax.random.PRNGKey(104)
        keys = jax.random.split(key, 2)
        x = jax.random.uniform(keys[0], (N_POINTS,), minval=-3.0, maxval=3.0)
        y = jax.random.uniform(keys[1], (N_POINTS,), minval=-3.0, maxval=3.0)
        true_val = x * y

        x_lb = jnp.full(N_POINTS, -3.0)
        x_ub = jnp.full(N_POINTS, 3.0)
        y_lb = jnp.full(N_POINTS, -3.0)
        y_ub = jnp.full(N_POINTS, 3.0)

        def eval_one(xi, yi, xlb, xub, ylb, yub, fi):
            xy = jnp.stack([xi, yi])
            xy_lb = jnp.stack([xlb, ylb])
            xy_ub = jnp.stack([xub, yub])
            return lr(xy, xy_lb, xy_ub, fi)

        cv, cc = jax.vmap(eval_one)(x, y, x_lb, x_ub, y_lb, y_ub, true_val)
        assert jnp.all(cv <= true_val + TOL)
        assert jnp.all(cc >= true_val - TOL)

    def test_soundness_square(self):
        """Runtime enforcement for square relaxation."""
        lr = create_learned_relaxation(jax.random.PRNGKey(5), "square")
        key = jax.random.PRNGKey(105)
        x = jax.random.uniform(key, (N_POINTS,), minval=-3.0, maxval=3.0)
        lb = jnp.full(N_POINTS, -3.0)
        ub = jnp.full(N_POINTS, 3.0)
        true_val = x**2

        cv, cc = jax.vmap(lambda xi, lbi, ubi, fi: lr(xi, lbi, ubi, fi))(x, lb, ub, true_val)
        assert jnp.all(cv <= true_val + TOL)
        assert jnp.all(cc >= true_val - TOL)

    def test_per_op_wrappers(self):
        """Per-operation wrapper functions match the expected signatures."""
        lr_exp = create_learned_relaxation(jax.random.PRNGKey(10), "exp")
        x, lb, ub = jnp.array(1.0), jnp.array(0.0), jnp.array(2.0)
        cv, cc = relax_exp_learned(x, lb, ub, lr_exp)
        assert cv.shape == ()
        assert cc.shape == ()
        assert cv <= jnp.exp(x) + TOL
        assert cc >= jnp.exp(x) - TOL

    def test_fallback_to_mccormick(self):
        """Registry.get returns None for unregistered ops."""
        registry = LearnedRelaxationRegistry()
        assert registry.get("exp") is None
        assert registry.get("unknown") is None


# =====================================================================
# Training Pipeline Tests
# =====================================================================


class TestTrainingPipeline:
    """Tests for the ICNN training pipeline."""

    def test_training_reduces_loss(self):
        """Loss decreases over epochs."""
        from discopt._jax.icnn_trainer import (
            _compute_predictions,
            generate_training_data,
            relaxation_loss,
            train_relaxation,
        )

        # Untrained model
        untrained = create_learned_relaxation(jax.random.PRNGKey(0), "exp")
        data = generate_training_data("exp", n_boxes=500, points_per_box=5)

        x_b = data.x[:256]
        lb_b = data.lb[:256]
        ub_b = data.ub[:256]
        f_b = data.f_x[:256]
        mc_cv_b = data.mc_cv[:256]
        mc_cc_b = data.mc_cc[:256]

        cv0, cc0 = _compute_predictions(untrained, x_b, lb_b, ub_b, False)
        loss0 = float(relaxation_loss(cv0, cc0, f_b, mc_cv_b, mc_cc_b))

        # Lightly trained model
        trained = train_relaxation(
            "exp", n_epochs=100, batch_size=256, n_boxes=500, points_per_box=5
        )
        cv1, cc1 = _compute_predictions(trained, x_b, lb_b, ub_b, False)
        loss1 = float(relaxation_loss(cv1, cc1, f_b, mc_cv_b, mc_cc_b))

        assert loss1 < loss0, f"Training didn't reduce loss: {loss1} >= {loss0}"

    def test_trained_model_sound(self):
        """Post-training soundness on held-out data (with runtime enforcement)."""
        from discopt._jax.icnn_trainer import train_relaxation

        trained = train_relaxation(
            "log", n_epochs=50, batch_size=256, n_boxes=500, points_per_box=5
        )

        key = jax.random.PRNGKey(999)
        x = jax.random.uniform(key, (1000,), minval=0.1, maxval=5.0)
        lb = jnp.full(1000, 0.1)
        ub = jnp.full(1000, 5.0)
        true_val = jnp.log(x)

        cv, cc = jax.vmap(lambda xi, lbi, ubi, fi: trained(xi, lbi, ubi, fi))(x, lb, ub, true_val)
        assert jnp.all(cv <= true_val + TOL)
        assert jnp.all(cc >= true_val - TOL)

    def test_data_generation_coverage(self):
        """Training data covers the domain uniformly."""
        from discopt._jax.icnn_trainer import generate_training_data

        data = generate_training_data("exp", n_boxes=1000, points_per_box=10)
        n_total = 1000 * 10

        assert data.x.shape == (n_total,)
        assert data.lb.shape == (n_total,)
        assert data.ub.shape == (n_total,)
        assert data.f_x.shape == (n_total,)
        assert data.mc_cv.shape == (n_total,)
        assert data.mc_cc.shape == (n_total,)

        # Check lb <= x <= ub
        assert jnp.all(data.x >= data.lb - TOL)
        assert jnp.all(data.x <= data.ub + TOL)

        # McCormick should be sound
        assert jnp.all(data.mc_cv <= data.f_x + TOL)
        assert jnp.all(data.mc_cc >= data.f_x - TOL)

    def test_data_generation_bilinear(self):
        """Bilinear training data has correct shape and soundness."""
        from discopt._jax.icnn_trainer import generate_training_data

        data = generate_training_data("bilinear", n_boxes=500, points_per_box=5)
        n_total = 500 * 5

        assert data.x.shape == (n_total, 2)
        assert data.lb.shape == (n_total, 2)
        assert data.ub.shape == (n_total, 2)
        assert data.f_x.shape == (n_total,)
        assert data.mc_cv.shape == (n_total,)
        assert data.mc_cc.shape == (n_total,)

        # McCormick soundness
        assert jnp.all(data.mc_cv <= data.f_x + TOL)
        assert jnp.all(data.mc_cc >= data.f_x - TOL)

    def test_save_load_roundtrip(self):
        """Serialization preserves model parameters."""
        lr = create_learned_relaxation(jax.random.PRNGKey(42), "exp")
        registry = LearnedRelaxationRegistry()
        registry.register("exp", lr)

        with tempfile.TemporaryDirectory() as tmpdir:
            registry.save(tmpdir)
            loaded = LearnedRelaxationRegistry.load(tmpdir)
            assert "exp" in loaded
            loaded_lr = loaded.get("exp")

            x = jnp.array(1.0)
            lb = jnp.array(0.0)
            ub = jnp.array(2.0)
            true_val = jnp.exp(x)

            cv_orig, cc_orig = lr(x, lb, ub, true_val)
            cv_loaded, cc_loaded = loaded_lr(x, lb, ub, true_val)

            assert jnp.allclose(cv_orig, cv_loaded, atol=1e-12)
            assert jnp.allclose(cc_orig, cc_loaded, atol=1e-12)

    @pytest.mark.slow
    def test_gap_reduction_target(self):
        """Trained relaxation achieves measurable gap reduction vs McCormick.

        With lightweight training (200 epochs, 5k boxes) we target >= 5%
        gap reduction. Full training (500 epochs, 50k boxes) achieves >= 30%.
        """
        from discopt._jax.icnn_trainer import compute_gap_reduction, train_relaxation

        trained = train_relaxation(
            "exp", n_epochs=200, batch_size=512, n_boxes=5000, points_per_box=10
        )
        reduction = compute_gap_reduction(trained, "exp", n_test_boxes=100, points_per_box=50)
        assert reduction >= 0.05, f"Gap reduction {reduction:.2%} < 5% target"

    def test_relaxation_loss_components(self):
        """Loss function penalizes soundness violations heavily."""
        from discopt._jax.icnn_trainer import relaxation_loss

        f_x = jnp.array([1.0, 2.0, 3.0])
        mc_cv = jnp.array([0.5, 1.5, 2.5])
        mc_cc = jnp.array([1.5, 2.5, 3.5])

        # Sound predictions: no soundness penalty
        cv_sound = jnp.array([0.8, 1.8, 2.8])
        cc_sound = jnp.array([1.2, 2.2, 3.2])
        loss_sound = relaxation_loss(cv_sound, cc_sound, f_x, mc_cv, mc_cc)

        # Unsound predictions: cv > f(x)
        cv_bad = jnp.array([1.5, 2.5, 3.5])
        loss_bad = relaxation_loss(cv_bad, cc_sound, f_x, mc_cv, mc_cc)

        assert float(loss_bad) > float(loss_sound), "Unsound predictions should have higher loss"


# =====================================================================
# Integration Tests
# =====================================================================


class TestIntegration:
    """Tests for relaxation compiler and solver integration."""

    def _make_test_model(self):
        """Create a simple model with exp(x) for testing."""
        import discopt.modeling.core as dm

        m = dm.Model()
        x = m.continuous("x", lb=0.0, ub=3.0)
        m.minimize(dm.exp(x))
        return m

    def _make_bilinear_model(self):
        """Create a model with x*y for testing."""
        import discopt.modeling.core as dm

        m = dm.Model()
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.continuous("y", lb=-2.0, ub=2.0)
        m.minimize(x * y)
        return m

    def test_compiler_learned_mode(self):
        """compile_relaxation(mode='learned') works with a registry."""
        from discopt._jax.relaxation_compiler import compile_relaxation

        model = self._make_test_model()
        lr = create_learned_relaxation(jax.random.PRNGKey(0), "exp")
        registry = LearnedRelaxationRegistry()
        registry.register("exp", lr)

        relax_fn = compile_relaxation(
            model._objective.expression, model, mode="learned", learned_registry=registry
        )

        x_cv = jnp.array([1.5])
        x_cc = jnp.array([1.5])
        lb = jnp.array([0.0])
        ub = jnp.array([3.0])

        cv, cc = relax_fn(x_cv, x_cc, lb, ub)
        true_val = jnp.exp(1.5)
        assert cv <= true_val + TOL, f"cv={cv} > exp(1.5)={true_val}"
        assert cc >= true_val - TOL, f"cc={cc} < exp(1.5)={true_val}"

    def test_compiler_learned_bilinear(self):
        """compile_relaxation(mode='learned') handles bilinear x*y."""
        from discopt._jax.relaxation_compiler import compile_relaxation

        model = self._make_bilinear_model()
        lr = create_learned_relaxation(jax.random.PRNGKey(0), "bilinear")
        registry = LearnedRelaxationRegistry()
        registry.register("bilinear", lr)

        relax_fn = compile_relaxation(
            model._objective.expression, model, mode="learned", learned_registry=registry
        )

        x_cv = jnp.array([1.0, 0.5])
        x_cc = jnp.array([1.0, 0.5])
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])

        cv, cc = relax_fn(x_cv, x_cc, lb, ub)
        true_val = 1.0 * 0.5
        assert cv <= true_val + TOL, f"cv={cv} > true_val={true_val}"
        assert cc >= true_val - TOL, f"cc={cc} < true_val={true_val}"

    def test_compiler_fallback(self):
        """Unknown ops fall through to standard McCormick in learned mode."""
        import discopt.modeling.core as dm
        from discopt._jax.relaxation_compiler import compile_relaxation

        model = dm.Model()
        x = model.continuous("x", lb=0.1, ub=5.0)
        model.minimize(dm.log2(x))

        # Empty registry — no learned model for log2
        registry = LearnedRelaxationRegistry()

        relax_fn = compile_relaxation(
            model._objective.expression, model, mode="learned", learned_registry=registry
        )

        x_cv = jnp.array([2.0])
        x_cc = jnp.array([2.0])
        lb = jnp.array([0.1])
        ub = jnp.array([5.0])

        cv, cc = relax_fn(x_cv, x_cc, lb, ub)
        true_val = jnp.log2(2.0)
        assert jnp.isfinite(cv)
        assert jnp.isfinite(cc)
        assert cv <= true_val + TOL
        assert cc >= true_val - TOL

    def test_no_regression_standard_mode(self):
        """mode='standard' produces identical results to default."""
        from discopt._jax.relaxation_compiler import compile_relaxation

        model = self._make_test_model()

        relax_default = compile_relaxation(model._objective.expression, model)
        relax_standard = compile_relaxation(model._objective.expression, model, mode="standard")

        x_cv = jnp.array([1.5])
        x_cc = jnp.array([1.5])
        lb = jnp.array([0.0])
        ub = jnp.array([3.0])

        cv1, cc1 = relax_default(x_cv, x_cc, lb, ub)
        cv2, cc2 = relax_standard(x_cv, x_cc, lb, ub)

        assert jnp.allclose(cv1, cv2, atol=1e-14)
        assert jnp.allclose(cc1, cc2, atol=1e-14)

    def test_solver_opt_in_flag(self):
        """solve_model accepts use_learned_relaxations parameter."""
        import inspect

        from discopt.solver import solve_model

        sig = inspect.signature(solve_model)
        assert "use_learned_relaxations" in sig.parameters
        param = sig.parameters["use_learned_relaxations"]
        assert param.default is False

    def test_registry_len_and_contains(self):
        """Registry supports len() and 'in' operator."""
        registry = LearnedRelaxationRegistry()
        assert len(registry) == 0
        assert "exp" not in registry

        lr = create_learned_relaxation(jax.random.PRNGKey(0), "exp")
        registry.register("exp", lr)
        assert len(registry) == 1
        assert "exp" in registry

    def test_load_pretrained_no_dir(self):
        """load_pretrained_registry warns when no pretrained dir exists."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry = load_pretrained_registry(path="/nonexistent/path")
            assert len(registry) == 0
            assert len(w) == 1
            assert "pretrained" in str(w[0].message).lower()

    def test_create_learned_relaxation_invalid_op(self):
        """Creating a learned relaxation for an unknown op raises ValueError."""
        with pytest.raises(ValueError, match="Unknown operation"):
            create_learned_relaxation(jax.random.PRNGKey(0), "unknown_op")
