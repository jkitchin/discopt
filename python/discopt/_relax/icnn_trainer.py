"""
Training pipeline for ICNN-based learned relaxations.

Generates training data from McCormick baselines, trains ICNN pairs to
minimize relaxation gap while enforcing soundness, and saves pretrained
models for use by the relaxation compiler.

Workflow:
  1. ``generate_training_data(op_name)`` — random boxes + function evals
  2. ``train_relaxation(op_name)`` — ICNN pair training with soundness loss
  3. ``train_all()`` — train all 6 operations and save registry
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from discopt._relax.icnn import enforce_nonneg
from discopt._relax.learned_relaxations import (
    _OP_CONFIGS,
    LearnedRelaxation,
    LearnedRelaxationRegistry,
    create_learned_relaxation,
)
from discopt._relax.mccormick import (
    relax_bilinear,
    relax_exp,
    relax_log,
    relax_sin,
    relax_sqrt,
    relax_square,
)


class TrainingData(NamedTuple):
    """Training data for a single operation.

    All arrays have shape ``(n_samples,)`` for univariate ops
    or ``(n_samples, 2)`` for bivariate ops.
    """

    x: jnp.ndarray  # evaluation points
    lb: jnp.ndarray  # box lower bounds
    ub: jnp.ndarray  # box upper bounds
    f_x: jnp.ndarray  # true function values
    mc_cv: jnp.ndarray  # McCormick convex underestimator
    mc_cc: jnp.ndarray  # McCormick concave overestimator


# ---------------------------------------------------------------------------
# McCormick evaluation wrappers (for training data generation)
# ---------------------------------------------------------------------------

_MC_FNS = {
    "exp": lambda x, lb, ub: relax_exp(x, lb, ub),
    "log": lambda x, lb, ub: relax_log(x, lb, ub),
    "sqrt": lambda x, lb, ub: relax_sqrt(x, lb, ub),
    "sin": lambda x, lb, ub: relax_sin(x, lb, ub),
    "square": lambda x, lb, ub: relax_square(x, lb, ub),
}

_TRUE_FNS = {
    "exp": jnp.exp,
    "log": jnp.log,
    "sqrt": jnp.sqrt,
    "sin": jnp.sin,
    "square": lambda x: x**2,
}

# Domain constraints per operation
_DOMAIN_RANGES = {
    "exp": (-5.0, 5.0),
    "log": (0.01, 10.0),
    "sqrt": (0.0, 10.0),
    "sin": (-2 * jnp.pi, 2 * jnp.pi),
    "square": (-5.0, 5.0),
    "bilinear": (-5.0, 5.0),
}


def generate_training_data(
    op_name: str,
    n_boxes: int = 50_000,
    points_per_box: int = 10,
    key: Optional[jax.Array] = None,
) -> TrainingData:
    """Generate training data for a single operation.

    Samples random boxes ``[lb, ub]`` with varied widths, then evaluates
    the true function and McCormick relaxation at uniformly-sampled points
    within each box.

    Args:
        op_name: Operation name (exp, log, sqrt, sin, square, bilinear).
        n_boxes: Number of random boxes to sample.
        points_per_box: Points per box.
        key: JAX PRNG key (default: PRNGKey(42)).

    Returns:
        Training data with ``n_boxes * points_per_box`` samples.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    domain_lo, domain_hi = _DOMAIN_RANGES[op_name]
    is_bilinear = op_name == "bilinear"

    k1, k2, k3 = jax.random.split(key, 3)

    if is_bilinear:
        return _generate_bilinear_data(k1, k2, k3, n_boxes, points_per_box, domain_lo, domain_hi)

    # --- Univariate operations ---
    # Sample box endpoints
    k_lb, k_width, k_pts = jax.random.split(k1, 3)
    lb_raw = jax.random.uniform(k_lb, (n_boxes,), minval=domain_lo, maxval=domain_hi)
    # Width: log-uniform from 0.01 to min(domain_hi - lb, 10.0)
    max_width = jnp.minimum(domain_hi - lb_raw, 10.0)
    max_width = jnp.maximum(max_width, 0.02)
    log_width = jax.random.uniform(k_width, (n_boxes,), minval=jnp.log(0.01), maxval=0.0)
    width = jnp.exp(log_width) * max_width
    ub_raw = jnp.minimum(lb_raw + width, domain_hi)
    lb_raw = jnp.maximum(lb_raw, domain_lo)

    # Sample points uniformly in each box
    alpha = jax.random.uniform(k_pts, (n_boxes, points_per_box))
    # Expand: (n_boxes, points_per_box)
    lb_exp = lb_raw[:, None]
    ub_exp = ub_raw[:, None]
    x_all = lb_exp + alpha * (ub_exp - lb_exp)

    # Flatten
    x_flat = x_all.reshape(-1)
    lb_flat = jnp.repeat(lb_raw, points_per_box)
    ub_flat = jnp.repeat(ub_raw, points_per_box)

    # Evaluate true function
    f_x = _TRUE_FNS[op_name](x_flat)

    # Evaluate McCormick
    mc_fn = _MC_FNS[op_name]
    mc_cv, mc_cc = mc_fn(x_flat, lb_flat, ub_flat)

    return TrainingData(x=x_flat, lb=lb_flat, ub=ub_flat, f_x=f_x, mc_cv=mc_cv, mc_cc=mc_cc)


def _generate_bilinear_data(
    k1, k2, k3, n_boxes, points_per_box, domain_lo, domain_hi
) -> TrainingData:
    """Generate training data for bilinear relaxation."""
    k_xlb, k_xw, k_ylb, k_yw, k_pts = jax.random.split(k1, 5)

    x_lb = jax.random.uniform(k_xlb, (n_boxes,), minval=domain_lo, maxval=domain_hi)
    max_xw = jnp.maximum(domain_hi - x_lb, 0.02)
    x_width = (
        jnp.exp(jax.random.uniform(k_xw, (n_boxes,), minval=jnp.log(0.01), maxval=0.0)) * max_xw
    )
    x_ub = jnp.minimum(x_lb + x_width, domain_hi)

    y_lb = jax.random.uniform(k_ylb, (n_boxes,), minval=domain_lo, maxval=domain_hi)
    max_yw = jnp.maximum(domain_hi - y_lb, 0.02)
    y_width = (
        jnp.exp(jax.random.uniform(k_yw, (n_boxes,), minval=jnp.log(0.01), maxval=0.0)) * max_yw
    )
    y_ub = jnp.minimum(y_lb + y_width, domain_hi)

    # Sample points
    alpha_x = jax.random.uniform(k_pts, (n_boxes, points_per_box))
    k_ay = jax.random.fold_in(k_pts, 1)
    alpha_y = jax.random.uniform(k_ay, (n_boxes, points_per_box))

    x_pts = x_lb[:, None] + alpha_x * (x_ub - x_lb)[:, None]
    y_pts = y_lb[:, None] + alpha_y * (y_ub - y_lb)[:, None]

    x_flat = x_pts.reshape(-1)
    y_flat = y_pts.reshape(-1)
    x_lb_flat = jnp.repeat(x_lb, points_per_box)
    x_ub_flat = jnp.repeat(x_ub, points_per_box)
    y_lb_flat = jnp.repeat(y_lb, points_per_box)
    y_ub_flat = jnp.repeat(y_ub, points_per_box)

    f_x = x_flat * y_flat
    mc_cv, mc_cc = relax_bilinear(x_flat, y_flat, x_lb_flat, x_ub_flat, y_lb_flat, y_ub_flat)

    # Stack into 2D for bilinear
    xy = jnp.stack([x_flat, y_flat], axis=-1)
    lb = jnp.stack([x_lb_flat, y_lb_flat], axis=-1)
    ub = jnp.stack([x_ub_flat, y_ub_flat], axis=-1)

    return TrainingData(x=xy, lb=lb, ub=ub, f_x=f_x, mc_cv=mc_cv, mc_cc=mc_cc)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def relaxation_loss(
    cv_pred: jnp.ndarray,
    cc_pred: jnp.ndarray,
    f_x: jnp.ndarray,
    mc_cv: jnp.ndarray,
    mc_cc: jnp.ndarray,
) -> jnp.ndarray:
    """Multi-objective loss for learned relaxation training.

    Components:
      1. **Soundness penalty** (weight 100): ``cv <= f(x)`` and ``cc >= f(x)``
      2. **Tightness** (weight 1): minimize gap ``cc - cv``
      3. **Improvement over McCormick** (weight 10): learned bounds should be
         at least as tight as McCormick

    Args:
        cv_pred: Predicted convex underestimator values.
        cc_pred: Predicted concave overestimator values.
        f_x: True function values.
        mc_cv: McCormick convex underestimator values.
        mc_cc: McCormick concave overestimator values.

    Returns:
        Scalar loss value.
    """
    # Soundness: penalize cv > f(x) and cc < f(x)
    sound_cv = jnp.mean(jax.nn.relu(cv_pred - f_x) ** 2)
    sound_cc = jnp.mean(jax.nn.relu(f_x - cc_pred) ** 2)

    # Tightness: minimize gap
    gap = jnp.mean(cc_pred - cv_pred)

    # Improvement over McCormick: cv should be >= mc_cv, cc should be <= mc_cc
    improve_cv = jnp.mean(jax.nn.relu(mc_cv - cv_pred))
    improve_cc = jnp.mean(jax.nn.relu(cc_pred - mc_cc))

    return 100.0 * (sound_cv + sound_cc) + 1.0 * gap + 10.0 * (improve_cv + improve_cc)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _compute_predictions(
    learned: LearnedRelaxation,
    x_batch: jnp.ndarray,
    lb_batch: jnp.ndarray,
    ub_batch: jnp.ndarray,
    is_bilinear: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute cv/cc predictions for a batch (no soundness clamping)."""

    def predict_one(x, lb, ub):
        width = jnp.maximum(ub - lb, 1e-15)
        x_norm = (x - lb) / width
        if is_bilinear:
            features = jnp.concatenate([x_norm, width])
        else:
            features = jnp.stack([x_norm, width])
        cv = learned.cv_net(features)
        cc = -learned.cc_net(-features)
        return cv, cc

    cv_batch, cc_batch = jax.vmap(predict_one)(x_batch, lb_batch, ub_batch)
    return cv_batch, cc_batch


def train_relaxation(
    op_name: str,
    n_epochs: int = 500,
    batch_size: int = 1024,
    lr: float = 1e-3,
    n_boxes: int = 50_000,
    points_per_box: int = 10,
    key: Optional[jax.Array] = None,
    verbose: bool = False,
) -> LearnedRelaxation:
    """Train an ICNN pair for a single operation.

    Steps:
      1. Generate training data (random boxes + McCormick baselines)
      2. Initialize ICNN pair
      3. Train with Adam, monitoring soundness on validation set
      4. Post-training: project weights non-negative, verify convexity
      5. Return trained LearnedRelaxation

    Args:
        op_name: Operation name (exp, log, sqrt, sin, square, bilinear).
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        n_boxes: Number of random boxes for data generation.
        points_per_box: Points per box.
        key: JAX PRNG key.
        verbose: Print training progress.

    Returns:
        Trained LearnedRelaxation.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    k_data, k_model, k_train = jax.random.split(key, 3)

    # 1. Generate training data
    data = generate_training_data(op_name, n_boxes, points_per_box, k_data)
    n_total = data.x.shape[0]

    # Split 90/10 train/val
    n_val = max(n_total // 10, batch_size)
    n_train = n_total - n_val

    is_bilinear = op_name == "bilinear"

    # 2. Initialize model
    learned = create_learned_relaxation(k_model, op_name)

    # 3. Training loop
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(learned, eqx.is_array))

    @eqx.filter_jit
    def train_step(learned, opt_state, x_b, lb_b, ub_b, f_b, mc_cv_b, mc_cc_b):
        def loss_fn(model):
            cv, cc = _compute_predictions(model, x_b, lb_b, ub_b, is_bilinear)
            return relaxation_loss(cv, cc, f_b, mc_cv_b, mc_cc_b)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(learned)
        params = eqx.filter(learned, eqx.is_array)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_model = eqx.apply_updates(learned, updates)
        return new_model, new_opt_state, loss

    best_loss = jnp.inf
    best_model = learned
    patience = 50
    no_improve = 0

    for epoch in range(n_epochs):
        # Shuffle training data
        k_train, k_shuffle = jax.random.split(k_train)
        perm = jax.random.permutation(k_shuffle, n_train)

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            x_b = data.x[idx]
            lb_b = data.lb[idx]
            ub_b = data.ub[idx]
            f_b = data.f_x[idx]
            mc_cv_b = data.mc_cv[idx]
            mc_cc_b = data.mc_cc[idx]

            learned, opt_state, loss = train_step(
                learned, opt_state, x_b, lb_b, ub_b, f_b, mc_cv_b, mc_cc_b
            )
            epoch_loss += float(loss)
            n_batches += 1

        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
        else:
            avg_loss = 0.0

        # Validation loss
        val_idx = jnp.arange(n_train, n_total)
        val_x = data.x[val_idx[:n_val]]
        val_lb = data.lb[val_idx[:n_val]]
        val_ub = data.ub[val_idx[:n_val]]
        val_f = data.f_x[val_idx[:n_val]]
        val_mc_cv = data.mc_cv[val_idx[:n_val]]
        val_mc_cc = data.mc_cc[val_idx[:n_val]]

        cv_val, cc_val = _compute_predictions(learned, val_x, val_lb, val_ub, is_bilinear)
        val_loss = float(relaxation_loss(cv_val, cc_val, val_f, val_mc_cv, val_mc_cc))

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = learned
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:4d}: train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # 4. Post-training: enforce non-negative weights
    best_model = LearnedRelaxation(
        cv_net=enforce_nonneg(best_model.cv_net),
        cc_net=enforce_nonneg(best_model.cc_net),
        op_name=best_model.op_name,
        input_dim=best_model.input_dim,
    )

    return best_model


def train_all(
    save_dir: Optional[str] = None,
    n_epochs: int = 500,
    batch_size: int = 1024,
    lr: float = 1e-3,
    n_boxes: int = 50_000,
    points_per_box: int = 10,
    key: Optional[jax.Array] = None,
    verbose: bool = True,
) -> LearnedRelaxationRegistry:
    """Train ICNN pairs for all 6 operations and save.

    Args:
        save_dir: Directory for ``.eqx`` files. If None, uses default
            ``discopt/_relax/pretrained/``.
        n_epochs: Training epochs per operation.
        batch_size: Mini-batch size.
        lr: Learning rate.
        n_boxes: Number of random boxes per operation.
        points_per_box: Points per box.
        key: JAX PRNG key.
        verbose: Print progress.

    Returns:
        Registry with all trained relaxations.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "pretrained")

    registry = LearnedRelaxationRegistry()
    ops = list(_OP_CONFIGS.keys())

    for i, op_name in enumerate(ops):
        k_op = jax.random.fold_in(key, i)
        if verbose:
            print(f"Training {op_name} ({i + 1}/{len(ops)})...")

        learned = train_relaxation(
            op_name,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            n_boxes=n_boxes,
            points_per_box=points_per_box,
            key=k_op,
            verbose=verbose,
        )
        registry.register(op_name, learned)

    registry.save(save_dir)
    if verbose:
        print(f"Saved {len(registry)} pretrained models to {save_dir}")

    return registry


def compute_gap_reduction(
    learned: LearnedRelaxation,
    op_name: str,
    n_test_boxes: int = 100,
    points_per_box: int = 50,
    key: Optional[jax.Array] = None,
) -> float:
    """Compute average gap reduction vs McCormick on test boxes.

    Returns:
        Fraction in [0, 1] representing gap reduction. E.g. 0.35 means
        learned relaxation gap is 35% smaller than McCormick gap.
    """
    if key is None:
        key = jax.random.PRNGKey(99)

    data = generate_training_data(op_name, n_test_boxes, points_per_box, key)
    is_bilinear = op_name == "bilinear"

    # McCormick gap
    mc_gap = jnp.mean(data.mc_cc - data.mc_cv)

    # Learned gap (with soundness enforcement + McCormick fallback)
    cv_pred, cc_pred = _compute_predictions(learned, data.x, data.lb, data.ub, is_bilinear)
    # Apply soundness clamping
    cv_clamped = jnp.minimum(cv_pred, data.f_x)
    cc_clamped = jnp.maximum(cc_pred, data.f_x)
    # Clamp against McCormick: learned should never be wider
    cv_final = jnp.maximum(cv_clamped, data.mc_cv)
    cc_final = jnp.minimum(cc_clamped, data.mc_cc)
    learned_gap = jnp.mean(cc_final - cv_final)

    # Gap reduction
    reduction = 1.0 - learned_gap / jnp.maximum(mc_gap, 1e-15)
    return float(jnp.clip(reduction, 0.0, 1.0))
