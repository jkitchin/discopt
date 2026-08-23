"""EXPERIMENT 3 -- the realistic ab-initio workflow: MECP on a *learned*
two-state surrogate, solved globally.

A real MECP problem is a black box: the two state energies come out of an
electronic-structure code, so there is no algebraic expression for discopt to
relax.  The practical bridge is a surrogate: fit W1 and W2 (or a diabatic
matrix) to reference points, then optimize the surrogate globally.  This
experiment tests whether discopt can actually do that, three ways:

  (A) ``discopt.nn`` -- embed two trained feedforward networks (tanh, smooth,
      as a PES fit should be) as algebraic constraints and solve
      ``min W1_nn s.t. W1_nn - W2_nn == 0`` globally.

  (B) ``dm.custom`` reduced-space -- write the surrogate as an opaque
      JAX-traceable callable.  If it traces through discopt's MCBox type the
      solver certifies globally while branching only on the geometry degrees
      of freedom, which is the scaling behaviour MECP needs.

  (C) ``solver="direct"`` -- genuinely opaque body (what you would get by
      calling a quantum-chemistry program).  Derivative-free global *search*,
      no certificate.  Included to measure what is on offer when no
      surrogate is available at all.

The reference "electronic structure code" here is the analytic TwoWell model,
so the *true* seam is known and every surrogate answer can be scored against
it: this measures both the optimizer and the surrogate error separately.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import discopt.modeling as dm  # noqa: E402
import mecp_models as M  # noqa: E402

CHECKS = 0
NOTES: list[str] = []
NDIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3

tw = M.TwoWellParams(n=NDIM)
BNDS = np.array(tw.bounds(), dtype=float)


def true_states(x):
    return tw.states(list(np.asarray(x, float)), exp=np.exp)


def note(msg):
    NOTES.append(msg)
    print(f"   NOTE: {msg}")


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        note(f"FAIL: {msg}")
    return cond


def score(tag, x):
    """Score a candidate MECP geometry against the TRUE surfaces."""
    x = np.asarray(x, float)
    w1, w2 = true_states(x)
    print(f"      {tag}: true W1={float(w1):.6f}  true gap|W1-W2|={abs(float(w1 - w2)):.3e}")
    return float(w1), abs(float(w1 - w2))


# reference: the true global MECP of the model
if NDIM == 2:
    E_TRUE, X_TRUE, _, _ = M.seam_oracle_2d(tw, n_grid=1601)
    kind = "grid(exact)"
else:
    E_TRUE, X_TRUE, npj = M.seam_oracle_sampled(tw, n_samples=300_000, n_refine=500, seed=3)
    kind = f"sampled({npj})"
print(f"reference MECP ({kind}): E={E_TRUE:.6f} at {np.round(X_TRUE, 4)}")
check(np.isfinite(E_TRUE), "reference MECP not finite")

# ==========================================================================
# Fit the surrogate: two small tanh networks on a Latin-hypercube-ish sample
# ==========================================================================
print(f"\n[fit] training two tanh surrogates for W1, W2 on n={NDIM} coordinates")
rng = np.random.default_rng(0)
N_TRAIN = 4000
Xtr = rng.uniform(BNDS[:, 0], BNDS[:, 1], size=(N_TRAIN, NDIM))
W1tr, W2tr = tw.states([Xtr[:, i] for i in range(NDIM)], exp=np.exp)
W1tr = np.asarray(W1tr, float)
W2tr = np.asarray(W2tr, float)

try:
    from sklearn.neural_network import MLPRegressor

    HAVE_SK = True
except ImportError:
    HAVE_SK = False
    note("sklearn unavailable -- fitting with a hand-rolled least-squares tanh net instead")


def fit_net(y, hidden=(12, 12), seed=0):
    """Return (predict_fn, NetworkDefinition-compatible weight list)."""
    from discopt.nn.network import Activation, DenseLayer, NetworkDefinition

    if HAVE_SK:
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="tanh",
            max_iter=4000,
            tol=1e-9,
            random_state=seed,
            learning_rate_init=3e-3,
        )
        # scale inputs to [-1,1] and centre the target: PES fits need it
        lo, hi = BNDS[:, 0], BNDS[:, 1]
        Xs = 2 * (Xtr - lo) / (hi - lo) - 1
        ym, ys = y.mean(), y.std()
        mlp.fit(Xs, (y - ym) / ys)
        layers = []
        acts = [Activation.TANH] * len(mlp.coefs_[:-1]) + [Activation.LINEAR]
        for Wm, b, a in zip(mlp.coefs_, mlp.intercepts_, acts):
            layers.append(DenseLayer(weights=np.asarray(Wm), biases=np.asarray(b), activation=a))
        net = NetworkDefinition(
            layers=layers,
            input_bounds=(-np.ones(NDIM), np.ones(NDIM)),
        )

        def predict(X):
            X = np.atleast_2d(np.asarray(X, float))
            Xs = 2 * (X - lo) / (hi - lo) - 1
            return mlp.predict(Xs) * ys + ym

        return predict, net, (lo, hi, ym, ys)
    raise SystemExit("sklearn required for experiment 3A")


t0 = time.time()
pred1, net1, sc1 = fit_net(W1tr, seed=0)
pred2, net2, sc2 = fit_net(W2tr, seed=1)
print(f"      fitted in {time.time() - t0:.1f}s")
Xte = rng.uniform(BNDS[:, 0], BNDS[:, 1], size=(2000, NDIM))
W1te, W2te = tw.states([Xte[:, i] for i in range(NDIM)], exp=np.exp)
r1 = float(np.sqrt(np.mean((pred1(Xte) - np.asarray(W1te)) ** 2)))
r2 = float(np.sqrt(np.mean((pred2(Xte) - np.asarray(W2te)) ** 2)))
print(f"      surrogate RMSE: W1={r1:.4f}  W2={r2:.4f}  (energy units)")
check(r1 < 0.5 and r2 < 0.5, f"surrogate fit too poor to be meaningful (RMSE {r1:.3f}/{r2:.3f})")

# ==========================================================================
print("\n[A] discopt.nn: embed both networks, solve min W1 s.t. W1-W2 == 0")
# ==========================================================================
try:
    from discopt.nn import add_predictor

    m = dm.Model("nn_surrogate_mecp")
    # scaled inputs live in [-1,1]; the geometry is recovered afterwards
    z = m.continuous("z", shape=(NDIM,), lb=-1.0, ub=1.0)
    y1, f1 = add_predictor(m, z, net1, method="reduced_space", prefix="w1")
    y2, f2 = add_predictor(m, z, net2, method="reduced_space", prefix="w2")
    lo1, hi1, ym1, ys1 = sc1
    lo2, hi2, ym2, ys2 = sc2
    W1e = y1[0] * ys1 + ym1
    W2e = y2[0] * ys2 + ym2
    m.minimize(W1e)
    m.subject_to(W1e - W2e == 0)
    t0 = time.time()
    r = m.solve(time_limit=600.0)
    el = time.time() - t0
    print(
        f"   status={r.status} obj={r.objective} bound={r.bound} "
        f"gap={r.gap} cert={r.gap_certified} nodes={r.node_count} t={el:.1f}s"
    )
    check(r.status in ("optimal", "feasible"), "nn-surrogate MECP did not solve")
    if r.x is not None:
        zv = np.asarray(r.x["z"], float).ravel()[:NDIM]
        xv = lo1 + (zv + 1) * (hi1 - lo1) / 2
        print(f"   geometry = {np.round(xv, 5)}")
        e_s, gap_s = score("surrogate-optimal point on TRUE surfaces", xv)
        # The surrogate answer should be within a few RMSE of the true MECP.
        check(
            abs(e_s - E_TRUE) < 8 * max(r1, r2) + 0.3,
            f"surrogate MECP energy {e_s:.4f} far from true {E_TRUE:.4f}",
        )
except Exception as exc:
    print(f"   RAISED: {type(exc).__name__}: {exc}")
    traceback.print_exc(limit=4)
    note(f"discopt.nn route raised {type(exc).__name__}: {exc}")
    CHECKS += 1

# ==========================================================================
print("\n[B] dm.custom reduced-space: opaque callable, branching on geometry only")
print("    B1 uses a raw jnp.exp (should NOT trace through MCBox -> local only)")
print("    B2 uses an MCBox-dispatching exp (should certify, DOF-only branching)")
# ==========================================================================
import jax.numpy as jnp  # noqa: E402
from discopt._relax.mcbox import MCBox  # noqa: E402


def mexp(x):
    """Dispatching exp: MCBox for the relaxation path, jnp for the value path."""
    return x.exp() if isinstance(x, MCBox) else jnp.exp(x)


for label, expfn in (("B1 raw jnp.exp", jnp.exp), ("B2 MCBox-dispatch", mexp)):
    try:

        def w1_fn(q, _e=expfn):
            return tw.states([q[i] for i in range(NDIM)], exp=_e)[0]

        def w2_fn(q, _e=expfn):
            return tw.states([q[i] for i in range(NDIM)], exp=_e)[1]

        m = dm.Model(f"custom_mecp_{label.split()[0]}")
        q = m.continuous("q", shape=(NDIM,), lb=BNDS[:, 0], ub=BNDS[:, 1])
        cw1 = dm.custom(w1_fn, name="W1")
        cw2 = dm.custom(w2_fn, name="W2")
        m.minimize(cw1(q))
        m.subject_to(cw1(q) - cw2(q) == 0)
        t0 = time.time()
        r = m.solve(time_limit=600.0)
        el = time.time() - t0
        print(
            f"   [{label}] status={r.status} obj={r.objective} bound={r.bound} "
            f"gap={r.gap} cert={r.gap_certified} nodes={r.node_count} t={el:.1f}s"
        )
        CHECKS += 1
        if r.x is not None:
            xv = np.asarray(r.x["q"], float).ravel()[:NDIM]
            score(f"{label} optimum", xv)
            if r.gap_certified:
                check(
                    r.bound is None or r.bound <= E_TRUE + 1e-3,
                    f"SOUNDNESS: certified bound {r.bound} > true MECP {E_TRUE}",
                )
    except Exception as exc:
        print(f"   [{label}] RAISED: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=4)
        note(f"dm.custom {label} raised {type(exc).__name__}: {exc}")
        CHECKS += 1

# ==========================================================================
print('\n[C] solver="direct": fully opaque body, derivative-free, NO certificate')
print("    (posed as the Levine penalty objective, since DIRECT needs it unconstrained)")
# ==========================================================================
try:
    import jax.numpy as jnp

    SIGMA, ALPHA = 8.0, 0.01

    def pen_fn(q):
        w1, w2 = tw.states([q[i] for i in range(NDIM)], exp=jnp.exp)
        d = w2 - w1
        return 0.5 * (w1 + w2) + SIGMA * d * d / (jnp.abs(d) + ALPHA)

    m = dm.Model("direct_penalty_mecp")
    q = m.continuous("q", shape=(NDIM,), lb=BNDS[:, 0], ub=BNDS[:, 1])
    m.minimize(dm.custom(pen_fn, name="F")(q))
    t0 = time.time()
    r = m.solve(solver="direct", time_limit=120.0)
    el = time.time() - t0
    print(
        f"   status={r.status} obj={r.objective} bound={r.bound} cert={r.gap_certified} t={el:.1f}s"
    )
    CHECKS += 1
    check(not r.gap_certified, "DIRECT reported a certificate -- it must not")
    if r.x is not None:
        xv = np.asarray(r.x["q"], float).ravel()[:NDIM]
        e_c, gap_c = score("DIRECT optimum", xv)
except Exception as exc:
    print(f"   RAISED: {type(exc).__name__}: {exc}")
    traceback.print_exc(limit=4)
    note(f'solver="direct" route raised {type(exc).__name__}: {exc}')
    CHECKS += 1

# ==========================================================================
print("\n" + "=" * 78)
print(f"EXECUTED CHECKS: {CHECKS}")
print(f"NOTES: {len(NOTES)}")
for s in NOTES:
    print(f"  - {s}")
if CHECKS == 0:
    print("PROBE FIRED NOTHING")
    sys.exit(2)
