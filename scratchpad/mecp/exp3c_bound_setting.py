"""EXPERIMENT 3c -- retest the two uncertified surrogate routes with the
mccormick_bounds='lp' setting the solver itself recommended.

Experiment 3b's reduced-space run printed:
  "McCormick 'nlp' objective bound is not a valid dual bound for nonconvex
   models (issue #120); falling back to the alphaBB underestimator. Use
   mccormick_bounds='lp' for a valid spatial relaxation on models with
   continuous variables."
so reporting "the reduced-space route does not certify" without trying that
setting would be reporting a default, not a capability.
"""

import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import discopt.modeling as dm  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mecp_models as M  # noqa: E402
from discopt._relax.mcbox import MCBox  # noqa: E402
from discopt.nn import add_predictor  # noqa: E402
from discopt.nn.network import Activation, DenseLayer, NetworkDefinition  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

NDIM, TL = 3, 300.0
tw = M.TwoWellParams(n=NDIM)
B = np.array(tw.bounds(), float)
E_REF = 1.558620  # certified by the factorable route (exp2/exp3b)
CHECKS = 0


def mexp(x):
    return x.exp() if isinstance(x, MCBox) else jnp.exp(x)


def show(tag, r, el):
    print(
        f"  {tag:<34s} status={r.status:<11s} obj={_n(r.objective)} bound={_n(r.bound)} "
        f"gap={_g(r.gap)} cert={str(r.gap_certified):<5s} nodes={r.node_count:<6d} t={el:7.2f}s",
        flush=True,
    )


def _n(v):
    return "     None" if v is None else f"{v:10.6f}"


def _g(v):
    return "    None" if v is None else f"{v:8.1e}"


print("=" * 100)
print("EXPERIMENT 3c -- mccormick_bounds='lp' on the routes that did not certify")
print("=" * 100)

# --- reduced-space dm.custom, lp bounds -----------------------------------
for mb in ("auto", "lp"):

    def w1s(*a):
        return tw.states(list(a), exp=mexp)[0]

    def w2s(*a):
        return tw.states(list(a), exp=mexp)[1]

    m = dm.Model(f"custom_mcbox_{mb}")
    q = [m.continuous(f"q{i}", lb=B[i, 0], ub=B[i, 1]) for i in range(NDIM)]
    c1, c2 = dm.custom(w1s, name="W1"), dm.custom(w2s, name="W2")
    m.minimize(c1(*q))
    m.subject_to(c1(*q) - c2(*q) == 0)
    t0 = time.time()
    r = m.solve(time_limit=TL, mccormick_bounds=mb)
    show(f"custom/mcbox  bounds={mb}", r, time.time() - t0)
    CHECKS += 1
    if r.gap_certified and r.bound is not None:
        CHECKS += 1
        assert r.bound <= E_REF + 1e-4, f"SOUNDNESS: bound {r.bound} > {E_REF}"

# --- discopt.nn surrogate, lp bounds --------------------------------------
rng = np.random.default_rng(0)
X = rng.uniform(B[:, 0], B[:, 1], size=(4000, NDIM))
W1t, W2t = tw.states([X[:, i] for i in range(NDIM)], exp=np.exp)
lo, hi = B[:, 0], B[:, 1]
Xs = 2 * (X - lo) / (hi - lo) - 1


def fit(y, hidden, seed):
    y = np.asarray(y, float)
    ym, ys = y.mean(), y.std()
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="tanh",
        max_iter=4000,
        tol=1e-9,
        random_state=seed,
        learning_rate_init=3e-3,
    )
    mlp.fit(Xs, (y - ym) / ys)
    acts = [Activation.TANH] * (len(mlp.coefs_) - 1) + [Activation.LINEAR]
    layers = [
        DenseLayer(weights=np.asarray(W), biases=np.asarray(b), activation=a)
        for W, b, a in zip(mlp.coefs_, mlp.intercepts_, acts)
    ]
    net = NetworkDefinition(layers=layers, input_bounds=(-np.ones(NDIM), np.ones(NDIM)))
    rmse = float(np.sqrt(np.mean((mlp.predict(Xs) * ys + ym - y) ** 2)))
    return net, ym, ys, rmse


# also try a much smaller net: relaxation tightness should depend on width
for hidden in ((12, 12), (5,)):
    n1, m1, s1, e1 = fit(W1t, hidden, 0)
    n2, m2, s2, e2 = fit(W2t, hidden, 1)
    print(f"\n  net {hidden}: train RMSE W1={e1:.4f} W2={e2:.4f}")
    for mb in ("auto", "lp"):
        m = dm.Model(f"nn_{hidden}_{mb}")
        z = m.continuous("z", shape=(NDIM,), lb=-1.0, ub=1.0)
        y1, _ = add_predictor(m, z, n1, method="reduced_space", prefix="w1")
        y2, _ = add_predictor(m, z, n2, method="reduced_space", prefix="w2")
        E1 = y1[0] * s1 + m1
        E2 = y2[0] * s2 + m2
        m.minimize(E1)
        m.subject_to(E1 - E2 == 0)
        t0 = time.time()
        r = m.solve(time_limit=TL, mccormick_bounds=mb)
        show(f"nn{hidden} bounds={mb}", r, time.time() - t0)
        CHECKS += 1
        if r.x is not None:
            zv = np.asarray(r.x["z"], float).ravel()[:NDIM]
            xv = lo + (zv + 1) * (hi - lo) / 2
            w1, w2 = tw.states(list(xv), exp=np.exp)
            print(
                f"     -> true W1={float(w1):.6f} (true MECP {E_REF:.6f}), "
                f"|W1-W2|={abs(float(w1 - w2)):.2e}"
            )

print(f"\nEXECUTED CHECKS: {CHECKS}")
if CHECKS == 0:
    sys.exit(2)
