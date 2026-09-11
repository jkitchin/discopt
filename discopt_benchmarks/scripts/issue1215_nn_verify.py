"""Emit + evaluate a matrix of NN embeddings for one formulation, for A/B diffing.

Usage: issue1215_nn_verify.py <full_space|reduced_space|relu_bigm> <outfile>

Capture the file, change the emitter, capture again, `cmp`. Written for #1215's
NN-emitter vectorisation, and the reason it exists is that the first attempt at
that verification used `.nl` + LP only and would have missed a GAMS-only change.

Two regimes, because byte-identical export is not evidence of solver neutrality
(§55): the .nl/LP TEXT, and the ARENA's residuals at sampled points. The second
replaces a solve-status comparison on purpose -- reduced_space fuses each layer's
affine and activation into one nonlinear body, so its smooth-activation solves run
past a 25 s limit and every status would read `time_limit`, making the comparison
vacuous. Residuals at fixed pseudo-random points test the same property (same
mathematics in the arena) in milliseconds and without B&B.

Four arms, because each covers something the others cannot:

  NL    the solver-facing format -- carries NO row names
  LP    carries names, but cannot represent a nonlinear activation at all
  GAMS  carries names AND accepts nonlinear bodies; the only arm that can show a
        row-NAME change on a nonlinear net
  EVAL  the ARENA's objective at fixed pseudo-random points, because
        byte-identical export is not evidence of solver neutrality (§55) -- the
        matmul form of a layer exports identically and loses the certificate

Known-weak column: `maxcon`/`maxbnd` come back NaN on these models in both arms,
so they carry no signal; the `obj` column is the equivalence check. Left in
rather than deleted so a future reader does not mistake NaN for a finding.

Guards, each earned: truncate up front plus a trailing sentinel so a crash cannot
leave a stale file for `cmp` to call identical (that produced a false "IDENTICAL"
twice); a minimum capture count so the matrix cannot silently shrink; exceptions
recorded as text so one unsupported combination does not hide the rest, and so
that a refusal appearing on BOTH sides is visibly zero coverage rather than a
pass; per-case progress to stderr (§10).

Expected refusals, which are correct behaviour and not failures: LP/MPS cannot
represent a nonlinear activation; `.nl` refuses `max()` (RELU in reduced_space)
as needing the DNLP model type. Compare the per-arm identical/differing counts,
not the raw file, so those do not read as coverage.
"""

import pathlib
import sys

import numpy as np
from discopt._rust import model_to_repr
from discopt.export import to_gams, to_lp, to_nl
from discopt.modeling import Model
from discopt.nn import DenseLayer, NetworkDefinition, OffsetScaling
from discopt.nn.formulations.full_space import FullSpaceFormulation
from discopt.nn.formulations.reduced_space import ReducedSpaceFormulation
from discopt.nn.formulations.relu_bigm import ReluBigMFormulation

METHOD, OUT = sys.argv[1], sys.argv[2]
pathlib.Path(OUT).write_text("")
FORMS = {
    "full_space": FullSpaceFormulation,
    "reduced_space": ReducedSpaceFormulation,
    "relu_bigm": ReluBigMFormulation,
}
ACTS = {
    "full_space": ("tanh", "sigmoid", "softplus", "linear"),
    "reduced_space": ("tanh", "sigmoid", "softplus", "relu"),
    "relu_bigm": ("relu",),
}


def net(sizes, act):
    rng = np.random.default_rng(7)
    return NetworkDefinition(
        layers=[
            DenseLayer(
                weights=rng.normal(0, 0.35, (sizes[i], sizes[i + 1])),
                biases=rng.normal(0, 0.2, sizes[i + 1]),
                activation=act if i < len(sizes) - 2 else "linear",
            )
            for i in range(len(sizes) - 1)
        ],
        input_bounds=(np.full(sizes[0], -1.0), np.full(sizes[0], 1.0)),
    )


def scaling(nin, nout):
    return OffsetScaling(
        x_offset=np.full(nin, 0.25),
        x_factor=np.full(nin, 2.0),
        y_offset=np.full(nout, -0.5),
        y_factor=np.full(nout, 3.0),
    )


def cap(label, fn):
    try:
        return f"--- {label} ---\n{fn()}"
    except Exception as e:
        return f"--- {label} ---\n<{type(e).__name__}: {e}>"


parts, n = [], 0
for sizes in ([3, 5, 2], [4, 8, 4, 1], [3, 4, 1], [6, 10, 10, 3], [2, 3, 3, 3]):
    for act in ACTS[METHOD]:
        for scaled in (False, True):
            tag = f"{METHOD} {sizes} {act} scaled={scaled}"
            print(f"# {tag}", file=sys.stderr, flush=True)

            def build(sizes=sizes, act=act, scaled=scaled):
                m = Model("nn")
                x = m.continuous("x", shape=(sizes[0],), lb=-1.0, ub=1.0)
                sc = scaling(sizes[0], sizes[-1]) if scaled else None
                inp, o = FORMS[METHOD](m, net(sizes, act), "pred", sc).build()
                m.subject_to(inp == x, name="link")
                m.minimize(o[0])
                return m

            parts.append(cap(f"{tag} NL", lambda b=build: to_nl(b())))
            parts.append(cap(f"{tag} LP", lambda b=build: to_lp(b())))
            # GAMS is the only format here that carries ROW NAMES *and* accepts a
            # nonlinear body, so it is the arm that proves a family name expands
            # to the same names the per-element loop wrote. LP covers names only
            # for linear nets; `.nl` carries none at all.
            parts.append(cap(f"{tag} GAMS", lambda b=build: to_gams(b())))

            def residuals(build=build):
                m = build()
                rep = model_to_repr(m, getattr(m, "_builder", None))
                rng = np.random.default_rng(11)
                out = [f"n_vars={rep.n_vars} n_cons={rep.n_constraints}"]
                for t in range(4):
                    pt = rng.uniform(-0.9, 0.9, rep.n_vars)
                    obj, con, bnd = rep.evaluate_point(list(pt))
                    out.append(f"  pt{t}: obj={obj:.12f} maxcon={con:.12f} maxbnd={bnd:.12f}")
                return "\n".join(out)

            parts.append(cap(f"{tag} EVAL", residuals))
            n += 4
parts.append(f"=== END {n} captures ===")
pathlib.Path(OUT).write_text("\n".join(parts))
assert n >= 30, f"only {n} captures"
errs = sum(1 for p in parts if "\n<" in p)
print(f"{METHOD}: {n} captures, {errs} raised")
