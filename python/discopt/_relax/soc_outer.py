"""Polyhedral outer approximation of convex norm-ball rows (#2059).

A convex quadratic row ``x'Qx + c'x + k <= 0`` reaches an outer-approximation (OA)
master as tangent cuts, one dense row per separation. That approximation is only
accurate where a tangent was taken, and the HiGHS ``lp_nlp_bb`` master can add
rows only between trees, so a branch that moves the quadratic part away from the
separated points sees a loose relaxation. On portfol_classical050_1 that left the
master 2.2% from the optimum after 18-30 rebuilt trees (performance-plan 103.13).

When the linear part lives on the quadratic support, completing the square turns
the row into a *norm ball* ``||L x + v||_2 <= rho`` (``Q ~ L'L``), a second-order
cone. :cite:t:`BenTalNemirovski2001soc` give a lifted polyhedral approximation of
such a cone that is uniformly accurate *everywhere*, with size logarithmic in the
accuracy: a binary tower of 2-D cones ``r >= ||(a, b)||``, each replaced by ``nu``
rotation levels,

    xi_0 >= |a|,  eta_0 >= |b|,
    xi_j  = cos(t_j) xi_{j-1} + sin(t_j) eta_{j-1},
    eta_j >= |-sin(t_j) xi_{j-1} + cos(t_j) eta_{j-1}|,     t_j = pi / 2**(j+1),
    eta_nu <= tan(t_nu) xi_nu,      r := xi_nu,

with ``||(a, b)|| <= xi_nu <= ||(a, b)|| / cos(t_nu)`` on the projection
:cite:p:`Vielma2017` (who use it in exactly this role, for MI-SOCP).

**Validity (it is an OUTER approximation).** Every point of the ball extends to a
point of the polyhedron: put each leaf at its value, ``xi_0 = |a|``,
``eta_0 = |b|``, apply the rotations exactly (they preserve the norm), and every
internal node at the exact norm of its children; the root is then the exact
``||L x + v||``. The polyhedron therefore contains the feasible set of the row:
adding it to a master that is a relaxation keeps it a relaxation, so the master's
dual bound stays valid.

What makes ``||L x + v|| <= rho`` hold at every feasible ``x``:

* ``L'L <= Q`` in the PSD order, proved rather than assumed (:func:`_factor`): a
  diagonal ``Q`` with any negative entry is declined, and any other ``Q`` must be
  proved ``>= margin*I`` by ``rigorous_psd_shift``, its factor's residual bounded
  below ``margin / 2``;
* ``v`` solves ``2 L'v = c_S`` in least squares; its residual ``delta`` is
  charged to ``rho**2`` as ``sum_j |delta_j| max(|lb_j|, |ub_j|)``, and a row with a
  nonzero residual on an unbounded column is declined;
* a linear part OFF the quadratic support is not a ball: declined.

Declined as well: equalities, non-scalar rows, rows the exact extractor will not
read, supports with fewer than two terms or more than :data:`MAX_SUPPORT`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Rotation levels per 2-D cone. Measured on portfol_classical050_1 (one HiGHS
#: solve of the approximated MILP, performance-plan 103.13): 6 -> bound 1.0e-3 from
#: the optimum, 7 -> 2.1e-4, 8 -> 5.0e-5 (14.3 s), 10 -> 2e-6 (41.8 s).
LEVELS = 8

#: Supports wider than this are not factored (O(m^3) eigendecomposition).
MAX_SUPPORT = 500

#: Cap on leaves (terms) over all rows of one master; a ball that does not fit is
#: skipped. The block adds about ``20`` rows and ``10`` columns per leaf, and the
#: master's LP cost grows faster than that. Measured on the MINLPLib portfol family
#: at 120 s (performance-plan 103.13): 50 leaves -- classical050_1 certified
#: (uncertified OFF), robust050_34 40.0 -> 9.5 s; 100 leaves -- robust100_09
#: unchanged; 200 leaves -- robust200_03 unchanged, classical200_2 spent 102 s in
#: its first root (4,388 x 2,590 LP) and lost its dual bound.
MAX_LEAVES = 128

#: The master is built dense (``oa._build_master_milp_data``): a block whose rows
#: times the master's columns exceed this many entries is not built.
MAX_DENSE_ENTRIES = 25_000_000


def soc_outer_enabled() -> bool:
    """``DISCOPT_OA_SOC_OUTER``: add the Ben-Tal--Nemirovski outer approximation of
    every convex norm-ball row to the ``lp_nlp_bb`` HiGHS master (#2059).

    **Graduated (default ON, #2059); ``=0`` restores the tangent-only master.**
    Bound-changing, so CLAUDE.md section 5 regime 2. Evidence as re-taken on main
    with #2060's ``DISCOPT_PERSPECTIVE_ROW_DISAGG`` (performance-plan 103.15),
    which this block supersedes on the rows it covers:

    * panel over the 102 routed instances, 30 s, interleaved, every incumbent
      verified: 612 checks, 0 problems, certified unchanged (none gained or lost),
      total wall 434.1 -> 436.8 s; the block is built on 6 (ball_mk2_10/30,
      nvs10/11/12, portfol_classical050_1), with the same result on all but the
      last, whose bound at 30 s is looser (-0.097110 -> -0.097498: the #1066 guard
      hands over before its single tree ends);
    * the MINLPLib portfol family at the issue's 120 s, interleaved, incumbents
      verified, no bound past its reference: portfol_classical050_1 feasible ->
      **optimal, 52.3 s**; portfol_robust050_34 optimal 17.1 -> **8.4 s**;
      robust100_09 uncertified in both, bound 1.6e-5 looser; the 200-asset rows
      are past :data:`MAX_LEAVES`; the shortfall rows are not balls.
    """
    from discopt.solver_tuning import _env_flag

    return _env_flag("DISCOPT_OA_SOC_OUTER", default=True)


@dataclass
class NormBallRow:
    """``||L x + v|| <= rho``: leaves ``u_k = L_idx[k] . x + v[k]``."""

    row: int
    L_idx: list[np.ndarray]
    L_val: list[np.ndarray]
    v: np.ndarray
    rho: float


def _factor(Q: np.ndarray, margin: float) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Rows ``L_k`` (support-local) with ``L'L <= Q`` in the PSD order, or ``None``.

    The guarantee is rigorous, not "PSD to within roundoff" (#1679/#1682); a row
    it cannot prove is declined. Review of #2064: the first version dropped
    eigenvalues in ``[-margin, 0)`` uncharged, and on ``x1**2 + x2**2 - 1e-10 x3**2
    <= 1`` the ball cut off the feasible point ``(0.9, 0.9, 1e5)``.

    * **Diagonal ``Q``.** Any negative entry declines the row. A kept term is
      ``fl(sqrt(fl(q - margin)))**2 <= (q - margin)(1 + 4u) <= q`` because
      ``margin >= 4u q``; an entry ``q <= margin`` is dropped, contributing
      ``q x**2 >= 0``.
    * **Otherwise.** ``Q >= margin*I`` is proved with
      :func:`~discopt._relax.convexity.eigenvalue.rigorous_psd_shift` on
      ``Q - 2 margin I`` (shift at most ``margin / 2``, the diagonal subtraction's
      rounding being far below that), ``L`` is taken from ``Q - margin I``, and the
      residual ``T = Q - margin I - L'L`` is bounded in the 2-norm by its computed
      Frobenius norm plus the entrywise error bound ``gamma_{k+3}(|L'||L| + |Q| +
      margin)`` (Higham 2002, eq. 3.13). With ``||T||_2 <= margin / 2``,
      ``Q - L'L = T + margin I >= 0``. A singular or near-singular ``Q`` therefore
      declines.
    """
    from .convexity.eigenvalue import rigorous_psd_shift

    m = Q.shape[0]
    u = np.finfo(np.float64).eps / 2.0
    if not np.any(Q - np.diag(np.diag(Q))):
        q = np.diag(Q)
        if float(q.min()) < 0.0:
            return None
        return [
            (np.array([j], dtype=np.intp), np.array([math.sqrt(float(q[j]) - margin)]))
            for j in range(m)
            if float(q[j]) - margin > 0.0
        ]
    delta = rigorous_psd_shift(Q - 2.0 * margin * np.eye(m))
    if delta is None or delta > 0.5 * margin:
        return None
    evals, vecs = np.linalg.eigh(Q - margin * np.eye(m))
    if float(evals.min()) <= 0.0:
        return None
    L = (vecs * np.sqrt(evals)).T  # rows L_k; L'L ~ Q - margin I
    k = L.shape[0]
    gamma = (k + 3) * u / (1.0 - (k + 3) * u)
    resid = (Q - margin * np.eye(m)) - L.T @ L
    err = gamma * (np.abs(L.T) @ np.abs(L) + np.abs(Q) + margin)
    if float(np.linalg.norm(resid)) + float(np.linalg.norm(err)) > 0.5 * margin:
        return None
    rows = []
    for t in range(k):
        keep = np.flatnonzero(L[t] != 0.0)
        rows.append((keep.astype(np.intp), L[t, keep]))
    return rows


def norm_ball_rows(model, row_map, lb, ub) -> list[NormBallRow]:
    """Every scalar convex quadratic row of ``model`` that is a norm ball."""
    from .problem_classifier import _EXTRACTION_DECLINES, _extract_quadratic_terms

    n = sum(v.size for v in model._variables)
    lo = np.asarray(lb, dtype=np.float64)
    hi = np.asarray(ub, dtype=np.float64)
    out: list[NormBallRow] = []
    leaves = 0
    for start, stop, con in row_map:
        if stop - start != 1 or con.sense not in ("<=", ">="):
            continue
        try:
            hterms, c, k = _extract_quadratic_terms(con.body, model, n)
        except _EXTRACTION_DECLINES:
            continue
        nz = {(i, j): v for (i, j), v in hterms.items() if v != 0.0}
        if not nz:
            continue
        sign = 1.0 if con.sense == "<=" else -1.0
        S = np.array(sorted({i for i, _ in nz} | {j for _, j in nz}), dtype=np.intp)
        if S.size < 2 or S.size > MAX_SUPPORT:
            continue
        c_full = sign * np.asarray(c, dtype=np.float64)
        off = np.ones(n, dtype=bool)
        off[S] = False
        if np.any(c_full[off] != 0.0):
            continue  # a linear part off the support: not a ball
        pos = {int(j): a for a, j in enumerate(S)}
        H = np.zeros((S.size, S.size))
        for (i, j), v in nz.items():
            H[pos[i], pos[j]] = v
        scale = max(1.0, float(np.abs(H).max()))
        if not np.allclose(H, H.T, rtol=0.0, atol=1e-12 * scale):
            continue
        Q = 0.25 * sign * (H + H.T)  # body = 0.5 x'Hx + c'x + k  =>  Q = H / 2
        factor = _factor(Q, 1e-9 * max(1.0, float(np.abs(Q).max())))
        if factor is None or len(factor) < 2:
            continue
        L = np.zeros((len(factor), S.size))
        for t, (idx, val) in enumerate(factor):
            L[t, idx] = val
        c_S = c_full[S]
        if np.any(c_S != 0.0):
            v, *_ = np.linalg.lstsq(2.0 * L.T, c_S, rcond=None)
            delta = c_S - 2.0 * L.T @ v
            if np.any(np.abs(delta) > 1e-9 * max(1.0, float(np.abs(c_S).max()))):
                continue  # c_S is not in the range of Q: not a ball
            reach = np.maximum(np.abs(lo[S]), np.abs(hi[S]))
            hit = delta != 0.0
            if np.any(hit & ~(np.isfinite(reach) & (reach < 1e20))):
                continue
            charge = float(np.sum(np.abs(delta[hit]) * reach[hit]))
        else:
            v = np.zeros(len(factor))
            charge = 0.0
        rho2 = -sign * float(k) + float(v @ v) + charge
        if not np.isfinite(rho2) or rho2 <= 0.0:
            continue
        if leaves + len(factor) > MAX_LEAVES:
            continue
        leaves += len(factor)
        out.append(
            NormBallRow(
                row=int(start),
                L_idx=[S[idx] for idx, _ in factor],
                L_val=[val for _, val in factor],
                v=np.asarray(v, dtype=np.float64),
                rho=math.sqrt(rho2) * (1.0 + 1e-12),
            )
        )
    return out


@dataclass
class SocOuterBlock:
    """The lifted rows of every :class:`NormBallRow`, in block-local columns.

    ``ub_rows`` are ``(x_terms, block_terms, rhs)`` with ``x_terms`` and
    ``block_terms`` as ``{column: coefficient}``, meaning ``x . a + block . b <=
    rhs``; every block column is ``>= 0``. There are no equality rows: each
    ``xi_j`` is written out as its combination of ``xi_0, eta_0 .. eta_{j-1}``
    (at most ``levels + 1`` terms), and a cone's output is one column
    ``r >= xi_nu``, which the parent cone uses as its ``xi_0``/``eta_0``. That is
    still an outer approximation -- the extension of a true point puts ``r`` at
    ``xi_nu`` exactly, and a larger ``r`` only tightens the cones above it -- and
    it is as accurate, since the rotations never shrink ``||(xi_j, eta_j)||``.
    Measured on portfol_classical050_1 with HiGHS presolve off (which the #1634
    cross-solve uses): the equality form took 42.8 s where presolve on took
    14.3 s.
    """

    rows: list[NormBallRow]
    levels: int = LEVELS
    n_cols: int = 0
    ub_rows: list[tuple[dict[int, float], dict[int, float], float]] = field(default_factory=list)
    #: Per 2-D cone, for :meth:`start_values`: ``(a, b, xi0, eta_cols, r)`` with
    #: ``a``/``b`` operands (``("leaf", row, k)`` or ``("col", c)``), ``xi0`` the
    #: column standing for ``xi_0`` (the child's ``r`` or a new leaf column, and
    #: likewise ``eta_cols[0]`` for ``eta_0``), ``eta_cols`` the ``eta_0..eta_nu``
    #: columns and ``r`` the output column.
    _cones: list[tuple[tuple, tuple, int, list[int], int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for j, r in enumerate(self.rows):
            level: list[tuple] = [("leaf", j, k) for k in range(len(r.L_idx))]
            while len(level) > 1:
                nxt = []
                for i in range(0, len(level) - 1, 2):
                    nxt.append(("col", self._cone(level[i], level[i + 1])))
                if len(level) % 2:
                    nxt.append(level[-1])
                level = nxt
            root = level[0]
            assert root[0] == "col"
            self.ub_rows.append(({}, {root[1]: 1.0}, r.rho))

    def _new(self) -> int:
        self.n_cols += 1
        return self.n_cols - 1

    def _magnitude(self, op) -> int:
        """A column ``>= |op|``: the operand itself when it is a (nonnegative) column."""
        if op[0] == "col":
            return int(op[1])
        r = self.rows[op[1]]
        k = op[2]
        col = self._new()
        x_terms = dict(zip(r.L_idx[k].tolist(), r.L_val[k].tolist()))
        const = float(r.v[k])
        for sgn in (1.0, -1.0):
            self.ub_rows.append(
                ({c: sgn * v for c, v in x_terms.items()}, {col: -1.0}, -sgn * const)
            )
        return col

    def _cone(self, a, b) -> int:
        """New column ``r >= ||(a, b)||`` (outer approximation); returns it."""
        xi0 = self._magnitude(a)
        eta_cols = [self._magnitude(b)]
        xi: dict[int, float] = {xi0: 1.0}
        for j in range(1, self.levels + 1):
            th = math.pi / 2 ** (j + 1)
            cs, sn = math.cos(th), math.sin(th)
            eta_prev = eta_cols[-1]
            eta = self._new()
            # eta_j >= |-sin * xi_{j-1} + cos * eta_{j-1}|
            for sgn in (1.0, -1.0):
                row = {c: -sgn * sn * v for c, v in xi.items()}
                row[eta_prev] = row.get(eta_prev, 0.0) + sgn * cs
                row[eta] = row.get(eta, 0.0) - 1.0
                self.ub_rows.append(({}, row, 0.0))
            # xi_j = cos * xi_{j-1} + sin * eta_{j-1}
            nxt = {c: cs * v for c, v in xi.items()}
            nxt[eta_prev] = nxt.get(eta_prev, 0.0) + sn
            xi = nxt
            eta_cols.append(eta)
        tan_nu = math.tan(math.pi / 2 ** (self.levels + 1))
        # eta_nu <= tan * xi_nu
        row = {c: -tan_nu * v for c, v in xi.items()}
        row[eta_cols[-1]] = row.get(eta_cols[-1], 0.0) + 1.0
        self.ub_rows.append(({}, row, 0.0))
        # r >= xi_nu
        out = self._new()
        row = dict(xi)
        row[out] = row.get(out, 0.0) - 1.0
        self.ub_rows.append(({}, row, 0.0))
        self._cones.append((a, b, xi0, eta_cols, out))
        return out

    def start_values(self, x) -> np.ndarray:
        """The block's columns at ``x`` by the exact construction (module docstring)."""
        xv = np.asarray(x, dtype=np.float64).ravel()
        vals = np.zeros(self.n_cols, dtype=np.float64)

        def value(op) -> float:
            if op[0] == "col":
                return float(vals[op[1]])
            r = self.rows[op[1]]
            return float(r.L_val[op[2]] @ xv[r.L_idx[op[2]]] + r.v[op[2]])

        for a, b, xi0, eta_cols, out in self._cones:  # bottom-up: children first
            xi, eta = abs(value(a)), abs(value(b))
            vals[xi0], vals[eta_cols[0]] = xi, eta
            for j in range(1, self.levels + 1):
                th = math.pi / 2 ** (j + 1)
                cs, sn = math.cos(th), math.sin(th)
                xi, eta = cs * xi + sn * eta, abs(-sn * xi + cs * eta)
                vals[eta_cols[j]] = eta
            vals[out] = xi
        return vals
