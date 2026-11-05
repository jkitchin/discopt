"""Column-eliminating presolve of a native spatial-kernel spec (#2168).

The native spatial kernel (``discopt._rust.solve_spatial_tree_py``) receives a
:func:`~discopt._relax.spatial_producer.build_spatial_kernel_spec` spec: linear
``<=`` rows, lifted terms (``term_kind`` 0=bilinear, 1=monomial, 2=affine square,
3=sqrt, 4=affine power) and affine-form products (BLF, ``w = (A x + a0)(B x + b0)``).
:func:`presolve_spec` rewrites that spec into an equivalent one with fewer free
columns, the reductions SCIP's presolve applies and the kernel otherwise never
sees (#2142 / performance-plan sections 115-116: discopt's own tree certifies GDPlib
``cstr`` under the homogenized hull on SCIP's presolved model and not on its own).

Reductions, iterated to closure:

1. **Fixings.** The kernel's own root propagation (``propagate_spec_py``) on the
   global box; a column with ``lo == hi`` exactly is fixed. A pair of opposite
   support-1 rows fixes its column too.
2. **Fixed-column substitution.** A fixed column is moved into the right-hand side
   of every row and into the constant of every BLF side and term input:

   * a bilinear term (kind 0) with a fixed factor becomes the linear equality
     ``w = v * x`` (two rows); a BLF with a constant side likewise;
   * a kind 1-4 term with a fixed input fixes its output to ``f(v)``.

   A fixed column keeps its index, its box ``[v, v]`` and its objective
   coefficient, so nothing has to be reconstructed for it.
3. **Doubleton aggregation.** A pair of opposite parallel rows of support 2 is the
   equality ``a_p x_p + a_q x_q = b``; one column is eliminated as
   ``x_p := a x_q + c`` and substituted everywhere:

   * an affine input of a monomial becomes an affine square / affine power, an
     affine input of a kind 2/3/4 term composes into its affine map, a bilinear
     factor turns the term into a BLF, a BLF side absorbs it directly;
   * a term or BLF *output* is eliminated only when ``c == 0`` (a BLF / bilinear
     output with any ``a``, by rescaling one side; a kind 1-4 output only when
     ``a == 1``); the representative then carries the product;
   * the representative is chosen so that outputs, integers and objective columns
     survive; ``|a|`` outside ``[1e-3, 1e3]`` is refused; an integer column is
     eliminated only with ``a = +-1`` and integral ``c``, and its representative
     inherits the integrality (``x_q = (x_p - c) / a`` is then integral);
   * the eliminated column's box is intersected into its representative's
     (rounded outward).
4. **Row cleanup.** Empty rows are checked and dropped, parallel duplicates keep the
   minimum right-hand side.

**Not done: multi-aggregation** (the issue's optional stage 4). SCIP's eight
multi-aggregations on cstr eliminate a BLF *output* into its linear context
(``w := x62 - x67 - 0.99``), which the kernel cannot express: a BLF needs one output
column. The two expressible variants were built and measured on cstr (10k-node
bound, seeded kernel harness, stages 1-3 alone 2.28): eliminating a non-output
column into rows AND BLF sides 0.17 (a side's interval becomes a sum of column
intervals, loosening its envelopes); into rows only 1.74; restricted to implied-free
columns, as SCIP does, nothing qualifies. Removed in the PR that built it;
performance-plan section 119.

An eliminated column keeps its index (``n_cols`` is unchanged, so the kernel's and
the verifier's index space is stable) with box ``[0, 0]``, integrality 0, objective
0, no occurrence anywhere, and no OBBT candidacy. ``expand`` maps a reduced point
back by replaying the eliminations in reverse.

**Soundness.** Every substitution is an equality implied by globally valid rows or
by an exact fixing of the global box (an equality is a pair of rows whose normalised
coefficients and right-hand sides are exact negatives, compared as floats with no
tolerance), so the reduced problem equals the original
on that box: the reduced optimum is the original optimum, and the reduced spec's
relaxation at a box is a relaxation of the original's restricted to the same
equalities. The caller re-verifies every expanded point against the model
(``_native_kernel_verify_point``). Anything this module cannot express exactly is
left alone; an inconsistency (an empty row with a negative right-hand side, an
empty intersected box) declines the presolve (``None``) so the kernel sees the
original spec and proves it itself.
"""

from __future__ import annotations

import collections
import math
from typing import Callable, Optional

import numpy as np

#: The kernel's infinite-bound sentinel (``INF`` in the Rust LP layer).
_INF = 1e20
#: Aggregation multipliers outside ``[1/_MAX_SCALE, _MAX_SCALE]`` are refused.
_MAX_SCALE = 1e3
#: Closure iterations (each one propagates, substitutes and aggregates).
_MAX_ROUNDS = 25

#: Counters of the last :func:`presolve_spec` calls (CLAUDE.md section 6 probe).
PRESOLVE_STATS: "collections.Counter[str]" = collections.Counter()

_SPEC_ARRAY_KEYS = (
    "c",
    "integrality",
    "global_lo",
    "global_hi",
    "fixed_row_ptr",
    "fixed_cols",
    "fixed_coeffs",
    "fixed_rhs",
    "term_kind",
    "term_i",
    "term_j",
    "term_out",
    "term_p",
    "term_coeff",
    "term_cst",
    "blf_w",
    "blf_a_ptr",
    "blf_a_cols",
    "blf_a_coeffs",
    "blf_a_const",
    "blf_b_ptr",
    "blf_b_cols",
    "blf_b_coeffs",
    "blf_b_const",
    "obbt_candidates",
)


class _Infeasible(Exception):
    """The reduction found an inconsistency; the presolve declines."""


def _down(v: float) -> float:
    return v if abs(v) >= _INF else float(np.nextafter(v, -np.inf))


def _up(v: float) -> float:
    return v if abs(v) >= _INF else float(np.nextafter(v, np.inf))


def _term_value(kind: int, v: float, p: int, coeff: float, cst: float) -> float:
    """Value of a kind 1-4 term's output at the input value ``v``."""
    if kind == 1:
        return float(v**p)
    u = coeff * v + cst
    if kind == 2:
        return float(u * u)
    if kind == 3:
        if u < 0.0:
            raise _Infeasible("sqrt of a negative fixed argument")
        return math.sqrt(u)
    if kind == 4:
        if u == 0.0 and p < 0:
            raise _Infeasible("negative power of a zero fixed argument")
        return float(u**p)
    raise ValueError(f"term kind {kind} has no single input")


class _Problem:
    """A mutable, dictionary-form copy of a kernel spec."""

    def __init__(self, spec: dict):
        self.n = int(spec["n_cols"])
        self.n_orig = int(spec["n_orig"])
        self.c = np.array(spec["c"], dtype=np.float64)
        self.integ = np.array(spec["integrality"]).astype(bool)
        self.lo = np.array(spec["global_lo"], dtype=np.float64)
        self.hi = np.array(spec["global_hi"], dtype=np.float64)
        ptr = np.asarray(spec["fixed_row_ptr"]).astype(np.int64)
        cols = np.asarray(spec["fixed_cols"]).astype(np.int64)
        coefs = np.asarray(spec["fixed_coeffs"], dtype=np.float64)
        rhs = np.asarray(spec["fixed_rhs"], dtype=np.float64)
        self.rows: dict[int, list] = {}
        self._next_row = 0
        self.col_rows: dict[int, set] = collections.defaultdict(set)
        for r in range(len(rhs)):
            row: dict[int, float] = {}
            for q in range(int(ptr[r]), int(ptr[r + 1])):
                row[int(cols[q])] = row.get(int(cols[q]), 0.0) + float(coefs[q])
            self._add_row(row, float(rhs[r]))
        self.terms: dict[int, dict] = {}
        self._next_term = 0
        for t in range(len(spec["term_kind"])):
            self.terms[self._next_term] = dict(
                kind=int(spec["term_kind"][t]),
                i=int(spec["term_i"][t]),
                j=int(spec["term_j"][t]),
                out=int(spec["term_out"][t]),
                p=int(spec["term_p"][t]),
                coeff=float(spec["term_coeff"][t]),
                cst=float(spec["term_cst"][t]),
            )
            self._next_term += 1
        self.blfs: dict[int, dict] = {}
        self._next_blf = 0
        for t in range(len(spec["blf_w"])):
            sides = []
            for s in ("a", "b"):
                p0 = int(spec[f"blf_{s}_ptr"][t])
                p1 = int(spec[f"blf_{s}_ptr"][t + 1])
                form: dict[int, float] = {}
                for q in range(p0, p1):
                    k = int(spec[f"blf_{s}_cols"][q])
                    form[k] = form.get(k, 0.0) + float(spec[f"blf_{s}_coeffs"][q])
                sides.append([form, float(spec[f"blf_{s}_const"][t])])
            self._add_blf(sides[0], sides[1], int(spec["blf_w"][t]))
        self.obbt = [int(j) for j in np.asarray(spec["obbt_candidates"]).tolist()]
        self._obbt_set = set(self.obbt)
        self._pending: list[tuple[int, float]] = []  # queued fixings (see ``fix``)
        self._depth = 0  # substitution nesting; fixings drain at depth 0
        self.fixed: set[int] = set()  # columns with lo == hi, substituted out
        self.dead: set[int] = set()  # eliminated columns
        self.elim: list[tuple[int, dict[int, float], float]] = []
        self.stats: "collections.Counter[str]" = collections.Counter()
        self._dtypes = {k: np.asarray(spec[k]).dtype for k in _SPEC_ARRAY_KEYS}

    # -- bookkeeping -----------------------------------------------------------

    def _add_row(self, row: dict, rhs: float) -> None:
        row = {k: a for k, a in row.items() if a != 0.0}
        r = self._next_row
        self._next_row += 1
        self.rows[r] = [row, rhs]
        for k in row:
            self.col_rows[k].add(r)

    def _drop_row(self, r: int) -> None:
        row, _ = self.rows.pop(r)
        for k in row:
            self.col_rows[k].discard(r)

    def _add_eq(self, row: dict, rhs: float) -> None:
        self._add_row(dict(row), rhs)
        self._add_row({k: -a for k, a in row.items()}, -rhs)

    def _add_blf(self, a_side: list, b_side: list, w: int) -> None:
        self.blfs[self._next_blf] = dict(A=a_side, B=b_side, w=w)
        self._next_blf += 1

    def _uses(self, j: int) -> dict:
        """Where column ``j`` occurs: term/BLF inputs and outputs."""
        out: dict[str, list[int]] = dict(t_in=[], t_out=[], b_in=[], b_out=[])
        for t, d in self.terms.items():
            if d["i"] == j or (d["kind"] == 0 and d["j"] == j):
                out["t_in"].append(t)
            if d["out"] == j:
                out["t_out"].append(t)
        for t, d in self.blfs.items():
            if j in d["A"][0] or j in d["B"][0]:
                out["b_in"].append(t)
            if d["w"] == j:
                out["b_out"].append(t)
        return out

    # -- substitution ----------------------------------------------------------

    def _subst_rows(self, j: int, form: dict, b: float) -> None:
        for r in list(self.col_rows.get(j, ())):
            row, rhs = self.rows[r]
            a = row.pop(j)
            self.col_rows[j].discard(r)
            rhs -= a * b
            for k, s in form.items():
                v = row.get(k, 0.0) + a * s
                if v == 0.0:
                    if k in row:
                        del row[k]
                        self.col_rows[k].discard(r)
                else:
                    row[k] = v
                    self.col_rows[k].add(r)
            self.rows[r][1] = rhs
            self._clean_row(r)

    def _clean_row(self, r: int) -> None:
        """Drop cancellation residue a x with |a| max|x| <= 1e-12, relaxing the rhs."""
        row, rhs = self.rows[r]
        scale = max((abs(a) for a in row.values()), default=0.0)
        relaxed = False
        for k in [k for k, a in row.items() if abs(a) <= 1e-12 * scale]:
            a = row[k]
            bnd = max(abs(self.lo[k]), abs(self.hi[k]))
            if bnd >= _INF or abs(a) * bnd > 1e-12 * (1.0 + abs(rhs)):
                continue
            # a x + rest <= rhs implies rest <= rhs + max(-a x) over the box.
            rhs += max(-a * self.lo[k], -a * self.hi[k])
            relaxed = True
            del row[k]
            self.col_rows[k].discard(r)
            self.stats["residue_dropped"] += 1
        if relaxed:
            self.rows[r][1] = _up(rhs)

    @staticmethod
    def _subst_form(side: list, j: int, form: dict, b: float) -> None:
        f, c0 = side
        a = f.pop(j)
        side[1] = c0 + a * b
        for k, s in form.items():
            v = f.get(k, 0.0) + a * s
            if v == 0.0:
                f.pop(k, None)
            else:
                f[k] = v

    def substitute(self, j: int, form: dict, b: float, *, eliminate: bool) -> None:
        """Replace ``x_j`` by ``sum(form) + b`` in every row, form and term input.

        With ``eliminate`` the column is also removed from term/BLF outputs (the
        caller checked that this is expressible), from the objective and the box.

        A fixing this exposes (a term whose input became constant fixes its output)
        is queued and applied after this substitution finishes, never recursively
        from inside it: a recursive fixing pops terms and BLFs this loop still holds,
        and can add rows over ``x_j`` after its rows were rewritten (#2171 review,
        ``x * x**2`` with ``x`` fixed raised ``KeyError``). So the inputs are
        re-read until ``x_j`` is in none, and the rows go last.
        """
        self._depth += 1
        try:
            self._substitute(j, form, b, eliminate=eliminate)
        finally:
            self._depth -= 1
        if self._depth == 0:
            self._drain()

    def _substitute(self, j: int, form: dict, b: float, *, eliminate: bool) -> None:
        while True:
            uses = self._uses(j)
            if not uses["b_in"] and not uses["t_in"]:
                break
            for t in uses["b_in"]:
                d = self.blfs[t]
                for side in (d["A"], d["B"]):
                    if j in side[0]:
                        self._subst_form(side, j, form, b)
                if not d["A"][0] or not d["B"][0]:
                    # a constant side: w = const * (affine form) is linear
                    self.blfs.pop(t)
                    self._linearise_product(d["A"], d["B"], d["w"])
            for t in uses["t_in"]:
                self._subst_term_input(t, j, form, b)
        # last: the linearisations above may have added rows over x_j
        self._subst_rows(j, form, b)
        if eliminate:
            ((k, a),) = form.items()  # aggregation is into one representative
            for t in self._uses(j)["t_out"]:
                d = self.terms[t]
                if d["kind"] == 0:
                    # x_j = x_i x_m and x_j = a x_k  ->  x_k = (x_i / a) x_m
                    self.terms.pop(t)
                    self._add_blf([{d["i"]: 1.0 / a}, 0.0], [{d["j"]: 1.0}, 0.0], k)
                else:
                    assert a == 1.0 and b == 0.0
                    d["out"] = k
            for t in self._uses(j)["b_out"]:
                d = self.blfs[t]
                d["A"] = [{c: s / a for c, s in d["A"][0].items()}, d["A"][1] / a]
                d["w"] = k
            if self.c[j] != 0.0:
                assert b == 0.0 or not form
                for k2, s in form.items():
                    self.c[k2] += self.c[j] * s
                self.c[j] = 0.0
            self._intersect_box(j, form, b)
            self._clamp_sqrt_domain(k)
            if self.integ[j]:
                for k2 in form:
                    self.integ[k2] = True
            if j in self._obbt_set:
                # the representative inherits the eliminated column's OBBT candidacy
                for k2 in form:
                    if k2 not in self._obbt_set:
                        self._obbt_set.add(k2)
                        self.obbt.append(k2)
            self.lo[j] = self.hi[j] = 0.0
            self.integ[j] = False
            self.dead.add(j)
            self.elim.append((j, dict(form), b))
            assert not any(self._uses(j).values()) and not self.col_rows.get(j)

    def _subst_term_input(self, t: int, j: int, form: dict, b: float) -> None:
        d = self.terms[t]
        kind = d["kind"]
        if kind == 0:
            self.terms.pop(t)
            sides = []
            for x in (d["i"], d["j"]):
                sides.append([dict(form), b] if x == j else [{x: 1.0}, 0.0])
            if not sides[0][0] or not sides[1][0]:
                self._linearise_product(sides[0], sides[1], d["out"])
            else:
                self._add_blf(sides[0], sides[1], d["out"])
                self.stats["bilinear_to_blf"] += 1
            return
        if not form:
            self.terms.pop(t)
            self.fix(d["out"], _term_value(kind, b, d["p"], d["coeff"], d["cst"]))
            self.stats["term_output_fixed"] += 1
            return
        ((k, a),) = form.items()
        if kind == 1:
            if a == 1.0 and b == 0.0:
                d["i"] = k
            else:
                d.update(kind=2 if d["p"] == 2 else 4, i=k, coeff=a, cst=b)
                self.stats["monomial_to_affine"] += 1
        else:
            d.update(i=k, coeff=d["coeff"] * a, cst=d["coeff"] * b + d["cst"])

    def _linearise_product(self, a_side: list, b_side: list, w: int) -> None:
        """``w = A * B`` with a constant side becomes a linear equality (or a fixing)."""
        if not a_side[0] and not b_side[0]:
            self.fix(w, a_side[1] * b_side[1])
            return
        if not a_side[0]:
            a_side, b_side = b_side, a_side
        v = b_side[1]
        row = {k: -v * s for k, s in a_side[0].items()}
        row[w] = row.get(w, 0.0) + 1.0
        self._add_eq(row, v * a_side[1])
        self.stats["product_linearised"] += 1

    def _intersect_box(self, j: int, form: dict, b: float) -> None:
        lo, hi = self.lo[j], self.hi[j]
        if not form:
            if b < lo - 1e-9 * (1 + abs(lo)) or b > hi + 1e-9 * (1 + abs(hi)):
                raise _Infeasible(f"column {j} fixed outside its box")
            return
        if len(form) > 1:
            row = dict(form)
            if hi < _INF:
                self._add_row(dict(row), hi - b)
            if lo > -_INF:
                self._add_row({k: -s for k, s in row.items()}, b - lo)
            return
        ((k, a),) = form.items()
        lo_k = (
            -_INF
            if (lo <= -_INF if a > 0 else hi >= _INF)
            else _down(((lo if a > 0 else hi) - b) / a)
        )
        hi_k = (
            _INF if (hi >= _INF if a > 0 else lo <= -_INF) else _up(((hi if a > 0 else lo) - b) / a)
        )
        nlo, nhi = max(self.lo[k], lo_k), min(self.hi[k], hi_k)
        if self.integ[k] or self.integ[j]:
            nlo = math.ceil(nlo - 1e-9) if nlo > -_INF else nlo
            nhi = math.floor(nhi + 1e-9) if nhi < _INF else nhi
        if nlo > nhi:
            if nlo - nhi > 1e-9 * (1.0 + abs(nhi)):
                raise _Infeasible(f"aggregating column {j} empties the box of {k}")
            nlo = nhi = 0.5 * (nlo + nhi)
        self.lo[k], self.hi[k] = float(nlo), float(nhi)

    def _clamp_sqrt_domain(self, k: int) -> None:
        """Keep every sqrt argument on ``k``'s box nonnegative.

        ``_intersect_box`` rounds the representative's box outward, so a sqrt input
        ``u = coeff * x_k + cst`` that was ``>= 0`` on the eliminated column's box can
        dip below 0 by rounding at an end of ``k``'s. The kernel then emits no sqrt
        envelope rows at all for that box (``EnvTerm::push_rows``: an undefined base
        box gives the aux floor only) -- sound, but the envelope is silently lost.
        Any feasible point has ``u >= 0`` (the term's domain), so moving such an end
        inward until ``u >= 0`` holds exactly is valid. Only rounding-level dips are
        moved; a genuine one is the kernel's to handle as before (#2171 review).
        """
        for d in self.terms.values():
            if d["kind"] != 3 or d["i"] != k or d["coeff"] == 0.0:
                continue
            a, c = d["coeff"], d["cst"]
            end = "lo" if a > 0 else "hi"  # the end where u is smallest
            x = self.lo[k] if end == "lo" else self.hi[k]
            if abs(x) >= _INF:
                continue
            u = a * x + c
            if u >= 0.0 or u < -1e-12 * (1.0 + abs(c) + abs(a * x)):
                continue
            step = np.inf if end == "lo" else -np.inf
            for _ in range(64):
                x = float(np.nextafter(x, step))
                if a * x + c >= 0.0:
                    break
            else:
                continue
            if end == "lo":
                self.lo[k] = min(x, self.hi[k])
            else:
                self.hi[k] = max(x, self.lo[k])
            self.stats["sqrt_domain_clamped"] += 1

    def fix(self, j: int, v: float) -> None:
        """Fix column ``j`` to ``v`` and substitute it into rows, forms and inputs.

        Inside a substitution the fixing is queued (see :meth:`substitute`)."""
        self._pending.append((j, v))
        if self._depth == 0:
            self._drain()

    def _drain(self) -> None:
        while self._pending:
            j, v = self._pending.pop(0)
            self._fix_now(j, v)

    def _fix_now(self, j: int, v: float) -> None:
        if j in self.dead:
            raise _Infeasible(f"a fixing reached the eliminated column {j}")
        if j in self.fixed:
            if abs(self.lo[j] - v) > 1e-9 * (1.0 + abs(v)):
                raise _Infeasible(f"column {j} fixed to two values")
            return
        lo, hi = self.lo[j], self.hi[j]
        tol = 1e-9 * (1.0 + abs(v))
        if v < lo - tol or v > hi + tol:
            raise _Infeasible(f"column {j} fixed to {v} outside [{lo}, {hi}]")
        if self.integ[j]:
            if abs(v - round(v)) > 1e-9:
                raise _Infeasible(f"integer column {j} fixed to {v}")
            v = float(round(v))
        v = min(max(v, lo), hi)
        self.lo[j] = self.hi[j] = v
        self.fixed.add(j)
        self.stats["fixed"] += 1
        self.substitute(j, {}, v, eliminate=False)

    # -- detection -------------------------------------------------------------

    @staticmethod
    def _key(row: dict, sign: float) -> tuple:
        """The row's coefficients scaled by ``sign / |a_k0|`` (``k0`` its first
        column), exactly: no rounding, so two rows match only when their normalised
        coefficients are the same floats (#2171 review)."""
        k0 = min(row)
        s = sign / abs(row[k0])
        return tuple((k, a * s) for k, a in sorted(row.items()))

    def equalities(self, max_support: int) -> list:
        """Pairs of opposite parallel rows of support <= ``max_support``: equalities."""
        index: dict[tuple, int] = {}
        for r, (row, _) in self.rows.items():
            if 0 < len(row) <= max_support:
                index.setdefault(self._key(row, 1.0), r)
        out: list[tuple] = []
        seen: set[int] = set()
        for key, r in index.items():
            if r in seen:
                continue
            row, rhs = self.rows[r]
            r2 = index.get(self._key(row, -1.0))
            if r2 is None or r2 in seen or r2 == r:
                continue
            row2, rhs2 = self.rows[r2]
            k0 = min(row)
            s = abs(row2[k0]) / abs(row[k0])  # row2 == -s * row
            # exact: every coefficient and the rhs are -s times row's, so the pair is
            # the equality row.x == rhs, not a thin range
            if any(row2[k] != -s * a for k, a in row.items()) or rhs2 != -s * rhs:
                continue
            seen.update((r, r2))
            out.append((r, r2, dict(row), dict(row2), rhs))
        return out

    def cleanup_rows(self) -> None:
        """Drop empty rows (checking them) and parallel duplicates (min rhs kept)."""
        best: dict[tuple, tuple] = {}
        for r in list(self.rows):
            row, rhs = self.rows[r]
            if not row:
                if rhs < -1e-9 * (1.0 + abs(rhs)):
                    raise _Infeasible(f"row {r} reduced to 0 <= {rhs}")
                self._drop_row(r)
                self.stats["rows_empty_dropped"] += 1
                continue
            key = self._key(row, 1.0)
            scale = abs(row[min(row)])
            norm_rhs = rhs / scale
            prev = best.get(key)
            if prev is None:
                best[key] = (r, norm_rhs)
            elif norm_rhs < prev[1]:
                self._drop_row(prev[0])
                best[key] = (r, norm_rhs)
                self.stats["rows_duplicate_dropped"] += 1
            else:
                self._drop_row(r)
                self.stats["rows_duplicate_dropped"] += 1

    # -- aggregation -------------------------------------------------------------

    def _elim_cost(self, p: int, q: int, a: float, b: float) -> Optional[tuple]:
        """Cost of eliminating ``x_p := a x_q + b``, or ``None`` if not expressible."""
        if p in self.fixed or p in self.dead or q in self.dead:
            return None
        if not (1.0 / _MAX_SCALE <= abs(a) <= _MAX_SCALE):
            return None
        if self.integ[p] and not (abs(a) == 1.0 and b == round(b)):
            return None
        if self.c[p] != 0.0 and b != 0.0:
            return None
        uses = self._uses(p)
        if set(uses["t_out"]) & set(uses["t_in"]) or set(uses["b_out"]) & set(uses["b_in"]):
            return None  # a term whose output is its own input
        for t in uses["t_out"]:
            d = self.terms[t]
            if b != 0.0 or (d["kind"] != 0 and a != 1.0):
                return None
            if q in (d["i"], d["j"] if d["kind"] == 0 else -1):
                return None
        for t in uses["b_out"]:
            d = self.blfs[t]
            if b != 0.0 or q in d["A"][0] or q in d["B"][0]:
                return None
        for t in uses["t_in"]:
            d = self.terms[t]
            if d["out"] == q:
                return None
        for t in uses["b_in"]:
            if self.blfs[t]["w"] == q:
                return None
        is_out = bool(uses["t_out"] or uses["b_out"])
        return (int(is_out), int(self.integ[p]), int(self.c[p] != 0.0), -p)

    def aggregate(self) -> int:
        n_new = 0
        for r1, r2, row, row2, rhs in self.equalities(2):
            if r1 not in self.rows or r2 not in self.rows:
                continue
            if self.rows[r1][0] != row or self.rows[r2][0] != row2:
                continue  # an earlier reduction in this pass touched it
            if len(row) == 1:
                ((k, a),) = row.items()
                self._drop_row(r1)
                self._drop_row(r2)
                self.fix(k, rhs / a)
                self.stats["fixed_by_singleton_eq"] += 1
                n_new += 1
                continue
            (x1, a1), (x2, a2) = sorted(row.items())
            cands = []
            for p, ap, q, aq in ((x1, a1, x2, a2), (x2, a2, x1, a1)):
                a, b = -aq / ap, rhs / ap
                cost = self._elim_cost(p, q, a, b)
                if cost is not None:
                    cands.append((cost, p, q, a, b))
            if not cands:
                self.stats["doubleton_skipped"] += 1
                continue
            _, p, q, a, b = min(cands)
            self._drop_row(r1)
            self._drop_row(r2)
            self.substitute(p, {q: a}, b, eliminate=True)
            self.stats["aggregated"] += 1
            n_new += 1
        return n_new

    # -- spec I/O ----------------------------------------------------------------

    def to_spec(self) -> dict:
        n = self.n
        for j in self.dead:
            assert not self.col_rows.get(j), j
        rows = list(self.rows.values())
        ptr, cols, coefs, rhs = [0], [], [], []
        for row, b in rows:
            for k in sorted(row):
                cols.append(k)
                coefs.append(row[k])
            rhs.append(b)
            ptr.append(len(cols))
        terms = list(self.terms.values())
        blfs = list(self.blfs.values())
        spec: dict = {"n_cols": n, "n_orig": self.n_orig}
        arr = {
            "c": self.c,
            "integrality": self.integ.astype(np.int64),
            "global_lo": self.lo,
            "global_hi": self.hi,
            "fixed_row_ptr": ptr,
            "fixed_cols": cols,
            "fixed_coeffs": coefs,
            "fixed_rhs": rhs,
            "term_kind": [d["kind"] for d in terms],
            "term_i": [d["i"] for d in terms],
            "term_j": [d["j"] if d["kind"] == 0 else -1 for d in terms],
            "term_out": [d["out"] for d in terms],
            "term_p": [d["p"] for d in terms],
            "term_coeff": [d["coeff"] for d in terms],
            "term_cst": [d["cst"] for d in terms],
            "blf_w": [d["w"] for d in blfs],
            "obbt_candidates": [j for j in self.obbt if j not in self.dead],
        }
        for s, key in (("a", "A"), ("b", "B")):
            p_, c_, co_, cst_ = [0], [], [], []
            for d in blfs:
                form, c0 = d[key]
                for k in sorted(form):
                    c_.append(k)
                    co_.append(form[k])
                cst_.append(c0)
                p_.append(len(c_))
            arr.update(
                {
                    f"blf_{s}_ptr": p_,
                    f"blf_{s}_cols": c_,
                    f"blf_{s}_coeffs": co_,
                    f"blf_{s}_const": cst_,
                }
            )
        for k, v in arr.items():
            spec[k] = np.asarray(v, dtype=self._dtypes[k])
        for t in terms:
            assert t["i"] not in self.dead and t["out"] not in self.dead
            assert t["kind"] != 0 or t["j"] not in self.dead
        for d in blfs:
            assert d["w"] not in self.dead
            assert not (set(d["A"][0]) | set(d["B"][0])) & self.dead
        return spec


def _propagate(spec: dict) -> Optional[tuple[np.ndarray, np.ndarray]]:
    from discopt import _rust, _timing

    kw = dict(spec)
    lo = np.array(spec["global_lo"], dtype=np.float64)
    hi = np.array(spec["global_hi"], dtype=np.float64)
    with _timing.charge("rust"):
        live, plo, phi = _rust.propagate_spec_py(**kw, lo=lo, hi=hi, cutoff=None, rounds=15)
    if not live:
        return None
    return np.asarray(plo, dtype=np.float64), np.asarray(phi, dtype=np.float64)


def presolve_spec(
    spec: dict,
) -> Optional[tuple[dict, Callable[[np.ndarray], np.ndarray], dict]]:
    """Reduce a kernel spec; ``(reduced_spec, expand, info)`` or ``None``.

    ``spec`` holds the kernel arrays only (no ``meta_*`` keys). ``None`` means the
    presolve eliminated nothing or found an inconsistency; the caller then hands
    the kernel the original spec. ``expand(x)`` maps a reduced-space point (length
    ``>= n_cols``; columns past ``n_cols`` are the LP's envelope columns and are
    returned untouched) to the original space. ``info`` holds the counters, the
    eliminated columns and ``compress`` (original point -> reduced point).
    """
    prob = _Problem(spec)
    try:
        for _ in range(_MAX_ROUNDS):
            before = (len(prob.fixed), len(prob.dead), len(prob.rows))
            box = _propagate(prob.to_spec())
            if box is None:
                prob.stats["declined_root_infeasible"] += 1
                PRESOLVE_STATS.update(prob.stats)
                return None
            plo, phi = box
            for j in range(prob.n):
                if j in prob.fixed or j in prob.dead:
                    continue
                if plo[j] == phi[j]:
                    prob.stats["fixed_by_fbbt"] += 1
                    prob.fix(j, float(plo[j]))
            while prob.aggregate():
                pass
            prob.cleanup_rows()
            if (len(prob.fixed), len(prob.dead), len(prob.rows)) == before:
                break
    except _Infeasible:
        prob.stats["declined_inconsistent"] += 1
        PRESOLVE_STATS.update(prob.stats)
        return None
    PRESOLVE_STATS.update(prob.stats)
    PRESOLVE_STATS["calls"] += 1
    if not prob.dead and not prob.fixed:
        return None
    reduced = prob.to_spec()
    n = prob.n
    elim = list(prob.elim)
    dead = sorted(prob.dead)

    def expand(x) -> np.ndarray:
        y = np.array(x, dtype=np.float64)
        if y.shape[0] < n:
            raise ValueError(f"expand needs a point of length >= n_cols={n}, got {y.shape[0]}")
        for j, form, b in reversed(elim):
            y[j] = b + sum(a * y[k] for k, a in form.items())
        return y

    def compress(x) -> np.ndarray:
        y = np.array(x, dtype=np.float64)
        y[dead] = 0.0
        return y

    info = dict(
        stats=dict(prob.stats),
        n_eliminated=len(prob.dead),
        n_fixed=len(prob.fixed),
        eliminated=dead,
        compress=compress,
        n_rows_before=len(spec["fixed_rhs"]),
        n_rows_after=len(reduced["fixed_rhs"]),
    )
    return reduced, expand, info
