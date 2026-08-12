"""Reader for the QPLIB native instance format.

QPLIB (https://qplib.zib.de/, CC-BY 4.0) is a library of 453 quadratic
programming instances -- 390 of them *nonconvex* -- with externally certified
reference objective values. It is the complement to the MINLPLib corpus: where
MINLPLib exercises general factorable nonlinearity, QPLIB concentrates on
nonconvex quadratic objectives and constraints, which is exactly the structure
the McCormick / alphaBB relaxation layer is built to bound.

A QPLIB instance is

.. math::

    \\text{sense} \\quad \\tfrac{1}{2} x^T Q^0 x + b^0 x + q^0

    \\text{s.t.} \\quad c^i_l \\le \\tfrac{1}{2} x^T Q^i x + b^i x \\le c^i_u

    l \\le x \\le u

with each variable continuous, binary, or general integer. The library contains
no semi-continuous variables and no SOS constraints (verified over all 453
instances against ``instancedata.csv``), so those cases are absent by
construction rather than silently dropped.

Format notes
------------
The ``.qplib`` file is line-oriented plain text with ``#`` comments, but its
*section layout is conditional on the three-character problem type code* on
line 2 (``OVC`` -- objective / variables / constraints):

* ``O = L``  -- linear objective: the objective-quadratic block is **absent**.
* ``V = B``  -- all binary: the variable bound and variable type blocks are
  **absent** (bounds are implicitly ``[0, 1]``).
* ``V = C``/``I`` -- uniformly continuous / integer: the variable *type* block
  is absent (bounds are present).
* ``V = M``/``G`` -- mixed: both bound and type blocks present.
* ``C = N``/``B`` -- no constraints: the ``ncons`` line itself is **absent**,
  along with every constraint block and the constraint-dual starting point.
* ``C = L``  -- linear constraints only: the constraint-quadratic block is absent.

Getting this conditioning wrong does not raise -- it silently shifts the token
stream and yields a well-formed but *wrong* model. Every field this reader
produces is therefore cross-checked against the library's own
``instancedata.csv`` in ``test_qplib_reader.py``.

Quadratic storage: only the lower triangle (``i >= j``, 1-based) is stored, and
**every stored entry carries a uniform factor of one half** --

.. math::

    \\tfrac{1}{2} x^T Q x = \\tfrac{1}{2} \\sum_{i \\ge j} v_{ij}\\, x_i x_j

That is, a stored off-diagonal entry :math:`v_{ij}` represents the *combined*
:math:`Q_{ij} + Q_{ji}` rather than one triangle half, so it is **not** doubled
on symmetrization. The tempting reading -- halve the diagonal, take the
off-diagonal at face value -- overstates every off-diagonal contribution by 2x
and yields objective values roughly twice the published ones.

This convention was not taken from the documentation (which states only the
:math:`\\tfrac{1}{2}` in the model form); it was *fitted* from the data. A
least-squares solve for the two coefficients over 40 all-continuous instances
(where :math:`x^2 \\ne x`, so the diagonal is identifiable) recovers
``(0.5, 0.5)`` at rank 2 with a maximum residual of 4.5e-12. The same fit is
re-run as a unit test so a silent convention change cannot pass.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "QplibInstance",
    "read_qplib",
    "read_solution",
    "to_model",
    "from_qplib",
    "read_solu",
]

# Variable classes exposed by this reader.
_CONTINUOUS = 0
_BINARY = 1
_INTEGER = 2

# Codes as they appear in the file's variable-type block. Only these two occur
# anywhere in the library (verified over every 'M'/'G' instance: 97103 records,
# codes {0, 1} only). The block marks *integrality*; binary-vs-general-integer
# is decided by the bounds, not by a third code.
_CODE_CONTINUOUS = 0
_CODE_INTEGER = 1

# The reader treats any |value| at or above this as infinite. QPLIB writes the
# IEEE double max (1.79769313486232E+308) as its infinity marker, but an
# instance may declare its own on the "value for infinity" line, which is what
# is actually honored below.
_DEFAULT_INFINITY = 1.79769313486232e308

#: Uniform factor applied to every stored quadratic entry, objective and
#: constraint alike. See the module docstring: this is 0.5 for the diagonal
#: *and* the off-diagonal, because a stored off-diagonal entry already
#: represents Q_ij + Q_ji.
_QUAD_SCALE = 0.5


@dataclass
class QplibInstance:
    """Raw contents of a ``.qplib`` file, before any modeling-layer conversion.

    Indices are 0-based here even though the file stores them 1-based.
    """

    name: str
    probtype: str
    sense: str  # "minimize" or "maximize"
    n_vars: int
    n_cons: int
    infinity: float

    #: Lower-triangle entries ``(i, j, v)`` with ``i >= j`` of the objective Q.
    obj_quad: list[tuple[int, int, float]] = field(default_factory=list)
    obj_lin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    obj_const: float = 0.0

    #: Lower-triangle entries ``(con, i, j, v)`` with ``i >= j``.
    con_quad: list[tuple[int, int, int, float]] = field(default_factory=list)
    #: Entries ``(con, i, v)``.
    con_lin: list[tuple[int, int, float]] = field(default_factory=list)

    lhs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rhs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ub: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: One of ``_CONTINUOUS`` / ``_BINARY`` / ``_INTEGER`` per variable.
    vtype: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    x0: np.ndarray = field(default_factory=lambda: np.zeros(0))

    var_names: dict[int, str] = field(default_factory=dict)
    con_names: dict[int, str] = field(default_factory=dict)

    @property
    def objective_type(self) -> str:
        return self.probtype[0]

    @property
    def variable_type(self) -> str:
        return self.probtype[1]

    @property
    def constraint_type(self) -> str:
        return self.probtype[2]

    @property
    def n_binary(self) -> int:
        """Integer variables whose domain lies in ``{0, 1}``.

        This is a *derived* classification and may disagree with the
        ``nbinvars``/``nintvars`` split in the library's ``instancedata.csv``,
        which reflects how the source model declared each variable -- something
        the ``.qplib`` file does not record. QPLIB_3525, for instance, reports
        ``nbinvars=0`` while carrying 831 integer variables bounded ``[0, 1]``.
        The two agree on the *total* number of integral variables in every
        instance, and that total is what the tests assert. Nothing downstream
        depends on the split: an integer variable on ``[0, 1]`` and a binary
        variable are the same feasible set.
        """
        return int(np.count_nonzero(self.vtype == _BINARY))

    @property
    def n_integer(self) -> int:
        """General integer variables (integral, domain not inside ``{0, 1}``)."""
        return int(np.count_nonzero(self.vtype == _INTEGER))

    @property
    def n_integral(self) -> int:
        """All integer-constrained variables. Matches ``nbinvars + nintvars``."""
        return int(np.count_nonzero(self.vtype != _CONTINUOUS))

    @property
    def is_maximize(self) -> bool:
        return self.sense == "maximize"

    def variable_name(self, i: int) -> str:
        """Name of variable *i*; QPLIB's default naming is 1-based ``x<k>``."""
        return self.var_names.get(i, f"x{i + 1}")

    def evaluate_objective(self, x: np.ndarray) -> float:
        """Objective value at *x*, in the instance's own sense.

        Applies the uniform ``1/2`` quadratic convention documented in the
        module docstring.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_vars,):
            raise ValueError(f"x has shape {x.shape}, expected ({self.n_vars},)")
        total = float(self.obj_const) + float(self.obj_lin @ x)
        for i, j, v in self.obj_quad:
            total += _QUAD_SCALE * v * x[i] * x[j]
        return total

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Row activities ``1/2 xᵀQⁱx + bⁱx`` for every constraint."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_vars,):
            raise ValueError(f"x has shape {x.shape}, expected ({self.n_vars},)")
        act = np.zeros(self.n_cons)
        for c, i, v in self.con_lin:
            act[c] += v * x[i]
        for c, i, j, v in self.con_quad:
            act[c] += _QUAD_SCALE * v * x[i] * x[j]
        return act

    def max_violation(self, x: np.ndarray, *, tol: float = 0.0) -> float:
        """Largest absolute violation of any row or bound at *x*.

        Used to independently feasibility-verify a reference point rather than
        trusting the published objective value alone.
        """
        x = np.asarray(x, dtype=float)
        viol = 0.0
        lo = np.where(self.lb <= -self.infinity, -np.inf, self.lb)
        hi = np.where(self.ub >= self.infinity, np.inf, self.ub)
        viol = max(viol, float(np.max(np.maximum(lo - x, 0.0), initial=0.0)))
        viol = max(viol, float(np.max(np.maximum(x - hi, 0.0), initial=0.0)))
        if self.n_cons:
            act = self.evaluate_constraints(x)
            clo = np.where(self.lhs <= -self.infinity, -np.inf, self.lhs)
            chi = np.where(self.rhs >= self.infinity, np.inf, self.rhs)
            viol = max(viol, float(np.max(np.maximum(clo - act, 0.0), initial=0.0)))
            viol = max(viol, float(np.max(np.maximum(act - chi, 0.0), initial=0.0)))
        integral = self.vtype != _CONTINUOUS
        if integral.any():
            xi = x[integral]
            viol = max(viol, float(np.max(np.abs(xi - np.round(xi)), initial=0.0)))
        return viol if viol > tol else 0.0


class _Cursor:
    """Line cursor over a comment-stripped ``.qplib`` file.

    Raises on exhaustion rather than returning a sentinel: a truncated or
    misparsed file must fail loudly, never degrade into a plausible model.
    """

    __slots__ = ("lines", "pos", "path")

    def __init__(self, lines: list[str], path: str) -> None:
        self.lines = lines
        self.pos = 0
        self.path = path

    def next_line(self) -> str:
        if self.pos >= len(self.lines):
            raise ValueError(f"{self.path}: unexpected end of file after {self.pos} records")
        line = self.lines[self.pos]
        self.pos += 1
        return line

    def next_tokens(self, expected: int) -> list[str]:
        line = self.next_line()
        parts = line.split()
        if len(parts) != expected:
            raise ValueError(
                f"{self.path}: line {self.pos} expected {expected} fields, got "
                f"{len(parts)}: {line!r}"
            )
        return parts

    def next_int(self) -> int:
        line = self.next_line()
        try:
            return int(line.split()[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{self.path}: line {self.pos} is not an integer: {line!r}") from exc

    def next_float(self) -> float:
        line = self.next_line()
        try:
            return float(line.split()[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{self.path}: line {self.pos} is not a float: {line!r}") from exc

    def at_end(self) -> bool:
        return self.pos >= len(self.lines)


def _strip(raw: str) -> list[str]:
    out = []
    for line in raw.splitlines():
        hash_at = line.find("#")
        if hash_at >= 0:
            line = line[:hash_at]
        line = line.strip()
        if line:
            out.append(line)
    return out


def _sparse_vector(cur: _Cursor, n: int, default: float, label: str) -> np.ndarray:
    """Read a ``default value`` / ``count`` / ``index value``* block."""
    vec = np.full(n, default, dtype=float)
    count = cur.next_int()
    for _ in range(count):
        idx_s, val_s = cur.next_tokens(2)
        idx = int(idx_s) - 1
        if not 0 <= idx < n:
            raise ValueError(f"{cur.path}: {label} index {idx + 1} out of range 1..{n}")
        vec[idx] = float(val_s)
    return vec


def read_qplib(path: str) -> QplibInstance:
    """Parse a ``.qplib`` file into a :class:`QplibInstance`.

    Parameters
    ----------
    path : str
        Path to a ``.qplib`` file.

    Raises
    ------
    ValueError
        On any structural inconsistency -- a short file, an out-of-range index,
        a malformed record, or trailing content. The reader never guesses.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        cur = _Cursor(_strip(fh.read()), os.path.basename(path))

    name = cur.next_line()
    probtype = cur.next_line().strip().upper()
    if len(probtype) != 3:
        raise ValueError(f"{cur.path}: problem type {probtype!r} is not 3 characters")
    # Indexed rather than unpacked: mypy disallows unpacking a str, and the
    # length was just checked above.
    otype, vtype_code, ctype = probtype[0], probtype[1], probtype[2]

    sense_raw = cur.next_line().strip().lower()
    if sense_raw not in ("minimize", "maximize"):
        raise ValueError(f"{cur.path}: unknown objective sense {sense_raw!r}")

    n_vars = cur.next_int()
    if n_vars <= 0:
        raise ValueError(f"{cur.path}: nonpositive variable count {n_vars}")

    # 'N' (unconstrained) and 'B' (bounds only) omit the constraint count line
    # entirely -- reading it anyway would consume the next section's header.
    has_cons = ctype not in ("N", "B")
    n_cons = cur.next_int() if has_cons else 0
    if n_cons < 0:
        raise ValueError(f"{cur.path}: negative constraint count {n_cons}")

    inst = QplibInstance(
        name=name,
        probtype=probtype,
        sense=sense_raw,
        n_vars=n_vars,
        n_cons=n_cons,
        infinity=_DEFAULT_INFINITY,
    )

    # ---- objective quadratic (absent for a linear objective) ----
    if otype != "L":
        n_obj_quad = cur.next_int()
        for _ in range(n_obj_quad):
            i_s, j_s, v_s = cur.next_tokens(3)
            i, j = int(i_s) - 1, int(j_s) - 1
            if not (0 <= i < n_vars and 0 <= j < n_vars):
                raise ValueError(f"{cur.path}: objective quadratic index out of range")
            if j > i:
                raise ValueError(
                    f"{cur.path}: objective quadratic entry ({i + 1},{j + 1}) is upper "
                    "triangular; QPLIB stores the lower triangle only"
                )
            inst.obj_quad.append((i, j, float(v_s)))

    # ---- objective linear + constant ----
    obj_lin_default = cur.next_float()
    inst.obj_lin = _sparse_vector(cur, n_vars, obj_lin_default, "objective linear")
    inst.obj_const = cur.next_float()

    # ---- constraints ----
    if has_cons:
        if ctype != "L":  # 'L' has no quadratic constraint block
            n_con_quad = cur.next_int()
            for _ in range(n_con_quad):
                c_s, i_s, j_s, v_s = cur.next_tokens(4)
                c, i, j = int(c_s) - 1, int(i_s) - 1, int(j_s) - 1
                if not 0 <= c < n_cons:
                    raise ValueError(f"{cur.path}: constraint index {c + 1} out of range")
                if not (0 <= i < n_vars and 0 <= j < n_vars):
                    raise ValueError(f"{cur.path}: constraint quadratic index out of range")
                if j > i:
                    raise ValueError(
                        f"{cur.path}: constraint {c + 1} quadratic entry is upper triangular"
                    )
                inst.con_quad.append((c, i, j, float(v_s)))

        n_con_lin = cur.next_int()
        for _ in range(n_con_lin):
            c_s, i_s, v_s = cur.next_tokens(3)
            c, i = int(c_s) - 1, int(i_s) - 1
            if not 0 <= c < n_cons:
                raise ValueError(f"{cur.path}: constraint index {c + 1} out of range")
            if not 0 <= i < n_vars:
                raise ValueError(f"{cur.path}: constraint linear index out of range")
            inst.con_lin.append((c, i, float(v_s)))

    inst.infinity = abs(cur.next_float())

    if has_cons:
        lhs_default = cur.next_float()
        inst.lhs = _sparse_vector(cur, n_cons, lhs_default, "lhs")
        rhs_default = cur.next_float()
        inst.rhs = _sparse_vector(cur, n_cons, rhs_default, "rhs")
    else:
        inst.lhs = np.zeros(0)
        inst.rhs = np.zeros(0)

    # ---- variable bounds / types ----
    if vtype_code == "B":
        # All binary: bounds are implicit and no bound or type block is written.
        inst.lb = np.zeros(n_vars)
        inst.ub = np.ones(n_vars)
        is_integer = np.ones(n_vars, dtype=bool)
    else:
        lb_default = cur.next_float()
        inst.lb = _sparse_vector(cur, n_vars, lb_default, "lower bound")
        ub_default = cur.next_float()
        inst.ub = _sparse_vector(cur, n_vars, ub_default, "upper bound")

        if vtype_code in ("M", "G"):
            # The block stores *integrality* only (0 = continuous, 1 = integer).
            # It does not distinguish binary from general integer -- that is a
            # property of the bounds, resolved below.
            type_default = cur.next_int()
            codes = np.full(n_vars, type_default, dtype=int)
            count = cur.next_int()
            for _ in range(count):
                i_s, t_s = cur.next_tokens(2)
                i, t = int(i_s) - 1, int(t_s)
                if not 0 <= i < n_vars:
                    raise ValueError(f"{cur.path}: variable type index out of range")
                if t not in (_CODE_CONTINUOUS, _CODE_INTEGER):
                    raise ValueError(
                        f"{cur.path}: unknown variable type code {t} "
                        f"(expected {_CODE_CONTINUOUS} or {_CODE_INTEGER})"
                    )
                codes[i] = t
            is_integer = codes == _CODE_INTEGER
        elif vtype_code == "C":
            is_integer = np.zeros(n_vars, dtype=bool)
        elif vtype_code == "I":
            is_integer = np.ones(n_vars, dtype=bool)
        else:
            raise ValueError(f"{cur.path}: unknown variable class {vtype_code!r} in {probtype}")

    # Binary is not stored: it is an integer variable whose domain is contained
    # in {0, 1}. Containment, not bound equality -- an integer fixed at 1
    # (lb = ub = 1) counts as binary too, which is how instancedata.csv's
    # nbinvars/nintvars split and the b/i/x prefixes in the .sol files behave.
    binary = is_integer & (inst.lb >= 0.0) & (inst.ub <= 1.0)
    inst.vtype = np.where(binary, _BINARY, np.where(is_integer, _INTEGER, _CONTINUOUS)).astype(int)

    # ---- starting point (read to keep the cursor aligned; x0 is retained) ----
    x0_default = cur.next_float()
    inst.x0 = _sparse_vector(cur, n_vars, x0_default, "starting point")
    if has_cons:
        dual_default = cur.next_float()
        _sparse_vector(cur, n_cons, dual_default, "constraint dual")
    bdual_default = cur.next_float()
    _sparse_vector(cur, n_vars, bdual_default, "bound dual")

    # ---- names ----
    n_var_names = cur.next_int()
    for _ in range(n_var_names):
        idx_s, nm = cur.next_tokens(2)
        inst.var_names[int(idx_s) - 1] = nm
    n_con_names = cur.next_int()
    for _ in range(n_con_names):
        idx_s, nm = cur.next_tokens(2)
        inst.con_names[int(idx_s) - 1] = nm

    if not cur.at_end():
        raise ValueError(
            f"{cur.path}: {len(cur.lines) - cur.pos} unconsumed record(s) after the name "
            "sections -- the section layout was misread"
        )
    return inst


#: ``.sol`` records are named ``<prefix><k>`` where ``prefix`` is ``b`` for a
#: binary variable and ``x`` otherwise, and ``k`` is a **GAMS** variable number
#: that counts ``objvar`` as variable 1. The QPLIB variable at 0-based index
#: ``i`` therefore appears as ``k = i + 2``.
#:
#: This offset is not documented; it was established empirically over the whole
#: library (offset 1: 0 out-of-range and 0 prefix/type mismatches across ~376k
#: records; offset 0: 127 violations). Both invariants are re-checked on every
#: read below, so a wrong offset fails loudly instead of silently scrambling the
#: point -- which would turn every downstream feasibility check into a no-op.
_SOL_INDEX_OFFSET = 2

_SOL_NAME_RE = re.compile(r"^([A-Za-z_]+)(\d+)$")

#: GAMS names solution records ``x``/``b``/``i`` by variable class. Only the
#: integral-vs-continuous split is checked (see read_solution).
_CLASS_NAME = {_CONTINUOUS: "continuous", _BINARY: "binary", _INTEGER: "integer"}


def read_solution(path: str, inst: QplibInstance) -> tuple[np.ndarray, Optional[float]]:
    """Read a QPLIB ``.sol`` reference point.

    The file lists ``objvar`` plus the *nonzero* entries by variable name, so
    absent variables take value zero.

    Returns
    -------
    (x, objvar)
        ``x`` is the dense point; ``objvar`` is the published objective value
        from the file, or ``None`` if the file did not carry one.

    Raises
    ------
    ValueError
        If a record's name is malformed, its index falls outside the variable
        range, or its ``b``/``x`` prefix disagrees with the variable's type in
        the ``.qplib`` file.
    """
    base = os.path.basename(path)
    x = np.zeros(inst.n_vars)
    objvar: Optional[float] = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            key, val = parts
            if key == "objvar":
                objvar = float(val)
                continue
            m = _SOL_NAME_RE.match(key)
            if m is None:
                raise ValueError(f"{base}: malformed solution record name {key!r}")
            prefix, num = m.group(1), int(m.group(2))
            idx = num - _SOL_INDEX_OFFSET
            if not 0 <= idx < inst.n_vars:
                raise ValueError(
                    f"{base}: solution name {key!r} maps to index {idx}, outside "
                    f"0..{inst.n_vars - 1}"
                )
            # Only integrality is cross-checkable: GAMS distinguishes 'b' from
            # 'i' by the *source model's* declaration, which the .qplib file
            # does not record (see the note on n_binary). Integral-vs-continuous
            # is exact -- 0 violations over 686531 records -- and is enough to
            # catch a misaligned index.
            integral_prefix = prefix in ("b", "i")
            is_integral = bool(inst.vtype[idx] != _CONTINUOUS)
            if integral_prefix != is_integral:
                raise ValueError(
                    f"{base}: solution name {key!r} has prefix {prefix!r} but variable "
                    f"{idx} is {_CLASS_NAME[int(inst.vtype[idx])]}"
                )
            x[idx] = float(val)
    return x, objvar


def read_solu(path: str) -> dict[str, float]:
    """Parse a ``qplib.solu`` oracle file into ``{instance: best objective}``.

    Same ``=best= NAME VALUE`` layout as ``minlplib.solu``.
    """
    out: dict[str, float] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[0] in ("=best=", "=opt="):
                out[parts[1]] = float(parts[2])
    return out


def to_model(inst: QplibInstance):
    """Build a :class:`discopt.modeling.Model` from a parsed instance.

    Two-sided rows ``c_l <= a(x) <= c_u`` become an equality when the sides
    coincide, a single inequality when one side is infinite, and two rows
    otherwise. A row with both sides infinite is dropped as vacuous.
    """
    import discopt.modeling as dm

    m = dm.Model(inst.name)
    inf = inst.infinity

    xs = []
    for i in range(inst.n_vars):
        nm = inst.variable_name(i)
        lo, hi = float(inst.lb[i]), float(inst.ub[i])
        if inst.vtype[i] == _BINARY:
            xs.append(m.binary(nm))
        elif inst.vtype[i] == _INTEGER:
            xs.append(m.integer(nm, lb=lo, ub=hi))
        else:
            xs.append(
                m.continuous(
                    nm,
                    lb=-9.999e19 if lo <= -inf else lo,
                    ub=9.999e19 if hi >= inf else hi,
                )
            )

    obj_terms = [_QUAD_SCALE * v * xs[i] * xs[j] for i, j, v in inst.obj_quad]
    obj_terms += [
        float(inst.obj_lin[i]) * xs[i] for i in range(inst.n_vars) if inst.obj_lin[i] != 0.0
    ]
    expr = dm.sum(obj_terms) if obj_terms else None
    if expr is None:
        # A pure-constant objective still needs a well-formed expression.
        expr = 0.0 * xs[0]
    if inst.obj_const:
        expr = expr + float(inst.obj_const)
    if inst.is_maximize:
        m.maximize(expr)
    else:
        m.minimize(expr)

    rows: list[list] = [[] for _ in range(inst.n_cons)]
    for c, i, j, v in inst.con_quad:
        rows[c].append(_QUAD_SCALE * v * xs[i] * xs[j])
    for c, i, v in inst.con_lin:
        rows[c].append(v * xs[i])

    for c in range(inst.n_cons):
        lo, hi = float(inst.lhs[c]), float(inst.rhs[c])
        lo_free, hi_free = lo <= -inf, hi >= inf
        if lo_free and hi_free:
            continue
        if not rows[c]:
            # An empty row is a constant restriction; only infeasible constants
            # matter, and silently dropping a violated one would be unsound.
            if (not lo_free and lo > 0.0) or (not hi_free and hi < 0.0):
                raise ValueError(
                    f"{inst.name}: constraint {c + 1} is empty but its bounds "
                    f"[{lo}, {hi}] exclude zero -- the instance is trivially infeasible"
                )
            continue
        body = dm.sum(rows[c])
        nm = inst.con_names.get(c, f"c{c + 1}")
        if not lo_free and not hi_free and lo == hi:
            m.subject_to(body == lo, name=nm)
        else:
            if not hi_free:
                m.subject_to(body <= hi, name=f"{nm}_ub" if not lo_free else nm)
            if not lo_free:
                m.subject_to(body >= lo, name=f"{nm}_lb" if not hi_free else nm)
    return m


def from_qplib(path: str):
    """Read a ``.qplib`` file directly into a :class:`discopt.modeling.Model`."""
    return to_model(read_qplib(path))
