"""Parse MINLPLib instancedata.csv and score solver results against it.

The instancedata.csv file is published at https://www.minlplib.org/instancedata.csv
and shipped into the cache by ``discopt_benchmarks.scripts.fetch_minlplib``.

Two responsibilities live here:

1. **Parse** the CSV into ``InstanceMeta`` records, schema-tolerant against
   column renames or extra fields.

2. **Score** a :class:`SolveResult` against the reference: classify the outcome
   as one of ``optimal_proven`` | ``feasible_only`` | ``incorrect`` | ``timeout``
   | ``error`` and surface a ``best_known_objective`` when the reference declares
   one as proven.

The scoring is intentionally separate from the metrics module so the
classification rules live next to the field semantics they depend on.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from benchmarks.metrics import SolveResult, SolveStatus


# Column names we look for; we accept several aliases since MINLPLib's CSV
# layout has changed over the years.
_NAME_KEYS = ("name", "Instance", "instance", "Name")
_PROBTYPE_KEYS = ("probtype", "ProbType", "type")
_NVARS_KEYS = ("nvars", "NVars", "n_vars")
_NBINVARS_KEYS = ("nbinvars", "NBinVars", "n_binvars")
_NINTVARS_KEYS = ("nintvars", "NIntVars", "n_intvars")
_NCONS_KEYS = ("ncons", "NCons", "n_cons", "nlincons")
_OBJSENSE_KEYS = ("objsense", "ObjSense", "objective_sense")
_CONVEX_KEYS = ("convex", "Convex", "isconvex")
_PRIMAL_KEYS = ("primalbound", "primal_bound", "PrimalBound", "best_known_objective")
_DUAL_KEYS = ("dualbound", "dual_bound", "DualBound")
_PROVEN_KEYS = ("provenoptimal", "proven_optimal", "ProvenOptimal", "isoptimal")


def _first(row: dict, keys: Iterable[str], default: str = "") -> str:
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            return str(row[k]).strip()
    return default


def _to_int(s: str, default: int = 0) -> int:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default


def _to_float(s: str) -> float | None:
    if s is None:
        return None
    t = str(s).strip()
    if not t or t.lower() in {"nan", "none", "inf", "-inf", "1e+20", "-1e+20"}:
        return None
    try:
        return float(t)
    except (TypeError, ValueError):
        return None


def _to_bool(s: str) -> bool | None:
    t = str(s).strip().lower()
    if t in {"t", "true", "1", "yes", "y"}:
        return True
    if t in {"f", "false", "0", "no", "n"}:
        return False
    return None


@dataclass
class InstanceMeta:
    """Reference metadata for a single MINLPLib instance."""

    name: str
    probtype: str = "unknown"          # LP, MILP, NLP, MINLP, QP, QCQP, MIQP, MIQCP, ...
    n_vars: int = 0
    n_binvars: int = 0
    n_intvars: int = 0
    n_constraints: int = 0
    objsense: str = "min"              # "min" | "max"
    is_convex: bool | None = None
    primal_bound: float | None = None  # best-known feasible objective
    dual_bound: float | None = None    # best-known bound (LB for min)
    proven_optimal: bool = False       # true iff a solver has proven this optimal
    raw: dict = field(default_factory=dict)  # original row, for forward-compat queries

    @property
    def category_bucket(self) -> str:
        """Coarse category bucket for reporting (LP/MILP/QP/MIQP/QCQP/MIQCP/NLP/MINLP)."""
        pt = self.probtype.upper().strip()
        if pt in {"LP", "MILP", "QP", "MIQP", "QCQP", "MIQCP", "NLP", "MINLP"}:
            return pt
        # Best-effort fall-back classification from variable / constraint counts.
        is_mip = (self.n_binvars + self.n_intvars) > 0
        if pt.startswith("MI") or is_mip:
            return "MINLP"
        if "Q" in pt:
            return "QCQP" if is_mip else "QP"
        return "MINLP" if pt else "unknown"

    @property
    def size_bucket(self) -> str:
        n = self.n_vars
        if n <= 10:
            return "<=10"
        if n <= 30:
            return "<=30"
        if n <= 100:
            return "<=100"
        if n <= 500:
            return "<=500"
        return ">500"

    @property
    def known_optimum(self) -> float | None:
        """Return primal bound if proven optimal, else None.

        Used as the canonical reference for the ``incorrect`` check.
        """
        return self.primal_bound if self.proven_optimal else None


def load_instance_data(csv_path: Path) -> dict[str, InstanceMeta]:
    """Parse instancedata.csv into a {name: InstanceMeta} dict.

    Unknown columns are preserved in ``InstanceMeta.raw``.
    """
    out: dict[str, InstanceMeta] = {}
    if not csv_path.exists():
        return out

    with open(csv_path, newline="", encoding="utf-8") as f:
        # MINLPLib uses ';' (modern releases embed commas inside set-valued
        # fields like {'nl','gms'}, which fools the sniffer). Inspect the
        # header line directly and pick the delimiter that yields more fields.
        header = f.readline()
        f.seek(0)
        if header.count(";") > header.count(","):
            delimiter = ";"
        else:
            delimiter = ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            name = _first(row, _NAME_KEYS)
            if not name:
                continue
            meta = InstanceMeta(
                name=name,
                probtype=_first(row, _PROBTYPE_KEYS, "unknown"),
                n_vars=_to_int(_first(row, _NVARS_KEYS, "0")),
                n_binvars=_to_int(_first(row, _NBINVARS_KEYS, "0")),
                n_intvars=_to_int(_first(row, _NINTVARS_KEYS, "0")),
                n_constraints=_to_int(_first(row, _NCONS_KEYS, "0")),
                objsense=_first(row, _OBJSENSE_KEYS, "min").lower(),
                is_convex=_to_bool(_first(row, _CONVEX_KEYS, "")),
                primal_bound=_to_float(_first(row, _PRIMAL_KEYS, "")),
                dual_bound=_to_float(_first(row, _DUAL_KEYS, "")),
                proven_optimal=bool(_to_bool(_first(row, _PROVEN_KEYS, "")) or False),
                raw=dict(row),
            )
            # If no explicit proven_optimal flag but primal == dual within tolerance,
            # treat as proven.
            if not meta.proven_optimal and meta.primal_bound is not None and meta.dual_bound is not None:
                if math.isclose(meta.primal_bound, meta.dual_bound, abs_tol=1e-6,
                                rel_tol=1e-6):
                    meta.proven_optimal = True

            out[name] = meta
    return out


def known_optima_from_index(index: dict[str, InstanceMeta]) -> dict[str, float]:
    """Extract a {name: known_optimum} dict for proven-optimal instances."""
    return {
        name: meta.primal_bound
        for name, meta in index.items()
        if meta.proven_optimal and meta.primal_bound is not None
    }


# ── Outcome scoring ─────────────────────────────────────────────────────────

OUTCOME_OPTIMAL = "optimal_proven"   # solver proved optimal and matches reference
OUTCOME_FEASIBLE = "feasible_only"   # solver returned a feasible solution, didn't prove
OUTCOME_INCORRECT = "incorrect"      # solver claimed optimal but disagrees with reference
OUTCOME_TIMEOUT = "timeout"          # solver hit time/node limit, no useful solution
OUTCOME_INFEASIBLE = "infeasible"    # solver reports infeasible
OUTCOME_ERROR = "error"              # solver crashed or numerical error
OUTCOME_UNKNOWN = "unknown"

ALL_OUTCOMES = (
    OUTCOME_OPTIMAL,
    OUTCOME_FEASIBLE,
    OUTCOME_INCORRECT,
    OUTCOME_TIMEOUT,
    OUTCOME_INFEASIBLE,
    OUTCOME_ERROR,
    OUTCOME_UNKNOWN,
)


def score_result(
    result: SolveResult,
    meta: InstanceMeta | None,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3,
) -> str:
    """Classify a SolveResult against the reference into an OUTCOME_* tag.

    Reference comparison only triggers when ``meta.known_optimum`` is set
    (i.e. MINLPLib reports the instance as proven-optimal). For unproven
    instances we trust the solver's own status without an objective check.
    """
    status = result.status

    if status == SolveStatus.OPTIMAL:
        # Did the solver agree with the reference?
        ref = meta.known_optimum if meta else None
        if ref is not None and result.objective is not None:
            diff = abs(result.objective - ref)
            if diff > abs_tol + rel_tol * abs(ref):
                return OUTCOME_INCORRECT
        return OUTCOME_OPTIMAL

    if status == SolveStatus.FEASIBLE:
        return OUTCOME_FEASIBLE

    if status == SolveStatus.INFEASIBLE:
        return OUTCOME_INFEASIBLE

    if status in (SolveStatus.TIME_LIMIT, SolveStatus.MEMORY_LIMIT):
        return OUTCOME_TIMEOUT

    if status in (SolveStatus.NUMERICAL_ERROR, SolveStatus.ERROR):
        return OUTCOME_ERROR

    return OUTCOME_UNKNOWN


def tally_outcomes(
    results: list[SolveResult],
    index: dict[str, InstanceMeta],
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3,
) -> dict[str, int]:
    """Group results into per-outcome counts."""
    tally: dict[str, int] = {o: 0 for o in ALL_OUTCOMES}
    for r in results:
        meta = index.get(r.instance)
        outcome = score_result(r, meta, abs_tol=abs_tol, rel_tol=rel_tol)
        tally[outcome] += 1
    return tally
