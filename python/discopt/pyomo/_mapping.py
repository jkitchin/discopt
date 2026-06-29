"""Translate between discopt ``SolveResult`` and Pyomo results/options.

Three concerns live here:

* :func:`translate_options` — Pyomo solver options -> ``Model.solve`` kwargs.
* :data:`STATUS_MAP` / :func:`termination_for` — discopt status -> Pyomo enums.
* :func:`solution_flat` / :func:`load_solution` / :func:`load_duals` — index-aligned
  primal/dual loading (the alignment is guaranteed by the ``.nl`` column/row order;
  see ``_writer.write_nl``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# discopt SolveResult.status -> (TerminationCondition name, SolverStatus name).
# Strings (resolved lazily so importing this module never requires pyomo).
STATUS_MAP: dict[str, tuple[str, str]] = {
    "optimal": ("optimal", "ok"),
    "feasible": ("feasible", "ok"),
    "infeasible": ("infeasible", "ok"),
    "unbounded": ("unbounded", "ok"),
    "infeasible_or_unbounded": ("infeasibleOrUnbounded", "ok"),
    "time_limit": ("maxTimeLimit", "ok"),
    "node_limit": ("maxIterations", "ok"),
    "iteration_limit": ("maxIterations", "ok"),
    "error": ("error", "error"),
}


def termination_for(status: str):
    """Resolve a discopt status string to ``(TerminationCondition, SolverStatus)``."""
    from pyomo.opt import SolverStatus, TerminationCondition

    tc_name, ss_name = STATUS_MAP.get(status, ("unknown", "warning"))
    return getattr(TerminationCondition, tc_name), getattr(SolverStatus, ss_name)


# Pyomo option name (lower-cased) -> discopt Model.solve kwarg.
_OPTION_ALIASES: dict[str, str] = {
    "timelimit": "time_limit",
    "time_limit": "time_limit",
    "maxtime": "time_limit",
    "mipgap": "gap_tolerance",
    "gap": "gap_tolerance",
    "gap_tolerance": "gap_tolerance",
    "threads": "threads",
    "num_threads": "threads",
}


def translate_options(options: dict[str, Any]) -> dict[str, Any]:
    """Map Pyomo solver options to ``Model.solve`` kwargs.

    Recognised aliases are normalised; any other option is forwarded verbatim so a
    new ``Model.solve`` kwarg works without touching this plugin.
    """
    out: dict[str, Any] = {}
    for key, val in options.items():
        out[_OPTION_ALIASES.get(str(key).lower(), str(key))] = val
    return out


def solution_flat(discopt_model: Any, result: Any) -> np.ndarray:
    """Flatten ``SolveResult.x`` (keyed by var name) into ``.nl`` column order.

    Iterates the discopt model's variables (declaration order == ``.nl`` column
    order) and ravels each value, matching the per-element column expansion the NL
    writer / ``result_io.write_sol`` use.
    """
    parts: list[np.ndarray] = []
    for var in discopt_model._variables:
        parts.append(np.asarray(result.x[var.name], dtype=np.float64).ravel())
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def load_solution(pyomo_vars: list, flat: np.ndarray) -> None:
    """Load *flat* (``.nl`` column order) into *pyomo_vars* (same order).

    Integer/binary variables are rounded to defeat ~1e-5 integrality drift and
    clamped to their declared bounds so ``set_value`` validation cannot trip.
    """
    if len(flat) != len(pyomo_vars):
        raise AssertionError(
            f"discopt/pyomo column-order mismatch: {len(flat)} solution values vs "
            f"{len(pyomo_vars)} .nl columns — refusing to load a misaligned solution"
        )
    for vdata, raw in zip(pyomo_vars, flat):
        val = float(raw)
        if vdata.is_integer() or vdata.is_binary():
            val = float(round(val))
        lb, ub = vdata.lb, vdata.ub
        if lb is not None and val < lb:
            val = lb
        if ub is not None and val > ub:
            val = ub
        vdata.set_value(val, skip_validation=True)


def load_duals(
    model: Any,
    discopt_model: Any,
    result: Any,
    pyomo_vars: list,
    pyomo_cons: list,
) -> None:
    """Best-effort dual / reduced-cost loading into the model's Suffixes.

    Only runs when the model declares an importable ``dual`` (and/or ``rc``) Suffix
    and discopt populated the corresponding multipliers (it does so only on the
    KKT/NLP path; ``constraint_duals`` / ``bound_duals_*`` are ``None`` otherwise).

    Both mappings are by **index/row order**, not by name: discopt and Pyomo name
    variables/constraints differently. ``constraint_duals`` is keyed
    ``c.name or f"c{idx}"`` in row order (``solver._unpack_constraint_duals``), so
    we rebuild those keys in row order; ``bound_duals_*`` is keyed by discopt's
    variable name, aligned to ``pyomo_vars[i]`` by column index.
    """
    dual_suf = _importable_suffix(model, "dual")
    if dual_suf is not None and result.constraint_duals is not None:
        flat_duals: list[float] = []
        ok = True
        for idx, con in enumerate(discopt_model._constraints):
            key = con.name if con.name else f"c{idx}"
            mult = result.constraint_duals.get(key)
            if mult is None:
                ok = False
                break
            flat_duals.extend(np.asarray(mult, dtype=np.float64).ravel().tolist())
        if ok and len(flat_duals) == len(pyomo_cons):
            for cdata, mu in zip(pyomo_cons, flat_duals):
                dual_suf[cdata] = float(mu)

    rc_suf = _importable_suffix(model, "rc")
    if rc_suf is not None and (
        result.bound_duals_lower is not None or result.bound_duals_upper is not None
    ):
        lower = result.bound_duals_lower or {}
        upper = result.bound_duals_upper or {}
        # rc = lower-bound multiplier - upper-bound multiplier; align by column index.
        for dvar, vdata in zip(discopt_model._variables, pyomo_vars):
            lo = np.asarray(lower.get(dvar.name, 0.0), dtype=np.float64).ravel()
            hi = np.asarray(upper.get(dvar.name, 0.0), dtype=np.float64).ravel()
            if lo.size == 1 and hi.size == 1:
                rc_suf[vdata] = float(lo[0]) - float(hi[0])


def _importable_suffix(model: Any, name: str):
    """Return the model's Suffix *name* if present and import-enabled, else None."""
    from pyomo.core.base.suffix import Suffix

    suf = getattr(model, name, None)
    if suf is None or not isinstance(suf, Suffix):
        return None
    if not suf.import_enabled():
        return None
    return suf
