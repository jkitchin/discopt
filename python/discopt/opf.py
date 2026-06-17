"""AC optimal power flow (AC-OPF) in rectangular voltage coordinates.

AC-OPF is the canonical hard QCQP of power systems: choose generator setpoints
and bus voltages to meet demand at minimum cost, subject to the (nonconvex)
power-flow physics. In **rectangular** coordinates ``V_i = e_i + j f_i`` the
power injections are *quadratic* in ``(e, f)``, so the whole problem is a QCQP —
exactly the structure the Wave-2 PSD/SOC cuts target.

For a bus ``i`` with network admittance ``Y_ik = G_ik + j B_ik`` the injected
power is ``S_i = V_i * conj(sum_k Y_ik V_k)``, giving

    P_i = sum_k [ G_ik (e_i e_k + f_i f_k) + B_ik (f_i e_k - e_i f_k) ]
    Q_i = sum_k [ G_ik (f_i e_k - e_i f_k) - B_ik (e_i e_k + f_i f_k) ].

Power balance sets ``P_i = Pg_i - Pd_i`` and ``Q_i = Qg_i - Qd_i``; voltage
magnitudes obey ``Vmin^2 <= e_i^2 + f_i^2 <= Vmax^2``. The slack bus pins the
reference angle (``f = 0``). The objective minimises linear generation cost.

This module is a self-contained builder over a small network spec; it emits a
standard :class:`discopt.modeling.core.Model`, so the global solver (with
``psd_cuts=True`` for the moment strengthening) handles it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import discopt.modeling.core as dm
from discopt.modeling.core import Expression, Variable

__all__ = ["Bus", "Line", "ACOPF", "build_ac_opf_rectangular"]


@dataclass
class Bus:
    """A network bus.

    ``kind`` is ``"slack"`` (reference, hosts a generator), ``"gen"`` (PV/PQ
    generator), or ``"load"`` (no generator). Demand ``pd``/``qd`` and the
    generator limits/cost are in per-unit.
    """

    name: str
    kind: str = "load"
    pd: float = 0.0
    qd: float = 0.0
    vmin: float = 0.95
    vmax: float = 1.05
    pg_min: float = 0.0
    pg_max: float = 0.0
    qg_min: float = 0.0
    qg_max: float = 0.0
    cost: float = 0.0  # linear $/pu of real generation


@dataclass
class Line:
    """A transmission line with series impedance ``z = r + j x`` (per-unit)."""

    frm: str
    to: str
    r: float
    x: float


@dataclass
class ACOPF:
    buses: list[Bus]
    lines: list[Line]
    # Reference real voltage at the slack bus (f is pinned to 0 there).
    slack_e: float = 1.0
    _: bool = field(default=False, repr=False)


def admittance_matrix(acopf: ACOPF) -> tuple[np.ndarray, np.ndarray]:
    """Build the bus admittance matrix ``Y = G + jB`` (series lines, no shunts)."""
    idx = {b.name: i for i, b in enumerate(acopf.buses)}
    n = len(acopf.buses)
    Y = np.zeros((n, n), dtype=complex)
    for ln in acopf.lines:
        y = 1.0 / complex(ln.r, ln.x)
        a, b = idx[ln.frm], idx[ln.to]
        Y[a, a] += y
        Y[b, b] += y
        Y[a, b] -= y
        Y[b, a] -= y
    return Y.real.copy(), Y.imag.copy()


def build_ac_opf_rectangular(acopf: ACOPF) -> dm.Model:
    """Build the rectangular AC-OPF QCQP as a discopt :class:`Model`."""
    G, B = admittance_matrix(acopf)
    n = len(acopf.buses)
    m = dm.Model("ac_opf")

    e = [m.continuous(f"e_{b.name}", lb=-b.vmax, ub=b.vmax) for b in acopf.buses]
    f = [m.continuous(f"f_{b.name}", lb=-b.vmax, ub=b.vmax) for b in acopf.buses]
    pg: dict[int, Variable] = {}
    qg: dict[int, Variable] = {}
    for i, b in enumerate(acopf.buses):
        if b.kind in ("slack", "gen"):
            pg[i] = m.continuous(f"pg_{b.name}", lb=b.pg_min, ub=b.pg_max)
            qg[i] = m.continuous(f"qg_{b.name}", lb=b.qg_min, ub=b.qg_max)

    def _p_inj(i: int):
        terms = []
        for k in range(n):
            if G[i, k] != 0.0 or B[i, k] != 0.0:
                terms.append(G[i, k] * (e[i] * e[k] + f[i] * f[k]))
                terms.append(B[i, k] * (f[i] * e[k] - e[i] * f[k]))
        return dm.sum(terms)

    def _q_inj(i: int):
        terms = []
        for k in range(n):
            if G[i, k] != 0.0 or B[i, k] != 0.0:
                terms.append(G[i, k] * (f[i] * e[k] - e[i] * f[k]))
                terms.append(-B[i, k] * (e[i] * e[k] + f[i] * f[k]))
        return dm.sum(terms)

    for i, b in enumerate(acopf.buses):
        pg_i: Expression | float = pg[i] if i in pg else 0.0
        qg_i: Expression | float = qg[i] if i in qg else 0.0
        # P_i = Pg_i - Pd_i ; Q_i = Qg_i - Qd_i
        m.subject_to(_p_inj(i) == pg_i - b.pd, name=f"pbal_{b.name}")
        m.subject_to(_q_inj(i) == qg_i - b.qd, name=f"qbal_{b.name}")
        # Voltage magnitude limits: Vmin^2 <= e^2 + f^2 <= Vmax^2.
        m.subject_to(e[i] * e[i] + f[i] * f[i] <= b.vmax * b.vmax, name=f"vmax_{b.name}")
        m.subject_to(e[i] * e[i] + f[i] * f[i] >= b.vmin * b.vmin, name=f"vmin_{b.name}")
        if b.kind == "slack":
            # Reference angle: f = 0, e fixed to the reference voltage.
            m.subject_to(f[i] == 0.0, name=f"ref_f_{b.name}")
            m.subject_to(e[i] == acopf.slack_e, name=f"ref_e_{b.name}")

    cost_terms = [b.cost * pg[i] for i, b in enumerate(acopf.buses) if i in pg and b.cost != 0.0]
    m.minimize(dm.sum(cost_terms) if cost_terms else dm.sum([0.0 * e[0]]))
    return m


def two_bus_example() -> ACOPF:
    """A 2-bus AC-OPF: slack generator feeds a PQ load over one lossy line."""
    return ACOPF(
        buses=[
            Bus(
                "g",
                kind="slack",
                vmin=0.95,
                vmax=1.05,
                pg_min=0.0,
                pg_max=2.0,
                qg_min=-1.0,
                qg_max=1.0,
                cost=1.0,
            ),
            Bus("l", kind="load", pd=0.5, qd=0.2, vmin=0.95, vmax=1.05),
        ],
        lines=[Line("g", "l", r=0.01, x=0.1)],
        slack_e=1.0,
    )
