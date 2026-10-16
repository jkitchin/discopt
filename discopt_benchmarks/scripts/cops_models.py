# ruff: noqa: N803, N806  -- names follow the COPS/JuMP source so each line can be checked against it
"""COPS 3.0 continuous-NLP benchmark models, ported to discopt.

Source: the JuMP backend of MadNLP/COPSBenchmark.jl (``ext/COPSBenchmarkJuMP/*.jl``,
fetched 2026-10-16 at ``149d2d10``), itself adapted from Dolan, Moré and Munson,
"Benchmarking Optimization Software with COPS 3.0", ANL/MCS-TM-273 (2004). Each
builder follows its JuMP file statement by statement (Julia's 1-based indices are
shifted to 0-based; the comments name the JuMP constraint each block mirrors).

Every builder returns ``(model, start)``. ``start`` maps each ``Variable`` to the
COPS starting point, ready for ``Model.solve(initial_solution=...)``. A JuMP
variable declared with no ``start`` gets 0.0, Ipopt's default.

Not ported: ``tetra_*`` and ``triangle_*`` (they need the mesh files under
``data/``) and the PDE transition-state models (``dirichlet``, ``henon``,
``lane_emden``, which need ``pde_utils``' domain builder).

**Known deviations from the JuMP source** -- each one changes only a starting point,
never the model:

* ``elec``: the start is uniform-random on the sphere from Julia's
  ``Random.seed!(2713)``. numpy cannot reproduce Julia's stream, so this uses
  ``np.random.default_rng(2713)`` for the same distribution and a different draw.

Coefficients that vary by row are applied row by row as Python floats rather than
as numpy-array constants. An array ``Constant`` inside a nonlinear term used to
crash ``milp_relaxation._product_to_monomial``; that was found while writing these
ports and fixed in #1717, and the scalar form is kept so the models build the same
expressions they were validated with.

**Validation (2026-10-16).** At the reference size with the COPS start, 15 of 17
models reproduce ``COPS_REFERENCE`` through ``Model.solve`` (worst relative error
3.5e-4, glider). The other two are checked with a local POUNCE solve
(``solvers.nlp_pounce.solve_nlp_from_model``) from the COPS start: ``pinene``
19.87217 (reference 19.8721) and ``torsion`` -0.418087 (reference -0.418087). Through
``Model.solve`` they return a worse uncertified point (pinene) or ran out of memory
(torsion, fixed by #1718).

``COPS_REFERENCE`` gives the reference objective at the reference size, copied from
``test/runtests.jl`` of the same repository ("Reference values from
http://www.mcs.anl.gov/~more/cops/cops3.pdf"). They are best-known LOCAL optima
reported by the COPS authors, not certified global optima, so they can check that a
port is faithful but cannot serve as a dual-bound oracle.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np


def _fact(k: int) -> float:
    return float(math.factorial(k))


# name -> (builder args at the reference size, reference objective, sense)
COPS_REFERENCE: dict[str, tuple[tuple, float, str]] = {
    "bearing": ((50, 50), -1.5482e-1, "min"),
    "camshape": ((200,), 4.2791, "max"),
    "catmix": ((100,), -4.80556e-2, "min"),
    "chain": ((800,), 5.06891, "min"),
    "channel": ((100,), 1.0, "min"),
    "elec": ((50,), 1.0552e3, "min"),
    "gasoil": ((100,), 5.2366e-3, "min"),
    "glider": ((100,), -1.25505e3, "min"),
    "marine": ((100,), 1.97462e7, "min"),
    "methanol": ((100,), 9.02229e-3, "min"),
    "minsurf": ((50, 50), 2.51488, "min"),
    "pinene": ((100,), 1.98721e1, "min"),
    "polygon": ((100,), -0.674981, "min"),
    "robot": ((200,), 9.14138, "min"),
    "rocket": ((400,), 1.01283, "max"),
    "steering": ((200,), 5.54577e-1, "min"),
    "torsion": ((50, 50), -4.18087e-1, "min"),
}


# --------------------------------------------------------------------------- chain
def chain(n: int):
    nh = max(2, (n - 4) // 4)
    L, a, b = 4.0, 1.0, 3.0
    tmin = 0.25 if b > a else 0.75
    h = 1.0 / nh
    k = np.arange(1, nh + 2, dtype=float)  # Julia k = 1:nh+1
    m = dm.Model("cops_chain")
    u = m.continuous("u", shape=(nh + 1,), lb=-np.inf, ub=np.inf)
    x1 = m.continuous("x1", shape=(nh + 1,), lb=-np.inf, ub=np.inf)
    x2 = m.continuous("x2", shape=(nh + 1,), lb=-np.inf, ub=np.inf)
    x3 = m.continuous("x3", shape=(nh + 1,), lb=-np.inf, ub=np.inf)
    m.minimize(x2[nh])
    m.subject_to(x1[1:] - x1[:-1] - 0.5 * h * (u[:-1] + u[1:]) == 0)
    m.subject_to(x1[0] == a)
    m.subject_to(x1[nh] == b)
    m.subject_to(x2[0] == 0)
    m.subject_to(x3[0] == 0)
    m.subject_to(x3[nh] == L)
    s_lo = dm.sqrt(1 + u[:-1] ** 2)
    s_hi = dm.sqrt(1 + u[1:] ** 2)
    m.subject_to(x2[1:] - x2[:-1] - 0.5 * h * (x1[:-1] * s_lo + x1[1:] * s_hi) == 0)
    m.subject_to(x3[1:] - x3[:-1] - 0.5 * h * (s_lo + s_hi) == 0)
    base = 4 * abs(b - a) * (k / nh - tmin)
    x1s = 4 * abs(b - a) * k / nh * (0.5 * k / nh - tmin) + a
    start = {u: base, x1: x1s, x2: x1s * base, x3: base.copy()}
    return m, start


# -------------------------------------------------------------------------- rocket
def rocket(nh: int):
    h_0, v_0, m_0, g_0 = 1.0, 0.0, 1.0, 1.0
    T_c, h_c, v_c, m_c = 3.5, 500.0, 620.0, 0.6
    c = 0.5 * math.sqrt(g_0 * h_0)
    m_f = m_c * m_0
    D_c = 0.5 * v_c * (m_0 / g_0)
    T_max = T_c * m_0 * g_0
    i = np.arange(nh + 1, dtype=float)
    m = dm.Model("cops_rocket")
    hh = m.continuous("h", shape=(nh + 1,), lb=1.0, ub=np.inf)
    v = m.continuous("v", shape=(nh + 1,), lb=0.0, ub=np.inf)
    mm = m.continuous("m", shape=(nh + 1,), lb=m_f, ub=m_0)
    T = m.continuous("T", shape=(nh + 1,), lb=0.0, ub=T_max)
    step = m.continuous("step", lb=0.0, ub=np.inf)

    def dv(sl):
        D = D_c * v[sl] ** 2 * dm.exp(-h_c * (hh[sl] - h_0)) / h_0
        g = g_0 * (h_0 / hh[sl]) ** 2
        return (T[sl] - D - mm[sl] * g) / mm[sl]

    lo, hi = slice(0, nh), slice(1, nh + 1)
    m.maximize(hh[nh])
    m.subject_to(hh[hi] == hh[lo] + 0.5 * step * (v[hi] + v[lo]))  # con_dh
    m.subject_to(v[hi] == v[lo] + 0.5 * step * (dv(hi) + dv(lo)))  # con_dv
    m.subject_to(mm[hi] == mm[lo] + 0.5 * step * (-T[hi] / c - T[lo] / c))  # con_dm
    m.subject_to(hh[0] == h_0)
    m.subject_to(v[0] == v_0)
    m.subject_to(mm[0] == m_0)
    m.subject_to(mm[nh] == m_f)
    start = {
        hh: np.ones(nh + 1),
        v: i / nh * (1.0 - i / nh),
        mm: (m_f - m_0) * (i / nh) + m_0,
        T: np.full(nh + 1, T_max / 2.0),
        step: np.array(1.0 / nh),
    }
    return m, start


# -------------------------------------------------------------------------- glider
def glider(nh: int):
    x_0, y_0, y_f = 0.0, 1000.0, 900.0
    vx_0 = vx_f = 13.23
    vy_0 = vy_f = -1.288
    u_c, r_0, mass, g = 2.5, 100.0, 100.0, 9.81
    c0, c1, S, rho = 0.034, 0.069662, 14.0, 1.13
    cL_min, cL_max = 0.0, 1.4
    k = np.arange(nh + 1, dtype=float)
    m = dm.Model("cops_glider")
    t_f = m.continuous("t_f", lb=0.0, ub=np.inf)
    x = m.continuous("x", shape=(nh + 1,), lb=0.0, ub=np.inf)
    y = m.continuous("y", shape=(nh + 1,), lb=-np.inf, ub=np.inf)
    vx = m.continuous("vx", shape=(nh + 1,), lb=0.0, ub=np.inf)
    vy = m.continuous("vy", shape=(nh + 1,), lb=-np.inf, ub=np.inf)
    cL = m.continuous("cL", shape=(nh + 1,), lb=cL_min, ub=cL_max)
    step = t_f / nh

    def dots(sl):
        r = (x[sl] / r_0 - 2.5) ** 2
        u = u_c * (1 - r) * dm.exp(-r)
        w = vy[sl] - u
        vv = dm.sqrt(vx[sl] ** 2 + w**2)
        D = 0.5 * (c0 + c1 * cL[sl] ** 2) * rho * S * vv**2
        Lf = 0.5 * cL[sl] * rho * S * vv**2
        vx_dot = (-Lf * (w / vv) - D * (vx[sl] / vv)) / mass
        vy_dot = (Lf * (vx[sl] / vv) - D * (w / vv)) / mass - g
        return vx_dot, vy_dot

    lo, hi = slice(0, nh), slice(1, nh + 1)
    vxd_hi, vyd_hi = dots(hi)
    vxd_lo, vyd_lo = dots(lo)
    m.minimize(-x[nh])
    m.subject_to(x[hi] == x[lo] + 0.5 * step * (vx[hi] + vx[lo]))
    m.subject_to(y[hi] == y[lo] + 0.5 * step * (vy[hi] + vy[lo]))
    m.subject_to(vx[hi] == vx[lo] + 0.5 * step * (vxd_hi + vxd_lo))
    m.subject_to(vy[hi] == vy[lo] + 0.5 * step * (vyd_hi + vyd_lo))
    m.subject_to(x[0] == x_0)
    m.subject_to(y[0] == y_0)
    m.subject_to(y[nh] == y_f)
    m.subject_to(vx[0] == vx_0)
    m.subject_to(vx[nh] == vx_f)
    m.subject_to(vy[0] == vy_0)
    m.subject_to(vy[nh] == vy_f)
    start = {
        t_f: np.array(1.0),
        x: x_0 + vx_0 * (k / nh),
        y: y_0 + (k / nh) * (y_f - y_0),
        vx: np.full(nh + 1, vx_0),
        vy: np.full(nh + 1, vy_0),
        cL: np.full(nh + 1, cL_max / 2.0),
    }
    return m, start


# --------------------------------------------------------------------------- robot
def robot(nh: int):
    L, rho0, phi0 = 5.0, 4.5, math.pi / 4
    k = np.arange(1, nh + 2, dtype=float)
    m = dm.Model("cops_robot")
    rho = m.continuous("rho", shape=(nh + 1,), lb=0.0, ub=L)
    the = m.continuous("the", shape=(nh + 1,), lb=-math.pi, ub=math.pi)
    phi = m.continuous("phi", shape=(nh + 1,), lb=0.0, ub=math.pi)
    rho_dot = m.continuous("rho_dot", shape=(nh + 1,), lb=-np.inf, ub=np.inf)
    the_dot = m.continuous("the_dot", shape=(nh + 1,), lb=-np.inf, ub=np.inf)
    phi_dot = m.continuous("phi_dot", shape=(nh + 1,), lb=-np.inf, ub=np.inf)
    u_rho = m.continuous("u_rho", shape=(nh + 1,), lb=-1.0, ub=1.0)
    u_the = m.continuous("u_the", shape=(nh + 1,), lb=-1.0, ub=1.0)
    u_phi = m.continuous("u_phi", shape=(nh + 1,), lb=-1.0, ub=1.0)
    tf = m.continuous("tf", lb=0.0, ub=np.inf)
    step = tf / nh

    def inertia(sl):
        cube = (L - rho[sl]) ** 3 + rho[sl] ** 3
        return cube * dm.sin(phi[sl]) ** 2 / 3.0, cube / 3.0

    lo, hi = slice(0, nh), slice(1, nh + 1)
    It_hi, Ip_hi = inertia(hi)
    It_lo, Ip_lo = inertia(lo)
    m.minimize(tf)
    m.subject_to(rho[hi] == rho[lo] + 0.5 * step * (rho_dot[hi] + rho_dot[lo]))
    m.subject_to(phi[hi] == phi[lo] + 0.5 * step * (phi_dot[hi] + phi_dot[lo]))
    m.subject_to(the[hi] == the[lo] + 0.5 * step * (the_dot[hi] + the_dot[lo]))
    m.subject_to(rho_dot[hi] == rho_dot[lo] + 0.5 * step * (u_rho[hi] + u_rho[lo]) / L)
    m.subject_to(the_dot[hi] == the_dot[lo] + 0.5 * step * (u_the[hi] / It_hi + u_the[lo] / It_lo))
    m.subject_to(phi_dot[hi] == phi_dot[lo] + 0.5 * step * (u_phi[hi] / Ip_hi + u_phi[lo] / Ip_lo))
    m.subject_to(rho[0] == 4.5)
    m.subject_to(the[0] == 0.0)
    m.subject_to(phi[0] == math.pi / 4.0)
    m.subject_to(rho[nh] == 4.5)
    m.subject_to(the[nh] == 2.0 * math.pi / 3)
    m.subject_to(phi[nh] == math.pi / 4.0)
    for d in (rho_dot, the_dot, phi_dot):
        m.subject_to(d[0] == 0.0)
        m.subject_to(d[nh] == 0.0)
    z = np.zeros(nh + 1)
    start = {
        rho: np.full(nh + 1, rho0),
        the: 2 * math.pi / 3 * (k / nh) ** 2,
        phi: np.full(nh + 1, phi0),
        rho_dot: z.copy(),
        the_dot: 4 * math.pi / 3 * (k / nh),
        phi_dot: z.copy(),
        u_rho: z.copy(),
        u_the: z.copy(),
        u_phi: z.copy(),
        tf: np.array(1.0),
    }
    return m, start


# ------------------------------------------------------------------------ steering
def steering(nh: int):
    a = 100.0
    m = dm.Model("cops_steering")
    u = m.continuous("u", shape=(nh + 1,), lb=-math.pi / 2.0, ub=math.pi / 2.0)
    x = m.continuous("x", shape=(nh + 1, 4), lb=-np.inf, ub=np.inf)
    tf = m.continuous("tf", lb=-np.inf, ub=np.inf)
    h = tf / nh
    lo, hi = slice(0, nh), slice(1, nh + 1)
    m.minimize(tf)
    m.subject_to(tf >= 0.0)  # a constraint row in the JuMP source, not a bound
    m.subject_to(x[hi, 0] == x[lo, 0] + 0.5 * h * (x[lo, 2] + x[hi, 2]))
    m.subject_to(x[hi, 1] == x[lo, 1] + 0.5 * h * (x[lo, 3] + x[hi, 3]))
    m.subject_to(x[hi, 2] == x[lo, 2] + 0.5 * h * (a * dm.cos(u[lo]) + a * dm.cos(u[hi])))
    m.subject_to(x[hi, 3] == x[lo, 3] + 0.5 * h * (a * dm.sin(u[lo]) + a * dm.sin(u[hi])))
    for j in range(4):
        m.subject_to(x[0, j] == 0.0)
    for j, val in ((1, 5.0), (2, 45.0), (3, 0.0)):
        m.subject_to(x[nh, j] == val)
    k = np.arange(1, nh + 2, dtype=float)
    xs = np.zeros((nh + 1, 4))
    xs[:, 1] = 5 * k / nh
    xs[:, 2] = 45.0 * k / nh
    start = {u: np.zeros(nh + 1), x: xs, tf: np.array(1.0)}
    return m, start


# -------------------------------------------------------------------------- catmix
def catmix(nh: int):
    ne, nc = 2, 3
    h = 1.0 / nh
    rho = [0.11270166537926, 0.50000000000000, 0.88729833462074]
    bc = [1.0, 0.0]
    m = dm.Model("cops_catmix")
    u = m.continuous("u", shape=(nh, nc), lb=0.0, ub=1.0)
    v = m.continuous("v", shape=(nh, ne), lb=-np.inf, ub=np.inf)
    w = m.continuous("w", shape=(nh, nc, ne), lb=-np.inf, ub=np.inf)
    pp = m.continuous("pp", shape=(nh, nc, ne), lb=-np.inf, ub=np.inf)
    Dpp = m.continuous("Dpp", shape=(nh, nc, ne), lb=-np.inf, ub=np.inf)
    ppf = m.continuous("ppf", shape=(ne,), lb=-np.inf, ub=np.inf)
    # alpha = 0 in the source, so the smoothing term is identically zero.
    m.minimize(-1.0 + ppf[0] + ppf[1])
    for k in range(nc):
        for s in range(ne):
            m.subject_to(
                pp[:, k, s]
                == v[:, s] + h * sum(w[:, j - 1, s] * (rho[k] ** j / _fact(j)) for j in (1, 2, 3))
            )
            m.subject_to(
                Dpp[:, k, s]
                == sum(w[:, j - 1, s] * (rho[k] ** (j - 1) / _fact(j - 1)) for j in (1, 2, 3))
            )
    for s in range(ne):
        m.subject_to(
            ppf[s] == v[nh - 1, s] + h * sum(w[nh - 1, j - 1, s] / _fact(j) for j in (1, 2, 3))
        )
        m.subject_to(
            v[: nh - 1, s] + sum(w[: nh - 1, j - 1, s] * (h / _fact(j)) for j in (1, 2, 3))
            == v[1:, s]
        )
        m.subject_to(v[0, s] == bc[s])
    for j in range(nc):
        m.subject_to(Dpp[:, j, 0] == u[:, j] * (10.0 * pp[:, j, 1] - pp[:, j, 0]))
        m.subject_to(
            Dpp[:, j, 1]
            == u[:, j] * (pp[:, j, 0] - 10.0 * pp[:, j, 1]) - (1 - u[:, j]) * pp[:, j, 1]
        )
    mod = np.array([1.0, 0.0])  # mod(s, ne) for s = 1, 2
    start = {
        u: np.zeros((nh, nc)),
        v: np.tile(mod, (nh, 1)),
        w: np.zeros((nh, nc, ne)),
        pp: np.tile(mod, (nh, nc, 1)),
        Dpp: np.zeros((nh, nc, ne)),
        ppf: mod.copy(),
    }
    return m, start


# ------------------------------------------------------------------------ camshape
def camshape(n: int):
    R_v, R_max, R_min, alpha = 1.0, 2.0, 1.0, 1.5
    d_theta = 2 * math.pi / (5 * (n + 1))
    cd = math.cos(d_theta)
    lim = alpha * d_theta
    m = dm.Model("cops_camshape")
    r = m.continuous("r", shape=(n,), lb=R_min, ub=R_max)
    m.maximize(((math.pi * R_v) / n) * dm.sum(r))
    m.subject_to(-r[:-2] * r[1:-1] - r[1:-1] * r[2:] + 2 * r[:-2] * r[2:] * cd <= 0.0)
    m.subject_to(-R_min * r[0] - r[0] * r[1] + 2 * R_min * r[1] * cd <= 0.0)
    m.subject_to(-(R_min**2) - R_min * r[0] + 2 * R_min * r[0] * cd <= 0.0)
    m.subject_to(-r[n - 2] * r[n - 1] - r[n - 1] * R_max + 2 * r[n - 2] * R_max * cd <= 0.0)
    m.subject_to(-2 * R_max * r[n - 1] + 2 * r[n - 1] ** 2 * cd <= 0.0)
    m.subject_to(r[1:] - r[:-1] >= -lim)
    m.subject_to(r[1:] - r[:-1] <= lim)
    m.subject_to(r[0] - R_min >= -lim)
    m.subject_to(r[0] - R_min <= lim)
    m.subject_to(R_max - r[n - 1] >= -lim)
    m.subject_to(R_max - r[n - 1] <= lim)
    return m, {r: np.full(n, (R_min + R_max) / 2.0)}


# ---------------------------------------------------------------------------- elec
def elec(np_: int):
    rng = np.random.default_rng(2713)  # see the module docstring: not Julia's draw
    theta = 2 * math.pi * rng.random(np_)
    phi = math.pi * rng.random(np_)
    m = dm.Model("cops_elec")
    x = m.continuous("x", shape=(np_,), lb=-np.inf, ub=np.inf)
    y = m.continuous("y", shape=(np_,), lb=-np.inf, ub=np.inf)
    z = m.continuous("z", shape=(np_,), lb=-np.inf, ub=np.inf)
    terms = []
    for i in range(np_ - 1):
        d2 = (x[i] - x[i + 1 :]) ** 2 + (y[i] - y[i + 1 :]) ** 2 + (z[i] - z[i + 1 :]) ** 2
        terms.append(dm.sum(1.0 / dm.sqrt(d2)))
    m.minimize(sum(terms))
    m.subject_to(x**2 + y**2 + z**2 == 1)
    start = {
        x: np.cos(theta) * np.sin(phi),
        y: np.sin(theta) * np.sin(phi),
        z: np.cos(phi),
    }
    return m, start


# ------------------------------------------------------------------------- polygon
def polygon(n: int):
    N = n // 2
    m = dm.Model("cops_polygon")
    r = m.continuous("r", shape=(N,), lb=0.0, ub=1.0)
    th = m.continuous("theta", shape=(N,), lb=0.0, ub=math.pi)
    m.subject_to(th[N - 1] == math.pi)
    m.subject_to(r[N - 1] == 0)
    m.subject_to(th[1:] - th[:-1] >= 0.0)
    for i in range(N - 1):
        m.subject_to(
            r[i] ** 2 + r[i + 1 :] ** 2 - 2 * r[i] * r[i + 1 :] * dm.cos(th[i] - th[i + 1 :]) - 1
            <= 0
        )
    m.minimize(-0.5 * dm.sum(r[:-1] * r[1:] * dm.sin(th[1:] - th[:-1])))
    i = np.arange(1, N + 1, dtype=float)
    return m, {r: np.ones(N), th: i * math.pi / (N - 1) - math.pi / (N - 1)}


# ------------------------------------------------------------------------- bearing
def bearing(nx: int, ny: int):
    b, e = 10.0, 0.1
    hx = 2 * math.pi / (nx + 1)
    hy = 2 * b / (ny + 1)
    wq = [(1.0 + e * math.cos(i * hx)) ** 3 for i in range(nx + 2)]
    m = dm.Model("cops_bearing")
    v = m.continuous("v", shape=(nx + 2, ny + 2), lb=0.0, ub=np.inf)
    ex1 = []
    for i in range(nx + 1):  # Julia i = 1:nx+1, j = 1:ny+1
        dx = (v[i + 1, : ny + 1] - v[i, : ny + 1]) / hx
        dy = (v[i, 1 : ny + 2] - v[i, : ny + 1]) / hy
        ex1.append((wq[i] + 2 * wq[i + 1]) * dm.sum(dx**2 + dy**2))
    ex2 = []
    for i in range(1, nx + 2):  # Julia i = 2:nx+2, j = 2:ny+2
        dx = (v[i - 1, 1:] - v[i, 1:]) / hx
        dy = (v[i, : ny + 1] - v[i, 1:]) / hy
        ex2.append((2 * wq[i] + 2 * wq[i - 1]) * dm.sum(dx**2 + dy**2))
    ex3 = [e * math.sin(i * hx) * dm.sum(v[i, :]) for i in range(nx + 2) if math.sin(i * hx)]
    m.minimize(0.5 * (hx * hy / 6.0) * (sum(ex1) + sum(ex2)) - hx * hy * sum(ex3))
    m.subject_to(v[:, 0] == 0.0)
    m.subject_to(v[:, ny + 1] == 0.0)
    m.subject_to(v[0, :] == 0.0)
    m.subject_to(v[nx + 1, :] == 0.0)
    s = np.maximum(np.sin(np.arange(nx + 2) * hx), 0.0)
    return m, {v: np.tile(s[:, None], (1, ny + 2))}


# ------------------------------------------------------------------------- torsion
def torsion(nx: int, ny: int):
    c = 5.0
    hx = 1.0 / (nx + 1.0)
    hy = 1.0 / (ny + 1.0)
    area = 0.5 * hx * hy
    D = np.array(
        [
            [min(min(i, nx - i + 1) * hx, min(j, ny - j + 1) * hy) for j in range(ny + 2)]
            for i in range(nx + 2)
        ]
    )
    m = dm.Model("cops_torsion")
    v = m.continuous("v", shape=(nx + 2, ny + 2), lb=-np.inf, ub=np.inf)
    lin_lower = dm.sum(v[1:, : ny + 1] + v[: nx + 1, : ny + 1] + v[: nx + 1, 1:])
    lin_upper = dm.sum(v[1:, 1:] + v[: nx + 1, 1:] + v[1:, : ny + 1])
    quad_lower = dm.sum(
        ((v[1:, : ny + 1] - v[: nx + 1, : ny + 1]) / hx) ** 2
        + ((v[: nx + 1, 1:] - v[: nx + 1, : ny + 1]) / hy) ** 2
    )
    quad_upper = dm.sum(
        ((v[1:, 1:] - v[: nx + 1, 1:]) / hx) ** 2 + ((v[1:, 1:] - v[1:, : ny + 1]) / hy) ** 2
    )
    m.minimize(area * ((quad_lower + quad_upper) / 2.0 - c * (lin_lower + lin_upper) / 3.0))
    for i in range(nx + 2):  # -D <= v <= D: constraint rows in the JuMP source
        for j in range(ny + 2):
            m.subject_to(v[i, j] >= -float(D[i, j]))
            m.subject_to(v[i, j] <= float(D[i, j]))
    return m, {v: D.copy()}


# ------------------------------------------------------------------------- minsurf
def minsurf(nx: int, ny: int):
    x_mesh = np.linspace(0, 1, nx + 2)
    v0 = np.tile((1 - (2 * x_mesh - 1) ** 2)[:, None], (1, ny + 2))
    hx = 1 / (nx + 1)
    hy = 1 / (ny + 1)
    area = 0.5 * hx * hy
    m = dm.Model("cops_minsurf")
    v = m.continuous("v", shape=(nx + 2, ny + 2), lb=-np.inf, ub=np.inf)
    s1 = dm.sum(
        dm.sqrt(
            1
            + ((v[1:, : ny + 1] - v[: nx + 1, : ny + 1]) / hx) ** 2
            + ((v[: nx + 1, 1:] - v[: nx + 1, : ny + 1]) / hy) ** 2
        )
    )
    s2 = dm.sum(
        dm.sqrt(
            1
            + ((v[: nx + 1, 1:] - v[1:, 1:]) / hx) ** 2
            + ((v[1:, : ny + 1] - v[1:, 1:]) / hy) ** 2
        )
    )
    m.minimize(area * s1 + area * s2)
    m.subject_to(v[0, :] == 0)
    m.subject_to(v[nx + 1, :] == 0)
    for i in range(nx + 2):
        val = 1 - (2 * i * hx - 1) ** 2
        m.subject_to(v[i, 0] == val)
        m.subject_to(v[i, ny + 1] == val)
    m.subject_to(v >= 0)
    i_lo, i_hi = int(math.floor(0.25 / hx)), int(math.ceil(0.75 / hx))
    j_lo, j_hi = int(math.floor(0.25 / hy)), int(math.ceil(0.75 / hy))
    m.subject_to(v[i_lo : i_hi + 1, j_lo : j_hi + 1] >= 1)
    return m, {v: v0}


# ------------------------------------------------------------------------- channel
def channel(nh: int):
    nc = nd = 4
    R = 10.0
    h = 1.0 / nh
    rho = [0.06943184420297, 0.33000947820757, 0.66999052179243, 0.93056815579703]
    t = np.arange(nh + 1) * h
    v0 = np.zeros((nh, nd))
    v0[:, 0] = t[:nh] ** 2 * (3.0 - 2.0 * t[:nh])
    v0[:, 1] = 6 * t[:nh] * (1.0 - t[:nh])
    v0[:, 2] = 6 * (1.0 - 2.0 * t[:nh])
    v0[:, 3] = -12.0
    m = dm.Model("cops_channel")
    v = m.continuous("v", shape=(nh, nd), lb=-np.inf, ub=np.inf)
    w = m.continuous("w", shape=(nh, nc), lb=-np.inf, ub=np.inf)
    uc = m.continuous("uc", shape=(nh, nc, nd), lb=-np.inf, ub=np.inf)
    Duc = m.continuous("Duc", shape=(nh, nc, nd), lb=-np.inf, ub=np.inf)
    m.minimize(1.0 + 0.0 * v[0, 0])  # constant objective: a feasibility problem
    for j in range(nc):
        for s in range(1, nd + 1):  # 1-based s, k as in the source
            m.subject_to(
                uc[:, j, s - 1]
                == v[:, s - 1]
                + h * sum(w[:, k - 1] * (rho[j] ** k / _fact(k)) for k in range(1, nc + 1))
            )
            m.subject_to(
                Duc[:, j, s - 1]
                == sum(
                    v[:, k - 1] * ((rho[j] * h) ** (k - s) / _fact(k - s)) for k in range(s, nd + 1)
                )
                + h ** (nd - s + 1)
                * sum(
                    w[:, k - 1] * (rho[j] ** (k + nd - s) / _fact(k + nd - s))
                    for k in range(1, nc + 1)
                )
            )
    m.subject_to(v[0, 0] == 0.0)  # bc_1
    m.subject_to(v[0, 1] == 0.0)  # bc_2
    m.subject_to(  # bc_3
        sum(v[nh - 1, k - 1] * (h ** (k - 1) / _fact(k - 1)) for k in range(1, nd + 1))
        + h**nd * sum(w[nh - 1, k - 1] / _fact(k + nd - 1) for k in range(1, nc + 1))
        == 1.0
    )
    m.subject_to(  # bc_4
        sum(v[nh - 1, k - 1] * (h ** (k - 2) / _fact(k - 2)) for k in range(2, nd + 1))
        + h ** (nd - 1) * sum(w[nh - 1, k - 1] / _fact(k + nd - 2) for k in range(1, nc + 1))
        == 0.0
    )
    for s in range(1, nd + 1):  # continuity
        m.subject_to(
            sum(v[: nh - 1, k - 1] * (h ** (k - s) / _fact(k - s)) for k in range(s, nd + 1))
            + h ** (nd - s + 1)
            * sum(w[: nh - 1, k - 1] / _fact(k + nd - s) for k in range(1, nc + 1))
            == v[1:, s - 1]
        )
    for j in range(nc):  # collocation
        m.subject_to(
            sum(w[:, k - 1] * (rho[j] ** (k - 1) / _fact(k - 1)) for k in range(1, nc + 1))
            == R * (Duc[:, j, 1] * Duc[:, j, 2] - Duc[:, j, 0] * Duc[:, j, 3])
        )
    start = {
        v: v0,
        w: np.zeros((nh, nc)),
        uc: np.repeat(v0[:, None, :], nc, axis=1),
        Duc: np.zeros((nh, nc, nd)),
    }
    return m, start


# ------------------------------------------------- collocation parameter estimation
def _itau(tau, h, nh):
    # Julia: itau = Int[min(nh, floor(tau[i]/h)+1)] (1-based) -> 0-based row index
    return [min(nh, int(math.floor(tj / h)) + 1) - 1 for tj in tau]


def _v0_from_measurements(nh, ne, itau, first_row, z):
    """The source's three-loop v0 fill, with ``itau`` 0-based."""
    v0 = np.zeros((nh, ne))
    v0[: itau[0] + 1, :] = first_row
    for j in range(1, len(itau)):
        v0[itau[j - 1] + 1 : itau[j] + 1, :] = z[j]
    v0[itau[-1] + 1 :, :] = z[-1]
    return v0


def _collocation_block(m, nh, nc, ne, h, rho, tau, z, itau, v, w, uc, Duc):
    """Objective (squared measurement error) and the shared collocation rows."""
    t = [i * h for i in range(nh + 1)]
    err = []
    for j in range(len(tau)):
        row = itau[j]
        for s in range(ne):
            e = v[row, s] - float(z[j][s])
            for k in range(1, nc + 1):
                coef = (tau[j] - t[row]) ** k / (_fact(k) * h ** (k - 1))
                if coef:
                    e = e + coef * w[row, k - 1, s]
            err.append(e**2)
    m.minimize(sum(err))
    for j in range(nc):
        for s in range(ne):
            m.subject_to(
                uc[:, j, s]
                == v[:, s]
                + h * sum(w[:, k - 1, s] * (rho[j] ** k / _fact(k)) for k in range(1, nc + 1))
            )
            m.subject_to(
                Duc[:, j, s]
                == sum(
                    w[:, k - 1, s] * (rho[j] ** (k - 1) / _fact(k - 1)) for k in range(1, nc + 1)
                )
            )
    for s in range(ne):  # continuity
        m.subject_to(
            v[: nh - 1, s] + sum(w[: nh - 1, j - 1, s] * (h / _fact(j)) for j in range(1, nc + 1))
            == v[1:, s]
        )


def _collocation_vars(m, nh, nc, ne):
    v = m.continuous("v", shape=(nh, ne), lb=-np.inf, ub=np.inf)
    w = m.continuous("w", shape=(nh, nc, ne), lb=-np.inf, ub=np.inf)
    uc = m.continuous("uc", shape=(nh, nc, ne), lb=-np.inf, ub=np.inf)
    Duc = m.continuous("Duc", shape=(nh, nc, ne), lb=-np.inf, ub=np.inf)
    return v, w, uc, Duc


def _collocation_start(v, w, uc, Duc, v0, nc):
    nh, ne = v0.shape
    return {
        v: v0,
        w: np.zeros((nh, nc, ne)),
        uc: np.repeat(v0[:, None, :], nc, axis=1),
        Duc: np.zeros((nh, nc, ne)),
    }


def pinene(nh: int):
    nc, ne, np_ = 3, 5, 5
    rho = [0.11270166537926, 0.5, 0.88729833462074]
    bc = [100.0, 0.0, 0.0, 0.0, 0.0]
    tau = [1230.0, 3060.0, 4920.0, 7800.0, 10680.0, 15030.0, 22620.0, 36420.0]
    z = [
        [88.35, 7.3, 2.3, 0.4, 1.75],
        [76.4, 15.6, 4.5, 0.7, 2.8],
        [65.1, 23.1, 5.3, 1.1, 5.8],
        [50.4, 32.9, 6.0, 1.5, 9.3],
        [37.5, 42.7, 6.0, 1.9, 12.0],
        [25.9, 49.1, 5.9, 2.2, 17.0],
        [14.0, 57.4, 5.1, 2.6, 21.0],
        [4.5, 63.1, 3.8, 2.9, 25.7],
    ]
    h = tau[-1] / nh
    itau = _itau(tau, h, nh)
    v0 = _v0_from_measurements(nh, ne, itau, bc, z)
    m = dm.Model("cops_pinene")
    th = m.continuous("theta", shape=(np_,), lb=0.0, ub=np.inf)
    v, w, uc, Duc = _collocation_vars(m, nh, nc, ne)
    _collocation_block(m, nh, nc, ne, h, rho, tau, z, itau, v, w, uc, Duc)
    for s in range(ne):
        m.subject_to(v[0, s] == bc[s])
    for j in range(nc):
        u = [uc[:, j, s] for s in range(ne)]
        m.subject_to(Duc[:, j, 0] == -(th[0] + th[1]) * u[0])
        m.subject_to(Duc[:, j, 1] == th[0] * u[0])
        m.subject_to(Duc[:, j, 2] == th[1] * u[0] - (th[2] + th[3]) * u[2] + th[4] * u[4])
        m.subject_to(Duc[:, j, 3] == th[2] * u[2])
        m.subject_to(Duc[:, j, 4] == th[3] * u[2] - th[4] * u[4])
    start = _collocation_start(v, w, uc, Duc, v0, nc)
    start[th] = np.zeros(np_)
    return m, start


def gasoil(nh: int):
    nc, ne, np_ = 4, 2, 3
    rho = [0.06943184420297, 0.33000947820757, 0.66999052179243, 0.93056815579703]
    bc = [1.0, 1.0]  # bc[s] for s = 1:ne of the source's [1, 1, 2, 0]
    tau = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.150, 0.175, 0.20, 0.225, 0.250, 0.30]
    tau += [0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95]
    z = [
        [1.0000, 0.0],
        [0.8105, 0.2000],
        [0.6208, 0.2886],
        [0.5258, 0.3010],
        [0.4345, 0.3215],
        [0.3903, 0.3123],
        [0.3342, 0.2716],
        [0.3034, 0.2551],
        [0.2735, 0.2258],
        [0.2405, 0.1959],
        [0.2283, 0.1789],
        [0.2071, 0.1457],
        [0.1669, 0.1198],
        [0.1530, 0.0909],
        [0.1339, 0.0719],
        [0.1265, 0.0561],
        [0.1200, 0.0460],
        [0.0990, 0.0280],
        [0.0870, 0.0190],
        [0.0770, 0.0140],
        [0.0690, 0.0100],
    ]
    h = tau[-1] / nh
    itau = _itau(tau, h, nh)
    v0 = _v0_from_measurements(nh, ne, itau, bc, z)
    m = dm.Model("cops_gasoil")
    th = m.continuous("theta", shape=(np_,), lb=0.0, ub=np.inf)
    v, w, uc, Duc = _collocation_vars(m, nh, nc, ne)
    _collocation_block(m, nh, nc, ne, h, rho, tau, z, itau, v, w, uc, Duc)
    for s in range(ne):
        m.subject_to(v[0, s] == z[0][s])
    for j in range(nc):
        u1, u2 = uc[:, j, 0], uc[:, j, 1]
        m.subject_to(Duc[:, j, 0] == -(th[0] + th[2]) * u1**2)
        m.subject_to(Duc[:, j, 1] == th[0] * u1**2 - th[1] * u2)
    start = _collocation_start(v, w, uc, Duc, v0, nc)
    start[th] = np.zeros(np_)
    return m, start


def marine(nh: int):
    nc, ne = 1, 8
    rho = [0.5]
    tau = list(np.linspace(0.0, 10.0, 21))
    z = [
        [20000.0, 17000.0, 10000.0, 15000.0, 12000.0, 9000.0, 7000.0, 3000.0],
        [12445.0, 15411.0, 13040.0, 13338.0, 13484.0, 8426.0, 6615.0, 4022.0],
        [7705.0, 13074.0, 14623.0, 11976.0, 12453.0, 9272.0, 6891.0, 5020.0],
        [4664.0, 8579.0, 12434.0, 12603.0, 11738.0, 9710.0, 6821.0, 5722.0],
        [2977.0, 7053.0, 11219.0, 11340.0, 13665.0, 8534.0, 6242.0, 5695.0],
        [1769.0, 5054.0, 10065.0, 11232.0, 12112.0, 9600.0, 6647.0, 7034.0],
        [943.0, 3907.0, 9473.0, 10334.0, 11115.0, 8826.0, 6842.0, 7348.0],
        [581.0, 2624.0, 7421.0, 10297.0, 12427.0, 8747.0, 7199.0, 7684.0],
        [355.0, 1744.0, 5369.0, 7748.0, 10057.0, 8698.0, 6542.0, 7410.0],
        [223.0, 1272.0, 4713.0, 6869.0, 9564.0, 8766.0, 6810.0, 6961.0],
        [137.0, 821.0, 3451.0, 6050.0, 8671.0, 8291.0, 6827.0, 7525.0],
        [87.0, 577.0, 2649.0, 5454.0, 8430.0, 7411.0, 6423.0, 8388.0],
        [49.0, 337.0, 2058.0, 4115.0, 7435.0, 7627.0, 6268.0, 7189.0],
        [32.0, 228.0, 1440.0, 3790.0, 6474.0, 6658.0, 5859.0, 7467.0],
        [17.0, 168.0, 1178.0, 3087.0, 6524.0, 5880.0, 5562.0, 7144.0],
        [11.0, 99.0, 919.0, 2596.0, 5360.0, 5762.0, 4480.0, 7256.0],
        [7.0, 65.0, 647.0, 1873.0, 4556.0, 5058.0, 4944.0, 7538.0],
        [4.0, 44.0, 509.0, 1571.0, 4009.0, 4527.0, 4233.0, 6649.0],
        [2.0, 27.0, 345.0, 1227.0, 3677.0, 4229.0, 3805.0, 6378.0],
        [1.0, 20.0, 231.0, 934.0, 3197.0, 3695.0, 3159.0, 6454.0],
        [1.0, 12.0, 198.0, 707.0, 2562.0, 3163.0, 3232.0, 5566.0],
    ]
    h = tau[-1] / nh
    itau = _itau(tau, h, nh)
    v0 = _v0_from_measurements(nh, ne, itau, z[0], z)
    m = dm.Model("cops_marine")
    g = m.continuous("g", shape=(ne - 1,), lb=0.0, ub=np.inf)
    mm = m.continuous("m", shape=(ne,), lb=0.0, ub=np.inf)
    v, w, uc, Duc = _collocation_vars(m, nh, nc, ne)
    _collocation_block(m, nh, nc, ne, h, rho, tau, z, itau, v, w, uc, Duc)
    u = [uc[:, 0, s] for s in range(ne)]
    m.subject_to(Duc[:, 0, 0] == -(mm[0] + g[0]) * u[0])
    m.subject_to(Duc[:, 0, ne - 1] == g[ne - 2] * u[ne - 2] - mm[ne - 1] * u[ne - 1])
    for s in range(1, ne - 1):
        m.subject_to(Duc[:, 0, s] == g[s - 1] * u[s - 1] - (mm[s] + g[s]) * u[s])
    start = _collocation_start(v, w, uc, Duc, v0, nc)
    start[g] = np.zeros(ne - 1)  # no start in the source -> Ipopt's 0
    start[mm] = np.zeros(ne)
    return m, start


def methanol(nh: int):
    ne, np_, nc = 3, 5, 3
    rho = [0.11270166537926, 0.5, 0.88729833462074]
    tau = [0, 0.050, 0.065, 0.080, 0.123, 0.233, 0.273, 0.354, 0.397, 0.418, 0.502]
    tau += [0.553, 0.681, 0.750, 0.916, 0.937, 1.122]
    z = [
        [1.0000, 0.0, 0.0],
        [0.7085, 0.1621, 0.0811],
        [0.5971, 0.1855, 0.0965],
        [0.5537, 0.1989, 0.1198],
        [0.3684, 0.2845, 0.1535],
        [0.1712, 0.3491, 0.2097],
        [0.1198, 0.3098, 0.2628],
        [0.0747, 0.3576, 0.2467],
        [0.0529, 0.3347, 0.2884],
        [0.0415, 0.3388, 0.2757],
        [0.0261, 0.3557, 0.3167],
        [0.0208, 0.3483, 0.2954],
        [0.0085, 0.3836, 0.2950],
        [0.0053, 0.3611, 0.2937],
        [0.0019, 0.3609, 0.2831],
        [0.0018, 0.3485, 0.2846],
        [0.0006, 0.3698, 0.2899],
    ]
    bc = [1.0, 0.0, 0.0]
    h = tau[-1] / nh
    itau = _itau(tau, h, nh)
    v0 = np.full((nh, ne), 0.001)  # the source overwrites v0 with 0.001
    m = dm.Model("cops_methanol")
    th = m.continuous("theta", shape=(np_,), lb=0.0, ub=np.inf)
    v, w, uc, Duc = _collocation_vars(m, nh, nc, ne)
    _collocation_block(m, nh, nc, ne, h, rho, tau, z, itau, v, w, uc, Duc)
    for s in range(ne):
        m.subject_to(v[0, s] == bc[s])
    for j in range(nc):
        u1, u2 = uc[:, j, 0], uc[:, j, 1]
        den = (th[1] + th[4]) * u1 + u2
        m.subject_to(Duc[:, j, 0] == -(2 * th[1] - (th[0] * u2) / den + th[2] + th[3]) * u1)
        m.subject_to(Duc[:, j, 1] == (th[0] * u1 * (th[1] * u1 - u2)) / den + th[2] * u1)
        m.subject_to(Duc[:, j, 2] == (th[0] * u1 * (u2 + th[4] * u1)) / den + th[3] * u1)
    start = _collocation_start(v, w, uc, Duc, v0, nc)
    start[th] = np.ones(np_)
    return m, start


BUILDERS = {
    "bearing": bearing,
    "camshape": camshape,
    "catmix": catmix,
    "chain": chain,
    "channel": channel,
    "elec": elec,
    "gasoil": gasoil,
    "glider": glider,
    "marine": marine,
    "methanol": methanol,
    "minsurf": minsurf,
    "pinene": pinene,
    "polygon": polygon,
    "robot": robot,
    "rocket": rocket,
    "steering": steering,
    "torsion": torsion,
}
assert set(BUILDERS) == set(COPS_REFERENCE)
