"""
Synthetic gas network MINLP benchmark for discopt vs GAMS comparison.

Topology (8 nodes, 6 pipes, 2 compressor stations)::

    s1 --pipe0--> n1 --[CS1]--> n2 --pipe2--\\
                                              n5 --pipe4--> d1
    s2 --pipe1--> n3 --[CS2]--> n4 --pipe3--/    \\--pipe5--> d2

Sources s1, s2 supply gas at fixed pressures. Demands d1, d2 require
specified mass flow rates at minimum delivery pressures. Node n5 is a
mixing/splitting junction. Compressor stations CS1 and CS2 each have a
binary on/off decision variable.

MINLP structure::

    - 2 binary variables  (compressor on/off)
    - ~15 continuous vars  (pressures, flows, compression ratios)
    - Nonconvex constraints: Weymouth pressure-drop (f^2 = C*(p_in^2 - p_out^2))
    - Nonconvex objective:  compressor power (w * (beta^kappa - 1))
    - Big-M coupling:       flow and ratio linked to on/off binaries

Expected optimal: both compressors on, equal flow split (~35 kg/s each),
total cost ~3.07 MW. Single-compressor solutions cost ~9.4 MW.

Reference: https://github.com/jkitchin/discopt/issues/15
"""

import numpy as np

import discopt.modeling as dm

# ---------------------------------------------------------------------------
# Network parameters
# ---------------------------------------------------------------------------

# Source pressures (bar)
P_S1 = 50.0
P_S2 = 50.0

# Demand mass flows (kg/s)
DEMAND_D1 = 40.0
DEMAND_D2 = 30.0
TOTAL_DEMAND = DEMAND_D1 + DEMAND_D2  # 70 kg/s

# Minimum delivery pressures (bar)
P_D1_MIN = 45.0
P_D2_MIN = 45.0

# Internal node pressure bounds (bar)
P_MIN = 30.0
P_MAX = 70.0

# Pipe Weymouth constants: f^2 = C * (p_in^2 - p_out^2)
# Units: (kg/s)^2 / bar^2.  Derived from D^(16/3) / (friction * L * T * Z).
PIPE_C = {
    0: 3.5,  # s1 -> n1  (50 km, 500 mm diameter)
    1: 3.5,  # s2 -> n3  (50 km, 500 mm diameter)
    2: 5.0,  # n2 -> n5  (30 km, 600 mm diameter)
    3: 5.0,  # n4 -> n5  (30 km, 600 mm diameter)
    4: 3.5,  # n5 -> d1  (50 km, 500 mm diameter)
    5: 3.5,  # n5 -> d2  (40 km, 500 mm diameter)
}

# Compressor parameters
BETA_MAX = 2.0  # maximum compression ratio
W_CS_MAX = 100.0  # maximum compressor throughput (kg/s)
KAPPA = 0.2857  # (gamma - 1) / gamma, for gamma = 1.4 (natural gas)

# Power coefficient: P_MW = K_POWER * w * (beta^KAPPA - 1)
# Derived from cp * T_in / eta / 1e6, with cp=2340 J/(kg*K), T=283 K, eta=0.8
K_POWER = 0.828  # MW per (kg/s) when (beta^kappa - 1) = 1

# Fixed cost of running a compressor (MW-equivalent, represents startup/standby)
FIXED_COST = 0.5


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_gas_network_minlp() -> dm.Model:
    """Build synthetic gas network MINLP.

    Returns a discopt Model ready for solve().
    """
    m = dm.Model("gas_network_minlp")

    # -- Node pressures (bar) ------------------------------------------------
    # Internal nodes: n1, n2, n3, n4, n5 (indices 0..4)
    p = m.continuous("p_node", shape=(5,), lb=P_MIN, ub=P_MAX)
    # n1=p[0], n2=p[1], n3=p[2], n4=p[3], n5=p[4]

    # Delivery-point pressures
    p_d1 = m.continuous("p_d1", lb=P_D1_MIN, ub=P_MAX)
    p_d2 = m.continuous("p_d2", lb=P_D2_MIN, ub=P_MAX)

    # -- Pipe flows (kg/s), non-negative (directions fixed by topology) ------
    f = m.continuous("f_pipe", shape=(6,), lb=0.0, ub=150.0)

    # -- Compressor variables ------------------------------------------------
    w_cs = m.continuous("w_cs", shape=(2,), lb=0.0, ub=W_CS_MAX)
    beta = m.continuous("beta", shape=(2,), lb=1.0, ub=BETA_MAX)
    y = m.binary("y_cs", shape=(2,))

    # -- Objective: minimize total compressor power --------------------------
    power_1 = K_POWER * w_cs[0] * (beta[0] ** KAPPA - 1.0)
    power_2 = K_POWER * w_cs[1] * (beta[1] ** KAPPA - 1.0)
    m.minimize(power_1 + power_2 + FIXED_COST * (y[0] + y[1]))

    # -- Weymouth pressure-drop on each pipe ---------------------------------
    # f^2 = C * (p_from^2 - p_to^2)
    m.subject_to(f[0] ** 2 == PIPE_C[0] * (P_S1**2 - p[0] ** 2), name="weymouth_s1_n1")
    m.subject_to(f[1] ** 2 == PIPE_C[1] * (P_S2**2 - p[2] ** 2), name="weymouth_s2_n3")
    m.subject_to(f[2] ** 2 == PIPE_C[2] * (p[1] ** 2 - p[4] ** 2), name="weymouth_n2_n5")
    m.subject_to(f[3] ** 2 == PIPE_C[3] * (p[3] ** 2 - p[4] ** 2), name="weymouth_n4_n5")
    m.subject_to(f[4] ** 2 == PIPE_C[4] * (p[4] ** 2 - p_d1**2), name="weymouth_n5_d1")
    m.subject_to(f[5] ** 2 == PIPE_C[5] * (p[4] ** 2 - p_d2**2), name="weymouth_n5_d2")

    # -- Compressor pressure ratio -------------------------------------------
    m.subject_to(p[1] == beta[0] * p[0], name="cs1_pressure")
    m.subject_to(p[3] == beta[1] * p[2], name="cs2_pressure")

    # -- Big-M: flow and ratio coupled to on/off binary ----------------------
    m.subject_to(w_cs[0] <= W_CS_MAX * y[0], name="cs1_flow_bigM")
    m.subject_to(w_cs[1] <= W_CS_MAX * y[1], name="cs2_flow_bigM")
    m.subject_to(beta[0] <= 1.0 + (BETA_MAX - 1.0) * y[0], name="cs1_ratio_bigM")
    m.subject_to(beta[1] <= 1.0 + (BETA_MAX - 1.0) * y[1], name="cs2_ratio_bigM")

    # -- Flow conservation at each node --------------------------------------
    m.subject_to(f[0] == w_cs[0], name="balance_n1")  # s1 feed = CS1 inlet
    m.subject_to(w_cs[0] == f[2], name="balance_n2")  # CS1 outlet = pipe to n5
    m.subject_to(f[1] == w_cs[1], name="balance_n3")  # s2 feed = CS2 inlet
    m.subject_to(w_cs[1] == f[3], name="balance_n4")  # CS2 outlet = pipe to n5
    m.subject_to(f[2] + f[3] == f[4] + f[5], name="balance_n5")  # mixing node

    # -- Fixed demand flows --------------------------------------------------
    m.subject_to(f[4] == DEMAND_D1, name="demand_d1")
    m.subject_to(f[5] == DEMAND_D2, name="demand_d2")

    # -- At least one compressor must operate --------------------------------
    m.subject_to(y[0] + y[1] >= 1, name="min_one_cs")

    return m


# ---------------------------------------------------------------------------
# Analytical reference solution (both compressors on, equal split)
# ---------------------------------------------------------------------------


def gas_network_reference_solution():
    """Compute the equal-split reference point for verification.

    Returns a dict with keys: w_each, p_n1, p_n2, p_n5, beta,
    power_each, total_cost.
    """
    w_each = TOTAL_DEMAND / 2.0  # 35 kg/s

    # Pipe 0 (s1 -> n1): p_n1
    p_n1_sq = P_S1**2 - w_each**2 / PIPE_C[0]
    p_n1 = np.sqrt(p_n1_sq)

    # Pipe 4,5 determine p_n5 (binding constraint is pipe 4, higher flow)
    p_n5_sq = max(
        P_D1_MIN**2 + DEMAND_D1**2 / PIPE_C[4],
        P_D2_MIN**2 + DEMAND_D2**2 / PIPE_C[5],
    )
    p_n5 = np.sqrt(p_n5_sq)

    # Pipe 2 (n2 -> n5): p_n2
    p_n2_sq = p_n5_sq + w_each**2 / PIPE_C[2]
    p_n2 = np.sqrt(p_n2_sq)

    beta_val = p_n2 / p_n1
    power_each = K_POWER * w_each * (beta_val**KAPPA - 1.0)
    total = 2 * power_each + 2 * FIXED_COST

    return {
        "w_each": w_each,
        "p_n1": p_n1,
        "p_n2": p_n2,
        "p_n5": p_n5,
        "beta": beta_val,
        "power_each": power_each,
        "total_cost": total,
    }


# ---------------------------------------------------------------------------
# Main: build, solve, report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Print reference solution
    ref = gas_network_reference_solution()
    print("=== Analytical reference (equal split, both CS on) ===")
    print(f"  Flow per CS:     {ref['w_each']:.1f} kg/s")
    print(f"  p_n1 = p_n3:     {ref['p_n1']:.2f} bar")
    print(f"  p_n2 = p_n4:     {ref['p_n2']:.2f} bar")
    print(f"  p_n5:            {ref['p_n5']:.2f} bar")
    print(f"  beta:            {ref['beta']:.4f}")
    print(f"  Power per CS:    {ref['power_each']:.4f} MW")
    print(f"  Total cost:      {ref['total_cost']:.4f} MW")
    print()

    # Build and solve
    model = build_gas_network_minlp()
    print(model)
    print()

    result = model.solve(time_limit=60, gap_tolerance=1e-4)
    assert hasattr(result, "status")  # SolveResult, not Iterator
    print("\n=== Solve result ===")
    print(f"  Status:      {result.status}")  # type: ignore[union-attr]
    print(f"  Objective:   {result.objective:.4f} MW")  # type: ignore[union-attr]
    print(f"  Bound:       {result.bound:.4f} MW")  # type: ignore[union-attr]
    print(f"  Gap:         {result.gap:.2%}" if result.gap is not None else "  Gap: N/A")  # type: ignore[union-attr]
    print(f"  Nodes:       {result.node_count}")  # type: ignore[union-attr]
    print(f"  Wall time:   {result.wall_time:.3f} s")  # type: ignore[union-attr]
    print()

    # Extract solution
    y_val = result.value(model._vars_by_name["y_cs"])  # type: ignore[union-attr, attr-defined]
    w_val = result.value(model._vars_by_name["w_cs"])  # type: ignore[union-attr, attr-defined]
    beta_val = result.value(model._vars_by_name["beta"])  # type: ignore[union-attr, attr-defined]
    p_val = result.value(model._vars_by_name["p_node"])  # type: ignore[union-attr, attr-defined]

    print("=== Solution details ===")
    for i in range(2):
        status = "ON" if y_val[i] > 0.5 else "OFF"
        print(f"  CS{i + 1}: {status}  flow={w_val[i]:.2f} kg/s  beta={beta_val[i]:.4f}")
    print("  Node pressures: " + ", ".join(f"n{i + 1}={p_val[i]:.2f}" for i in range(5)))
