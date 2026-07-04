"""
discopt API Examples

Demonstrates the modeling syntax across real problem types,
from textbook examples to industrial-scale formulations.
Each example is self-contained and runnable (once the solver backend exists).
"""

import numpy as np

# The standard import
import discopt.modeling as dm

# ═══════════════════════════════════════════════════════════════
# EXAMPLE 1: Simple MINLP (textbook)
#
#   minimize    x₁² + x₂² + x₃
#   subject to  x₁ + x₂ ≥ 1
#               x₁² + x₂ ≤ 3
#               x₃ ∈ {0, 1}
#               x₁, x₂ ∈ [0, 5]
# ═══════════════════════════════════════════════════════════════


def example_simple_minlp():
    m = dm.Model("textbook")

    x1 = m.continuous("x1", lb=0, ub=5)
    x2 = m.continuous("x2", lb=0, ub=5)
    x3 = m.binary("x3")

    m.minimize(x1**2 + x2**2 + x3)
    m.subject_to(x1 + x2 >= 1)
    m.subject_to(x1**2 + x2 <= 3)

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 2: Pooling Problem (Haverly)
#
# Classic nonconvex blending problem. Three pools, bilinear
# quality mixing constraints. This is a key benchmark class
# where GPU batching should provide advantage.
# ═══════════════════════════════════════════════════════════════


def example_pooling_haverly():
    m = dm.Model("haverly_pooling")

    # Sources, pools, products

    # Flow variables
    # y[i] = flow from source i to pool (sources 0,1 feed pool)
    y = m.continuous("y_source_to_pool", shape=(2,), lb=0, ub=100)
    # x[j] = flow from pool to product j
    x = m.continuous("x_pool_to_product", shape=(2,), lb=0, ub=100)
    # z = direct flow from source 2 to product 1
    z = m.continuous("z_direct", lb=0, ub=100)

    # Pool quality (bilinear: this is what makes it nonconvex). p is the pool
    # sulfur CONCENTRATION (quality), bounded by the source qualities [1, 3].
    # Modelling p as the concentration (not the sulfur *mass*) is what keeps the
    # product specs from vanishing when the pool is bypassed — see below (E1).
    p = m.continuous("pool_sulfur_concentration", lb=0.0, ub=3.0)

    # Source qualities
    quality = np.array([3.0, 1.0, 2.0])  # sulfur content

    # Objective: maximize profit
    revenue = np.array([9.0, 15.0])  # price per unit of product
    source_cost = np.array([6.0, 16.0, 10.0])
    m.maximize(
        revenue[0] * x[0]
        + revenue[1] * (x[1] + z)
        - source_cost[0] * y[0]
        - source_cost[1] * y[1]
        - source_cost[2] * z
    )

    # Pool mass balance
    m.subject_to(y[0] + y[1] == x[0] + x[1], name="pool_mass_balance")

    # Pool quality balance (bilinear): concentration * pool inflow = sulfur mass in.
    m.subject_to(
        p * (y[0] + y[1]) == quality[0] * y[0] + quality[1] * y[1],
        name="pool_quality_balance",
    )

    # Product quality specs: sulfur delivered <= spec * total flow to the product.
    # Product 0 draws only pool flow (at concentration p); product 1 draws the pool
    # stream x1 (at concentration p) plus the direct source-2 stream z (concentration
    # quality[2]). Using the concentration p directly — rather than clearing a
    # 1/(y0+y1) denominator by multiplying the spec through by the pool flow — keeps
    # both constraints binding even when the pool is bypassed (pool flow -> 0). The
    # original mass form multiplied through by (y0+y1), so an all-direct solution
    # satisfied the spec vacuously (0 <= 0) and shipped 2 %-sulfur product against a
    # 1.5 % spec, certifying a false optimum of 500 (true Haverly-I optimum 400) (E1).
    m.subject_to(p * x[0] <= 2.5 * x[0], name="product0_sulfur_spec")
    m.subject_to(p * x[1] + quality[2] * z <= 1.5 * (x[1] + z), name="product1_sulfur_spec")

    # Product demand
    m.subject_to(x[0] <= 100, name="product0_demand")
    m.subject_to(x[1] + z <= 200, name="product1_demand")

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 3: Process Network Synthesis
#
# Choose which processing units to build (binary) and optimize
# continuous flows through the network. Classic MINLP with
# indicator constraints.
# ═══════════════════════════════════════════════════════════════


def example_process_synthesis():
    m = dm.Model("process_synthesis")

    n_units = 4
    n_streams = 6

    # Binary: which units to build
    y = m.binary("build", shape=(n_units,))

    # Continuous: flow rates through streams
    f = m.continuous("flow", shape=(n_streams,), lb=0, ub=1000)

    # Continuous: unit operating levels
    x = m.continuous("capacity", shape=(n_units,), lb=0, ub=500)

    # Cost data
    fixed_cost = np.array([1000, 1500, 800, 1200])
    variable_cost = np.array([2.5, 3.0, 1.8, 2.2])
    revenue_per_unit = 10.0

    # Objective: maximize profit
    m.maximize(
        revenue_per_unit * f[5]  # revenue from final product
        - dm.sum(lambda i: fixed_cost[i] * y[i], over=range(n_units))
        - dm.sum(lambda i: variable_cost[i] * x[i], over=range(n_units))
    )

    # Feed availability
    m.subject_to(f[0] <= 500, name="feed_limit")

    # Mass balance at each node
    m.subject_to(f[0] == f[1] + f[2], name="splitter")
    m.subject_to(f[1] == x[0], name="unit0_feed")
    m.subject_to(f[2] == x[1], name="unit1_feed")
    m.subject_to(x[0] * 0.9 + x[1] * 0.8 == f[3], name="mixer1")
    m.subject_to(f[3] == x[2] + x[3], name="splitter2")
    m.subject_to(x[2] * 0.95 + x[3] * 0.85 == f[5], name="mixer2")

    # Indicator constraints: unit operates only if built
    for i in range(n_units):
        m.if_then(
            y[i],
            [
                x[i] >= 10,  # Minimum operating level
                x[i] <= 500,  # Maximum capacity
            ],
            name=f"unit{i}_active",
        )

        # If not built, capacity is zero
        m.subject_to(x[i] <= 500 * y[i], name=f"unit{i}_bigM")

    # At least 2 units must be built
    m.subject_to(dm.sum(y) >= 2, name="min_units")

    # Nonlinear yield: unit 2 has diminishing returns
    m.subject_to(f[5] <= dm.log(1 + x[2]) * 100 + x[3] * 0.85, name="nonlinear_yield")

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 4: Portfolio Optimization with Cardinality Constraints
#
# Minimize risk (quadratic) subject to return target and
# cardinality constraint (invest in at most K assets).
# This is MIQCQP.
# ═══════════════════════════════════════════════════════════════


def example_portfolio():
    n_assets = 20
    np.random.seed(42)

    # Generate random covariance matrix and expected returns
    A = np.random.randn(n_assets, n_assets) * 0.1
    cov = A.T @ A + 0.01 * np.eye(n_assets)  # Positive definite
    expected_return = np.random.uniform(0.02, 0.15, n_assets)
    target_return = 0.08
    max_assets = 8

    m = dm.Model("portfolio")

    # Continuous: portfolio weights
    w = m.continuous("weight", shape=(n_assets,), lb=0, ub=0.3)

    # Binary: whether we invest in each asset
    z = m.binary("invest", shape=(n_assets,))

    # Minimize portfolio variance (quadratic objective)
    # w^T Σ w — this is a quadratic form
    m.minimize(
        dm.sum(
            lambda i: dm.sum(lambda j: cov[i, j] * w[i] * w[j], over=range(n_assets)),
            over=range(n_assets),
        )
    )

    # Return target
    m.subject_to(
        dm.sum(lambda i: expected_return[i] * w[i], over=range(n_assets)) >= target_return,
        name="min_return",
    )

    # Weights sum to 1
    m.subject_to(dm.sum(w) == 1.0, name="budget")

    # Cardinality: invest in at most K assets
    m.subject_to(dm.sum(z) <= max_assets, name="cardinality")

    # Linking: w[i] > 0 only if z[i] = 1
    m.subject_to([w[i] <= 0.3 * z[i] for i in range(n_assets)], name="linking")

    # Minimum investment if selected
    m.subject_to([w[i] >= 0.02 * z[i] for i in range(n_assets)], name="min_invest")

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 5: Chemical Reactor Design
#
# Nonlinear experiment: Arrhenius kinetics, heat balance,
# integer number of reactor stages. Signomial terms.
# ═══════════════════════════════════════════════════════════════


def example_reactor_design():
    m = dm.Model("reactor_design")

    # Design variables
    T = m.continuous("temperature", shape=(3,), lb=300, ub=800)  # K
    V = m.continuous("volume", shape=(3,), lb=0.1, ub=10)  # m³
    F = m.continuous("feed_rate", lb=0.1, ub=100)  # mol/s
    m.integer("n_stages", lb=1, ub=5)

    # Kinetic parameters
    k0 = 1e6  # pre-exponential factor
    Ea = 50000.0  # activation energy (J/mol)
    R = 8.314  # gas constant

    # Minimize total reactor volume (cost proxy)
    m.minimize(dm.sum(V))

    # Conversion target: 95% of feed must react
    # Arrhenius kinetics at each stage (nonlinear!)
    for i in range(3):
        rate_constant = k0 * dm.exp(-Ea / (R * T[i]))
        m.subject_to(
            rate_constant * V[i] >= F * 0.3,  # Each stage converts ≥30%
            name=f"conversion_stage{i}",
        )

    # Heat balance: adiabatic temperature rise per stage.
    # ΔT = -dH·X/Cp is intensive (independent of the feed rate F), so the original
    # `F/(Cp*F)` was a needless division-by-variable that merely cancels to 1/Cp.
    # With the original dH=-80000 the rise is 320 K/stage, which forces
    # T[2] >= 940 K against the 750 K material limit below — the model was provably
    # infeasible. A moderate exotherm keeps the 3-stage cascade within limits (E3).
    Cp = 75.0  # J/(mol·K)
    dH = -30000.0  # J/mol (exothermic)
    conversion = 0.3  # fractional conversion per stage
    delta_T = -dH * conversion / Cp  # adiabatic rise per stage, K (= 120 K)
    for i in range(3):
        if i == 0:
            m.subject_to(
                T[i] <= 400,  # Feed temperature limit
                name="feed_temp",
            )
        else:
            # Temperature rises due to reaction (constant adiabatic increment).
            m.subject_to(T[i] == T[i - 1] + delta_T, name=f"heat_balance_stage{i}")

    # Material limit on temperature
    m.subject_to([T[i] <= 750 for i in range(3)], name="max_temperature")

    # Only n_stages stages are active (others have zero volume)
    for i in range(3):
        # V[i] = 0 if i >= n_stages — encoded via big-M
        # This couples the integer and continuous variables
        pass  # Would use indicator constraints in full implementation

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 6: Facility Location with Nonlinear Transportation
#
# Choose which warehouses to open (binary) and allocate
# customer demand. Transportation cost is nonlinear
# (economies of scale: cost per unit decreases with volume).
# ═══════════════════════════════════════════════════════════════


def example_facility_location():
    n_facilities = 5
    n_customers = 20
    np.random.seed(123)

    # Data
    fixed_cost = np.random.uniform(10000, 50000, n_facilities)
    capacity = np.random.uniform(500, 2000, n_facilities)
    demand = np.random.uniform(50, 200, n_customers)
    distance = np.random.uniform(10, 500, (n_facilities, n_customers))

    m = dm.Model("facility_location")

    # Binary: open facility
    y = m.binary("open", shape=(n_facilities,))

    # Continuous: shipment from facility i to customer j
    x = m.continuous("ship", shape=(n_facilities, n_customers), lb=0, ub=2000)

    # Objective: minimize fixed + transportation cost
    # Transportation has economies of scale: cost = distance * sqrt(shipment)
    transport_cost = dm.sum(
        lambda i: dm.sum(
            lambda j: distance[i, j] * dm.sqrt(x[i, j] + 1),  # +1 avoids sqrt(0) issues
            over=range(n_customers),
        ),
        over=range(n_facilities),
    )

    m.minimize(dm.sum(lambda i: fixed_cost[i] * y[i], over=range(n_facilities)) + transport_cost)

    # Demand satisfaction
    m.subject_to(
        [
            dm.sum(lambda i: x[i, j], over=range(n_facilities)) >= demand[j]
            for j in range(n_customers)
        ],
        name="demand",
    )

    # Capacity limits (active only if open)
    m.subject_to(
        [
            dm.sum(lambda j: x[i, j], over=range(n_customers)) <= capacity[i] * y[i]
            for i in range(n_facilities)
        ],
        name="capacity",
    )

    # At least 2 facilities must be open
    m.subject_to(dm.sum(y) >= 2, name="min_open")

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 7: Parametric Optimization with Sensitivity
#
# Demonstrates dm.Parameter for differentiating through the solve.
# Useful for: what-if analysis, bilevel optimization, pricing.
# ═══════════════════════════════════════════════════════════════


def example_parametric():
    m = dm.Model("parametric_blending")

    # Parameters: values that are fixed per solve but differentiable
    price_A = m.parameter("price_A", value=50.0)
    price_B = m.parameter("price_B", value=30.0)
    demand = m.parameter("demand", value=np.array([100.0, 150.0]))

    # Variables
    x = m.continuous("blend", shape=(2, 2), lb=0, ub=200)  # blend[source, product]
    y = m.binary("use_source", shape=(2,))

    # Minimize cost
    m.minimize(
        price_A * dm.sum(x[0, :]) + price_B * dm.sum(x[1, :]) + 1000 * dm.sum(y)  # fixed costs
    )

    # Meet demand for each product
    for j in range(2):
        m.subject_to(x[0, j] + x[1, j] >= demand[j], name=f"demand_product{j}")

    # Source activation
    for i in range(2):
        m.subject_to(dm.sum(x[i, :]) <= 300 * y[i], name=f"source{i}_activation")

    # Quality constraint (nonlinear)
    m.subject_to(
        0.3 * x[0, 0] + 0.7 * x[1, 0] >= 0.5 * (x[0, 0] + x[1, 0]), name="quality_product0"
    )

    # After solving:
    # result = m.solve(sensitivity=True)
    # dprice = result.gradient(price_A)   # How does optimal cost change if price_A increases?
    # ddemand = result.gradient(demand)   # Sensitivity to demand changes

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 8: Logical Constraints and Disjunctive Programming
#
# Demonstrates BooleanVar, propositional logic (land/lor/lnot),
# atleast/atmost/exactly, and GDP reformulation methods.
# ═══════════════════════════════════════════════════════════════


def example_logical_constraints():
    """GDP model with boolean variables and propositional logic.

    Project selection problem: choose a subset of projects
    subject to precedence, exclusivity, and budget constraints
    expressed as logical propositions.
    """
    m = dm.Model("project_selection")

    # Continuous: investment level for each project (if selected)
    invest = m.continuous("invest", shape=(4,), lb=0, ub=100)

    # Boolean: whether each project is selected
    active = m.boolean("active", shape=(4,))

    # Objective: maximize weighted return
    returns = np.array([12.0, 8.0, 15.0, 6.0])
    m.maximize(dm.sum(lambda i: returns[i] * invest[i], over=range(4)))

    # Budget constraint (at most 3 projects active). Propositional-logic
    # constraints go through m.logical(), not m.subject_to() (which takes
    # algebraic Constraints only) (E2).
    m.logical(dm.atmost(3, active), name="budget")

    # Project 3 requires project 0 (precedence): the implication a3 -> a0, i.e.
    # lor(~a3, a0). The original land(~a3, a0) instead *asserts* both ~a3 and a0
    # unconditionally (forcing project 3 off and project 0 on), not the
    # requirement being modelled (E2).
    m.logical(dm.lor(~active[3], active[0]), name="precedence")

    # At least one of projects 0 or 1 must be active
    m.logical(dm.lor(active[0], active[1]), name="require_one")

    # Investment only allowed if project is active (indicator constraints)
    for i in range(4):
        m.if_then(active[i].variable, [invest[i] >= 10, invest[i] <= 80])

    # Disjunction: project 2 is either high-intensity (≥50) or low-intensity (≤20).
    # Disjunct blocks are created with make_disjunct() (E2).
    d_high = m.make_disjunct("project2_high")
    d_high.subject_to(invest[2] >= 50)
    d_low = m.make_disjunct("project2_low")
    d_low.subject_to(invest[2] <= 20)
    m.add_disjunction([d_high, d_low], name="project2_mode")

    # Solve with default big-M:        m.solve(gdp_method="big-m")
    # Solve with LP-tightened big-M:   m.solve(gdp_method="mbigm")
    # Solve with convex hull:          m.solve(gdp_method="hull")
    # Solve with LOA decomposition:    m.solve(gdp_method="loa")

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 9: Neural Network Surrogate Optimization
#
# Embeds a trained feedforward NN as MINLP constraints and
# optimizes over it. Demonstrates NNFormulation with three
# strategies: relu_bigm, full_space, reduced_space.
# ═══════════════════════════════════════════════════════════════


def example_nn_surrogate():
    """Optimize over a trained NN surrogate using discopt.nn.

    A 2-input, 1-output network predicts process yield.
    We find the input conditions that maximize predicted yield
    subject to operating bounds.
    """
    from discopt.nn import NetworkDefinition, NNFormulation
    from discopt.nn.network import Activation, DenseLayer

    # Simulate a trained network (normally loaded from file / ONNX)
    np.random.seed(0)
    W1 = np.random.randn(2, 8) * 0.5
    b1 = np.random.randn(8) * 0.1
    W2 = np.random.randn(8, 4) * 0.5
    b2 = np.random.randn(4) * 0.1
    W3 = np.random.randn(4, 1) * 0.5
    b3 = np.random.randn(1) * 0.1

    net = NetworkDefinition(
        layers=[
            DenseLayer(W1, b1, Activation.RELU),
            DenseLayer(W2, b2, Activation.RELU),
            DenseLayer(W3, b3, Activation.LINEAR),
        ],
        input_bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
    )

    # Build optimization model
    m = dm.Model("nn_surrogate")

    # Embed the network using ReLU big-M MILP formulation
    nn = NNFormulation(m, net, strategy="relu_bigm", prefix="yield_model")
    nn.formulate()

    # Maximize predicted yield (NN output)
    m.maximize(nn.outputs[0])

    # Add operating constraints on inputs
    m.subject_to(nn.inputs[0] + nn.inputs[1] <= 1.5, name="combined_limit")

    # For smooth activations (sigmoid/tanh), use full_space with McCormick relaxations:
    #   nn = NNFormulation(m, net_smooth, strategy="full_space")
    # For small networks, use reduced_space (no intermediate variables):
    #   nn = NNFormulation(m, net_small, strategy="reduced_space")
    # Load from ONNX:
    #   from discopt.nn import load_onnx
    #   net = load_onnx("model.onnx", input_bounds=(lb, ub))

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 10: Import from Pyomo
#
# Shows the bridge from existing Pyomo models to discopt.
# ═══════════════════════════════════════════════════════════════


def example_pyomo_import():
    """
    # Assuming you have an existing Pyomo model:

    import pyomo.environ as pyo
    import discopt

    # Build Pyomo model (existing workflow)
    pyo_model = pyo.ConcreteModel()
    pyo_model.x = pyo.Var([1,2,3], bounds=(0, 10))
    pyo_model.y = pyo.Var([1,2], within=pyo.Binary)
    pyo_model.obj = pyo.Objective(
        expr=sum(pyo_model.x[i]**2 for i in [1,2,3])
             + sum(100*pyo_model.y[j] for j in [1,2])
    )
    pyo_model.c1 = pyo.Constraint(expr=pyo_model.x[1] + pyo_model.x[2] >= 5)
    pyo_model.c2 = pyo.Constraint(expr=pyo_model.x[3] * pyo_model.y[1] <= 8)

    # One-line import to discopt
    dm_model = dm.from_pyomo(pyo_model)

    # Now solve with discopt
    result = dm_model.solve()

    # Or use as discopt solver plugin directly in Pyomo:
    solver = pyo.SolverFactory('discopt')
    results = solver.solve(pyo_model)
    """
    print("Pyomo import example (requires pyomo and solver backend)")


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 11: Import from .nl file (AMPL)
#
# Standard interchange format for optimization problems.
# ═══════════════════════════════════════════════════════════════


def example_nl_import():
    """
    import discopt

    # Load from AMPL .nl format (parsed by Rust for speed)
    model = dm.from_nl("benchmarks/minlplib/ex1221.nl")
    result = model.solve()

    print(f"Optimal: {result.objective}")
    print(result.explain())
    """
    print("NL import example (requires Rust parser)")


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 12: Natural Language Formulation (LLM)
#
# The LLM-native interface: describe your problem, get a model.
# ═══════════════════════════════════════════════════════════════


def example_llm_formulation():
    """
    import discopt
    import pandas as pd

    # Your data
    costs = pd.DataFrame({
        'warehouse': ['NYC', 'LA', 'Chicago'],
        'capacity': [500, 700, 400],
        'fixed_cost': [10000, 12000, 8000],
    })
    customers = pd.DataFrame({
        'city': ['Boston', 'Miami', 'Denver', 'Seattle', 'Dallas'],
        'demand': [120, 80, 90, 110, 70],
    })
    shipping_rates = np.random.uniform(5, 50, (3, 5))

    # Describe the problem in natural language
    model = dm.from_description(
        '''
        Minimize total cost (fixed + shipping) of serving customers from warehouses.
        Each warehouse has a fixed cost to open and a maximum capacity.
        Each customer's demand must be fully met.
        Decide which warehouses to open and how much to ship from each
        warehouse to each customer.
        Shipping cost is proportional to the rate times quantity shipped.
        ''',
        data={
            'warehouses': costs,
            'customers': customers,
            'shipping_rates': shipping_rates,
        },
    )

    # The LLM generates a validated model. Review it:
    print(model)
    print(model.summary())

    # Solve with LLM explanation
    result = model.solve(llm=True)
    print(result.explain())
    # "Opened warehouses NYC and Chicago (total fixed cost $18,000).
    #  Boston and Miami served from NYC; Denver, Seattle, Dallas from Chicago.
    #  Total cost: $24,350. Opening LA would reduce shipping by $3,200
    #  but add $12,000 in fixed costs — not economical."
    """
    print("LLM formulation example (requires LLM integration)")


# ═══════════════════════════════════════════════════════════════
# EXAMPLE 13: Streaming Solve with Live Progress
# ═══════════════════════════════════════════════════════════════


def example_streaming():
    """
    import discopt

    m = dm.Model("large_problem")
    # ... build model ...

    # Streaming solve: get updates during branch-and-bound
    for update in m.solve(stream=True, llm=True):
        print(f"  {update.elapsed:6.1f}s | "
              f"Best: {update.incumbent:10.2f} | "
              f"Bound: {update.lower_bound:10.2f} | "
              f"Gap: {update.gap:6.2%} | "
              f"Nodes: {update.node_count}")

        if update.message:
            print(f"    LLM: {update.message}")

        # User-defined stopping
        if update.gap is not None and update.gap < 0.01:
            print("  1% gap reached — stopping early")
            break

    # Output:
    #    0.5s | Best:    142350 | Bound:     98200 | Gap: 31.02% | Nodes: 128
    #    2.1s | Best:    142350 | Bound:    121400 | Gap: 14.72% | Nodes: 1024
    #      LLM: OBBT tightened crude flow bounds by 35%. Root gap improving.
    #    8.4s | Best:    138900 | Bound:    132100 | Gap:  4.90% | Nodes: 8192
    #      LLM: Found better feasible solution by diving into left subtree.
    #   14.2s | Best:    138900 | Bound:    137800 | Gap:  0.79% | Nodes: 23401
    #   1% gap reached — stopping early
    """
    print("Streaming solve example (requires solver backend)")


# ═══════════════════════════════════════════════════════════════
# EXAMPLE: Transportation (named sets, sparse-friendly indexing)
#
#   minimize    Σ_{(p,k)} cost[p,k] · ship[p,k]
#   subject to  Σ_k ship[p,k] ≤ supply[p]     ∀ p ∈ PLANTS
#               Σ_p ship[p,k] ≥ demand[k]     ∀ k ∈ MARKETS
#               ship[p,k] ≥ 0
# ═══════════════════════════════════════════════════════════════


def example_transportation():
    """Balanced transportation problem indexed by named sets."""
    m = dm.Model("transportation")

    plants = m.set("plants", ["pitt", "sf"])
    markets = m.set("markets", ["a", "b", "c"])

    supply = {"pitt": 20.0, "sf": 30.0}
    demand = {"a": 10.0, "b": 25.0, "c": 15.0}
    cost = {
        ("pitt", "a"): 4.0,
        ("pitt", "b"): 6.0,
        ("pitt", "c"): 8.0,
        ("sf", "a"): 5.0,
        ("sf", "b"): 3.0,
        ("sf", "c"): 7.0,
    }

    ship = m.continuous("ship", over=plants * markets, lb=0, ub=1000)
    m.minimize(dm.sum(cost[(p, k)] * ship[p, k] for p in plants for k in markets))
    m.constraint(plants, lambda p: dm.sum(ship[p, k] for k in markets) <= supply[p], name="supply")
    m.constraint(markets, lambda k: dm.sum(ship[p, k] for p in plants) >= demand[k], name="demand")

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE: Assignment (indexed binaries over a product set)
# ═══════════════════════════════════════════════════════════════


def example_assignment():
    """Assign workers to tasks one-to-one at minimum cost."""
    m = dm.Model("assignment")

    workers = m.set("workers", ["w1", "w2", "w3"])
    tasks = m.set("tasks", ["a", "b", "c"])
    cost = {
        ("w1", "a"): 9,
        ("w1", "b"): 2,
        ("w1", "c"): 7,
        ("w2", "a"): 6,
        ("w2", "b"): 4,
        ("w2", "c"): 3,
        ("w3", "a"): 5,
        ("w3", "b"): 8,
        ("w3", "c"): 1,
    }

    assign = m.binary("assign", over=workers * tasks)
    m.minimize(dm.sum(cost[(w, t)] * assign[w, t] for w in workers for t in tasks))
    m.constraint(workers, lambda w: dm.sum(assign[w, t] for t in tasks) == 1, name="one_task")
    m.constraint(tasks, lambda t: dm.sum(assign[w, t] for w in workers) == 1, name="one_worker")

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE: Multi-commodity flow (set algebra: product + filter)
# ═══════════════════════════════════════════════════════════════


def example_multicommodity_flow():
    """Min-cost multi-commodity flow on a small sparse network.

    Demonstrates set algebra: arcs are a *sparse* subset of node×node, and the
    variable is indexed over arcs × commodities (a product set).
    """
    m = dm.Model("mcf")

    nodes = m.set("nodes", ["s", "u", "v", "t"])
    # Arcs are a *sparse* subset of the dense node-pair set ``nodes * nodes``.
    arcs = (nodes * nodes).where(
        lambda i, j: (i, j) in {("s", "u"), ("s", "v"), ("u", "t"), ("v", "t"), ("u", "v")}
    )
    commodities = m.set("commodities", ["c1", "c2"])

    cap = {a: 10.0 for a in arcs}
    flow = m.continuous("flow", over=arcs * commodities, lb=0, ub=1000)

    # Arc capacity shared across commodities.
    m.constraint(
        arcs,
        lambda i, j: dm.sum(flow[(i, j), c] for c in commodities) <= cap[(i, j)],
        name="capacity",
    )
    m.minimize(dm.sum(flow[a, c] for a in arcs for c in commodities))

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# EXAMPLE: Bilevel toll setting (Stackelberg)
#
# A road authority (leader) sets a toll `t` on a tolled arc to
# maximize revenue `t * f`. A driver population (follower) then
# chooses the flow `f` on that arc to minimize its own convex
# congestion-plus-toll cost. The follower is a convex QP in `f`,
# so it is replaced by its KKT conditions (an MPEC) via
# `discopt.bilevel.BilevelProblem`. Unconstrained follower optimum
# is f = 5 - t, giving revenue t(5 - t), maximized at t = 2.5.
# ═══════════════════════════════════════════════════════════════


def example_bilevel_toll():
    from discopt.bilevel import BilevelProblem

    m = dm.Model("bilevel_toll")
    t = m.continuous("toll", lb=0, ub=5)  # leader: toll on the tolled arc
    f = m.continuous("flow", lb=0, ub=8)  # follower: flow on the tolled arc

    # Leader maximizes toll revenue t * f  ->  minimize -(t * f).
    m.minimize(-(t * f))

    # Follower minimizes congestion cost 0.5 f^2 - 5 f (desired free-flow 5)
    # plus the toll it pays, t * f. Convex QP in f (Hessian = 1).
    bl = BilevelProblem(
        m,
        upper_vars=[t],
        lower_vars=[f],
        lower_objective=0.5 * f * f - 5 * f + t * f,
        lower_constraints=[],  # only the follower's flow bounds (folded in as KKT)
    )
    bl.formulate(method="kkt", mpec_method="gdp")

    print(m)
    return m


# ═══════════════════════════════════════════════════════════════
# Run all examples that don't require the solver backend
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("discopt API Examples")
    print("=" * 60)

    examples = [
        ("Simple MINLP", example_simple_minlp),
        ("Pooling (Haverly)", example_pooling_haverly),
        ("Process Synthesis", example_process_synthesis),
        ("Portfolio Optimization", example_portfolio),
        ("Reactor Design", example_reactor_design),
        ("Facility Location", example_facility_location),
        ("Parametric / Sensitivity", example_parametric),
        ("Logical Constraints / GDP", example_logical_constraints),
        ("NN Surrogate Optimization", example_nn_surrogate),
        ("Transportation (named sets)", example_transportation),
        ("Assignment (indexed binaries)", example_assignment),
        ("Multi-commodity Flow (set algebra)", example_multicommodity_flow),
    ]

    for name, func in examples:
        print(f"\n{'─' * 60}")
        print(f"Example: {name}")
        print(f"{'─' * 60}")
        try:
            model = func()
            if model:
                print("\n  ✓ Model built successfully")
        except Exception as e:
            print(f"\n  ✗ Error: {e}")

    print(f"\n{'─' * 60}")
    print("Import examples (stubs):")
    print(f"{'─' * 60}")
    example_pyomo_import()
    example_nl_import()
    example_llm_formulation()
    example_streaming()

    print(f"\n{'=' * 60}")
    print("All model-building examples completed.")
    print("Solve requires the Rust+JAX backend (under development).")
    print("=" * 60)
