import sys, time, inspect, numpy as np, discopt, discopt.modeling as dm
import discopt.solvers.lp_milp_highs as H
from discopt.ml import NNFormulation, NetworkDefinition, DenseLayer, Activation
z = np.load(sys.argv[1]); Ws=[z[f"W{i}"] for i in range(3)]; bs=[z[f"b{i}"] for i in range(3)]
L=[DenseLayer(Ws[i], bs[i], Activation.RELU) for i in range(2)]+[DenseLayer(Ws[2], bs[2], Activation.LINEAR)]
m=dm.Model("s"); nn=NNFormulation(m, NetworkDefinition(L, input_bounds=(z["lo"], z["hi"])), strategy="relu_extended"); nn.formulate(); m.minimize(dm.sum(nn.outputs))
t=time.perf_counter(); r=m.solve(time_limit=600, threads=1, deterministic=True); w=time.perf_counter()-t
print(f"{sys.argv[2]} marker={'_repair_tree_below_root' in inspect.getsource(H)} file={discopt.__file__[-40:]} {r.status} obj={r.objective} bound={r.bound} nodes={r.node_count} wall={w:.1f}", flush=True)
