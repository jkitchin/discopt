"""#1975 (retired): solve one .nl through the outer-piece prototype, as the flag did.

    python -u outer_piece_solve_one.py <path.nl> <time_limit>
"""
import importlib.util, os, sys, time
import discopt.modeling as dm

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("outer_piece_prototype", f"{here}/outer_piece_prototype.py")
op = importlib.util.module_from_spec(spec); sys.modules[spec.name] = op; spec.loader.exec_module(op)
m = dm.from_nl(sys.argv[1]); tl = float(sys.argv[2]); t = time.time()
try:
    p = op.plan(m)
except op.OuterPieceUnsupported as e:
    sys.exit(f"UNSUPPORTED: {e}")
if p is None:
    sys.exit("no open column")
r = op.solve_with_outer_piece(m, p, dict(time_limit=tl, gap_tolerance=1e-4))
if r is None:
    sys.exit(f"declined: {p.stats}")
print(f"status={r.status} obj={r.objective} bound={r.bound} cert={r.gap_certified} "
      f"nodes={r.node_count} wall={time.time()-t:.1f}")
print({k: v for k, v in r.solver_stats.items() if k.startswith("outer")})
