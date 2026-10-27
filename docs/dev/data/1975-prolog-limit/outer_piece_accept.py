"""#1975: does the outer-piece rewrite accept each instance? (structural, no solve)"""
import json, sys, signal, time
import importlib.util, os
import discopt.modeling as dm
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("outer_piece_prototype", f"{_here}/outer_piece_prototype.py")
op = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = op; _spec.loader.exec_module(op)
D, names, out = sys.argv[1], open(sys.argv[2]).read().split(), sys.argv[3]
def alarm(*a): raise TimeoutError
signal.signal(signal.SIGALRM, alarm)
n_checked = 0
with open(out, "w") as fh:
    for name in names:
        row = {"name": name}; t = time.time()
        signal.alarm(60)
        try:
            m = dm.from_nl(f"{D}/{name}.nl")
            p = op.plan(m)
            if p is None:
                row["verdict"] = "no open column"
            else:
                o, d = op.build_outer(p, 1e4)
                row.update(verdict="accepted", n_open=len(p.cols),
                           n_gen=sum(c.in_T for c in p.cols), degree=d,
                           max_row_degree=p.stats.get("outer_piece/max_row_degree"))
        except op.OuterPieceUnsupported as e:
            row["verdict"] = "unsupported"; row["reason"] = str(e)[:120]
        except Exception as e:
            row["verdict"] = "error"; row["reason"] = f"{type(e).__name__}: {e}"[:200]
        finally:
            signal.alarm(0)
        row["t"] = round(time.time() - t, 2); n_checked += 1
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print(row, flush=True)
print("checked", n_checked)
assert n_checked == len(names)
