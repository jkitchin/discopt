"""highs-parity P1.1: run solve_lp_std on the L/INF panels' standard forms (exact-identity check).

    python discopt_benchmarks/scripts/lp_engine_seam_check.py build SFS.pkl
    python discopt_benchmarks/scripts/lp_engine_seam_check.py run SFS.pkl OUT.json [highs|rust]

Run "run" once on the pre-seam tree (no engine argument) and once per engine on the seam;
the JSON outputs compare field by field (status, objective repr, bound, certificate,
labels, message, engine status, stats without wall times).
"""

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "discopt_benchmarks/scripts")
mode = sys.argv[1]
if mode == "build":
    import lp_milp_loader as L  # noqa: N812
    from discopt.modeling.core import from_nl
    from discopt.solver import _highs_std_form

    man = json.loads(Path("discopt_benchmarks/data/lp_milp/manifest.json").read_text())
    out = []
    for m in man["members"]:
        if m["panel"] not in ("L", "INF") or m["n_int"]:
            continue
        inst = L.read_instance(m["path"])
        with tempfile.TemporaryDirectory() as td:
            nl = os.path.join(td, "m.nl")
            L.write_nl(inst, nl)
            model = from_nl(nl)
        _, _, sf = _highs_std_form(model)
        out.append((m["panel"], m["name"], m.get("oracle_status"), m.get("oracle_objective"), sf))
    Path(sys.argv[2]).write_bytes(pickle.dumps(out))
    print("built", len(out))
else:
    import discopt.solvers.lp_milp_highs as H  # noqa: N812

    print("FILE", H.__file__, "engine-seam" if hasattr(H, "_lp_engine") else "pre-seam", flush=True)
    sfs = pickle.loads(Path(sys.argv[2]).read_bytes())
    kw = {} if len(sys.argv) < 5 else {"engine": sys.argv[4]}
    res = {}
    for panel, name, ost, oobj, sf in sfs:
        o = H.solve_lp_std(sf, time_limit=60, **kw)
        stats = {k: v for k, v in sorted(o.stats.items()) if "time" not in k}
        res[name] = {
            "panel": panel,
            "oracle_status": ost,
            "oracle": oobj,
            "status": o.status,
            "objective": repr(o.objective),
            "bound": repr(o.bound),
            "cert": o.gap_certified,
            "labels": dict(sorted(o.labels.items())),
            "message": o.message,
            "highs_status": o.highs_status,
            "stats": stats,
        }
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
    print("ran", len(res))
