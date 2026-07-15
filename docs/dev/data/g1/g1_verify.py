"""G1 byte-identity gate: solve one instance in a fresh subprocess and print a
JSON row of {node_count, objective, bound, status}.

  --baseline  monkeypatch the two G1 fusion gates OFF (exact pre-change path),
              so a differential run proves byte-identity fused-vs-unfused.

Usage:
    PYTHONPATH=<wt>/python python g1_verify.py <path.nl> [--baseline] [--tl N]
"""

from __future__ import annotations

import json
import sys


def _disable_fusion():
    import discopt._jax.nlp_evaluator as nev

    orig_init = nev.NLPEvaluator.__init__

    def patched(self, *a, **k):
        orig_init(self, *a, **k)
        # Force every evaluate_* method onto its original per-quantity branch.
        self._fused_fc_jit = None
        self._gj_fusable_cache = False

    nev.NLPEvaluator.__init__ = patched


def main() -> None:
    args = sys.argv[1:]
    baseline = "--baseline" in args
    args = [a for a in args if a != "--baseline"]
    tl = 300.0
    if "--tl" in args:
        i = args.index("--tl")
        tl = float(args[i + 1])
        del args[i : i + 2]
    path = args[0]

    if baseline:
        _disable_fusion()

    from discopt.modeling.core import from_nl

    model = from_nl(path)
    res = model.solve(time_limit=tl, threads=1)

    def g(name):
        for n in (name, f"_{name}"):
            if hasattr(res, n):
                v = getattr(res, n)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return str(v)
        return None

    row = {
        "instance": path.split("/")[-1].replace(".nl", ""),
        "baseline": baseline,
        "node_count": g("node_count"),
        "objective": g("objective"),
        "bound": g("bound"),
        "status": g("status"),
    }
    print(json.dumps(row))


if __name__ == "__main__":
    main()
