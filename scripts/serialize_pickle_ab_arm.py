"""One arm of the WP-7 pickle A/B: time pickle / deepcopy with one tree's discopt.

Plan: ``docs/dev/serialization-completion-plan.md`` WP-7 (entry experiment). Run
once per arm per repeat, the arms interleaved and their order rotated, each in a
fresh interpreter (the baseline needs its own built Rust extension)::

    python -u scripts/serialize_pickle_ab_arm.py <tree> <marker> python/tests/data/minlplib_nl

``marker`` names the version under test (CLAUDE.md §8): ``absent`` (no
``_PICKLE_SCOPES``: before #1764), ``present`` (#1764, state-borne children-first
list) or ``variant`` (WP-7: ``_new_after_children``). The imported ``core`` must
come from ``<tree>``. Times (a) the 66-file corpus pickled and unpickled, (b) the
same deep-copied, (c) three synthetic loop-built sums (5k/50k/200k nodes) pickled,
unpickled and deep-copied, with ``gc.collect()`` before each timed op -- without it
a full collection lands on whichever op crosses the threshold and moves between
arms (measured: a ~170 ms cell that changed places). Prints one JSON line; exits
non-zero if nothing was timed (§6).

Results: ``docs/dev/data/serialize-pickle-ab-2026-10-19.jsonl``.
"""

import copy
import gc
import glob
import json
import os
import pickle
import sys
import time
import warnings


def _deep(n: int):
    import discopt.modeling as dm

    m = dm.Model(f"deep{n}")
    x = m.continuous("x", lb=-1, ub=1, shape=(10,))
    e = x[0]
    for i in range(n // 3):
        e = e + 0.5 * x[i % 10]
    m.subject_to(e <= 10.0)
    m.minimize(x[0])
    return m


def _timed(fn) -> float:
    gc.collect()
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main() -> int:
    warnings.simplefilter("ignore")
    tree, marker, corpus = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, os.path.join(tree, "python"))
    import discopt.modeling.core as core

    assert os.path.realpath(core.__file__).startswith(os.path.realpath(tree)), core.__file__
    src = open(core.__file__).read()
    assert ("_PICKLE_SCOPES" in src) == (marker != "absent"), (marker, core.__file__)
    assert ("_new_after_children" in src) == (marker == "variant"), (marker, core.__file__)

    sys.setrecursionlimit(100000)
    models = [core.from_nl(p) for p in sorted(glob.glob(os.path.join(corpus, "*.nl")))]
    assert len(models) == 66, len(models)
    synth = {n: _deep(n) for n in (5000, 50000, 200000)}

    out: dict = {"marker": marker, "file": core.__file__}
    out["corpus_pickle_rt"] = _timed(lambda: [pickle.loads(pickle.dumps(m)) for m in models])
    out["corpus_deepcopy"] = _timed(lambda: [copy.deepcopy(m) for m in models])
    timed = 2 * len(models)
    for n, m in synth.items():
        held: list = []
        out[f"deep{n}_dumps"] = _timed(lambda m=m: held.append(pickle.dumps(m)))
        out[f"deep{n}_loads"] = _timed(lambda: pickle.loads(held[0]))
        out[f"deep{n}_deepcopy"] = _timed(lambda m=m: copy.deepcopy(m))
        timed += 3
    out["timed"] = timed
    print(json.dumps(out), flush=True)
    return 0 if timed else 3


if __name__ == "__main__":
    sys.exit(main())
