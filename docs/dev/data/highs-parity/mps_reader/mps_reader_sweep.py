"""Exact differential: discopt's MPS reader vs HiGHS's, over every MPS file given.

Usage: python mps_reader_sweep.py FILE_OR_DIR ... > rows.jsonl
Prints one JSON row per file (agree / differences / which side failed) and a final
executed count; exits non-zero when nothing was compared or anything disagreed.
"""

import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../python/tests"))
import _mps_differential as D  # noqa: E402

from discopt._mps import read_mps  # noqa: E402


def files(args):
    for a in args:
        if os.path.isdir(a):
            for pat in ("*.mps", "*.mps.gz", "*.MPS"):
                yield from sorted(glob.glob(os.path.join(a, "**", pat), recursive=True))
        else:
            yield a


def main():
    n_cmp = n_bad = 0
    for f in files(sys.argv[1:]):
        row = {"file": f}
        t0 = time.perf_counter()
        try:
            ref = D.highs_arrays(f)
            row["highs_s"] = round(time.perf_counter() - t0, 3)
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            ref = None
            row["highs_error"] = repr(e)
        t0 = time.perf_counter()
        try:
            ours = read_mps(f)
            row["ours_s"] = round(time.perf_counter() - t0, 3)
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            ours = None
            row["ours_error"] = repr(e)
        if ref is not None and ours is not None:
            diffs = D.differences(ours, ref)
            row["agree"] = not diffs
            row["differences"] = diffs
            n_cmp += 1
            n_bad += bool(diffs)
        else:
            row["agree"] = (ref is None) and (ours is None)
            n_bad += not row["agree"]
        print(json.dumps(row), flush=True)
    print(f"# compared {n_cmp}, disagreements {n_bad}", file=sys.stderr, flush=True)
    return 1 if (n_cmp == 0 or n_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
