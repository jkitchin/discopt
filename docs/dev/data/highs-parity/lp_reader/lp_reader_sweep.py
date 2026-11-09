"""Exact differential: discopt's LP-format reader vs HiGHS's, over every LP file given
plus (``--probes``) the synthetic cases of ``python/tests/_lp_probe_cases.py``.

Usage: python lp_reader_sweep.py --out rows.jsonl [--probes] FILE_OR_DIR ...
Writes one JSON row per file or case (agree / differences / which side failed) to
``--out`` (not stdout: HiGHS's reader ``printf``s to it) and prints a final executed
count; exits non-zero when nothing was compared or anything disagreed.
"Agree" for a file HiGHS rejects means ours rejects it too.
"""

import glob
import json
import os
import sys
import tempfile
import time

TESTS = os.path.join(os.path.dirname(__file__), "../../../../../python/tests")
sys.path.insert(0, TESTS)
import _mps_differential as D  # noqa: E402
from _lp_probe_cases import PROBE_CASES  # noqa: E402
from discopt._lpfile import read_lp  # noqa: E402


def files(args):
    for a in args:
        if os.path.isdir(a):
            for pat in ("*.lp", "*.lp.gz", "*.LP"):
                yield from sorted(glob.glob(os.path.join(a, "**", pat), recursive=True))
        else:
            yield a


def compare(path, row):
    t0 = time.perf_counter()
    try:
        ref = D.highs_arrays(path)
        row["highs_s"] = round(time.perf_counter() - t0, 3)
    except UnicodeDecodeError:
        raise  # highspy cannot expose a non-UTF-8 name: not a kError, so no verdict
    except ValueError as e:  # HiGHS kError: recorded, not swallowed
        ref = None
        row["highs_error"] = str(e)
    t0 = time.perf_counter()
    try:
        ours = read_lp(path)
        row["ours_s"] = round(time.perf_counter() - t0, 3)
    except ValueError as e:  # our kError: recorded, not swallowed
        ours = None
        row["ours_error"] = str(e)
    if ref is not None and ours is not None:
        diffs = D.differences(ours, ref)
        row["agree"] = not diffs
        row["differences"] = diffs
        row["n"], row["m"] = ref["n"], ref["m"]
        row["hessian_nnz"] = None if ref["hessian"] is None else int(ref["hessian"].nnz)
    else:
        row["agree"] = (ref is None) and (ours is None)
    return row


def main():
    args = sys.argv[1:]
    probes = "--probes" in args
    args = [a for a in args if a != "--probes"]
    i_out = args.index("--out")
    out = open(args[i_out + 1], "w")  # noqa: SIM115 - closed below
    del args[i_out : i_out + 2]
    n_rows = n_bad = n_both_read = 0
    for f in files(args):
        row = compare(f, {"file": f})
        n_rows += 1
        n_both_read += "differences" in row
        n_bad += not row["agree"]
        print(json.dumps(row), file=out, flush=True)
    if probes:
        with tempfile.TemporaryDirectory(prefix="lpsweep_") as tmp:
            for k, (tag, text) in enumerate(PROBE_CASES):
                path = os.path.join(tmp, f"case_{k}.lp")
                with open(path, "wb") as fh:
                    fh.write(text.encode("utf-8"))
                row = compare(path, {"case": k, "tag": tag})
                n_rows += 1
                n_both_read += "differences" in row
                n_bad += not row["agree"]
                print(json.dumps(row), file=out, flush=True)
    out.close()
    print(
        f"# compared {n_rows} (both read {n_both_read}), disagreements {n_bad}",
        file=sys.stderr,
        flush=True,
    )
    return 1 if (n_rows == 0 or n_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
