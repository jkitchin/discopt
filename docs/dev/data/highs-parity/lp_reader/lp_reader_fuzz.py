"""Mutation fuzz: discopt's LP-format reader vs HiGHS's on randomly mutated LP texts.

Usage: python lp_reader_fuzz.py SEED N [EXTRA_SEED_DIR ...]
Seeds are the synthetic cases of ``python/tests/_lp_probe_cases.py`` plus every LP file
under 4 kB in the given directories (e.g. HiGHS's ``check/instances``); each iteration
applies 1-4 random insertions/deletions of LP lexemes and compares both readers
exactly (kError included). A file HiGHS reads but highspy cannot expose (a non-UTF-8
name raises ``UnicodeDecodeError`` in ``getLp``) gets no verdict and is counted apart.
Prints the executed counts; exits non-zero when nothing was compared or anything
disagreed.
"""

import glob
import os
import random
import sys
import tempfile

TESTS = os.path.join(os.path.dirname(__file__), "../../../../../python/tests")
sys.path.insert(0, TESTS)
import _mps_differential as D  # noqa: E402
from _lp_probe_cases import PROBE_CASES  # noqa: E402
from discopt._lpfile import read_lp  # noqa: E402

PIECES = [
    b" ", b"\n", b"+", b"-", b"[", b"]", b"^", b"*", b"/", b":", b"::", b"<", b">", b"=",
    b"<=", b">=", b"2", b"0", b"1e30", b"inf", b"nan", b"x", b"y", b"st", b"end", b"bin",
    b"gen", b"semi", b"free", b"bounds", b"min", b"max", b"\\", b";", b"/*", b"*/", b"\t",
    b"\r", b"\x0b", b"S1", b"sos", b"0x1p3", b"-1e-400", b"1e-10", b"x^2", b"[ x^2 ]/2",
    b"[ x*y ]/2", b"subject to", b"HiGHS_R1:",
]  # fmt: skip


def main():
    seed, n_iter = int(sys.argv[1]), int(sys.argv[2])
    seeds = [t.encode("utf-8") for _, t in PROBE_CASES]
    for d in sys.argv[3:]:
        for f in sorted(glob.glob(os.path.join(d, "*.lp"))):
            if os.path.getsize(f) < 4000:
                with open(f, "rb") as fh:
                    seeds.append(fh.read())
    rng = random.Random(seed)
    n_cmp = n_bad = n_read = n_undecodable = 0
    with tempfile.TemporaryDirectory(prefix="lpfuzz_") as tmp:
        path = os.path.join(tmp, "fz.lp")
        for _ in range(n_iter):
            b = bytearray(rng.choice(seeds))
            for _ in range(rng.randint(1, 4)):
                op, pos = rng.random(), rng.randint(0, len(b))
                if op < 0.5:
                    b[pos:pos] = rng.choice(PIECES)
                elif op < 0.8 and len(b) > 1:
                    del b[pos : pos + rng.randint(1, 3)]
                else:
                    b[pos:pos] = b" " + rng.choice(PIECES) + b" "
            with open(path, "wb") as fh:
                fh.write(bytes(b))
            try:
                ref = D.highs_arrays(path)
            except UnicodeDecodeError:
                n_undecodable += 1
                continue
            except ValueError:  # HiGHS kError
                ref = None
            try:
                ours = read_lp(path)
            except ValueError:  # our kError
                ours = None
            n_cmp += 1
            if ref is not None and ours is not None:
                n_read += 1
                diffs = D.differences(ours, ref)
            elif (ref is None) != (ours is None):
                diffs = [f"highs_error={ref is None} ours_error={ours is None}"]
            else:
                diffs = []
            if diffs:
                n_bad += 1
                print("DISAGREE", diffs, repr(bytes(b))[:300], flush=True)
    print(
        f"# fuzz seed {seed}: compared {n_cmp} (both read {n_read}), disagreements "
        f"{n_bad}, no verdict (non-UTF-8 name) {n_undecodable}",
        flush=True,
    )
    return 1 if (n_cmp == 0 or n_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
