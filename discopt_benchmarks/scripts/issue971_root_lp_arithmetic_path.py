#!/usr/bin/env python
"""Issue #971 entry experiment: is the root LP bound a function of the matrix alone?

``diff_root_lp`` excuses a root-LP difference only when the relaxation
*fingerprint* differs too, on the reasoning that "a difference is only
attributable when both sides solved the same LP". That rule assumes
**same matrix bytes ⇒ same LP answer**. #971 reports CI failures on ``contvar``
bucketed ``changed`` — i.e. the fingerprint MATCHED and the bound still moved by
8.8% — which falsifies the assumption if reproducible.

This probe holds the tree, the instance and the matrix fixed and varies only the
*arithmetic path*, via OpenBLAS's runtime kernel dispatch (``OPENBLAS_CORETYPE``;
the wheels ship ``DYNAMIC_ARCH``, so the same binary runs a Nehalem / Haswell /
SkylakeX / Zen kernel on demand — the same knob a heterogeneous CI runner pool
turns implicitly). For each core type it reports, from a **fresh process**, the
relaxation fingerprint and the root LP ``(status, bound)``.

Kill criterion: if every core type yields the same ``(status, bound)`` with the
same fingerprint, the arithmetic path is NOT the mechanism and #971's premise
needs another explanation. If any two agree on the fingerprint and disagree on
the bound, ``diff_root_lp``'s attributability rule is provably incomplete.

Discipline: asserts the tree under test by path (CLAUDE.md §8), never swallows an
exception (§7), prints a completed-rep count and exits non-zero when it is zero
(§6), and streams per-rep progress (§10).

Usage (from the repo root, with the built extension importable)::

    python -u discopt_benchmarks/scripts/issue971_root_lp_arithmetic_path.py \
        [--instance contvar] [--coretypes Haswell,Nehalem,SkylakeX,Zen]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"

# Kernel families an x86-64 runner pool actually spans: pre-FMA (Nehalem),
# FMA/AVX2 (Haswell), AVX-512 (SkylakeX), and AMD (Zen). OpenBLAS falls back to
# a generic kernel for a core type the CPU cannot run, which is itself a
# distinct arithmetic path, so an unsupported name is still a useful arm.
_DEFAULT_CORETYPES = ("Haswell", "Nehalem", "SkylakeX", "Zen", "Prescott")

_CHILD = r"""
import json, os, sys
sys.path.insert(0, {python_dir!r})
import discopt
assert discopt.__file__ == {expect_init!r}, (
    "wrong tree under test: %s != %s" % (discopt.__file__, {expect_init!r})
)
sys.path.insert(0, {tests_dir!r})
from support.claim_differential import current_root_lp, current_row

row = current_row({name!r})
status, bound = current_root_lp({name!r})
print("RESULT " + json.dumps(
    {{"coretype": os.environ.get("OPENBLAS_CORETYPE", ""),
      "fingerprint": row["fingerprint"],
      "n_rows": row["n_rows"], "n_cols": row["n_cols"],
      "status": status, "bound": bound}}))
"""


def _run_one(name: str, coretype: str) -> dict:
    """Build + solve ``name`` in a fresh process under ``coretype``.

    No exception handling (CLAUDE.md §7): a crashing arm must surface as a
    crash, not as a missing row that reads like agreement.
    """
    env = dict(os.environ)
    env["OPENBLAS_CORETYPE"] = coretype
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    src = _CHILD.format(
        python_dir=str(_REPO / "python"),
        tests_dir=str(_REPO / "python" / "tests"),
        expect_init=str(_REPO / "python" / "discopt" / "__init__.py"),
        name=name,
    )
    proc = subprocess.run(
        [sys.executable, "-u", "-c", src], cwd=str(_REPO), env=env, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{coretype} arm failed (rc={proc.returncode}):\n{proc.stderr[-4000:]}")
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(f"{coretype} arm printed no RESULT:\n{proc.stdout[-2000:]}")
    return json.loads(line[len("RESULT ") :])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="contvar")
    ap.add_argument("--coretypes", default=",".join(_DEFAULT_CORETYPES))
    args = ap.parse_args()

    if not (_NL_DIR / f"{args.instance}.nl").exists():
        raise SystemExit(f"no such instance: {args.instance}")

    rows: list[dict] = []
    for ct in [c for c in args.coretypes.split(",") if c]:
        row = _run_one(args.instance, ct)
        rows.append(row)
        print(
            f"  {ct:<10} fp={row['fingerprint'][:12]} status={row['status']} bound={row['bound']}",
            flush=True,
        )

    print(f"\nREPS_COMPLETED={len(rows)}")
    if not rows:
        print("PROBE MEASURED NOTHING", file=sys.stderr)
        return 2

    fps = {r["fingerprint"] for r in rows}
    answers = {(r["status"], r["bound"]) for r in rows}
    print(f"distinct_fingerprints={len(fps)}  distinct_answers={len(answers)}")

    # The falsifying observation: same matrix bytes, different answer.
    same_bytes_diff_answer = False
    for fp in fps:
        arms = [r for r in rows if r["fingerprint"] == fp]
        if len({(r["status"], r["bound"]) for r in arms}) > 1:
            same_bytes_diff_answer = True
            print(f"SAME_BYTES_DIFFERENT_ANSWER fingerprint={fp[:12]}:")
            for r in arms:
                print(f"    {r['coretype']:<10} -> ({r['status']}, {r['bound']})")
    print(f"same_bytes_different_answer={same_bytes_diff_answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
