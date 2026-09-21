#!/usr/bin/env python3
"""#1397: what does the denominator-sign slack cost in *tightening strength*?

The fix in ``_find_clearable_denominator`` refuses to clear a denominator whose
interval error :func:`bound_expression_error` cannot bound -- a transcendental or a
nested quotient returns ``inf``. That is the sound direction (clearing is optional;
clearing a sign-indefinite denominator flips the constraint), but CLAUDE.md §5 still
requires the change be measurably neutral-or-better, not merely sound. So: over the
in-repo corpus, how many denominators does the pass clear with the slack ON vs OFF?

Both arms run in ONE process, interleaved per instance, by swapping the slack helper
-- so there is no load-gate ambiguity and no between-run drift. Per §6 it prints an
executed-comparison count and exits non-zero when that is zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

from discopt._relax import factorable_reform as fr
from discopt.modeling.core import Constraint, from_nl

CORPUS = Path(__file__).resolve().parents[1] / "python" / "tests" / "data" / "minlplib_nl"


def _count_clears(model) -> int:
    n = 0
    for c in model._constraints:
        if isinstance(c, Constraint) and fr._find_clearable_denominator(c.body, model) is not None:
            n += 1
    if model._objective is not None:
        expr = getattr(model._objective, "expr", model._objective)
        if fr._find_clearable_denominator(expr, model) is not None:
            n += 1
    return n


def main() -> int:
    files = sorted(CORPUS.glob("*.nl"))
    print(f"corpus : {CORPUS}  ({len(files)} instances)")
    print(f"file   : {fr.__file__}")
    src = Path(fr.__file__).read_text()
    marker = "_denominator_sign_slack"
    print(f"marker : {'PRESENT' if marker in src else 'ABSENT'} ({marker})")
    if marker not in src:
        print("FAIL: this probe measures the fix; it is not in the loaded tree (§8)")
        return 2

    real_slack = fr._denominator_sign_slack
    compared = 0
    total_on = total_off = 0
    regressed: list[tuple[str, int, int]] = []
    errors = 0

    for path in files:
        try:
            model = from_nl(str(path))
        except Exception as exc:
            errors += 1
            print(f"  {path.stem:<28} LOAD ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        fr._denominator_sign_slack = lambda *_a, **_k: (0.0, 0.0)
        off = _count_clears(model)
        fr._denominator_sign_slack = real_slack
        on = _count_clears(model)
        compared += 1
        total_off += off
        total_on += on
        if on != off:
            regressed.append((path.stem, off, on))
            print(f"  {path.stem:<28} clears OFF={off} ON={on}  <-- DIFFERS", flush=True)

    print(f"\nEXECUTED COMPARISONS: {compared} of {len(files)} (load errors: {errors})")
    if compared == 0:
        print("FAIL: nothing was compared (§6)")
        return 2
    print(f"denominators cleared, slack OFF (today) : {total_off}")
    print(f"denominators cleared, slack ON  (fix)   : {total_on}")
    print(f"instances whose clear count changed     : {len(regressed)}")
    for name, off, on in regressed:
        print(f"    {name:<28} {off} -> {on}")
    if total_on == total_off:
        print("\nPASS: no strength lost on the corpus (the refusal is unreachable here)")
    else:
        print(f"\nNOTE: {total_off - total_on} clear(s) lost; each must be justified below")
    return 0


if __name__ == "__main__":
    sys.exit(main())
