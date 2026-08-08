"""Independent re-check of ``panel_corpus.json`` against a *second* oracle (#945).

``panel_corpus.py`` judges soundness against the in-repo registry
(``python/tests/data/known_optima.toml``), which covers 32 of the 66 corpus
instances and is the table this repo's certification tests already trust. That
makes it the right gate, and a poor *witness*: a fix whose whole point is that
incumbents stop being better-than-possible should be shown moving the number,
not merely failing to trip a threshold.

So this reads the panel's own output back and scores both arms against
MINLPLib's ``minlplib.solu`` (``=opt=`` rows only — ``=best=`` is not an oracle)
at a threshold four orders tighter than the gate's, and prints the signed
super-optimality per arm. A positive number is an incumbent *better than the
published global optimum*, which is impossible; the interesting quantity is how
many instances carry one in each arm, and whether any instance moves the wrong
way.

Not a gate — ``panel_corpus.py`` is the gate. This is the witness for the PR.

§6: prints ``COMPARED=`` and exits non-zero if it compared nothing, so a moved
corpus or a renamed key cannot report "0 regressions" from an empty loop.
§7: no exception handling — a missing panel JSON or solu file must crash.

Usage:
    python -u scratchpad/issue945/panel_superopt_recheck.py \
        scratchpad/issue945/panel_corpus.json [path/to/minlplib.solu]
"""

from __future__ import annotations

import json
import os
import sys

_DEFAULT_SOLU = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")

# Tight on purpose: the gate uses 1e-6 relative, which every violation this fix
# removes sits *underneath*. The defect is real at 1e-8, so look at 1e-12.
_THRESHOLD = 1e-12


def load_optima(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            # "=best=" is an incumbent someone found, not a proven optimum, so it
            # cannot witness super-optimality. Only "=opt=" rows are an oracle.
            if len(parts) == 3 and parts[0] == "=opt=":
                out[parts[1]] = float(parts[2])
    return out


def main() -> int:
    panel_path = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/issue945/panel_corpus.json"
    solu_path = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_SOLU
    rows = json.load(open(panel_path))["rows"]
    optima = load_optima(solu_path)

    compared = 0
    per_arm = {"pre": 0, "post": 0}
    detail: list[tuple[str, float, float]] = []
    wrong_way: list[str] = []

    for name in sorted(rows):
        pre, post = rows[name]["pre"], rows[name]["post"]
        if name not in optima:
            continue
        if pre["objective"] is None or post["objective"] is None:
            continue
        opt = optima[name]
        maximize = bool(pre["maximize"])
        compared += 1

        def excess(obj: float, _opt: float = opt, _mx: bool = maximize) -> float:
            """Signed, scaled: > 0 means BETTER than the optimum, i.e. impossible."""
            raw = (obj - _opt) if _mx else (_opt - obj)
            return raw / max(1.0, abs(_opt))

        a, b = excess(pre["objective"]), excess(post["objective"])
        if a > _THRESHOLD:
            per_arm["pre"] += 1
        if b > _THRESHOLD:
            per_arm["post"] += 1
        if a > _THRESHOLD or b > _THRESHOLD:
            detail.append((name, a, b))
        # An instance that was sound in pre and is super-optimal in post is the
        # only outcome that would sink this change.
        if a <= _THRESHOLD < b:
            wrong_way.append(name)

    for name, a, b in detail:
        tag = "FIXED" if a > _THRESHOLD >= b else ""
        print(f"  {name:18s} pre={a:+.3e}  post={b:+.3e}  {tag}")
    print(f"COMPARED={compared}")
    print(f"SUPER_OPTIMAL_INSTANCES pre={per_arm['pre']}  post={per_arm['post']}")
    print(f"WRONG_WAY={len(wrong_way)} {wrong_way}")

    if compared == 0:
        print("FAIL: compared nothing — the panel JSON and the solu share no instance names")
        return 1
    if wrong_way:
        print("FAIL: an instance became super-optimal in the post arm")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
