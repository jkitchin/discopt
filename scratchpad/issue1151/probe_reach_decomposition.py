"""#1151 reach probe: does the failing decomposition test touch the changed code?

Runs the failing case's own solve calls directly, counting every entry into the
three sites this PR changed. Zero calls means the diff provably cannot move that
bound. Prints an executed-call count and exits non-zero if the probe itself
never ran a solve (§6).
"""
import sys

from discopt.validation import feasibility as F

STATS = {"verify_point": 0, "jacobian_row_scales": 0, "_row_scales": 0}
_vp, _jrs, _rs = F.verify_point, F.jacobian_row_scales, F._row_scales


def _wrap(name, fn):
    def inner(*a, **k):
        STATS[name] += 1
        return fn(*a, **k)

    return inner


F.verify_point = _wrap("verify_point", _vp)
F.jacobian_row_scales = _wrap("jacobian_row_scales", _jrs)
F._row_scales = _wrap("_row_scales", _rs)

import discopt._dual_recovery as _dr  # noqa: E402
import discopt.validation.examiner as _ex  # noqa: E402

for _m in (_ex, _dr):
    if hasattr(_m, "jacobian_row_scales"):
        setattr(_m, "jacobian_row_scales", F.jacobian_row_scales)

sys.path.insert(0, "python/tests")
from test_decomposition_adversarial import _rand_gap  # noqa: E402

from discopt.decomposition.lagrangian import solve_lagrangian  # noqa: E402

solves = 0
seq = solve_lagrangian(_rand_gap(4), method="kelley", backend="sequential", time_limit=15)
solves += 1
thr = solve_lagrangian(_rand_gap(4), method="kelley", backend="threads", time_limit=15)
solves += 1
print(f"seq: status={seq.status} bound={seq.bound!r} objective={seq.objective!r}")
print(f"thr: status={thr.status} bound={thr.bound!r} objective={thr.objective!r}")
print(f"bit-equal bounds: {seq.bound == thr.bound}")
print(f"[#1151 reach] calls into the changed sites: {STATS}  total={sum(STATS.values())}")
print(f"executed solves: {solves}")
if solves == 0:
    sys.exit("PROBE RAN NO SOLVE")
