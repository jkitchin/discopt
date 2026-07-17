import os
import sys
import warnings

sys.path.insert(0, "python")
from discopt.modeling.core import from_nl

NL = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
SOLU = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")


def oracle(name):
    with open(SOLU) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and p[1] == name and p[0] in ("=opt=", "=best="):
                return float(p[2])
    return None


insts = sys.argv[1].split(",")
tl = float(sys.argv[2])
_hdr_l = f"{'instance':24s} {'oracle':>9s}"
_hdr_off = f"{'OFF obj':>8s} {'OFF bnd':>9s}"
_hdr_on = f"{'ON obj':>8s} {'ON bnd':>9s}"
print(f"{_hdr_l} | {_hdr_off} | {_hdr_on} | verdict")
for name in insts:
    opt = oracle(name)
    row = []
    for flag in ("0", "1"):
        os.environ["DISCOPT_MILP_SWAP_RESEED"] = flag
        m = from_nl(f"{NL}/{name}.nl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.solve(time_limit=tl)
        row.append((r.objective, r.bound))
    (o0, b0), (o1, b1) = row
    # soundness: for these min problems bound <= opt <= obj
    snd = all(b is None or opt is None or b <= opt + 1e-4 for b in (b0, b1)) and all(
        o is None or opt is None or o >= opt - 1e-4 for o in (o0, o1)
    )
    # primal verdict
    if o0 is None or o1 is None:
        v = "?"
    elif o1 < o0 - 1e-6:
        v = "ON BETTER primal"
    elif o1 > o0 + 1e-6:
        v = "ON WORSE primal"
    else:
        v = "tie primal"

    def fo(x):
        return f"{x:.2f}" if x is not None else "None"

    mark = "" if snd else "  !!!UNSOUND"
    left = f"{name:24s} {fo(opt):>9s}"
    off = f"{fo(o0):>8s} {fo(b0):>9s}"
    on = f"{fo(o1):>8s} {fo(b1):>9s}"
    print(f"{left} | {off} | {on} | {v}{mark}")
