# ruff: noqa -- frozen #1731 panel record, kept as run.
"""#1731 panel population: corpus instances on which ``_objective_lattice`` fires.

Prefilter (metadata only): has a discrete column, <= 2000 vars, an oracle.
Then load each model and run the detector. Prints one line per instance and an
executed-detector count; exits non-zero if the detector ran on nothing.
"""

import csv
import multiprocessing as mp
import os
import sys

D = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark")
MAXV = 2000


def solu(path):
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 3 and p[0] == "=opt=":
            out[p[1]] = float(p[2])
        elif len(p) >= 3 and p[0] == "=best=":
            out.setdefault(p[1], float(p[2]))
    return out


def candidates():
    msolu = solu(f"{D}/minlplib.solu")
    for r in csv.DictReader(open(f"{D}/problem_sizes.csv")):
        if int(r["discrete"]) > 0 and int(r["vars"]) <= MAXV and r["name"] in msolu:
            yield ("minlplib", r["name"], f"{D}/minlplib/nl/{r['name']}.nl", msolu[r["name"]])
    for r in csv.DictReader(open(f"{D}/qplib/qplib_manifest.csv")):
        if r["usable_oracle"] == "True" and int(r["n_integral"]) > 0 and int(r["nvars"]) <= MAXV:
            yield ("qplib", r["name"], f"{D}/qplib/qplib/{r['name']}.qplib", float(r["reference"]))


def probe(c):
    try:
        return _probe(c)
    except Exception as e:  # reported per instance and counted, never hidden
        return c[0], c[1], c[3], ("ERR", f"{type(e).__name__}: {e}"[:160])


def _probe(c):
    lib, name, path, ref = c
    import discopt.solver as S

    if lib == "minlplib":
        from discopt.modeling.core import from_nl

        m = from_nl(path)
    else:
        from discopt.interfaces.qplib import from_qplib

        m = from_qplib(path)
    return lib, name, ref, S._objective_lattice(m)


def main():
    import discopt.solver as S

    assert S.__file__.startswith(os.getcwd()), S.__file__
    cs = list(candidates())
    print(f"candidates {len(cs)}", flush=True)
    ran = 0
    errs = 0
    hits = []
    with mp.get_context("spawn").Pool(6) as pool:
        for res in pool.imap_unordered(probe, cs):
            ran += 1
            lib, name, ref, lat = res
            if isinstance(lat, tuple) and lat[0] == "ERR":
                errs += 1
                print(f"ERR {lib} {name} {lat[1]}", flush=True)
            elif lat is not None:
                hits.append(res)
                print(f"HIT {lib} {name} g={lat[0]} c0={lat[1]} ref={ref}", flush=True)
    with open(sys.argv[1], "w") as f:
        w = csv.writer(f)
        w.writerow(["lib", "name", "ref", "g", "c0"])
        for lib, name, ref, lat in sorted(hits):
            w.writerow([lib, name, ref, lat[0], lat[1]])
    print(f"detector executed {ran - errs}, load errors {errs}, hits {len(hits)}", flush=True)
    sys.exit(0 if ran - errs else 1)


if __name__ == "__main__":
    main()
