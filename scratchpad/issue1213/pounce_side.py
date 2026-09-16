#!/usr/bin/env python3
"""pounce-nl side of the issue #1213 differential: run pounce.read_nl over the
same corpus discopt-core/examples/nl_audit.rs walks, and emit one JSON line per
file. Prints an attempted count and exits non-zero if it is zero."""

import json
import pathlib
import sys

import pounce

corpus = pathlib.Path(sys.argv[1])
files = sorted(corpus.glob("*.nl")) if corpus.is_dir() else [corpus]
attempted = 0
out = []
for f in files:
    attempted += 1
    rec = {"name": f.stem}
    try:
        p = pounce.read_nl(str(f))
        rec.update(
            ok=True,
            n=p.n,
            m=p.m,
            nnz_jac=p.nnz_jac,
            nnz_hess=p.nnz_hess,
            variant=str(
                p.variant()
                if callable(getattr(p, "variant", None))
                else getattr(p, "variant", None)
            ),
        )
    except BaseException as e:  # instruments never swallow: record the type
        rec.update(ok=False, err_kind=type(e).__name__, err=str(e).replace("\n", " ")[:300])
    out.append(rec)
for r in out:
    print(json.dumps(r))
print(f"pounce_side: attempted {attempted} file(s)", file=sys.stderr)
if attempted == 0:
    sys.exit("PROBE DID NOT FIRE: zero .nl files attempted")
