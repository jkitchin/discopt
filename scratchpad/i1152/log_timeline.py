"""#1152: timestamped solver log timeline for one instance."""

from __future__ import annotations

import logging
import os
import sys
import time

from discopt.modeling.core import from_nl

name, T = sys.argv[1], float(sys.argv[2])
T0 = time.perf_counter()


class _Fmt(logging.Formatter):
    def format(self, record):
        return f"{record.relativeCreated / 1000.0:7.2f} {record.name} {record.getMessage()}"


h = logging.StreamHandler(sys.stdout)
h.setFormatter(_Fmt())
logging.root.addHandler(h)
logging.root.setLevel(logging.DEBUG)
for noisy in ("jax", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

m = from_nl(os.path.join("python/tests/data/minlplib_nl", name + ".nl"))
t0 = time.perf_counter()
r = m.solve(time_limit=T)
print(f"RESULT wall={time.perf_counter() - t0:.2f} bound={r.bound} nodes={r.node_count}")
