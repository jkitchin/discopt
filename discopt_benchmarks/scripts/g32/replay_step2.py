"""#1917 G3.2 step 2 differential: DISCOPT_RUST_MILP_TABLEAU_CMIR=0 vs =1 on the strong root
profile, with coefficient tightening (its measured precondition) held ON in both arms.

The pre-#2136 version of this script toggled DISCOPT_RUST_MILP_COEF_TIGHTEN, not the
tableau separator: its ``cmir_cuts=True`` is #2107's aggregated c-MIR, so it never
measured step 2. It also trusted HiGHS's status. Both are fixed in ``replay.py``.
usage: replay_step2.py T out.jsonl pkl..."""

import pathlib
import subprocess
import sys

here = pathlib.Path(__file__).resolve().parent
sys.exit(
    subprocess.call(
        [
            sys.executable,
            "-u",
            str(here / "replay.py"),
            "--flag",
            "tabcmir",
            "--profile",
            "strong",
            "--base",
            "DISCOPT_RUST_MILP_COEF_TIGHTEN=1",
            *sys.argv[1:],
        ]
    )
)
