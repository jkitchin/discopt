"""#1917 G3.2 step 1 differential: DISCOPT_RUST_MILP_COEF_TIGHTEN=0 vs =1 on the captured
options. Thin wrapper over ``replay.py`` (#2136 fixed the oracle there).
usage: replay_step1.py T out.jsonl pkl..."""

import pathlib
import subprocess
import sys

here = pathlib.Path(__file__).resolve().parent
sys.exit(
    subprocess.call(
        [sys.executable, "-u", str(here / "replay.py"), "--flag", "coef", *sys.argv[1:]]
    )
)
