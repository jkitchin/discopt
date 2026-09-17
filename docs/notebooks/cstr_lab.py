#!/usr/bin/env python
"""The "laboratory" behind ``cstr_fit_doe_optimize.ipynb``.

This script is the *only* place the true kinetics live. The notebook fits a model
to the numbers this produces and never imports this file, so nothing downstream
can accidentally read the answer it is supposed to estimate -- the same reason a
real study keeps the instrument and the analysis apart.

It writes a CSV of steady-state measurements from a single isothermal CSTR fed
pure A, with the series reaction A -> B -> C:

    F (C_A0 - C_A) = V k1 C_A          F (0 - C_B) = V (k2 C_B - k1 C_A)

solved in closed form and corrupted with Gaussian measurement noise.

Usage
-----
Initial campaign (five flow rates, the notebook's committed input)::

    python cstr_lab.py --flows 0.6,0.9,1.3,1.9,2.6 --seed 7 \
        --out data/cstr_campaign1.csv

One run at a flow rate the notebook's experiment design asked for::

    python cstr_lab.py --flows 0.57 --seed 11 --out data/cstr_run6.csv

The true constants, for scoring an analysis that is already finished::

    python cstr_lab.py --reveal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ── The rig's specification: known to the experimenter, and to the notebook. ──
V = 1.0  # reactor volume, L
CA0 = 2.0  # feed concentration of A, mol/L
SIGMA = 0.03  # 1-sigma repeatability of the concentration assay, mol/L

# ── Ground truth: known to nobody. Do not import these into the notebook. ──
K1_TRUE = 1.50  # 1/min
K2_TRUE = 0.40  # 1/min


def outlet(F, k1, k2, CA0=CA0, V=V):
    """Steady-state outlet concentrations ``(CA, CB, CC)`` at flow rate *F*."""
    tau = V / np.asarray(F, dtype=float)
    CA = CA0 / (1.0 + k1 * tau)
    CB = k1 * tau * CA / (1.0 + k2 * tau)
    return CA, CB, CA0 - CA - CB


def measure(flow_rates, seed):
    """Run the reactor at each flow rate and assay the outlet."""
    F = np.atleast_1d(np.asarray(flow_rates, dtype=float))
    if np.any(F <= 0.0):
        raise ValueError(f"flow rates must be positive, got {F.tolist()}")
    CA, CB, _ = outlet(F, K1_TRUE, K2_TRUE)
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            F,
            CA + SIGMA * rng.standard_normal(F.shape),
            CB + SIGMA * rng.standard_normal(F.shape),
            np.full(F.shape, SIGMA),
        ]
    )


def write_csv(rows, path):
    """Write the assay table, with the rig's specification in the header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Steady-state CSTR assay, written by cstr_lab.py\n")
        fh.write(f"# rig: V={V} L, CA0={CA0} mol/L\n")
        fh.write("# F: L/min; CA, CB, sigma: mol/L\n")
        fh.write("F,CA,CB,sigma\n")
        for F, CA, CB, sig in rows:
            fh.write(f"{F:.6f},{CA:.6f},{CB:.6f},{sig:.6f}\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--flows",
        help="comma-separated flow rates in L/min, e.g. 0.6,0.9,1.3",
    )
    ap.add_argument("--seed", type=int, help="measurement-noise seed")
    ap.add_argument("--out", help="CSV path to write")
    ap.add_argument(
        "--reveal",
        action="store_true",
        help="print the true constants as JSON. Only for scoring an analysis "
        "that is already finished -- reading this before you fit is cheating.",
    )
    args = ap.parse_args(argv)

    if args.reveal:
        print(json.dumps({"k1": K1_TRUE, "k2": K2_TRUE, "V": V, "CA0": CA0, "sigma": SIGMA}))
        return 0

    missing = [f"--{n}" for n in ("flows", "seed", "out") if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"{', '.join(missing)} required unless --reveal is given")

    flows = [float(tok) for tok in args.flows.split(",") if tok.strip()]
    if not flows:
        raise SystemExit("no flow rates given")
    rows = measure(flows, args.seed)
    path = write_csv(rows, args.out)
    print(f"wrote {len(rows)} run(s) to {path}")
    for F, CA, CB, _ in rows:
        print(f"  F = {F:6.3f} L/min   CA = {CA:7.4f}   CB = {CB:7.4f}  mol/L")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
