# #2142 §114 entry experiment — pre-registered 2026-11-05 before any run of the flag

Diagnosis (c10/c11): 91 of the 94 captured cstr-hull boxes are empty for every integral
binary assignment (exact forward evaluation: a perspective `w` of a binary fixed at 0 must be
1e8, or a product of such `w` 1e16 / 1e24, and the box excludes it). The kernel keeps them open
because its step-5 integer rounding ignores a change <= EPS = 1e-9 while `cap_hi(.., 0)` pads a
binary's upper bound to exactly 1e-9. Pre-rounding the integer bounds lets the unchanged kernel
propagation empty 91/94 (c11).

Fix: `DISCOPT_INT_ROUND_EXACT` (default OFF) — snap on any strict improvement.

E1 (instrument, the 94 boxes, 15 rounds, no cutoff, no pre-rounding):
OFF must empty 0/94 (as c11); ON must empty >= 85/94. Kill: ON < 85.

E2 (entry, cstr hull, `gdp_method="hull"`, deterministic, max_nodes=10000, OFF vs ON):
Kill: ON bound @10k < 1.0, or any bound > 3.0620073 (oracle) + 1e-6 (soundness, stop at once).

Then, only if E2 passes: cstr hull to certification under the 300 s GDP-parity budget, and the
§5 panels as §112 (corpus 74 at max_nodes=2000, the §107 extras, GDP parity 24 x {big-m, hull}
at max_nodes=5000), `--assert-unfired-identical`. Graduate only if cert-clean AND net-positive.
