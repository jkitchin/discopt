# Issue #2067 fixture

`QPLIB_8153.qplib` and its reference solution `QPLIB_8153.sol` (objective
`-1.076351999364640`, feasible to 2.7e-15), vendored unmodified from
[QPLIB](https://qplib.zib.de/) for `test_2067_convex_node_safe_bound.py`.

**License**: CC-BY 4.0. Cite Furini, Traversi, Belotti, Frangioni, Gleixner,
Gould, Liberti, Lodi, Misener, Mittelmann, Sahinidis, Vigerske and Wiegele,
*QPLIB: a library of quadratic programming instances*, Mathematical Programming
Computation 11(2), 2019, doi:10.1007/s12532-018-0147-4.

124 variables (31 integer), 97 constraints, convex MINLP (`LMQ`). Its NLP-B&B root
relaxation returns an objective 1.49e-7 *above* the reference optimum, which is
what the node bound used to be; it is kept apart from `../qplib/`, whose nine
instances are chosen for reader coverage.
