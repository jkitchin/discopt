# QPLIB test fixtures

Nine instances vendored from [QPLIB](https://qplib.zib.de/) to test
`discopt.interfaces.qplib` without requiring the full corpus.

**License**: CC-BY 4.0. Cite Furini, Traversi, Belotti, Frangioni, Gleixner,
Gould, Liberti, Lodi, Misener, Mittelmann, Sahinidis, Vigerske and Wiegele,
*QPLIB: a library of quadratic programming instances*, Mathematical Programming
Computation 11(2), 2019, doi:10.1007/s12532-018-0147-4.

The `.qplib` section layout is conditional on the three-character `probtype`
code (objective / variables / constraints), so these nine were chosen as the
smallest instances that between them exercise every branch the reader has to
take, rather than for their mathematical content:

| fixture | probtype | branch it covers |
| --- | --- | --- |
| `QPLIB_3562` | LIQ | `O=L` (no objective-quadratic block), `V=I`, `C=Q` |
| `QPLIB_3814` | QMQ | `O=Q`, `V=M` (both bound and type blocks present) |
| `QPLIB_3871` | DML | `O=D` (diagonal objective) |
| `QPLIB_3815` | QBL | `V=B` (no bound *or* type block; implicit `[0,1]`) |
| `QPLIB_3385` | LCQ | `V=C` (bounds present, type block absent) |
| `QPLIB_3496` | LGQ | `V=G`, and the `i`-prefixed solution names that forced integrality to be derived from bounds rather than read off a type code |
| `QPLIB_3852` | QBN | `C=N` (the `ncons` line itself is absent) |
| `QPLIB_0031` | QML | `C=L` (no constraint-quadratic block) |
| `QPLIB_2967` | QCC | `C=C` |

Not covered in-repo: `C=B`, `C=D` and `O=C`. The smallest instance of each is
139 KB–1.9 MB, too large to vendor. Those three branches are exercised by
`test_qplib_corpus.py`, which runs against the full corpus when it is present.

`instancedata.csv` and `qplib.solu` are the upstream metadata and best-known
objective values, subset to these nine rows.
