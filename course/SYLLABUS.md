# Syllabus

Thirty lessons in three tracks. Each row links the lesson directory and the
primary literature backing it (full BibTeX entries in `docs/references.bib`).

## Basic track — modeling and solving

| #  | Title                              | Primary references                        |
| -- | ---------------------------------- | ----------------------------------------- |
| 01 | Introduction to optimization       | Bertsimas1997, Williams2013, Wolsey1998   |
| 02 | Linear programming fundamentals    | Bertsimas1997, Dantzig1963                |
| 03 | Modeling in `discopt`              | Williams2013                              |
| 04 | Sensitivity, duality, shadow prices| Bertsimas1997, Boyd2004                   |
| 05 | Integer programming basics         | Wolsey1998, Williams2013, Conforti2014    |
| 06 | Branch-and-bound, conceptually     | Land1960, Dakin1965, Wolsey1998           |
| 07 | Convexity and why it matters       | Boyd2004, Rockafellar1970                 |
| 08 | Unconstrained nonlinear optimization | Nocedal2006                             |
| 09 | Constrained NLP — KKT and IPOPT    | Nocedal2006, Wachter2006, KKT references  |
| 10 | Capstone (basic): a real model     | Williams2013                              |

## Intermediate track — algorithms and structure

| #  | Title                              | Primary references                        |
| -- | ---------------------------------- | ----------------------------------------- |
| 11 | Simplex revealed                   | Dantzig1963, Bland1977, Bertsimas1997     |
| 12 | Interior-point methods             | Karmarkar1984, Mehrotra1992, Wright1997   |
| 13 | Cutting planes                     | Gomory1958, Balas1993, Cornuejols2008     |
| 14 | Branching rules                    | Achterberg2005, Achterberg2007            |
| 15 | Presolve and FBBT                  | Belotti2009, Savelsbergh1994              |
| 16 | Convex relaxations                 | McCormick1976, Adjiman1998, Tawarmalani2005 |
| 17 | Conic optimization (SOCP/SDP)      | BenTal2001, Boyd2004                      |
| 18 | Decomposition methods              | Benders1962, DantzigWolfe1960             |
| 19 | Stochastic programming             | Birge2011, Shapiro2009                    |
| 20 | Capstone (intermediate)            | mixed                                     |

## Advanced track — frontiers

| #  | Title                              | Primary references                        |
| -- | ---------------------------------- | ----------------------------------------- |
| 21 | Spatial branch-and-bound           | Tawarmalani2005, Smith1999, Belotti2009   |
| 22 | Global optimization theory         | Floudas1999, HorstTuy1996                 |
| 23 | Generalized disjunctive programming| Raman1994, Grossmann2013, Trespalacios2015|
| 24 | Neural-net-embedded MINLP          | Fischetti2018, Anderson2020, Ceccon2022   |
| 25 | Robust optimization                | BenTal2009, BertsimasSim2004              |
| 26 | Machine learning for optimization  | Bengio2021, Gasse2019                     |
| 27 | Differentiable optimization        | Amos2017, Agrawal2019                     |
| 28 | Bilevel and complementarity        | Dempe2002, LuoPangRalph1996               |
| 29 | Reformulations and symmetry        | Liberti2012, Margot2010                   |
| 30 | Capstone (advanced)                | mixed                                     |

## Pacing

The course is self-paced. A reasonable target is **one lesson per week** for
basic, **one every 1-2 weeks** for intermediate, and **one every 2-3 weeks**
for advanced. The total course is roughly equivalent to a one-year graduate
sequence in optimization.

## Prerequisites

- **Basic**: calculus, linear algebra, comfort with Python.
- **Intermediate**: completion of basic track or equivalent (an undergraduate
  OR or convex optimization course).
- **Advanced**: completion of intermediate track and at least one of: a
  graduate optimization course, research experience, or significant industrial
  experience formulating MINLPs.

## Assessment

Each lesson awards up to 100 points distributed across exercise correctness,
writing quality, and (on capstones) experimental rigor. The `course-assessor`
skill normalizes grading across sessions. Track completion requires ≥ 70 / 100
on every lesson.
