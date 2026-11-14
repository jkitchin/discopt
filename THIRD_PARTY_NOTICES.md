# Third-party notices

discopt is licensed under the Eclipse Public License 2.0 (see `LICENSE`). Parts of it are
derived from third-party software under the licenses reproduced below. Each derived file says
so in its header.

## Why discopt ports rather than links

discopt translates selected algorithms from other open-source solvers into Rust instead of
linking their native libraries. The reasons: deep integration with discopt's own LP, tree and
certificate machinery; behaviour compatible with the original (checked where possible by
bit-exact replay against the original program); and a pure-Rust build that needs no C, C++ or
Fortran compiler and ships no native solver library, which keeps installation and distribution
simple. Every translated or derived file says where it comes from and under which license.

## HiGHS (MIT License)

`crates/discopt-core/src/lp/highs_cutgen/` (`cdouble`, `random`, `integers`, `core`,
`sparse_vec`, `transformed_lp`, `separators`, `pool`, `vbound`, `domain`, `modk`, and the
pdqsort part of `stdalgo`), and the HiGHS-derived primal heuristics, child selection and
strong-branching budget in `crates/discopt-core/src/bnb/milp_driver.rs`, and the trivial
heuristics and Feasibility Jump solver in `crates/discopt-core/src/bnb/feas_jump.rs`
(from `highs/mip/HighsMipSolverData.cpp`, `HighsFeasibilityJump.cpp` and
`feasibilityjump.hh`), are Rust translations
of, or are derived from, HiGHS (https://github.com/ERGO-Code/HiGHS), modified for discopt. The
HiGHS patches under `docs/dev/data/highs-parity/gap-plan-p3/` are diffs of HiGHS source.

```
MIT License

Copyright (c) 2026 HiGHS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## pdqsort (zlib License)

`crates/discopt-core/src/lp/highs_cutgen/stdalgo.rs` contains a Rust translation of Orson
Peters' pdqsort (as vendored by HiGHS in `extern/pdqsort`). This is an altered version of the
original software.

```
Copyright (c) 2021 Orson Peters <orsonpeters@gmail.com>

This software is provided 'as-is', without any express or implied warranty. In no event will the
authors be held liable for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial
applications, and to alter it and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the
   original software. If you use this software in a product, an acknowledgment in the product
   documentation would be appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be misrepresented as
   being the original software.

3. This notice may not be removed or altered from any source distribution.
```
