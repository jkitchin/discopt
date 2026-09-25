# Run discopt in your browser

There is a version of discopt that needs no installation at all:

```{raw} html
<p style="font-size:1.15rem;margin:1.2rem 0;">
  <a href="wasm/index.html"><strong>&rarr; Open the in-browser solver</strong></a>
</p>
```

It is a single page with a code editor, a Run button and a console. Pick a model
from the dropdown — one for each problem class, LP through MINLP — edit it, and
solve it. The editor is real Python: anything you can write against
{mod}`discopt.modeling` works.

## What is actually running

The whole solver. [Pyodide] supplies CPython compiled to WebAssembly, and
discopt is installed into it as an emscripten wheel together with
[POUNCE][pounce], the pure-Rust interior-point code discopt uses for its NLP
subproblems {cite:p}`Wachter2006`. The Rust core — the simplex, the
branch-and-bound tree, the AD tape — is the same code as a desktop install,
compiled to wasm rather than to native.

So the solve happens in your tab. Nothing is uploaded, there is no server doing
the work, and the page keeps running if you go offline after it loads. The cost
is the download: roughly 12 MB of Pyodide, numpy and scipy plus about 7 MB of
the two wheels on first visit, which the browser then caches.

## How it differs from a local install

Two differences are worth knowing before you compare numbers with a desktop run.

**LP and MILP go through the in-house Rust simplex, not HiGHS.** A local install
routes pure LP and MILP models to HiGHS; `highspy` has no WebAssembly build, so
the page sets `DISCOPT_LP_MILP_BACKEND=rust`, which is discopt's own supported
opt-out. Both routes return discopt-verified certificates, but they are different
engines and may report different node counts and timings for the same model.

**There are no threads.** WebAssembly in this configuration has no pthreads, so
the parallel node waves fall back to the serial path they already had. Results
are identical — the serial branch-and-bound tree is deterministic and is the
reference the parallel one is checked against — but a large model will be slower
in the tab than on your machine, on top of the usual wasm penalty.

Everything else is the same solver, so a model that solves here solves locally.

## When to use it

For trying discopt before installing it, for a small model you want to share as
a link, and for teaching — a class can open a URL instead of debugging six
different Python environments. For real work, install it: a native build is
faster, has HiGHS and the parallel paths, and can read your files.

[Pyodide]: https://pyodide.org
[pounce]: https://github.com/jkitchin/pounce
