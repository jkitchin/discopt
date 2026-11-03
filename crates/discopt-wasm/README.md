# discopt in the browser

`web/` is a single page that runs discopt entirely client-side: [Pyodide] supplies
CPython compiled to WebAssembly, discopt is installed into it as an emscripten wheel,
and a dropdown offers one model of each problem class — LP, MILP, QP, MIQP, NLP, MINLP —
plus one infeasible model, whose IIS the page draws as a table and a graph.
Nothing is uploaded and there is no server component; the solve happens in the tab.

This mirrors POUNCE's `crates/pounce-wasm/web-python/` page, deliberately: same Pyodide
version, same worker shape, same install discipline. The two pages are the same idea
applied to the two solvers, and the versions below must stay in lockstep with POUNCE's
because **both wheels install into one Pyodide instance** — discopt's Rust core calls
POUNCE for its NLP subproblems.

## Why Pyodide and not a standalone wasm module

POUNCE ships two browser pages. One is a standalone `wasm32-wasip1` module with a
four-function C ABI and no Python at all; the other is Pyodide. discopt cannot take the
first route: its branch-and-bound orchestration is Python (`python/discopt/solver.py`
alone is larger than several of the Rust crates), so there is no small C-ABI surface to
export. The Python layer has to come along, which means CPython has to come along.

## Layout

| Path | What it is |
| --- | --- |
| `web/index.html` | The page: masthead, dropdown, editor, console, IIS view. |
| `web/app.js` | Page logic — dropdown, Run/Cancel, theme, console. |
| `web/worker.js` | Pyodide bootstrap, the wheel install, and running a script. |
| `web/examples.js` | The six example models, one per problem class, and the infeasible one. |
| `web/iis_view.py` | Turns an `IISResult` into the dict the IIS view draws (runs in Pyodide). |
| `web/iis_render.js` | Draws that dict: summary, table, constraint–variable graph. |
| `web/vendor/` | CodeMirror 6, bundled and committed (see below). |
| `web/wheels/` | The two emscripten wheels and their manifests, **committed**. |

## Running it locally

The page is static, but it must be *served* — ES modules and workers do not load from
`file://`:

```bash
python -m http.server -d crates/discopt-wasm/web 8000
# then open http://localhost:8000/
```

The wheels are committed, so a fresh clone runs. To point the page at wheels hosted
elsewhere instead:

```
http://localhost:8000/?discopt=<url>&pounce=<url>&pyodide=<url>
```

## The examples are executed, not just shipped

`web/examples.js` holds the seven models as strings, and `python/tests/test_wasm_examples.py`
parses that file and **runs every one of them** against the local discopt. An example
that raises is worse than no example, and the failure would otherwise only ever appear
in somebody's browser. The test lives under `python/tests/` rather than next to the page
because that is the only tree `pyproject.toml` lists in `testpaths`.

The six problem-class models are the same six as `docs/notebooks/problem_classes.ipynb`,
so the page and the notebook cannot drift into disagreeing about what discopt does with
each class. The seventh, `IIS`, is infeasible on purpose and is expected to print
`infeasible`.

## The IIS view

When a script leaves an `IISResult` in a top-level variable (`iis = m.compute_iis()`),
the page shows it below the output, under "Why is it infeasible?":

- a **summary**: how many members (constraints and bounds) over how many variables, the
  model's size, and whether the IIS is *proven* minimal — `compute_iis` keeps a member
  when a feasibility solve is inconclusive, and the page says so instead of implying
  minimality;
- a **table**: each member, its kind, the variable *elements* it touches (`a[1]`, not
  `a`), and a button that selects the line declaring it in the editor. Constraints are
  found by their literal `name="..."`, bounds by the `m.continuous("x", ...)` call; a
  name built at run time (an f-string) is reported as such rather than guessed;
- a **constraint–variable graph**: members on the left, variables on the right, an edge
  for each variable a member touches. Constraints outside the IIS that share a variable
  with it are drawn dashed and muted below a "not in the IIS" heading (at most 12, the
  rest counted), to show where the conflict sits in the model. They are not part of it.

The table is the primary view and holds everything the graph shows; the graph is an
image to assistive technology, labelled to point at the table. Focusing a row's line
button, or pointing at a row or node, highlights that member and its links. Membership
is never shown by colour alone (solid heavy border vs. dashed, plus separate headings).

The script stays plain discopt: nothing page-specific is imported, so it runs unchanged
outside the browser. After the script finishes, `worker.js` calls
`discopt_iis_view.collect_json` on the script's namespace. Each run gets a **fresh
namespace**, as `python script.py` would, so an IIS from an earlier run is never redrawn.
`iis_view.py` is served beside the page and written into Pyodide's file system rather
than shipped in the wheel, so the view works with the committed wheel. It depends on
`discopt.infeasibility.IISResult`, `discopt.modeling.core` and
`discopt.serialize._children`, a private helper, which is why
`python/tests/test_wasm_iis_view.py` imports it from its path and checks its output
against the `IISResult` it describes.

### Testing it

| Check | What it covers |
| --- | --- |
| `pytest python/tests/test_wasm_iis_view.py` | The dict, against the `IISResult`: members, element-level variables, whole-vector uses, context and its cap, model matching. |
| `node examples_in_pyodide.mjs` | The same helper under real Pyodide on the IIS example, and that no feasible example yields one. |
| `npm run test:render` | `iis_render.js`'s pure functions (line lookup, summary) under `node --test`. |
| `npm run test:page` | The page in headless Chromium: the run, table and graph agreeing, highlighting, the editor link, axe (WCAG 2.2 AA) in both themes, no sideways scroll at 320 px, a later run clearing the view. |

`test:page` serves the page and the npm copy of Pyodide itself, so it needs no server.
The pytest module is marked `smoke` and runs in the ordinary Python lanes; the other three
run in the `Browser page` workflow (`.github/workflows/wasm.yml`).

## CodeMirror is vendored, not fetched

`web/vendor/codemirror.bundle.js` is committed (~460 kB, ~155 kB gzipped). A CDN
`<script>` would put a third-party request on every visit and break offline and
self-hosted deployments. `web/vendor/codemirror.versions.json` records exactly which
package versions went in, and `web/vendor/build-codemirror.sh` reproduces the bundle
from them — run it by hand to bump CodeMirror, and commit the regenerated bundle
together with the updated versions file.

## The install order, and the three dependencies that are skipped

`worker.js` installs in three steps and the order matters:

```js
await pyodide.loadPackage(['numpy', 'scipy']);   // Pyodide's own builds
await micropip.install(pounceWheelUrl);          // deps on; already satisfied
await micropip.install.callKwargs(discoptWheelUrl, { deps: false });
```

`callKwargs` is not decoration. Pyodide binds a trailing JS object to a *positional*
argument, so `micropip.install(url, { deps: false })` hands the object to micropip's
second parameter and drops the flag silently — the install then tries to resolve jax and
dies. The call that looks right is the one that is wrong.

discopt's `[project] dependencies` declare `jax`, `jaxlib` and `highspy` alongside
numpy, scipy and `pounce-solver`. Three of those cannot be installed under Pyodide:
`jaxlib` and `highspy` are C++ extension wheels with no emscripten build, and `jax`
hard-pins `jaxlib`. Every use site is a function-local import, so installing without
them succeeds and `import discopt.modeling` pulls in none of them, and the wheel goes in
with `deps: false`.

That is a real divergence between the browser's dependency contract and the declared
one, so it is checked rather than assumed: immediately after the install, `worker.js`
imports `discopt.modeling` and raises if `jax`, `jaxlib` or `highspy` turned up in
`sys.modules`. If discopt ever grows an eager import of one of them, this page fails
loudly at install time instead of confusingly at the first solve.

That is *not* the same as saying no route reaches them, and an earlier version of this
file claimed it was. Lazy imports move the failure from install time to solve time; they
do not remove it. Pure LP and MILP really do route to highspy by default, and both
examples raised `HighsUnavailable` the first time they were run under these constraints.
So `worker.js` also sets

```js
os.environ["DISCOPT_LP_MILP_BACKEND"] = "rust"
os.environ["DISCOPT_CONVEX_MINLP_ROUTE"] = "0"
```

The first is discopt's documented opt-out from the #1229 HiGHS route, selecting the
certified in-house Rust simplex; the second turns off the convex-MINLP OA route, whose
master is always HiGHS (performance-plan §25.13), so a convex MINLP takes the default
spatial path. The consequence worth knowing: **the LP and MILP numbers
this page prints come from the Rust simplex, not from HiGHS.** Both are certified, and
`python/tests/test_wasm_examples.py` runs the examples with that variable set and with
the three modules made unimportable, so the page's environment is the tested one.

Because `deps: false` is all-or-nothing, `pounce-solver` **must** be installed as its
own step, before discopt. Left to discopt's metadata it would simply never be installed,
and the first solve would fail on a lazy `import pounce` deep inside the solver.

## Threads: there are none

Pyodide has no pthreads. `threading.Thread.start()` raises `RuntimeError: can't start new
thread`, `os.cpu_count()` is 1, and `sys.platform` is `emscripten`.

This matters because rayon builds its global pool lazily and **panics** when that build
fails, and the panic crosses the PyO3 boundary as `pyo3_runtime.PanicException` — which
derives from `BaseException` deliberately, so none of the `except Exception` serial
fallbacks in `solver.py` catch it. The MIQP example died exactly this way; the other five
survived only because they sat under the batch-size floors that guard the parallel
branches, so a larger model would have hit it too.

The fix is not in this directory and not specific to the browser: each parallel entry
point now asks whether threads exist before entering, via
`discopt_core::parallel::threads_available` in Rust and `_os_threads_available` in
`solver.py`. A threadless platform takes the serial path that was always there. That also
covers POUNCE's waves, since discopt decides before calling into them.

Consequently the wheel is built with the **default** feature set, exactly like a desktop
build — see the comment in `build-wheel.sh` for the `--no-default-features` attempt that
did not work and why.

## Version pins

These must match POUNCE's `crates/pounce-wasm/build-wheel.sh` byte for byte. An
emscripten wheel is valid for exactly the Pyodide build it was compiled against, and two
wheels sharing one interpreter must have been compiled against the same one:

| | |
| --- | --- |
| Pyodide | 0.28.3 |
| pyodide-build | 0.39.0 |
| emscripten | 4.0.9 |
| Rust | `nightly-2025-06-27`, with the wasm-EH sysroot from [`pyodide/rust-emscripten-wasm-eh-sysroot`] release `emcc-4.0.9_nightly-2025-06-27` |

`worker.js` checks the staged wheel's manifest against the Pyodide version it is loading
and refuses a mismatch with a message naming the rebuild script, because micropip's own
error for this reads like the package does not exist.

Nothing currently enforces the lockstep *across the repository boundary*: a Pyodide bump
here requires a coordinated rebuild in the POUNCE repo, and the only thing that will
tell you it was missed is this page failing to install.

[Pyodide]: https://pyodide.org
[`pyodide/rust-emscripten-wasm-eh-sysroot`]: https://github.com/pyodide/rust-emscripten-wasm-eh-sysroot
