# discopt in the browser

`web/` is a single page that runs discopt entirely client-side: [Pyodide] supplies
CPython compiled to WebAssembly, discopt is installed into it as an emscripten wheel,
and a dropdown offers one model of each problem class — LP, MILP, QP, MIQP, NLP, MINLP.
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
| `web/index.html` | The page: masthead, dropdown, editor, console. |
| `web/app.js` | Page logic — dropdown, Run/Cancel, theme, console. |
| `web/worker.js` | Pyodide bootstrap, the wheel install, and running a script. |
| `web/examples.js` | The six example models, one per problem class. |
| `web/vendor/` | CodeMirror 6, bundled and committed (see below). |
| `web/wheels/` | Staging for the built wheels; contents are not committed. |

## Running it locally

The page is static, but it must be *served* — ES modules and workers do not load from
`file://`:

```bash
python -m http.server -d crates/discopt-wasm/web 8000
# then open http://localhost:8000/
```

Without a staged wheel the page loads, the editor works, and the first Run reports that
no wheel is deployed. To point it at wheels hosted elsewhere:

```
http://localhost:8000/?discopt=<url>&pounce=<url>&pyodide=<url>
```

## The examples are executed, not just shipped

`web/examples.js` holds the six models as strings, and `python/tests/test_wasm_examples.py`
parses that file and **runs every one of them** against the local discopt. An example
that raises is worse than no example, and the failure would otherwise only ever appear
in somebody's browser. The test lives under `python/tests/` rather than next to the page
because that is the only tree `pyproject.toml` lists in `testpaths`.

The six models are the same six as `docs/notebooks/problem_classes.ipynb`, so the page
and the notebook cannot drift into disagreeing about what discopt does with each class.

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
await micropip.install(discoptWheelUrl, { deps: false });
```

discopt's `[project] dependencies` declare `jax`, `jaxlib` and `highspy` alongside
numpy, scipy and `pounce-solver`. Three of those cannot be installed under Pyodide:
`jaxlib` and `highspy` are C++ extension wheels with no emscripten build, and `jax`
hard-pins `jaxlib`. None is reachable from anything this page does — every use site is a
function-local import behind a route the page does not take — so the wheel goes in with
`deps: false`.

That is a real divergence between the browser's dependency contract and the declared
one, so it is checked rather than assumed: immediately after the install, `worker.js`
imports `discopt.modeling` and raises if `jax`, `jaxlib` or `highspy` turned up in
`sys.modules`. If discopt ever grows an eager import of one of them, this page fails
loudly at install time instead of confusingly at the first solve.

Because `deps: false` is all-or-nothing, `pounce-solver` **must** be installed as its
own step, before discopt. Left to discopt's metadata it would simply never be installed,
and the first solve would fail on a lazy `import pounce` deep inside the solver.

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
