// Pyodide worker: owns CPython, installs discopt, runs one script at a time.
//
// Everything heavy lives here rather than on the page, because a solve is a
// synchronous call into wasm. On the main thread that would freeze the tab for
// the duration -- and the whole point of the page is to run solves that take
// long enough to be interesting.
//
// The install is the subtle part. discopt's `[project] dependencies` declare
// jax, jaxlib, highspy and pounce-solver alongside numpy and scipy, and three of
// those cannot be installed under Pyodide at all (jaxlib and highspy publish no
// emscripten wheel and never will; jax is pure Python but hard-pins jaxlib).
// None of the three is imported eagerly: every use site is a function-local
// import, so `import discopt.modeling` loads zero jax and zero highspy modules.
// So the wheel is installed with `deps: false` and the dependencies this page
// really needs are installed explicitly, in order. That is a deliberate
// divergence between the browser's dependency contract and the declared one,
// and it is recorded here rather than hidden.
//
// What that argument does NOT establish -- and an earlier version of this
// comment wrongly claimed it did -- is that no *route* reaches them. Lazy
// imports move the failure from install time to solve time; they do not remove
// it. Pure LP and MILP really do route to highspy by default, and both examples
// raised `HighsUnavailable` when first run under this page's constraints. The
// fix is below, at the install; the lesson is here: "nothing imports it at
// startup" is not "nothing needs it". The check that would have caught it is
// `python/tests/test_wasm_examples.py`, which now runs every example with these
// three modules blocked, because a plain pytest run has all three installed and
// therefore cannot see this class of failure at all.

const PYODIDE_VERSION = '0.28.3';
const params = new URLSearchParams(self.location.search);
const PYODIDE_URL = params.get('pyodide') || `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

const say = (text) => self.postMessage({ type: 'status', text });
const out = (text) => self.postMessage({ type: 'stdout', text });

// --- wheels ----------------------------------------------------------------

// Both wheels are named by a manifest written beside them by the build script,
// not by a constant here: an emscripten wheel's ABI tag carries the exact
// emscripten version Pyodide was built with, so the file name changes whenever
// either version moves. A stale constant would surface as an unhelpful micropip
// resolution error that reads like a missing package.
async function wheelFromManifest(which, overrideParam) {
  const override = params.get(overrideParam);
  if (override) return override;

  const res = await fetch(`./wheels/${which}-wheel.json`);
  if (!res.ok) {
    throw new Error(
      `no ${which} wheel is deployed here — run crates/discopt-wasm/build-wheel.sh, ` +
        `or pass ?${overrideParam}=<wheel url>.`,
    );
  }
  const manifest = await res.json();
  // An emscripten wheel is valid for exactly the Pyodide build it was compiled
  // against. Say that plainly rather than letting micropip report "no matching
  // distribution", which reads like the package does not exist.
  if (manifest.pyodide_version !== PYODIDE_VERSION) {
    throw new Error(
      `the deployed ${which} wheel was built for Pyodide ${manifest.pyodide_version}, but ` +
        `this page runs ${PYODIDE_VERSION} — rebuild it with crates/discopt-wasm/build-wheel.sh`,
    );
  }
  return `./wheels/${manifest.wheel}`;
}

// --- Pyodide ---------------------------------------------------------------

let pyodide = null;
let micropip = null;

const ready = (async () => {
  say(`loading Pyodide ${PYODIDE_VERSION} (~10 MB, cached after the first run)…`);
  // `pyodide.mjs`, not `pyodide.js`: this is a module worker, where
  // `importScripts` does not exist.
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  pyodide = await loadPyodide({
    indexURL: PYODIDE_URL,
    stdout: (line) => out(line + '\n'),
    stderr: (line) => out(line + '\n'),
  });
  await pyodide.loadPackage('micropip');
  micropip = pyodide.pyimport('micropip');
  say('ready');
  self.postMessage({ type: 'ready' });
})().catch((err) => {
  self.postMessage({ type: 'fatal', message: String(err && err.message ? err.message : err) });
});

// --- installing discopt ----------------------------------------------------

let installing = null;
// A failed install must not be cached as done: drop the rejected promise so the
// next Run retries rather than replaying the same network error forever.
const ensureDiscopt = () =>
  (installing ??= installDiscopt().catch((err) => {
    installing = null;
    throw err;
  }));

async function installDiscopt() {
  say('installing discopt (numpy, scipy, pounce, then discopt)…');

  // numpy and scipy come from Pyodide's own build. Letting micropip resolve
  // them would send it to PyPI for source distributions it cannot compile.
  await pyodide.loadPackage(['numpy', 'scipy']);

  // pounce-solver must be installed as its own step, BEFORE discopt. `deps:
  // false` below is all-or-nothing, so if pounce were left to discopt's
  // metadata it would simply never be installed, and the first solve would fail
  // on a lazy `import pounce` deep inside the solver -- long after the install
  // looked like it succeeded. Its own dependencies are only numpy and scipy,
  // which the line above has already satisfied, so this one resolves normally.
  const pounceWheel = await wheelFromManifest('pounce', 'pounce');
  await micropip.install(pounceWheel);

  // See the header: `deps: false` skips jax, jaxlib and highspy, none of which
  // has an emscripten wheel.
  //
  // `callKwargs`, NOT `micropip.install(wheel, { deps: false })`. Pyodide maps
  // a trailing JS object to a *positional* argument, so the plain call binds
  // `{deps: false}` to micropip's second parameter (`keep_going`) and the
  // `deps` flag is silently dropped -- the install then tries to resolve the
  // real dependency list and dies with
  //
  //     ValueError: Can't find a pure Python 3 wheel for:
  //     'jaxlib>=0.4', 'highspy>=1.10', 'jax>=0.4'
  //
  // Nothing about the call site looks wrong; it just quietly means something
  // else. `callKwargs` passes the object as keyword arguments, which is what
  // was meant.
  const discoptWheel = await wheelFromManifest('discopt', 'discopt');
  await micropip.install.callKwargs(discoptWheel, { deps: false });

  // Route pure LP and MILP through the in-house Rust simplex instead of HiGHS.
  //
  // This is NOT a workaround for the page: `DISCOPT_LP_MILP_BACKEND=rust` is
  // discopt's shipped, supported opt-out from the #1229 HiGHS route, and the
  // Rust route it selects is the fully certified engine that was the default
  // before that route landed. Bounds are still certified; nothing is weakened.
  //
  // It is required, not cosmetic. A pure LP dispatches to `_solve_lp_highs`
  // unconditionally when the backend is `highs` (solver.py:11939), and that
  // route deliberately has no fallback -- it raises `HighsUnavailable` rather
  // than quietly solving by some other means. Since highspy cannot install
  // here, the LP and MILP examples -- the first two in the dropdown, LP being
  // what the page opens on -- both died on:
  //
  //   discopt.solvers.lp_milp_highs.HighsUnavailable:
  //   the LP/MILP HiGHS route needs highspy>=1.10
  //
  // An earlier note in this file claimed every jax/highspy use site was
  // unreachable from the page. That was wrong; this is the correction. The
  // backend is read per solve (`_lp_milp_backend()`), not at import, but it is
  // set before the import below so the environment is settled in one place.
  //
  // The consequence is worth stating plainly rather than hiding: the LP and
  // MILP numbers this page prints come from the Rust simplex, so they are not a
  // measurement of the HiGHS route a desktop install takes by default.
  await pyodide.runPythonAsync(`
import os

os.environ["DISCOPT_LP_MILP_BACKEND"] = "rust"
`);

  // Fail loudly and immediately if the deps-false gamble was wrong, rather than
  // letting the first solve report a confusing ImportError from inside the
  // solver. This import is the exact claim the header makes.
  //
  // Note what this guard does and does not cover: it proves nothing was pulled
  // in at *import* time. Both examples above imported cleanly and failed at
  // *solve* time, which is why `python/tests/test_wasm_examples.py` runs every
  // example with these three modules blocked.
  await pyodide.runPythonAsync(`
import discopt.modeling as dm  # noqa: F401
import sys

_leaked = sorted({n.split(".")[0] for n in sys.modules} & {"jax", "jaxlib", "highspy"})
if _leaked:
    raise ImportError(
        "discopt imported " + ", ".join(_leaked) + " on this page, which the browser "
        "build installs no wheel for. The install needs revisiting: see the header "
        "of crates/discopt-wasm/web/worker.js."
    )
`);
  say('ready');
}

// --- running ---------------------------------------------------------------

self.onmessage = async (event) => {
  if (event.data.type !== 'run') return;
  try {
    await ready;
    if (!pyodide) return;

    // Installing happens before the timer starts: the download is not the
    // model's solve time, and reporting it as such would be a lie about the
    // solver.
    await ensureDiscopt();

    self.postMessage({ type: 'running' });
    const started = performance.now();
    await pyodide.runPythonAsync(event.data.code);
    self.postMessage({ type: 'done', ms: performance.now() - started });
  } catch (err) {
    // A Python exception arrives with its traceback in `message`; show it as the
    // script's own output rather than as a page-level failure.
    self.postMessage({ type: 'error', message: String(err && err.message ? err.message : err) });
  }
};
