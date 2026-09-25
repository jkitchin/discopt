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
// None of them is needed to solve: measured on a desktop, `import discopt`
// loads zero jax and zero highspy modules, and a full MINLP solve still loads
// zero of either -- every use site is a function-local import behind a route
// this page does not take.
//
// So the wheel is installed with `deps: false` and the dependencies this page
// really needs are installed explicitly, in order. That is a deliberate
// divergence between the browser's dependency contract and the declared one,
// and it is recorded here rather than hidden: if discopt ever grows an eager
// import of one of those three, this page breaks and the declared metadata was
// right all along.

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

  // See the header: `deps: false` skips jax, jaxlib and highspy, which are
  // declared but unreachable here and unnecessary for every route this page
  // takes.
  const discoptWheel = await wheelFromManifest('discopt', 'discopt');
  await micropip.install(discoptWheel, { deps: false });

  // Fail loudly and immediately if the deps-false gamble was wrong, rather than
  // letting the first solve report a confusing ImportError from inside the
  // solver. This import is the exact claim the header makes.
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
