// End-to-end test of the emscripten wheels under real Pyodide.
//
// This is the same Pyodide build, the same two wheels, the same install order
// and the same environment variable the page's worker uses -- fetched over HTTP
// from the same local server, so the wheels are exercised exactly as a visitor
// would get them. What it does not cover is the DOM: CodeMirror, the worker
// plumbing and the theme are still unverified by this.
//
// Every example is run and counted; the script exits non-zero if fewer than six
// ran or if any failed, so a harness that silently did nothing cannot report a
// pass.
import { loadPyodide } from 'pyodide';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// The page must be *served* -- micropip installs wheels over HTTP, not from
// disk. Start one first:
//
//   python -m http.server -d crates/discopt-wasm/web 8731
//   node crates/discopt-wasm/tests/examples_in_pyodide.mjs
//
// Override the origin with the first argument or DISCOPT_WASM_BASE.
const BASE = (process.argv[2] || process.env.DISCOPT_WASM_BASE || 'http://localhost:8731')
  .replace(/\/$/, '');
const WEB = join(dirname(fileURLToPath(import.meta.url)), '..', 'web');
const PYODIDE_VERSION = '0.28.3';

const fail = (msg) => { console.error('FAIL: ' + msg); process.exit(1); };

// Parse examples.js the same way the Python test does.
const js = readFileSync(join(WEB, 'examples.js'), 'utf8');
const start = js.indexOf('[', js.indexOf('export const EXAMPLES'));
let body = js.slice(start, js.lastIndexOf(']') + 1);
body = body.replace(/^(\s*)(id|label|title|code):/gm, '$1"$2":').replace(/,(\s*[}\]])/g, '$1');
const examples = JSON.parse(body);
if (examples.length !== 6) fail(`parsed ${examples.length} examples, expected 6`);

// The worker resolves wheels through a manifest; do the same, including its
// Pyodide-version check, so a mismatched manifest fails here too.
async function wheelFromManifest(which) {
  const res = await fetch(`${BASE}/wheels/${which}-wheel.json`);
  if (!res.ok) fail(`no ${which} manifest: HTTP ${res.status}`);
  const m = await res.json();
  if (m.pyodide_version !== PYODIDE_VERSION) {
    fail(`${which} wheel targets Pyodide ${m.pyodide_version}, running ${PYODIDE_VERSION}`);
  }
  return `${BASE}/wheels/${m.wheel}`;
}

const t0 = Date.now();
const pyodide = await loadPyodide();
console.log(`pyodide up (${Date.now() - t0} ms)`);

// An emscripten wheel is valid for exactly one Pyodide build. If the installed
// npm package drifts from the version the wheels were compiled against, this
// harness would be testing a combination nobody ships -- passing or failing for
// reasons that say nothing about the page.
if (pyodide.version !== PYODIDE_VERSION) {
  fail(`loaded Pyodide ${pyodide.version}, but the wheels target ${PYODIDE_VERSION}`);
}

await pyodide.loadPackage(['numpy', 'scipy', 'micropip']);
const micropip = pyodide.pyimport('micropip');

const pounceWheel = await wheelFromManifest('pounce');
console.log(`installing ${pounceWheel.split('/').pop()}`);
await micropip.install(pounceWheel);

const discoptWheel = await wheelFromManifest('discopt');
console.log(`installing ${discoptWheel.split('/').pop()} (deps: false)`);
try {
  await micropip.install.callKwargs(discoptWheel, { deps: false });
} catch (e) {
  console.error('INSTALL FAILED: ' + (e && e.message ? e.message : String(e)).slice(0, 3000));
  process.exit(1);
}

// Exactly what worker.js does, in the same order.
await pyodide.runPythonAsync(`
import os

os.environ["DISCOPT_LP_MILP_BACKEND"] = "rust"
`);

// The worker's leak guard, verbatim in effect.
await pyodide.runPythonAsync(`
import discopt.modeling as dm  # noqa: F401
import sys

_leaked = sorted({n.split(".")[0] for n in sys.modules} & {"jax", "jaxlib", "highspy"})
if _leaked:
    raise ImportError("leaked: " + ", ".join(_leaked))
print("discopt imported; no jax/jaxlib/highspy in sys.modules")
`);

let ran = 0;
const failures = [];
for (const ex of examples) {
  ran += 1;
  const started = Date.now();
  try {
    pyodide.runPython(`
import io, sys
_buf = io.StringIO()
_stdout = sys.stdout
sys.stdout = _buf
`);
    await pyodide.runPythonAsync(ex.code);
    const printed = pyodide.runPython(`
sys.stdout = _stdout
_buf.getvalue()
`);
    const ms = Date.now() - started;
    if (!printed.toLowerCase().includes('optimal')) {
      failures.push([ex.id, 'no optimal status']);
      console.log(`  ${ex.id.padEnd(6)} FAIL  printed no optimal status`);
    } else {
      console.log(`  ${ex.id.padEnd(6)} ok    ${ms} ms`);
    }
  } catch (err) {
    pyodide.runPython('import sys; sys.stdout = _stdout');
    const line = String(err).trim().split('\n').filter(Boolean).pop();
    failures.push([ex.id, line]);
    console.log(`  ${ex.id.padEnd(6)} FAIL  ${line}`);
  }
}

console.log(`\nEXAMPLES_RUN=${ran} FAILURES=${failures.length}`);
if (ran !== 6) fail('probe did not run six examples');
process.exit(failures.length ? 1 : 0);
