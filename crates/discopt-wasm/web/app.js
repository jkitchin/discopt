// Page logic: pick an example, run it in the Pyodide worker, show what it
// printed. The worker owns CPython and the solver; this side is a dropdown, an
// editor and a console.

import { EXAMPLES } from './examples.js';
import { attachEditor } from './vendor/codemirror.bundle.js';
import { locate, renderIIS } from './iis_render.js';

const $ = (id) => document.getElementById(id);
const codeBox = $('code');
const outBox = $('out');
const runButton = $('run');
const cancelButton = $('cancel');
const statusLine = $('status');
const titleLine = $('example-title');
const iisSection = $('iis-section');
const iisBox = $('iis');
const iisNote = $('iis-note');

// --- editor ----------------------------------------------------------------

const select = $('example');
for (const example of EXAMPLES) {
  const option = document.createElement('option');
  option.value = example.id;
  option.textContent = example.label;
  select.append(option);
}

const byId = (id) => EXAMPLES.find((e) => e.id === id) ?? EXAMPLES[0];

// CodeMirror wraps the textarea rather than replacing it, so `codeBox.value`
// stays the single source of truth and the page still works as a plain
// textarea if the bundle fails to load.
const editor = attachEditor(codeBox, { onRun: () => run() });

function showExample(id) {
  const example = byId(id);
  editor.set(example.code);
  titleLine.textContent = example.title;
}

select.addEventListener('change', () => showExample(select.value));
showExample(select.value);

// --- worker ----------------------------------------------------------------

// A solve runs synchronously inside the worker, so nothing short of killing the
// thread can interrupt one -- Cancel terminates the worker and starts a fresh
// one. That is a real reset: the replacement re-loads Pyodide (from cache) and
// re-installs discopt before the next run.
let worker = null;
let ready = false;

function spawnWorker() {
  if (worker) worker.terminate();
  ready = false;
  runButton.disabled = true;
  cancelButton.disabled = true;
  // The worker inherits this page's query string, so ?pyodide= / ?discopt= /
  // ?pounce= overrides for a self-hosted deployment reach it.
  worker = new Worker(`./worker.js${self.location.search}`, { type: 'module' });
  worker.onmessage = onWorkerMessage;
}

function onWorkerMessage({ data }) {
  if (data.type === 'status') {
    setStatus(data.text);
  } else if (data.type === 'ready') {
    ready = true;
    runButton.disabled = false;
    // discopt is not installed yet -- the first Run pulls the wheels, and says
    // so on this line while it does.
    setStatus('ready — Python is loaded');
  } else if (data.type === 'stdout') {
    append(data.text);
  } else if (data.type === 'running') {
    outBox.textContent = '';
    clearIIS();
    cancelButton.disabled = false;
    setStatus('solving…');
  } else if (data.type === 'iis') {
    showIIS(data.items);
  } else if (data.type === 'iis-error') {
    iisSection.hidden = false;
    iisNote.textContent = `The page could not draw this IIS: ${data.message}`;
  } else if (data.type === 'done') {
    runButton.disabled = false;
    cancelButton.disabled = true;
    const drawn = iisBox.childElementCount ? ' — IIS shown below the output' : '';
    setStatus(`done in ${(data.ms / 1000).toFixed(2)} s${drawn}`);
  } else if (data.type === 'error') {
    runButton.disabled = false;
    cancelButton.disabled = true;
    append(`\n${data.message}\n`);
    setStatus('the script raised an exception', true);
  } else if (data.type === 'fatal') {
    runButton.disabled = true;
    cancelButton.disabled = true;
    setStatus(data.message, true);
  }
}

spawnWorker();

function setStatus(text, isError = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle('err', isError);
}

function append(text) {
  outBox.textContent += text;
  outBox.scrollTop = outBox.scrollHeight;
}

// The script as it was when it ran: the IIS table's line numbers are computed
// against it.
let ranCode = '';

function run() {
  if (!ready) return;
  runButton.disabled = true;
  ranCode = codeBox.value;
  worker.postMessage({ type: 'run', code: ranCode });
}

// --- IIS -------------------------------------------------------------------

function clearIIS() {
  iisBox.replaceChildren();
  iisNote.textContent = '';
  iisSection.hidden = true;
}

// Select a member's line in the editor. If the script was edited after it ran,
// the line is looked up again in the current text, so the link follows the
// declaration rather than pointing at whatever moved into its old place.
function showInEditor(member, line) {
  const current = codeBox.value;
  const target = current === ranCode ? line : locate(current, member);
  if (target === null) {
    setStatus(`${member.name} is no longer in the script`, true);
    return;
  }
  const view = editor.view;
  if (!view) {
    codeBox.focus();
    return;
  }
  const at = view.state.doc.line(target);
  view.dispatch({ selection: { anchor: at.from, head: at.to }, scrollIntoView: true });
  view.focus();
  setStatus(`editor: line ${target}, ${member.name}`);
}

function showIIS(items) {
  const n = renderIIS(iisBox, items, { source: ranCode, onLocate: showInEditor });
  iisSection.hidden = n === 0;
  // The summary of the first one is announced; the section holds the rest.
  iisNote.textContent = n
    ? `Infeasibility explained: ${items[0].members.length} members in the IIS` +
      (n > 1 ? `, and ${n - 1} more IIS below.` : '.')
    : '';
}

runButton.addEventListener('click', run);
cancelButton.addEventListener('click', () => {
  append('\n^C  cancelled — restarting Python\n');
  setStatus('cancelled — reloading Python…');
  spawnWorker();
});

$('dl-code').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([codeBox.value], { type: 'text/x-python' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `${select.value.toLowerCase()}_model.py`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
});

// --- theme -----------------------------------------------------------------

// The inline script in index.html has already resolved "auto" into an explicit
// data-theme so the first paint is right; this re-resolves it on a change to
// the OS setting, which only matters while the preference *is* "auto". The
// editor needs telling separately -- CodeMirror carries its own palette, which
// no amount of page CSS reaches.
const themeSelect = $('theme');
const systemDark = matchMedia('(prefers-color-scheme: dark)');

function readTheme() {
  try {
    return localStorage.getItem('discopt-theme') || 'auto';
  } catch {
    return 'auto';
  }
}

function applyTheme(pref) {
  const dark = pref === 'dark' || (pref === 'auto' && systemDark.matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  editor.setDark(dark);
}

themeSelect.value = readTheme();
applyTheme(themeSelect.value);
themeSelect.addEventListener('change', () => {
  try {
    localStorage.setItem('discopt-theme', themeSelect.value);
  } catch {
    // A blocked store is not a reason to refuse the change for this page view.
  }
  applyTheme(themeSelect.value);
});
systemDark.addEventListener('change', () => applyTheme(readTheme()));
