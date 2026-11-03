// End-to-end test of the page's IIS view, in a real browser.
//
// examples_in_pyodide.mjs proves the wheels solve and that iis_view.py builds
// its dict under Pyodide; it never touches the DOM. This drives the page itself
// in headless Chromium: the worker, the IIS table and graph, the line links
// into the editor, axe (WCAG 2.2 AA) on the IIS section in both themes, and
// reflow at 320 px.
//
// The script serves `web/` and the npm copy of Pyodide from one local origin,
// and points the page at that Pyodide with `?pyodide=`. The npm package does
// not ship numpy or scipy: Node's Pyodide downloads them from its CDN on first
// use and caches them in `node_modules/pyodide`. So the script loads them in
// Node first -- a download on a fresh install, a no-op after -- and the page
// then finds them on the local server.
//
//   cd crates/discopt-wasm/tests && npm install
//   node iis_page.mjs
//
// Playwright and axe-core are pinned dev dependencies of this package. On a
// fresh machine, `npx playwright install chromium` fetches the browser that
// version expects.
//
// Every check is counted, and the script exits non-zero if any failed or if
// none ran.
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, extname, join, normalize } from 'node:path';
import { chromium } from 'playwright';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = join(HERE, '..', 'web');
const PYODIDE = join(HERE, 'node_modules', 'pyodide');
const require = createRequire(import.meta.url);

const AXE = readFileSync(require.resolve('axe-core/axe.min.js'), 'utf8');

{
  const { loadPyodide } = await import('pyodide');
  const py = await loadPyodide();
  await py.loadPackage(['numpy', 'scipy', 'micropip']);
}

// --- a static server for web/ and Pyodide ---------------------------------

const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.py': 'text/x-python', '.wasm': 'application/wasm',
  '.whl': 'application/zip', '.zip': 'application/zip', '.map': 'application/json',
};

const server = createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const [root, rel] = path.startsWith('/pyodide/')
    ? [PYODIDE, path.slice('/pyodide/'.length)]
    : [WEB, path === '/' ? 'index.html' : path.slice(1)];
  const file = normalize(join(root, rel));
  if (!file.startsWith(root)) {
    res.writeHead(403).end();
    return;
  }
  try {
    if (!(await stat(file)).isFile()) throw new Error('not a file');
    res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream' });
    res.end(await readFile(file));
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const BASE = `http://127.0.0.1:${server.address().port}/`;

// --- checks ----------------------------------------------------------------

let checks = 0;
let failed = 0;
const check = (cond, msg) => {
  checks += 1;
  if (!cond) failed += 1;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${msg}`);
};

// Click Run and wait for *this* run to finish. The status passes through
// "solving" first, so the previous run's "done in" cannot satisfy the wait.
async function run(page) {
  await page.click('#run');
  for (const re of [/solving/, /done in|raised/]) {
    await page.waitForFunction(
      (src) => new RegExp(src).test(document.getElementById('status').textContent),
      re.source,
      { timeout: 240000 },
    );
  }
}

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));

  await page.goto(`${BASE}?pyodide=${encodeURIComponent(`${BASE}pyodide/`)}`);
  await page.waitForFunction(() => !document.getElementById('run').disabled, null, { timeout: 120000 });
  check(await page.isHidden('#iis-section'), 'IIS section hidden before any run');

  await page.selectOption('#example', 'LP');
  await run(page);
  check(/done in/.test(await page.textContent('#status')), 'LP example ran');
  check(await page.isHidden('#iis-section'), 'a feasible model draws no IIS');

  await page.selectOption('#example', 'IIS');
  await run(page);
  check(/infeasible/.test(await page.textContent('#out')), 'IIS example printed "infeasible"');
  check(await page.isVisible('#iis-section'), 'IIS section shown');
  check(/IIS shown below/.test(await page.textContent('#status')), 'status line points at the IIS');
  check(/\d+ members in the IIS/.test(await page.textContent('#iis-note')), 'live region announces it');

  // The table and the graph must agree with each other and with the summary.
  const summary = await page.textContent('.iis-summary');
  const nMembers = Number(/^(\d+) members/.exec(summary)?.[1]);
  const nVars = Number(/over (\d+) variables/.exec(summary)?.[1]);
  const memberRows = await page.$$('tr.iis-member');
  const contextRows = await page.$$('tr.iis-context');
  check(nMembers > 0 && memberRows.length === nMembers, `table has the ${nMembers} members`);
  const nodes = await page.$$eval('svg.iis-graph .iis-node', (g) => g.length);
  check(nodes === nMembers + contextRows.length + nVars, `graph has ${nodes} nodes`);
  const rowLinks = await page.$$eval('tr[data-node] td:nth-child(3) code', (c) =>
    c.reduce((n, x) => n + (x.textContent === '—' ? 0 : x.textContent.split(', ').length), 0));
  const edges = await page.$$eval('svg.iis-graph .iis-edge', (e) => e.length);
  check(edges === rowLinks, `graph draws ${edges} edges, one per variable the table lists`);

  // Focusing a row's line button highlights that member and its variables.
  const soy = page.locator('tr', { hasText: 'soy_supply' }).locator('button.iis-line');
  await soy.focus();
  const lit = await page.$$eval('svg .iis-node.iis-hl text', (t) => t.map((x) => x.textContent).sort());
  check(JSON.stringify(lit) === JSON.stringify(['a[1]', 'b[1]', 'soy_supply']), `focus lights ${lit}`);
  check((await page.$$('svg .iis-edge.iis-hl')).length === 2, 'and exactly its two edges');

  // Pressing it selects the declaring line in the editor.
  await soy.press('Enter');
  const selection = await page.evaluate(() => String(window.getSelection()));
  check(/name="soy_supply"/.test(selection), `editor selects: ${selection.trim()}`);
  const inEditor = await page.evaluate(() =>
    document.querySelector('.cm-content').contains(document.activeElement));
  check(inEditor, 'focus moves to the editor');

  for (const theme of ['light', 'dark']) {
    await page.selectOption('#theme', theme);
    await page.addScriptTag({ content: AXE });
    const violations = await page.evaluate(async () =>
      (await window.axe.run(document.getElementById('iis-section'), {
        runOnly: ['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'],
      })).violations.map((v) => `${v.id} (${v.nodes.length})`));
    check(violations.length === 0, `axe, ${theme} theme: ${violations.join(', ') || 'clean'}`);
  }

  await page.setViewportSize({ width: 320, height: 800 });
  const overflow = await page.evaluate(() =>
    document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check(overflow <= 0, `no horizontal page scroll at 320 px (${overflow} px)`);
  await page.setViewportSize({ width: 1100, height: 900 });

  // The next run starts from a fresh namespace, so the old IIS is not redrawn.
  await page.selectOption('#example', 'QP');
  await run(page);
  check(await page.isHidden('#iis-section'), 'running a feasible model clears the IIS');

  check(errors.length === 0, `no page errors${errors.length ? `: ${errors.join(' | ')}` : ''}`);
} finally {
  await browser.close();
  server.close();
}

console.log(`\nCHECKS=${checks} FAILED=${failed}`);
process.exit(checks === 0 || failed ? 1 : 0);
