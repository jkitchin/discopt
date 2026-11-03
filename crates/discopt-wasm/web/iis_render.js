// Draw an IIS: a summary, a table, and a constraint--variable graph.
//
// The input is the list of dicts `iis_view.py` builds, one per IISResult the
// script left in a variable. Nothing here knows about discopt's expressions;
// everything it shows is in that dict.
//
// The table is the primary view and holds everything the graph shows, as text:
// every member, the variables it touches, and where it is in the script. The
// graph is the same links drawn as a bipartite picture, members on the left,
// variables on the right. It is an image to assistive technology (role="img",
// with a label that points at the table), and nothing in it needs a pointer:
// focusing a row's "line" button highlights that member in the graph too.
//
// Color is never the only cue. IIS members are drawn with a solid, heavy border
// and listed first; constraints outside the IIS have a dashed border, muted
// text, a dashed edge and their own labelled block in the table and the graph.

export const SCHEMA = 1;

const SVG_NS = 'http://www.w3.org/2000/svg';

// --- finding a member in the script ----------------------------------------

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/**
 * 1-based line of the script that declares `member`, or null.
 *
 * A constraint is found by its `name="..."` argument, a bound by the
 * `m.continuous("x", ...)` call that declares its variable. A name built at
 * run time (an f-string in a loop) has no literal to find, so it gets null,
 * and the table says so rather than guessing a line.
 */
export function locate(source, member) {
  if (!source || !member) return null;
  let re;
  if (member.kind === 'bound') {
    const v = escapeRe(member.variable);
    re = new RegExp(`\\.(?:continuous|integer|binary)\\(\\s*(["'])${v}\\1`);
  } else {
    if (member.named === false) return null;
    re = new RegExp(`\\bname\\s*=\\s*(["'])${escapeRe(member.name)}\\1`);
  }
  const lines = source.split('\n');
  for (let i = 0; i < lines.length; i += 1) {
    if (re.test(lines[i])) return i + 1;
  }
  return null;
}

// --- text ------------------------------------------------------------------

const pretty = (s) => s.replace(/<=/g, '≤').replace(/>=/g, '≥');
const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

function kindLabel(member) {
  if (member.kind === 'bound') return member.side === 'lower' ? 'lower bound' : 'upper bound';
  return member.kind === 'context' ? 'constraint, not in the IIS' : 'constraint';
}

/** One-paragraph summary of a payload, as plain text. */
export function summarize(p) {
  const nCon = p.members.filter((m) => m.kind === 'constraint').length;
  const nBnd = p.members.length - nCon;
  const parts = [];
  const split = [nCon && plural(nCon, 'constraint'), nBnd && plural(nBnd, 'bound')]
    .filter(Boolean)
    .join(' and ');
  parts.push(
    `${plural(p.members.length, 'member')} (${split}) over ` +
      `${plural(p.variables.length, 'variable')}.`,
  );
  if (p.model) {
    parts.push(
      `The model has ${plural(p.model.n_constraints, 'constraint')} and ` +
        `${plural(p.model.n_variables, 'variable')}.`,
    );
  }
  if (p.proven_irreducible) {
    parts.push(
      `Proven minimal: dropping any one member makes the rest feasible ` +
        `(${plural(p.n_solves, 'feasibility solve')}).`,
    );
  } else {
    parts.push(
      `Not proven minimal: some of the ${plural(p.n_solves, 'feasibility solve')} were ` +
        `inconclusive, so this subsystem is infeasible but may hold members it does not need.`,
    );
  }
  return parts.join(' ');
}

// --- DOM helpers -----------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else node.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children) if (c !== null && c !== undefined) node.append(c);
  return node;
}

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text !== undefined) node.textContent = text;
  return node;
}

// --- table -----------------------------------------------------------------

function table(p, source, highlight, onLocate, uid) {
  const caption = el('caption', {
    text: 'Members of the IIS, the variables each one touches, and the line of the ' +
      'script that declares it',
  });
  const head = el(
    'thead',
    {},
    el(
      'tr',
      {},
      el('th', { scope: 'col', text: 'Member' }),
      el('th', { scope: 'col', text: 'Kind' }),
      el('th', { scope: 'col', text: 'Variables' }),
      el('th', { scope: 'col', text: 'In the script' }),
    ),
  );

  const row = (m) => {
    const line = locate(source, m);
    let where;
    if (line === null) {
      let why = 'not found';
      if (m.kind !== 'bound') why = m.named === false ? 'unnamed' : 'name built at run time';
      where = el('span', { class: 'iis-muted', text: why });
    } else {
      const button = el('button', {
        type: 'button',
        class: 'iis-line',
        text: `line ${line}`,
        'aria-label': `Show ${m.name} at line ${line} in the editor`,
      });
      button.addEventListener('click', () => onLocate(m, line));
      button.addEventListener('focus', () => highlight(m.id));
      button.addEventListener('blur', () => highlight(null));
      where = button;
    }
    const tr = el(
      'tr',
      { class: m.kind === 'context' ? 'iis-context' : 'iis-member', 'data-node': m.id },
      el('th', { scope: 'row' }, el('code', { text: pretty(m.name) })),
      el('td', { text: kindLabel(m) }),
      el('td', {}, el('code', { text: m.vars.join(', ') || '—' })),
      el('td', {}, where),
    );
    tr.addEventListener('pointerenter', () => highlight(m.id));
    tr.addEventListener('pointerleave', () => highlight(null));
    return tr;
  };

  const body = el('tbody', {}, ...p.members.map(row));
  const parts = [caption, head, body];
  if (p.context.length) {
    const shown =
      p.context_total > p.context.length
        ? `${p.context.length} of ${p.context_total} shown`
        : `${plural(p.context_total, 'constraint')}`;
    parts.push(
      el(
        'tbody',
        { class: 'iis-context-block' },
        el(
          'tr',
          {},
          el('th', {
            scope: 'rowgroup',
            colspan: 4,
            id: `${uid}-ctx`,
            text: `Not in the IIS, but sharing a variable with it (${shown})`,
          }),
        ),
        ...p.context.map(row),
      ),
    );
  }
  return el('div', { class: 'iis-table-wrap' }, el('table', { class: 'iis-table' }, ...parts));
}

// --- graph -----------------------------------------------------------------

const ROW = 28; // node height
const GAP = 8; // vertical gap between nodes
const PITCH = ROW + GAP;
const CHAR = 7.4; // ~width of one 12px monospace glyph
const PAD = 14; // horizontal padding inside a node
const MAX_CHARS = 26; // longer labels are cut, with the full name in a <title>
const COL_GAP = 150; // space between the columns, where the edges run
const HEADER = 26; // room for the column headings
const DIVIDER = 26; // room for the "not in the IIS" heading in the left column

const clip = (s) => (s.length > MAX_CHARS ? `${s.slice(0, MAX_CHARS - 1)}…` : s);
const widthOf = (labels) =>
  Math.max(80, ...labels.map((s) => clip(s).length * CHAR + 2 * PAD));

/**
 * The SVG graph, plus an `edges` map the highlighter uses. Pure layout: every
 * coordinate comes from the payload's order, so the same payload always draws
 * the same picture.
 */
function graph(p, uid) {
  const left = [...p.members, ...p.context];
  const leftLabels = left.map((m) => pretty(m.name));
  const rightLabels = p.variables.map((v) => v.id + (v.whole ? ' (all)' : ''));
  const wl = widthOf(leftLabels);
  const wr = widthOf(rightLabels);
  const xr = wl + COL_GAP;
  const width = xr + wr + 2;

  // Left column: members, then (after a labelled gap) the context block.
  const yLeft = left.map((m, i) =>
    HEADER + i * PITCH + (m.kind === 'context' ? DIVIDER : 0),
  );
  const leftHeight = left.length
    ? yLeft[yLeft.length - 1] + ROW
    : HEADER;
  const rightSpan = p.variables.length * PITCH - GAP;
  // The variable column is centred against the member block it mostly serves.
  const memberSpan = p.members.length * PITCH - GAP;
  const yr0 = HEADER + Math.max(0, (memberSpan - rightSpan) / 2);
  const yRight = p.variables.map((_, i) => yr0 + i * PITCH);
  const height = Math.max(leftHeight, yr0 + rightSpan) + 4;

  const root = svg('svg', {
    class: 'iis-graph',
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: 'img',
    'aria-labelledby': `${uid}-gtitle`,
  });
  root.append(
    svg(
      'title',
      { id: `${uid}-gtitle` },
      `Constraint–variable graph: ${plural(p.members.length, 'IIS member')} on the left, ` +
        `linked to the ${plural(p.variables.length, 'variable')} they touch on the right` +
        (p.context.length
          ? `, with ${plural(p.context.length, 'other constraint')} that share those variables, dashed.`
          : '.') +
        ' The table above lists the same links as text.',
    ),
  );

  root.append(
    svg('text', { x: 0, y: 14, class: 'iis-colhead' }, 'IIS members'),
    svg('text', { x: xr, y: 14, class: 'iis-colhead' }, 'variables'),
  );
  if (p.context.length) {
    const y = yLeft[p.members.length] - 9;
    root.append(svg('text', { x: 0, y, class: 'iis-colhead' }, 'not in the IIS'));
  }

  const varIndex = new Map(p.variables.map((v, i) => [v.id, i]));
  const edges = new Map(); // node id -> [edge elements]
  const neighbours = new Map(); // node id -> Set of node ids
  const touch = (a, b, e) => {
    for (const [x, y] of [[a, b], [b, a]]) {
      if (!edges.has(x)) edges.set(x, []);
      edges.get(x).push(e);
      if (!neighbours.has(x)) neighbours.set(x, new Set());
      neighbours.get(x).add(y);
    }
  };

  // Edges first, so nodes paint over their ends.
  const edgeLayer = svg('g', { class: 'iis-edges' });
  root.append(edgeLayer);
  left.forEach((m, i) => {
    const y1 = yLeft[i] + ROW / 2;
    for (const vid of m.vars) {
      const j = varIndex.get(vid);
      if (j === undefined) continue;
      const y2 = yRight[j] + ROW / 2;
      const mid = wl + COL_GAP / 2;
      const path = svg('path', {
        d: `M${wl},${y1} C${mid},${y1} ${mid},${y2} ${xr},${y2}`,
        class: m.kind === 'context' ? 'iis-edge iis-edge-context' : 'iis-edge',
      });
      edgeLayer.append(path);
      touch(m.id, `v:${vid}`, path);
    }
  });

  const nodes = new Map(); // node id -> <g>
  const node = (id, x, y, w, label, cls) => {
    const g = svg('g', { class: `iis-node ${cls}`, 'data-node': id });
    g.append(
      svg('rect', { x: x + 1, y, width: w - 2, height: ROW, rx: cls.includes('iis-var') ? 14 : 4 }),
      svg('text', { x: x + PAD, y: y + ROW / 2 + 4 }, clip(label)),
    );
    if (clip(label) !== label) g.append(svg('title', {}, label));
    nodes.set(id, g);
    root.append(g);
    return g;
  };
  left.forEach((m, i) => {
    const cls = m.kind === 'context' ? 'iis-ctx' : m.kind === 'bound' ? 'iis-bound' : 'iis-con';
    node(m.id, 0, yLeft[i], wl, leftLabels[i], cls);
  });
  p.variables.forEach((v, j) => node(`v:${v.id}`, xr, yRight[j], wr, rightLabels[j], 'iis-var'));

  return { root, nodes, edges, neighbours };
}

// --- putting it together ---------------------------------------------------

/**
 * Render `payloads` into `container`, replacing what was there.
 *
 * `source` is the script that produced them, for the "line" links.
 * `onLocate(member, line)` is called when one is pressed.
 * Returns the number of IIS results drawn.
 */
export function renderIIS(container, payloads, { source = '', onLocate = () => {} } = {}) {
  container.replaceChildren();
  if (!payloads || !payloads.length) return 0;

  payloads.forEach((p, k) => {
    const uid = `iis${k}`;
    const block = el('div', { class: 'iis-block' });
    if (p.schema !== SCHEMA) {
      block.append(
        el('p', {
          class: 'iis-muted',
          text: `This IIS was described in format ${p.schema}; the page draws format ${SCHEMA}.`,
        }),
      );
      container.append(block);
      return;
    }
    if (payloads.length > 1 || p.label) {
      block.append(el('h3', {}, 'IIS ', el('code', { text: p.label || String(k + 1) })));
    }
    block.append(el('p', { class: 'iis-summary', text: summarize(p) }));
    if (!p.model) {
      block.append(
        el('p', {
          class: 'iis-muted',
          text: 'The model it came from is not in a top-level variable, so constraints ' +
            'outside the IIS are not shown.',
        }),
      );
    }

    const g = graph(p, uid);
    const highlight = (id) => {
      const on = id === null ? null : new Set([id, ...(g.neighbours.get(id) || [])]);
      g.root.classList.toggle('iis-focus', on !== null);
      for (const [nid, n] of g.nodes) n.classList.toggle('iis-hl', on !== null && on.has(nid));
      for (const list of g.edges.values()) for (const e of list) e.classList.remove('iis-hl');
      if (on !== null) for (const e of g.edges.get(id) || []) e.classList.add('iis-hl');
      for (const tr of block.querySelectorAll('tr[data-node]')) {
        tr.classList.toggle('iis-hl', on !== null && tr.dataset.node === id);
      }
    };
    for (const [nid, n] of g.nodes) {
      n.addEventListener('pointerenter', () => highlight(nid));
      n.addEventListener('pointerleave', () => highlight(null));
    }

    block.append(table(p, source, highlight, onLocate, uid));
    block.append(
      el(
        'figure',
        { class: 'iis-figure' },
        el('div', { class: 'iis-graph-wrap' }, g.root),
        el('figcaption', {
          text: 'Solid boxes are the IIS: constraints, and bounds on a single variable. ' +
            'Dashed boxes are other constraints that share a variable with it; they are ' +
            'not part of the conflict, and their links to variables outside the IIS are ' +
            'not drawn. Point at a box or a table row, or focus a row\'s line button, to ' +
            'highlight its links.',
        }),
      ),
    );
    container.append(block);
  });
  return payloads.length;
}
