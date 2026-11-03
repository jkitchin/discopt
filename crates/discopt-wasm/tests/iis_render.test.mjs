// Unit tests for the pure parts of web/iis_render.js: finding a member's line
// in the script, and the summary sentence. Run with `node --test`.
// The DOM half is exercised in a real browser by iis_page.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { locate, summarize } from '../web/iis_render.js';

const SCRIPT = [
  'import discopt.modeling as dm',
  'm = dm.Model("t")',
  'a = m.continuous("a", shape=(3,), lb=0, ub=40)',
  "b = m.integer('b', lb=0, ub=5)",
  'm.subject_to(a[0] + b >= 1, name="need")',
  "m.subject_to(a[1] <= 2, name='cap.1')",
  'm.subject_to(a[2] <= 3, name="need_more")',
  'for i in range(3):',
  '    m.subject_to(a[i] >= 0, name=f"pos{i}")',
].join('\n');

test('a constraint is found by its literal name, in either quote style', () => {
  assert.equal(locate(SCRIPT, { kind: 'constraint', name: 'need', named: true }), 5);
  assert.equal(locate(SCRIPT, { kind: 'constraint', name: 'cap.1', named: true }), 6);
  assert.equal(locate(SCRIPT, { kind: 'constraint', name: 'need_more', named: true }), 7);
});

test('regex characters in a name are literal', () => {
  // "cap.1" must not match "capX1"; "need" must not match "need_more" first.
  assert.equal(locate('x = 1\nm.subject_to(y, name="capX1")', { kind: 'constraint', name: 'cap.1' }), null);
  assert.equal(locate(SCRIPT, { kind: 'constraint', name: 'need' }), 5);
});

test('a name built at run time, or no name, finds nothing rather than guessing', () => {
  assert.equal(locate(SCRIPT, { kind: 'constraint', name: 'pos1', named: true }), null);
  assert.equal(locate(SCRIPT, { kind: 'constraint', name: 'constraint 0', named: false }), null);
});

test('a bound is found at the declaration of its variable', () => {
  assert.equal(locate(SCRIPT, { kind: 'bound', variable: 'a', name: 'a[1] <= 40' }), 3);
  assert.equal(locate(SCRIPT, { kind: 'bound', variable: 'b', name: 'b <= 5' }), 4);
  assert.equal(locate(SCRIPT, { kind: 'bound', variable: 'zz', name: 'zz <= 1' }), null);
  assert.equal(locate('', { kind: 'bound', variable: 'a' }), null);
});

const payload = (over = {}) => ({
  members: [
    { kind: 'constraint' }, { kind: 'constraint' }, { kind: 'bound' },
  ],
  variables: [{}, {}],
  model: { n_constraints: 9, n_variables: 1 },
  proven_irreducible: true,
  n_solves: 1,
  ...over,
});

test('the summary counts members, kinds and variables', () => {
  const s = summarize(payload());
  assert.match(s, /^3 members \(2 constraints and 1 bound\) over 2 variables\./);
  assert.match(s, /The model has 9 constraints and 1 variable\./);
  assert.match(s, /Proven minimal: .* \(1 feasibility solve\)\./);
});

test('the summary says so when the IIS is not proven minimal, and when there is no model', () => {
  const s = summarize(payload({ proven_irreducible: false, model: null, n_solves: 4 }));
  assert.match(s, /Not proven minimal: some of the 4 feasibility solves were inconclusive/);
  assert.doesNotMatch(s, /The model has/);
});
