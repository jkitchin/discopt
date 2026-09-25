// Bundle entry: everything the page needs from CodeMirror 6, and nothing else.
//
// The export is deliberately the same `attachEditor(textarea) -> {set, get}`
// contract POUNCE's dependency-free editor.js offers, so the page wires up the
// same way and the textarea stays the source of truth.

import { EditorView, basicSetup } from 'codemirror';
import { EditorState, Compartment } from '@codemirror/state';
import { keymap } from '@codemirror/view';
import { indentWithTab } from '@codemirror/commands';
import { python } from '@codemirror/lang-python';
import { oneDark } from '@codemirror/theme-one-dark';

export function attachEditor(textarea, { onRun } = {}) {
  const theme = new Compartment();

  // Tab indents instead of leaving the editor. That traps keyboard users, so
  // Escape-then-Tab still moves focus out -- CodeMirror's default behaviour
  // once indentWithTab is bound.
  const runKey = onRun
    ? [{ key: 'Mod-Enter', preventDefault: true, run: () => (onRun(), true) }]
    : [];

  const view = new EditorView({
    state: EditorState.create({
      doc: textarea.value,
      extensions: [
        basicSetup,
        python(),
        keymap.of([...runKey, indentWithTab]),
        theme.of([]),
        EditorView.updateListener.of((update) => {
          // Mirror into the textarea on every change so `textarea.value` is
          // always current: the run path reads it, and a page whose bundle
          // failed to load still has a working plain textarea.
          if (update.docChanged) textarea.value = view.state.doc.toString();
        }),
      ],
    }),
  });

  textarea.parentNode.insertBefore(view.dom, textarea);
  textarea.style.display = 'none';

  const set = (code) => {
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: code } });
    textarea.value = code;
  };

  const setDark = (on) => view.dispatch({ effects: theme.reconfigure(on ? oneDark : []) });

  return { set, get: () => view.state.doc.toString(), setDark, view };
}
