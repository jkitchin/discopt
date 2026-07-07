# LLM Module Review — Security, Correctness, and the Safety Invariant

**Date:** 2026-07-03
**Scope:** `python/discopt/llm/` (`tools.py`, `chat.py`, `reformulation.py`,
`advisor.py`, `diagnosis.py`, `commentary.py`, `prompts.py`, `serializer.py`,
`provider.py`, `safety.py`, `__init__.py` — 3,063 lines) and `test_llm_modules.py`.
**Method:** Full read of the safety-critical files (`tools.py`, `safety.py`,
`advisor.py`, `provider.py`, `chat.py`, `reformulation.py`); the security finding
reproduced end-to-end (working RCE). Baseline: **104 tests passed** (0.4 s).

CLAUDE.md states the module's invariant verbatim: *"Safety invariant: LLM outputs
never affect solver math. Formulations pass validate(). Explanations are sanitized.
Graceful degradation when litellm is unavailable."* This review's central question
is whether that invariant holds. **It does not** — and the breach is worse than a
math bug: it is a remote-code-execution vulnerability.

---

## 1. Summary of findings

> **✅ RESOLVED LLM-1** — `python/tests/test_llm_eval_safety.py` (this PR). The
> `eval()` path is replaced by a whitelisted-AST evaluator (`_safe_eval_node` in
> `tools.py`): only arithmetic, indexing, and calls to a fixed set of `dm` math /
> reduction functions are permitted; `Attribute`, `Lambda`, comprehensions, starred
> args, and calls to non-whitelisted callables are rejected — closing the escape
> class. RCE repro (`os.system` via `np.__loader__…__globals__`) now raises
> `ValueError` and runs no code; verified fails-before/passes-after (9 escape tests
> fail on the pre-fix `eval`). `sum`/`abs` now resolve to the modeling reductions,
> fixing a latent hang where builtin `sum(var)` iterated a Variable forever. The
> full finding text is preserved below.

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| LLM-1 | **P0 SECURITY (RCE)** — ✅ RESOLVED | `tools.py:546-566` | `ModelBuilder._eval_expression` runs `eval(expr_str, {"__builtins__": {}, ...np, dm...})` — the classic incomplete sandbox. With `np` in scope it is trivially escaped to **arbitrary code execution** (`os.system` confirmed). Reachable through the public `discopt.chat()` and `discopt.from_description()`, i.e. any LLM (or a prompt-injected one) can run shell commands on the user's machine. Docstring falsely claims "No builtins or arbitrary code execution" [CONFIRMED with working RCE] |
| LLM-2 | **P1 false-assurance** | `safety.py:92-146` | `sanitize_tool_args` sanitizes only the `name` field; the **eval'd fields** (`lhs`, `rhs`, `expression`, `value`) pass through raw. The function's presence in the pipeline reads as "inputs are sanitized" while the injection surface is untouched [CONFIRMED] |
| LLM-3 | P1 invariant | `advisor.py:51-59, 259-302` | `suggest_solver_params(llm=True)` does `params.update(llm_suggestion)` on **unvalidated LLM JSON** — no key whitelist despite the prompt naming one. An LLM-returned `gap_tolerance`/`nlp_solver`/`time_limit` flows straight into the returned params dict; if the user forwards it to `solve(**params)` (the documented use), LLM output *has* affected the solve. Violates the invariant's spirit [CONFIRMED mechanism] |
| LLM-4 | P2 dead capability | `reformulation.py:85-121` | `apply_bound_tightening` documents "Number of bounds tightened" and "modified in place" but the body only `logger.debug`s and **always returns 0** — a stub masquerading as a working, model-mutating function [CONFIRMED] |
| LLM-5 | P3 | `advisor.py:160-166` | `_analyze_structure` detects big-M / bilinear by **string-scanning `str(constraint)`** (`" * " in c_str`, `float(token) >= 1000`) — brittle heuristics that misfire on names containing digits or reprs with `*`. Advisory-only, so low stakes, but wrong advice erodes trust |

Checked and found **sound** (the invariant holds *here*):

- **Explanations are genuinely sanitized/bounded** (`validate_explanation`:
  truncation + strip; the diagnosis/commentary paths are read-only text over a
  serialized model).
- **Graceful degradation** is real and consistent: every entry point guards
  `import litellm` and raises an actionable `ImportError`; `is_available()` gates
  the optional paths; LLM-augmentation failures are caught and fall back to the
  rule-based result (`advisor`, `reformulation._llm_analyze`).
- **Reformulation suggestions are advisory** — `analyze_reformulations` returns
  `ReformulationSuggestion` objects; nothing auto-rewrites the model from LLM
  output. `apply_bound_tightening` is the only mutator and it is deterministic
  (and, per LLM-4, a no-op).
- **The tool layer maps to real Model methods** (no free-form code-gen for model
  structure) and `execute_tool` catches handler exceptions into error strings —
  the *architecture* is right; LLM-1 is a single unsafe primitive inside it.
- `provider.py` is a clean litellm wrapper (deterministic `temperature=0`,
  timeouts, retry-to-RuntimeError); the OpenAI-format tool schemas are well-formed.

---

## 2. LLM-1: the RCE, in detail

`_eval_expression` builds a namespace with `__builtins__` emptied and the model's
variables + `np` + `dm` added, then `eval()`s the LLM-supplied string:

```python
safe_ns = {"__builtins__": {}, "sum": sum, "range": range, "abs": abs, ...}
safe_ns.update(self._namespace)          # includes np, dm, and every variable
return eval(expr_str, safe_ns)           # noqa: S307   ← the ruff hint was ignored
```

Emptying `__builtins__` is the textbook *insufficient* sandbox: any object in
scope is a bridge back to builtins via its type's method-resolution chain.
Reproduced, all through the public `execute_tool("add_constraint", …)` path:

```
().__class__.__bases__[0].__subclasses__()                    → object subclass list
np.__loader__.__init__.__globals__['__builtins__']['__import__']('os').system(...)
                                                              → arbitrary shell command
```

The last payload **wrote a file and ran `echo` via `os.system`** in this review.
`np` alone is sufficient (any imported module's `__loader__`/`__globals__` reaches
`__builtins__`); removing `np` would not fix it because `dm`, and indeed any
`Variable`/`Expression` object, exposes the same `.__class__.__…__` chain.

**Threat model.** The inputs to `eval` are LLM tool-call arguments. The attacker
is therefore anyone who can influence the LLM's output: a prompt-injected problem
description (`from_description(untrusted_text)`), a compromised/hostile model
endpoint (`DISCOPT_LLM_MODEL` pointing at an attacker's server), or simply a
capable model coaxed into emitting a crafted expression. The blast radius is the
user's machine (file read/write, network, process spawn) with the user's
privileges. This is the highest-severity class of finding in the entire review
series — not a wrong number, but code execution.

**Fix (do not try to harden `eval`).** Replace the string-expression path with a
**safe AST evaluator**: parse with `ast.parse(expr_str, mode="eval")`, walk the
tree with an explicit allowlist of node types (`BinOp`, `UnaryOp`, `Call` to a
whitelisted function set `{sum, exp, log, sqrt, sin, cos, …}`, `Subscript`,
`Name` resolving only to the model namespace, `Constant`, `Tuple`, `List`), and
**reject every other node** (`Attribute` — kills `.__class__`; `Lambda`;
comprehensions with calls; starred; etc.). Attribute access is the escape vector,
so a no-`Attribute` policy alone closes the class. Alternatively, drop
string-expressions entirely and have the tool schema pass **structured operand
trees** (the tools already enumerate the operations) so no parsing/eval is needed
at all — the more robust design. Either way: delete the `# noqa: S307` and the
false docstring claim. Regression tests: every §2 payload must raise; the legit
`x[0]**2 + 5*y` cases must still evaluate.

---

## 3. LLM-2 / LLM-3: assurance and invariant gaps

**LLM-2.** `sanitize_tool_args` clamps `add_continuous` bounds and regex-cleans
`name`, but the fields that reach `eval` (`lhs`/`rhs`/`expression`/`value`) are
never touched. Its position right before `execute_tool` gives a false sense that
tool inputs are validated. After LLM-1 is fixed by a safe evaluator this is moot
for RCE, but the function should either validate the expression fields (parse-check
them) or its docstring should stop implying it guards them.

**LLM-3.** The advisor's `params.update(llm_suggestion)` accepts any keys/values
the model returns. The prompt *says* "Valid keys: nlp_solver, partitions, …" but
nothing enforces it — an LLM can return `{"gap_tolerance": 1.0}` (or a bogus
`nlp_solver`) and the caller who does `model.solve(**suggest_solver_params(m,
llm=True))` then solves with LLM-chosen numerics. Per CLAUDE.md's invariant, LLM
output must not affect solver math; here it can. **Fix:** whitelist the keys and
range-validate each (`gap_tolerance ∈ [1e-9, 1e-1]`, `nlp_solver ∈ {known set}`,
`partitions ∈ [0, 64]`, …); drop unknown keys with a debug log. Cheap, and it
makes the invariant true for this path.

---

## 4. LLM-4 / LLM-5: honesty items

**LLM-4.** `apply_bound_tightening` is a no-op with a working-function docstring
and return contract. Either implement the DAG walk it describes (the comment
admits "full implementation would walk the expression DAG") or delete it and its
export — a stub that claims to tighten bounds is worse than absent, because a
caller trusts the returned count. (Note: real bound tightening already exists in
FBBT/`tightening.py`; this LLM-module duplicate should probably just be removed.)

**LLM-5.** `_analyze_structure`'s `str(constraint)` scanning is a fragile way to
detect structure the model already knows through its DAG (and through
`problem_classifier`, which the same function already calls for `problem_class`).
Advisory-only, so cosmetic — but it should reuse the structural analysis the
solver layer computes rather than re-deriving it from reprs.

---

## 5. Test coverage gaps

`test_llm_modules.py` (104 tests) covers the tool handlers, serialization,
rule-based advisor output, reformulation detection, and graceful degradation
well. The gap is **adversarial**: nothing tests that `_eval_expression` rejects
hostile input (the entire finding class), nothing asserts the sanitizer covers
the injection surface, and nothing pins the advisor's param-key whitelist. Add:

1. A **sandbox-escape test suite** — each §2 payload (and the canonical
   `().__class__…__subclasses__()`, `__import__`, attribute-access, lambda,
   comprehension forms) must raise; legit expressions must evaluate. This is the
   regression harness for LLM-1 and belongs in a security-marked test.
2. An advisor test asserting an out-of-whitelist LLM key is dropped and an
   out-of-range `gap_tolerance` is rejected (LLM-3).
3. A test that `apply_bound_tightening` either tightens a known-tightenable model
   or is removed (LLM-4).

---

## 6. Assessment

The LLM module's *architecture* is sound and its intent is right: tool-calling to
real Model methods (not code-gen), advisory-only reformulations, sanitized
explanations, clean optional-dependency degradation, and a stated invariant that
LLM output must not touch solver math. Most of that holds. But the single
`eval()` primitive at the bottom of the model-building path turns "the LLM writes
your model" into "the LLM runs code on your machine," which is both a
critical security vulnerability and a direct falsification of the module's own
"no arbitrary code execution" claim. That one line is the most urgent fix found
in this entire review series — ahead of the solver P0s, because its failure mode
is RCE rather than a wrong answer. The remaining items (unvalidated advisor
params, the sanitizer's false coverage, the no-op tightening stub) are the kind
of gaps that let LLM-1 read as safe for as long as it did.

---

## 7. Implementation plan (for Opus)

### Phase 1 — close the RCE (PR `fix(llm): LLM-1` — security, highest priority)

- Replace `_eval_expression`'s `eval()` with an AST-allowlist evaluator (no
  `Attribute` nodes; `Call` only to a whitelisted math-function set; `Name` only
  into the model namespace) **or** switch the tool schema to structured operand
  trees and remove parsing entirely. Delete `# noqa: S307` and the false docstring
  line.
- **Acceptance:** the §2 payloads (subclass walk, `__import__`, `os.system`,
  attribute access, lambda, comprehension-with-call) all raise `ValueError`;
  `x[0]**2 + 5*y`, `sum(x[i] for i in range(3))`-style legit expressions still
  evaluate; `chat()` / `from_description()` behavior on benign input unchanged.
  Security-marked regression test lands with the fix.

### Phase 2 — invariant + assurance (PR `fix(llm): LLM-2..LLM-3`)

- LLM-3: whitelist + range-validate advisor param keys before `params.update`.
- LLM-2: either validate the expression fields in `sanitize_tool_args` or correct
  its docstring/scope so it no longer implies coverage it lacks.
- Tests: advisor drops out-of-whitelist keys and rejects out-of-range values.

### Phase 3 — honesty (PR `fix(llm): LLM-4..LLM-5`)

- Remove `apply_bound_tightening` (deferring to FBBT/`tightening.py`) or implement
  the documented DAG walk; either way make the return contract true.
- Route `_analyze_structure` through the existing `problem_classifier`/DAG
  structure instead of `str(constraint)` scanning.
