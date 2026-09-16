# Audit — is the `.nl` decoder duplicated between discopt and `pounce-nl`? (2026-09-16)

**Issue.** [#1213](https://github.com/jkitchin/discopt/issues/1213) — discopt maintains
`crates/discopt-core/src/nl_parser.rs` while its `pounce-solver` dependency ships
`pounce-nl`, which also reads AMPL `.nl`. Two readers of one format spec in one
dependency closure, same author. The issue asks one narrow question: **is the shared
*decoding* layer large enough and stable enough to be worth factoring into one crate
with two builders on top?**

**Decision: keep both. Do not factor a shared decoder crate.** The rationale and the
measurements behind it are below. This decision is reversible — §6 names the two
things that would change it.

**Environment.** discopt @ `3868cd9`, `pounce-solver` 0.11.0 (the wheel a fresh
`pip install discopt` resolves today; the issue was written against the 0.10 floor
in `pyproject.toml`). Corpus: the 66-instance in-repo snapshot
`python/tests/data/minlplib_nl/`.

## Summary

| Task from #1213 | Result |
|---|---|
| A. Line-level decode/build split | ✅ measured — **213 lines** of genuinely shared decode surface, not 3,527 |
| B. Opcode/segment coverage, both directions | ✅ measured — each side accepts what the other refuses; the refusals are *policy*, not gaps |
| C. Decide and record | ✅ **keep both**, this document |
| D. Corpus gate (only if factoring) | n/a — not factoring |

Two findings not anticipated by the issue:

- **`pounce-nl` cannot read binary (`b`-header) `.nl` at all.** One instance in the
  in-repo corpus (`st_miqp5.nl`) is binary; `pounce.read_nl` refuses it with
  `stream did not contain valid UTF-8`. discopt reads it. So the 313-line binary
  transcoder — 60 % of discopt's IR-free decode surface — has no counterpart to
  share with, which removes most of the apparent overlap.
- **The two decoders agree everywhere they overlap.** 65/65 objectives and 4,009
  variable bounds match at a common point (§4). There is no format-level bug
  currently being fixed twice, which is the cost the issue was written to avoid.

## 1 — The shared surface is 213 lines, not 3,527

`scratchpad/issue1213/classify_nl_parser.py` scans top-level items with a
brace-depth scanner, assigns each an explicit role (an unlisted item is a hard
error, so the table cannot silently go stale), and classifies every code line in
the parsing functions as IR-coupled or IR-free. It asserts that every item marked
format-only is in fact free of any IR type mention, prints its executed-classification
count (1,420) and exits non-zero if that count is zero.

```
total lines                                3642   (3627 before this audit's doc note)
  test module                              1305   (47% of all code lines)
non-test code lines                        1471
  IR-free decoding, movable as-is           526   (35.8%)
    of which binary->text transcoder        313
  fused decode+build functions              864   (58.7%)
    IR-coupled lines inside them            185
    IR-free lines inside them               679
    coupled<->free transitions              199
  public API wrappers / error type / data    73
```

Three corrections to the issue's framing fall out of this:

1. **3,527 is mostly tests.** The implementation is 1,471 code lines; 1,305 of the file's 2,776 code lines (47 %)
   are the `#[cfg(test)]` module. The maintenance surface is under half what the
   issue states.
2. **The movable decode surface is 526 lines, and 313 of those are the binary
   transcoder** (`BinCursor`, `transcode_binary_body`, `transcode_expr`,
   `opcode_arity`, `bound_value_count`, `fmt_f64`, `transcode_binary_nl`). Since
   `pounce-nl` does not read binary `.nl` (§3), that code has nothing to share
   with. **Genuinely shared decode surface: 526 − 313 = 213 lines** — the header
   parser and its header struct (146 + 26), the line reader (23), and the
   numeric lexers (18) — i.e. **14 %
   of the non-test file.**
3. **The remaining 864 lines are not layered, they are fused.** `parse_opcode` and
   `parse_nl_full` interleave "read the next token" and "intern an arena node" at
   statement granularity: 199 transitions between IR-coupled and IR-free lines.
   Every `parse_opcode` arm is *decode one operand, build one node*. Splitting them
   means inventing a neutral event/token stream and rewriting both halves against
   it — a rewrite of the battle-tested path, not a move.

## 2 — The two libraries produce different things, and that has not changed

The issue's central qualification holds against 0.11.0, re-checked rather than
assumed. `pounce.read_nl` returns `NlProblem`, whose entire public surface is
evaluation:

```
con_names constraints g_l g_u gradient hessian hessian_structure
hessian_vector_product jacobian jacobian_structure m minimize n nnz_hess
nnz_jac obj_constant objective var_names variant x0 x_l x_u
```

There is no expression accessor. pounce 0.11.0 *does* newly export an `NlExpr`
DAG type, which was worth checking — but it is a **builder** (`const_`, `var`,
`sum`, `sin`, …, feeding `build_nl_problem`), not an inspector: no opcode, no
operand traversal, and no path from an `NlProblem` back to one. So there is still
no `NlProblem → ModelRepr` lowering and still cannot be one. McCormick envelopes,
FBBT, term classification and convexity detection all need the arena that only
discopt's parser produces.

## 3 — Coverage diverges in *both* directions, and the divergences are deliberate

`scratchpad/issue1213/gen_opcode_probes.py` writes one minimal single-variable
`.nl` per opcode; both readers are then asked the same 40 questions
(`scratchpad/issue1213/op_*.jsonl`).

| | discopt | `pounce-nl` 0.11.0 |
|---|---|---|
| text (`g`) `.nl` | accepts | accepts |
| **binary (`b`) `.nl`** | **accepts** (transcodes to text, then one parser) | **refuses** — not valid UTF-8 |
| o0/1/2/3/5/11/12/15/16/37–47/49–54 | accepts | accepts |
| **o76/o77/o78** (`x^c`, `x^2`, `c^x` shorthands) | **accepts** | **refuses** — `unsupported opcode` |
| **o35** if-then-else | **refuses** — `UnsupportedOpcode` | **accepts** |
| **o48** atan2 | **refuses** — `UnsupportedOpcode` | **accepts** |
| o13/14/55/57/58 (floor/ceil/intdiv/round/trunc) | refuses, named | refuses |
| o56/59/60/61 | refuses, `UnknownOpcode` | refuses |
| complementarity (type-5 `r` rows) | recovered → MPEC path | no API |
| suffixes (`S` segment) | parsed | no API |
| `AMPLFUNC` external functions | not supported | resolved |
| objective structure | `ModelRepr` arena | none |
| derivatives | objective eval only | full TNLP (grad/Jac/Hess) |
| "infinity" sentinel | `1e20` | `1e19` |

*Honest scope note on o76/77/78* (CLAUDE.md §4 — a mechanism validated only on a
synthetic proxy may be a no-op on the real class): this gap is demonstrated on the
synthetic probes, **not** observed on the corpus. The 65 text instances use only
o0, o1, o2, o3, o5, o16, o39, o43, o44, o54 — MINLPLib's writer emits `o5` for
powers, never the shorthands. The binary-format gap, by contrast, *is* observed on
a real corpus file.

The important structural point is not the size of the table but the shape of it.
**discopt refuses o35 and o48 on purpose**: they have no sound representation in
the IR, and per CLAUDE.md §1/§3 a loud refusal beats a silent substitution that
would certify the optimum of a different problem (correctness issue C-5). pounce
accepts them because a tape can evaluate anything it can differentiate. So a
shared decoder would have to decode a *superset* and push every accept/refuse
decision up into the two builders — leaving the refusal policy, which is the part
carrying the correctness risk, duplicated anyway. Factoring would share the
213 easy lines and none of the dangerous ones.

## 4 — Where they overlap, they already agree

`scratchpad/issue1213/agree.py` evaluates both readers at the same point
(`x_j = clamp(0.5, x_l, x_u)`) for every corpus instance both accept:

```
objective comparisons: 65      (all agree, rtol 1e-7)
bound comparisons    : 4009    (all agree)
skipped              : 1       st_miqp5 — binary, pounce cannot read it
```

The only systematic difference is the infinity sentinel (pounce 1e19, the Ipopt
`nlp_*_bound_inf` convention; discopt 1e20), which is a convention, not a
disagreement about the file. Both probes print an executed-comparison count and
exit non-zero if it is zero.

This is the measurement that settles the cost side of the issue. The premise of
factoring is "format-level bugs have to be fixed twice"; on the corpus that
justifies the parser's existence, there are currently **no** such bugs to fix even
once. The duplication is costing nothing today.

## 5 — Costs of factoring, against that benefit

- **Zero user-visible payoff.** Maintenance-only, as the issue states, and §4 shows
  the maintenance is not currently being paid.
- **It is a rewrite, not a move.** 199 interleave transitions (§1); only 14 % of the
  file moves as-is.
- **It churns the battle-tested path.** `nl_parser.rs` is validated against the
  ~4,800-instance MINLPLib snapshot and the 66-instance in-repo corpus, and the
  issue's own gate (byte-identical `ModelRepr` on all 66, complementarity and
  suffixes preserved) is the right bar — it is just an expensive bar to clear for
  no gain.
- **A new cross-crate edge has a measured cost here.** `crates/discopt-core/Cargo.toml`
  records `feral` 0.11.2 → 0.11.3 silently regressing set-covering from ~3 s to a
  >30 s timeout on a *patch* bump (LU refactor going quadratic→cubic in row count),
  now guarded by `test_setcover_lp_regression.py`. Every new edge also inherits the
  bound-neutrality bump protocol (the 49/49 bit-identical panel). Putting the `.nl`
  front door behind a version bump buys a class of failure the repo has already
  been burned by.
- **The repo already applies the "one source of truth" principle where it pays.**
  The binary transcoder's header comment is explicit: rather than duplicate the
  recursive-descent parser against a byte cursor — "a second source of truth that
  could silently diverge" — it transcodes binary into the exact text token stream
  the proven parser consumes, reusing 100 % of the model-building and soundness
  logic. That is the same principle the factoring proposal invokes, and discopt
  already spent it on the case where the duplicated logic was the *risky* part.
  Between discopt and pounce the duplicated part is the *safe* part (§3), which is
  why the same principle does not apply.

## 6 — What would change this decision

Neither of these holds today; record a falsification here if one starts to.

1. **A format-level bug is found and has to be fixed in both readers.** §4's
   agreement check is the detector: re-run `scratchpad/issue1213/agree.py` after any
   `.nl` reader change on either side. The first genuine double-fix is the evidence
   that the maintenance cost the issue hypothesized is real.
2. **`pounce-nl` grows a structure-preserving output** — an inspectable expression
   DAG reachable from a parsed `NlProblem`, not just the `NlExpr` builder. That
   would make an `NlProblem → ModelRepr` lowering possible for the first time and
   reopens the much larger question the issue deliberately scoped out.

A third, narrower opportunity exists and is *not* recommended now: if pounce ever
needs to read binary `.nl`, discopt's transcoder (313 lines, IR-free, already
IR-independent) is the one piece that could move across as-is without touching
`ModelRepr`.

## Reproducing

```bash
# 1 — line-level split (prints 1,420 executed classifications)
python3 scratchpad/issue1213/classify_nl_parser.py

# 2 — what discopt's parser accepts/refuses over a directory of .nl files
cargo run -p discopt-core --release --example nl_audit -- python/tests/data/minlplib_nl

# 3 — the same question to pounce-nl  (pip install pounce-solver)
python3 -u scratchpad/issue1213/pounce_side.py python/tests/data/minlplib_nl

# 4 — per-opcode probes, both readers
python3 scratchpad/issue1213/gen_opcode_probes.py scratchpad/issue1213/opprobes

# 5 — cross-implementation agreement (exits non-zero on any disagreement)
python3 -u scratchpad/issue1213/agree.py python/tests/data/minlplib_nl
```

`crates/discopt-core/examples/nl_audit.rs` is kept in-tree as the discopt side of
that differential: it walks a directory of `.nl` files reporting accept/refuse plus
model shape, and `--eval <file.nl> <x.txt>` prints the objective at a given point
for the agreement check. It prints the number of files attempted and exits non-zero
if that number is zero, so a probe that traverses nothing cannot read as a pass.

## Related

- **#1212** (standalone `discopt-solve` binary) is unblocked by this: it needs
  `ModelRepr`, which only discopt's parser produces. The `pounce-cli` / `discopt-solve`
  overlap is a separate question and is not addressed here.
- The broader "do discopt's Rust components belong in the POUNCE crate family"
  question (notably `lp/simplex/`, which POUNCE lacks) remains out of scope, as
  #1213 states.
