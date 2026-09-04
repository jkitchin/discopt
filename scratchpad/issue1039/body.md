## Summary

Running the `slow` and `correctness` tiers for the v0.8.0 release turned up **102
failures** (1384 passed, 16 skipped, 5 xfailed, 2 xpassed, 68m49s). Neither tier runs in
CI (excluded by the default `addopts`; the slow job is on-demand), so these have
accumulated invisibly — see #1034.

Triage of all 102:

| bucket | count | disposition |
|---|---|---|
| Presolve-box duals reported against the user's model | 79 | #1037 (single root cause) |
| Callback veto ignored, vetoed point certified optimal | 1 | #1038 (highest severity) |
| `hartman_3` surrogate budget miss | 1 | #1036 |
| Load artifacts (re-ran green on a quiet machine) | 3 | not defects — see below |
| Everything else | 18 | **this issue** |

### The 3 load artifacts

The tier ran while three unrelated `pounce` benchmark processes held ~100% CPU each
(load 5.4–8.8). Re-running the 23 non-minlptests failures at load 3.0 turned these
green:

* `test_912_work_budget.py::test_solve_deadline_still_stops_a_work_budgeted_search`
* `test_issue654_deadline_root_setup.py::test_issue654_dod_panel_honors_and_scales_with_time_limit[eg_all_s-7.657752093]`
* `test_sparsity.py::TestPerformance::test_sparse_faster_for_large_sparse`

All three are deadline/throughput assertions — exactly the class CLAUDE.md §9 warns
about. The other 20 reproduced under quiet load and are real.

---

## A. The HDA opt-out is inert — 3 tests, 1 root cause

All three assert that turning a flag **off** restores the legacy loose bound, and all
three get the same tight bound instead:

```
test_issue_517_numerical_dual_bound.py:59
    flag disabled must be the no-bound baseline, got -64473.44239643668
test_issue_671_lp_iterative_refinement.py:114
    flag OFF should be the loose candidate-A floor, got -64473.44239643685
test_relax_row_filter.py:119
    opt-out bound -64473.4 is unexpectedly tight -- the legacy no-filter path
    should give the loose candidate-A floor
```

The identical value across all three says one mechanism: the HDA / candidate-A
row-filter opt-out no longer disables anything.

This is policy-relevant, not cosmetic. The graduation rule in CLAUDE.md §5 requires that
a graduated flag "keep the `=0` opt-out and the legacy path intact". If the opt-out is
inert, the escape hatch promised when these flags graduated does not exist, and any
future A/B measurement using the flag silently compares a path against itself — the same
failure mode as #1035 (`relaxation_arithmetic="superposition"` inert since #632).

A tighter bound is not itself unsound, so this is a contract/measurement defect rather
than a correctness one. But it disarms the instrument used to validate correctness
changes, which is why it leads this list.

## B. `time_limit` overrun — 2 tests

```
test_875_root_setup_budget.py:479
    solve took 61.1s against a 30s time_limit (2.0x); root setup is still unbounded
    solve took 90.0s against a 60s time_limit (1.5x); root setup is still unbounded
```

Reproduced at load 3.0, so not a load artifact. `watercontamination0202` overruns its
budget by 1.5–2.0x, i.e. root setup remains outside the deadline — the thing #875 set
out to bound.

## C. Mechanisms that never fire — 3 tests

```
test_c42_cut_inherit_coldpath.py:213  pool-drop retry never fired
test_c42_cut_inherit_coldpath.py:289  no re-separation trigger fired on the stalling class
test_tx1_adaptive_nlp.py:107          expected the adaptive node-NLP back-off to fire on nvs09
```

Each asserts an adaptive mechanism engages on an instance chosen to provoke it. Same
shape as bucket A: the feature may be inert, or the probe may no longer select for it.
Worth checking whether the trigger conditions still match the instances.

## D. Node-count / gap targets missed — 3 tests

```
test_auto_cut_policy.py:92        assert 111 < (181 / 2)
test_psd_node_reduction.py:56     assert 111 < (181 / 2)
test_cut_recognizer.py:478        assert (3.0024852206298087 is None or ... < 2.0)
```

The first two are byte-identical numbers, so they are one instance measured twice: the
policy delivers 111 nodes where the test demands < 90.5 against a 181-node baseline —
a real reduction (39%), just short of the 50% the test encodes. These are performance
assertions of the kind PR #1033 removed elsewhere; they need either a re-derived
threshold backed by a measurement or removal, not a nudge.

## E. Accuracy / status misses — 4 tests

```
test_gauss_newton_hessian.py:369   assert 'feasible' == 'optimal'
test_gdpopt_loa.py:112             assert 'unknown' in ('optimal', 'feasible')
test_gp_corpus.py:199              assert 1.998683979470214 == 2.0 +- 1.0e-04
test_relax_row_filter.py:149       bchoco07: bound drifted
                                   (1.0000000000002582 -> 1.0000000000002498)
```

`test_gdpopt_loa` returning `unknown` is the most concerning of these. The `bchoco07`
drift is 8.4e-15 relative — a byte-identical assertion tripping on last-bit
nondeterminism, which needs deciding on its own terms (either the path is genuinely
deterministic and something broke it, or the assertion is too strict to be meaningful).

## F. Completeness misses — 2 tests

```
test_incumbent_injection_soundness.py:79
    nvs19 returned -1001.2 in 60s; known optimum is -1098.4
test_issue654_deadline_root_setup.py:282
    sonet23v4 lost its dual bound -- a §8 truncation regression
```

`test_incumbent_injection_soundness` is explicitly documented as "a *completeness*
check ... not a soundness one", and its sibling soundness test
(`test_suboptimal_warm_start_never_false_certifies`) passes — no false certificate here,
the search just does not reach the optimum in budget.

## G. `gap_certified` contract ambiguity — 1 test

```
test_tainted_tree_bound.py:85
    assert 338.2746666667722 <= ((0.0001 * 1100.4000000000003) + 1e-06)
```

Measured on nvs17:

```
time_limit=6s : status=time_limit  objective=-1100.4  bound=-1442.2
                gap=31.06%  gap_certified=True   nodes=19614
time_limit=20s: status=optimal     objective=-1100.4  bound=-1100.4
                gap=9.5e-14  gap_certified=True  nodes=60451
```

**Nothing here is unsound.** The bound −1442.2 is a valid lower bound (≤ the true
optimum −1100.4), the incumbent *is* the true optimum, the 31% gap is honestly
reported, and the status is `time_limit`, not `optimal`.

The failure is a genuine disagreement about what `gap_certified` means:

* Its docstring says "True if the reported optimality **gap** is mathematically
  certified" — under which `True` with a rigorous 31% gap is correct.
* The `__post_init__` guard comments say "the graduation panels count a
  `gap_certified=True` instance as certified" and call premature `True` "a false
  certification (the benchmark gate would miscount it as a solved/certified instance)"
  — under which the test is right and the flag should be `False` here.

Both readings are load-bearing somewhere. This needs an owner decision on the contract
before either the field or the test is touched; changing the test to match current
behavior without settling it would be weakening a validation to make a test pass.

---

## Suggested handling

Buckets A and B are the ones with contracts behind them (a promised opt-out; a promised
deadline). C and D are probe-health questions. G needs a decision, not a code change.

Prerequisite for all of it: #1034. Until these tiers run somewhere on a schedule, any fix
here can regress the same way — #1038 is a fixed correctness guard that broke and stayed
broken because nothing ran its test.
