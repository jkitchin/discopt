You are implementing GitHub issue #861 for the discopt repo, working from the
evidence-based plan in `docs/dev/issue-861-incremental-admission-plan.md`. Read
that plan file COMPLETELY before doing anything else — its §0 protocol is
binding, its §1 evidence is already measured (do not re-derive it unless the
tree has drifted under the cited files), and its §5 task list (T1–T7) is your
work queue. Also read `CLAUDE.md` and honor it fully (especially §§1–11:
correctness before performance, entry experiments before implementation,
executed-assertion counts, no swallowed exceptions in instruments).

Each invocation of this prompt is ONE iteration of a loop. Do exactly one
plan task (or one T5 sub-task) per iteration, end to end:

1. **Recover state.** Determine which plan tasks are already done by looking at
   the repo itself: the "## Progress ledger" section at the bottom of the plan
   doc (create it on the first iteration), merged/open PRs (`gh pr list
   --state all --search "861"`), and the latest admission-sweep results in
   `discopt_benchmarks/results/`.
2. **Pick the next unstarted task** in plan order (T1, T2, T3, T4, T5a, T5b,
   T5c, T6, T7). Re-read the task's entry in §5 plus the design section (§2)
   and evidence (§1) it references.

   **Do not wait on CI or on a merge to start the next task** (owner decision,
   2026-07-28 — three consecutive iterations were spent idling on ~20-minute CI
   cycles). Concretely:
   - Merge a finished PR with `gh pr merge --merge --auto` and move on in the
     same iteration. Do not block on the full CI matrix; the *local* gates below
     are what you actually rely on.
   - If a previous PR is still open when you start, branch the next task **off
     that PR's branch** (stacked) rather than off `main`, and target the PR's
     branch in `--base`. Rebase onto `main` once the parent merges.
   - Still check back on any PR you left open: if its CI went red, fixing it is
     the *first* thing you do in the iteration that notices, before continuing
     the new task.

   **This relaxes PROCESS gates only, never CORRECTNESS gates.** The §4 gates
   stay mandatory and stay *local* (they are what makes an admission widening
   safe): the admission sweep with `--baseline`, the node-parity panel on newly
   admitted instances, the oracle check, and — the lesson from T3 — CI's exact
   fast marker selection run locally before pushing:
   `pytest python/tests/ -m "not slow and not correctness and not integration and
   not amp_benchmark and not requires_cyipopt and not memory_heavy"
   --ignore=python/tests/test_correctness.py --ignore=python/tests/test_amp.py`.
   Never merge a PR whose local gates you have not run, and never merge one whose
   CI has actually gone red.
3. **Run the task's entry experiment first** if the plan marks it as not yet
   executed, or if `python/discopt/_relax/incremental_mccormick.py`,
   `python/discopt/_relax/uniform_relax.py`, or
   `python/discopt/_relax/spatial_producer.py` changed since the plan's HEAD
   (`bdc104bb`). If the experiment falsifies the task's premise, STOP the task,
   record the falsification in plan §7 and the ledger, comment on issue #861,
   and end the iteration — do not implement against a falsified premise.
4. **Implement on a feature branch** (`git checkout -b <type>/861-<task> main`
   after `git pull`), with the regression tests the task's G1 gate names
   (fail-before/pass-after). NEVER commit to main, never use --no-verify,
   never weaken `_validate`'s comparisons or any other gate to make something
   pass — if a task can only pass that way, it is dead; record why and stop.
5. **Run the task's verification gates** exactly as §4 specifies (sweep
   before/after table, LP-value parity panel, certifying panel, pytest smoke +
   adversarial suite as applicable) and paste the numbers — not adjectives —
   into the PR description (write PR/issue bodies to a file and use
   --body-file; never inline --body). The admission count must be monotone
   non-decreasing and no previously-admitted instance may flip to declined
   (plan §0.5 tells you what to do if one does).
6. **Open the PR** (`Contributes to #861`; only T7 uses `Closes #861`), then
   merge it yourself once CI is green (`gh pr merge --merge`). If CI cannot go
   green this iteration, leave the PR open with a comment describing exactly
   what remains, update the ledger, and end the iteration.
7. **Close the loop for the iteration:** append to the plan doc's Progress
   ledger (task id, PR number, sweep numbers, gates run, anything falsified)
   in the SAME PR as the task when possible, and post a short comment on
   issue #861 summarizing the landed task and the current admitted/declined
   count.

Rules of engagement:
- Prefer finishing over documenting: if you notice three artifacts and no code
  change in your session, stop analyzing and build (CLAUDE.md working-on-an-
  issue §6).
- Scope discipline: plan §6 non-goals are hard fences. The ball_mk2_30 primal
  gap, engine route-pinning, and native-kernel family extensions are OFF
  limits (except the T3 spatial_producer classifier audit).
- Measurement discipline: every probe/panel prints an executed-assertion count
  and exits non-zero when zero; timing claims need interleaved A/B, a load
  check, and a spread; long jobs print incremental progress and are run with
  `python -u`.
- If a measurement contradicts the plan, THE MEASUREMENT WINS: record the
  falsification in plan §7, re-scope the task in the plan (edit it in the same
  PR), and continue with the corrected scope.

Termination: after T7's PR is merged and issue #861 has the close-out comment
(explicit "issue can be closed: yes/no + what remains"), verify the issue's
final state matches that verdict, write the line `RALPH-861-COMPLETE` as the
last line of the plan doc's Progress ledger, and output `RALPH-861-COMPLETE`
as the final line of your reply. If the ledger already contains
`RALPH-861-COMPLETE` when you start, do nothing and output it again.
