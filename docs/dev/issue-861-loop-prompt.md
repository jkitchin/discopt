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
   `discopt_benchmarks/results/`. If an iteration before you left a PR open,
   your job this iteration is to finish it: check CI (`gh pr checks`), fix
   failures, merge it with `gh pr merge --merge`, verify the linked issue state
   afterwards, and update the ledger. Never start a new task while the previous
   task's PR is unmerged.
2. **Pick the next unstarted task** in plan order (T1, T2, T3, T4, T5a, T5b,
   T5c, T6, T7). Re-read the task's entry in §5 plus the design section (§2)
   and evidence (§1) it references.
3. **Run the task's entry experiment first** if the plan marks it as not yet
   executed, or if `python/discopt/_jax/incremental_mccormick.py`,
   `python/discopt/_jax/uniform_relax.py`, or
   `python/discopt/_jax/spatial_producer.py` changed since the plan's HEAD
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
