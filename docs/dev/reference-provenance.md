# Reading HiGHS and SCIP: the provenance policy

`ref/` holds reading copies of the HiGHS and SCIP sources. It is gitignored and
never vendored. HiGHS is MIT, SCIP 10 and SoPlex are Apache-2.0; discopt is
EPL-2.0. The reference-reading agent definitions (`.claude/agents/highs-expert.md`,
`.claude/agents/scip-expert.md`) used to end this topic by telling the reader to
"flag that to the owner as a licensing/attribution decision rather than deciding
it yourself." The owner has now made that call, and this file is where it lives
so it survives outside a gitignored directory.

## The decision (owner, 2026-09-05)

> "We can't port it directly, but we can use it for inspiration and implement
> something similar. ... We should be careful to acknowledge and make it
> different"

**Read for the idea, implement it independently, cite the source.** Three things
follow. The third is a correction, not a restatement.

### 1. Work from the published paper, not the source file

Where a paper exists, implement from the paper. cMIR is {cite:t}`Marchand2001`;
lifted cover inequalities are {cite:t}`Gu1998`; lifted flow covers are
{cite:t}`Gu1999`. This produces an independent expression *and* usually a better
implementation, because a paper states the conditions that code leaves implicit.
Add the entry to `docs/references.bib` and cite it in the module header.

Use the reference source to settle a *specific* question the paper leaves
ambiguous — a tolerance, a degenerate case, an ordering — not as the thing being
transcribed.

### 2. Acknowledge

A module whose design came from reading a reference implementation says so in its
header, naming the file that informed it. That is the "acknowledge" half of the
decision and it costs one line. `lp/aggregation.rs` already carries this shape.

### 3. Do not lean on "it's in a different language, so it isn't copying"

This premise gets offered as the reason the whole question is moot. It is not
sound and the policy must not rest on it: **a translation into another language
is still a derivative work**, and reimplementing a C++ routine in Rust does not
by itself make a close transcription safe.

What actually protects an independent reimplementation is that **algorithms and
mathematical methods are not copyrightable — only their expression is.** The
protection therefore comes from writing discopt's own expression of the method,
which is exactly what (1) and (2) produce. It does not come from the change of
language. Stating the real reason matters because it tells you *which* practices
are load-bearing: working from the paper is, and being in Rust is not.

## What still gets escalated

One case remains genuinely open: a routine with **no published description**,
where the reference implementation *is* the specification and any correct version
would closely resemble it. Bring those to the owner rather than deciding them.
