# Pre-registration: 10 s panel for #966 deficiency (1)
Written 2026-08-11 ~20:22, BEFORE any 10 s data exists.

## Why add a budget at all
Post-#990, the 20 s panel has 0/19 instances with base wall > 1.05x budget.
A panel where the defect never appears cannot measure its fix; the 20 s arm now
contributes only noise (-0.70 +/- 0.69 s) to a gate that requires a win there.
15 s shows the deficiency on exactly 2/19 (hda, contvar). 10 s should widen it.

## Falsifiable predictions (either outcome is reportable; neither is a pass condition by itself)
P1. At 10 s, MORE than 2/19 instances show base wall > 1.05x budget.
    If 10 s shows <= 2, deficiency (1) is genuinely narrow and "broadly helpful"
    is NOT met -> recommend the flag stays OFF.
P2. On instances satisfying P1, cand punctuality beats base by >= +0.10x.
    If cand is within +/-0.05x of base, the flag does not fix the class -> OFF.
P3. Soundness holds: 0 unsound, 0 cert_regressions, 0 lost_incumbents,
    0 lost_bound, 0 incumbent_verification_failed, over a NON-ZERO oracle
    comparison count.
    ANY violation -> flag stays OFF, full stop, regardless of P1/P2. Soundness
    is not tradeable (CLAUDE.md sec.1).

## Kill criterion
If P3 fails, or if BOTH P1 and P2 fail, I will recommend closing #966 with the
flags OFF as a recorded negative result (the DISCOPT_CUT_INHERIT precedent).

## What this is NOT
Not a replacement for the 15 s / 20 s panels -- those are reported as run.
Adding a tighter budget is not a re-roll of a failed gate: it is testing the
mechanism in the regime where the mechanism exists. The 20 s result stands as
measured and is reported as measured.
