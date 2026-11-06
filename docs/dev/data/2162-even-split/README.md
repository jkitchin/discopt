# PR #2162: `DISCOPT_ROOT_FIXPOINT_EVEN_SPLIT` (retired)

- `engagement_probe.py` / `.jsonl`: the legacy arm on `main` `f43f29f7` at `time_limit=120`.
  - Each solve stops right after its outer root fixpoint.
  - Each OBBT round's allotment, time used and tightenings are recorded.
  - 122 cases: the 74 in-repo `.nl` plus 24 GDPlib instances under hull and big-m.
  - 53 reach the fixpoint and 13 engage.
- `engaged_nl_120s.*`: the 6 engaged `.nl`, ON vs OFF, 120 s, 2 reps.
  - Command: `flag_onoff_corpus_panel.py --only 4stufen,bchoco06,bchoco07,bchoco08,beuster,hda`.
  - `4stufen`'s dual bound is looser ON in both reps (21567.94 -> 21381.17, minimize).
- The GDPlib panel (192 solves) and the 20 s corpus panel are summarized in PR #2162's
  comments; their rows were not committed by the introducing session.
