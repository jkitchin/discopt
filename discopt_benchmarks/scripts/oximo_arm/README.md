# oximo arm of the cross-tool modelling panel (#1215)

`main.rs` is the oximo half of `issue1215_cross_tool_panel.py`. It is kept as a
source file rather than a vendored crate so the repo does not carry a second
Cargo workspace; build it in a scratch crate:

```bash
cargo new --bin oximo_arm && cd oximo_arm
cargo add oximo --features io      # oximo is a RUST crate; `pip install oximo`
                                   # finds an unrelated package and fails
cp <this dir>/main.rs src/main.rs
cargo run --release > oximo.tsv
```

Then run the Python arm and combine:

```bash
python -u discopt_benchmarks/scripts/issue1215_cross_tool_panel.py \
    --oximo oximo.tsv
```

`main.rs --memory` measures retained memory instead of wall time, emitting
`tool family idiom rows vars retained_b`. It re-executes itself once per point,
reads the same `VmRSS` field of `/proc/self/status` the Python arm reads, and
warms up on a tiny model first. Pair it with the panel's `--memory`:

```bash
cargo run --release -- --memory > oximo-memory.tsv
python -u discopt_benchmarks/scripts/issue1215_cross_tool_panel.py \
    --memory --oximo oximo-memory.tsv
```

`oximo-2026-09-10.tsv` and `oximo-memory-2026-09-10.tsv` are the measured oximo
arms from the runs recorded in §48 and §50, checked in so the panel can be
recombined without cargo or a network fetch:

```bash
python -u discopt_benchmarks/scripts/issue1215_cross_tool_panel.py \
    --oximo discopt_benchmarks/scripts/oximo_arm/oximo-2026-09-10.tsv
python -u discopt_benchmarks/scripts/issue1215_cross_tool_panel.py --memory \
    --oximo discopt_benchmarks/scripts/oximo_arm/oximo-memory-2026-09-10.tsv
```

They are snapshots of one box, so re-measure the oximo arm before publishing a
new comparison; they are here for reproducing §48's and §50's tables, not as a
moving baseline.

To recombine stored measurements without re-running either arm:

```bash
python discopt_benchmarks/scripts/issue1215_cross_tool_panel.py \
    --from-tsv python.tsv --oximo oximo.tsv
```

Both arms emit the same TSV columns for a given mode
(`tool family idiom rows vars construct_s write_s` for timing,
`tool family idiom rows vars retained_b` for `--memory`; the combiner rejects
one mode's TSV in the other's run) and both read the row and
variable counts back out of each written `.nl` header, so an arm cannot look
fast by building a smaller model than it claims. The combiner checks that every
arm agrees on those counts before reporting a ratio, prints the number of
comparisons it made, and exits non-zero if it reported nothing.

`node_density.rs` is a second probe for the same scratch crate: it reads the
arena at each construction stage to show *why* oximo's per-element rows are
cheap (linear fusion in the operator overloads, handles instead of objects).
Swap it in for `src/main.rs` the same way. Its discopt counterpart is
`scripts/issue1215_node_density.py`.

Results and their reading: `docs/dev/performance-plan.md` §48 (the timing
panel), §49 (why oximo is fast without being vectorised) and §50 (memory).
