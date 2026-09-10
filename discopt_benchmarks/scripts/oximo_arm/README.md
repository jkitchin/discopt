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

`oximo-2026-09-10.tsv` is the measured oximo arm from the run recorded in §48,
checked in so the panel can be recombined without cargo or a network fetch:

```bash
python -u discopt_benchmarks/scripts/issue1215_cross_tool_panel.py \
    --oximo discopt_benchmarks/scripts/oximo_arm/oximo-2026-09-10.tsv
```

It is a snapshot of one box, so re-measure the oximo arm before publishing a new
comparison; it is here for reproducing §48's tables, not as a moving baseline.

To recombine stored measurements without re-running either arm:

```bash
python discopt_benchmarks/scripts/issue1215_cross_tool_panel.py \
    --from-tsv python.tsv --oximo oximo.tsv
```

Both arms emit the same TSV columns
(`tool family idiom rows vars construct_s write_s`) and both read the row and
variable counts back out of each written `.nl` header, so an arm cannot look
fast by building a smaller model than it claims. The combiner checks that every
arm agrees on those counts before reporting a ratio, prints the number of
comparisons it made, and exits non-zero if it reported nothing.

Results and their reading: `docs/dev/performance-plan.md` §48.
