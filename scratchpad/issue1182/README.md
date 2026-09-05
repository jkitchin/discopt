# #1182 entry experiment — exact continuous (simplex/CNF) lowering

The probes CLAUDE.md §4 requires before implementing the Theorem-1 lowering of
[arXiv:2601.03906v1](https://arxiv.org/abs/2601.03906v1). Results are written up in
`docs/dev/performance-plan.md` §26; the `*_run.log` files here are the raw output of
the runs quoted there.

Every probe prints an executed-comparison / executed-assertion count and exits
non-zero if it is zero (CLAUDE.md §6).

```bash
# E1 needs the benchmark harness on the path (native GDPlib builders); the others
# only need this directory.
export PYTHONPATH=discopt_benchmarks:scratchpad/issue1182

python -u scratchpad/issue1182/E1_entry_experiment.py --time-limit 60   # real GDP corpus
python -u scratchpad/issue1182/E2_paper_class.py --steps 3,5 --reps 2   # the paper's own class
python -u scratchpad/issue1182/E3_bigm_refusal.py                       # capability, 4 fixtures
python -u scratchpad/issue1182/E5_corpus_refusal_scan.py                # GDPlib refusal scan
```

`E5` additionally needs `pyomo` and `gdplib` **installed from source** (the PyPI
wheel omits the model data files), plus `pandas`/`matplotlib`/`openpyxl`:

```bash
pip install pyomo pandas matplotlib openpyxl \
    "gdplib @ git+https://github.com/SECQUOIA/gdplib.git"
```

`simplex_proto.py` / `source_check.py` are the **prototype** the entry experiment
ran against, deliberately kept: they are what produced the numbers in §26, and the
shipped `discopt._relax.simplex_lowering` reproduces their node counts exactly
(jobshop 251, small_batch 3), which is the cross-check that the productionized
lowering is the thing that was measured.

**Outcome.** The speed hypothesis is falsified (E1: 0 of 3 instances meet the kill
criterion; E2: the same on the class the paper targets). The capability motive
survives and is what shipped: E3's fourth fixture is a row both classical lowerings
refuse, and E5 finds 18 such rows in GDPlib's `stranded_gas`.
