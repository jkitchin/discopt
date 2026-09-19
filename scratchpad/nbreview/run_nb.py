#!/usr/bin/env python
"""Execute a notebook in place, saving outputs. Reports per-cell errors.

Usage: run_nb.py <notebook.ipynb> [timeout_seconds]
Exit 0 = all cells ran clean; 1 = at least one cell raised.
"""
import sys, os, time, json
import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 900

nb = nbformat.read(path, as_version=4)
client = NotebookClient(
    nb,
    timeout=timeout,
    kernel_name="python3",
    resources={"metadata": {"path": os.path.dirname(os.path.abspath(path))}},
    allow_errors=True,          # run every cell; we inspect errors ourselves
    record_timing=True,
)
t0 = time.time()
client.execute()
elapsed = time.time() - t0
nbformat.write(nb, path)

# Report: which cells errored.
errors = []
executed = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    if cell.source.strip():
        executed += 1
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            errors.append((i, out.get("ename"), out.get("evalue"),
                           "\n".join(out.get("traceback", []))[-3000:]))

print(f"=== {os.path.basename(path)}: executed {executed} code cells in {elapsed:.1f}s ===")
if executed == 0:
    print("PROBE DID NOT FIRE: zero code cells executed")
    sys.exit(2)
if not errors:
    print("RESULT: CLEAN (0 errors)")
    sys.exit(0)
print(f"RESULT: {len(errors)} ERRORING CELL(S)")
for idx, ename, evalue, tb in errors:
    print(f"\n--- cell index {idx}: {ename}: {evalue}\n{tb}")
sys.exit(1)
