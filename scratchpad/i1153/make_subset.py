"""Select the phase-B panel: instances still OPEN at the largest phase-A rung.

Chosen by that measured property, never by name (CLAUDE.md §2). An instance that
certifies optimality inside the smallest budget is monotone by construction and
would only dilute the panel.
"""
import json, sys
rows = json.load(open(sys.argv[1]))
last = lambda v: v[-1]
open_names = [
    n for n, v in sorted(rows.items())
    if v and "error" not in last(v) and last(v).get("status") not in ("optimal", "infeasible")
]
json.dump(open_names, open(sys.argv[2], "w"), indent=1)
print(f"# open at the largest rung: {len(open_names)} of {len(rows)}")
print(" ".join(open_names))
