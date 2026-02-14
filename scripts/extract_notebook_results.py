#!/usr/bin/env python3
"""Extract benchmark result lines from an executed Jupyter notebook.

Usage:
    python scripts/extract_notebook_results.py <notebook.ipynb> <output.json>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <notebook.ipynb> <output.json>", file=sys.stderr)
        sys.exit(1)

    nb_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    nb = json.loads(nb_path.read_text())
    lines = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for out in cell.get("outputs", []):
                text = "".join(out.get("text", []))
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped and ("obj=" in stripped or "status=" in stripped):
                        lines.append(stripped)

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "notebook",
        "source": str(nb_path.name),
        "n_results": len(lines),
        "results": lines,
    }
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Extracted {len(lines)} result lines -> {out_path}")


if __name__ == "__main__":
    main()
