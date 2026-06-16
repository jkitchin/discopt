#!/usr/bin/env python
"""Thin wrapper around ``discopt.gams.verify`` (kept for ``make gams-verify``).

The implementation and the ``.gms`` corpus now live inside the installed package
(``discopt.gams.verify`` / ``discopt.gams`` data), so pip users can run the same
end-to-end check via ``discopt gams-verify``. Prefer that; this wrapper just
forwards its arguments.
"""

from __future__ import annotations

from discopt.gams.verify import main

if __name__ == "__main__":
    raise SystemExit(main())
