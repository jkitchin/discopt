#!/usr/bin/env python
"""R1.2 entry experiment — claim-boundary vs canonical-atom census (issue #632).

Before replacing the federated univariate claimers with the canonical dominance
dispatch (R1.2), we must know the canonical atom model faithfully covers what the
federation actually claims — otherwise the cutover would silently drop or mis-own a
claim. This script measures that over the vendored corpus:

For every instance it (1) builds the relaxation with the auditor, recording each
raw-expression id the federation claimed via an expr-id owner family
(univariate / composite-univariate / composite-multivar), and (2) canonicalizes the
same model. Then for each claimed expr id it checks the corresponding canonical
node and reports how many are:

- **covered** — present in the canonical DAG and a genuine nonlinear atom node
  (neither affine nor opaque): the cutover can own it;
- **opaque** — the canonicalizer refused it (would fall back — a coverage gap);
- **affine** — the canonicalizer thinks it is linear (a real disagreement);
- **missed** — not walked by canonicalize (e.g. an issue-267 claim on a distributed
  node, handled by R2.3, not R1.2's univariate scope).

A clean R1.2 go signal is **zero opaque and zero affine** among covered claims: the
canonical model never disagrees that a federation-claimed node is a relaxable
nonlinear atom.

Usage (repo root, built extension importable)::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/r12_boundary_census.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
sys.path.insert(0, str(_REPO / "python"))


def census() -> dict:
    from discopt._jax.canonical_expr import canonicalize, is_affine
    from discopt._jax.claim_audit import audit_build
    from discopt.modeling.core import from_nl

    totals = {"claims": 0, "covered": 0, "opaque": 0, "affine": 0, "missed": 0}
    per_family: dict[str, int] = {}
    n_inst = 0
    for p in sorted(_NL_DIR.glob("*.nl")):
        try:
            model = from_nl(str(p))
            report = audit_build(model)
            dag = canonicalize(model)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {p.stem}: {exc!r}", file=sys.stderr)
            continue
        n_inst += 1
        for family, ids in report.claimed_expr_ids.items():
            per_family[family] = per_family.get(family, 0) + len(ids)
            for eid in ids:
                totals["claims"] += 1
                node = dag._memo.get(eid)
                if node is None:
                    totals["missed"] += 1
                elif node.is_opaque:
                    totals["opaque"] += 1
                elif is_affine(node):
                    totals["affine"] += 1
                else:
                    totals["covered"] += 1
    return {"instances": n_inst, "totals": totals, "per_family": per_family}


def main() -> int:
    result = census()
    t = result["totals"]
    print(f"instances scanned: {result['instances']}")
    print(f"expr-id claims:    {t['claims']}")
    print(f"  covered (nonlinear atom): {t['covered']}")
    print(f"  opaque  (would fall back): {t['opaque']}")
    print(f"  affine  (disagreement):    {t['affine']}")
    print(f"  missed  (not walked; R2.3 scope): {t['missed']}")
    print(f"per family: {result['per_family']}")
    go = t["opaque"] == 0 and t["affine"] == 0
    print(f"\nR1.2 boundary go-signal (0 opaque, 0 affine among covered): {'YES' if go else 'NO'}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
