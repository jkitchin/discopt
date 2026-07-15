"""Claim-boundary audit instrumentation (issue #632, stage R0).

Two things live here, both read-only with respect to the relaxation the solver
builds:

1. :func:`relaxation_fingerprint` — the SHA-256 of a built MILP relaxation's
   ``(_c, _A_ub, _b_ub, _bounds, _integrality)`` (extracted from the inline #630
   guardrail in ``test_lr2_offneutral_relaxation.py``). It is the byte-identity
   primitive the differential gate (plan §3.2) and the committed baseline
   (``docs/dev/data/claim-baseline.jsonl``) are built on.

2. :func:`audit_build` / :class:`AuditReport` — the **claim auditor**. For one
   ``build_milp_relaxation`` call it reconstructs, from the returned varmap, which
   *owner family* claimed each nonlinear node (which aux column each produced).
   This is (a) the "unchanged dispatch" classifier the differential gate uses to
   decide which corpus instances must stay byte-identical vs which changed and
   why, (b) the ownership diff used to *attribute* every changed instance to a
   dispatcher rule in later stages, and (c) from R2.5 onward the CI assertion that
   every nonlinear node has exactly one owner and the defer-list fires zero times.

The auditor derives everything from the build's *output* varmap, so it never
changes what the solver builds — proven in ``test_claim_audit.py`` by fingerprint
equality between an audited and an un-audited build.

The defer-fire counter (:func:`defer_audit` / :func:`note_defer`) was the hook the
federated defer predicates reported to. The #632 cutover removed the federation
(``build_milp_relaxation`` now delegates unconditionally to the uniform engine, and
no defer predicate remains), so this counter has **no production consumer** — it is
retained only as standalone-tested instrumentation (a no-op unless a
:func:`defer_audit` context is active) pending a decision to remove it or repurpose
it for the uniform engine. The still-live audit here is :func:`audit_build` /
:class:`AuditReport`, which run against the uniform build and are exercised by the
tests.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import hashlib
from collections import Counter
from typing import Any, Optional

import numpy as np

try:  # scipy is a hard dep of the relaxation path, but keep the import defensive.
    import scipy.sparse as _sp

    def _is_sparse(a: Any) -> bool:
        return bool(_sp.issparse(a))

except Exception:  # pragma: no cover - scipy always present in practice

    def _is_sparse(a: Any) -> bool:
        return False


__all__ = [
    "relaxation_fingerprint",
    "fingerprint_model",
    "AuditReport",
    "audit_build",
    "defer_audit",
    "note_defer",
]


# --------------------------------------------------------------------------- #
# R0.3 — relaxation fingerprint
# --------------------------------------------------------------------------- #
def relaxation_fingerprint(relax: Any) -> str:
    """Deterministic SHA-256 of a built MILP relaxation matrix.

    Hashes every array a claim (additive row/column) could touch: the objective
    ``_c``, the inequality matrix ``_A_ub`` (densified in a stable order), the RHS
    ``_b_ub``, the variable ``_bounds``, and the ``_integrality`` mask. Two equal
    digests mean identical LP relaxations — same columns, same rows, same numbers.
    """
    h = hashlib.sha256()

    def _feed(label: str, arr: Any) -> None:
        h.update(label.encode())
        if arr is None:
            h.update(b"None")
            return
        if _is_sparse(arr):
            arr = np.asarray(arr.todense())
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())

    _feed("c", relax._c)
    _feed("A_ub", relax._A_ub)
    _feed("b_ub", relax._b_ub)
    _feed("bounds", np.asarray(relax._bounds, dtype=np.float64) if relax._bounds else None)
    _feed(
        "integrality",
        np.asarray(relax._integrality, dtype=np.float64)
        if relax._integrality is not None
        else None,
    )
    return h.hexdigest()


def _build(model: Any, terms: Any, disc: Any):
    """Build the relaxation for ``model`` (importing lazily to avoid a cycle)."""
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms

    if terms is None:
        terms = classify_nonlinear_terms(model)
    if disc is None:
        disc = DiscretizationState()
    return build_milp_relaxation(model, terms, disc)


def fingerprint_model(model: Any, terms: Any = None, disc: Any = None) -> str:
    """Convenience: build ``model``'s relaxation and fingerprint it."""
    relax, _info = _build(model, terms, disc)
    return relaxation_fingerprint(relax)


# --------------------------------------------------------------------------- #
# R0.4 — claim auditor (ownership map derived from the build's varmap)
# --------------------------------------------------------------------------- #
# Each owner family and how to read its claimed (expr_id -> aux_col) pairs out of
# the varmap the build returns. "list_attr" families are lists of relaxation
# dataclasses carrying (expr_id/base_col, aux_col); "dict_id" families are
# id(expr)->aux_col maps; "dict_key" families are structural-key->aux_col maps
# (the product side — collision-free, reported for completeness).
_LIST_OWNERS = {
    # varmap key -> (id field, column field). These lists of relaxation
    # dataclasses are the authoritative per-owner claim records. The id-keyed
    # dict views (e.g. varmap["univariate"]) are the *same* claims re-indexed by
    # id(expr) for the linearizer, so they are NOT listed as separate owners here
    # (doing so would double-count every column as a spurious conflict).
    "univariate_relaxations": ("expr_id", "aux_col"),
    "composite_relaxations": ("expr_id", "aux_col"),
    "composite_multivar_relaxations": ("expr_id", "aux_col"),
    "univariate_square_relaxations": ("base_col", "aux_col"),
}
_DICT_KEY_OWNERS = ("bilinear", "monomial", "trilinear", "multilinear", "fractional_power")


@dataclasses.dataclass(frozen=True)
class AuditReport:
    """Ownership summary of one ``build_milp_relaxation`` call.

    ``owner_by_col`` maps each aux column to the owner family that produced it;
    ``columns`` is the per-family set of aux columns; ``defer_fires`` counts each
    defer-predicate site that fired during the build (empty unless the build ran
    inside a :func:`defer_audit` context with the predicates wired). ``conflicts``
    lists any aux column claimed by more than one family (must be empty — the R2.5
    exactly-one-owner invariant).
    """

    owner_by_col: dict[int, str]
    columns: dict[str, frozenset[int]]
    defer_fires: dict[str, int]
    conflicts: dict[int, tuple[str, ...]]
    fingerprint: str
    # Raw-expression ids the federation actually claimed, per owner family (from
    # the ``expr_id`` field of the emitted relaxation records). These are the
    # ids the R1.2 boundary census maps onto canonical atoms.
    claimed_expr_ids: dict[str, frozenset[int]] = dataclasses.field(default_factory=dict)

    @property
    def n_claims(self) -> int:
        return len(self.owner_by_col)

    def owners(self) -> frozenset[str]:
        return frozenset(self.columns)

    def all_claimed_expr_ids(self) -> frozenset[int]:
        out: set[int] = set()
        for ids in self.claimed_expr_ids.values():
            out |= ids
        return frozenset(out)


def _column_claims(info: dict) -> list[tuple[int, str]]:
    """Extract ``(aux_col, owner_family)`` pairs from a build's varmap."""
    claims: list[tuple[int, str]] = []
    for key, (_id_field, col_field) in _LIST_OWNERS.items():
        for rel in info.get(key, []) or []:
            col = getattr(rel, col_field, None)
            if col is not None:
                claims.append((int(col), key))
    for key in _DICT_KEY_OWNERS:
        for col in (info.get(key, {}) or {}).values():
            claims.append((int(col), key))
    return claims


def _expr_id_claims(info: dict) -> dict[str, frozenset[int]]:
    """Per-family set of raw-expression ids the federation claimed.

    Only the list-owner relaxation records carry an ``expr_id`` (the id of the
    original ``Expression`` node claimed); the structurally-keyed product side is
    keyed by (index) tuples, not expr ids, so it is omitted here.
    """
    out: dict[str, set[int]] = {}
    for key, (id_field, _col_field) in _LIST_OWNERS.items():
        # Only families whose id field is a genuine expression id carry a claim
        # on a raw Expression node; univariate_square keys by base column, not
        # expr id, so it is excluded from the expr-id census.
        if id_field != "expr_id":
            continue
        for rel in info.get(key, []) or []:
            eid = getattr(rel, id_field, None)
            if isinstance(eid, int):
                out.setdefault(key, set()).add(eid)
    return {k: frozenset(v) for k, v in out.items()}


def audit_build(model: Any, terms: Any = None, disc: Any = None) -> AuditReport:
    """Build ``model``'s relaxation and summarise which owner claimed each column.

    Read-only: it inspects the returned varmap and never alters the build. Runs the
    build inside a :func:`defer_audit` context so any wired defer-predicate sites
    are counted.
    """
    with defer_audit() as fires:
        relax, info = _build(model, terms, disc)

    owner_by_col: dict[int, str] = {}
    columns: dict[str, set[int]] = {}
    conflicts: dict[int, list[str]] = {}
    for col, owner in _column_claims(info):
        columns.setdefault(owner, set()).add(col)
        if col in owner_by_col and owner_by_col[col] != owner:
            conflicts.setdefault(col, [owner_by_col[col]]).append(owner)
        else:
            owner_by_col[col] = owner

    return AuditReport(
        owner_by_col=dict(owner_by_col),
        columns={k: frozenset(v) for k, v in columns.items()},
        defer_fires=dict(fires),
        conflicts={c: tuple(v) for c, v in conflicts.items()},
        claimed_expr_ids=_expr_id_claims(info),
        fingerprint=relaxation_fingerprint(relax),
    )


# --------------------------------------------------------------------------- #
# Defer-fire counter (mechanism only; wired into the predicates by R1.2/R2)
# --------------------------------------------------------------------------- #
_DEFER_SINK: contextvars.ContextVar[Optional[Counter]] = contextvars.ContextVar(
    "discopt_defer_sink", default=None
)


@contextlib.contextmanager
def defer_audit():
    """Activate defer-fire counting for the duration of the context.

    Yields a :class:`collections.Counter` that accumulates one increment per
    :func:`note_defer` call made while the context is active. Nested contexts are
    independent (the innermost sink wins). Outside any context, :func:`note_defer`
    is a no-op — so a ``note_defer`` call wired into a defer predicate cannot
    change that predicate's behaviour or the built relaxation.
    """
    sink: Counter = Counter()
    token = _DEFER_SINK.set(sink)
    try:
        yield sink
    finally:
        _DEFER_SINK.reset(token)


def note_defer(site: str) -> None:
    """Record that defer-predicate ``site`` fired (no-op unless auditing)."""
    sink = _DEFER_SINK.get()
    if sink is not None:
        sink[site] += 1
