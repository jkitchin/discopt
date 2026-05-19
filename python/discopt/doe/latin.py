"""Latin-square family designs for ANOVA on additive main effects.

These are classical Fisher / Cochran designs for experiments where the
goal is to estimate **additive main effects** of 1 treatment factor plus
1--3 blocking factors with k levels each, using only k**2 runs (instead
of the k**4 a full factorial would need).

* **Randomized complete block** (2 factors, 1 treatment + 1 block): k**2
  runs as a full factorial in k x k.
* **Latin square** (3 factors): k**2 runs, each treatment level appears
  once per row and once per column.
* **Graeco-Latin square** (4 factors): k**2 runs, superimposed pair of
  orthogonal Latin squares.
* **Hyper-Graeco-Latin square** (5 factors): k**2 runs, three mutually
  orthogonal Latin squares (MOLS).

The MOLS construction used here is the cyclic one over Z_k: for prime k
the squares ``L_a(i, j) = (i + a*j) mod k`` for ``a = 1..k-1`` are
pairwise orthogonal. ``k = 2`` admits no Latin pair (only one square),
and ``k = 6`` is the famous Euler exception — both raise.

Replication of the whole square is supported via the ``replicates``
argument; randomization (row/column permutation + run-order shuffle) is
applied per replicate.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

LevelSpec = Mapping[str, Sequence[object]]


@dataclass(frozen=True)
class LatinDesign:
    """A randomized Latin-family design ready for execution.

    Attributes
    ----------
    factors : tuple of str
        Factor names in the order they were specified by the caller.
    levels : tuple of tuple
        Per-factor level lists in the same order as ``factors``.
    rows : list of dict
        One dict per run with all factor values, a ``replicate`` index
        (0-based), and a ``run_order`` index (0-based, shuffled).
    family : str
        ``"rcb"``, ``"latin-square"``, ``"graeco-latin"``, or
        ``"hyper-graeco-latin"`` — which design was produced.
    """

    factors: tuple[str, ...]
    levels: tuple[tuple[object, ...], ...]
    rows: list[dict[str, object]]
    family: str

    def __len__(self) -> int:
        return len(self.rows)


def _mols_prime_power(k: int) -> list[list[list[int]]]:
    """Return k-1 mutually orthogonal Latin squares of order ``k``.

    Only handles k where Z_k is a field (k prime) or where we ship a
    hardcoded MOLS table (k = 4). For other k we fall back to the
    cyclic Z_k construction which is correct for prime k but only
    produces non-orthogonal squares otherwise — the caller must check.
    """
    if k <= 1:
        raise ValueError(f"order must be >= 2, got {k}")
    if k == 2:
        return [[[0, 1], [1, 0]]]
    if k == 4:
        # MOLS(4) over GF(4) -- three mutually orthogonal Latin squares.
        # Verified pairwise orthogonal (every (a, b) pair appears exactly once).
        return [
            [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
            [[0, 2, 3, 1], [1, 3, 2, 0], [2, 0, 1, 3], [3, 1, 0, 2]],
            [[0, 3, 1, 2], [1, 2, 0, 3], [2, 1, 3, 0], [3, 0, 2, 1]],
        ]
    if _is_prime(k):
        return [[[(i + a * j) % k for j in range(k)] for i in range(k)] for a in range(1, k)]
    # Composite non-prime-power: just return the cyclic square.
    # MOLS may not exist; caller validates.
    return [[[(i + j) % k for j in range(k)] for i in range(k)]]


def _is_prime(k: int) -> bool:
    if k < 2:
        return False
    if k % 2 == 0:
        return k == 2
    for d in range(3, int(math.isqrt(k)) + 1, 2):
        if k % d == 0:
            return False
    return True


def _check_orthogonal(a: list[list[int]], b: list[list[int]]) -> bool:
    k = len(a)
    seen: set[tuple[int, int]] = set()
    for i in range(k):
        for j in range(k):
            seen.add((a[i][j], b[i][j]))
    return len(seen) == k * k


def latin_square(k: int, *, seed: int | None = None) -> list[list[int]]:
    """A randomized k x k Latin square (cells in 0..k-1).

    Permutes rows, then columns of the canonical cyclic square. The
    result is uniform over a non-trivial subset of all Latin squares of
    order k — good enough for randomization purposes in DoE.
    """
    if k < 2:
        raise ValueError(f"latin_square requires k >= 2, got {k}")
    base = _mols_prime_power(k)[0]
    return _randomize_square(base, seed)


def graeco_latin_square(
    k: int, *, seed: int | None = None
) -> tuple[list[list[int]], list[list[int]]]:
    """Two orthogonal k x k Latin squares (Graeco-Latin).

    Raises ``ValueError`` for k = 2 and k = 6 (no MOLS pair exists).
    """
    if k == 2:
        raise ValueError("Graeco-Latin square does not exist for k = 2")
    if k == 6:
        raise ValueError("Graeco-Latin square does not exist for k = 6 (Euler)")
    squares = _build_squares(k, 2, seed)
    return squares[0], squares[1]


def hyper_graeco_latin_square(
    k: int, *, seed: int | None = None
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Three mutually orthogonal Latin squares (5 factors total)."""
    if k < 4:
        raise ValueError("hyper-Graeco-Latin requires k >= 4")
    if k == 6:
        raise ValueError("hyper-Graeco-Latin does not exist for k = 6")
    squares = _build_squares(k, 3, seed)
    return squares[0], squares[1], squares[2]


def _randomize_square(square: list[list[int]], seed: int | None) -> list[list[int]]:
    """Row-permute, then column-permute, then relabel cell values."""
    rng = random.Random(seed)
    k = len(square)
    row_perm = list(range(k))
    col_perm = list(range(k))
    rng.shuffle(row_perm)
    rng.shuffle(col_perm)
    val_perm = list(range(k))
    rng.shuffle(val_perm)
    return [[val_perm[square[row_perm[i]][col_perm[j]]] for j in range(k)] for i in range(k)]


def _build_squares(k: int, n_squares: int, seed: int | None) -> list[list[list[int]]]:
    """Randomize a set of MOLS using a single shared randomization.

    Returns ``n_squares`` Latin squares whose pairwise orthogonality is
    preserved by applying the same row/column permutation to all of them
    and an independent value relabelling to each.
    """
    squares = _mols_prime_power(k)
    if n_squares == 1:
        return [_randomize_square(squares[0], seed)]
    if n_squares > len(squares):
        raise ValueError(f"k = {k} provides {len(squares)} MOLS, need {n_squares}")
    for i in range(n_squares):
        for j in range(i + 1, n_squares):
            if not _check_orthogonal(squares[i], squares[j]):
                raise ValueError(f"MOLS for k = {k} are not pairwise orthogonal")
    rng = random.Random(seed)
    row_perm = list(range(k))
    col_perm = list(range(k))
    rng.shuffle(row_perm)
    rng.shuffle(col_perm)
    val_perms: list[list[int]] = []
    for _ in range(n_squares):
        p = list(range(k))
        rng.shuffle(p)
        val_perms.append(p)
    out: list[list[list[int]]] = []
    for s_idx in range(n_squares):
        sq = squares[s_idx]
        vp = val_perms[s_idx]
        out.append([[vp[sq[row_perm[i]][col_perm[j]]] for j in range(k)] for i in range(k)])
    return out


def latin_square_design(
    factors: LevelSpec,
    *,
    replicates: int = 1,
    seed: int | None = None,
) -> LatinDesign:
    """Build a randomized Latin-family design from per-factor level lists.

    The number of factors determines the design family:

    * 2 factors -> randomized complete block (full k x k factorial)
    * 3 factors -> Latin square
    * 4 factors -> Graeco-Latin square
    * 5 factors -> hyper-Graeco-Latin square

    Every factor must have the same number of levels ``k``. ``k = 6`` is
    rejected for 4+ factor designs (Euler's MOLS exception).

    Parameters
    ----------
    factors : mapping name -> sequence of levels
        Factor levels. Order is preserved (Python 3.7+ dict ordering).
    replicates : int, default 1
        Number of independent replications of the whole square.
        Each replicate gets its own randomization.
    seed : int, optional
        Reproducible randomization seed.

    Returns
    -------
    LatinDesign
        Rows are tagged with ``replicate`` (0..replicates-1) and
        ``run_order`` (0..total_runs-1). Run order is shuffled across
        the whole experiment.
    """
    if not factors:
        raise ValueError("at least one factor required")
    names = tuple(factors.keys())
    levels = tuple(tuple(factors[n]) for n in names)
    n_factors = len(names)
    k_set = {len(lv) for lv in levels}
    if len(k_set) != 1:
        raise ValueError(f"all factors must have the same number of levels, got {k_set}")
    k = k_set.pop()
    if k < 2:
        raise ValueError(f"each factor needs at least 2 levels, got {k}")
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")

    if n_factors == 1:
        family = "one-way"
    elif n_factors == 2:
        family = "rcb"
    elif n_factors == 3:
        family = "latin-square"
    elif n_factors == 4:
        family = "graeco-latin"
    elif n_factors == 5:
        family = "hyper-graeco-latin"
    else:
        raise ValueError(f"latin_square_design supports up to 5 factors, got {n_factors}")

    rows: list[dict[str, object]] = []
    rng = random.Random(seed)
    for r in range(replicates):
        rep_seed = rng.randrange(0, 2**31 - 1)
        rep_rows = _one_replicate(names, levels, k, family, rep_seed)
        for row in rep_rows:
            row["replicate"] = r
        rows.extend(rep_rows)

    order = list(range(len(rows)))
    rng.shuffle(order)
    for new_idx, original_idx in enumerate(order):
        rows[original_idx]["run_order"] = new_idx
    rows.sort(key=lambda d: d["run_order"])
    return LatinDesign(factors=names, levels=levels, rows=rows, family=family)


def _one_replicate(
    names: tuple[str, ...],
    levels: tuple[tuple[object, ...], ...],
    k: int,
    family: str,
    seed: int,
) -> list[dict[str, object]]:
    if family == "one-way":
        return [{names[0]: levels[0][i]} for i in range(k)]
    if family == "rcb":
        return [
            {names[0]: levels[0][i], names[1]: levels[1][j]} for i in range(k) for j in range(k)
        ]
    n_treatments = {"latin-square": 1, "graeco-latin": 2, "hyper-graeco-latin": 3}[family]
    if family != "rcb" and k == 6 and n_treatments >= 2:
        raise ValueError("Graeco-Latin / hyper-Graeco-Latin do not exist for k = 6")
    squares = _build_squares(k, n_treatments, seed)
    rows = []
    for i in range(k):
        for j in range(k):
            row: dict[str, object] = {names[0]: levels[0][i], names[1]: levels[1][j]}
            for t, sq in enumerate(squares):
                row[names[2 + t]] = levels[2 + t][sq[i][j]]
            rows.append(row)
    return rows


__all__ = [
    "LatinDesign",
    "graeco_latin_square",
    "hyper_graeco_latin_square",
    "latin_square",
    "latin_square_design",
]
