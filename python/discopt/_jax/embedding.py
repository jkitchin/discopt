"""Embedded-binary helpers for SOS2-compatible convex-hull formulations.

The helper here mirrors the Gray-code embedding idea used in Alpine.jl for
piecewise convex-hull relaxations: replace one binary per interval with a
logarithmic number of selector binaries while preserving the SOS2 adjacency
pattern on the lambda variables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingMap:
    """Structured SOS2 embedding metadata."""

    encoding: str
    bit_count: int
    codes: tuple[tuple[int, ...], ...]
    positive_sets: tuple[tuple[int, ...], ...]
    negative_sets: tuple[tuple[int, ...], ...]


def _binary_code(index: int, width: int) -> tuple[int, ...]:
    """Return the fixed-width binary representation of ``index``."""
    return tuple((index >> shift) & 1 for shift in range(width - 1, -1, -1))


def _gray_code(index: int, width: int) -> tuple[int, ...]:
    """Return the fixed-width Gray-code representation of ``index``."""
    return _binary_code(index ^ (index >> 1), width)


def _adjacent_intervals(lambda_idx: int, lambda_count: int) -> list[int]:
    """Return the interval indices adjacent to breakpoint ``lambda_idx``."""
    partition_count = lambda_count - 1
    adjacent: list[int] = []
    if lambda_idx > 0:
        adjacent.append(lambda_idx - 1)
    if lambda_idx < partition_count:
        adjacent.append(lambda_idx)
    return adjacent


def _is_sos2_compatible(codes: list[tuple[int, ...]]) -> bool:
    """Return True when adjacent interval codes differ in exactly one bit."""
    for first, second in zip(codes, codes[1:]):
        if sum(abs(a - b) for a, b in zip(first, second)) != 1:
            return False
    return True


def build_embedding_map(
    lambda_count: int,
    encoding: str = "gray",
) -> EmbeddingMap:
    """Build an SOS2-compatible embedding map for ``lambda_count`` breakpoints.

    Parameters
    ----------
    lambda_count : int
        Number of lambda variables, equal to the number of breakpoints.
    encoding : str, default ``"gray"``
        Interval encoding scheme. ``"gray"`` is SOS2-compatible for arbitrary
        partition counts. ``"binary"`` is only SOS2-compatible for exactly two
        partitions (``lambda_count == 3``) and is mainly kept to exercise the
        validation path.
    """
    if lambda_count < 2:
        raise ValueError("lambda_count must be at least 2")
    if encoding not in {"gray", "binary"}:
        raise ValueError(
            f"Unsupported embedding encoding: {encoding!r}. Choose from 'gray' or 'binary'."
        )

    partition_count = lambda_count - 1
    if partition_count == 1:
        return EmbeddingMap(
            encoding=encoding,
            bit_count=0,
            codes=((),),
            positive_sets=(),
            negative_sets=(),
        )

    bit_count = int(math.ceil(math.log2(partition_count)))

    if encoding == "gray":
        codes = [_gray_code(i, bit_count) for i in range(partition_count)]
    else:
        codes = [_binary_code(i, bit_count) for i in range(partition_count)]

    if not _is_sos2_compatible(codes):
        if encoding == "binary":
            raise ValueError(
                "Embedding encoding 'binary' is not SOS2-compatible for "
                f"{partition_count} partitions; it only works for exactly 2 "
                "partitions (lambda_count=3). Use 'gray' for larger partition "
                "counts."
            )
        raise ValueError(
            f"Embedding encoding {encoding!r} is not SOS2-compatible for "
            f"{partition_count} partitions."
        )

    positive_sets: list[tuple[int, ...]] = []
    negative_sets: list[tuple[int, ...]] = []

    for bit_idx in range(bit_count):
        positive: list[int] = []
        negative: list[int] = []
        for lambda_idx in range(lambda_count):
            adjacent = _adjacent_intervals(lambda_idx, lambda_count)
            adjacent_codes = [codes[idx] for idx in adjacent]
            if all(code[bit_idx] == 1 for code in adjacent_codes):
                positive.append(lambda_idx)
            if all(code[bit_idx] == 0 for code in adjacent_codes):
                negative.append(lambda_idx)
        positive_sets.append(tuple(positive))
        negative_sets.append(tuple(negative))

    return EmbeddingMap(
        encoding=encoding,
        bit_count=bit_count,
        codes=tuple(codes),
        positive_sets=tuple(positive_sets),
        negative_sets=tuple(negative_sets),
    )
