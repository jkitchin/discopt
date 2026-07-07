"""Feature extraction and instance fingerprinting for learning.

Turns a :class:`~discopt.decomposition.advisor.analyzer.StructureReport` into the
scalar :class:`~discopt.decomposition.learning.record.InstanceFeatures` the
learners consume, and computes a stable fingerprint so re-solves of the same
instance group together in the store.
"""

from __future__ import annotations

import hashlib

from discopt.decomposition.advisor.analyzer import StructureReport
from discopt.decomposition.learning.record import InstanceFeatures


def extract_features(report: StructureReport) -> InstanceFeatures:
    """Build :class:`InstanceFeatures` from a structure report (no extra passes)."""
    return InstanceFeatures(
        num_vars=report.num_vars,
        num_constraints=report.num_constraints,
        num_incidences=report.num_incidences,
        integer_fraction=report.integer_fraction,
        coupling_density=report.coupling_density,
        num_blocks=report.num_blocks,
        blocks_after_integer_projection=report.blocks_after_integer_projection,
        nonlinear=report.model_is_nonlinear,
    )


def fingerprint(report: StructureReport) -> str:
    """Stable structural fingerprint of an instance.

    Hashes the structural summary (sizes, integer fraction, coupling density,
    block counts, nonlinearity) rounded to avoid float jitter. Two instances with
    the same structure share a fingerprint even if variable names differ, so
    re-solves and near-duplicates group together.
    """
    key = "|".join(
        str(x)
        for x in (
            report.num_vars,
            report.num_constraints,
            report.num_incidences,
            round(report.integer_fraction, 4),
            round(report.coupling_density, 4),
            report.num_blocks,
            report.blocks_after_integer_projection,
            int(report.model_is_nonlinear),
        )
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


__all__ = ["extract_features", "fingerprint"]
