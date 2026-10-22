"""``partitions=k`` is refused, not silently ignored (#1362, then #1873).

#1362 found that ``solve(partitions=k)`` was read in exactly one place -- the
relaxation compiler of the McCormick-"nlp" route -- so on every model taking the
default LP-form McCormick route it did nothing, which is how
``docs/notebooks/advanced_features.ipynb`` came to print the same 5 nodes for
``partitions=0`` and ``partitions=4``. The fix then was a warning.

#1873 retired the route itself (docs/dev/nlp-mccormick-route-retirement.md): its
bound never changed a result on the corpus, ``partitions=4`` included. With no
route left to read it, a warning would announce a no-op on every model, so the
option is now refused before any route can take the model.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest


def _lp_route_minlp():
    """The advanced_features model: nonconvex, so it takes the LP route."""
    m = dm.Model("mccormick_demo")
    x = m.continuous("x", lb=0.1, ub=5)
    y = m.continuous("y", lb=0.1, ub=5)
    z = m.binary("z")
    m.minimize(dm.exp(x) * y + z)
    m.subject_to(x + y >= 2)
    m.subject_to(x * y <= 5 * z + 1)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("k", [1, 4, 8])
def test_a_nonzero_partitions_request_is_refused(k):
    with pytest.raises(ValueError, match=rf"partitions={k}.*has been removed"):
        _lp_route_minlp().solve(partitions=k, time_limit=10)


@pytest.mark.smoke
def test_partitions_zero_is_still_accepted_silently():
    """``0`` was always the default, so it stays accepted without comment."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        r = _lp_route_minlp().solve(partitions=0, time_limit=20)
    assert r.status == "optimal"
