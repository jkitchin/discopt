"""Native logic-based GDP algorithms (#1775 G3/G4).

``base`` holds the shared loop every native algorithm runs on: the deadline,
the LB/UB bookkeeping and the status rules of spec §5.4.
"""

from discopt.gdp.algorithms.base import Deadline, GDPAlgorithm

__all__ = ["Deadline", "GDPAlgorithm"]
