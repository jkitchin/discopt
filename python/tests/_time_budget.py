"""Per-test wall-clock budget for the CI fast lane (``--time-budget SECONDS``).

The fast lane was trimmed repeatedly (#390/#717, then the #1877 sharding) and grew
back each time, because every new regression test that runs a real solve lands in
it unmarked: the 465 tests that took >= 2 s carried 72% of the lane's time (#1877
profile). This makes the cost visible where it is added. With the option set, a
test whose **call phase** runs longer than the budget fails, unless its node id is
in ``time_budget_allowlist.json``.

The allowlist grandfathers the tests that were already slow when the budget was
introduced, each with its measured call time. It is meant to shrink. An entry is
removed when its test is sped up or marked ``slow``, and the session summary lists
allowlisted tests that ran well under the budget. Adding an entry is allowed, but
it is a deliberate, reviewed change to a data file, not something that happens by
default.

Only the call phase is timed: module/session fixtures are shared by many tests, so
charging their setup to whichever test happens to run first would make the verdict
depend on scheduling order. Off unless ``--time-budget`` is given, so local runs,
the coverage lane (coverage tracing inflates times 1.5-3x) and the correctness lanes
are unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ALLOWLIST_PATH = Path(__file__).with_name("time_budget_allowlist.json")

#: An allowlisted test that ran in under budget / RETIRE_FACTOR is reported as a
#: candidate for removal; the same factor sets the margin when the list is built.
RETIRE_FACTOR = 2.5


def load_allowlist(path: Path = ALLOWLIST_PATH) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def addoption(parser) -> None:
    group = parser.getgroup("time budget")
    group.addoption(
        "--time-budget",
        type=float,
        default=None,
        help="fail a test whose call phase exceeds SECONDS unless allowlisted "
        "(python/tests/time_budget_allowlist.json)",
    )
    group.addoption(
        "--time-budget-allowlist",
        default=str(ALLOWLIST_PATH),
        help="allowlist JSON (node id -> measured seconds) for --time-budget",
    )


class TimeBudgetPlugin:
    def __init__(self, budget: float, allowlist: dict[str, float]) -> None:
        if not budget > 0:
            raise ValueError(f"--time-budget must be positive, got {budget}")
        self.budget = budget
        self.allowlist = allowlist
        self.over: list[tuple[str, float]] = []
        self.retirable: list[tuple[str, float, float]] = []

    # makereport runs where the test runs (an xdist worker); the verdict travels to
    # the controller on the report itself, as user_properties, and is collected
    # there in logreport for the summary.
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        if report.when != "call" or not report.passed:
            return
        took = report.duration
        recorded = self.allowlist.get(item.nodeid)
        if recorded is not None:
            if took < self.budget / RETIRE_FACTOR:
                report.user_properties.append(("time_budget_retirable", recorded))
            return
        if took > self.budget:
            report.user_properties.append(("time_budget_over", took))
            report.outcome = "failed"
            report.longrepr = (
                f"{item.nodeid} took {took:.1f} s, over the fast-lane budget of "
                f"{self.budget:g} s (--time-budget). Speed it up, mark it "
                "@pytest.mark.slow (it then runs in the on-demand lane, with "
                "@pytest.mark.correctness if it validates a known optimum), or add "
                f'"{item.nodeid}": {took:.2f} to python/tests/time_budget_allowlist.json '
                "and justify the entry in the PR."
            )

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        props = dict(report.user_properties)
        if "time_budget_over" in props:
            self.over.append((report.nodeid, props["time_budget_over"]))
        if "time_budget_retirable" in props:
            self.retirable.append((report.nodeid, props["time_budget_retirable"], report.duration))

    def pytest_terminal_summary(self, terminalreporter):
        tr = terminalreporter
        tr.write_sep("-", f"fast-lane time budget: {self.budget:g} s per test (call phase)")
        tr.write_line(
            f"{len(self.over)} over budget, {len(self.allowlist)} allowlisted, "
            f"{len(self.retirable)} allowlisted ran under {self.budget / RETIRE_FACTOR:g} s"
        )
        for nodeid, took in self.over:
            tr.write_line(f"  OVER  {took:6.1f} s  {nodeid}")
        for nodeid, recorded, took in sorted(self.retirable, key=lambda r: r[2]):
            tr.write_line(f"  could leave the allowlist ({recorded:.1f} -> {took:.1f} s)  {nodeid}")


def configure(config) -> None:
    budget = config.getoption("--time-budget")
    if budget is None:
        return
    allowlist = load_allowlist(Path(config.getoption("--time-budget-allowlist")))
    config.pluginmanager.register(TimeBudgetPlugin(budget, allowlist), "discopt-time-budget")
