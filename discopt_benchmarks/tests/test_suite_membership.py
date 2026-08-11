"""A suite must run the instances it declares, and know their optima.

Both halves of this file are about a benchmark that reports a number for the
wrong thing:

* ``--suite global50`` ran **119** instances -- every ``.nl`` in the local
  corpus -- and labelled the result with the panel's name. The default loader
  (``_load_minlplib_instances``, used whenever ``--fetch``/``--use-cache`` is
  not passed) read only ``instance_list_inline`` and ignored ``instance_list``,
  the list *file* form that every ``[gates.cert*]`` panel is defined with.
  ``_load_minlplib_from_cache`` honoured both, so which loader ran silently
  decided what the suite meant.
* On that same path the oracle was a 34-entry literal covering **21 of the 50**
  panel names. An instance with ``best_known_objective=None`` can never be
  counted incorrect, so ``incorrect_count`` -- the gate CLAUDE.md gives zero
  slack -- read 0 largely by not being evaluated.

These tests exercise the real loader against the real config; they are not a
transcription of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import tomllib

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import run_benchmarks as rb  # noqa: E402

_CONFIG = _BENCH / "config" / "benchmarks.toml"
_PANEL_LIST = _BENCH / "config" / "baron_global50.txt"

pytestmark = pytest.mark.skipif(
    not _CONFIG.exists() or not _PANEL_LIST.exists(),
    reason="needs the benchmark config and the global50 instance list",
)


def _suite(name: str) -> dict:
    with open(_CONFIG, "rb") as f:
        return tomllib.load(f)["suites"][name]


def _panel_names() -> list[str]:
    return [
        ln.strip()
        for ln in _PANEL_LIST.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def test_global50_resolves_to_the_panel_not_the_whole_corpus():
    """The bug, stated as a number: 50 declared, 119 run.

    Asserted as a bound rather than an exact count because how many of the 50
    have a local ``.nl`` is a property of the vendored corpus, not of the
    selection logic. The corpus is larger than the panel, so "no more than the
    panel" is what distinguishes a working filter from an absent one.
    """
    declared = _panel_names()
    assert len(declared) == 50, "the panel list itself changed; update this test"

    instances, _ = rb._load_minlplib_instances(_suite("global50"))
    got = {i.name for i in instances}

    assert got, "the panel resolved to nothing -- fail-closed fired unexpectedly"
    assert got <= set(declared), (
        f"loader returned {len(got - set(declared))} instances that are NOT in "
        f"the panel list, e.g. {sorted(got - set(declared))[:5]}"
    )
    assert len(got) >= 40, (
        f"only {len(got)} of 50 panel instances resolved locally; the corpus or the list moved"
    )


def test_a_missing_list_file_filters_to_nothing_rather_than_everything():
    """Fail closed. A typo in a suite table must not silently run the corpus.

    This is the property that makes the assertion above trustworthy: without
    it, "filter applied" and "filter file unreadable" look identical from the
    outside, and the unreadable case is the dangerous one.
    """
    instances, _ = rb._load_minlplib_instances({"instance_list": "config/does-not-exist.txt"})
    assert instances == []


def test_no_suite_config_still_loads_the_whole_corpus():
    """The other direction: absence of a list means "no filter", not "nothing".

    ``_read_instance_list(None)`` returns None (no filter) while a missing file
    returns an empty set. Conflating them would either break the unfiltered
    suites or resurrect the bug.
    """
    instances, _ = rb._load_minlplib_instances(None)
    assert len(instances) > 50


def test_the_panel_has_an_oracle_for_nearly_every_instance():
    """A correctness gate over instances with no known optimum is vacuous.

    Two panel instances (casctanks, tspn05) have no vendored optimum; that is a
    real gap and the bound below admits exactly it. What it rejects is the prior
    state, where 29 of the 50 had none.
    """
    instances, optima = rb._load_minlplib_instances(_suite("global50"))
    assert instances, "no instances -- the assertion below would be vacuous"

    with_oracle = [i for i in instances if optima.get(i.name) is not None]
    ratio = len(with_oracle) / len(instances)
    assert ratio >= 0.90, (
        f"only {len(with_oracle)}/{len(instances)} panel instances have a known "
        f"optimum ({ratio:.0%}); incorrect_count cannot fire on the rest"
    )


def test_vendored_optima_agree_with_the_hardcoded_map():
    """Merging the two oracles must widen coverage, not re-baseline verdicts.

    If these ever disagree, one of them is wrong and a correctness verdict
    silently changes meaning depending on which loader ran -- exactly the class
    of thing this file exists to catch.
    """
    cert = rb._load_cert_optima()
    assert cert, "cert-optima.json did not load; the comparison would be vacuous"

    _, merged = rb._load_minlplib_instances(_suite("global50"))
    compared = 0
    for name, value in cert.items():
        if name in merged:
            compared += 1
            assert merged[name] == pytest.approx(value, abs=1e-6, rel=1e-6), (
                f"{name}: vendored optimum {value} disagrees with the merged map's {merged[name]}"
            )
    assert compared >= 40, f"only {compared} optima compared; expected the panel's ~48"
