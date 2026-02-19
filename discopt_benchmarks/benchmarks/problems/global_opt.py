"""Global optimization benchmark problems with gap-focused metrics."""

from __future__ import annotations

from pathlib import Path

from benchmarks.problems.base import TestProblem, register

_APPLICABLE = ["ipm", "ripopt", "ipopt"]

_NL_DIR = Path(__file__).parent.parent.parent.parent / "python" / "tests" / "data" / "minlplib_nl"

_KNOWN_OPTIMA = {
    "ex1221": 7.66718007,
    "ex1225": 31.0,
    "ex1226": -17.0,
    "st_e13": 2.0,
    "st_e15": 7.66718007,
    "st_e27": 2.0,
    "st_e38": 7197.72714900,
    "st_e40": 30.41421350,
    "nvs01": 12.46966882,
    "nvs02": 5.96418452,
    "nvs03": 16.0,
    "nvs04": 0.72,
    "nvs05": 5.47093411,
    "nvs06": 1.77031250,
    "nvs07": 4.0,
    "nvs08": 23.44972735,
    "nvs10": -310.80,
    "nvs11": -431.0,
    "nvs12": -481.20,
    "nvs14": -40358.15477,
    "nvs15": 1.0,
    "nvs16": 0.70312500,
    "nvs21": -5.68478250,
    "prob03": 10.0,
    "prob06": 1.17712434,
    "prob10": 3.44550379,
    "gear": 0.0,
    "gear3": 0.0,
    "gear4": 1.64342847,
    "chance": 29.89437816,
    "dispatch": 3155.28792700,
    "meanvar": 5.24339907,
    "alan": 2.9250,
}

# Smoke instances for global optimization
_SMOKE_NAMES = ["ex1221", "nvs01", "alan", "dispatch", "gear4"]


def _make_nl_builder(nl_path: Path):
    """Return a build function that loads a model from an .nl file."""

    def _build():
        import discopt.modeling as dm

        return dm.from_nl(str(nl_path))

    return _build


# ---------------------------------------------------------------------------
# Register smoke instances
# ---------------------------------------------------------------------------

for _name in _SMOKE_NAMES:
    _nl_path = _NL_DIR / f"{_name}.nl"
    if _nl_path.exists():
        register(
            TestProblem(
                name=f"global_{_name}",
                category="global_opt",
                level="smoke",
                build_fn=_make_nl_builder(_nl_path),
                known_optimum=_KNOWN_OPTIMA[_name],
                applicable_solvers=_APPLICABLE,
                source="nl_file",
                tags=["gap_focused", "minlplib"],
            )
        )


# ---------------------------------------------------------------------------
# Full: all instances with known optima (excluding smoke duplicates)
# ---------------------------------------------------------------------------


def _register_full_global_instances():
    """Register all .nl instances with known optima as full problems.

    Smoke instances are already included by get_problems(level="full")
    so we only register non-smoke instances here.
    """
    if not _NL_DIR.is_dir():
        return

    smoke_set = set(_SMOKE_NAMES)
    for nl_file in sorted(_NL_DIR.glob("*.nl")):
        instance_name = nl_file.stem
        if instance_name in smoke_set:
            continue

        optimum = _KNOWN_OPTIMA.get(instance_name)
        if optimum is None:
            continue

        register(
            TestProblem(
                name=f"global_{instance_name}",
                category="global_opt",
                level="full",
                build_fn=_make_nl_builder(nl_file),
                known_optimum=optimum,
                applicable_solvers=_APPLICABLE,
                source="nl_file",
                tags=["gap_focused", "minlplib"],
            )
        )


_register_full_global_instances()
