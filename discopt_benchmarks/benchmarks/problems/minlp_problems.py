"""MINLP benchmark problems from .nl files and programmatic models."""

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

# Smoke instances loaded from .nl files
_SMOKE_NL = ["ex1221", "nvs03", "st_e13", "gear4"]


def _make_nl_builder(nl_path: Path):
    """Return a build function that loads a model from an .nl file."""

    def _build():
        import discopt.modeling as dm

        return dm.from_nl(str(nl_path))

    return _build


# ---------------------------------------------------------------------------
# Register smoke .nl instances
# ---------------------------------------------------------------------------

for _name in _SMOKE_NL:
    _nl_path = _NL_DIR / f"{_name}.nl"
    if _nl_path.exists():
        register(
            TestProblem(
                name=f"minlp_{_name}",
                category="minlp",
                level="smoke",
                build_fn=_make_nl_builder(_nl_path),
                known_optimum=_KNOWN_OPTIMA[_name],
                applicable_solvers=_APPLICABLE,
                source="nl_file",
                tags=["minlplib"],
            )
        )


# ---------------------------------------------------------------------------
# Programmatic smoke: simple_minlp
# ---------------------------------------------------------------------------


def _build_simple_minlp():
    """min x^2 + 3*y, x in [0,5], y binary, s.t. x + y >= 1.

    y=0 => x>=1, obj=1.  y=1 => x=0, obj=3. Opt=1.0.
    """
    import discopt.modeling as dm

    m = dm.Model("simple_minlp")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.binary("y")
    m.minimize(x**2 + 3 * y)
    m.subject_to(x + y >= 1.0)
    return m


register(
    TestProblem(
        name="minlp_simple",
        category="minlp",
        level="smoke",
        build_fn=_build_simple_minlp,
        known_optimum=1.0,
        applicable_solvers=_APPLICABLE,
        n_vars=2,
        n_constraints=1,
        source="programmatic",
        tags=["simple"],
    )
)


# ---------------------------------------------------------------------------
# Full: scan .nl directory and register all instances with known optima
# ---------------------------------------------------------------------------


def _register_full_nl_instances():
    """Register all .nl files with known optima as full-level problems.

    Instances already registered as smoke are included in full level
    via the get_problems(level="full") logic in base.py (which
    returns both smoke and full), so we register them as "full"
    only if they are NOT in the smoke set.
    """
    if not _NL_DIR.is_dir():
        return

    smoke_set = set(_SMOKE_NL)
    for nl_file in sorted(_NL_DIR.glob("*.nl")):
        instance_name = nl_file.stem
        if instance_name in smoke_set:
            continue  # Already registered as smoke

        optimum = _KNOWN_OPTIMA.get(instance_name)
        if optimum is None:
            # Skip instances without known optima
            continue

        register(
            TestProblem(
                name=f"minlp_{instance_name}",
                category="minlp",
                level="full",
                build_fn=_make_nl_builder(nl_file),
                known_optimum=optimum,
                applicable_solvers=_APPLICABLE,
                source="nl_file",
                tags=["minlplib"],
            )
        )


_register_full_nl_instances()
