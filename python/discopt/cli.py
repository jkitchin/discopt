"""discopt CLI.

Usage:
    discopt about
    discopt test
    discopt convert input.gms output.nl
    discopt install-skills [--project-scope] [--dev] [--force]

Developer-only commands (``lit-scan``, ``adversary``, ``search-arxiv``,
``search-openalex``, ``write-report``) live under ``discopt-dev`` in
:mod:`discopt.dev.cli`.
"""

import argparse
import importlib.metadata
import os
import platform
import shutil
import sys
from pathlib import Path


def _cmd_about(_args):
    import discopt

    version = discopt.__version__

    install_location = os.path.dirname(os.path.abspath(discopt.__file__))

    try:
        meta = importlib.metadata.metadata("discopt")
        pkg_version = meta["Version"]
        summary = meta["Summary"] or ""
        license_text = meta["License"] or ""
    except importlib.metadata.PackageNotFoundError:
        pkg_version = version
        summary = "Hybrid MINLP solver combining Rust and JAX"
        license_text = "EPL-2.0"

    try:
        import discopt._rust

        rust_ext = os.path.abspath(discopt._rust.__file__)
    except (ImportError, AttributeError):
        rust_ext = "not available"

    print(f"discopt {pkg_version}")
    print(f"  Summary:      {summary}")
    print(f"  License:      {license_text}")
    print(f"  Location:     {install_location}")
    print(f"  Rust ext:     {rust_ext}")
    print(f"  Python:       {sys.version}")
    print(f"  Platform:     {platform.platform()}")
    print(f"  Executable:   {sys.executable}")

    deps = ["jax", "jaxlib", "numpy", "scipy"]
    optional_deps = ["cyipopt", "highspy", "litellm", "pycutest", "onnx", "onnxruntime"]
    for name in deps:
        try:
            ver = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            ver = "not installed"
        print(f"  {name}: {ver}")

    print("  Optional:")
    for name in optional_deps:
        try:
            ver = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            ver = "not installed"
        print(f"    {name}: {ver}")


def _cmd_test(_args):
    """Run a quick smoke test to verify the installation works."""
    errors = []
    passed = []

    try:
        import discopt

        passed.append(f"import discopt ({discopt.__version__})")
    except Exception as e:
        errors.append(f"import discopt: {e}")

    try:
        import discopt._rust  # noqa: F811

        passed.append("Rust extension loaded")
    except ImportError as e:
        errors.append(f"Rust extension: {e}")

    try:
        import jax
        import jax.numpy as jnp

        _ = jnp.array([1.0, 2.0])
        passed.append(f"JAX {jax.__version__} (backend: {jax.default_backend()})")
    except Exception as e:
        errors.append(f"JAX: {e}")

    try:
        import discopt

        m = discopt.Model("smoke_test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= 1)
        result = m.solve(verbose=False)
        obj = float(result.objective)
        if abs(obj - 1.0) > 1e-3:
            errors.append(f"Solve sanity: expected objective ~1.0, got {obj}")
        else:
            passed.append(f"Model build + solve (objective={obj:.6f})")
    except Exception as e:
        errors.append(f"Model build + solve: {e}")

    try:
        from discopt._jax.dag_compiler import compile_objective
        from discopt.modeling import Model

        m2 = Model("dag_test")
        x = m2.continuous("x", lb=0, ub=1)
        m2.minimize(x * x)
        _ = compile_objective(m2)
        passed.append("DAG compiler")
    except Exception as e:
        errors.append(f"DAG compiler: {e}")

    for msg in passed:
        print(f"  PASS  {msg}")
    for msg in errors:
        print(f"  FAIL  {msg}")

    total = len(passed) + len(errors)
    print(f"\n{len(passed)}/{total} checks passed.")

    if errors:
        sys.exit(1)


_IMPORT_EXTS = {".gms", ".nl"}
_EXPORT_EXTS = {".gms", ".nl", ".mps", ".lp"}


def _cmd_convert(args):
    """Convert between optimization model file formats."""
    in_path = args.input
    out_path = args.output

    in_ext = os.path.splitext(in_path)[1].lower()
    out_ext = os.path.splitext(out_path)[1].lower()

    if in_ext not in _IMPORT_EXTS:
        print(
            f"Error: unsupported input format '{in_ext}'. "
            f"Supported: {', '.join(sorted(_IMPORT_EXTS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    if out_ext not in _EXPORT_EXTS:
        print(
            f"Error: unsupported output format '{out_ext}'. "
            f"Supported: {', '.join(sorted(_EXPORT_EXTS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    import discopt.modeling as dm

    try:
        if in_ext == ".gms":
            model = dm.from_gams(in_path)
        else:
            model = dm.from_nl(in_path)

        exporters = {
            ".gms": lambda: model.to_gams(out_path),
            ".nl": lambda: model.to_nl(out_path),
            ".mps": lambda: model.to_mps(out_path),
            ".lp": lambda: model.to_lp(out_path),
        }
        exporters[out_ext]()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Converted {in_path} -> {out_path}")


def _cmd_install_skills(args):
    """Install packaged Claude Code skills/agents into the user's .claude/ tree.

    Defaults to ``~/.claude/`` (user scope). Use ``--project-scope`` to
    target the current directory's ``./.claude/`` instead. Copies files by
    default; ``--dev`` symlinks them instead so edits to package data show
    up live (useful with ``pip install -e``).
    """
    from discopt.skills import iter_agents, iter_commands

    if args.project_scope:
        base = Path.cwd() / ".claude"
    else:
        base = Path.home() / ".claude"

    dest_commands = base / "commands"
    dest_agents = base / "agents"
    dest_commands.mkdir(parents=True, exist_ok=True)
    dest_agents.mkdir(parents=True, exist_ok=True)

    def _install_one(src, dest_dir, verb_counts):
        dest = dest_dir / src.name
        exists = dest.exists() or dest.is_symlink()
        if exists and not args.force:
            print(f"  skip  {src.name} (already exists)")
            verb_counts["skip"] += 1
            return
        if exists:
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
        # ``Traversable`` exposes a filesystem path for most real-world
        # installs; ``importlib.resources.as_file`` would materialize a
        # temp copy for zipapps, but we explicitly want the *source* path
        # for --dev symlinks. Resolve via str() -> Path.
        src_path = Path(str(src))
        if args.dev:
            dest.symlink_to(src_path)
            print(f"  link  {src.name}")
            verb_counts["link"] += 1
        else:
            shutil.copy2(src_path, dest)
            print(f"  copy  {src.name}")
            verb_counts["copy"] += 1

    verb_counts = {"copy": 0, "link": 0, "skip": 0}
    n_commands = 0
    for src in iter_commands():
        _install_one(src, dest_commands, verb_counts)
        n_commands += 1
    n_agents = 0
    for src in iter_agents():
        _install_one(src, dest_agents, verb_counts)
        n_agents += 1

    print(f"\nInstalled {n_commands} command(s) and {n_agents} agent(s) into {base}")
    if verb_counts["skip"]:
        print(f"  {verb_counts['skip']} already existed; pass --force to overwrite.")


def main():
    parser = argparse.ArgumentParser(prog="discopt", description="discopt CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_about = subparsers.add_parser("about", help="Show version and installation info")
    p_about.set_defaults(func=_cmd_about)

    p_test = subparsers.add_parser("test", help="Run smoke tests to verify installation")
    p_test.set_defaults(func=_cmd_test)

    p_conv = subparsers.add_parser(
        "convert",
        help="Convert between model formats (.gms, .nl, .mps, .lp)",
    )
    p_conv.add_argument("input", help="Input file path (.gms or .nl)")
    p_conv.add_argument("output", help="Output file path (.gms, .nl, .mps, or .lp)")
    p_conv.set_defaults(func=_cmd_convert)

    p_skills = subparsers.add_parser(
        "install-skills",
        help="Install packaged Claude Code slash commands and agents into ~/.claude/",
    )
    p_skills.add_argument(
        "--project-scope",
        action="store_true",
        help="Install into ./.claude/ (current project) instead of ~/.claude/.",
    )
    p_skills.add_argument(
        "--dev",
        action="store_true",
        help="Symlink package files instead of copying (for in-place edits).",
    )
    p_skills.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files at the destination.",
    )
    p_skills.set_defaults(func=_cmd_install_skills)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
