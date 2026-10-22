"""discopt CLI.

Usage:
    discopt about
    discopt test
    discopt solve model.nl [--profile NAME] [--time-limit S] [--json] [--sol] ...
    discopt daemon {serve|stop|status}    # warm solve daemon (auto-spawned by solve)
    discopt convert input.gms output.nl
    discopt install-skills [--project-scope] [--dev] [--force]

Developer-only commands (``lit-scan``, ``adversary``, ``search-arxiv``,
``search-openalex``, ``write-report``) live under ``discopt-dev`` in
:mod:`discopt.dev.cli`.

Plugin subcommands
------------------
External packages can add subcommands through the ``"discopt.cli"``
entry-point group. The entry-point *name* is the subcommand name; the value
must resolve to a module exposing ``add_subparser(subparsers) -> None``
(which registers a subparser with exactly that name) and
``run(args) -> int | None``. Example::

    [project.entry-points."discopt.cli"]
    doe = "discopt.doe.cli"

Plugin modules are imported lazily: only for their own subcommand or when
full help is requested, never on the built-in command paths. Built-in names
cannot be shadowed.
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
        # Fallback only when the distribution metadata is unreadable (e.g. a
        # source tree with no installed dist). Keep in sync with the `description`
        # field in pyproject.toml, which is the real source.
        summary = "MINLP solver with a Rust core: spatial branch and bound over convex relaxations"
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

    deps = ["numpy", "scipy", "highspy", "pounce-solver"]
    # jax/jaxlib are the `discopt[jax]` extra (#1873).
    optional_deps = ["jax", "jaxlib", "cyipopt", "litellm", "pycutest", "onnx", "onnxruntime"]
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

    # JAX is the optional `discopt[jax]` extra (#1873): its absence is reported,
    # not counted as a failure, and the JAX-only checks below are skipped.
    from discopt._optional_jax import INSTALL_HINT, jax_available

    skipped = []
    have_jax = jax_available()
    if have_jax:
        try:
            import jax
            import jax.numpy as jnp

            _ = jnp.array([1.0, 2.0])
            passed.append(f"JAX {jax.__version__} (backend: {jax.default_backend()})")
        except Exception as e:
            errors.append(f"JAX: {e}")
    else:
        skipped.append(f"JAX not installed (optional; `{INSTALL_HINT}`)")

    try:
        import discopt

        m = discopt.Model("smoke_test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= 1)
        result = m.solve()
        obj = float(result.objective)
        if abs(obj - 1.0) > 1e-3:
            errors.append(f"Solve sanity: expected objective ~1.0, got {obj}")
        else:
            passed.append(f"Model build + solve (objective={obj:.6f})")
    except Exception as e:
        errors.append(f"Model build + solve: {e}")

    if have_jax:
        try:
            from discopt._relax.dag_compiler import compile_objective
            from discopt.modeling import Model

            m2 = Model("dag_test")
            x = m2.continuous("x", lb=0, ub=1)
            m2.minimize(x * x)
            _ = compile_objective(m2)
            passed.append("DAG compiler")
        except Exception as e:
            errors.append(f"DAG compiler: {e}")
    else:
        skipped.append("JAX DAG compiler (needs JAX)")

    for msg in passed:
        print(f"  PASS  {msg}")
    for msg in skipped:
        print(f"  SKIP  {msg}")
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


def _cmd_gams_register(args):
    """Register discopt as a solver with a GAMS system.

    Writes a ``discopt-gams`` run script and merges a discopt entry into
    ``gamsconfig.yaml`` in the target directory (default ``$HOME/.gams``, which
    GAMS reads automatically), so ``option minlp = discopt;`` dispatches to
    discopt. ``--check`` only runs the gamsapi diagnostic.
    """
    from discopt.gams import check_gamsapi, write_registration

    ok, message = check_gamsapi()
    if args.check:
        print(("OK: " if ok else "PROBLEM: ") + message)
        sys.exit(0 if ok else 1)

    out = Path(args.directory)
    written = write_registration(out)
    verb = {"created": "Created", "merged": "Merged into", "replaced": "Updated"}[written["action"]]
    print(f"{verb} {written['config']}")
    print(f"Run script: {written['script']}")
    if out == Path.home() / ".gams":
        print("\nGAMS reads $HOME/.gams/gamsconfig.yaml automatically. In GAMS:")
    else:
        print(
            f"\nMerge {written['config']} into your GAMS gamsconfig.yaml "
            "($HOME/.gams or the GAMS system dir), then in GAMS:"
        )
    print("    option minlp = discopt;\n    solve m using minlp minimizing z;")
    print(f"\ngamsapi check: {message}")


def _cmd_gams_daemon(args):
    """Control the warm discopt GAMS solver daemon."""
    from discopt.gams import daemon

    return daemon.main([args.action])


def _cmd_gams_verify(args):
    """Run the packaged .gms smoke corpus through GAMS with solver=discopt."""
    from discopt.gams import verify

    sys.exit(verify(gams=args.gams, solver=args.solver, keep=args.keep))


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


def _cmd_daemon(args):
    """Control the warm ``discopt solve`` daemon (serve/stop/status)."""
    from discopt import daemon

    return daemon.main([args.action])


def _coerce_tuning_value(s: str, type_hint):
    """Coerce a ``--tuning key=value`` string by a SolverTuning field's type."""
    name = type_hint if isinstance(type_hint, str) else getattr(type_hint, "__name__", "str")
    if name == "bool":
        low = s.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {s!r}")
    if name == "int":
        return int(s)
    if name == "float":
        return float(s)
    return s


def _parse_tuning(pairs):
    """``["key=val", ...]`` -> dict, validated/coerced against SolverTuning fields."""
    import dataclasses

    from discopt.solver_tuning import SolverTuning

    types = {f.name: f.type for f in dataclasses.fields(SolverTuning)}
    out = {}
    for p in pairs or []:
        if "=" not in p:
            print(f"Error: --tuning expects key=value, got {p!r}", file=sys.stderr)
            sys.exit(2)
        k, v = p.split("=", 1)
        k = k.strip()
        if k not in types:
            print(f"Error: unknown tuning field {k!r}; valid: {sorted(types)}", file=sys.stderr)
            sys.exit(2)
        try:
            out[k] = _coerce_tuning_value(v.strip(), types[k])
        except ValueError as exc:
            print(f"Error: --tuning {k}: {exc}", file=sys.stderr)
            sys.exit(2)
    return out


def _cmd_solve(args):
    """Solve a ``.nl`` model, warm-routed through the solve daemon when available."""
    import json

    nl = os.path.abspath(args.file)
    if not os.path.exists(nl):
        print(f"Error: no such file: {args.file}", file=sys.stderr)
        sys.exit(1)
    if os.path.splitext(nl)[1].lower() != ".nl":
        print(f"Error: discopt solve expects a .nl file, got {args.file!r}", file=sys.stderr)
        sys.exit(1)

    # Only the flags the user explicitly set become overrides (so a profile's
    # values are not clobbered by argparse defaults). CLI > profile > defaults.
    overrides = {}
    if args.time_limit is not None:
        overrides["time_limit"] = args.time_limit
    if args.gap is not None:
        overrides["gap_tolerance"] = args.gap
    if args.threads is not None:
        overrides["threads"] = args.threads
    if args.solver is not None:
        overrides["solver"] = args.solver
    if args.partitions is not None:
        overrides["partitions"] = args.partitions
    if args.rlt is not None:
        overrides["rlt"] = args.rlt
    if args.nlp_bb is not None:
        overrides["nlp_bb"] = args.nlp_bb
    tuning = _parse_tuning(args.tuning)
    if tuning:
        overrides["tuning"] = tuning

    from discopt.profiles import resolve_options

    try:
        options = resolve_options(args.profile, overrides)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    from discopt.result_io import (
        deserialize_result,
        options_from_payload,
        options_to_payload,
        serialize_result,
        summary_text,
        write_json,
        write_sol,
    )

    payload = options_to_payload(options)

    # Solve: warm daemon when available, in-process fallback otherwise.
    result = None
    if not args.no_daemon:
        from discopt import daemon

        resp = daemon.solve_via_daemon(nl, payload, hard_deadline=args.hard_timeout)
        if resp is not None:
            if not resp.get("ok"):
                print(f"Error (daemon): {resp.get('error')}", file=sys.stderr)
                sys.exit(1)
            result = deserialize_result(resp["result"])
    if result is None:
        if not args.no_daemon and not args.quiet:
            print("daemon unavailable; solving in-process", file=sys.stderr)
        from discopt.modeling.core import from_nl

        result = from_nl(nl).solve(**options_from_payload(payload))

    # Outputs: stdout by default; files only when explicitly requested.
    stub = os.path.splitext(os.path.basename(nl))[0]
    base_dir = Path(args.out_dir) if args.out_dir else Path(os.path.dirname(nl) or ".")
    wrote = []
    if args.json:
        p = base_dir / f"{stub}.result.json"
        write_json(result, p, options=payload)
        wrote.append(str(p))
    if args.sol:
        from discopt.modeling.core import from_nl

        var_names = [v.name for v in from_nl(nl)._variables]
        p = base_dir / f"{stub}.sol"
        write_sol(result, var_names, p)
        wrote.append(str(p))

    if args.format == "json":
        # Scripts redirect this into a file, so it is an archival path too (#1266).
        # `allow_nan=False`, as in `write_json`: never print a bare `NaN` token.
        doc = serialize_result(result, provenance=True, options=payload)
        print(json.dumps(doc, indent=2, allow_nan=False))
    elif not args.quiet:
        print(summary_text(result))
    for w in wrote:
        print(f"-> wrote {w}", file=sys.stderr)

    sys.exit(0 if result.status in ("optimal", "feasible") else 1)


def _keys_lost(old, new, path: str = "") -> list:
    """Paths of object keys present in *old* and absent from *new* (lists by position)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key, value in old.items():
            if path == "" and key in ("schema", "discopt", "provenance"):
                continue
            if key not in new:
                out.append(f"{path}/{key}")
            else:
                out += _keys_lost(value, new[key], f"{path}/{key}")
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path} (length {len(old)} -> {len(new)})"]
        return [
            p for i, (a, b) in enumerate(zip(old, new)) for p in _keys_lost(a, b, f"{path}[{i}]")
        ]
    return []


def _write_upgraded(fh, payload: bytes) -> None:
    """Write the rewritten document to the temporary file (a seam for tests)."""
    fh.write(payload)


def _cmd_upgrade_model(args):
    """Rewrite a native model document at the current schema (plan §3.5).

    Exit 0 on success (or, with ``--check``, when the document loads), 1 when it is
    refused -- unreadable, malformed, or written by a NEWER format, which a rewrite
    would downgrade -- and 2 for usage errors
    (argparse, or a ``--import`` module that fails to import).
    """
    import gzip
    import json
    import warnings

    from discopt.serialize import (
        _EXTENSION_PREFIX,
        _GZIP_MAGIC,
        SCHEMA,
        SchemaVersionWarning,
        SerializationError,
        dumps,
        loads,
    )

    src = Path(args.input)
    dst = Path(args.output) if args.output else src
    for module in args.imports:
        # Registered atoms are re-tagged on load only for names this process has
        # registered, as a program registers them at import before it loads.
        try:
            importlib.import_module(module)
        except Exception as exc:  # any failure of the user's module is a usage error
            print(f"Error: cannot import {module}: {type(exc).__name__}: {exc}", file=sys.stderr)
            sys.exit(2)
    try:
        raw = src.read_bytes()
        was_gzip = raw[:2] == _GZIP_MAGIC
        text = (gzip.decompress(raw) if was_gzip else raw).decode("utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("error", SchemaVersionWarning)
            model = loads(text)
        old_doc = json.loads(text)  # `loads` accepted it, so this parses
        schema_in = old_doc["schema"]
        result = model.saved_result
        # `dumps` records the document's own provenance block under `derived_from`.
        # The model's author is unchanged by an upgrade, so it is carried forward.
        prov = model.provenance if isinstance(model.provenance, dict) else {}
        new_text = dumps(model, result=result, author=prov.get("author"))
        # A top-level ``x-`` annotation is never read, so `dumps` cannot reproduce
        # it; it is carried through verbatim, since it is never refused either.
        extensions = {k: v for k, v in old_doc.items() if k.startswith(_EXTENSION_PREFIX)}
        if extensions:
            new_doc = json.loads(new_text)
            new_doc.update(extensions)
            new_text = json.dumps(new_doc, allow_nan=False)
        # Prove the rewrite before writing it: what comes back must be the model
        # that went in.
        reread = loads(new_text)
        if dumps(reread, result=reread.saved_result, provenance=False) != dumps(
            model, result=result, provenance=False
        ):
            raise SerializationError("the rewritten document does not reload to the same model")
        # A newer format only ever adds, so anything the old document said that the
        # new one does not is content this process could not carry -- above all a
        # registered-atom tag for an atom it has not registered (the one thing the
        # reader may legitimately leave behind; an unknown key it refuses, #1780).
        lost = _keys_lost(old_doc, json.loads(new_text))
        if lost:
            shown = ", ".join(lost[:5]) + (f" and {len(lost) - 5} more" if len(lost) > 5 else "")
            hint = (
                "; register the atoms first with --import <module>"
                if any(p.endswith("/atom") for p in lost)
                else ""
            )
            raise SerializationError(
                f"rewriting would drop content the document carries ({shown}){hint}"
            )
    except SchemaVersionWarning as exc:
        print(f"Error: {src}: refused; {exc}", file=sys.stderr)
        sys.exit(1)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        # SerializationError and json.JSONDecodeError are ValueErrors.
        print(f"Error: {src}: {exc}", file=sys.stderr)
        sys.exit(1)

    def _content(doc: dict) -> dict:
        doc = dict(doc)
        for key in ("schema", "discopt", "provenance"):
            doc.pop(key, None)
        return doc

    changed = _content(old_doc) != _content(json.loads(new_text))
    print(
        f"{src}: {schema_in} -> {SCHEMA}; content "
        f"{'changes' if changed else 'unchanged'} (schema, version and provenance are restamped)"
    )
    if args.check:
        return
    if dst.suffix == ".gz" or (dst == src and was_gzip):
        payload = gzip.compress(new_text.encode("utf-8"))
    else:
        payload = new_text.encode("utf-8")
    # Write beside the target and rename over it: an in-place rewrite must never
    # leave a truncated file where the only copy of the model was.
    import tempfile

    tmp_name = None
    try:
        fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            _write_upgraded(fh, payload)
            fh.flush()
            os.fsync(fh.fileno())
        if dst.exists():
            shutil.copymode(dst, tmp_name)
        else:  # `mkstemp` creates 0600; a new file gets the usual umask mode
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, dst)
        tmp_name = None
    except OSError as exc:
        print(f"Error: {dst}: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)
    print(f"-> wrote {dst}", file=sys.stderr)


_BUILTIN_COMMANDS = frozenset(
    {
        "about",
        "test",
        "convert",
        "gams-register",
        "solve",
        "daemon",
        "gams-daemon",
        "gams-verify",
        "upgrade-model",
        "install-skills",
        "help",
    }
)


def _cli_plugin_entry_points():
    """``discopt.cli`` entry points, sorted by name. Separate function = test seam."""
    try:
        return sorted(importlib.metadata.entry_points(group="discopt.cli"), key=lambda ep: ep.name)
    except Exception:
        return []


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

    p_gams = subparsers.add_parser(
        "gams-register",
        help="Register discopt as a solver with a GAMS system",
    )
    p_gams.add_argument(
        "--directory",
        default=str(Path.home() / ".gams"),
        help="Directory to write/merge gamsconfig.yaml + run script "
        "(default: $HOME/.gams, which GAMS reads automatically).",
    )
    p_gams.add_argument(
        "--check",
        action="store_true",
        help="Only run the gamsapi diagnostic; do not write any files.",
    )
    p_gams.set_defaults(func=_cmd_gams_register)

    p_solve = subparsers.add_parser(
        "solve",
        help="Solve a .nl model (warm-routed through the solve daemon)",
    )
    p_solve.add_argument("file", help="Input model file (.nl)")
    p_solve.add_argument("--profile", default=None, help="Named option profile (e.g. fast, exact).")
    p_solve.add_argument("--time-limit", type=float, default=None, help="Wall-clock limit (s).")
    p_solve.add_argument("--gap", type=float, default=None, help="Relative optimality gap tol.")
    p_solve.add_argument("--threads", type=int, default=None, help="Rust threads.")
    p_solve.add_argument("--solver", default=None, help="Backend selector (e.g. amp, gp).")
    p_solve.add_argument("--partitions", type=int, default=None, help="McCormick partitions.")
    p_solve.add_argument("--rlt", default=None, choices=["auto", "on", "off"], help="RLT control.")
    p_solve.add_argument(
        "--nlp-bb", action=argparse.BooleanOptionalAction, default=None, help="Force NLP-BB on/off."
    )
    p_solve.add_argument(
        "--tuning",
        action="append",
        metavar="KEY=VAL",
        help="SolverTuning field override (repeatable), e.g. --tuning rlt_quad=false.",
    )
    p_solve.add_argument(
        "--no-daemon", action="store_true", help="Solve in-process (do not use the daemon)."
    )
    p_solve.add_argument(
        "--hard-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Daemon-side hard wall: the daemon SIGKILLs itself if this solve "
        "overruns (enforced independently of the solver/client). Default: no limit.",
    )
    p_solve.add_argument(
        "--format", default="text", choices=["text", "json"], help="Stdout format (default text)."
    )
    p_solve.add_argument("--json", action="store_true", help="Write <stub>.result.json.")
    p_solve.add_argument("--sol", action="store_true", help="Write an AMPL-style <stub>.sol.")
    p_solve.add_argument("--out-dir", default=None, help="Directory for --json/--sol output.")
    p_solve.add_argument("--quiet", action="store_true", help="Suppress the stdout summary.")
    p_solve.set_defaults(func=_cmd_solve)

    p_daemon = subparsers.add_parser(
        "daemon",
        help="Control the warm discopt solve daemon (serve/stop/status)",
    )
    p_daemon.add_argument(
        "action",
        choices=["serve", "stop", "kill", "status"],
        help="serve (foreground), stop (graceful), kill (force, for a wedged daemon), or status.",
    )
    p_daemon.set_defaults(func=_cmd_daemon)

    p_gamsd = subparsers.add_parser(
        "gams-daemon",
        help="Control the warm GAMS solver daemon (serve/stop/status)",
    )
    p_gamsd.add_argument(
        "action",
        choices=["serve", "stop", "kill", "status"],
        help="serve (run in foreground), stop, kill (force), or status.",
    )
    p_gamsd.set_defaults(func=_cmd_gams_daemon)

    p_gamsv = subparsers.add_parser(
        "gams-verify",
        help="Run the packaged .gms corpus through GAMS with solver=discopt (needs GAMS)",
    )
    p_gamsv.add_argument("--gams", default="gams", help="Path to the gams executable.")
    p_gamsv.add_argument("--solver", default="discopt", help="GAMS solver name to force.")
    p_gamsv.add_argument("--keep", action="store_true", help="Keep the scratch directory.")
    p_gamsv.set_defaults(func=_cmd_gams_verify)

    p_upg = subparsers.add_parser(
        "upgrade-model",
        help="Rewrite a saved model document (.dopt/.json[.gz]) at the current schema",
    )
    p_upg.add_argument("input", help="Document written by Model.save / discopt.modeling.dumps.")
    p_upg.add_argument(
        "output", nargs="?", default=None, help="Where to write it (default: rewrite in place)."
    )
    p_upg.add_argument(
        "--check", action="store_true", help="Load and report schema in -> out; write nothing."
    )
    p_upg.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE",
        help="Import MODULE first (repeatable), e.g. the one that registers the model's atoms.",
    )
    p_upg.set_defaults(func=_cmd_upgrade_model)

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

    # Plugin subcommands can pull heavy optional deps
    # (e.g. ``discopt.doe.cli`` ~0.4 s: streamlit/pandas), so register them
    # lazily -- only for their own command or when full help is requested. This
    # keeps every other command (notably ``discopt solve``) at the light CLI
    # floor; built-in commands never even scan the entry-point metadata.
    _argv1 = sys.argv[1] if len(sys.argv) > 1 else None
    _want_all = _argv1 in (None, "help", "-h", "--help")

    plugin_runners = {}
    if _want_all or _argv1 not in _BUILTIN_COMMANDS:
        for ep in _cli_plugin_entry_points():
            if ep.name in _BUILTIN_COMMANDS or ep.name in plugin_runners:
                print(
                    f"warning: ignoring discopt.cli plugin {ep.name!r} "
                    f"({ep.value}): name already taken",
                    file=sys.stderr,
                )
                continue
            if not (_want_all or _argv1 == ep.name):
                continue
            try:
                mod = ep.load()
                mod.add_subparser(subparsers)
                plugin_runners[ep.name] = mod.run
            except Exception as exc:
                msg = f"discopt.cli plugin {ep.name!r} ({ep.value}) failed to load: {exc}"
                if _argv1 == ep.name:
                    print(f"Error: {msg}", file=sys.stderr)
                    sys.exit(1)
                print(f"warning: {msg} (skipped)", file=sys.stderr)

    p_help = subparsers.add_parser("help", help="Show this help message and exit")
    p_help.set_defaults(func=lambda _args: parser.print_help())

    args = parser.parse_args()
    if args.command == "help":
        parser.print_help()
        return
    runner = plugin_runners.get(args.command)
    if runner is not None:
        sys.exit(runner(args) or 0)
    args.func(args)


if __name__ == "__main__":
    main()
