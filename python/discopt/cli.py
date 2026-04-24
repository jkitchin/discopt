"""discopt CLI.

Usage:
    discopt about
    discopt test
    discopt convert input.gms output.nl
    discopt install-skills [--project-scope] [--dev] [--force]
    discopt lit-scan [topic ...]
    discopt adversary [-n COUNT] [topic ...]
    discopt search-arxiv "query" [--max-results 20] [--start-date 2026-01-01]
    discopt search-openalex "query" [--from-date ...] [--to-date ...] [--per-page 20]
    discopt write-report <output-path>
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


def search_arxiv(query: str, max_results: int = 20) -> list[dict]:
    """Query arXiv API and return parsed results."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"http://export.arxiv.org/api/query?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "discopt-discoptbot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read().decode("utf-8")

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_data)

    results = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        updated_el = entry.find("atom:updated", ns)

        title_text = title_el.text if title_el is not None else None
        summary_text = summary_el.text if summary_el is not None else None
        pub_text = published_el.text if published_el is not None else None
        upd_text = updated_el.text if updated_el is not None else None

        title = title_text.strip().replace("\n", " ") if title_text else ""
        abstract = summary_text.strip().replace("\n", " ") if summary_text else ""
        published = pub_text.strip() if pub_text else ""
        updated = upd_text.strip() if upd_text else ""

        authors = []
        for author_el in entry.findall("atom:author", ns):
            name_el = author_el.find("atom:name", ns)
            name_text = name_el.text if name_el is not None else None
            if name_text:
                authors.append(name_text.strip())

        arxiv_id = ""
        pdf_link = ""
        for link_el in entry.findall("atom:link", ns):
            href = link_el.get("href", "")
            if link_el.get("title") == "pdf":
                pdf_link = href
            elif "/abs/" in href:
                arxiv_id = href.split("/abs/")[-1]

        category_el = entry.find("arxiv:primary_category", ns)
        category = category_el.get("term", "") if category_el is not None else ""

        doi_el = entry.find("arxiv:doi", ns)
        doi_text = doi_el.text if doi_el is not None else None
        doi = doi_text.strip() if doi_text else ""

        results.append(
            {
                "title": title,
                "authors": authors,
                "arxiv_id": arxiv_id,
                "published": published[:10],
                "updated": updated[:10],
                "abstract": abstract[:500],
                "category": category,
                "doi": doi,
                "pdf_link": pdf_link,
                "link": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            }
        )

    return results


def search_openalex(
    query: str,
    from_date: str | None = None,
    to_date: str | None = None,
    per_page: int = 20,
    api_key: str | None = None,
) -> list[dict]:
    """Query OpenAlex API and return parsed results."""
    filters = [f"default.search:{query}"]
    if from_date:
        filters.append(f"from_created_date:{from_date}")
    if to_date:
        filters.append(f"to_created_date:{to_date}")

    params: dict = {
        "filter": ",".join(filters),
        "sort": "publication_date:desc",
        "per_page": per_page,
    }
    if api_key:
        params["api_key"] = api_key

    url = f"https://api.openalex.org/works?{urllib.parse.urlencode(params)}"

    # OpenAlex uses API keys for authentication. Get a free key at
    # openalex.org/settings/api and set OPENALEX_API_KEY or pass --api-key.
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "discopt/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    for work in data.get("results", []):
        title = work.get("title", "") or ""
        doi = work.get("doi", "") or ""
        pub_date = work.get("publication_date", "") or ""
        oa_id = work.get("id", "") or ""

        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        abstract = ""
        inv_index = work.get("abstract_inverted_index")
        if inv_index:
            word_positions = []
            for word, positions in inv_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            abstract = " ".join(w for _, w in word_positions)[:500]

        source = ""
        primary_loc = work.get("primary_location") or {}
        source_obj = primary_loc.get("source") or {}
        source = source_obj.get("display_name", "") or ""

        link = ""
        best_oa = work.get("best_oa_location") or {}
        link = best_oa.get("pdf_url", "") or best_oa.get("landing_page_url", "") or ""
        if not link and doi:
            link = doi if doi.startswith("http") else f"https://doi.org/{doi}"

        results.append(
            {
                "title": title,
                "authors": authors[:5],
                "doi": doi.replace("https://doi.org/", "")
                if doi.startswith("https://doi.org/")
                else doi,
                "published": pub_date,
                "abstract": abstract,
                "source": source,
                "openalex_id": oa_id,
                "link": link,
            }
        )

    return results


def _cmd_search_arxiv(args):
    try:
        results = search_arxiv(args.query, args.max_results)
    except Exception as e:
        print(json.dumps({"error": str(e), "results": []}))
        sys.exit(1)

    if args.start_date:
        results = [r for r in results if r["published"] >= args.start_date]

    print(json.dumps({"query": args.query, "count": len(results), "results": results}, indent=2))


def _cmd_search_openalex(args):
    try:
        results = search_openalex(
            args.query,
            from_date=args.from_date,
            to_date=args.to_date,
            per_page=args.per_page,
            api_key=args.api_key,
        )
    except Exception as e:
        print(json.dumps({"error": str(e), "results": []}))
        sys.exit(1)

    print(
        json.dumps(
            {"query": args.query, "count": len(results), "results": results},
            indent=2,
        )
    )


def _cmd_write_report(args):
    path = args.output_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    content = sys.stdin.read()
    with open(path, "w") as f:
        f.write(content)
    print(f"Wrote {len(content)} bytes to {path}")


def _cmd_about(_args):
    import discopt

    version = discopt.__version__

    # Installation location
    install_location = os.path.dirname(os.path.abspath(discopt.__file__))

    # Try to get package metadata from pip
    try:
        meta = importlib.metadata.metadata("discopt")
        pkg_version = meta["Version"]
        summary = meta["Summary"] or ""
        license_text = meta["License"] or ""
    except importlib.metadata.PackageNotFoundError:
        pkg_version = version
        summary = "Hybrid MINLP solver combining Rust and JAX"
        license_text = "EPL-2.0"

    # Check for Rust extension
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

    # Key dependency versions
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


def _run_claude_command(slash_command: str, args_str: str = ""):
    """Invoke a Claude Code slash command via the claude CLI."""
    import shutil
    import subprocess

    claude_bin = shutil.which("claude")
    if claude_bin is None:
        print(
            "Error: 'claude' CLI not found on PATH.\nInstall Claude Code: https://claude.ai/code",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt = f"/{slash_command}"
    if args_str:
        prompt = f"{prompt} {args_str}"

    subprocess.run([claude_bin, "-p", prompt], check=False)


def _cmd_lit_scan(args):
    """Run the discoptbot literature scanner."""
    topic = " ".join(args.topic) if args.topic else ""
    _run_claude_command("discoptbot", topic)


def _cmd_adversary(args):
    """Run the adversary solver correctness agent."""
    parts = []
    if args.count != 1:
        parts.append(str(args.count))
    if args.topic:
        parts.extend(args.topic)
    _run_claude_command("adversary", " ".join(parts))


def _cmd_test(_args):
    """Run a quick smoke test to verify the installation works."""
    errors = []
    passed = []

    # 1. Import core package
    try:
        import discopt

        passed.append(f"import discopt ({discopt.__version__})")
    except Exception as e:
        errors.append(f"import discopt: {e}")

    # 2. Rust extension
    try:
        import discopt._rust  # noqa: F811

        passed.append("Rust extension loaded")
    except ImportError as e:
        errors.append(f"Rust extension: {e}")

    # 3. JAX
    try:
        import jax
        import jax.numpy as jnp

        _ = jnp.array([1.0, 2.0])
        passed.append(f"JAX {jax.__version__} (backend: {jax.default_backend()})")
    except Exception as e:
        errors.append(f"JAX: {e}")

    # 4. Build and solve a tiny model
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

    # 5. DAG compiler (JAX tracing)
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

    # Print results
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
        # Import
        if in_ext == ".gms":
            model = dm.from_gams(in_path)
        else:
            model = dm.from_nl(in_path)

        # Export
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

    # about
    p_about = subparsers.add_parser("about", help="Show version and installation info")
    p_about.set_defaults(func=_cmd_about)

    # test
    p_test = subparsers.add_parser("test", help="Run smoke tests to verify installation")
    p_test.set_defaults(func=_cmd_test)

    # convert
    p_conv = subparsers.add_parser(
        "convert",
        help="Convert between model formats (.gms, .nl, .mps, .lp)",
    )
    p_conv.add_argument("input", help="Input file path (.gms or .nl)")
    p_conv.add_argument("output", help="Output file path (.gms, .nl, .mps, or .lp)")
    p_conv.set_defaults(func=_cmd_convert)

    # search-arxiv
    p_arxiv = subparsers.add_parser("search-arxiv", help="Search arXiv API")
    p_arxiv.add_argument("query", help="arXiv search query")
    p_arxiv.add_argument("--max-results", type=int, default=20)
    p_arxiv.add_argument(
        "--start-date",
        help="Filter results after this date (YYYY-MM-DD)",
    )
    p_arxiv.set_defaults(func=_cmd_search_arxiv)

    # search-openalex
    p_oa = subparsers.add_parser("search-openalex", help="Search OpenAlex API")
    p_oa.add_argument("query", help="Search query")
    p_oa.add_argument("--from-date", help="Start date (YYYY-MM-DD)")
    p_oa.add_argument("--to-date", help="End date (YYYY-MM-DD)")
    p_oa.add_argument("--per-page", type=int, default=20)
    p_oa.add_argument(
        "--api-key",
        default=os.environ.get("OPENALEX_API_KEY"),
        help="OpenAlex API key (or set OPENALEX_API_KEY env var)",
    )
    p_oa.set_defaults(func=_cmd_search_openalex)

    # write-report
    p_wr = subparsers.add_parser("write-report", help="Write stdin to a file")
    p_wr.add_argument("output_path", help="Output file path")
    p_wr.set_defaults(func=_cmd_write_report)

    # lit-scan (Claude-powered literature scanner)
    p_lit = subparsers.add_parser(
        "lit-scan",
        help="Search recent literature for papers relevant to discopt",
    )
    p_lit.add_argument(
        "topic",
        nargs="*",
        help="Optional topic focus (e.g., 'McCormick relaxations')",
    )
    p_lit.set_defaults(func=_cmd_lit_scan)

    # adversary (Claude-powered solver correctness testing)
    p_adv = subparsers.add_parser(
        "adversary",
        help="Test discopt against benchmark problems with known solutions",
    )
    p_adv.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="Number of problems to test (default: 1)",
    )
    p_adv.add_argument(
        "topic",
        nargs="*",
        help="Optional problem class focus (e.g., 'MINLP', 'convex NLP')",
    )
    p_adv.set_defaults(func=_cmd_adversary)

    # install-skills
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
