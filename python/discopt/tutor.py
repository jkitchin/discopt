"""``discopt tutor`` — entry point for the Claude-Code-driven course.

Thin CLI sugar that locates the ``course/`` tree, resolves a lesson id,
and shells out to ``claude`` with the matching ``/course:`` slash
command (lesson, hint, assess, progress). The lesson library, grading
rubrics, and progress tracking all live under ``course/``; this module
adds no separate state.

The course content ships as package data under ``discopt.course`` so a
pip install includes it. ``discopt tutor install`` materializes a
writable copy of the tree into the user's working directory and drops
the ``/course:`` slash commands into ``./.claude/``.

Subcommands:

* ``discopt tutor``                       — dashboard (counts + next lesson)
* ``discopt tutor list``                  — every lesson with completion status
* ``discopt tutor start <lesson>``        — launch ``claude /course:lesson <lesson>``
* ``discopt tutor resume``                — start whatever ``current_lesson`` is
* ``discopt tutor next``                  — start the next-numbered lesson after ``current_lesson``
* ``discopt tutor reset [<lesson>]``      — drop one entry from progress.yaml (or all)
* ``discopt tutor install [--force]``     — copy the ``/course:`` slash commands
                                            into ``./.claude/``
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_TRACKS = ("basic", "intermediate", "advanced")


def _packaged_course_dir() -> Path | None:
    """Return the read-only course/ tree shipped inside the wheel."""
    try:
        from discopt.course import package_root
    except Exception:
        return None
    p = package_root()
    return p if (p / "SYLLABUS.md").exists() else None


def _find_course_dir() -> Path | None:
    """Locate the ``course/`` tree.

    Resolution order:
    1. ``$DISCOPT_COURSE_DIR``
    2. Walk upward from cwd looking for ``course/SYLLABUS.md``
       (so a writable copy in the user's project always wins).
    3. The packaged copy shipped with the wheel (read-only).
    """
    env = os.environ.get("DISCOPT_COURSE_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "SYLLABUS.md").exists():
            return p
        return None
    cwd = Path.cwd().resolve()
    for cand in [cwd, *cwd.parents]:
        if (cand / "course" / "SYLLABUS.md").exists():
            return cand / "course"
    return _packaged_course_dir()


def _is_read_only(course_dir: Path) -> bool:
    """Heuristic: True when ``course_dir`` is the packaged copy in site-packages."""
    pkg = _packaged_course_dir()
    if pkg is None:
        return False
    try:
        return course_dir.resolve() == pkg.resolve()
    except OSError:
        return False


def _list_lessons(course_dir: Path) -> list[str]:
    """Return ``<track>/<id>`` strings in syllabus order."""
    out: list[str] = []
    for track in _TRACKS:
        td = course_dir / track
        if not td.is_dir():
            continue
        for sub in sorted(td.iterdir()):
            if sub.is_dir() and (sub / "reading.ipynb").exists():
                out.append(f"{track}/{sub.name}")
    return out


def _resolve_lesson(course_dir: Path, name: str) -> str | None:
    """Map a user-supplied lesson name to a canonical ``<track>/<id>``.

    Accepts:
    * ``"basic/02_lp_fundamentals"`` — already canonical
    * ``"02"`` or ``"02_lp_fundamentals"`` — looked up across tracks
    """
    all_lessons = _list_lessons(course_dir)
    if name in all_lessons:
        return name
    # Strip leading track if present but unknown
    short = name.split("/", 1)[-1]
    matches = [le for le in all_lessons if le.split("/", 1)[1].startswith(short)]
    if not matches and re.fullmatch(r"\d+", short):
        # bare lesson number, e.g. "16"
        matches = [le for le in all_lessons if le.split("/", 1)[1].startswith(short + "_")]
    if len(matches) == 1:
        return matches[0]
    return None


def _progress_path(course_dir: Path) -> Path:
    return course_dir / "progress.yaml"


def _load_progress(course_dir: Path) -> dict | None:
    """Best-effort load of ``course/progress.yaml``.

    Returns ``None`` if the file is missing or PyYAML is unavailable.
    """
    path = _progress_path(course_dir)
    if not path.exists():
        return None
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None


def _scores_for(progress: dict | None) -> dict:
    if not progress:
        return {}
    scores = progress.get("scores") or {}
    return scores if isinstance(scores, dict) else {}


def _current_lesson(progress: dict | None) -> str | None:
    if not progress:
        return None
    cl = progress.get("current_lesson")
    return cl if isinstance(cl, str) else None


def _claude_binary() -> str | None:
    return shutil.which("claude")


def _print_no_claude() -> None:
    print(
        "claude binary not found on PATH. Install Claude Code "
        "(https://claude.com/claude-code) and re-run.",
        file=sys.stderr,
    )


def _launch_slash(slash_command: str, *extra_args: str) -> int:
    """Spawn ``claude "<slash_command> <extra_args>"`` and inherit its TTY."""
    claude = _claude_binary()
    if claude is None:
        _print_no_claude()
        return 1
    initial = " ".join((slash_command, *extra_args)).strip()
    # Inherit stdin/stdout/stderr so the user gets a real interactive session.
    return subprocess.call([claude, initial])


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_dashboard(_args, course_dir: Path) -> int:
    lessons = _list_lessons(course_dir)
    progress = _load_progress(course_dir)
    scores = _scores_for(progress)
    current = _current_lesson(progress)

    print(f"discopt tutor  ({course_dir})")
    if _is_read_only(course_dir):
        print(
            "  (read-only packaged copy; run `discopt tutor install` to "
            "materialize a writable course/ in cwd)"
        )
    print()
    for track in _TRACKS:
        track_lessons = [le for le in lessons if le.startswith(track + "/")]
        done = [le for le in track_lessons if _is_passed(scores.get(le))]
        avg = _avg([scores[le].get("total") for le in track_lessons if le in scores])
        avg_s = f"{avg:.0f}" if avg is not None else "-"
        print(f"  {track:12s}  {len(done):>2d} / {len(track_lessons):<2d} complete   avg {avg_s}")
    print()
    if current:
        print(f"  current lesson:  {current}")
    next_id = _next_lesson(lessons, scores, current)
    if next_id:
        print(f"  next lesson:     {next_id}")
    print()
    print("  discopt tutor list             show every lesson")
    print("  discopt tutor start <lesson>   launch a lesson in claude")
    print("  discopt tutor resume           continue current_lesson")
    return 0


def _is_passed(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    total = entry.get("total")
    return isinstance(total, (int, float)) and total >= 70


def _avg(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _next_lesson(lessons: list[str], scores: dict, current: str | None) -> str | None:
    if current and current in lessons:
        idx = lessons.index(current)
        # If current is already passed, jump to the one after it.
        if _is_passed(scores.get(current)):
            return lessons[idx + 1] if idx + 1 < len(lessons) else None
        return current
    # No current lesson: first unpassed lesson.
    for le in lessons:
        if not _is_passed(scores.get(le)):
            return le
    return None


def _cmd_list(_args, course_dir: Path) -> int:
    lessons = _list_lessons(course_dir)
    progress = _load_progress(course_dir)
    scores = _scores_for(progress)
    current = _current_lesson(progress)

    if progress is None and not _progress_path(course_dir).exists():
        print("(no progress.yaml yet — run /course:progress in claude to initialize)\n")

    for le in lessons:
        entry = scores.get(le) or {}
        total = entry.get("total")
        if isinstance(total, (int, float)):
            tag = f"{int(total):3d}"
        else:
            tag = " - "
        marker = "*" if le == current else " "
        print(f"  {marker} {tag}   {le}")
    return 0


def _cmd_start(args, course_dir: Path) -> int:
    lesson = _resolve_lesson(course_dir, args.lesson)
    if lesson is None:
        print(f"unknown lesson: {args.lesson!r}", file=sys.stderr)
        print("run `discopt tutor list` to see available lessons.", file=sys.stderr)
        return 1
    return _launch_slash(f"/course:lesson {lesson}")


def _cmd_resume(_args, course_dir: Path) -> int:
    progress = _load_progress(course_dir)
    current = _current_lesson(progress)
    if current is None:
        if progress is None and not _progress_path(course_dir).exists():
            print(
                "no progress.yaml yet. Start your first lesson with "
                "`discopt tutor start basic/01_intro_to_optimization`.",
                file=sys.stderr,
            )
        else:
            print(
                "no current_lesson recorded in progress.yaml. "
                "Use `discopt tutor start <lesson>` to pick one.",
                file=sys.stderr,
            )
        return 1
    return _launch_slash(f"/course:lesson {current}")


def _cmd_next(_args, course_dir: Path) -> int:
    lessons = _list_lessons(course_dir)
    progress = _load_progress(course_dir)
    scores = _scores_for(progress)
    current = _current_lesson(progress)
    nxt = _next_lesson(lessons, scores, current)
    if nxt is None:
        print("you've completed every lesson — nothing left to start.")
        return 0
    return _launch_slash(f"/course:lesson {nxt}")


def _cmd_reset(args, course_dir: Path) -> int:
    if _is_read_only(course_dir):
        print(
            "course/ is the read-only packaged copy. Run "
            "`discopt tutor install` first to make it writable.",
            file=sys.stderr,
        )
        return 1
    path = _progress_path(course_dir)
    if not path.exists():
        print("no progress.yaml to reset.")
        return 0
    if args.lesson is None:
        # Wipe the whole file: restore from template.
        tmpl = course_dir / "progress.template.yaml"
        if not tmpl.exists():
            print("progress.template.yaml is missing; refusing to delete progress.yaml.")
            return 1
        confirm = input(f"reset all progress at {path}? [y/N] ").strip().lower()
        if confirm != "y":
            print("aborted.")
            return 0
        shutil.copy2(tmpl, path)
        print(f"reset {path} from template.")
        return 0
    # Drop a single entry.
    lesson = _resolve_lesson(course_dir, args.lesson)
    if lesson is None:
        print(f"unknown lesson: {args.lesson!r}", file=sys.stderr)
        return 1
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        print(
            "per-lesson reset needs PyYAML. Install it or edit progress.yaml by hand.",
            file=sys.stderr,
        )
        return 1
    data = yaml.safe_load(path.read_text()) or {}
    scores = data.get("scores") or {}
    if lesson in scores:
        del scores[lesson]
        data["scores"] = scores or None
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"dropped progress entry for {lesson}.")
    else:
        print(f"no progress entry for {lesson}; nothing to do.")
    return 0


_PY_ARTIFACT_SUFFIXES = (".pyc", ".pyo")


def _is_py_artifact(rel: Path) -> bool:
    """Reject Python build artifacts that leak through editable installs."""
    if "__pycache__" in rel.parts:
        return True
    if rel.name == "__init__.py" and len(rel.parts) == 1:
        # The course/__init__.py we added to make it a package — student
        # shouldn't see it in their materialized copy.
        return True
    return rel.suffix in _PY_ARTIFACT_SUFFIXES


def _copy_tree(src_root: Path, dest_root: Path, *, force: bool, skip: set[str]) -> tuple[int, int]:
    """Copy ``src_root`` recursively into ``dest_root``.

    Skips any top-level entry whose name is in *skip* (so the course copy
    can exclude ``_claude_assets`` while still pulling lessons across) and
    Python build artifacts (``__pycache__``, ``*.pyc``, the package's own
    ``__init__.py``). Returns ``(copied, skipped)`` file counts.
    """
    copied = 0
    skipped = 0
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        if rel.parts and rel.parts[0] in skip:
            continue
        if _is_py_artifact(rel):
            continue
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            print(f"  skip  {rel} (exists; pass --force to overwrite)")
            skipped += 1
            continue
        shutil.copy2(src, dest)
        print(f"  copy  {rel}")
        copied += 1
    return copied, skipped


def _cmd_install(args, course_dir: Path) -> int:
    """Materialize a writable course/ tree + Claude assets into cwd.

    Copies:
    * ``course/_claude_assets/`` -> ``./.claude/``
    * everything else under ``course/`` -> ``./course/``

    Idempotent: existing files are preserved unless ``--force`` is passed.
    """
    src_assets = course_dir / "_claude_assets"
    if not src_assets.is_dir():
        print(f"missing {src_assets}; can't install.", file=sys.stderr)
        return 1
    cwd = Path.cwd()
    claude_dest = cwd / ".claude"
    course_dest = cwd / "course"

    print(f"installing Claude assets into {claude_dest}")
    n_claude, n_claude_skip = _copy_tree(src_assets, claude_dest, force=args.force, skip=set())

    course_copied = course_skipped = 0
    if course_dest.resolve() != course_dir.resolve():
        print(f"\ninstalling course content into {course_dest}")
        course_copied, course_skipped = _copy_tree(
            course_dir,
            course_dest,
            force=args.force,
            skip={"_claude_assets"},
        )
    else:
        print(f"\nskipping course content copy: already at {course_dest}")

    total = n_claude + course_copied
    total_skipped = n_claude_skip + course_skipped
    print(f"\ninstalled {total} file(s); {total_skipped} skipped.")
    if total_skipped:
        print("  pass --force to overwrite existing files.")
    return 0


# ---------------------------------------------------------------------------
# argparse entry point
# ---------------------------------------------------------------------------


def add_subparser(subparsers) -> None:
    """Register the ``tutor`` subcommand on the top-level ``discopt`` parser."""
    p = subparsers.add_parser(
        "tutor",
        help="Interactive Claude-Code-driven discopt course.",
        description=(
            "Entry point for the discopt course. Locates the course/ tree, "
            "resolves a lesson id, and launches a claude session running the "
            "matching /course: slash command."
        ),
    )
    tutor_sub = p.add_subparsers(dest="tutor_cmd", metavar="<subcommand>")

    p_list = tutor_sub.add_parser("list", help="list every lesson with status")
    p_list.set_defaults(tutor_func=_cmd_list)

    p_start = tutor_sub.add_parser("start", help="launch a specific lesson")
    p_start.add_argument("lesson", help="lesson id, e.g. basic/02_lp_fundamentals or just 02")
    p_start.set_defaults(tutor_func=_cmd_start)

    p_resume = tutor_sub.add_parser("resume", help="continue current_lesson")
    p_resume.set_defaults(tutor_func=_cmd_resume)

    p_next = tutor_sub.add_parser("next", help="start the next-numbered lesson")
    p_next.set_defaults(tutor_func=_cmd_next)

    p_reset = tutor_sub.add_parser("reset", help="reset progress (one lesson, or all)")
    p_reset.add_argument("lesson", nargs="?", help="lesson id; omit to reset all")
    p_reset.set_defaults(tutor_func=_cmd_reset)

    p_install = tutor_sub.add_parser(
        "install", help="install /course: slash commands into ./.claude/"
    )
    p_install.add_argument("--force", action="store_true", help="overwrite existing files")
    p_install.set_defaults(tutor_func=_cmd_install)

    # Bare `discopt tutor` (no subcommand) shows the dashboard.
    p.set_defaults(tutor_func=_cmd_dashboard, tutor_cmd=None)


def run(args) -> int:
    """Dispatch ``discopt tutor ...`` after argparse parsing."""
    course_dir = _find_course_dir()
    # ``install`` is the only command that legitimately runs against a course
    # tree the user might still need help finding; all others require it.
    if course_dir is None:
        print(
            "course/ directory not found. Run from a discopt checkout that "
            "contains course/SYLLABUS.md, or set DISCOPT_COURSE_DIR.",
            file=sys.stderr,
        )
        return 1
    rc: int = args.tutor_func(args, course_dir)
    return rc
