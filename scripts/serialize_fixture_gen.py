"""Write the serialization golden fixtures for one schema version (WP-1).

Plan: ``docs/dev/serialization-completion-plan.md`` §3.3. Run it with the
interpreter of a venv in which the discopt under test is installed -- a
*historical* tree checked out at the last commit that wrote ``--expect-schema``,
built there with ``maturin develop --release`` (the Rust ABI must match that
commit's Python side)::

    /tmp/wp1/venv-1.1/bin/python -u scripts/serialize_fixture_gen.py \\
        --tree /tmp/wp1/wt-1.1 --expect-schema discopt.model/1.1 \\
        --out python/tests/data/serialize_fixtures/v1.1

It refuses to write anything unless the imported ``discopt`` *and* its compiled
``_rust`` extension live under ``--tree`` and ``discopt.serialize.SCHEMA`` equals
``--expect-schema`` (CLAUDE.md §8). Each case in
``python/tests/serialize_fixture_models.py`` is built and written with
``provenance=False`` (byte-stable); a case that cannot be built at this version is
recorded as ``unbuildable`` with the exception, never skipped. With ``--forward DIR
--own OWN`` it instead (or also) reads a newer version's documents with *this*
reader and records each outcome -- a forward cell of the compatibility matrix.

Prints ``cases written: N, unbuildable: M, forward cells: K`` and exits non-zero
when N is 0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import traceback

REGISTRY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "python",
    "tests",
    "serialize_fixture_models.py",
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _load_registry():
    spec = importlib.util.spec_from_file_location("serialize_fixture_models", REGISTRY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["serialize_fixture_models"] = mod
    spec.loader.exec_module(mod)
    return mod


def _isolate_tree(tree: str) -> None:
    """Drop every ``sys.path`` entry that provides some *other* discopt checkout.

    A venv built with ``--system-site-packages`` also sees the system's editable
    install, and ``discopt``'s package path then spans both trees: a submodule
    missing from the old tree (``discopt.operators`` before 1.1) silently imports
    from the current one, and the "historical" fixture is written by a mix of two
    versions. Found while generating the first 1.0 fixtures; see also
    ``_check_no_foreign_modules``, which proves the isolation held.
    """
    tree = os.path.realpath(tree)
    keep = []
    for entry in sys.path:
        real = os.path.realpath(entry or ".")
        provides = os.path.isfile(os.path.join(real, "discopt", "__init__.py"))
        if provides and not real.startswith(tree + os.sep):
            print(f"  isolating: dropped {entry} from sys.path", flush=True)
            continue
        keep.append(entry)
    sys.path[:] = keep


def _check_loaded_code(tree: str, expect_schema: str):
    _isolate_tree(tree)
    import discopt
    import discopt._rust as rust
    import discopt.serialize as ser

    tree = os.path.realpath(tree)
    for what, path in (("discopt", discopt.__file__), ("discopt._rust", rust.__file__)):
        if not os.path.realpath(path).startswith(tree + os.sep):
            sys.exit(f"refusing: {what} imported from {path}, not from --tree {tree}")
    if ser.SCHEMA != expect_schema:
        sys.exit(
            f"refusing: discopt.serialize.SCHEMA is {ser.SCHEMA!r}, expected {expect_schema!r}"
        )
    return discopt, ser


def _check_no_foreign_modules(tree: str) -> list[str]:
    """Every loaded ``discopt.*`` module must come from *tree*; returns offenders."""
    tree = os.path.realpath(tree)
    bad = []
    for name, mod in sorted(sys.modules.items()):
        if name != "discopt" and not name.startswith("discopt."):
            continue
        path = getattr(mod, "__file__", None)
        if path and not os.path.realpath(path).startswith(tree + os.sep):
            bad.append(f"{name} <- {path}")
    return bad


def _to_nl(model):
    try:
        return model.to_nl(), None
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed: it IS the datum
        return None, f"{type(exc).__name__}: {exc}"


def _dumps(ser, model, kwargs):
    if "provenance" in inspect.signature(ser.dumps).parameters:
        kwargs = dict(kwargs, provenance=False)
    return ser.dumps(model, indent=1, **kwargs)


def _write_case(ser, reg, name, out_dir):
    model, kwargs, fixed = reg.build(name)
    if fixed is not None:
        var, value = fixed
        with var.fixed(value):
            text = _dumps(ser, model, kwargs)
            nl, nl_why = _to_nl(model)
    else:
        text = _dumps(ser, model, kwargs)
        nl, nl_why = _to_nl(model)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
        fh.write(text)
    entry = {
        "file": f"{name}.json",
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "to_nl_sha256": None if nl is None else hashlib.sha256(nl.encode()).hexdigest(),
        "to_nl_refused": nl_why,
    }
    if name in reg.NL_FILES:
        entry["input_sha256"] = _sha256_file(reg.NL_FILES[name])
    return entry


def _forward_cell(ser, path, own_dir):
    """Read a NEWER version's document with this reader and classify the outcome.

    ``loads_identical``: this reader's re-dump of it equals this version's own
    fixture for the same case byte for byte -- the newer document decoded into
    exactly the model this version itself writes for that case (content this
    reader does not know, such as a newer tag, is necessarily absent from both).
    ``loads_different``: it loaded, but into something else -- a silent misread,
    the one outcome the matrix test forbids. ``refused``: the reader raised.
    """
    name = os.path.basename(path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        loaded = ser.loads(text)
    except Exception as exc:  # noqa: BLE001 - the outcome being recorded
        return {"outcome": "refused", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    own = os.path.join(own_dir, name)
    if not os.path.exists(own):
        return {"outcome": "loaded", "note": "no own-version fixture to compare (unbuildable here)"}
    with open(own, encoding="utf-8") as fh:
        own_text = fh.read()
    redump = _dumps(ser, loaded, {})
    if '"solution"' in own_text:  # compare model content; the embedded result is checked live
        import json as _json

        own_doc = _json.loads(own_text)
        own_doc.pop("solution", None)
        own_text = _json.dumps(own_doc, indent=1)
        redump = _json.dumps(_json.loads(redump), indent=1)
    if redump == own_text:
        return {"outcome": "loads_identical"}
    return {"outcome": "loads_different", "diff_hint": _first_difference(own_text, redump)}


def _first_difference(a: str, b: str) -> str:
    for i, (x, y) in enumerate(zip(a.splitlines(), b.splitlines())):
        if x != y:
            return f"line {i + 1}: own={x.strip()[:120]!r} reloaded={y.strip()[:120]!r}"
    return f"length differs: own={len(a)} reloaded={len(b)}"


def _merge_manifest(path: str, label: str, report: dict, forward_dir) -> None:
    """Merge one run into ``MANIFEST.json``.

    ``versions[label]`` holds the writer facts and per-case hashes;
    ``versions[label]["forward"][writer]`` holds the cells for documents written by
    the newer version ``writer`` (taken from the ``--forward`` directory's name,
    ``v<writer>``) and read by ``label``.
    """
    manifest = {"versions": {}}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    entry = manifest["versions"].setdefault(label, {})
    for key in ("schema", "commit", "discopt_version", "python"):
        if entry.get(key) not in (None, report[key]):
            sys.exit(f"refusing: manifest {label}.{key}={entry[key]!r}, this run {report[key]!r}")
        entry[key] = report[key]
    if "cases" in report:
        entry["cases"] = report["cases"]
    if forward_dir:
        writer = os.path.basename(os.path.normpath(forward_dir))
        if not writer.startswith("v"):
            sys.exit(f"--forward directory must be named v<version>, got {writer!r}")
        entry.setdefault("forward", {})[writer[1:]] = report["forward"]
    manifest["versions"] = dict(sorted(manifest["versions"].items(), key=lambda kv: _key(kv[0])))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=False)
        fh.write("\n")


def _key(label: str) -> tuple:
    return tuple(int(p) for p in label.split("."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", required=True, help="root of the discopt checkout under test")
    ap.add_argument("--expect-schema", required=True, help='e.g. "discopt.model/1.1"')
    ap.add_argument("--out", help="directory to write this version's fixtures into")
    ap.add_argument("--forward", help="a newer version's fixture directory to read")
    ap.add_argument("--own", help="this version's fixtures, compared against (with --forward)")
    ap.add_argument("--label", help='this version\'s label in the manifest, e.g. "1.0"')
    ap.add_argument("--manifest", help="MANIFEST.json to merge this run's results into")
    args = ap.parse_args()

    discopt, ser = _check_loaded_code(args.tree, args.expect_schema)
    reg = _load_registry()
    commit = subprocess.run(
        ["git", "-C", args.tree, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    report: dict = {
        "schema": ser.SCHEMA,
        "commit": commit,
        "discopt_version": getattr(discopt, "__version__", None),
        "python": sys.version.split()[0],
    }

    written = unbuildable = forward = 0
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        cases: dict = {}
        for name in reg.CASES:
            try:
                cases[name] = _write_case(ser, reg, name, args.out)
                written += 1
                print(f"  wrote {name}", flush=True)
            except Exception as exc:  # noqa: BLE001 - recorded per §3.3, never skipped
                cases[name] = {
                    "unbuildable": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "trace_tail": traceback.format_exc().splitlines()[-3:],
                }
                unbuildable += 1
                print(f"  UNBUILDABLE {name}: {type(exc).__name__}: {exc}", flush=True)
        report["cases"] = cases
    if args.forward:
        if not args.own:
            sys.exit("--forward needs --own (this version's fixtures to compare against)")
        # Build every case first, as a program defines its registrations at import: a
        # registered atom's tag is re-derived on load only for a name registered in
        # the reading process, so without this the cell would measure process state
        # (the registry), not the document.
        for name in reg.CASES:
            try:
                reg.build(name)
            except Exception as exc:  # noqa: BLE001 - reported; the cell says "loaded"
                print(f"  (cannot build {name} here: {type(exc).__name__}: {exc})", flush=True)
        cells: dict = {}
        for fname in sorted(os.listdir(args.forward)):
            if fname.endswith(".json"):
                cells[fname] = _forward_cell(ser, os.path.join(args.forward, fname), args.own)
                forward += 1
                print(f"  forward {fname}: {cells[fname]['outcome']}", flush=True)
        report["forward"] = cells

    foreign = _check_no_foreign_modules(args.tree)
    if foreign:
        # Written files are from a mixed import; say so and fail rather than keep them.
        print("REFUSING: discopt modules imported from outside --tree:", *foreign, sep="\n  ")
        return 3
    if args.manifest:
        if not args.label:
            sys.exit("--manifest needs --label")
        _merge_manifest(args.manifest, args.label, report, args.forward)
    print(json.dumps(report, indent=1, sort_keys=True))
    print(f"cases written: {written}, unbuildable: {unbuildable}, forward cells: {forward}")
    return 0 if (written or forward) else 1


if __name__ == "__main__":
    sys.exit(main())
