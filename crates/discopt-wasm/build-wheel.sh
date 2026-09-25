#!/usr/bin/env bash
# Build `discopt` for Pyodide and stage it for the browser page.
#
#   crates/discopt-wasm/build-wheel.sh          # build and stage
#   crates/discopt-wasm/build-wheel.sh --check  # verify the staged wheel (CI)
#
# This is the *real* package compiled for emscripten, not a second
# implementation: the wheel micropip installs is built from the root
# `pyproject.toml` by the same maturin backend that builds the desktop wheels,
# so `import discopt` in the browser gets the modeling API, the Rust
# branch-and-bound core and the in-house simplex with the same behaviour.
#
# ## Why this needs its own toolchain
#
# An emscripten extension module is not a `rustup target add` away. Pyodide
# links CPython and every extension as emscripten *side modules*, which pins
# four versions to each other:
#
#   * PYODIDE_VERSION       — what the page loads (`web/worker.js`)
#   * EMSCRIPTEN_VERSION    — the emsdk that Pyodide release was built with;
#                             it is baked into the wheel's ABI tag, which is
#                             why a wheel is valid for one Pyodide and no other
#   * RUST_TOOLCHAIN        — Rust for emscripten needs a std built with wasm
#                             exception handling, which rustup does not ship
#   * SYSROOT_RELEASE       — that std, from pyodide/rust-emscripten-wasm-eh-sysroot,
#                             published per (emcc version, nightly) pair
#
# All four are the values POUNCE's `crates/pounce-wasm/build-wheel.sh` uses, and
# they must stay byte-identical to them. The page installs *two* emscripten
# wheels into one interpreter — discopt's and `pounce-solver`'s, because discopt
# calls POUNCE for its NLP subproblems — and two side modules sharing an
# interpreter must have been compiled against the same emscripten. Nothing
# enforces that across the repository boundary; this comment is the enforcement.
#
# ## The emsdk must be Pyodide's, not one you cloned
#
# `pyodide xbuildenv install-emscripten` installs emsdk *and applies Pyodide's
# patches to it*, and one of them is load-bearing for every Rust package:
# upstream emscripten 4.0.9 runs each side-module export through
# `str.isidentifier()` (tools/emscripten.py) and aborts on the first Rust
# legacy-mangled name, which contain `$` and `..`:
#
#   emcc: error: invalid export name: _ZN4core3fmt5float52_$LT$impl$u20$core..fmt..Display...
#
# A stock emsdk fails that way on a twelve-line crate that does
# `format!("{}", 1.5f64)` — the symbol comes out of the precompiled std, so
# nothing about the crate being built changes it. If you see that error, the
# build found the wrong emcc; it is not an LTO, codegen-units, visibility or
# `debug` setting, and no combination of those fixes it.
#
# When bumping PYODIDE_VERSION: read the new release's emscripten version, pick
# the newest `emcc-<that>_nightly-<date>` sysroot release, update all four below
# together, AND rebuild POUNCE's wheel against the same set.
set -euo pipefail

# Keep in step with `web/worker.js`'s PYODIDE_VERSION; --check enforces it.
PYODIDE_VERSION=0.28.3
PYODIDE_BUILD_VERSION=0.39.0
EMSCRIPTEN_VERSION=4.0.9
RUST_TOOLCHAIN=nightly-2025-06-27
SYSROOT_RELEASE="emcc-${EMSCRIPTEN_VERSION}_nightly-2025-06-27"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
wheels="$here/web/wheels"
toolchain="${DISCOPT_PYODIDE_TOOLCHAIN:-$root/target/pyodide-toolchain}"

pinned_pyodide_version() {
  sed -n "s/^const PYODIDE_VERSION = '\(.*\)';$/\1/p" "$here/web/worker.js"
}

project_version() {
  python3 -c 'import re,sys; print(re.search(r"^version = \"(.*)\"", open(sys.argv[1]).read(), re.M).group(1))' \
    "$root/pyproject.toml"
}

# --check: no build, no toolchain — just the invariants CI cares about. Both
# wheels are checked, because the page needs both and a missing `pounce-solver`
# would surface only as a lazy `import pounce` failing mid-solve.
if [[ "${1:-}" == "--check" ]]; then
  page="$(pinned_pyodide_version)"
  fail=0
  if [[ "$page" != "$PYODIDE_VERSION" ]]; then
    echo "worker.js pins Pyodide $page but this script builds for $PYODIDE_VERSION" >&2
    fail=1
  fi

  check_manifest() {
    local which="$1" manifest="$wheels/$1-wheel.json" prefix="$2"
    if [[ ! -f "$manifest" ]]; then
      echo "no $which wheel staged at $manifest" >&2
      return 1
    fi
    local have wheel
    have=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pyodide_version"])' "$manifest")
    wheel=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["wheel"])' "$manifest")
    local bad=0
    if [[ "$have" != "$page" ]]; then
      echo "staged $which wheel targets Pyodide $have; the page loads $page" >&2
      bad=1
    fi
    if [[ ! -f "$wheels/$wheel" ]]; then
      echo "$which manifest names $wheel but it is not in $wheels" >&2
      bad=1
    fi
    if [[ "$wheel" != "$prefix"* ]]; then
      echo "staged $which wheel is $wheel, expected to start with $prefix" >&2
      bad=1
    fi
    return $bad
  }

  check_manifest discopt "discopt-$(project_version)-" || fail=1
  # `pounce-solver` is not built here -- it is copied from the POUNCE repo,
  # which builds and commits it. Only its presence and Pyodide version are ours
  # to check.
  check_manifest pounce "pounce_solver-" || fail=1

  [[ $fail == 0 ]] && echo "wheel check OK: discopt $(project_version) + pounce-solver for Pyodide $page"
  exit $fail
fi

# --- toolchain -------------------------------------------------------------
# Everything lands under `target/` so it is disposable and gitignored. Each
# step is idempotent: re-running the script re-uses what is already there.

mkdir -p "$toolchain"

# 1. A host CPython whose minor version matches Pyodide's. pyodide-build refuses
#    to install its cross-build environment otherwise, with a message ("Version
#    X is not compatible with the current environment") that does not name the
#    Python version as the reason.
py_minor=$(python3 -c 'import sys; print(sys.version_info[1])')
host_python=python3
if [[ "$py_minor" != 13 ]]; then
  host_python=$(command -v python3.13 || true)
  if [[ -z "$host_python" ]]; then
    echo "need CPython 3.13 on PATH (Pyodide $PYODIDE_VERSION is CPython 3.13); found 3.$py_minor" >&2
    exit 1
  fi
fi

venv="$toolchain/venv"
if [[ ! -x "$venv/bin/pyodide" ]]; then
  echo "==> pyodide-build $PYODIDE_BUILD_VERSION"
  "$host_python" -m venv "$venv"
  "$venv/bin/pip" install --quiet "pyodide-build==$PYODIDE_BUILD_VERSION"
fi
"$venv/bin/pyodide" xbuildenv install "$PYODIDE_VERSION" >/dev/null 2>&1 || true

# 2. emsdk — Pyodide's own, patched copy. See the header: a hand-cloned emsdk of
#    the same version cannot link a Rust side module.
emsdk="$("$venv/bin/pyodide" config get emsdk_dir)"
if [[ ! -f "$emsdk/upstream/emscripten/emcc" ]]; then
  echo "==> emsdk $EMSCRIPTEN_VERSION (Pyodide's, with its emscripten patches)"
  "$venv/bin/pyodide" xbuildenv install-emscripten
fi
if ! grep -q 'if not settings.SIDE_MODULE:' "$emsdk/upstream/emscripten/tools/emscripten.py"; then
  echo "emsdk at $emsdk is missing Pyodide's side-module export patch" >&2
  echo "re-run: $venv/bin/pyodide xbuildenv install-emscripten --force" >&2
  exit 1
fi

# 3. Rust, plus the wasm-exception-handling std. `rustup target add
#    wasm32-unknown-emscripten` installs a std built the *other* way and links
#    against it silently, so the sysroot has to be dropped in by hand.
echo "==> rust $RUST_TOOLCHAIN + $SYSROOT_RELEASE"
rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal >/dev/null
sysroot="$(rustc "+$RUST_TOOLCHAIN" --print sysroot)/lib/rustlib"
if [[ ! -d "$sysroot/wasm32-unknown-emscripten" ]]; then
  tarball="$toolchain/$SYSROOT_RELEASE.tar.bz2"
  [[ -f "$tarball" ]] || curl -sSL -o "$tarball" \
    "https://github.com/pyodide/rust-emscripten-wasm-eh-sysroot/releases/download/$SYSROOT_RELEASE/$SYSROOT_RELEASE.tar.bz2"
  mkdir -p "$sysroot"
  tar xjf "$tarball" -C "$sysroot"
fi

# --- build -----------------------------------------------------------------

# shellcheck disable=SC1091
source "$emsdk/emsdk_env.sh" >/dev/null 2>&1
export RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN"

# Debug info is roughly 10x the code size in a wasm module, and this is a
# download a visitor waits on. `lto` and `codegen-units` are left alone: neither
# affects whether the link succeeds.
export CARGO_PROFILE_RELEASE_DEBUG=0

# The wheel is built with the DEFAULT feature set, exactly like the desktop
# build. An earlier revision of this script built it `--no-default-features` to
# drop rayon, because Pyodide has no pthreads and rayon *panics* rather than
# erroring when its global pool cannot be built:
#
#   thread '<unnamed>' panicked at rayon-core-1.13.0/src/registry.rs:171:
#   The global thread pool has not been initialized.:
#   ThreadPoolBuildError { kind: IOError(Os { code: 6, kind: WouldBlock, ... }) }
#
# That did not work, and the measurement is why this comment exists: the MIQP
# example still died with the same panic, because the pool is also built inside
# `pounce-solver`, whose wheel this script does not compile. Dropping discopt's
# own feature could never have fixed it.
#
# The real fix is a capability check at each parallel entry point --
# `discopt_core::parallel::threads_available` in Rust, `_os_threads_available`
# in `solver.py` -- so a threadless platform takes the serial path that was
# already there instead of panicking past every handler. That works for POUNCE
# too, since discopt decides before calling in. With it in place there is no
# reason for the wasm build to differ from any other, and a build that differs
# is a build whose behaviour has to be reasoned about separately.

echo "==> building discopt $(project_version) for Pyodide $PYODIDE_VERSION"
rm -rf "$root/dist"
(cd "$root" && "$venv/bin/pyodide" build)

# Two wheels come out of one build and only one of them installs. maturin writes
# `...-emscripten_${EMSCRIPTEN_VERSION}_wasm32.whl` under target/wheels, then
# pyodide-build repacks it into dist/ under the newer ABI-versioned tag
# `pyemscripten_2025_0_wasm32`. The micropip that ships inside Pyodide 0.28.3
# does not know that tag and refuses it:
#
#   ValueError: Wheel was built with Emscripten vpyemscripten.2025.0
#               but Pyodide was built with Emscripten v4.0.9
#
# so the page is served the emscripten-tagged one. Revisit when the Pyodide pin
# moves — a newer micropip prefers the repacked tag.
built=$(ls "$root/target/wheels/discopt-"*"-emscripten_${EMSCRIPTEN_VERSION//./_}_wasm32.whl")
name=$(basename "$built")

mkdir -p "$wheels"
rm -f "$wheels"/discopt-*.whl

# Strip what the browser cannot use on the way in.
python3 - "$built" "$wheels/$name" <<'SLIM'
import base64, hashlib, sys, zipfile

src, dst = sys.argv[1], sys.argv[2]

# `pyproject.toml`'s `include` carries the Rust sources, Cargo.toml and
# Cargo.lock so a source build works from the wheel's sdist sibling. None of it
# is reachable at runtime, and in the browser it is a download a visitor waits
# on. The wheel also carries __pycache__ for whichever CPython built it;
# Pyodide is 3.13.
KEEP_PYC = "cpython-313"


def drop(name):
    if name.startswith("crates/") or name in ("Cargo.toml", "Cargo.lock"):
        return True
    return "__pycache__" in name and KEEP_PYC not in name


zin = zipfile.ZipFile(src)
record = next(n for n in zin.namelist() if n.endswith(".dist-info/RECORD"))
rows = []
kept = dropped = 0
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
    for info in zin.infolist():
        if info.filename == record:
            continue
        if drop(info.filename):
            dropped += 1
            continue
        data = zin.read(info.filename)
        zout.writestr(info, data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        rows.append(f"{info.filename},sha256={digest.decode()},{len(data)}")
        kept += 1
    rows.append(f"{record},,")
    zout.writestr(record, "\n".join(rows) + "\n")

# A slimmer that silently matched nothing would look exactly like a successful
# run, so say what it did and refuse a wheel it emptied.
print(f"    slimmed: kept {kept} members, dropped {dropped}")
if kept == 0:
    raise SystemExit("slimming kept nothing -- refusing to stage an empty wheel")
SLIM

# A relaxed-SIMD instruction anywhere in this module stops the *whole* module
# from compiling in a browser that lacks the proposal: wasm is validated up
# front, so an instruction nothing can reach is as fatal as one on the hot path.
# That is not hypothetical. pulp 0.22.2 — pulled in through feral, never asked
# for — compiled its wasm `RelaxedSimd` backend unconditionally, and the 46
# `f64x2.relaxed_madd` it left in three unreachable functions took POUNCE's demo
# page down with
#
#   CompileError: WebAssembly.Module doesn't parse at byte 1171:
#   relaxed simd instructions not supported, in function at index 486
#
# pulp 0.22.3 puts that backend behind a cargo feature. Nothing *pins* it off,
# though, and the failure only shows up in a browser old enough to reject it —
# so check the artifact, not the lockfile.
echo "==> extracting the extension module for artifact checks"
probe="$toolchain/_discopt_rust.so"
python3 - "$wheels/$name" "$probe" <<'EXTRACT'
import sys, zipfile
src, dst = sys.argv[1:3]
zin = zipfile.ZipFile(src)
member = next(n for n in zin.namelist() if "_rust" in n and n.endswith(".so"))
open(dst, "wb").write(zin.read(member))
print(f"    probing {member}")
EXTRACT
relaxed_bad=0
# GOT symbol names survive `-C strip=symbols`, so this needs no extra tooling
# and works on any runner.
if strings "$probe" | grep -q RelaxedSimd; then
  echo "wheel carries pulp's RelaxedSimd backend -- see the note in $(basename "${BASH_SOURCE[0]}")" >&2
  relaxed_bad=1
fi
# wabt is not installed everywhere; when it is, check the instructions directly
# rather than trusting a symbol name to stand in for them.
if command -v wasm-objdump >/dev/null 2>&1; then
  relaxed_n=$(wasm-objdump -d "$probe" | grep -cE '\b[fi][0-9]+x[0-9]+\.relaxed_[a-z0-9_]+' || true)
  if [[ "$relaxed_n" != 0 ]]; then
    echo "wheel carries $relaxed_n relaxed-SIMD instructions -- see the note above" >&2
    relaxed_bad=1
  fi
  echo "    wasm-objdump: $relaxed_n relaxed-SIMD instructions"
else
  echo "    wasm-objdump absent; checked the GOT symbol only"
fi
rm -f "$probe"
[[ $relaxed_bad == 0 ]] || exit 1

# The page reads this rather than a hard-coded file name: the ABI tag moves with
# the emscripten version, and a stale constant would surface as a micropip
# resolution error that reads like a missing package.
python3 - "$wheels/discopt-wheel.json" "$name" "$PYODIDE_VERSION" "$EMSCRIPTEN_VERSION" <<'PY'
import json, sys
path, wheel, pyodide, emscripten = sys.argv[1:5]
with open(path, "w") as fh:
    json.dump(
        {
            "wheel": wheel,
            "version": wheel.split("-")[1],
            "pyodide_version": pyodide,
            "emscripten_version": emscripten,
        },
        fh,
        indent=2,
    )
    fh.write("\n")
PY

size=$(wc -c < "$wheels/$name")
printf 'web/wheels/%s  %s bytes (%.1f MB)\n' "$name" "$size" "$(echo "$size" | awk '{print $1/1048576}')"

if [[ ! -f "$wheels/pounce-wheel.json" ]]; then
  echo
  echo "note: no pounce-solver wheel staged. It is built and committed by the" >&2
  echo "POUNCE repo; copy its crates/pounce-wasm/web-python/wheels/ contents here." >&2
fi
