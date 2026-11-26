# syntax=docker/dockerfile:1
#
# discopt's entry for the MIP Competition 2027 (small infeasible subsystems).
# Build from the repository root (the context must be the repo root):
#
#   docker/mipcc27/build.sh                      # or, by hand:
#   docker build --platform linux/amd64 -f docker/mipcc27/Dockerfile \
#       --build-arg DISCOPT_COMMIT="$(git rev-parse HEAD)" -t discopt-mipcc27 .
#
# Run:  <image> INSTANCE SOLUTION   (see docker/mipcc27/README.md)
#
# Base images are pinned by digest (multi-arch indexes: linux/amd64 and
# linux/arm64 both resolve), the Rust toolchain by version, and every Python
# package by exact version (constraints.txt). The Rust crates are pinned by
# Cargo.lock (`--locked`).

ARG PYTHON_IMAGE=docker.io/library/python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
ARG RUST_IMAGE=docker.io/library/rust:1.91.1-slim-bookworm@sha256:8514999d4786ef12efe89239e86b3d0a021b94b9d35108c8efe6c79ca7dc1a65

FROM ${RUST_IMAGE} AS rust

# ---------------------------------------------------------------- build stage
FROM ${PYTHON_IMAGE} AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# Parallel rustc jobs for the build (not a runtime setting).
ARG CARGO_BUILD_JOBS=4
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
RUN pip install maturin==1.14.1

WORKDIR /src
# Only what the wheel is built from (the root .dockerignore is an allow-list).
COPY Cargo.toml Cargo.lock pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY crates ./crates
COPY python/discopt ./python/discopt
RUN maturin build --release --locked --strip --out /wheels \
    && ls -l /wheels

# -------------------------------------------------------------- runtime stage
FROM ${PYTHON_IMAGE} AS runtime
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY docker/mipcc27/constraints.txt /opt/discopt/constraints.txt
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install -c /opt/discopt/constraints.txt /wheels/discopt-*.whl

# The build the image was made from, printed by the entry on every run (and by
# `<image> --version`), so a result can be traced to a commit.
ARG DISCOPT_COMMIT
RUN test -n "${DISCOPT_COMMIT}" \
        || { echo "pass --build-arg DISCOPT_COMMIT=\$(git rev-parse HEAD)" >&2; exit 1; } \
    && python - "${DISCOPT_COMMIT}" <<'EOF'
import hashlib, importlib.metadata as md, sys
import discopt, discopt.mipcc, discopt._rust
# The installed package, not a stray copy on sys.path.
assert "/site-packages/discopt/" in discopt.__file__, discopt.__file__
src = open(discopt.mipcc.__file__, "rb").read()
info = [
    f"discopt_commit={sys.argv[1]}",
    f"discopt_version={md.version('discopt')}",
    f"mipcc_sha256={hashlib.sha256(src).hexdigest()[:16]}",
    f"highspy={md.version('highspy')}",
    f"python={sys.version.split()[0]}",
]
open("/opt/discopt/BUILD_INFO", "w").write("\n".join(info) + "\n")
print("\n".join(info))
EOF
LABEL org.opencontainers.image.title="discopt-mipcc27" \
      org.opencontainers.image.description="discopt IIS entry for the MIP Competition 2027" \
      org.opencontainers.image.source="https://github.com/jkitchin/discopt" \
      org.opencontainers.image.licenses="EPL-2.0" \
      org.opencontainers.image.revision="${DISCOPT_COMMIT}"

COPY docker/mipcc27/mipcc-entry /usr/local/bin/mipcc-entry
COPY docker/mipcc27/selftest.mps /opt/discopt/selftest.mps

# Sandbox (PROGRAM_INTERFACE.md): read-only root, writable /tmp and /out only,
# an arbitrary non-root uid, HOME possibly unwritable. Everything that would
# write under HOME (discopt's cache, XDG caches) goes to /tmp; no bytecode is
# written at run time (pip compiled it at install).
ENV HOME=/tmp \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Self-test at build time, as an unprivileged user writing only to /tmp: the
# SOLUTION_FORMAT.md example must give {c1, c2} with both certificates.
USER 10001:10001
RUN cd /tmp && MIPCC_TIME_LIMIT=60 MIPCC_THREADS=2 \
        mipcc-entry /opt/discopt/selftest.mps /tmp/selftest.json \
    && python - <<'EOF'
import json
doc = json.load(open("/tmp/selftest.json"))
rows = doc["rows"]
assert rows["c3"] == {"action": "relax"}, rows
assert all(rows[c]["action"] == "keep" and "certificate" in rows[c] for c in ("c1", "c2")), rows
print("selftest: OK", rows)
import os; os.remove("/tmp/selftest.json")
EOF

ENTRYPOINT ["/usr/local/bin/mipcc-entry"]
