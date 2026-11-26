#!/bin/sh
# Build the MIPcc27 image from a clean checkout, recording the commit in it.
#   docker/mipcc27/build.sh [TAG]          (default tag: discopt-mipcc27:<short sha>)
# Environment: PLATFORM (default linux/amd64), ENGINE (docker or podman, default
# docker), CARGO_BUILD_JOBS (default 4), ALLOW_DIRTY=1 to build uncommitted changes
# (the commit is then recorded with a "-dirty" suffix).
set -eu
root=$(git rev-parse --show-toplevel)
cd "$root"
commit=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    if [ "${ALLOW_DIRTY:-0}" != 1 ]; then
        echo "build.sh: the tree has uncommitted changes; commit them or set ALLOW_DIRTY=1" >&2
        exit 1
    fi
    commit="${commit}-dirty"
fi
tag=${1:-discopt-mipcc27:$(git rev-parse --short=9 HEAD)}
${ENGINE:-docker} build --platform "${PLATFORM:-linux/amd64}" \
    -f docker/mipcc27/Dockerfile \
    --build-arg DISCOPT_COMMIT="$commit" \
    --build-arg CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}" \
    -t "$tag" .
echo "built $tag ($commit)"
