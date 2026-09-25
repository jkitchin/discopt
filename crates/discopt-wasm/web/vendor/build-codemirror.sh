#!/usr/bin/env bash
# Rebuild the vendored CodeMirror 6 bundle.
#
# The bundle is committed, not fetched at page load. The page's whole premise is
# that the solving happens on your machine, and a CDN <script> would put a
# third-party request on every visit and break the offline and self-hosted
# deployments outright. The cost of that choice is a ~460 kB artifact in the
# tree whose provenance has to be checkable, which is what this script is for:
# `codemirror.versions.json` records exactly what went in, and re-running this
# with those versions reproduces the file.
#
# Not run by CI or by the docs build. Run it by hand to bump CodeMirror, then
# commit the regenerated bundle together with the updated versions file.
#
# Usage:  ./build-codemirror.sh            # versions from codemirror.versions.json
#         ./build-codemirror.sh --latest   # newest releases, then update that file
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
versions="$here/codemirror.versions.json"

for tool in node npm npx; do
    command -v "$tool" >/dev/null || { echo "error: $tool is required" >&2; exit 1; }
done

if [[ "${1:-}" == "--latest" ]]; then
    specs=(codemirror @codemirror/lang-python @codemirror/theme-one-dark)
    esbuild_spec=esbuild
else
    [[ -f "$versions" ]] || { echo "error: $versions is missing (try --latest)" >&2; exit 1; }
    # Pinned, so a rebuild on another machine produces the same bundle rather
    # than silently picking up whatever npm has published since.
    mapfile -t specs < <(node -e '
        const v = require(process.argv[1]).bundled;
        for (const [name, version] of Object.entries(v)) console.log(`${name}@${version}`);
    ' "$versions")
    esbuild_spec="esbuild@$(node -e 'console.log(require(process.argv[1]).esbuild)' "$versions")"
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
echo '{ "name": "discopt-cm6-vendor", "private": true, "type": "module" }' > "$work/package.json"
cp "$here/codemirror.entry.js" "$work/entry.js"

echo "installing: ${specs[*]}"
(cd "$work" && npm install --silent --no-fund --no-audit "${specs[@]}")

echo "bundling with $esbuild_spec"
(cd "$work" && npx --yes "$esbuild_spec" entry.js \
    --bundle --format=esm --minify --target=es2020 \
    --outfile=codemirror.bundle.js)

cp "$work/codemirror.bundle.js" "$here/codemirror.bundle.js"

# Record what actually went in, resolved from the lock file rather than from the
# request: `--latest` asked for a range and got specific versions, and those are
# what the committed bundle contains.
node -e '
    const lock = require(process.argv[1] + "/package-lock.json");
    const names = ["codemirror", "@codemirror/lang-python", "@codemirror/theme-one-dark"];
    const bundled = Object.fromEntries(
        names.map((n) => [n, lock.packages["node_modules/" + n].version]),
    );
    const esbuild = process.argv[2].replace(/^esbuild@?/, "") || "latest";
    process.stdout.write(JSON.stringify({ bundled, esbuild }, null, 2) + "\n");
' "$work" "$esbuild_spec" > "$versions"

echo
echo "wrote $(basename "$here")/codemirror.bundle.js ($(wc -c < "$here/codemirror.bundle.js") bytes)"
cat "$versions"
