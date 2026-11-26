# discopt for the MIP Competition 2027 (container)

This directory packages discopt's entry for the
[MIP Competition 2027](https://www.mixedinteger.org/2027/competition), "Explainability and
Small Infeasible Subsystems", as a container image. The image follows the competition's
`PROGRAM_INTERFACE.md`: it is called as `<image> INSTANCE SOLUTION` and writes the JSON
described in `SOLUTION_FORMAT.md`. The solver is `python -m discopt.mipcc`
(`python/discopt/mipcc.py`). The method and its measurements are in
`docs/dev/iis-mipcc27.md`.

| File | Purpose |
|---|---|
| `Dockerfile` | two-stage build: a Rust/maturin wheel, then a slim runtime |
| `build.sh` | builds from a clean checkout and records the commit in the image |
| `constraints.txt` | exact versions of the runtime Python packages |
| `mipcc-entry` | the entrypoint; prints the build info, then `exec`s the solver |
| `selftest.mps` | the `SOLUTION_FORMAT.md` example, solved once during the build |

The repository root's `.dockerignore` is an allow-list, so only the sources the wheel is
built from go into the build context.

## Build

Run this from a clean checkout. The build context must be the repository root.

```bash
docker/mipcc27/build.sh                         # tag discopt-mipcc27:<short sha>, linux/amd64
PLATFORM=linux/arm64 docker/mipcc27/build.sh    # native build on an arm64 host
ENGINE=podman docker/mipcc27/build.sh           # build with podman instead of docker
```

You can also build by hand. `DISCOPT_COMMIT` is required:

```bash
docker build --platform linux/amd64 -f docker/mipcc27/Dockerfile \
    --build-arg DISCOPT_COMMIT="$(git rev-parse HEAD)" -t discopt-mipcc27 .
```

The build needs network access to fetch the base images, the crates and the wheels. The
finished image never touches the network. The pins are:

- the base images, by digest: `python:3.12.12-slim-bookworm` for the build and runtime
  stages, and `rust:1.91.1-slim-bookworm` for the toolchain;
- maturin, by version;
- the crates, by `Cargo.lock` (the build passes `--locked`);
- the runtime packages, by `constraints.txt`.

`CARGO_BUILD_JOBS` limits the rustc jobs (the default is 4). The last build step solves
`selftest.mps` as an unprivileged user and fails the build unless the answer is `{c1, c2}`
with both certificates.

## Run

Run the image the way the organizers do, with the flags from `PROGRAM_INTERFACE.md`:

```bash
podman run --rm --init --network none --read-only --tmpfs /tmp \
    --cap-drop ALL --security-opt no-new-privileges --userns=keep-id \
    -e MIPCC_TIME_LIMIT=3600 -e MIPCC_THREADS=8 \
    -v "$PWD/stein15inf.mps.gz:/data/stein15inf.mps.gz:ro" -v "$PWD/out:/out" \
    discopt-mipcc27 /data/stein15inf.mps.gz /out/stein15inf.json
mipcc-iis-checker --instance stein15inf.mps.gz --solution out/stein15inf.json
```

With docker, replace `--userns=keep-id` with `--user <uid>:<gid>`. The image runs under any
uid. `<image> --version` prints the build info: the commit, the discopt version, a hash of
`mipcc.py`, and the HiGHS and Python versions. Every run also prints that line to stderr
first, so each log can be traced to a commit.

## How it meets the run contract

- **Budget.** `MIPCC_TIME_LIMIT` is read at start-up, with 3600 as the default. The engine
  gets the limit minus a margin of `min(max(10, 5% of the limit), 120)` seconds. A watchdog
  thread exits with status 0 at the limit minus half that margin, whatever the engine is
  doing. With the 3600 s limit, the engine budget ends at 3480 s and the hard stop is at
  3540 s.
- **Threads.** `MIPCC_THREADS` sets the thread budget. The portfolio members split it, and
  each member's HiGHS gets an equal share. When the variable is unset, the image uses every
  core.
- **A valid answer from the start.** The first thing written is the whole model, with
  every element kept. That answer is valid whenever the instance is infeasible, which the
  competition guarantees. It stays the answer if the engine fails or runs out of time.
  Every write is atomic: a temporary file in the same directory, then a rename. So the file
  is never half-written, it is always a regular file (a symlink already at `SOLUTION` is
  replaced, not written through), and its mode is `0644`.
- **Exit status.** The entry exits 0 once a solution is on disk. That includes an engine
  exception or a timeout, where the whole model is the answer. It exits non-zero only when
  it cannot write `SOLUTION` at all, when a `MIPCC_*` variable is not a number, or when it
  is called with the wrong arguments (status 2). It never uses status 125.
- **SIGTERM.** The solver is `exec`ed under `--init`, so a SIGTERM reaches it directly. It
  terminates at once: measured 0.1 s with `docker kill -s TERM`. The portfolio members are
  separate interpreters that exit when their stdin reaches EOF. The competition discards a
  run that was stopped, so the watchdog's earlier exit is what matters.
- **Sandbox.** `HOME`, `TMPDIR` and `XDG_CACHE_HOME` point into `/tmp`, and no bytecode is
  written at run time. The only other file written is `SOLUTION` under `/out`. Both
  `.mps` and `.mps.gz` are read.
- **Certificates.** Every certificate is checked again against the original MPS data with
  the checker's 1e-5 rule, plus a safety margin. A certificate that fails is dropped,
  because one wrong certificate rejects the whole file.
- **Allowed software.** The solver uses discopt's Rust core and HiGHS (MIT). It uses no
  commercial solver.
