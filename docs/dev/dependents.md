# Tracking dependent packages (issue #414)

As application/teaching code moves out of the core into satellite packages
(`discopt-doe`, `discopt-mkm`, `discopt-aggregation`, `discopt-apps`,
`discopt-course`, …), we need to (1) keep a single source of truth for **what
depends on discopt**, and (2) find out when a discopt change **breaks** one of
those dependents. This page describes the mechanism.

## Components

| File | Role |
|---|---|
| [`.github/dependents.yml`](../../.github/dependents.yml) | Machine-readable registry of dependent repos (the source of truth, incl. private repos). |
| [`.github/scripts/notify_dependents.py`](../../.github/scripts/notify_dependents.py) | Reads the registry and, per dependent, sends a `repository_dispatch` and/or opens a review issue. |
| [`.github/workflows/notify-dependents.yml`](../../.github/workflows/notify-dependents.yml) | Runs the script on every published release (and manually, as a dry run). |
| [`.github/dependent-ci-template.yml`](../../.github/dependent-ci-template.yml) | Template each dependent copies in to receive the dispatch and run its tests. |

## How it works

On a **published discopt release**, `notify-dependents.yml` runs the script,
which for each entry in `dependents.yml`:

- **`dispatch: true`** → sends a `repository_dispatch` event (type
  `discopt-updated`) to the dependent. The dependent's
  `discopt-integration.yml` workflow (from the template) catches it, installs
  discopt from `main`, and runs the dependent's own test suite. Breakage shows
  up in the dependent's Actions tab.
- **`notify: true`** → opens a *"Review / update against discopt vX.Y.Z"* issue
  in the dependent (deduplicated: it won't reopen one that already exists),
  so the update is tracked.

Releases (not every push) are the trigger, to keep per-push noise out of the
dependents' issue trackers while still catching every shipped change.

## One-time setup: the dispatch token

The default `GITHUB_TOKEN` in Actions is scoped to the current repo and
**cannot** dispatch to or open issues in other repos. You must provide a token
with cross-repo access as the `DEPENDENTS_DISPATCH_TOKEN` secret on
`jkitchin/discopt`.

**Fine-grained personal access token (quickest):**

1. GitHub → Settings → Developer settings → Fine-grained tokens → *Generate new
   token*.
2. Resource owner: `jkitchin`. Repository access: *Only select repositories* →
   pick every repo listed in `dependents.yml` (public **and** private).
3. Repository permissions:
   - **Contents: Read and write** (required to send `repository_dispatch`).
   - **Issues: Read and write** (required to open review issues).
4. Copy the token, then add it at `jkitchin/discopt` → Settings → Secrets and
   variables → Actions → *New repository secret* named
   `DEPENDENTS_DISPATCH_TOKEN`.

A GitHub App installed on the dependents is a cleaner long-term alternative
(not tied to a personal account); the workflow only needs the resulting token
in the same secret.

If the secret is absent, a real release run fails loudly with a clear error
rather than silently skipping dependents.

## Adding / onboarding a dependent

1. Add an entry to `.github/dependents.yml`:
   ```yaml
     - repo: jkitchin/discopt-newthing
       private: false      # true if the repo is private
       dispatch: true      # run its CI on discopt release
       notify: true        # open a review issue on discopt release
   ```
2. If the token is a fine-grained PAT scoped to *selected* repos, add the new
   repo to that token's repository access list.
3. In the dependent repo, copy `.github/dependent-ci-template.yml` to
   `.github/workflows/discopt-integration.yml` and adjust its install/test
   steps to that package's layout.

## Testing without a release

Run the **Notify dependents** workflow manually
(`Actions → Notify dependents → Run workflow`). It defaults to `dry_run: true`,
which logs exactly what it *would* do (dispatch / open issue) per dependent in
the run summary, without calling the write APIs.
