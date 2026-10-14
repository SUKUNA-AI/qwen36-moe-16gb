# Remaining review findings

Scope: the changed CPU/harness/config/provenance paths and the proposed decode
intersection, not a whole-repository T5 audit. Fixed findings are documented in
`NEXT_UPDATE_HANDOFF.md`; they are not listed as open bugs here.

## M1 — Public operational launchers still need owner-local site configuration

**Severity: medium; confirmed.** `harness/run-experiment.sh:3`,
`harness/experiment.py:4`, and `harness/profile.sh:8` retain sanitized runtime,
model/toolchain and profiler paths. Read-only reproduction:

```bash
rg -n '/home/USER|/path/to/LLM' harness/run-experiment.sh harness/experiment.py harness/profile.sh
```

The public CPU entry point works without those launchers. Executing them directly
as a GPU runbook in an arbitrary checkout can select nonexistent paths and cannot
establish that the intended overlay was imported. The new preflight rejects
unconfigured native launchers. Owner-local experiment launcher configuration and
reviewed staging remain prerequisites; production/install/sync scripts were not
migrated in this task. No cloud service operation was attempted.

## M2 — Source exporter still assumes a standalone upstream-based fork Git tree

**Severity: medium; confirmed.** `harness/export-release-evidence.py:273` resolves
the correct source directory, but lines 274–277 require the upstream
`cac247a860e316e06580d05aeb05f2e647bde214` object and `HEAD == BASE`. This public
repository contains the fork under `freetoken/`, with monorepo history. Read-only
reproduction in this checkout:

```bash
git -C freetoken cat-file -e cac247a860e316e06580d05aeb05f2e647bde214
```

Observed exit 1: the object is absent. Even fetching it would not fix the HEAD/path
mapping assumptions. Feature mapping now covers threshold/FLA and the added CPU
helpers, and all exporter fixture tests execute, but a **real source export from
this monorepo remains unavailable**. Use reviewed Git diffs/commits for this
handoff. A monorepo/upstream export migration needs its own explicit base and path
mapping; do not weaken the exporter's refusal or manufacture an upstream basis.

No additional CUDA race is claimed as confirmed. T1 GPU transport/model behavior
and T2 fused execution remain unvalidated on SM120. T3 remains DEFERRED with its
specific ownership/capture requirements in
`next-update/gather-prefetch/OVERLAP-DESIGN.md`; no unsafe overlap implementation
or active flag was left behind.
