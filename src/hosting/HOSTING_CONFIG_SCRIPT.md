# Hosting configuration

The only static hosting authority is
`<config root>/hosting/hosting_config.json`. It uses the strict
`hosting.configuration.v3` contract and contains `control`,
`package_management`, and `environment_management` sections. Mutable keys,
sessions, audits, claims, operations, uploads, and environments are separate
records under the resolved hosting, package, and environment roots.

The top-level MP13 configuration owns the logical roots. Persistent root
definitions use only `@home`, `@config`, or `@temp`; normal hosting values use
`@hosting`, `@packages`, and `@environments`. Logical values are preserved on
disk and resolved only on the host.

## Commands

The project-root `hosting_config.py` is a thin proxy to the importable CLI:

```powershell
python hosting_config.py inspect --mp13-config-file C:\config\mp13_config.json
python hosting_config.py plan --mp13-config-file C:\config\mp13_config.json `
  --hosting-root '@config/host-data' `
  --packages-root '@config/package-data' `
  --environments-root '@config/environment-data' `
  --hosting-config-file C:\staging\hosting_config.json
python hosting_config.py apply --mp13-config-file C:\config\mp13_config.json `
  --hosting-root '@config/host-data' `
  --packages-root '@config/package-data' `
  --environments-root '@config/environment-data' `
  --hosting-config-file C:\staging\hosting_config.json `
  --expected-config-revision sha256:... `
  --expected-hosting-revision sha256:... --confirm
```

The same behavior is available from `hosting.hosting_setup_api` through
plan/apply/inspect/status/reset functions and the `hosting.setup.v1` request.
Apply and reset are host-local and require explicit confirmation. Root changes
use optimistic revisions, preflight validation, restrictive atomic writes, and
a recovery journal. Active-daemon relocation, non-empty destinations, and
cross-volume relocation are refused unless the local plan explicitly permits
the applicable condition.

Inspection reports logical roots and subsystem health. Resolved paths are
available only from host-local administrative inspection. Remote status never
returns credential values, tokens, key material, sensitive source queries, or
unrestricted host paths.

## Static schema

```json
{
  "contract": "hosting.configuration.v3",
  "control": {
    "authentication": {},
    "roles": {},
    "session_policy": {},
    "audit": {}
  },
  "package_management": {
    "artifact_root": "@packages/artifacts",
    "lock_root": "@packages/locks",
    "sources": {},
    "credentials": {},
    "dependency_policy": {},
    "verification": {"hash_algorithm": "sha256"}
  },
  "environment_management": {
    "environment_root": "@environments",
    "scratch_root": "@hosting/scratch",
    "retention": {},
    "cache": {}
  }
}
```

Unknown fields, wrong types, unsupported contracts, unresolved or escaping
labels, unsafe authentication policy, missing credential references, and a
non-SHA-256 baseline are rejected before replacement. Static changes take
effect after a deliberate daemon restart; the daemon never rewrites this file.
