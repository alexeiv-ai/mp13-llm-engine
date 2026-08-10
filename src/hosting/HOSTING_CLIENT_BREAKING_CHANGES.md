# Hosting client breaking changes

Status: reset after dependent adoption (2026-08-09)

The `HOSTED-TOOLBOX-DEFINITION` migration handoff has been consumed by every
listed dependent project. `mp13-docs` adopted parent release
`83b35e20604c8f0c2fbe27467980b6a49385d918` at dependent commit
`125d20f232bf5b755d18c1b23bc1e4b8929edf21`. No pending client-breaking-change
action remains in this handoff.

Supported behavior is defined by:

- [Hosting Access §11.6](HOSTING_ACCESS.md#116-durable-hosted-operation-and-capability-contract)
- [Hosted Toolbox Definition Contract](HOSTED_TOOLBOX_CONTRACT.md)
- [Toolbox Worker](sandbox/TOOLBOX_WORKER.md)

This file is intentionally retained as the canonical path for a future
client-visible breaking change. When a new break is introduced, replace this
reset marker with the complete migration instructions before the replacement
is released.
