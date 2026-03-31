# Hosting Worker Sandbox Spec And Plan

Date: 2026-03-29
Scope: Windows-first managed worker sandboxing, with explicit unsupported labels and Brokered I/O integration.

This document is both:

1. a normative specification for sandbox policy shape and enforcement expectations
2. an implementation plan with phase gates and testable exit criteria

## 1. Goals

Windows-first sandboxing for generic managed workers should target these outcomes:

1. Limit worker filesystem access to explicit allowed folders.
2. Prevent write/modify operations against daemon-owned and normal user-profile files by default.
3. Avoid inheriting parent process handles unless explicitly required.
4. Allow optional network disablement and future brokered network policy.
5. Allow optional child-process creation policy.
6. Keep current hosting daemon and worker RPC architecture usable.

Non-goals for the first Windows cut:

1. Full URL-level network enforcement at the OS boundary.
2. Strong same-account read isolation for arbitrary user files.
3. AppContainer, Docker, or VM dependency.
4. Perfect cross-platform semantic parity.

## 2. Current Baseline

Current managed workers are spawned by plain `subprocess.Popen(...)` with configurable `cwd` and `env` in [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py#L3736) and [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py#L3856).

Current baseline properties:

1. No worker sandbox policy schema exists.
2. No Windows restricted token / integrity-level lowering exists.
3. No Job Object restrictions exist for workers.
4. No brokered filesystem or brokered HTTP path exists for workers.
5. No explicit handle-inheritance hardening is enforced for worker spawn.

## 3. Normative Policy Model

This section is normative.

### 3.1 Required Top-Level Fields

Any worker sandbox policy must contain these top-level fields:

1. `enabled`
2. `profile`
3. `filesystem`
4. `process`
5. `network`
6. `brokered_io`
7. `platform_policy`

### 3.2 Required Semantics

1. `filesystem.default_access="deny"` means any direct path access not covered by an explicit rule is not trusted by policy.
2. `platform_status` labels describe implementation support, not user intent.
3. `supported` means hosting has a concrete enforcement mechanism for that permission on that platform.
4. `partial` means hosting can reduce risk or enforce only part of the requested semantics.
5. `unsupported` means hosting must not claim enforcement for that permission on that platform.
6. `network.allow_hosts` and `network.allow_url_prefixes` are broker-policy inputs, not raw OS firewall rules in the first Windows cut.
7. `brokered_io.http=true` means HTTP access must be requested over worker RPC and evaluated by hosting policy.
8. `brokered_io.filesystem=true` means sensitive filesystem access must be requested over worker RPC and evaluated by hosting policy.

### 3.3 Transport Requirement For Brokered I/O

Brokered I/O must use the existing host-managed worker IPC RPC transport:

1. Windows: named pipe (`AF_PIPE`)
2. Linux/macOS: Unix socket (`AF_UNIX`)

Brokered I/O must not require a new worker-facing HTTP listener.

Reason:

1. worker IPC transport already exists and is the trusted daemon-to-worker control path
2. adding a separate local HTTP listener would widen attack surface and duplicate auth/routing logic
3. brokered `http.fetch` should be implemented as an RPC method over pipe/socket transport, not as raw worker HTTP transport

### 3.4 Recommended First Policy Shape

Recommended first policy shape:

```json
{
  "sandbox": {
    "enabled": true,
    "profile": "generic_worker_v1",
    "platform_policy": {
      "windows": {
        "integrity_level": "low",
        "restricted_token": true,
        "job_object": true
      },
      "linux": {
        "launcher": "bwrap"
      }
    },
    "filesystem": {
      "default_access": "deny",
      "rules": [
        {
          "path": "C:\\\\workers\\\\venvs\\\\gw1",
          "access": ["read", "execute"],
          "platform_status": {
            "windows": "partial",
            "linux": "supported"
          }
        },
        {
          "path": "C:\\\\workers\\\\scratch\\\\gw1",
          "access": ["read", "write"],
          "platform_status": {
            "windows": "partial",
            "linux": "supported"
          }
        },
        {
          "path": "C:\\\\Users\\\\me\\\\.mp13-llm\\\\hosting",
          "access": [],
          "platform_status": {
            "windows": "supported",
            "linux": "supported"
          }
        }
      ]
    },
    "process": {
      "allow_subprocess": false,
      "inherit_parent_handles": false,
      "platform_status": {
        "windows": {
          "allow_subprocess": "partial",
          "inherit_parent_handles": "supported"
        },
        "linux": {
          "allow_subprocess": "supported",
          "inherit_parent_handles": "supported"
        }
      }
    },
    "network": {
      "mode": "disabled",
      "allow_hosts": [],
      "allow_url_prefixes": [],
      "platform_status": {
        "windows": {
          "disabled": "partial",
          "allow_hosts": "unsupported",
          "allow_url_prefixes": "unsupported"
        },
        "linux": {
          "disabled": "supported",
          "allow_hosts": "unsupported",
          "allow_url_prefixes": "unsupported"
        }
      }
    },
    "brokered_io": {
      "filesystem": true,
      "http": true,
      "subprocess": false
    }
  }
}
```

### 3.5 Recommended Python Structures

Recommended Python-side structures:

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

PlatformSupport = Literal["supported", "partial", "unsupported"]
FsAccess = Literal["read", "write", "execute"]
IntegrityLevel = Literal["untrusted", "low", "medium"]
NetworkMode = Literal["disabled", "direct", "brokered_only"]


@dataclass
class SandboxFsRule:
    path: str
    access: list[FsAccess] = field(default_factory=list)
    windows_status: PlatformSupport = "partial"
    linux_status: PlatformSupport = "supported"


@dataclass
class SandboxProcessPolicy:
    allow_subprocess: bool = False
    inherit_parent_handles: bool = False
    windows_allow_subprocess_status: PlatformSupport = "partial"
    windows_inherit_parent_handles_status: PlatformSupport = "supported"
    linux_allow_subprocess_status: PlatformSupport = "supported"
    linux_inherit_parent_handles_status: PlatformSupport = "supported"


@dataclass
class SandboxNetworkPolicy:
    mode: NetworkMode = "disabled"
    allow_hosts: list[str] = field(default_factory=list)
    allow_url_prefixes: list[str] = field(default_factory=list)
    windows_disabled_status: PlatformSupport = "partial"
    windows_host_allowlist_status: PlatformSupport = "unsupported"
    windows_url_allowlist_status: PlatformSupport = "unsupported"
    linux_disabled_status: PlatformSupport = "supported"
    linux_host_allowlist_status: PlatformSupport = "unsupported"
    linux_url_allowlist_status: PlatformSupport = "unsupported"


@dataclass
class WindowsSandboxPolicy:
    restricted_token: bool = True
    integrity_level: IntegrityLevel = "low"
    job_object: bool = True


@dataclass
class BrokeredIoPolicy:
    filesystem: bool = True
    http: bool = True
    subprocess: bool = False


@dataclass
class WorkerSandboxPolicy:
    enabled: bool = False
    profile: str = "generic_worker_v1"
    filesystem_rules: list[SandboxFsRule] = field(default_factory=list)
    process: SandboxProcessPolicy = field(default_factory=SandboxProcessPolicy)
    network: SandboxNetworkPolicy = field(default_factory=SandboxNetworkPolicy)
    windows: WindowsSandboxPolicy = field(default_factory=WindowsSandboxPolicy)
    brokered_io: BrokeredIoPolicy = field(default_factory=BrokeredIoPolicy)
```

### 3.6 Normative Direct-vs-Brokered Meaning

Policy interpretation rules:

1. Direct filesystem access on Windows is only trusted for write-protection outcomes explicitly marked `supported`.
2. Folder allowlist semantics on Windows must be treated as `partial` unless the access happens through brokered operations.
3. Hostname and URL allowlists must be treated as `unsupported` for direct worker networking in the first Windows cut.
4. Hostname and URL allowlists may be treated as `supported` only for brokered `http.fetch` after hosting-side validation is implemented.

## 4. Platform Support Matrix

### 4.1 Windows

| Permission / Control | Status | Notes |
|---|---|---|
| deny parent handle inheritance | supported | use `close_fds=True`, non-inheritable handles, narrow stdio passing |
| deny child processes | partial | Job Object and child-process policy help, but not full semantic parity across versions |
| deny writes to normal medium-integrity user files | supported | lower worker to Low IL |
| deny reads from normal medium-integrity user files | unsupported | integrity level does not block read-up by default |
| arbitrary folder allowlist with trusted enforcement | partial | requires ACL preparation plus low IL and/or brokering; same-account reads remain weak |
| network disabled | partial | practical via firewall/WFP-style controls or broker-only design, but not trivial in-process |
| allowed hostnames for direct worker networking | unsupported | needs broker or external mediation |
| allowed HTTP URLs for direct worker networking | unsupported | needs broker or external mediation |
| allowed hostnames for brokered `http.fetch` | planned_supported | enforced by daemon policy after Brokered I/O phase |
| allowed HTTP URLs for brokered `http.fetch` | planned_supported | enforced by daemon policy after Brokered I/O phase |
| protect daemon files from worker write | supported | daemon state remains medium-integrity and ACL-restricted |
| protect daemon files from worker read | partial | ACLs can help, but same-account readable files remain readable unless explicitly denied |

### 4.2 Linux

| Permission / Control | Status | Notes |
|---|---|---|
| deny parent handle inheritance | supported | standard spawn hygiene |
| deny child processes | supported | seccomp / launcher policy |
| filesystem allowlist | supported | `bwrap` / mount namespace model |
| network disabled | supported | namespace isolation |
| allowed hostnames for direct worker networking | unsupported | needs broker |
| allowed HTTP URLs for direct worker networking | unsupported | needs broker |
| allowed hostnames for brokered `http.fetch` | planned_supported | enforced by daemon policy after Brokered I/O phase |
| allowed HTTP URLs for brokered `http.fetch` | planned_supported | enforced by daemon policy after Brokered I/O phase |

## 5. Evaluation Of The Low Integrity Advice

Advice under review:

> A process running at a Low or Untrusted integrity level is automatically prevented by the Windows OS from writing to or modifying files at the default Medium integrity level.

Assessment:

1. This advice is substantially correct for the write/modify case.
2. It does contradict the earlier narrower statement that restricted token + Job Object alone is not enough.
3. The important correction is:
   - restricted token + Job Object alone are not enough
   - Low Integrity Level adds a real Windows boundary against write-up

What Low IL does help with:

1. Preventing writes, metadata modification, and DACL-like mutation attempts against standard medium-integrity user files.
2. Preventing worker modification of daemon-owned medium-integrity state files even when same-account DACLs would otherwise allow write.
3. Providing a meaningful first Windows containment layer without AppContainer.

What Low IL does not solve:

1. It does not reliably block reads from medium-integrity files.
2. It does not provide URL/hostname policy.
3. It does not provide a general folder allowlist by itself.
4. It does not replace handle-inheritance hardening.
5. It does not replace brokered I/O if direct access must be policy-mediated.

Revised Windows conclusion:

1. Low IL should be part of the first Windows sandbox design.
2. Restricted token + Low IL + Job Object is a sensible Windows-first starter.
3. Brokered I/O is still needed for:
   - path allowlist semantics
   - URL/host allowlists
   - reducing read exposure

## 6. Windows-First Design

### 6.1 Spawn Model

Managed worker launch should move from plain `subprocess.Popen(...)` to a sandbox-aware launcher:

1. Build effective `WorkerSandboxPolicy`.
2. Prepare worker-specific scratch and allowed directories.
3. Spawn worker with:
   - `close_fds=True`
   - minimal env
   - no inherited stdin
   - explicit stdout/stderr sinks only
4. On Windows:
   - create restricted token
   - lower token integrity level to Low
   - attach process to Job Object
   - pass only explicitly intended handles

Normative requirement:

1. Worker launch must preserve the current worker RPC contract over `AF_PIPE`.
2. Sandboxing must not require changing worker-facing control transport from pipe/socket RPC to HTTP.

### 6.2 Folder Protection Strategy

For Windows-first scope:

1. Protect daemon state and config by keeping them medium-integrity and ACL-restricted.
2. Give workers one explicit writable scratch root.
3. Treat direct worker reads outside declared roots as unsupported for strong enforcement.
4. Route sensitive file access through Brokered I/O instead of trying to enforce all path rules by token tricks alone.

### 6.3 Handle Inheritance Strategy

Required baseline:

1. `close_fds=True` on worker spawn where supported.
2. Ensure any IPC/log handles are created non-inheritable unless explicitly needed.
3. Do not hand parent daemon control handles to worker.
4. Review `stdout`/`stderr` redirection objects so they do not accidentally widen inherited-handle surface.

## 7. Brokered I/O Design

Brokered I/O is the policy enforcement layer for actions that Windows Low IL and Job Object do not express well.

### 7.1 Brokering Principles

1. Worker gets a minimal direct filesystem view.
2. Worker requests sensitive file or HTTP actions over host IPC.
3. Daemon evaluates request against `WorkerSandboxPolicy`.
4. Daemon performs allowed operations and returns structured results.

Normative transport rule:

1. Brokered I/O requests are worker RPC methods sent over existing worker IPC transport.
2. `proxy-rpc-*` on the host side remains the bridge for async/sync worker RPC.
3. No separate broker HTTP port is added between daemon and worker.

### 7.2 Brokered Filesystem API

Proposed worker RPC surface:

1. `fs.list`
2. `fs.read_text`
3. `fs.read_bytes`
4. `fs.write_text`
5. `fs.write_bytes`
6. `fs.mkdir`
7. `fs.stat`

Each request should carry:

1. `capability_id` or logical root id
2. `relative_path`
3. operation-specific payload

Daemon-side policy:

1. logical roots map to configured real paths
2. read/write/execute policy is evaluated before host access
3. path normalization must reject traversal outside root

Example:

```json
{
  "kind": "rpc",
  "method": "fs.read_text",
  "arguments": {
    "root_id": "worker_input",
    "relative_path": "task/config.json"
  }
}
```

Normative validation rules:

1. `root_id` must resolve to one configured logical root only.
2. `relative_path` must reject traversal outside root after normalization.
3. brokered `write_*` methods must fail if root policy does not include `write`.
4. brokered `read_*` methods must fail if root policy does not include `read`.

### 7.3 Brokered HTTP API

What brokered HTTP achieves:

1. It moves outbound HTTP authorization from worker code to the hosting daemon.
2. It gives hosting one policy decision point for:
   - whether network access is allowed at all
   - which hosts are allowed
   - which URL prefixes are allowed
   - which request headers are allowed to leave the host
   - how large request and response bodies may be
3. It prevents worker code from unilaterally deciding where sensitive HTTP traffic goes when policy requires broker-only networking.
4. It makes host/URL restrictions meaningful on Windows, where direct worker networking does not have a strong native allowlist mechanism in this first cut.
5. It preserves the existing worker RPC transport model by using the current pipe/socket channel instead of introducing a second network-facing control surface.

What brokered HTTP does not achieve by itself:

1. It does not magically disable all direct worker networking unless the platform sandbox also blocks or meaningfully restricts that direct path.
2. It does not provide a raw OS firewall equivalent for arbitrary worker sockets in the first Windows cut.
3. It does not make the worker unable to read arbitrary same-account files.
4. It does not by itself prove anything about remote vs local origin of the final outbound request; it only centralizes policy enforcement in hosting.
5. It does not replace the need for direct-network status labels to remain `partial` or `unsupported` where the OS boundary is weak.

Proposed worker RPC surface:

1. `http.fetch`

Payload:

1. `method`
2. `url`
3. `headers`
4. `body`
5. optional timeout

Daemon-side checks:

1. only allow when `network.mode == brokered_only`
2. match URL against allowlisted host / prefix rules
3. strip disallowed headers
4. apply body and response size caps
5. perform the outbound request on behalf of the worker only after policy passes

Normative worker-behavior rule:

1. If a worker profile declares `network.mode == brokered_only`, worker code must treat brokered `http.fetch` as the supported HTTP path.
2. Hosting must not describe direct worker HTTP egress as equivalent to brokered `http.fetch`.
3. A worker that bypasses brokered HTTP and attempts direct network egress is outside policy and must not be treated as complying with URL/host restrictions.

Normative transport note:

1. `http.fetch` is an RPC method over worker IPC pipe/socket transport.
2. It is not worker-side direct HTTP egress.
3. It is not a new daemon<->worker HTTP tunnel.

### 7.4 Brokered Subprocess API

Do not implement in first cut.

Status:

1. Windows: unsupported
2. Linux: unsupported

Rationale:

1. subprocess brokering complicates capability model significantly
2. first cut should prefer `allow_subprocess=false`

## 8. Prerequisites

### 8.1 Code And Runtime

1. Introduce `WorkerSandboxPolicy` structures in hosting.
2. Add spawn-spec fields for sandbox policy.
3. Add a dedicated Windows worker launcher module.
4. Add brokered RPC methods in daemon and worker IPC layer.
5. Add path normalization and policy evaluation helpers.
6. Add policy-to-registration serialization so diagnostics can report effective sandbox status.

### 8.2 Windows OS Primitives

1. Create restricted token helper.
2. Apply Low Integrity Level SID to worker token.
3. Create and attach Job Object.
4. Confirm spawned process inherits no unintended handles.
5. Optional later: firewall/WFP integration for stronger network disablement.

### 8.3 Testing

1. Unit tests for policy normalization and support labeling.
2. Windows-only tests for:
   - worker at Low IL cannot modify medium-integrity protected file
   - parent handles are not inherited
   - Job Object assignment succeeds
3. Integration tests for brokered filesystem RPC.
4. Negative tests for path traversal and denied brokered operations.

## 9. Implementation Plan

This section is normative for rollout order.

Each phase below has:

1. required implementation steps
2. minimum exit criteria
3. a testable outcome

### Phase 1: Policy Schema And Status Labels

1. Add policy dataclasses / JSON schema for worker sandbox.
2. Add platform support labels to every permission field.
3. Persist policy in managed worker registration metadata.

Deliverable:

1. sandbox policy can be declared, normalized, inspected, and surfaced in diagnostics

Exit criteria:

1. a worker registration can persist normalized sandbox policy
2. diagnostics can show `supported|partial|unsupported` per permission family
3. unsupported direct-network allowlist fields are surfaced explicitly, not implied

Testable outcome:

1. unit test proves policy normalization and status labeling for Windows and Linux examples

### Phase 2: Windows Spawn Hygiene

1. Add `close_fds=True` and explicit non-inheritable handle setup.
2. Audit worker stdout/stderr/log redirection for handle inheritance.
3. Add tests proving worker does not inherit daemon handles by default.

Deliverable:

1. supported `inherit_parent_handles=false`

Exit criteria:

1. worker spawn uses `close_fds=True` where applicable
2. no unintended inheritable handles are observed in worker
3. current worker logging and IPC remain functional

Testable outcome:

1. Windows test proves worker cannot use inherited parent handles and still answers RPC health/hello over pipe

### Phase 3: Windows Restricted Token + Low IL + Job Object

1. Create restricted token launcher.
2. Apply Low Integrity Level to worker token.
3. Launch worker under that token.
4. Attach worker to Job Object.
5. Add tests against medium-integrity protected files.

Deliverable:

1. supported `deny writes to medium-integrity daemon/user files`

Exit criteria:

1. worker runs with restricted token
2. worker runs at Low IL
3. worker is assigned to Job Object
4. worker can still service existing pipe RPC requests
5. worker cannot modify a medium-integrity protected file owned by the daemon user

Testable outcome:

1. Windows integration test proves a sandboxed worker can answer pipe RPC `hello`
2. the same worker fails to modify a daemon-owned medium-integrity file

First minimal outcome that can be tested end-to-end:

1. A managed worker spawned with:
   - `inherit_parent_handles=false`
   - `restricted_token=true`
   - `integrity_level=low`
   - `job_object=true`
2. still responds over existing `AF_PIPE` RPC
3. cannot modify daemon-owned medium-integrity files

This is the first required milestone before Brokered I/O work starts.

### Phase 4: Brokered Filesystem I/O

1. Add brokered filesystem RPC methods.
2. Add logical root mapping and root-relative path policy.
3. Give workers only minimal direct writable scratch path.
4. Move sensitive file access examples to broker path.

Deliverable:

1. practical filesystem allowlist model for Windows-first generic workers

Exit criteria:

1. worker can perform `fs.read_*` and `fs.write_*` only through brokered RPC for declared logical roots
2. traversal outside logical root is denied
3. direct same-account read semantics remain documented as partial/unsupported where applicable

Testable outcome:

1. integration test proves brokered read/write succeeds inside declared root and fails outside it

### Phase 5: Brokered HTTP

1. Add brokered `http.fetch`.
2. Enforce host/prefix allowlists there.
3. Keep direct network as disabled or unsupported under policy labels.

Deliverable:

1. supported host/URL policy via broker, not via raw OS sandbox

Exit criteria:

1. worker `http.fetch` goes over existing worker RPC pipe/socket transport
2. daemon enforces host/prefix allowlists
3. direct worker networking remains `partial` or `unsupported` according to platform matrix

Testable outcome:

1. integration test proves allowed brokered URL succeeds and denied URL is rejected before network egress

### Phase 6: Linux Backend

1. Add `bwrap` launcher backend.
2. Map same policy schema onto Linux support labels.
3. Keep unsupported fields explicitly marked when no direct Linux equivalent exists.

Deliverable:

1. cross-platform policy schema with intentionally different enforcement backends

Exit criteria:

1. same policy schema can be normalized on Linux
2. Linux backend marks direct filesystem/network support accurately
3. brokered RPC methods stay transport-compatible with existing worker IPC

Testable outcome:

1. Linux integration test proves `bwrap` worker still serves existing IPC RPC and brokered filesystem methods

## 10. Recommended First Cut

Recommended Windows-first MVP:

1. policy schema with support labels
2. `inherit_parent_handles=false`
3. restricted token
4. Low Integrity Level
5. Job Object
6. one writable scratch dir
7. brokered filesystem RPC

Do not block MVP on:

1. raw URL allowlists
2. raw hostname allowlists
3. full direct-read sandboxing of same-account files
4. child-process broker support

Required first acceptance slice:

1. Windows-only
2. existing worker pipe RPC remains working
3. no inherited daemon handles
4. worker cannot modify daemon-owned medium-integrity files

If this slice does not pass, the design must not claim Windows sandbox support yet.

## 11. Final Recommendation

Windows-first sandboxing is worth doing if framed correctly:

1. Low IL is a real and useful write-protection boundary.
2. It should be combined with restricted token and Job Object.
3. Brokered I/O is still needed for practical allowlist semantics.
4. The first trustworthy Windows promise should be:
   - worker cannot modify daemon and normal medium-integrity files
   - worker does not inherit parent handles
   - worker sensitive file/network access must go through daemon broker over existing pipe/socket RPC transport

That is a coherent starter design without Docker or AppContainer.
