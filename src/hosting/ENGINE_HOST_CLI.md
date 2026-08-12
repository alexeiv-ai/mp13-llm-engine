# Engine host CLI

The hosting daemon starts from one host-local input: the top-level MP13
configuration file. That file resolves the unified `hosting.configuration.v3`
authority and all hosting/package/environment roots before any externally
reachable listener is bound.

```powershell
python -m hosting.engine_host_cli --daemon `
  --mp13-config-file C:\config\mp13_config.json

python -m hosting.engine_host_cli --daemon --background `
  --mp13-config-file C:\config\mp13_config.json

python -m hosting.engine_host_cli --daemon-http `
  --mp13-config-file C:\config\mp13_config.json
```

Local channel settings use `engine_host_mp13_config_file`. SSH relay startup
forwards the same option to the host; it does not forward credentials, package
source values, trust material, or resolved root paths in process arguments.

The daemon/control contract is `hosting.control.v3`, major `3`, beginning with
daemon version `3.0.0`. A request envelope contains `contract`, `request_id`,
`command`, and `payload`. Unsupported or absent majors fail closed with
`hosting_contract_major_unsupported`.

Startup health distinguishes control configuration from package and
environment subsystem health. Authorized diagnostic sessions remain usable
when a non-control subsystem is degraded. Remote health output contains the
configuration contract/revision, logical roots, bounded source availability,
and bounded environment health; host paths and secrets are local-only.

Use `python -m hosting.engine_host_cli --help` for the current command surface.
Static policy is changed only with `hosting_config.py`; daemon commands mutate
only authorized dynamic records.
