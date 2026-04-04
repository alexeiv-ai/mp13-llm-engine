from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hosting import HostedToolBoxRef
from hosting.engine_host_service import EngineHostService
from mp13_engine.mp13_toolbox import Toolbox


def SimpleCalc(expr: Optional[str] = None, **kwargs: Any) -> str:
    """
    Evaluate a simple numeric Python expression such as `2 + 2 * 10`.

    Args:
        expr (str): Numeric expression to evaluate.
    """
    expression = str(expr or "").strip()
    if not expression:
        return "Error: expr is required."
    safe_globals = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "math": math,
    }
    try:
        return str(eval(expression, safe_globals, {}))
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


def ProjectFilePeek(
    relative_path: str = "src/app/mp13chat.py",
    max_chars: int = 400,
    **kwargs: Any,
) -> str:
    """
    Read a project file and return its first characters for inspection.

    Args:
        relative_path (str): Path relative to the project root.
        max_chars (int): Maximum number of characters to return.
    """
    root = Path(kwargs.get("project_root") or Path.cwd()).resolve()
    target = (root / str(relative_path or "")).resolve()
    try:
        target.relative_to(root)
    except Exception:
        return "Error: relative_path escapes the project root."
    if not target.exists() or not target.is_file():
        return f"Error: file not found: {relative_path}"
    limit = max(1, int(max_chars or 400))
    text = target.read_text(encoding="utf-8")
    preview = text[:limit]
    return f"{relative_path}\n---\n{preview}"


def ExampleHttpPeek(
    url: str = "https://example.com/",
    max_chars: int = 300,
    **kwargs: Any,
) -> str:
    """
    Fetch a URL and return the first characters of the response body.

    Args:
        url (str): URL to fetch.
        max_chars (int): Maximum number of characters to return.
    """
    try:
        with urllib.request.urlopen(str(url), timeout=15) as resp:  # nosec B310 - intentional demo helper
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"
    limit = max(1, int(max_chars or 300))
    preview = body[:limit]
    return f"{url}\n---\n{preview}"


@dataclass
class HostedChatDemoPlan:
    toolbox_id: str
    project_root: Path
    local_tool_names: List[str]
    auto_requests: List[Dict[str, Any]]
    suggested_prompts: List[str]


@dataclass
class HostedChatDemoRuntime:
    service: EngineHostService
    toolbox_ref: HostedToolBoxRef
    plan: HostedChatDemoPlan


def build_hosted_chat_demo_plan(
    *,
    toolbox_id: str,
    project_root: Path,
) -> HostedChatDemoPlan:
    root = Path(project_root).expanduser().resolve()
    calc_source = """
import math

def SimpleCalc(expr=None, **kwargs):
    \"\"\"
    Evaluate a simple numeric Python expression such as `2 + 2 * 10`.

    Args:
        expr (str): Numeric expression to evaluate.
    \"\"\"
    expression = str(expr or '').strip()
    if not expression:
        return 'Error: expr is required.'
    safe_globals = {
        '__builtins__': {},
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'pow': pow,
        'math': math,
    }
    try:
        return str(eval(expression, safe_globals, {}))
    except Exception as exc:
        return f'Error: {type(exc).__name__}: {exc}'
""".strip() + "\n"
    file_source = """
def ProjectFilePeek(relative_path='src/app/mp13chat.py', max_chars=400, **kwargs):
    \"\"\"
    Read a project file and return its first characters for inspection.

    Args:
        relative_path (str): Path relative to the project root.
        max_chars (int): Maximum number of characters to return.
    \"\"\"
    ctx = kwargs.get('context')
    if ctx is None:
        return 'Error: missing execution context.'
    out = ctx.fs.read_text(root_id='project_ro', relative_path=str(relative_path or ''))
    text = str(dict(out or {}).get('text') or '')
    limit = max(1, int(max_chars or 400))
    preview = text[:limit]
    return f\"{relative_path}\\n---\\n{preview}\"
""".strip() + "\n"

    return HostedChatDemoPlan(
        toolbox_id=str(toolbox_id or "").strip() or "chat-hosted-demo",
        project_root=root,
        local_tool_names=["SimpleCalc", "ProjectFilePeek", "ExampleHttpPeek"],
        auto_requests=[
            {
                "relative_path": "hosted_demo_math.py",
                "content": calc_source,
                "module_name": "hosted_demo_math",
                "callable_name": "SimpleCalc",
                "environment_name": "base",
                "required_imports": [],
                "sandbox_policy": {
                    "sandbox": {
                        "enabled": True,
                    }
                },
            },
            {
                "relative_path": "hosted_demo_fs.py",
                "content": file_source,
                "module_name": "hosted_demo_fs",
                "callable_name": "ProjectFilePeek",
                "environment_name": "project-read",
                "required_imports": [],
                "sandbox_policy": {
                    "sandbox": {
                        "enabled": True,
                        "filesystem": {
                            "rules": [
                                {
                                    "root_id": "project_ro",
                                    "path": str(root),
                                    "access": ["read"],
                                }
                            ]
                        },
                        "brokered_io": {"filesystem": True, "http": False, "subprocess": False},
                    }
                },
            },
            {
                "relative_path": "hosted_demo_http.py",
                "content": """
import base64

def ExampleHttpPeek(url='https://example.com/', max_chars=300, **kwargs):
    \"\"\"
    Fetch a URL and return the first characters of the response body.

    Args:
        url (str): URL to fetch.
        max_chars (int): Maximum number of characters to return.
    \"\"\"
    ctx = kwargs.get('context')
    if ctx is None:
        return 'Error: missing execution context.'
    out = ctx.http.fetch(url=str(url or 'https://example.com/'), method='GET', timeout_seconds=15.0, max_response_bytes=65536)
    body_b64 = str(dict(out or {}).get('body_b64') or '')
    try:
        body = base64.b64decode(body_b64).decode('utf-8', errors='replace') if body_b64 else ''
    except Exception:
        body = ''
    limit = max(1, int(max_chars or 300))
    preview = body[:limit]
    return f\"{url}\\n---\\n{preview}\"
""".strip() + "\n",
                "module_name": "hosted_demo_http",
                "callable_name": "ExampleHttpPeek",
                "environment_name": "brokered-http",
                "required_imports": [],
                "sandbox_policy": {
                    "sandbox": {
                        "enabled": True,
                        "brokered_io": {"filesystem": False, "http": True, "subprocess": False},
                        "network": {
                            "mode": "brokered_only",
                            "allow_url_prefixes": ["https://example.com/"],
                        },
                    }
                },
            },
        ],
        suggested_prompts=[
            "Use SimpleCalc to evaluate 12 * (7 + 5).",
            "Use ProjectFilePeek to read src/app/mp13chat.py and show the first 300 characters.",
            "Use ExampleHttpPeek to fetch https://example.com/ and show the first 200 characters.",
            "Use both SimpleCalc and ProjectFilePeek in one answer.",
            "Use all three tools in one answer.",
        ],
    )


def register_local_hosted_chat_demo_tools(toolbox: Toolbox, *, project_root: Path) -> List[str]:
    names: List[str] = []
    if not toolbox.get_tool("SimpleCalc"):
        toolbox.add_tool_callable(SimpleCalc, activate=True)
    else:
        toolbox.activate_tool("SimpleCalc")
        toolbox.user_tool_callables["SimpleCalc"] = SimpleCalc
    names.append("SimpleCalc")

    if not toolbox.get_tool("ProjectFilePeek"):
        toolbox.add_tool_callable(ProjectFilePeek, activate=True)
    else:
        toolbox.activate_tool("ProjectFilePeek")
    toolbox.user_tool_callables["ProjectFilePeek"] = lambda **kwargs: ProjectFilePeek(project_root=project_root, **kwargs)
    names.append("ProjectFilePeek")
    if not toolbox.get_tool("ExampleHttpPeek"):
        toolbox.add_tool_callable(ExampleHttpPeek, activate=True)
    else:
        toolbox.activate_tool("ExampleHttpPeek")
    toolbox.user_tool_callables["ExampleHttpPeek"] = ExampleHttpPeek
    names.append("ExampleHttpPeek")
    return names


def setup_hosted_chat_demo(
    *,
    toolbox: Toolbox,
    hosting_root: Path,
    project_root: Path,
    toolbox_id: str,
    python_executable: Optional[str] = None,
    worker_profile_class: str = "generic",
) -> HostedChatDemoRuntime:
    plan = build_hosted_chat_demo_plan(toolbox_id=toolbox_id, project_root=project_root)
    register_local_hosted_chat_demo_tools(toolbox, project_root=plan.project_root)
    root = Path(hosting_root).expanduser().resolve()
    service = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    toolbox_ref = HostedToolBoxRef(
        toolbox_id=plan.toolbox_id,
        host=service,
        python_executable=python_executable,
        worker_profile_class=worker_profile_class,
    )
    for request in list(plan.auto_requests or []):
        toolbox_ref.register_auto_callable(
            relative_path=str(request["relative_path"]),
            content=str(request["content"]),
            module_name=str(request["module_name"]),
            callable_name=str(request["callable_name"]),
            environment_name=str(request["environment_name"]),
            required_imports=list(request.get("required_imports") or []),
            sandbox_policy=dict(request.get("sandbox_policy") or {}),
            activate=True,
        )
    return HostedChatDemoRuntime(service=service, toolbox_ref=toolbox_ref, plan=plan)


def shutdown_hosted_chat_demo(runtime: Optional[HostedChatDemoRuntime]) -> None:
    if runtime is None:
        return
    try:
        runtime.service.toolbox_unregister_auto(
            toolbox_id=runtime.plan.toolbox_id,
            tool_keys=[
                f"{str(item.get('module_name') or '').strip()}:{str(item.get('callable_name') or '').strip()}"
                for item in list(runtime.plan.auto_requests or [])
            ],
            python_executable=runtime.toolbox_ref.python_executable,
            worker_profile_class=runtime.toolbox_ref.worker_profile_class,
        )
    except Exception:
        pass
