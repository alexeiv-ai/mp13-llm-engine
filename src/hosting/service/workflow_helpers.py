from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional


class WorkflowHelperMixin:
    @staticmethod
    def workflow_js_helper_default_sandbox_policy() -> Dict[str, Any]:
        return {
            "sandbox": {
                "enabled": True,
                "profile": "workflow_js_helper_v1",
                "process": {
                    "allow_subprocess": False,
                },
                "network": {
                    "mode": "disabled",
                },
                "brokered_io": {
                    "filesystem": False,
                    "http": False,
                    "subprocess": False,
                },
            }
        }

    def spawn_workflow_js_helper(
        self,
        *,
        engine_id: str = "workflow-js-helper",
        node_executable: Optional[str] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip() or "workflow-js-helper"
        call_capacity = max(1, min(int(capacity or 1), 256))
        env = {
            "MP13_WORKER_CONTRACT": "hosting.workflow_helper.worker.v1",
            "MP13_WORKFLOW_HELPER_WORKER_ID": eid,
            "MP13_ENGINE_ID": eid,
            "MP13_WORKFLOW_JS_HELPER_CAPACITY": str(call_capacity),
        }
        src_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = str(os.environ.get("PYTHONPATH") or "").strip()
        env["PYTHONPATH"] = src_root if not existing_pythonpath else os.pathsep.join([src_root, existing_pythonpath])
        node = str(node_executable or "").strip()
        if node:
            env["MP13_WORKFLOW_JS_NODE"] = node
        policy = dict(sandbox_policy or self.workflow_js_helper_default_sandbox_policy())
        return self.spawn(
            engine_id=eid,
            command=[sys.executable, "-m", "hosting.workflow_js_helper_ipc"],
            env=env,
            worker_profile_class=str(worker_profile_class or "generic").strip() or "generic",
            sandbox_policy=policy,
            executor_kind="workflow_js_helper",
            capabilities={
                "workflow_js_helper": True,
                "execution_contract": "hosting.workflow_helper.worker.v1",
                "sandbox_profile": "workflow_js_helper_v1",
                "capacity": call_capacity,
            },
        )
