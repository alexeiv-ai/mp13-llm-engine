import asyncio

from mp13_engine import mp13_engine_api as api
from mp13_engine.mp13_errors import EngineInitializationError


class FailingEngine:
    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id

    async def initialize_global_resources(self, config):
        raise EngineInitializationError(
            {
                "message": "Global Engine Initialization Failed: OSError: missing model",
                "details": {"errors": ["missing model"]},
            }
        )


def test_failed_initialize_cleans_registered_engine_and_reports_details() -> None:
    original_class = api._MP13_ENGINE_CLASS
    original_instances = dict(api._ENGINE_INSTANCES)
    original_aliases = dict(api._ALIAS_TO_ID)
    original_default = api._DEFAULT_ENGINE_ALIAS
    try:
        api._MP13_ENGINE_CLASS = FailingEngine
        api._ENGINE_INSTANCES.clear()
        api._ALIAS_TO_ID.clear()
        api._DEFAULT_ENGINE_ALIAS = None

        resp = asyncio.run(
            api.handle_call_tool(
                "initialize-engine",
                {"base_model_name_or_path": "C:/models/missing"},
            )
        )

        assert resp.status == "error"
        assert "missing model" in resp.message
        assert resp.details == {"errors": ["missing model"]}
        assert api._ENGINE_INSTANCES == {}
        assert api._ALIAS_TO_ID == {}
        assert api._DEFAULT_ENGINE_ALIAS is None
    finally:
        api._MP13_ENGINE_CLASS = original_class
        api._ENGINE_INSTANCES.clear()
        api._ENGINE_INSTANCES.update(original_instances)
        api._ALIAS_TO_ID.clear()
        api._ALIAS_TO_ID.update(original_aliases)
        api._DEFAULT_ENGINE_ALIAS = original_default
