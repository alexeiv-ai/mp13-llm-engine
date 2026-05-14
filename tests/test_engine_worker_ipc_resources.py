from __future__ import annotations

import sys

from hosting import engine_worker_ipc


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 2

    @staticmethod
    def memory_allocated(idx: int) -> int:
        return [1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024][idx]

    @staticmethod
    def memory_reserved(idx: int) -> int:
        return [3 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024][idx]


class _FakeTorch:
    cuda = _FakeCuda()


def test_worker_resource_status_reads_loaded_torch_module(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    out = engine_worker_ipc._worker_resource_status()
    data = out["result"]["data"]

    assert out["status"] == "ok"
    assert out["result"]["status"] == "success"
    assert data["current_gpu_mem_allocated_mb"] == 3072.0
    assert data["current_gpu_mem_reserved_mb"] == 7168.0
    assert data["gpu_info"] == [
        {"device_id": 0, "memory_allocated_mb": 1024.0, "memory_reserved_mb": 3072.0},
        {"device_id": 1, "memory_allocated_mb": 2048.0, "memory_reserved_mb": 4096.0},
    ]


def test_worker_resource_status_reports_pending_without_torch(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    out = engine_worker_ipc._worker_resource_status()

    assert out["status"] == "ok"
    assert out["result"]["status"] == "pending"
    assert out["result"]["message"] == "torch_module_not_loaded"
    assert out["result"]["data"]["gpu_vram_pending"] is True
