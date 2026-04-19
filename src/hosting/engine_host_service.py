"""
Compatibility import path for the engine host service.

The implementation lives in :mod:`hosting.service.host_service`. This module is
kept as an alias so existing imports and test monkeypatches against
``hosting.engine_host_service`` continue to target the implementation module.
"""
from __future__ import annotations

import sys as _sys

from .service import host_service as _host_service

_sys.modules[__name__] = _host_service

