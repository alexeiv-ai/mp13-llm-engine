"""Compatibility import path for engine host daemon APIs.

The implementation lives in :mod:`hosting.daemon`. This module re-exports the
public API and keeps selected globals available for older callers and tests that
monkeypatch them through ``hosting.engine_host_daemon``.
"""
from __future__ import annotations

import http
import os
import signal
import subprocess
import time
from multiprocessing.connection import Client as MPClient
from multiprocessing.connection import Listener as MPListener

from .daemon import *
from .daemon import __all__
