"""
Shared test fixtures.

Hard invariant (CLAUDE.md §2): **don't mock data sources**. Tests that need
network hit the real API; they skip cleanly when the network is down.
"""

from __future__ import annotations

import socket

import pytest


def _has_network(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def has_network() -> bool:
    return _has_network()


@pytest.fixture(autouse=True)
def _skip_when_offline(request, has_network):
    if request.node.get_closest_marker("network") and not has_network:
        pytest.skip("network unavailable")
