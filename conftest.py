"""
conftest.py  (repo root)
──────────────────────────
Global pytest configuration.
Disables the web3/ethereum plugin that conflicts with our Python environment.
"""


def pytest_configure(config):
    """Block the ethereum pytest plugin installed in this environment."""
    try:
        config.pluginmanager.set_blocked("ethereum")
    except Exception:
        pass

