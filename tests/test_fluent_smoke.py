"""Smoke test: Python can start the local Ansys Fluent, talk to it and close it.

Skipped unless PyFluent is installed AND the environment variable RUN_FLUENT=1
is set, because starting Fluent takes 1-2 minutes and needs the licence.
Normal `python -m pytest` runs stay fast and pass on PCs without Ansys.
"""

import os

import pytest

pytestmark = pytest.mark.fluent

if os.environ.get("RUN_FLUENT") != "1":
    pytest.skip(
        "Fluent smoke test off: set RUN_FLUENT=1 to launch Ansys Fluent",
        allow_module_level=True,
    )

pytest.importorskip(
    "ansys.fluent.core",
    reason="PyFluent not installed: python -m pip install -r requirements-ansys.txt",
)

from ansys.fluent_session import check_connection  # noqa: E402


def test_fluent_starts_reports_version_and_closes(tmp_path):
    result = check_connection(tmp_path)
    print(
        f"\nPyFluent {result.pyfluent_version} | Fluent {result.fluent_version} | "
        f"address {result.address} | startup {result.startup_s} s | "
        f"clean shutdown {result.shutdown_ok}"
    )
    for note in result.notes:
        print(f"  - {note}")
    assert result.fluent_version != "unknown"
    assert result.shutdown_ok
