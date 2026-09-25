"""The Render blueprint has to stay valid and consistent with the app.

A malformed or stale ``render.yaml`` fails the deploy rather than the build, so
it is worth asserting its contract here: the runtime, the commands, the health
check path the API actually serves, and that the requirements file it installs
is the API-only one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BLUEPRINT = PROJECT_ROOT / "render.yaml"


@pytest.fixture(scope="module")
def service() -> dict:
    data = yaml.safe_load(BLUEPRINT.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "render.yaml must be a mapping"
    services = data.get("services")
    assert isinstance(services, list) and services, "render.yaml must define at least one service"
    return services[0]


def test_blueprint_is_a_web_service(service: dict) -> None:
    assert service["type"] == "web"
    # Render's native Python runtime also ships node/npm, which is what lets the
    # frontend build in the same service instead of needing Docker.
    assert service["runtime"] == "python"


def test_build_command_installs_the_api_requirements(service: dict) -> None:
    command = service["buildCommand"]
    assert "requirements-api.txt" in command
    # It must not pull in the Streamlit app's requirements, which would drag
    # streamlit, scikit-learn and XGBoost back into the deployment.
    assert "requirements.txt" not in command.replace("requirements-api.txt", "")
    assert "requirements-web.txt" not in command


def test_build_command_builds_the_frontend(service: dict) -> None:
    command = service["buildCommand"]
    assert "npm ci" in command
    assert "npm run build" in command


def test_start_command_matches_the_entrypoint(service: dict) -> None:
    assert service["startCommand"].strip() == "python -m api"


def test_health_check_path_is_one_the_api_serves(service: dict) -> None:
    """The path must exist and stay unauthenticated, or deploys never go live."""
    from api.main import app

    assert service["healthCheckPath"] == "/api/health"
    assert any(getattr(route, "path", None) == "/api/health" for route in app.routes)


def test_binds_all_interfaces(service: dict) -> None:
    """Render only reaches the process if it binds 0.0.0.0."""
    env = {item["key"]: item.get("value") for item in service["envVars"]}
    assert env.get("TAILGATE_HOST") == "0.0.0.0"


def test_secrets_are_not_stored_in_the_blueprint(service: dict) -> None:
    """CFBD_API_KEY must be set in the dashboard, never committed."""
    env = {item["key"]: item for item in service["envVars"]}
    assert "CFBD_API_KEY" in env
    assert env["CFBD_API_KEY"].get("sync") is False
    assert "value" not in env["CFBD_API_KEY"]


def test_region_is_pinned(service: dict) -> None:
    """Render cannot move a service between regions after creation.

    Leaving this out means every re-create silently takes the dashboard default,
    which is how the service ended up in Oregon while its users were elsewhere.
    """
    assert service.get("region") in {"oregon", "ohio", "virginia", "frankfurt", "singapore"}


def test_every_env_var_has_a_key(service: dict) -> None:
    """Render rejects the blueprint if an env var entry is missing its key."""
    for item in service["envVars"]:
        assert item.get("key"), item
