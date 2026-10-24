"""
Unit Test Suite for Step 15: Deployment (Render, Streamlit Cloud & Docker)
========================================================================
Tests Docker multi-stage configuration, docker-compose orchestration,
render.yaml blueprint specification, and Streamlit Cloud configuration.
"""

from pathlib import Path
import yaml
import pytest


def test_dockerfile_structure_and_security():
    """Verify Dockerfile contains multi-stage builds, non-root user, and healthcheck."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile missing from root"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "AS builder" in content, "Missing Stage 1 builder"
    assert "AS runner" in content, "Missing Stage 2 runner"
    assert "useradd -r -g appgroup" in content, "Missing non-root user creation"
    assert "USER appuser" in content, "Container does not run as non-root user"
    assert "HEALTHCHECK" in content, "Missing container HEALTHCHECK instruction"
    assert "EXPOSE 8000 8501" in content, "Missing required ports exposure"


def test_docker_compose_configuration():
    """Verify docker-compose.yml defines both api and web services with dependencies."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml missing"

    with open(compose_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "services" in data
    assert "api" in data["services"]
    assert "web" in data["services"]

    # Verify API service configuration
    api_svc = data["services"]["api"]
    assert "8000:8000" in api_svc["ports"]
    assert "healthcheck" in api_svc

    # Verify Web service configuration
    web_svc = data["services"]["web"]
    assert "8501:8501" in web_svc["ports"]
    assert "api" in web_svc["depends_on"]
    assert "API_URL=http://api:8000" in web_svc["environment"]


def test_render_yaml_blueprint_specification():
    """Verify render.yaml declares salessight-ai-backend web service with health check."""
    render_path = Path("render.yaml")
    assert render_path.exists(), "render.yaml missing"

    with open(render_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "services" in data
    assert len(data["services"]) >= 1

    svc = data["services"][0]
    assert svc["name"] == "salessight-ai-backend"
    assert svc["type"] == "web"
    assert svc["runtime"] == "python"
    assert "uvicorn src.api.main:app" in svc["startCommand"]
    assert svc["healthCheckPath"] == "/health"


def test_streamlit_cloud_config_toml():
    """Verify .streamlit/config.toml defines required server and dark theme settings."""
    config_path = Path(".streamlit/config.toml")
    assert config_path.exists(), ".streamlit/config.toml missing"

    content = config_path.read_text(encoding="utf-8")
    assert "[theme]" in content
    assert 'primaryColor = "#2563EB"' in content
    assert 'backgroundColor = "#0E1117"' in content
    assert "[server]" in content
    assert "headless = true" in content


def test_dockerignore_file_rules():
    """Verify .dockerignore excludes cache, virtualenvs, and test directories."""
    ignore_path = Path(".dockerignore")
    assert ignore_path.exists(), ".dockerignore missing"

    content = ignore_path.read_text(encoding="utf-8")
    assert "__pycache__/" in content
    assert ".git" in content
    assert "tests/" in content
