"""
Unit Test Suite for Step 16: Production Requirements & Quality Assurance
========================================================================
Tests dependency compatibility, NumPy 2.0 guardrail, structured logging,
reproducibility seed enforcement, and environment variable fallbacks.
"""

from pathlib import Path
import os
import pytest
import numpy as np

from src.utils import get_logger, set_seed


def test_requirements_file_numpy_guardrail():
    """Verify requirements.txt enforces NumPy 2.0 incompatibility guardrail."""
    req_path = Path("requirements.txt")
    assert req_path.exists(), "requirements.txt missing"

    content = req_path.read_text(encoding="utf-8")
    assert "numpy>=1.24.0,<2.0.0" in content, "NumPy 2.0 incompatibility guardrail missing"
    assert "fastapi>=" in content
    assert "streamlit>=" in content
    assert "pydantic>=2.0.0" in content
    assert "scikit-learn>=" in content


def test_requirements_dev_file():
    """Verify requirements-dev.txt includes testing and quality tools."""
    dev_path = Path("requirements-dev.txt")
    assert dev_path.exists(), "requirements-dev.txt missing"

    content = dev_path.read_text(encoding="utf-8")
    assert "pytest" in content
    assert "pytest-cov" in content
    assert "flake8" in content


def test_env_example_declarative_schema():
    """Verify .env.example contains necessary production variables."""
    env_path = Path(".env.example")
    assert env_path.exists(), ".env.example missing"

    content = env_path.read_text(encoding="utf-8")
    assert "API_URL=" in content
    assert "ENVIRONMENT=" in content
    assert "PORT=" in content


def test_structured_logger_instantiation(tmp_path):
    """Verify structured logger writes to file and handles standard formats."""
    log_file = tmp_path / "test_run.log"
    logger = get_logger("TestLogger", log_file=str(log_file))

    logger.info("Production health test event")
    assert log_file.exists()
    assert "Production health test event" in log_file.read_text(encoding="utf-8")


def test_reproducibility_seed_deterministic():
    """Verify set_seed ensures reproducible pseudorandom generation."""
    set_seed(42)
    sample_a = np.random.rand(5)

    set_seed(42)
    sample_b = np.random.rand(5)

    assert np.allclose(sample_a, sample_b)
