"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from agentic_runner import models
from agentic_runner.settings import reset_settings_cache


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture()
def tmp_workspace(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        monkeypatch.setenv("AGENTIC_RUNNER_WORKSPACE_DIR", str(path))
        reset_settings_cache()
        yield path


@pytest.fixture()
def db_session() -> Iterator[sessionmaker[Session]]:
    """Return a sessionmaker bound to a fresh SQLite-in-memory DB."""
    engine = create_engine("sqlite://", future=True)
    models.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    yield factory
    engine.dispose()


@pytest.fixture()
def fake_provider():
    from agentic_runner.providers.fake import FakeProvider

    return FakeProvider()


@pytest.fixture(autouse=True)
def _force_fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_RUNNER_PROVIDER", "fake")
    reset_settings_cache()


@pytest.fixture(autouse=True)
def _reset_query_db() -> Iterator[None]:
    """Each test gets a fresh in-memory SQLite for query_db."""
    from agentic_runner.tools.query_db import reset_demo_db

    reset_demo_db()
    yield
    reset_demo_db()
