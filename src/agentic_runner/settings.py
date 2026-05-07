"""Runtime settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings."""

    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_RUNNER_",
        env_file=".env",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/agentic_runner",
        description="SQLAlchemy database URL",
    )
    provider: str = Field(default="fake", description="Default LLM provider")
    workspace_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "workspace",
        description="Sandboxed file root for read_file / write_file",
    )
    http_allowlist: tuple[str, ...] = Field(
        default=("localhost", "127.0.0.1", "example.com"),
        description="Allowed hosts for http_get tool",
    )
    max_tool_runtime_ms: int = Field(default=5000, description="Per-tool timeout")
    max_file_bytes: int = Field(default=64_000, description="Cap for read_file/write_file")
    log_level: str = Field(default="INFO")


_cached: Settings | None = None


def get_settings() -> Settings:
    """Lazy-cached settings accessor."""
    global _cached
    if _cached is None:
        _cached = Settings()
    return _cached


def reset_settings_cache() -> None:
    """Reset the settings cache (used in tests)."""
    global _cached
    _cached = None
