"""Structured tracing helpers backed by OpenTelemetry."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

_LOG_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Set up structlog + OpenTelemetry tracer once per process."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level, logging.INFO)),
        cache_logger_on_first_use=True,
    )

    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())

    _LOG_CONFIGURED = True


def get_logger(name: str) -> Any:
    configure_logging()
    return structlog.get_logger(name)


def get_tracer(name: str = "agentic_runner") -> trace.Tracer:
    configure_logging()
    return trace.get_tracer(name)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[trace.Span]:
    """Convenience span context manager that records attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as sp:
        for key, value in attrs.items():
            if value is not None:
                sp.set_attribute(key, value)
        yield sp
