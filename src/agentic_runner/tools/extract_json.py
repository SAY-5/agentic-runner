"""LLM-backed structured extraction tool."""

from __future__ import annotations

import json
import warnings
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from agentic_runner.providers import build_provider
from agentic_runner.providers.base import ChatMessage
from agentic_runner.settings import get_settings
from agentic_runner.tools._base import ToolInvocationError, register_tool

warnings.filterwarnings(
    "ignore",
    message=r'Field name "schema" .* shadows an attribute in parent "BaseModel"',
    category=UserWarning,
)


class ExtractJsonInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(min_length=1, max_length=20_000)
    schema: dict[str, Any] = Field(  # type: ignore[assignment]
        description="JSON Schema describing the expected payload"
    )
    hint: str = Field(default="", max_length=500)


class ExtractJsonOutput(BaseModel):
    extracted: dict[str, Any]


def _build_validator(schema: dict[str, Any]) -> type[BaseModel]:
    if schema.get("type") != "object":
        raise ToolInvocationError("extract_json: only object schemas are supported")
    fields: dict[str, Any] = {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    for fname, fspec in properties.items():
        ftype = type_map.get(fspec.get("type", "string"), str)
        default = ... if fname in required else None
        anno = ftype if fname in required else (ftype | None)
        fields[fname] = (anno, default)
    return create_model("ExtractedSchema", **fields)


@register_tool
class ExtractJsonTool:
    name: ClassVar[str] = "extract_json"
    description: ClassVar[str] = (
        "Extract structured JSON from prose against the supplied JSON Schema."
    )
    input_model: ClassVar[type[BaseModel]] = ExtractJsonInput
    output_model: ClassVar[type[BaseModel]] = ExtractJsonOutput
    max_runtime_ms: ClassVar[int] = 3000
    idempotent: ClassVar[bool] = True

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        parsed = ExtractJsonInput.model_validate(args)
        provider = build_provider(get_settings().provider)
        strict = bool(parsed.hint)
        prompt = (
            f"EXTRACT: {parsed.text}\n"
            f"SCHEMA: {json.dumps(parsed.schema)}\n"
            f"STRICT={'1' if strict else '0'}"
        )
        resp = provider.chat([ChatMessage(role="user", content=prompt)])
        try:
            payload = json.loads(resp.text)
        except json.JSONDecodeError as exc:
            raise ToolInvocationError(f"extract_json: provider returned non-JSON: {exc}") from exc

        validator = _build_validator(parsed.schema)
        try:
            validator.model_validate(payload)
        except ValidationError as exc:
            raise ToolInvocationError(
                f"extract_json: payload does not match supplied schema: {exc}"
            ) from exc

        return {"extracted": payload}
