"""Deterministic FakeProvider for hermetic CI.

The provider keys responses off the ``role=user`` content of the latest
message so each prompt deterministically yields the same plan/tool-call
pair. The scripted table below covers the 15 eval goals plus a few
intentionally-failing scenarios that exercise the replan path.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from agentic_runner.providers.base import (
    ChatMessage,
    ChatResponse,
    ToolCallRequest,
    ToolSpec,
)

# ---------------------------------------------------------------------------
# Scripted plans: each goal hash maps to an ordered list of plans.
# Successive replans pick the next plan in the list.
# ---------------------------------------------------------------------------

_PLAN_SCRIPT: dict[str, list[list[dict[str, Any]]]] = {
    "calc_avg_eng": [
        [
            {
                "description": "Query the average salary in the engineering department",
                "tool_hint": "query_db",
            },
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "calc_simple": [
        [
            {"description": "Compute the arithmetic expression", "tool_hint": "calculate"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "list_employees": [
        [
            {"description": "Query all employees", "tool_hint": "query_db"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "count_orders": [
        [
            {"description": "Count rows in orders", "tool_hint": "query_db"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "read_notes": [
        [
            {"description": "Read workspace notes file", "tool_hint": "read_file"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "write_then_read": [
        [
            {"description": "Write the report to the workspace", "tool_hint": "write_file"},
            {"description": "Read it back to confirm", "tool_hint": "read_file"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "fetch_status": [
        [
            {"description": "Fetch the example.com homepage", "tool_hint": "http_get"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "summarize_doc": [
        [
            {"description": "Read the document", "tool_hint": "read_file"},
            {"description": "Summarize it", "tool_hint": "summarize"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "extract_struct": [
        [
            {"description": "Extract structured data from the prose", "tool_hint": "extract_json"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "math_two_step": [
        [
            {"description": "Compute the first sub-expression", "tool_hint": "calculate"},
            {"description": "Compute the final expression", "tool_hint": "calculate"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "department_count": [
        [
            {"description": "Count distinct departments", "tool_hint": "query_db"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "find_john_salary": [
        [
            {"description": "Look up John's salary in the database", "tool_hint": "query_db"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ],
        [
            {"description": "Read salaries from the workspace file", "tool_hint": "read_file"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ],
    ],
    "send_email": [
        [
            {
                "description": "Send the email to alice@example.com",
                "tool_hint": "send_email",  # not registered — triggers abort
            }
        ]
    ],
    "summarize_short": [
        [
            {
                "description": "Summarize with a strict short word cap",
                "tool_hint": "summarize",
                "args_override": {"max_words": 5, "strict": True},
            },
            {"description": "Return the final answer", "tool_hint": "finish"},
        ]
    ],
    "extract_strict": [
        [
            {"description": "Extract the JSON", "tool_hint": "extract_json"},
            {"description": "Return the final answer", "tool_hint": "finish"},
        ],
        [
            {
                "description": "Extract the JSON with explicit hint",
                "tool_hint": "extract_json",
                "args_override": {"hint": "use the structured prompt"},
            },
            {"description": "Return the final answer", "tool_hint": "finish"},
        ],
    ],
}


_TOOL_ARGS: dict[tuple[str, str], list[dict[str, Any]]] = {
    ("calc_avg_eng", "query_db"): [
        {"sql": "SELECT AVG(salary) AS avg_salary FROM employees " "WHERE department='engineering'"}
    ],
    ("calc_avg_eng", "finish"): [{"result": "Average engineering salary: 101666.7"}],
    ("calc_simple", "calculate"): [{"expression": "(2+3)*7"}],
    ("calc_simple", "finish"): [{"result": "Result: 35"}],
    ("list_employees", "query_db"): [{"sql": "SELECT name, department FROM employees"}],
    ("list_employees", "finish"): [{"result": "Listed employees from the employees table."}],
    ("count_orders", "query_db"): [{"sql": "SELECT COUNT(*) AS n FROM orders"}],
    ("count_orders", "finish"): [{"result": "Order count returned."}],
    ("read_notes", "read_file"): [{"path": "notes.txt"}],
    ("read_notes", "finish"): [{"result": "Notes file contents returned."}],
    ("write_then_read", "write_file"): [
        {"path": "report.txt", "content": "Quarterly report content."}
    ],
    ("write_then_read", "read_file"): [{"path": "report.txt"}],
    ("write_then_read", "finish"): [{"result": "Report written and read back."}],
    ("fetch_status", "http_get"): [{"url": "http://example.com/"}],
    ("fetch_status", "finish"): [{"result": "Homepage fetched successfully."}],
    ("summarize_doc", "read_file"): [{"path": "long_doc.txt"}],
    ("summarize_doc", "summarize"): [{"text": "__USE_PRIOR_OUTPUT__", "max_words": 20}],
    ("summarize_doc", "finish"): [{"result": "Document summary returned."}],
    ("extract_struct", "extract_json"): [
        {
            "text": "Order #42 placed on 2024-03-01 for $99.50",
            "schema": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer"},
                    "date": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["order_id", "date", "amount"],
            },
            "hint": "extract structured fields",
        }
    ],
    ("extract_struct", "finish"): [{"result": "Structured order extracted."}],
    ("math_two_step", "calculate"): [{"expression": "10*5"}, {"expression": "50+25"}],
    ("math_two_step", "finish"): [{"result": "Final value: 75"}],
    ("department_count", "query_db"): [
        {"sql": "SELECT COUNT(DISTINCT name) AS n FROM departments"}
    ],
    ("department_count", "finish"): [{"result": "Distinct department count returned."}],
    ("find_john_salary", "query_db"): [
        {"sql": "SELECT salary FROM employees WHERE name='John Doe'"}
    ],
    ("find_john_salary", "read_file"): [{"path": "salaries.txt"}],
    ("find_john_salary", "finish"): [{"result": "John's salary located in salaries.txt."}],
    ("send_email", "send_email"): [{"to": "alice@example.com", "body": "hi"}],
    ("summarize_short", "summarize"): [
        {
            "text": "The quick brown fox jumps over the lazy dog repeatedly",
            "max_words": 5,
            "strict": True,
        },
    ],
    ("summarize_short", "finish"): [{"result": "Short summary returned."}],
    ("extract_strict", "extract_json"): [
        {
            "text": "Maybe order 7? unclear date, amount around 10",
            "schema": {
                "type": "object",
                "properties": {"order_id": {"type": "integer"}},
                "required": ["order_id"],
            },
        },
        {
            "text": "Maybe order 7? unclear date, amount around 10",
            "schema": {
                "type": "object",
                "properties": {"order_id": {"type": "integer"}},
                "required": ["order_id"],
            },
            "hint": "use the structured prompt",
        },
    ],
    ("extract_strict", "finish"): [{"result": "Order id extracted."}],
}


_GOAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Order matters: most-specific first, generic last.
    (re.compile(r"\(2\+3\)\*7", re.I), "calc_simple"),
    (re.compile(r"average salary.*engineering", re.I), "calc_avg_eng"),
    (re.compile(r"two-step|10\*5", re.I), "math_two_step"),
    (re.compile(r"distinct.*departments|count.*departments", re.I), "department_count"),
    (re.compile(r"find.*john.*salary", re.I), "find_john_salary"),
    (re.compile(r"send.*email.*alice", re.I), "send_email"),
    (re.compile(r"short summary.*fox|summary.*fox.*strict", re.I), "summarize_short"),
    (re.compile(r"extract.*unclear|extract.*strict", re.I), "extract_strict"),
    (re.compile(r"summarize.*long_doc|summarize.*document", re.I), "summarize_doc"),
    (re.compile(r"extract.*order.*structured|extract json from order", re.I), "extract_struct"),
    (re.compile(r"list (all )?employees", re.I), "list_employees"),
    (re.compile(r"count.*orders|rows.*orders", re.I), "count_orders"),
    (re.compile(r"read.*notes", re.I), "read_notes"),
    (re.compile(r"write.*report.*read|read.*back", re.I), "write_then_read"),
    (re.compile(r"fetch.*example\.com|http_get example", re.I), "fetch_status"),
]


def _classify_goal(text: str) -> str | None:
    for pattern, key in _GOAL_PATTERNS:
        if pattern.search(text):
            return key
    return None


class FakeProvider:
    """Deterministic provider that scripts plans + tool calls."""

    name = "fake"

    def __init__(self) -> None:
        self._state: dict[str, dict[str, Any]] = {}

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        user_msgs = [m for m in messages if m.role == "user"]
        if not user_msgs:
            return _empty_response()

        latest = user_msgs[-1].content
        anchor = user_msgs[0].content
        goal_key = _classify_goal(anchor) or _classify_goal(latest)

        if latest.startswith("PLAN:"):
            return self._plan(goal_key)
        if latest.startswith("SELECT:"):
            return self._select(goal_key, latest)
        if latest.startswith("SUMMARIZE:"):
            return self._summarize(latest)
        if latest.startswith("EXTRACT:"):
            return self._extract(latest)

        return _empty_response(text="(fake-provider: no scripted handler)")

    def _plan(self, goal_key: str | None) -> ChatResponse:
        if goal_key is None or goal_key not in _PLAN_SCRIPT:
            return _empty_response(text=json.dumps({"subtasks": []}))

        plans = _PLAN_SCRIPT[goal_key]
        st = self._state.setdefault(goal_key, {"plan_idx": 0, "tool_idx": {}})
        idx = min(st["plan_idx"], len(plans) - 1)
        plan = plans[idx]
        st["plan_idx"] = idx + 1
        return ChatResponse(
            text=json.dumps({"subtasks": plan}),
            tokens_in=120,
            tokens_out=80,
            cost_usd=0.0008,
            model_version="fake-1.0",
        )

    def _select(self, goal_key: str | None, prompt: str) -> ChatResponse:
        if goal_key is None:
            return _empty_response()

        m = re.search(r"SELECT:\s*(\S+)", prompt)
        if not m:
            return _empty_response()
        tool_name = m.group(1)

        st = self._state.setdefault(goal_key, {"plan_idx": 0, "tool_idx": {}})
        tool_idx_map = st["tool_idx"]
        per_tool = tool_idx_map.get(tool_name, 0)
        args_list = _TOOL_ARGS.get((goal_key, tool_name), [])
        if not args_list:
            return ChatResponse(
                text=f"(no scripted args for {goal_key}/{tool_name})",
                tokens_in=20,
                tokens_out=5,
                cost_usd=0.0001,
                model_version="fake-1.0",
            )

        args = args_list[min(per_tool, len(args_list) - 1)]
        tool_idx_map[tool_name] = per_tool + 1

        return ChatResponse(
            tool_calls=[
                ToolCallRequest(id=str(uuid.uuid4()), name=tool_name, arguments=dict(args))
            ],
            tokens_in=80,
            tokens_out=30,
            cost_usd=0.0004,
            model_version="fake-1.0",
        )

    def _summarize(self, prompt: str) -> ChatResponse:
        strict = "STRICT=1" in prompt
        words = (
            "Quick fox jumps repeatedly."
            if strict
            else "The quick brown fox jumps over the lazy dog and continues running endlessly."
        )
        return ChatResponse(
            text=words,
            tokens_in=50,
            tokens_out=20,
            cost_usd=0.0002,
            model_version="fake-1.0",
        )

    def _extract(self, prompt: str) -> ChatResponse:
        """Build a payload conforming to the schema embedded in the prompt.

        ``STRICT=1`` makes every field conformant; ``STRICT=0`` injects a
        type-mismatched value into the first numeric field to exercise
        the validator-driven replan path.
        """
        strict = "STRICT=1" in prompt
        schema: dict[str, Any] = {}
        for line in prompt.splitlines():
            if line.startswith("SCHEMA:"):
                try:
                    schema = json.loads(line[len("SCHEMA:") :].strip())
                except json.JSONDecodeError:
                    schema = {}
                break

        properties = schema.get("properties", {})
        sample_values = {
            "integer": 7,
            "number": 99.5,
            "string": "2024-03-01",
            "boolean": True,
        }
        payload: dict[str, Any] = {}
        first_field = next(iter(properties), None)
        for fname, fspec in properties.items():
            ftype = fspec.get("type", "string")
            if not strict and fname == first_field and ftype in {"integer", "number"}:
                payload[fname] = "seven"
            else:
                payload[fname] = sample_values.get(ftype, "x")

        if not properties:
            payload = {"order_id": 7} if strict else {"order_id": "seven"}

        return ChatResponse(
            text=json.dumps(payload),
            tokens_in=40,
            tokens_out=15,
            cost_usd=0.0002,
            model_version="fake-1.0",
        )


def _empty_response(text: str = "") -> ChatResponse:
    return ChatResponse(
        text=text, tokens_in=0, tokens_out=0, cost_usd=0.0, model_version="fake-1.0"
    )
