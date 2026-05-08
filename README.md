# agentic-runner

A general-purpose agentic loop in Python: a goal arrives, a planner decomposes
it into subtasks, an LLM-driven selector picks a tool for each subtask, the
tool's output is validated against a Pydantic schema, and **on validation
failure the runner re-plans with a typed `FailureReason` rather than retrying
the same call**. The replan path is the load-bearing piece — it can swap tools,
decompose differently, or abort honestly when no path forward exists.

## What this studies

- The plan -> select -> invoke -> validate -> replan cycle as a single loop,
  not retry-on-error.
- Typed `FailureReason` (`OUTPUT_SCHEMA_MISMATCH`, `TOOL_RETURNED_ERROR`,
  `CONFIDENCE_TOO_LOW`, `PRECONDITION_VIOLATED`, `RESOURCE_EXHAUSTED`,
  `PLAN_INVALID`, `NO_TOOL_FOR_SUBTASK`, `TIMEOUT`) feeding back into the
  planner so it can choose a different decomposition.
- Hard budgets at four levels: `max_steps`, `max_replans`, `max_cost_usd`,
  `max_wall_clock_s`. Budget exhaustion produces `RunResult.aborted(reason)`
  rather than a retry storm.
- A function-calling-shaped `ChatProvider` Protocol with a hermetic
  `FakeProvider` for CI (scripted plans + tool calls keyed off the user
  prompt).

## Eval numbers

Real numbers from the committed baseline at
[`eval/baselines/runner_v1_fake.json`](eval/baselines/runner_v1_fake.json),
produced by running the 20-goal suite (15 short + 5 long-horizon at 10 steps
each) through the deterministic `FakeProvider`:

| Metric | Value |
| --- | --- |
| goals (n) | 20 |
| success rate | 0.9500 |
| abort rate (honest) | 0.0500 |
| replan rate | 0.1000 |
| avg steps per goal | 4.10 |
| avg cost per goal (USD) | 0.002680 |
| tool-sequence Jaccard (avg) | 1.0000 |
| rubric pass rate | 1.0000 |
| matches-expected rate | 1.0000 |

The 5 long-horizon goals (`g16`..`g20`) chain 10 tool calls each across
`query_db`, `calculate`, `write_file`, `read_file`, `summarize`, `http_get`,
and `extract_json`. They pass at 100% under the FakeProvider.

Re-run with `make eval-smoke` to assert the baseline match within `1e-6`.

### Bench-regress gate

`make bench-regress` re-runs the suite and compares the headline aggregate
metrics (`success_rate`, `abort_rate`, `avg_steps`, `avg_cost_usd`,
`rubric_pass_rate`, `tool_sequence_jaccard_avg`) against the committed
baseline. Any metric whose relative drift exceeds 30% trips the gate and
fails CI. The `eval-smoke` job remains the strict equality check; the
`bench-regress` job is the looser drift gate intended to survive hermetic
LLM jitter once non-fake providers are wired in.

## Tools

Eight base tools and one composed primitive, all deterministic in CI. Each
ships a Pydantic input schema, a Pydantic output schema, a `max_runtime_ms`,
and an `idempotent` flag.

| Tool | Purpose |
| --- | --- |
| `calculate` | Safe arithmetic via `ast.parse` + numeric/op whitelist (no `eval()`, no names, no calls). |
| `query_db` | Read-only SELECT against a small demo schema (employees / departments / orders). Multi-statement and DDL keywords are rejected. |
| `read_file` | UTF-8 read confined to the `workspace/` sandbox; max-size cap; path-traversal blocked. |
| `write_file` | Same sandbox + size cap. |
| `http_get` | GET against an allowlist of hosts; CI uses `httpx.MockTransport` so no real network. |
| `summarize` | LLM-backed summarization. CI's `FakeProvider` returns deterministic scripted summaries. |
| `extract_json` | LLM-backed structured extraction validated against a user-supplied JSON Schema. |
| `finish` | Terminal tool that signals completion and returns the final string. |
| `summarize_document` | **Composed primitive** — `read_file -> summarize -> finish` packaged as one atomic Tool with its own input/output schemas. |

### Composed tools

A `ComposedTool` wraps a fixed sequence of inner tool calls as a single
primitive that satisfies the `Tool` protocol. The runner sees one tool
call; the composed tool internally executes its inner steps in order,
threads each step's output to the next, and produces a combined output
validated against its own declared output schema.

```python
from agentic_runner.tools import ComposedTool, ComposedStep, register_tool
from pydantic import BaseModel, Field


class MyInput(BaseModel):
    path: str = Field(min_length=1)
    max_words: int = 20


class MyOutput(BaseModel):
    summary: str
    source_bytes: int


@register_tool
class MyComposed(ComposedTool):
    name = "my_composed"
    description = "Read a file and summarize it as one atomic action."
    input_model = MyInput
    output_model = MyOutput
    steps = [
        ComposedStep("read_file", lambda p, q: {"path": p["path"]}),
        ComposedStep("summarize", lambda p, q: {"text": q[0]["content"], "max_words": p["max_words"]}),
        ComposedStep("finish",    lambda p, q: {"result": q[1]["summary"]}),
    ]
    output_combiner = staticmethod(
        lambda p, q: {"summary": q[1]["summary"], "source_bytes": q[0]["bytes"]}
    )
```

Inner-step failures propagate as a single `ToolInvocationError` whose
message names the failing step index and tool. The combiner output is
validated against `output_model` before returning, so a mismatched
combiner is caught at invocation rather than silently passed through.

## Modules

| Module | Role |
| --- | --- |
| `providers/` | `ChatProvider` Protocol + `FakeProvider` (hermetic) and stub adapters for OpenAI / Anthropic. |
| `tools/` | Tool Protocol + registry (`@register_tool`). One module per tool. |
| `planner.py` | Builds a planning prompt with goal + state + optional `FailureReason`; parses the JSON response into a `Plan`. |
| `selector.py` | Asks the model to pick a tool + arguments for one subtask. Returns `SelectorResult` (tool_call or failure). |
| `validator.py` | Runs Pydantic validation; emits typed `FailureReason` on mismatch or low confidence. |
| `runner.py` | The plan -> select -> invoke -> validate -> replan loop with budget enforcement. |
| `failure.py` | `FailureKind` enum + `FailureReason` model + `AbortGoal`. |
| `models.py` | SQLAlchemy ORM: goals, subtasks, tool_calls, validation_results, replan_events. |
| `api.py` | FastAPI: `POST /v1/goals`, `GET /v1/goals/{id}[/trace]`, `GET /healthz`. |
| `cli.py` | Click CLI: `agentic-runner run`, `agentic-runner eval run/smoke`, `agentic-runner seed`. |
| `eval_harness.py` | YAML suite loader + metrics + baseline diff (1e-6 float tolerance). |
| `trace.py` | structlog + OpenTelemetry helpers. |

## Quickstart

```sh
make dev            # install editable + dev extras
make up             # start postgres + the api via docker-compose
make migrate        # alembic upgrade head
make test           # unit tests
make typecheck      # strict mypy
make lint           # ruff + black --check
make eval           # produce a fresh baseline
make eval-smoke     # assert baseline match
```

Run a single goal from the CLI:

```sh
python -m agentic_runner.cli run \
    --goal "Calculate the average salary in the engineering department" \
    --provider fake
```

Hit the API:

```sh
curl -X POST localhost:8000/v1/goals \
    -H 'content-type: application/json' \
    -d '{"goal_text": "Compute (2+3)*7 with calculate"}'
```

## Architecture (the loop)

```
goal arrives -> Planner.plan(goal, state) -> [Subtask, Subtask, ...]
                       ^                              |
                       |                              v
                       |                   for each subtask:
                       |                       Selector.choose_tool(subtask, available_tools)
                       |                              |
                       |                              v
                       |                       ToolRegistry.invoke(tool, args)
                       |                              |
                       |                              v
                       |                       Validator.validate(output, output_schema)
                       |                              |
                       |            +-----------------+-----------------+
                       |            |                                   |
                       |          valid                              invalid
                       |            |                                   |
                       |            v                                   v
                       |     advance / record         FailureReason -> Planner.replan(goal, state, reason)
                       |            |                                   |
                       +------------+-----------------------------------+
                                     final -> return result -> persist trace
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the failure-reason -> replan
strategy mapping, budget enforcement points, and Provider Protocol shape.

## What this is *not*

- Not a browser-using agent — see SAY-5/pagerunner for that.
- Not a bug-classification one-shot — see SAY-5/bug-triage for that.
- Not an LLM gateway — see SAY-5/pulseroute for that.
- No real-internet HTTP without an explicit allowlist; the CI tests use
  `httpx.MockTransport` exclusively.
- No privileged subprocess execution. `calculate` evaluates arithmetic via an
  AST whitelist — never `eval()`. `query_db` only allows SELECT against a tiny
  demo schema.
- No multi-agent coordination, no learned tool selection (the LLM is the only
  selector).
- No fabricated benchmarks. Every number in this README comes from a committed
  baseline that CI re-asserts on every push.

## License

MIT — see [LICENSE](LICENSE).
