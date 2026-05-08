# Architecture

This document covers the parts of `agentic-runner` that are not obvious from
reading the source: the replan cycle, where budgets are checked, the Provider
Protocol shape, and what's deliberately not implemented.

## The replan cycle

Every loop iteration walks four stages: plan -> select -> invoke -> validate.
Validation is the only stage whose failure does **not** map to "retry the same
call". A `FailureReason` is constructed and fed back into the planner, which
produces a new subtask list. Successive replans see the prior failure context
in their planning prompt and can pick different tools, decompose more
granularly, or surface `AbortGoal`.

```
                          +---- REPLAN <-- FailureReason -+
                          |                               |
       +--------> PLAN ---+                               |
       |                  +--> [subtasks] --> for each ---+
       |                                          |
       |                                          v
       |                                       SELECT
       |                                          |
       |                                          v
       |                                       INVOKE  --> TOOL_RETURNED_ERROR --+
       |                                          |                              |
       |                                          v                              |
       |                                      VALIDATE  --> OUTPUT_SCHEMA_... ---+
       |                                          |     --> CONFIDENCE_TOO_LOW --+
       |                                          v                              |
       |                                      next subtask                       |
       |                                          |                              |
       |                                       finish?                           |
       |                                          |                              |
       |                                          v                              |
       +------- plan exhausted --> ABORT      SUCCEEDED                          |
                                                                                 |
                budget exhausted --> ABORT  <----------------------------------- +
```

### `FailureReason` -> replan strategy

| Reason | What it means | What the planner can do next |
| --- | --- | --- |
| `OUTPUT_SCHEMA_MISMATCH` | Tool returned a payload that didn't fit its declared `output_model`. | Pick a different tool, tighten the prompt, or change the args (e.g. set `strict=True`). |
| `TOOL_RETURNED_ERROR` | Tool raised `ToolInvocationError`. | Same as above. |
| `CONFIDENCE_TOO_LOW` | Output had a `confidence` field below `subtask.confidence_threshold`. | Re-issue with stricter constraints or via a different tool. |
| `PRECONDITION_VIOLATED` | Subtask depended on state the runner can't satisfy. | Reorder subtasks; insert a setup step. |
| `RESOURCE_EXHAUSTED` | Tool reported it was rate-limited or out of quota. | Defer or give up. |
| `PLAN_INVALID` | Planner returned malformed JSON or empty subtasks. | One more replan, then abort. |
| `NO_TOOL_FOR_SUBTASK` | Selector asked for a tool that isn't registered. | The runner records this then asks the planner for a new plan that uses only available tools; if the next plan is also unsatisfiable, abort. |
| `TIMEOUT` | Tool exceeded its declared `max_runtime_ms`. | Substitute a cheaper tool or abort. |

### Honest abort

The runner's job is *not* to make every goal succeed. A goal with no path
forward must abort with a typed reason rather than burn budget. The eval
suite's `g13_send_email_abort` exercises this: there's no email tool, so
the runner replans up to `max_replans` and then aborts with
`max_replans_exceeded :: no_tool_for_subtask: requested tool 'send_email' is
not registered`. That's the desired outcome for that goal.

## Budgets

`Runner.run` accepts a `RunBudget` with four caps:

- `max_steps` — total tool invocations across all subtasks.
- `max_replans` — distinct replanning rounds.
- `max_cost_usd` — running total of LLM call costs.
- `max_wall_clock_s` — measured from `Runner.run` entry.

Budgets are checked at three points:

1. **Top of the planning loop** (before each plan call) — wall clock + replans.
2. **After the plan call** — cost (a planner can over-spend before any tool
   runs).
3. **Before each tool invocation** — steps + cost.

Each check that trips returns `RunResult.aborted(reason)` with a string that
encodes which budget was hit. The full state is persisted before the abort
returns, so the trace is queryable via `GET /v1/goals/{id}/trace`.

## Tool registry

A tool is a class that satisfies the `Tool` Protocol — `name`, `description`,
`input_model`, `output_model`, `max_runtime_ms`, `idempotent`, and an
`invoke(args)` method. The `@register_tool` decorator instantiates the class
and adds it to the global `REGISTRY`. The eight built-in tools auto-register
when `agentic_runner.tools` is imported.

The runner calls each tool through `invoke_with_guard`, which:

1. Validates `args` against the tool's `input_model`.
2. Wraps the call in an OpenTelemetry span.
3. Measures wall-clock and refuses to return outputs that exceeded
   `max_runtime_ms` (raised as `ToolTimeoutError`).
4. Wraps `Exception` from the tool body in `ToolInvocationError` so the
   runner can persist a typed status.

The validator runs *after* `invoke_with_guard` returns. Tools may raise on
their own input validation; the validator on top is a second line of defense
that also handles the `confidence` threshold check.

## Provider Protocol — function-calling shape

`ChatProvider.chat()` returns a `ChatResponse` with a `tool_calls` field.
That field is the load-bearing one: every selector decision flows through
`tool_calls[0]`. The text channel exists for cases where the LLM is asked to
produce a JSON plan (the planner) or a free-form payload (the `summarize` and
`extract_json` tools), but the runner never tries to parse free text into a
tool invocation.

`FakeProvider` covers the hermetic CI path: it scripts plans + tool calls
keyed by a regex match against the user prompt. Live BYOK adapters
(`OpenAIProvider`, `AnthropicProvider`) are stubs that satisfy the Protocol so
the registry stays uniform; the actual SDK integration is intentionally
excluded from CI to keep the test suite hermetic.

## Persistence

Five tables, all with cascade-on-goal-delete:

- `goals` — the request, the budgets, the final result/abort_reason, totals.
- `subtasks` — ordered (`goal_id`, `idx`); supports parent linkage so future
  decomposition can record nested replans.
- `tool_calls` — args/output/latency/cost/status per invocation.
- `validation_results` — one row per `tool_call`, with `schema_violations`
  captured as JSON.
- `replan_events` — ordered (`goal_id`, `idx`); records the triggering tool
  call and the failure reason.

Indexes on `(goal_id, idx)` and `(subtask_id, idx)` support the
trace-tree-build query without an N+1 walk.

## Parallel subtask execution (DAG scheduler)

The legacy `Runner.run` walks subtasks one at a time. `Runner.run_parallel`
uses the same plan-validate-replan shape but schedules subtasks against an
explicit dependency DAG.

### Plan shape

`PlannedSubtask` carries two new fields:

- `parallel: bool` — opt-in to concurrent execution.
- `dependencies: list[int]` — indices (within the same plan) of subtasks
  that must complete before this one can start.

If `dependencies` is empty AND `parallel` is true, the subtask is treated as
a root with no implicit dependency on prior siblings. If `dependencies` is
empty AND `parallel` is false, the subtask inherits an implicit dependency
on every earlier subtask, preserving the legacy sequential semantics for
plans that don't opt in.

### Scheduler

```
plan -> _build_dependency_map -> while in_flight:
                                    schedule every ready idx (deps satisfied)
                                    asyncio.wait(FIRST_COMPLETED, capped at
                                                 Semaphore(MAX_CONCURRENCY=4))
                                    on success: append to completed, may unblock more
                                    on failure: drain remaining, capture last_failure,
                                                bubble for replan
```

Concurrency is capped at 4 simultaneous tasks (`MAX_CONCURRENCY`). Each
subtask runs through the same select -> invoke -> validate sequence as the
sequential path; sync tool invocation is wrapped in `asyncio.to_thread` so a
slow tool does not block the event loop.

### Failure -> replan with sibling state

When any subtask fails, the scheduler stops scheduling new work, drains
the remaining in-flight tasks (so their state is captured), and returns
the typed `FailureReason` of the first failing subtask. The runner feeds
that failure into the planner along with `state["prior_outputs"]` —
which now contains the outputs from every completed sibling subtask, not
just those before the failure point. This means a replan after a parallel
branch failure has access to the latest state the rest of the DAG produced.

### Persistence

`subtasks.dependencies` is stored as JSON (list of subtask ID strings) and
`subtasks.parallel` is a boolean. Both are added by Alembic revision
`20260508_0001`. The DAG is rebuilt on each replan from the in-memory
`Plan` rather than the DB row, so a replan is free to choose a different
parallel/sequential layout.

### Wall-time evidence

For a 5-subtask plan with two independent parallel branches at
`sleep_s=0.10` per task, the parallel scheduler completes in ~0.42s vs
~0.51s for the same workload run sequentially (~1.23x speedup; saves
roughly one `sleep_s` by overlapping the parallel pair). The
`tests/unit/test_parallel_runner.py::test_parallel_subtasks_overlap_in_wall_time`
test asserts this directly: it requires the parallel run to beat
`5 * sleep_s - 0.5 * sleep_s` and verifies the parallel pair's
start/finish windows overlap.

## What's deliberately not here

- No retry-with-backoff. A failed tool call goes through the planner, not back
  to the same call site. (This is the entire point of the project.)
- No streaming responses. The provider Protocol returns a complete
  `ChatResponse`; the runner is synchronous step-by-step (`run_parallel`
  uses asyncio under the hood but the plan/replan loop itself is still
  iterative).
- No vector store or RAG layer. The state passed back into the planner is
  literal subtask + output history, not a retrieval index.
- No subprocess execution, no shell, no eval(). `calculate` is an AST
  walker; `query_db` is read-only SQL with parsed-statement guards.
- No real-network HTTP in CI. The default `httpx.MockTransport` returns a
  deterministic 200; the live transport is settable but never wired in tests.
- No live LLM provider in CI. OpenAI/Anthropic adapters raise
  `NotImplementedError` and require explicit BYOK setup outside the test
  suite.
