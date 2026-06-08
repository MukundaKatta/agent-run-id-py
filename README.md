# agent-run-id-py

Generate, thread, and propagate **run IDs** for agent executions. Trace agent
calls across steps, logs, threads, and async contexts — with no third-party
dependencies.

When an LLM agent does work, it usually fans out into many nested steps, tool
calls, and (often) concurrent async tasks. Attaching a single stable run ID to
all of that makes logs, metrics, and traces correlate-able. This library gives
you that run ID and propagates it for you using `contextvars`, so it survives
across `await` boundaries and into tasks/threads spawned within a context.

## Install

```bash
pip install agent-run-id-py
```

Requires Python 3.9+. No runtime dependencies. The package ships type
information (PEP 561 `py.typed`).

## Usage

```python
from agent_run_id import (
    new_run_id, current_run_id, set_run_id, require_run_id,
    RunContext, RunRegistry, wrap, wrap_with_id, inject,
)

# Generate a run ID
rid = new_run_id()            # "run_a3f2..."
rid = new_run_id("task")      # "task_a3f2..."

# Context manager — sets the run ID for a block
with RunContext("run_abc") as ctx:
    print(current_run_id())   # "run_abc"
    # ... do agent work ...

# Auto-generate when no ID is passed
with RunContext() as ctx:
    print(ctx.run_id)         # "run_<hex>"

# Nested contexts restore the previous ID on exit
with RunContext("outer"):
    with RunContext("inner"):
        print(current_run_id())  # "inner"
    print(current_run_id())      # "outer"

# Inject the current ID into a dict (handy for log records / metadata)
with RunContext("run_xyz"):
    payload = inject({"model": "claude"})
    # {"model": "claude", "run_id": "run_xyz"}   (input dict is not mutated)

# Wrap a function so each call gets a fresh ID
@wrap
def handle_request():
    print(current_run_id())   # unique each call

# Inject the ID as a keyword argument
@wrap_with_id
def handle(run_id=None):
    print(run_id)             # "run_<hex>"

# Require an ID (raise if none is set)
with RunContext("run_1"):
    rid = require_run_id()    # "run_1"

# Track active / completed runs
registry = RunRegistry()
registry.start("run_001", metadata={"user": "alice"})
registry.finish("run_001", result="ok")
registry.fail("run_001", RuntimeError("crash"))
registry.active()             # list of active run IDs
registry.get("run_001")       # full record dict (a copy)
```

### Async propagation

Because the current run ID is stored in a `contextvars.ContextVar`, it
propagates into tasks created within a `RunContext` and stays isolated between
concurrently running tasks:

```python
import asyncio
from agent_run_id import RunContext, current_run_id

async def step():
    return current_run_id()

async def main():
    with RunContext("run_async"):
        # The ID is visible inside the task spawned within the context.
        return await asyncio.create_task(step())

asyncio.run(main())  # -> "run_async"
```

## API

| Name | Description |
| --- | --- |
| `new_run_id(prefix="run") -> str` | Generate a new unique ID as `"{prefix}_{32-hex}"`. |
| `current_run_id() -> str \| None` | The run ID active in the current context, or `None`. |
| `set_run_id(run_id) -> None` | Set the run ID for the current context. |
| `require_run_id() -> str` | Return the current run ID or raise `RuntimeError` if unset. |
| `inject(d, key="run_id") -> dict` | Return a copy of `d` with the current run ID added (no-op if unset). |
| `RunContext(run_id=None, prefix="run")` | Context manager; sets a run ID for the block and restores the previous one on exit (auto-generates if `run_id` is `None`). |
| `wrap(fn, prefix="run")` | Decorator: each call runs inside a fresh `RunContext`. |
| `wrap_with_id(fn, prefix="run")` | Decorator: passes a fresh `run_id` keyword argument into each call. |
| `RunRegistry()` | Thread-safe tracker of run records (`start`, `finish`, `fail`, `get`, `active`, `all_ids`, `clear`). |

`RunRegistry` records are dicts with the keys: `run_id`, `status`
(`"active"`, `"done"`, or `"failed"`), `started_at`, `finished_at`, `result`,
`error`, and `metadata`. `get()` returns a copy, so mutating the returned dict
does not affect the registry.

## Development

Run the test suite with the standard library only (no extra installs):

```bash
python -m unittest discover -s tests
```

## License

MIT
