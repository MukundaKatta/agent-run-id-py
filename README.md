# agent-run-id-py

Generate, thread, and propagate run IDs for agent executions. Trace agent calls across steps, logs, and async contexts.

## Install

```bash
pip install agent-run-id-py
```

## Usage

```python
from agent_run_id import (
    new_run_id, current_run_id, RunContext, RunRegistry, wrap, wrap_with_id, inject
)

# Generate a run ID
rid = new_run_id()            # "run_a3f2..."
rid = new_run_id("task")      # "task_a3f2..."

# Context manager — sets run ID for a block
with RunContext("run_abc") as ctx:
    print(current_run_id())   # "run_abc"
    # do agent work

# Auto-generate
with RunContext() as ctx:
    print(ctx.run_id)          # "run_<hex>"

# Nested contexts
with RunContext("outer"):
    with RunContext("inner"):
        print(current_run_id())  # "inner"
    print(current_run_id())      # "outer"

# Inject into dicts (useful for logging/metadata)
with RunContext("run_xyz"):
    payload = inject({"model": "claude"})
    # {"model": "claude", "run_id": "run_xyz"}

# Wrap a function — fresh ID per call
@wrap
def handle_request():
    print(current_run_id())  # unique each call

# Inject ID as kwarg
@wrap_with_id
def handle(run_id=None):
    print(run_id)            # "run_<hex>"

# Track active/completed runs
registry = RunRegistry()
registry.start("run_001", metadata={"user": "alice"})
registry.finish("run_001", result="ok")
registry.fail("run_001", RuntimeError("crash"))
registry.active()   # list of active run IDs
registry.get("run_001")  # full record dict
```

## License

MIT
