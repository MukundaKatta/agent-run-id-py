"""agent-run-id-py — generate, thread, and propagate run IDs for agent executions."""

from __future__ import annotations

import functools
import threading
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Async-safe context variable for the current run ID
_current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)


def new_run_id(prefix: str = "run") -> str:
    """Generate a new unique run ID."""
    return f"{prefix}_{uuid.uuid4().hex}"


def current_run_id() -> str | None:
    """Return the run ID active in the current async/thread context, or None."""
    return _current_run_id.get()


def set_run_id(run_id: str) -> None:
    """Set the run ID for the current context."""
    _current_run_id.set(run_id)


def require_run_id() -> str:
    """Return the current run ID or raise RuntimeError if not set."""
    rid = _current_run_id.get()
    if rid is None:
        raise RuntimeError(
            "No run ID set in the current context. Call set_run_id() or use RunContext."
        )
    return rid


def inject(d: dict[str, Any], key: str = "run_id") -> dict[str, Any]:
    """Return a copy of d with the current run ID injected under key."""
    rid = current_run_id()
    if rid is None:
        return d
    return {**d, key: rid}


class RunContext:
    """
    Context manager that sets a run ID for the duration of a block.

    A new ID is generated automatically if none is provided.

    Example::

        with RunContext("run_abc123"):
            assert current_run_id() == "run_abc123"

        with RunContext() as ctx:
            print(ctx.run_id)  # auto-generated
    """

    def __init__(self, run_id: str | None = None, prefix: str = "run") -> None:
        self.run_id: str = run_id if run_id is not None else new_run_id(prefix)
        self._token: Token[str | None] | None = None

    def __enter__(self) -> "RunContext":
        self._token = _current_run_id.set(self.run_id)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._token is not None:
            _current_run_id.reset(self._token)
            self._token = None


def wrap(fn: F, prefix: str = "run") -> F:
    """Decorator that creates a fresh run ID for each call.

    The wrapped function sees ``current_run_id()`` return a new, unique ID for
    the duration of every call. The original signature is preserved.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with RunContext(prefix=prefix):
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def wrap_with_id(fn: F, prefix: str = "run") -> F:
    """Decorator that injects the run_id as a keyword argument into each call.

    The wrapped function must accept a ``run_id`` keyword argument; a fresh ID
    is generated and passed in on every call (and also set as the current run
    ID for the duration of the call).
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with RunContext(prefix=prefix) as ctx:
            return fn(*args, run_id=ctx.run_id, **kwargs)

    return wrapper  # type: ignore[return-value]


@dataclass
class RunRegistry:
    """
    Track active and completed run IDs with associated metadata.

    Example::

        registry = RunRegistry()
        registry.start("run_abc", metadata={"model": "claude-sonnet"})
        registry.finish("run_abc", result="ok")
        print(registry.get("run_abc"))
    """

    _runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def start(self, run_id: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "status": "active",
                "started_at": time.time(),
                "finished_at": None,
                "result": None,
                "error": None,
                "metadata": metadata or {},
            }

    def finish(self, run_id: str, result: Any = None) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].update(
                    {
                        "status": "done",
                        "finished_at": time.time(),
                        "result": result,
                    }
                )

    def fail(self, run_id: str, error: Exception) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].update(
                    {
                        "status": "failed",
                        "finished_at": time.time(),
                        "error": repr(error),
                    }
                )

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._runs[run_id]) if run_id in self._runs else None

    def active(self) -> list[str]:
        with self._lock:
            return [rid for rid, r in self._runs.items() if r["status"] == "active"]

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._runs.keys())

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


__all__ = [
    "new_run_id",
    "current_run_id",
    "set_run_id",
    "require_run_id",
    "inject",
    "RunContext",
    "RunRegistry",
    "wrap",
    "wrap_with_id",
]
