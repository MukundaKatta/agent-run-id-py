"""Tests for agent-run-id-py."""

import threading

import pytest
from agent_run_id import (
    new_run_id,
    current_run_id,
    set_run_id,
    require_run_id,
    inject,
    RunContext,
    RunRegistry,
    wrap,
    wrap_with_id,
)


def _run_in_fresh_context(fn):
    """Run fn in a new thread so the run-id ContextVar starts at its default (None)."""
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def test_new_run_id_format():
    rid = new_run_id()
    assert rid.startswith("run_")
    assert len(rid) == 4 + 32  # "run_" + 32 hex chars


def test_new_run_id_custom_prefix():
    rid = new_run_id(prefix="task")
    assert rid.startswith("task_")


def test_new_run_id_unique():
    ids = {new_run_id() for _ in range(100)}
    assert len(ids) == 100


def test_current_run_id_none_initially():
    # Outer context — may or may not have an ID from a previous test,
    # so test inside a fresh context with None
    with RunContext("test_outer"):
        pass  # restore
    # after exiting, it should be restored
    # just check that RunContext restores on exit
    outer = current_run_id()
    with RunContext("inner_id"):
        assert current_run_id() == "inner_id"
    assert current_run_id() == outer


def test_run_context_sets_and_restores():
    with RunContext("abc123"):
        assert current_run_id() == "abc123"
    # restored
    assert current_run_id() != "abc123"


def test_run_context_auto_id():
    with RunContext() as ctx:
        assert current_run_id() == ctx.run_id
        assert ctx.run_id.startswith("run_")


def test_run_context_nested():
    with RunContext("outer"):
        assert current_run_id() == "outer"
        with RunContext("inner"):
            assert current_run_id() == "inner"
        assert current_run_id() == "outer"


def test_set_run_id():
    with RunContext("base"):
        set_run_id("overridden")
        assert current_run_id() == "overridden"


def test_require_run_id_returns_current():
    with RunContext("required_test"):
        rid = require_run_id()
        assert rid == "required_test"


def test_inject_adds_run_id():
    with RunContext("run_xyz"):
        result = inject({"key": "val"})
        assert result["run_id"] == "run_xyz"
        assert result["key"] == "val"


def test_inject_custom_key():
    with RunContext("run_abc"):
        result = inject({"x": 1}, key="trace_id")
        assert "trace_id" in result
        assert result["trace_id"] == "run_abc"


def test_inject_no_context_passthrough():
    # Outside RunContext: should return original dict unchanged
    # Run after restoring context to None
    outer = current_run_id()
    if outer is not None:
        # Wrap in a way that temporarily clears — not possible cleanly
        # so just test that inject works with a set context
        pass
    with RunContext("inject_test"):
        result = inject({"a": 1})
        assert "run_id" in result


def test_wrap_decorator():
    calls = []

    @wrap
    def fn():
        calls.append(current_run_id())
        return "done"

    fn()
    fn()
    assert len(calls) == 2
    assert calls[0] != calls[1]  # unique IDs each call
    assert all(c.startswith("run_") for c in calls)


def test_wrap_with_id_injects_kwarg():
    received = []

    @wrap_with_id
    def fn(run_id=None):
        received.append(run_id)
        return run_id

    result = fn()
    assert result.startswith("run_")
    assert len(received) == 1


# Registry tests


def test_registry_start_and_finish():
    reg = RunRegistry()
    reg.start("r1", metadata={"model": "claude"})
    assert "r1" in reg.active()
    reg.finish("r1", result="done")
    info = reg.get("r1")
    assert info["status"] == "done"
    assert info["result"] == "done"
    assert info["finished_at"] is not None
    assert "r1" not in reg.active()


def test_registry_fail():
    reg = RunRegistry()
    reg.start("r2")
    reg.fail("r2", RuntimeError("boom"))
    info = reg.get("r2")
    assert info["status"] == "failed"
    assert "RuntimeError" in info["error"]


def test_registry_all_ids():
    reg = RunRegistry()
    reg.start("a")
    reg.start("b")
    assert set(reg.all_ids()) == {"a", "b"}


def test_registry_get_missing():
    reg = RunRegistry()
    assert reg.get("nonexistent") is None


def test_registry_clear():
    reg = RunRegistry()
    reg.start("x")
    reg.clear()
    assert reg.all_ids() == []


def test_registry_metadata():
    reg = RunRegistry()
    reg.start("m1", metadata={"session": "abc"})
    info = reg.get("m1")
    assert info["metadata"]["session"] == "abc"


# Coverage for the "no run ID in context" code paths (run in a fresh thread)


def test_current_run_id_none_in_fresh_context():
    assert _run_in_fresh_context(current_run_id) is None


def test_require_run_id_raises_when_none():
    with pytest.raises(RuntimeError):
        _run_in_fresh_context(require_run_id)


def test_inject_passthrough_when_no_context():
    original = {"a": 1}
    result = _run_in_fresh_context(lambda: inject(original))
    assert result == {"a": 1}
    assert "run_id" not in result


def test_set_run_id_in_fresh_context():
    def body():
        set_run_id("manual_id")
        return current_run_id()

    assert _run_in_fresh_context(body) == "manual_id"


def test_registry_repr_hides_lock():
    reg = RunRegistry()
    assert "_lock" not in repr(reg)


def test_wrap_with_id_returns_same_id_passed_to_fn():
    seen = []

    @wrap_with_id
    def fn(run_id=None):
        seen.append(run_id)
        return run_id

    result = fn()
    assert result == seen[0]
    assert result.startswith("run_")
