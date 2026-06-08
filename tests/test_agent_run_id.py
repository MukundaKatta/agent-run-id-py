"""Tests for agent-run-id-py.

These tests use only the Python standard library (``unittest``) so they can be
run without any third-party dependencies::

    python3 -m unittest discover -s tests
"""

import asyncio
import os
import sys
import threading
import unittest

# Make the package importable from a source checkout (src/ layout) without
# requiring an editable install, so the suite runs with a bare
# ``python3 -m unittest discover -s tests``.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent_run_id import (
    RunContext,
    RunRegistry,
    current_run_id,
    inject,
    new_run_id,
    require_run_id,
    set_run_id,
    wrap,
    wrap_with_id,
)


def _run_in_fresh_context(fn):
    """Run ``fn`` in a new thread so the run-id ContextVar starts at its default (None)."""
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


class NewRunIdTests(unittest.TestCase):
    def test_format(self):
        rid = new_run_id()
        self.assertTrue(rid.startswith("run_"))
        self.assertEqual(len(rid), 4 + 32)  # "run_" + 32 hex chars

    def test_custom_prefix(self):
        rid = new_run_id(prefix="task")
        self.assertTrue(rid.startswith("task_"))

    def test_unique(self):
        ids = {new_run_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_hex_suffix(self):
        rid = new_run_id()
        suffix = rid[len("run_"):]
        # The suffix must be valid lowercase hexadecimal.
        int(suffix, 16)
        self.assertEqual(suffix, suffix.lower())


class RunContextTests(unittest.TestCase):
    def test_sets_and_restores(self):
        outer = current_run_id()
        with RunContext("abc123"):
            self.assertEqual(current_run_id(), "abc123")
        # Restored to whatever it was before entering.
        self.assertEqual(current_run_id(), outer)

    def test_auto_id(self):
        with RunContext() as ctx:
            self.assertEqual(current_run_id(), ctx.run_id)
            self.assertTrue(ctx.run_id.startswith("run_"))

    def test_auto_id_custom_prefix(self):
        with RunContext(prefix="job") as ctx:
            self.assertTrue(ctx.run_id.startswith("job_"))

    def test_nested(self):
        with RunContext("outer"):
            self.assertEqual(current_run_id(), "outer")
            with RunContext("inner"):
                self.assertEqual(current_run_id(), "inner")
            self.assertEqual(current_run_id(), "outer")

    def test_restores_on_exception(self):
        outer = current_run_id()
        with self.assertRaises(ValueError):
            with RunContext("boom"):
                self.assertEqual(current_run_id(), "boom")
                raise ValueError("explode")
        self.assertEqual(current_run_id(), outer)


class SetAndRequireTests(unittest.TestCase):
    def test_set_run_id(self):
        with RunContext("base"):
            set_run_id("overridden")
            self.assertEqual(current_run_id(), "overridden")

    def test_require_returns_current(self):
        with RunContext("required_test"):
            self.assertEqual(require_run_id(), "required_test")

    def test_set_run_id_in_fresh_context(self):
        def body():
            set_run_id("manual_id")
            return current_run_id()

        self.assertEqual(_run_in_fresh_context(body), "manual_id")

    def test_current_run_id_none_in_fresh_context(self):
        self.assertIsNone(_run_in_fresh_context(current_run_id))

    def test_require_raises_when_none(self):
        with self.assertRaises(RuntimeError):
            _run_in_fresh_context(require_run_id)


class InjectTests(unittest.TestCase):
    def test_adds_run_id(self):
        with RunContext("run_xyz"):
            result = inject({"key": "val"})
            self.assertEqual(result["run_id"], "run_xyz")
            self.assertEqual(result["key"], "val")

    def test_custom_key(self):
        with RunContext("run_abc"):
            result = inject({"x": 1}, key="trace_id")
            self.assertEqual(result["trace_id"], "run_abc")

    def test_does_not_mutate_input(self):
        original = {"a": 1}
        with RunContext("run_keep"):
            result = inject(original)
        self.assertNotIn("run_id", original)
        self.assertIn("run_id", result)
        self.assertIsNot(result, original)

    def test_passthrough_when_no_context(self):
        original = {"a": 1}
        result = _run_in_fresh_context(lambda: inject(original))
        self.assertEqual(result, {"a": 1})
        self.assertNotIn("run_id", result)


class WrapTests(unittest.TestCase):
    def test_wrap_decorator(self):
        calls = []

        @wrap
        def fn():
            calls.append(current_run_id())
            return "done"

        self.assertEqual(fn(), "done")
        fn()
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])  # unique IDs each call
        self.assertTrue(all(c.startswith("run_") for c in calls))

    def test_wrap_preserves_metadata(self):
        @wrap
        def documented():
            """My docstring."""

        self.assertEqual(documented.__name__, "documented")
        self.assertEqual(documented.__doc__, "My docstring.")

    def test_wrap_passes_args_and_kwargs(self):
        @wrap
        def add(a, b, c=0):
            return a + b + c

        self.assertEqual(add(1, 2, c=3), 6)

    def test_wrap_with_id_injects_kwarg(self):
        received = []

        @wrap_with_id
        def fn(run_id=None):
            received.append(run_id)
            return run_id

        result = fn()
        self.assertTrue(result.startswith("run_"))
        self.assertEqual(len(received), 1)
        self.assertEqual(result, received[0])

    def test_wrap_with_id_preserves_positional_args(self):
        @wrap_with_id
        def fn(value, run_id=None):
            return (value, run_id)

        value, rid = fn("payload")
        self.assertEqual(value, "payload")
        self.assertTrue(rid.startswith("run_"))


class RegistryTests(unittest.TestCase):
    def test_start_and_finish(self):
        reg = RunRegistry()
        reg.start("r1", metadata={"model": "claude"})
        self.assertIn("r1", reg.active())
        reg.finish("r1", result="done")
        info = reg.get("r1")
        self.assertEqual(info["status"], "done")
        self.assertEqual(info["result"], "done")
        self.assertIsNotNone(info["finished_at"])
        self.assertNotIn("r1", reg.active())

    def test_fail(self):
        reg = RunRegistry()
        reg.start("r2")
        reg.fail("r2", RuntimeError("boom"))
        info = reg.get("r2")
        self.assertEqual(info["status"], "failed")
        self.assertIn("RuntimeError", info["error"])

    def test_all_ids(self):
        reg = RunRegistry()
        reg.start("a")
        reg.start("b")
        self.assertEqual(set(reg.all_ids()), {"a", "b"})

    def test_get_missing(self):
        reg = RunRegistry()
        self.assertIsNone(reg.get("nonexistent"))

    def test_clear(self):
        reg = RunRegistry()
        reg.start("x")
        reg.clear()
        self.assertEqual(reg.all_ids(), [])

    def test_metadata(self):
        reg = RunRegistry()
        reg.start("m1", metadata={"session": "abc"})
        info = reg.get("m1")
        self.assertEqual(info["metadata"]["session"], "abc")

    def test_finish_missing_is_noop(self):
        reg = RunRegistry()
        reg.finish("ghost", result="ok")  # should not raise
        self.assertIsNone(reg.get("ghost"))

    def test_fail_missing_is_noop(self):
        reg = RunRegistry()
        reg.fail("ghost", RuntimeError("x"))  # should not raise
        self.assertIsNone(reg.get("ghost"))

    def test_get_returns_copy(self):
        reg = RunRegistry()
        reg.start("c1")
        snapshot = reg.get("c1")
        snapshot["status"] = "tampered"
        self.assertEqual(reg.get("c1")["status"], "active")

    def test_repr_hides_lock(self):
        reg = RunRegistry()
        self.assertNotIn("_lock", repr(reg))

    def test_thread_safe_concurrent_starts(self):
        reg = RunRegistry()

        def worker(base):
            for i in range(50):
                reg.start(f"{base}-{i}")

        threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(reg.all_ids()), 8 * 50)


class AsyncContextTests(unittest.TestCase):
    def test_propagates_into_task(self):
        async def child():
            return current_run_id()

        async def main():
            with RunContext("async_run"):
                return await asyncio.create_task(child())

        self.assertEqual(asyncio.run(main()), "async_run")

    def test_isolated_between_concurrent_tasks(self):
        async def labelled(label):
            with RunContext(label):
                await asyncio.sleep(0)
                return current_run_id()

        async def main():
            return await asyncio.gather(labelled("one"), labelled("two"))

        self.assertEqual(asyncio.run(main()), ["one", "two"])


if __name__ == "__main__":
    unittest.main()
