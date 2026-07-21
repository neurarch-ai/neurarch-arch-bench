"""Keyless smoke test for the v1 taskset.

Runs WITHOUT verifiers installed: injects a minimal stub of the `verifiers.v1`
API surface the taskset uses (TaskData/Task/TasksetConfig/Taskset/State/Trace +
reward/metric decorators), then exercises the real HTTP loop against a live
env server:

    node env-server.mjs &          # from the repo root
    python3 environments/neurarch_arch_design_v1/test_v1_smoke.py

Asserts: load() materializes the requested split; a noop plan parses but fails
(reward is a finite float, task_pass 0); garbage text hits the parse-failure
reward; balanced extraction survives prose after the JSON. This proves the
package's own logic end-to-end; running under real verifiers additionally needs
`uv run eval` (see README).
"""
import asyncio
import importlib.util
import json
import os
import sys
import types
import urllib.request
from pathlib import Path

ENV_URL = os.environ.get("NEURARCH_ENV_URL", "http://localhost:8737")

# ── Stub the verifiers.v1 surface the taskset imports ─────────────────────────

def _make_stub():
    v1 = types.ModuleType("verifiers.v1")

    class _Base:
        def __init__(self, *args, **kwargs):
            # Positional style: Task(data, task_config)
            if args and not kwargs:
                self.data = args[0]
                self.config = args[1] if len(args) > 1 else None
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __class_getitem__(cls, _item):
            return cls

    class TaskData(_Base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class _Config(_Base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class Taskset(_Base):
        def __init__(self, config):
            self.config = config

    def _identity_decorator(fn=None, **_kw):
        if fn is None:
            return lambda f: f
        return fn

    v1.TaskData = TaskData
    v1.TaskConfig = _Config
    v1.TasksetConfig = _Config
    v1.State = object
    v1.Task = _Base
    v1.Taskset = Taskset
    v1.Trace = object
    v1.reward = _identity_decorator
    v1.metric = _identity_decorator
    v1.stop = _identity_decorator
    v1.tool = _identity_decorator

    verifiers = types.ModuleType("verifiers")
    verifiers.v1 = v1
    sys.modules["verifiers"] = verifiers
    sys.modules["verifiers.v1"] = v1


def _load_taskset_module():
    pkg_dir = Path(__file__).parent / "neurarch_arch_design_v1"
    # Register the package so `from neurarch_arch_design_v1.taskset import ...` works
    pkg = types.ModuleType("neurarch_arch_design_v1")
    pkg.__path__ = [str(pkg_dir)]
    sys.modules["neurarch_arch_design_v1"] = pkg
    spec = importlib.util.spec_from_file_location(
        "neurarch_arch_design_v1.taskset", pkg_dir / "taskset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["neurarch_arch_design_v1.taskset"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeTrace:
    def __init__(self, last_reply):
        self.last_reply = last_reply


def main():
    # Server must be up
    with urllib.request.urlopen(f"{ENV_URL}/health", timeout=10) as r:
        assert json.loads(r.read()).get("ok", True) is not False

    _make_stub()
    m = _load_taskset_module()

    # parse_actions: balanced extraction, fenced input, prose-after-JSON
    plan = '{"actions":[{"type":"add_component","componentType":"relu","name":"a1"}]}'
    assert m.parse_actions(plan) is not None
    assert m.parse_actions(f"```json\n{plan}\n```") is not None
    assert m.parse_actions(plan + "\nWait, let me reconsider {oops") is not None
    assert m.parse_actions("no json at all") is None
    print("parse_actions: OK")

    # load() against the live server
    cfg = m.ArchDesignConfig()
    cfg.count = 12
    cfg.seed = 7
    cfg.task = m.ArchDesignTaskConfig()
    cfg.task.env_url = ENV_URL
    ts = m.NeurarchArchDesignTaskset(cfg)
    tasks = ts.load()
    assert len(tasks) == 12, f"expected 12 tasks, got {len(tasks)}"
    assert "SPEC:" in tasks[0].data.prompt and "CURRENT MODEL:" in tasks[0].data.prompt
    print(f"load(): OK ({len(tasks)} tasks)")

    # Grade paths through the real server
    t0 = tasks[0]
    noop_reward = asyncio.run(t0.arch_reward(FakeTrace('{"actions": []}')))
    assert isinstance(noop_reward, float), noop_reward
    noop_pass = asyncio.run(t0.task_pass(FakeTrace('{"actions": []}')))
    assert noop_pass == 0.0, "a noop plan must not pass an untouched-failing task"
    garbage = asyncio.run(t0.arch_reward(FakeTrace("I refuse to answer in JSON")))
    assert garbage == m.PARSE_FAILURE_REWARD, garbage
    parse_metric = asyncio.run(t0.parse_ok(FakeTrace('{"actions": []}')))
    assert parse_metric == 1.0
    print(f"grade paths: OK (noop reward {noop_reward:.3f}, parse-failure {garbage})")

    # Every task in the split grades without server error (noop sweep)
    fails = 0
    for t in tasks:
        graded = t._grade(FakeTrace('{"actions": []}'))
        assert graded is not None and "reward" in graded and "pass" in graded
        fails += 0 if graded["pass"] else 1
    print(f"noop sweep: OK ({fails}/{len(tasks)} tasks correctly not passed by noop)")

    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
