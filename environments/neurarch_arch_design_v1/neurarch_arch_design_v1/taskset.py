"""neurarch-arch-design-v1: architecture design as a verifiers v1 taskset.

Single-turn: the model reads a design spec plus a serialized starting graph and
emits ONE JSON object of structured graph edits. Reward comes from the
neurarch-arch-bench deterministic verifier (structural blockers, connectivity,
parameter budgets, required layer families) served over HTTP by the
zero-dependency env server in the parent repo:

    node env-server.mjs        # from the neurarch-arch-bench checkout

Tasks are procedurally generated from a seed, so train and eval splits are just
different seeds and a held-out split never appears on the public web.

This is the verifiers-v1-native port (taskset/harness split, verifiers>=0.2).
The legacy v0 package lives in ../neurarch_arch_design and is frozen. Run with
the tool-less `null` harness — the task is pure text-in / JSON-out:

    uv run eval --taskset.id neurarch-arch-design-v1 --harness.id null \
        --taskset.env-url http://localhost:8737

Rewards and metrics:
- `arch_reward` (weight 1.0): the server's shaped reward — pass ~1.0..1.5,
  valid-but-failing graphs earn a dense partial signal; completions that do not
  parse as a JSON action plan score -0.5.
- `task_pass` (metric): raw pass rate, undiluted by shaping.
- `parse_ok` (metric): share of completions that parsed as an action plan.
"""

import json
import re
import urllib.request

import verifiers.v1 as vf

SYSTEM_PROMPT = """You are a neural-architecture design agent. You edit a structured model graph by emitting actions.
Respond with ONE JSON object and nothing else:
{ "actions": [ <action> ... ] }

Action types:
- { "type": "add_component", "componentType": "<layer type>", "name": "<unique name>", "afterName": "<existing node>", "params": { ... } }
- { "type": "add_connection", "fromName": "<node>", "toName": "<node>" }
- { "type": "update_params", "name": "<node>", "params": { ... } }
- { "type": "delete_component", "name": "<node>" }
- { "type": "replace_model", "components": [ { "componentType": "...", "name": "...", "params": {...} } ], "connections": [ { "from": "...", "to": "..." } ] }

Rules:
- Param keys: linear {inFeatures,outFeatures}; conv2d {inChannels,outChannels,kernelSize}; embedding {numEmbeddings,embeddingDim}; multiHeadAttention {embedDim,numHeads}; groupedQueryAttention {embedDim,numHeads,numKVHeads}; batchNorm1d {numFeatures}; layerNorm {normalizedShape}; concatenate {dim}.
- For attention, embedDim MUST be divisible by numHeads; for GQA, numHeads MUST be divisible by numKVHeads.
- Chain linear layers so each inFeatures matches the upstream output width.
- Respect any parameter budget in the spec.
- All integer values must be plain integers, never arithmetic expressions.
- If the spec says to repair or edit in place, use surgical actions; do NOT use replace_model or clear_canvas.
- Output only the JSON object."""

PARSE_FAILURE_REWARD = -0.5


def _http_json(url, payload=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def parse_actions(text):
    """Balanced extraction of the {"actions": [...]} object from model text."""
    s = re.sub(r"```json\n?|```\n?", "", str(text or "").strip())
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    actions = obj.get("actions")
                    if isinstance(actions, list):
                        return actions
                    break
        start = s.find("{", start + 1)
    return None


class ArchDesignData(vf.TaskData):
    # Coordinates the env server needs to regenerate + grade this exact task.
    seed: int
    count: int
    index: int


class ArchDesignTaskConfig(vf.TaskConfig):
    env_url: str = "http://localhost:8737"
    """Base URL of the neurarch-arch-bench env server (node env-server.mjs)."""


class ArchDesignTask(vf.Task[ArchDesignData, vf.State, ArchDesignTaskConfig]):
    def _grade(self, trace: vf.Trace):
        actions = parse_actions(trace.last_reply)
        if actions is None:
            return None
        return _http_json(
            f"{self.config.env_url}/grade",
            {
                "seed": self.data.seed,
                "count": self.data.count,
                "index": self.data.index,
                "actions": actions,
            },
        )

    @vf.reward(weight=1.0)
    async def arch_reward(self, trace: vf.Trace) -> float:
        graded = self._grade(trace)
        if graded is None:
            return PARSE_FAILURE_REWARD
        return float(graded["reward"])

    @vf.metric
    async def task_pass(self, trace: vf.Trace) -> float:
        graded = self._grade(trace)
        return 1.0 if graded is not None and graded["pass"] else 0.0

    @vf.metric
    async def parse_ok(self, trace: vf.Trace) -> float:
        return 1.0 if parse_actions(trace.last_reply) is not None else 0.0


class ArchDesignConfig(vf.TasksetConfig):
    count: int = 256
    """Task-set size requested from the procedural generator."""
    seed: int = 123
    """Generator seed. Different seed = disjoint split (train/eval/held-out)."""
    task: ArchDesignTaskConfig = ArchDesignTaskConfig()


class NeurarchArchDesignTaskset(vf.Taskset[ArchDesignTask, ArchDesignConfig]):
    def load(self) -> list[ArchDesignTask]:
        env_url = self.config.task.env_url
        try:
            _http_json(f"{env_url}/health")
        except Exception as err:  # pragma: no cover - env misconfiguration path
            raise RuntimeError(
                f"neurarch-arch-bench env server not reachable at {env_url}. "
                "Start it with `node env-server.mjs` from the neurarch-arch-bench "
                "checkout (zero dependencies, node >= 18), or set "
                "--taskset.task.env-url."
            ) from err

        tasks = _http_json(
            f"{env_url}/tasks?count={self.config.count}&seed={self.config.seed}"
        )
        return [
            ArchDesignTask(
                ArchDesignData(
                    idx=t["index"],
                    prompt=(
                        f"{SYSTEM_PROMPT}\n\n"
                        f"SPEC:\n{t['spec']}\n\n"
                        f"CURRENT MODEL:\n{t['observation']}\n\n"
                        "Return the actions that fulfil the spec."
                    ),
                    seed=self.config.seed,
                    count=self.config.count,
                    index=t["index"],
                ),
                self.config.task,
            )
            for t in tasks
        ]
