"""neurarch-arch-design: a verifiers environment for neural-architecture design.

Single-turn: the model reads a design spec plus a serialized starting graph and
emits a JSON plan of structured graph edits. Reward comes from the
neurarch-arch-bench deterministic verifier (structural blockers, connectivity,
parameter budgets, required layer families) served over HTTP by the
zero-dependency env server in the parent repo:

    node env-server.mjs        # from the neurarch-arch-bench checkout

Tasks are procedurally generated from a seed, so train and eval splits are
just different seeds and a held-out split never appears on the public web.

Usage:
    import verifiers as vf
    env = vf.load_environment("neurarch-arch-design", count=256, seed=123)

Reward: the server's shaped reward (pass ~1.0..1.5, valid-but-failing graphs
earn a dense partial signal, malformed edits cost a little); completions that
do not parse as a JSON action plan score -0.5. A weight-0 `task_pass` metric
reports the raw pass rate.
"""
import json
import os
import re
import urllib.request

import verifiers as vf
from datasets import Dataset

DEFAULT_ENV_URL = os.environ.get("NEURARCH_ENV_URL", "http://localhost:8737")

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


def _parse_actions(text):
    s = re.sub(r"```json\n?|```\n?", "", str(text).strip())
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    actions = obj.get("actions")
    return actions if isinstance(actions, list) else None


def _completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return ""


def _grade(env_url, info, actions):
    return _http_json(f"{env_url}/grade", {
        "seed": info["seed"], "count": info["count"], "index": info["index"],
        "actions": actions,
    })


def load_environment(
    env_url: str = DEFAULT_ENV_URL,
    count: int = 256,
    seed: int = 123,
    **kwargs,
) -> vf.Environment:
    try:
        _http_json(f"{env_url}/health")
    except Exception as err:
        raise RuntimeError(
            f"neurarch-arch-bench env server not reachable at {env_url}. "
            "Start it with `node env-server.mjs` from the neurarch-arch-bench "
            "checkout (zero dependencies, node >= 18), or set NEURARCH_ENV_URL."
        ) from err

    tasks = _http_json(f"{env_url}/tasks?count={count}&seed={seed}")
    dataset = Dataset.from_list([
        {
            "prompt": [{
                "role": "user",
                "content": f"SPEC:\n{t['spec']}\n\nCURRENT MODEL:\n{t['observation']}\n\nReturn the actions that fulfil the spec.",
            }],
            "info": {"index": t["index"], "seed": seed, "count": count},
        }
        for t in tasks
    ])

    def arch_reward(completion, info) -> float:
        actions = _parse_actions(_completion_text(completion))
        if actions is None:
            return PARSE_FAILURE_REWARD
        return float(_grade(env_url, info, actions)["reward"])

    def task_pass(completion, info) -> float:
        """Weight-0 metric: raw pass rate, undiluted by shaping."""
        actions = _parse_actions(_completion_text(completion))
        if actions is None:
            return 0.0
        return 1.0 if _grade(env_url, info, actions)["pass"] else 0.0

    rubric = vf.Rubric(funcs=[arch_reward, task_pass], weights=[1.0, 0.0])

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
        **kwargs,
    )
