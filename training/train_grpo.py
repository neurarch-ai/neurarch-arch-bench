#!/usr/bin/env python3
"""GRPO-train a small open model to design neural architectures against the
neurarch-arch-bench verifiable-reward environment.

The environment is the HTTP server in ../env-server.mjs (plain node, zero
deps). It serves deterministic task splits and grades action plans with the
same programmatic verifier the benchmark leaderboard uses. This script:

  1. pulls a training split from the server (seed-addressable, so the
     held-out split is just a different seed),
  2. builds prompts (spec + serialized start graph),
  3. runs TRL's GRPOTrainer with a reward function that POSTs each sampled
     completion to /grade and returns the server's shaped reward.

Reward (computed server-side, documented in env-server.mjs):
  pass -> ~1.0..1.5, valid-but-failing -> ~0.2..0.5, broken -> ~0..0.2.
  A completion that does not parse as a JSON action plan gets -0.5 here,
  below every server reward, so "emit valid JSON" is learned first.

Usage (env server must be running: `node ../env-server.mjs`):

  # baseline pass rate on a held-out split (no training)
  python train_grpo.py --eval-only --seed 999 --count 64

  # train on the seed-123 split
  python train_grpo.py --steps 300 --count 512 --seed 123 --lora

  # re-evaluate the trained checkpoint
  python train_grpo.py --eval-only --seed 999 --count 64 \
      --model out/grpo-arch/checkpoint-final

Hardware: the default Qwen2.5-1.5B-Instruct with --lora fits a single T4
(Colab free tier) in 4-bit; an A100/H100 trains comfortably in bf16.
"""
import argparse
import json
import re
import urllib.request


# ── Env-server client ────────────────────────────────────────────────────────

def http_json(url: str, payload=None):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def fetch_tasks(env_url: str, count: int, seed: int):
    return http_json(f"{env_url}/tasks?count={count}&seed={seed}")


def grade(env_url: str, seed: int, count: int, index: int, actions):
    return http_json(
        f"{env_url}/grade",
        {"seed": seed, "count": count, "index": index, "actions": actions},
    )


# ── Prompting ────────────────────────────────────────────────────────────────

SYSTEM = """You are a neural-architecture design agent. You edit a structured model graph by emitting actions.
Respond with ONE JSON object and nothing else:
{ "actions": [ <action> ... ] }

Action types:
- { "type": "add_component", "componentType": "<layer type>", "name": "<unique name>", "afterName": "<existing node>", "params": { ... } }
- { "type": "add_connection", "fromName": "<node>", "toName": "<node>" }
- { "type": "update_params", "name": "<node>", "params": { ... } }
- { "type": "delete_component", "name": "<node>" }
- { "type": "replace_model", "components": [ { "componentType": "...", "name": "...", "params": {...} } ], "connections": [ { "from": "...", "to": "..." } ] }

Rules:
- Param keys: linear {inFeatures,outFeatures}; conv2d {inChannels,outChannels,kernelSize}; embedding {numEmbeddings,embeddingDim}; multiHeadAttention {embedDim,numHeads}; groupedQueryAttention {embedDim,numHeads,numKVHeads}; batchNorm1d {numFeatures}; layerNorm {normalizedShape}.
- For attention, embedDim MUST be divisible by numHeads; for GQA, numHeads MUST be divisible by numKVHeads.
- Chain linear layers so each inFeatures matches the upstream output width.
- Respect any parameter budget in the spec.
- If the spec says to repair or edit in place, use surgical actions; do NOT use replace_model or clear_canvas.
- Output only the JSON object."""


def build_prompt(task) -> list:
    user = (
        f"SPEC:\n{task['spec']}\n\n"
        f"CURRENT MODEL:\n{task['observation']}\n\n"
        "Return the actions that fulfil the spec."
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_actions(text: str):
    """Extract {"actions": [...]} from a completion. None if unparseable."""
    s = text.strip()
    s = re.sub(r"```json\n?|```\n?", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    actions = obj.get("actions")
    return actions if isinstance(actions, list) else None


def completion_text(completion) -> str:
    """TRL passes either a plain string or a chat-format message list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return ""


PARSE_FAILURE_REWARD = -0.5


# ── Eval (pass@1 on a split, no training) ───────────────────────────────────

def run_eval(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = fetch_tasks(args.env_url, args.count, args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    passed, parse_failures, total_reward = 0, 0, 0.0
    for t in tasks:
        text = tok.apply_chat_template(
            build_prompt(t), tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_completion,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        actions = parse_actions(reply)
        if actions is None:
            parse_failures += 1
            total_reward += PARSE_FAILURE_REWARD
            print(f"[PARSE-FAIL] {t['id']}")
            continue
        g = grade(args.env_url, args.seed, args.count, t["index"], actions)
        passed += 1 if g["pass"] else 0
        total_reward += g["reward"]
        status = "PASS" if g["pass"] else "FAIL"
        print(f"[{status}] {t['id']} score={g['score']} reward={g['reward']:.2f}"
              + ("" if g["pass"] else f"  ({'; '.join(g['failures'][:2])})"))
    n = len(tasks)
    print(f"\nmodel={args.model}")
    print(f"split: seed={args.seed} count={n}")
    print(f"pass@1: {passed}/{n} = {passed / n:.3f}")
    print(f"parse failures: {parse_failures}/{n}")
    print(f"mean reward: {total_reward / n:.3f}")


# ── GRPO training ────────────────────────────────────────────────────────────

def run_train(args):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    tasks = fetch_tasks(args.env_url, args.count, args.seed)
    ds = Dataset.from_list([
        {
            "prompt": build_prompt(t),
            "task_index": t["index"],
            "task_seed": args.seed,
            "task_count": args.count,
        }
        for t in tasks
    ])
    print(f"training split: seed={args.seed}, {len(ds)} tasks")

    def arch_reward(completions, task_index=None, task_seed=None, task_count=None, **kwargs):
        rewards = []
        for completion, idx, sd, cnt in zip(completions, task_index, task_seed, task_count):
            actions = parse_actions(completion_text(completion))
            if actions is None:
                rewards.append(PARSE_FAILURE_REWARD)
                continue
            try:
                rewards.append(grade(args.env_url, sd, cnt, idx, actions)["reward"])
            except Exception as err:  # env server hiccup: neutral, not fatal
                print(f"grade error on task {idx}: {err}")
                rewards.append(0.0)
        return rewards

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

    cfg = GRPOConfig(
        output_dir=args.out,
        max_steps=args.steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        logging_steps=1,
        save_steps=max(50, args.steps // 4),
        bf16=args.bf16,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=arch_reward,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train()
    final = f"{args.out}/checkpoint-final"
    trainer.save_model(final)
    print(f"saved {final}")
    print(f"reward curve: {args.out}/trainer_state.json (log_history[].reward)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-url", default="http://localhost:8737")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--count", type=int, default=256, help="tasks in the split")
    p.add_argument("--seed", type=int, default=123, help="split seed (use a different seed for held-out eval)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8, help="GRPO group size")
    p.add_argument("--max-prompt", type=int, default=1024)
    p.add_argument("--max-completion", type=int, default=512)
    p.add_argument("--out", default="out/grpo-arch")
    p.add_argument("--lora", action="store_true", help="LoRA instead of full finetune")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="report pass@1 on the split, no training")
    args = p.parse_args()

    ok = http_json(f"{args.env_url}/health")
    if not ok.get("ok"):
        raise SystemExit(f"env server not healthy at {args.env_url}; run: node ../env-server.mjs")

    if args.eval_only:
        run_eval(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
