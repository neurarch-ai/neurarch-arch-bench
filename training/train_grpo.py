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

    import os, json as _json
    if getattr(args, "curated", False):
        tasks = http_json(f"{args.env_url}/tasks?split=curated")
        print(f"evaluating on the curated split ({len(tasks)} hand-authored tasks)")
    else:
        tasks = fetch_tasks(args.env_url, args.count, args.seed)

    # fp16 on GPUs without bf16 (e.g. T4); low_cpu_mem_usage avoids a RAM spike
    # that can kill the Colab kernel while loading.
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    load_kw = dict(torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True)

    is_local = os.path.isdir(args.model)
    adapter_cfg = os.path.join(args.model, "adapter_config.json")
    if is_local and os.path.exists(adapter_cfg):
        # A LoRA adapter (what --lora training saves): load the base model, then
        # apply the adapter. Loading the adapter dir directly would fail.
        from peft import PeftModel
        base = _json.load(open(adapter_cfg)).get("base_model_name_or_path") or "Qwen/Qwen2.5-1.5B-Instruct"
        tok_src = args.model if os.path.exists(os.path.join(args.model, "tokenizer_config.json")) else base
        tok = AutoTokenizer.from_pretrained(tok_src)
        model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
        model = PeftModel.from_pretrained(model, args.model)
    else:
        if not is_local and args.model.count("/") != 1:
            raise SystemExit(
                f"'{args.model}' is neither a local directory nor a valid HF repo id "
                f"('namespace/name'). If it is a trained checkpoint, run eval from the "
                f"directory that contains '{args.model}' (watch for a nested "
                f"neurarch-arch-bench/neurarch-arch-bench folder)."
            )
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
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
        if getattr(args, "curated", False):
            g = http_json(f"{args.env_url}/grade", {"taskId": t["id"], "actions": actions})
        else:
            g = grade(args.env_url, args.seed, args.count, t["index"], actions)
        passed += 1 if g["pass"] else 0
        total_reward += g["reward"]
        status = "PASS" if g["pass"] else "FAIL"
        print(f"[{status}] {t['id']} score={g['score']} reward={g['reward']:.2f}"
              + ("" if g["pass"] else f"  ({'; '.join(g['failures'][:2])})"))
    n = len(tasks)
    print(f"\nmodel={args.model}")
    print("split: curated" if getattr(args, "curated", False) else f"split: seed={args.seed} count={n}")
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

    # GRPO requires the per-device batch (the generation batch) to be divisible
    # by num_generations (the group size). Auto-adjust so any batch/group combo
    # runs instead of erroring in TRL.
    if args.batch_size % args.num_generations != 0:
        divisors = [d for d in range(2, args.batch_size + 1)
                    if args.batch_size % d == 0 and d <= args.num_generations]
        ng = max(divisors) if divisors else args.batch_size
        print(f"[warn] num_generations {args.num_generations} does not divide "
              f"batch_size {args.batch_size}; using num_generations={ng}.")
        args.num_generations = ng

    # T4 (Turing) GPUs have no bf16; fall back to fp16 so --bf16 does not crash.
    use_bf16, use_fp16 = args.bf16, False
    if args.bf16:
        import torch
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            use_bf16, use_fp16 = False, True
            print("[warn] bf16 not supported on this GPU (e.g. T4); using fp16 instead.")

    # Build kwargs, then keep only fields this TRL version's GRPOConfig accepts,
    # so the script survives TRL API drift (e.g. renamed/removed length args).
    import dataclasses
    desired = dict(
        output_dir=args.out,
        max_steps=args.steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        logging_steps=1,
        save_steps=max(50, args.steps // 4),
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
    )
    accepted = {f.name for f in dataclasses.fields(GRPOConfig)}
    cfg_kwargs = {k: v for k, v in desired.items() if k in accepted}
    dropped = [k for k in desired if k not in accepted]
    if dropped:
        print(f"[warn] GRPOConfig (TRL {getattr(__import__('trl'), '__version__', '?')}) "
              f"does not accept {dropped}; using its defaults for those.")
    cfg = GRPOConfig(**cfg_kwargs)
    # If --model is a LoRA adapter dir (what train_sft.py saves), GRPOTrainer
    # cannot load it as a path; merge the adapter into its base model first.
    import os as _os, json as _json2
    model_arg = args.model
    _acfg = _os.path.join(str(args.model), "adapter_config.json")
    if _os.path.isdir(str(args.model)) and _os.path.exists(_acfg):
        import torch as _torch
        from transformers import AutoModelForCausalLM as _AM
        from peft import PeftModel as _PM
        _base = _json2.load(open(_acfg)).get("base_model_name_or_path") or "Qwen/Qwen2.5-1.5B-Instruct"
        _dtype = _torch.bfloat16 if (_torch.cuda.is_available() and _torch.cuda.is_bf16_supported()) else _torch.float16
        print(f"[info] {args.model} is a LoRA adapter; merging into base {_base} for RL")
        _m = _AM.from_pretrained(_base, torch_dtype=_dtype, low_cpu_mem_usage=True)
        model_arg = _PM.from_pretrained(_m, args.model).merge_and_unload()

    trainer = GRPOTrainer(
        model=model_arg,
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
    p.add_argument("--lr", type=float, default=1e-5)  # 1e-6 was too low to learn; 1e-5 is a better LoRA-RL default
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=4, help="GRPO group size (must divide batch-size)")
    p.add_argument("--max-prompt", type=int, default=1024)
    p.add_argument("--max-completion", type=int, default=384)  # valid designs are short; 512 wastes gen time on ramblers
    p.add_argument("--out", default="out/grpo-arch")
    p.add_argument("--lora", action="store_true", help="LoRA instead of full finetune")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="report pass@1 on the split, no training")
    p.add_argument("--curated", action="store_true", help="eval on the 12 curated tasks instead of a generated split")
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
