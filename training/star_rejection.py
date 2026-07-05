#!/usr/bin/env python3
"""STaR-style self-improvement on the architecture-design environment.

The third leg of the post-training tripod (next to GRPO in train_grpo.py and
plain SFT on build_sft_dataset output): rejection sampling with a verifier.

Each round:
  1. SAMPLE  k action plans per training task at temperature > 0
  2. VERIFY  each plan with the deterministic env server (/grade)
  3. KEEP    only plans that PASS the task's constraints
  4. SFT     the model on its own verified successes
  5. EVAL    pass@1 on a held-out seed the model never trains on

No labels, no reward model, no LLM judge anywhere in the loop: the verifier
is the whole supervision signal. That this loop runs at all is the point the
script demonstrates; expect diminishing returns after 2-3 rounds on a small
model.

Usage (env server must be running: `node ../env-server.mjs`):
  pip install "trl>=0.14" transformers datasets accelerate peft
  python star_rejection.py --rounds 2 --count 256 --k 4 --seed 123 \
      --eval-seed 999 --eval-count 64 --lora

Hardware: defaults (Qwen2.5-1.5B-Instruct + --lora) fit a Colab T4.
"""
import argparse
import json
import re
import urllib.request


def http_json(url, payload=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


SYSTEM = None  # loaded lazily from train_grpo to keep one prompt definition
from train_grpo import SYSTEM as _SYSTEM, build_prompt, parse_actions, fetch_tasks, grade  # noqa: E402
SYSTEM = _SYSTEM


def generate_plans(model, tok, tasks, k, max_new, temperature):
    """Sample k completions per task; returns list of (task, text)."""
    import torch
    out = []
    for t in tasks:
        text = tok.apply_chat_template(build_prompt(t), tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=k,
                pad_token_id=tok.eos_token_id,
            )
        for row in gen:
            out.append((t, tok.decode(row[inputs["input_ids"].shape[1]:], skip_special_tokens=True)))
    return out


def evaluate(model, tok, env_url, seed, count, max_new):
    import torch
    tasks = fetch_tasks(env_url, count, seed)
    passed = 0
    for t in tasks:
        text = tok.apply_chat_template(build_prompt(t), tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        reply = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        actions = parse_actions(reply)
        if actions is None:
            continue
        if grade(env_url, seed, count, t["index"], actions)["pass"]:
            passed += 1
    return passed / len(tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-url", default="http://localhost:8737")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--count", type=int, default=256, help="training tasks per round")
    ap.add_argument("--k", type=int, default=4, help="samples per task")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--eval-seed", type=int, default=999)
    ap.add_argument("--eval-count", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--out", default="out/star-arch")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    ok = http_json(f"{args.env_url}/health")
    if not ok.get("ok"):
        raise SystemExit(f"env server not healthy at {args.env_url}; run: node ../env-server.mjs")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    baseline = evaluate(model, tok, args.env_url, args.eval_seed, args.eval_count, args.max_completion)
    print(f"[round 0] held-out pass@1 = {baseline:.3f}")

    history = [{"round": 0, "heldout_pass1": baseline}]
    for rnd in range(1, args.rounds + 1):
        # Fresh task seed per round so the model can't memorize one split.
        round_seed = args.seed + rnd
        tasks = fetch_tasks(args.env_url, args.count, round_seed)

        plans = generate_plans(model, tok, tasks, args.k, args.max_completion, args.temperature)
        kept = []
        for t, text in plans:
            actions = parse_actions(text)
            if actions is None:
                continue
            g = grade(args.env_url, round_seed, args.count, t["index"], actions)
            if g["pass"]:
                kept.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": build_prompt(t)[1]["content"]},
                        {"role": "assistant", "content": json.dumps({"actions": actions})},
                    ],
                })
        accept = len(kept) / max(1, len(plans))
        print(f"[round {rnd}] sampled {len(plans)}, verifier kept {len(kept)} ({accept:.1%})")
        if not kept:
            print("Nothing passed; raise --k or --temperature, or start from a stronger model.")
            break

        peft_config = None
        if args.lora:
            from peft import LoraConfig
            peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                     task_type="CAUSAL_LM")
        cfg = SFTConfig(
            output_dir=f"{args.out}/round-{rnd}",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            logging_steps=5,
            report_to=[],
        )
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(kept),
                             processing_class=tok, peft_config=peft_config)
        trainer.train()
        model = trainer.model

        score = evaluate(model, tok, args.env_url, args.eval_seed, args.eval_count, args.max_completion)
        print(f"[round {rnd}] held-out pass@1 = {score:.3f} (baseline {baseline:.3f})")
        history.append({"round": rnd, "heldout_pass1": score, "kept": len(kept), "acceptance": accept})

    with open(f"{args.out.rstrip('/')}-history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"history -> {args.out.rstrip('/')}-history.json")


if __name__ == "__main__":
    main()
