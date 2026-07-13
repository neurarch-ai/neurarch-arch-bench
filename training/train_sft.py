#!/usr/bin/env python3
"""SFT a small model on VERIFIED (spec -> actions) pairs minted by the
environment itself — the standard fix when a policy is too weak for RL.

Every generated task carries a reference solution that provably passes the
grader, so the environment is an unlimited, keyless source of verified
supervised data (training/build_sft_dataset.mjs). A small model that fails RL
because it cannot even emit valid graph-edit JSON learns exactly that here;
after SFT, evaluate with train_grpo.py --eval-only (same held-out protocol),
and optionally run GRPO on top of the SFT checkpoint.

Usage (data first, no API key needed):
  node build_sft_dataset.mjs --count=3000 --seed=20260713 --out=sft-3k
  python train_sft.py --data sft-3k.chat.jsonl --epochs 2
  python train_grpo.py --eval-only --seed 999 --count 64 --model out/sft-arch/checkpoint-final

Hardware: Qwen2.5-1.5B + LoRA fits a free Colab T4 (~30-60 min for 3k rows).
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="chat-format jsonl from build_sft_dataset.mjs (<out>.chat.jsonl)")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="out/sft-arch")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4, help="LoRA SFT default; far higher than RL lr is normal")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig

    rows = [json.loads(l) for l in open(args.data)]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print(f"training on {len(ds)} verified rows from {args.data}")

    # T4 has no bf16; fall back to fp16 (same guard as train_grpo.py).
    use_bf16 = args.bf16 or True
    use_fp16 = False
    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        use_bf16, use_fp16 = False, True
        print("[warn] bf16 not supported on this GPU; using fp16.")

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # Keep only kwargs this TRL version's SFTConfig accepts (survives API drift,
    # same trick as train_grpo.py).
    import dataclasses
    desired = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        max_seq_length=args.max_len,       # older TRL name; one of the two survives
        logging_steps=20,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
    )
    accepted = {f.name for f in dataclasses.fields(SFTConfig)}
    cfg_kwargs = {k: v for k, v in desired.items() if k in accepted}
    dropped = [k for k in desired if k not in accepted]
    if dropped:
        print(f"[warn] SFTConfig does not accept {dropped}; using defaults for those.")
    cfg = SFTConfig(**cfg_kwargs)

    trainer = SFTTrainer(
        model=args.model,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train()
    final = f"{args.out}/checkpoint-final"
    trainer.save_model(final)
    print(f"saved {final}")
    print("now evaluate: python training/train_grpo.py --eval-only --seed 999 --count 64 "
          f"--model {final}")


if __name__ == "__main__":
    main()
