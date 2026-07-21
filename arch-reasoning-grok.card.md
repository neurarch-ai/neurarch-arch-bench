---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- reasoning
- neural-architecture-design
- verified
- rejection-sampling
- rlvr
pretty_name: Verified architecture-design reasoning traces (grok-4)
size_categories:
- n<1K
---

# arch-reasoning-grok

**376 verified (spec -> reasoning -> design) traces for neural-architecture
design, minted by grok-4 and filtered by a deterministic verifier.**

Each row pairs a natural-language architecture spec with grok-4's reasoning and
its final structured design (a batch of typed graph edits). Every row's design
was applied to the task's start graph and re-graded by the deterministic
verifier of [neurarch-arch-bench](https://github.com/neurarch-ai/neurarch-arch-bench):
no trace enters this set unless its design provably solves its spec. No human
judge, no LLM judge.

## Provenance and yield

- Model: `grok-4` (xAI API), rejection-sampled at up to 2 tries per task.
- Tasks: 500 procedurally generated (generator v2, seed 20260708), spanning
  ten families (MLP, autoencoder, CNN, transformer, GQA encoder, two-tower,
  and four edit-in-place repair/scaling families).
- Mint-time yield: 439/500 = 87.8% verified (zero API failures).
- Released rows: **376**, each re-verified under the repository's current
  rubric (v2), which adds the linear-width consistency check of the paper's
  Algorithm 1. The 63 mint-time-verified rows that fail v2's stricter width
  check are excluded; they are exactly the blind-spot class the paper's
  grounding study documents.

## Schema

One JSON object per line: `task_id`, `spec`, `observation` (serialized start
graph), `reasoning`, `actions` (the verified edit batch), `verified: true`,
`source`, and a chat-format `messages` array ready for SFT.

## Reproduce / extend

```bash
git clone https://github.com/neurarch-ai/neurarch-arch-bench
XAI_API_KEY=... node training/reasoning_traces.mjs --provider=grok --count=500 --delay=800 --out=my-traces
```

A private seed produces reasoning data whose designs never appeared on the
public web. Companion set: [arch-reasoning-claude](https://huggingface.co/datasets/neurarch-ai/arch-reasoning-claude).
