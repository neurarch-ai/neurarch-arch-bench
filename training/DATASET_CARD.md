---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - neural-architecture
  - agents
  - verifiable
  - synthetic
  - sft
pretty_name: arch-design-sft
size_categories:
  - 10K<n<100K
---

# arch-design-sft

Verified supervised fine-tuning data for **neural-architecture design as
structured graph editing**. Each row pairs a natural-language design spec and
a serialized starting graph with a reference action plan, and **every row is
re-graded by a deterministic verifier before it is written**: structural
blockers (attention head divisibility, connectivity), parameter budgets and
bands, required layer families. 100% of targets pass the
[neurarch-arch-bench](https://github.com/neurarch-ai/neurarch-arch-bench)
constraints; the build fails if even one does not.

## Why this data is different

- **Machine-checked targets.** No scraped code of unknown quality: the
  assistant turn is an action plan proven to satisfy the task's constraints
  by the same verifier that grades the public benchmark.
- **Contamination-free by construction.** Tasks are procedurally generated
  from a seed, not collected from the web. Evaluate on a different seed (or a
  private one) and the eval split has never existed anywhere.
- **Ten task families.** Six design-from-spec (MLP, autoencoder, CNN,
  transformer encoder, GQA encoder, two-tower retrieval) and four
  edit-in-place (repair a broken attention config, trim over a budget, grow
  into a param band, insert normalization) where wholesale rebuilds are
  forbidden, so the data teaches surgical edits, not just generation.

## Files

- `*.chat.jsonl`: `{messages: [system, user, assistant]}`; the assistant
  message is the JSON action plan. Drop-in for TRL `SFTTrainer` and most chat
  SFT stacks.
- `*.raw.jsonl`: `{id, family, seed, spec, observation, constraints, actions,
  grade}` for custom formatting or filtering.

## Reproduce / scale / mint a private split

```bash
git clone https://github.com/neurarch-ai/neurarch-arch-bench
cd neurarch-arch-bench/training
node build_sft_dataset.mjs --count=10000 --seed=20260704 --out=arch-design-sft
```

Same (count, seed) reproduces the file byte-for-byte; a new seed mints a
disjoint split of the same distribution.

## Honest limitations

- Targets are structurally valid and budget-respecting; the verifier does not
  claim they train to the best final metric (see the repo's grounding study:
  the pass/blocked boundary is grounded against real PyTorch behavior; the
  score magnitude is a validity margin, not a quality ranking).
- Reference plans mostly use `replace_model` for design-from-spec families
  and surgical actions for edit-in-place families; models trained only on
  this data will inherit that style.
- Synthetic distribution: real user specs are messier. Mix with your own
  instruction data as appropriate.

## License and citation

MIT. Built by [Neurarch](https://neurarch.com) from the open
neurarch-arch-bench generator and verifier.
