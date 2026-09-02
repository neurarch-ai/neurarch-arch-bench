# results/

Raw outputs behind the numbers in the paper that were measured on rented GPUs
or through provider APIs after 2026-08-19. Everything here is written by the
scripts in this repo; nothing is transcribed by hand.

- `modal/<tag>/eval-*.json` — pass@1 evaluations written by
  `training/modal_chain.py::save` (`passed`, `n`, `parse_failures`,
  `mean_reward`, `rubric_version`). Tags: `qwen1.5b-v3` and `qwen7b-v3` are the
  rubric-v3 training chain; `ood` is the family-holdout transfer split.
- `modal/shared/mint-*.json` — the minted SFT data manifests (row counts and
  how many rows were dropped for colliding with the eval holdout).
- `modal/qwen1.5b-v3/grpo-checkpoint-50-trainer_state.json` — the GRPO run of
  2026-08-20 that stopped at step 50 of 100; kept because it explains why the RL
  row carried v1 numbers until the rerun below.
- `grounding/clean200-triples-shard*.jsonl` — (architecture, verdict, training
  curve) triples from `modal_chain.py::grounding`, the input to
  `training/quality_head.py`.
- `contamination-*.log` — `contamination.mjs` output, controls included.
- `amp-*.json` — `amplify.mjs` records (`AMPLIFY_OUT`), one per provider.
- `reward-*.log` — `reward_anchor.mjs` output per reward model and tier.

Numbers in `VERIFICATION.md` point here by file name.
