# VERIFICATION.md — claim-by-claim source map

Every headline number in the paper, mapped to its source. Two tiers, stated
honestly:

- **Tier A — replayable artifact.** The raw run output is committed to this
  repository (or the check is keyless and runs in CI). Anyone can recompute the
  number from the artifact without an API key.
- **Tier B — transcribed run.** The number was transcribed from the console
  output of a logged interactive run (API keys or GPU required). The exact
  reproduction command is given; the numbers are internally cross-locked
  (per-family counts must sum to the reported totals, and Wilson intervals are
  recomputable from the reported fractions). Future runs should set the
  documented output variable so these become Tier A.

| Paper claim | Value | Tier | Source / reproduce |
| --- | --- | --- | --- |
| Amplification, v1 generator | 82.1% -> 100% (96/117 -> 117/117, 3 errors excluded) | A | `amp-full.json` (`rows[]`: recompute from `passAtTurn1` / `passFinal`) |
| Amplification, v2, claude | 79.2% -> 100% (95/120 -> 120/120, zero exclusions) | B | run 2026-07-16; `AMPLIFY_OUT=amp-claude-v2.json node amplify.mjs --providers=claude --generate=120 --seed=7 --turns=3` |
| Amplification, v2, deepseek | 59.3% -> 89.8% (35/59 -> 53/59) | B | same command with `--providers=deepseek` |
| Amplification, v2, grok-4 | 86.2% -> 100% (100/116 -> 116/116; 4 parse-error rows excluded; all 16 fixes at k=2) | A | `amp-grok.json` (2026-07-20) |
| Leaderboard, grok-4 | curated 10/12 avg 71 (2474 tok/solve); generated seed-7 88/120 avg 67 | A | `lb-grok-curated.json`, `lb-grok-gen.json` (2026-07-20; the generated run includes transport errors counted as misses) |
| Repair-round ablation (all fixes at k=2) | Table in paper | A (v1) / B (v2) | v1: `amp-full.json` `turnsUsed`; v2: transcribed |
| Per-family pass rates (Table 2) | 7 families 100%, conv 83%, txf 8.3%, norm 0% | B | arithmetic-locked: 7x12 + 10 + 1 + 0 = 95/120 = the reported 79.2% overall |
| Training chain, corrected protocol | 17.2% -> 76.0% (SFT) -> 80.2% (+GRPO), n=192 | B | Colab T4 runs 2026-07-16/17; commands in the paper's Reproducibility block; an independent GRPO rerun reproduced 154/192 exactly |
| SFT replications | 77.1% (n=192), 68.8% (n=64), 85.9% (v1 protocol) | B | logged runs, same commands with fresh sessions |
| SFT data-scaling curve (v2) | 12.5% / 15.1% / 26.6% / 76.0% at 236/474/945/3010 pairs (baseline 17.2%); parse fails 97/58/56/14 | B | datasets committed (`training/sft-{250,500,1k}.chat.jsonl`, holdout-excluded); Colab T4 runs 2026-07-17: `python training/train_sft.py --data training/sft-<size>.chat.jsonl --out out/sft-<size> --epochs 2` then `python training/train_grpo.py --eval-only --seed 999 --count 192 --model out/sft-<size>/checkpoint-final` |
| Train/eval overlap | 0 of 192 eval tasks in 3,010 training rows | A | keyless: `node --input-type=module -e "import {generateCases} from './generate.mjs'; import fs from 'fs'; const ev=new Set(generateCases(192,999).map(c=>c.task.spec)); const rows=fs.readFileSync('training/sft-3k.chat.jsonl','utf8').trim().split('\n').map(JSON.parse); let o=0; for (const r of rows){const u=r.messages.find(m=>m.role==='user').content; for (const s of ev) if (u.includes(s)){o++;break;}} console.log({overlap:o})"` |
| Generator v2 distinctness | 17,462 distinct tasks in a 20k draw (seed 999) | A | keyless: `node --input-type=module -e "import {generateCases} from './generate.mjs'; console.log(new Set(generateCases(20000,999).map(c=>c.task.spec)).size)"` |
| Reward audit, blatant tier (8 models) | 0% FP across all; agreement 86.7-98.3% | B | `node reward_anchor.mjs --provider=<p>` per provider (v2, n=60); the tool now prints the resolved model id |
| Reward audit, Grok variants | 95.0% agreement, 0% FP, 5.0% FN, each | A | `grok-4.5-blatant.txt`, `grok-4.20-blatant.txt`, `grok-4-fast-blatant.txt` (runs of 2026-07-17, `XAI_MODEL` per file name) |
| Near-miss collapse | qwen 46.7%, claude 33.3%, grok 25.5% FP | B | `node reward_anchor.mjs --provider=<p> --near-miss` |
| Grounding study (rubric v2) | 184/184 corrupted blocked and 0/184 complete a forward pass; 80/80 clean construct and run, 90% train; Spearman -0.15 | A | `grounding_results.csv` (2026-07-21, torch 2.8, seed 123). Keyless, no GPU: `node training/dump_grounding_set.mjs --count=80 --seed=123 --out=g.jsonl` then `python training/grounding.py --set g.jsonl --steps 60` |
| Grounding study (rubric v1, superseded) | 96/96 blocked fail forward; 88 width-corrupted graphs passed the rubric and crashed; Spearman -0.581 | B | `training/README.md` run table (2026-07-04, seeds 123+777). Retained because the v1-to-v2 delta measures what a shape-blind rubric misses |
| Curated split | claude 7/12, mean score 76; 5 failed tasks, 7 failure reasons | A | `leaderboard-data.json` on the public leaderboard (transcribed from the harness run of 2026-07-08; failure categories are per-reason, a task can trigger more than one) |
| Reasoning traces (claude) | mint yield 327/451 = 72.5% (49 API errors excluded); 306 survive rubric v2 and are the released set | A | `arch-reasoning-claude.stats.json` + `arch-reasoning-claude.card.md`; dataset public on Hugging Face |
| Reasoning traces (grok-4) | two mint runs, 439/500 = 87.8% and 437/500 = 87.4% (zero API errors in both); released set = 392-row union, every row re-graded under rubric v2 at merge | A | `arch-reasoning-grok.stats.json` + `arch-reasoning-grok.card.md`; merge via `training/combine_traces.mjs` (2026-07-21) |
| Verifier-as-tool (grok-4) | generated: raw 24/28 = 85.7% -> tool 29/29 = 100% (+14.3 pts, 1.17 audits/task); frontier: raw 28/29 = 96.6% -> tool 30/30 = 100% | A | `tooluse-grok.json`, `tooluse-grok-frontier.json` (2026-07-17); `node tool_use.mjs --provider=grok --generate=30 --seed=7 [--tier=frontier]`; one provider 429 excluded from both arms |
| Harness self-tests | reference 100%, noop 0% | A | keyless, in CI: `node calibrate.mjs --policy=reference` / `--policy=noop` |
| Satisfiability + non-vacuity | 500/500 references pass; edit-in-place starts fail untouched | A | `npx vitest run` (`generate.test.ts`) |
| Wilson 95% intervals | all bracketed values | A | recompute: `python3 -c "import math; k,n=95,120; z=1.959964; p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); print(round(100*(c-h)/d), round(100*(c+h)/d))"` |

Notes:

- **Rubric versioning.** Every number above was measured under rubric v1
  except the grounding study and the oracle row, which were re-measured under
  v2 on 2026-07-21 (both are keyless, so they were the two that could be
  re-run immediately).
  Rubric v2 (commit `f3f3bff`) later implemented Algorithm 1's linear-width
  and orphan checks, closing the grounding study's documented blind spot.
  Tier A replays that recompute from stored rows (e.g. `amp-full.json`) are
  unaffected; live re-runs at HEAD grade stricter, so convolutional-head
  tasks score lower than the v1 numbers. To reproduce v1 grading exactly:
  `git checkout f8e95fd -- bench.mjs`. `bench.mjs` exports `RUBRIC_VERSION`.
- Tier B exists because early runs did not set `AMPLIFY_OUT` (amplify) or `tee`
  (reward audits, training evals). The numbers were transcribed at run time and
  are consistent under the cross-locks above, but the raw JSON for those
  specific runs was not retained. Every command above writes an artifact when
  re-run as shown.
- The v1 amplification artifact (`amp-full.json`) is committed precisely so the
  replication claim ("v1 82 -> 100 replicates on v2 within three points") has a
  replayable side.
