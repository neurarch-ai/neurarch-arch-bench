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
| Repair-round ablation (all fixes at k=2) | Table in paper | A (v1) / B (v2) | v1: `amp-full.json` `turnsUsed`; v2: transcribed |
| Per-family pass rates (Table 2) | 7 families 100%, conv 83%, txf 8.3%, norm 0% | B | arithmetic-locked: 7x12 + 10 + 1 + 0 = 95/120 = the reported 79.2% overall |
| Training chain, corrected protocol | 17.2% -> 76.0% (SFT) -> 80.2% (+GRPO), n=192 | B | Colab T4 runs 2026-07-16/17; commands in the paper's Reproducibility block; an independent GRPO rerun reproduced 154/192 exactly |
| SFT replications | 77.1% (n=192), 68.8% (n=64), 85.9% (v1 protocol) | B | logged runs, same commands with fresh sessions |
| Train/eval overlap | 0 of 192 eval tasks in 3,010 training rows | A | keyless: `node --input-type=module -e "import {generateCases} from './generate.mjs'; import fs from 'fs'; const ev=new Set(generateCases(192,999).map(c=>c.task.spec)); const rows=fs.readFileSync('training/sft-3k.chat.jsonl','utf8').trim().split('\n').map(JSON.parse); let o=0; for (const r of rows){const u=r.messages.find(m=>m.role==='user').content; for (const s of ev) if (u.includes(s)){o++;break;}} console.log({overlap:o})"` |
| Generator v2 distinctness | 17,462 distinct tasks in a 20k draw (seed 999) | A | keyless: `node --input-type=module -e "import {generateCases} from './generate.mjs'; console.log(new Set(generateCases(20000,999).map(c=>c.task.spec)).size)"` |
| Reward audit, blatant tier (8 models) | 0% FP across all; agreement 86.7-98.3% | B | `node reward_anchor.mjs --provider=<p>` per provider (v2, n=60); the tool now prints the resolved model id |
| Reward audit, Grok variants | 95.0% agreement, 0% FP, 5.0% FN, each | A | `grok-4.5-blatant.txt`, `grok-4.20-blatant.txt`, `grok-4-fast-blatant.txt` (runs of 2026-07-17, `XAI_MODEL` per file name) |
| Near-miss collapse | qwen 46.7%, claude 33.3%, grok 25.5% FP | B | `node reward_anchor.mjs --provider=<p> --near-miss` |
| Grounding study | 96/96 blocked fail forward; 80/80 clean construct, 90% train; 88/88 width-corrupted crash; Spearman -0.581 | B | `training/README.md` records the run table (2026-07-04, torch 2.8, seeds 123+777); regenerate via `dump_grounding_set.mjs` + `training/grounding_at_scale.py` |
| Curated split | claude 7/12, mean score 76; 5 failed tasks, 7 failure reasons | A | `leaderboard-data.json` on the public leaderboard (transcribed from the harness run of 2026-07-08; failure categories are per-reason, a task can trigger more than one) |
| Reasoning traces | 327 verified / 451 graded = 72.5% (49 API errors excluded) | A | `arch-reasoning-claude.stats.json`; dataset public on Hugging Face |
| Harness self-tests | reference 100%, noop 0% | A | keyless, in CI: `node calibrate.mjs --policy=reference` / `--policy=noop` |
| Satisfiability + non-vacuity | 500/500 references pass; edit-in-place starts fail untouched | A | `npx vitest run` (`generate.test.ts`) |
| Wilson 95% intervals | all bracketed values | A | recompute: `python3 -c "import math; k,n=95,120; z=1.959964; p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); print(round(100*(c-h)/d), round(100*(c+h)/d))"` |

Notes:

- Tier B exists because early runs did not set `AMPLIFY_OUT` (amplify) or `tee`
  (reward audits, training evals). The numbers were transcribed at run time and
  are consistent under the cross-locks above, but the raw JSON for those
  specific runs was not retained. Every command above writes an artifact when
  re-run as shown.
- The v1 amplification artifact (`amp-full.json`) is committed precisely so the
  replication claim ("v1 82 -> 100 replicates on v2 within three points") has a
  replayable side.
