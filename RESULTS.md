# Results

All numbers below are graded by rubric v2 (`RUBRIC_VERSION` in `bench.mjs`),
which propagates widths along edges. Rubric v1 did not, and scored these same
models 13 to 14 points higher; `node rubric_delta.mjs <artifact>` recomputes
that gap from any leaderboard artifact without an API key.

Reproduce a row: `node leaderboard.mjs --providers=<model> --format=md`.

## Reference (oracle)

The `reference` provider replays a known-good solution per task. It needs no
API key and establishes the ceiling (every task is solvable) and the floor
every real model is measured against.

```bash
node leaderboard.mjs --providers=reference --format=md
```

| Model | Passed | Avg score |
| --- | --- | --- |
| reference (oracle) | 12/12 | 75 |

## Curated split (12 hand-authored production architectures)

| Model | Passed | Avg score |
| --- | --- | --- |
| google/gemini-2.5-flash | 11/12 | 75 |
| grok-4 | 9/12 | 60 |
| openai/gpt-4o | 9/12 | 72 |
| claude-sonnet-4-6 | 8/12 | 81 |

Twelve tasks is a small split: an independent rerun of claude landed at 7/12,
so read these as plus or minus one task. Pass count and mean score disagree on
purpose. claude leads on score and trails on passes, because its failures are
near-misses (one missing layer type) while grok's carry hard shape blockers.

Artifacts: `lb-claude-grok-curated.json`, `lb-gpt4o-curated.json`,
`lb-gemini-curated.json`.

## Generated split (120 tasks, seed 7)

```bash
node leaderboard.mjs --providers=<model> --generate=120 --seed=7 --format=md
```

| Model | Passed | Avg score | v1 would have reported |
| --- | --- | --- | --- |
| grok-4 | 86/119 | 63 | 86.6% (+14.3) |
| claude-sonnet-4-6 | 79/120 | 72 | 79.2% (+13.3) |
| google/gemini-2.5-flash | 77/120 | 67 | see `rubric_delta.mjs` |
| openai/gpt-4o | 66/120 | 66 | see `rubric_delta.mjs` |

One grok row lost to a transport error and is excluded. The `openai/` and
`google/` rows were served through OpenRouter, which routes to third-party
upstreams, so treat them as slightly less reproducible than the direct-API
rows.

Artifacts: `lb-claude-grok-gen.json`, `lb-gpt4o-gen.json`, `lb-gemini-gen.json`.

## Verifier-in-the-loop lift (up to 3 repair rounds)

```bash
AMPLIFY_OUT=out.json node amplify.mjs --providers=<model> --generate=120 --seed=7 --turns=3
```

| Model | k=1 | k=2 | k=3 | lift |
| --- | --- | --- | --- | --- |
| claude-sonnet-4-6 | 66.7% | 90.8% | 92.5% | +25.8 |
| grok-4 | 78.3% | 90.0% | 95.8% | +17.5 |
| deepseek-chat (V3) | 48.3% | 90.0% | 93.3% | +45.0 |

Under rubric v1 a single repair round carried the entire lift. It no longer
does: the defects a third round fixes are shape defects v1 could not see.

Artifacts: `amp-claude-v3.json`, `amp-grok-v3.json`, `amp-deepseek-v3.json`.

## Difficulty calibration

```bash
node calibrate.mjs --provider=<model> --per-family=12 --seed=11
node calibrate.mjs --provider=<model> --per-family=12 --seed=11 --tier=frontier
```

Core tier, claude-sonnet-4-6: 63.3% overall. Three families at 0% (transformer
encoder, insert-norm, two-tower), one eval-useful (GQA 58%), six saturated.

Frontier tier: claude-sonnet-4-6 27.8% (all three families eval-useful), grok-4
97.2% (all three saturated). The tier is harder than core for one frontier
model and easier for the other, so quote it with the model attached.

Artifacts: `calib-claude-core-v2.json`, `calib-claude-frontier.json`,
`calib-grok-frontier.json`.
