# Results

Run the benchmark yourself: `node leaderboard.mjs --providers=<model> --format=md`.

## Reference (oracle)

The `reference` provider replays a known-good solution per task. It needs no API key and establishes the ceiling (every task is solvable) and the floor every real model is measured against. Reproduce with:

```bash
node leaderboard.mjs --providers=reference --format=md
```

| Model | Passed | Avg score |
| --- | --- | --- |
| reference (oracle) | 8/8 | 74 |

## Frontier models

Bring API keys and run them yourself; paste rows here. Example:

```bash
XAI_API_KEY=xai-... ANTHROPIC_API_KEY=sk-... GEMINI_API_KEY=AIza... \
  node leaderboard.mjs --providers=grok,claude,gemini --format=md
```

| Model | Passed | Avg score |
| --- | --- | --- |
| grok | _run it_ | _run it_ |
| claude | _run it_ | _run it_ |
| gemini | _run it_ | _run it_ |

Numbers are not checked into this repo by the maintainer to avoid stale or cherry-picked claims. The harness is deterministic given a model's outputs, so anyone can reproduce a row.
