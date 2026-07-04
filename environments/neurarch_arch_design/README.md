# neurarch-arch-design

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment where
a model designs neural-network architectures as structured graph edits and a
deterministic verifier grades the result. No human judge, no LLM judge, no GPU
in the reward loop: grading is a sub-millisecond pure function.

- **Task**: a natural-language spec plus a serialized starting graph. Ten
  families, from design-from-spec (CNN, transformer, GQA encoder, two-tower
  retrieval) to edit-in-place repair, where `replace_model` is forbidden so a
  rebuild can't masquerade as a surgical fix.
- **Reward**: structural blockers (attention divisibility, connectivity),
  parameter budgets and bands, required layer families, plus a dense 0..100
  health score. Shaped so the learning order is: valid JSON, then valid
  graphs, then constraint satisfaction.
- **Splits**: procedurally generated from a seed. Train and eval are just
  different seeds; a held-out split never appears on the public web.

## Quickstart

The reward comes from the zero-dependency HTTP server in the parent repo
(plain node >= 18, no build step):

```bash
git clone https://github.com/neurarch-ai/neurarch-arch-bench
cd neurarch-arch-bench && node env-server.mjs   # serves :8737
```

Then:

```bash
prime env install neurarch-arch-design
vf-eval neurarch-arch-design -m gpt-4.1-mini -n 20
```

Or in code:

```python
import verifiers as vf
env = vf.load_environment("neurarch-arch-design", count=256, seed=123)
```

Point at a remote server with `NEURARCH_ENV_URL` or `env_url=...`.

## Args

| Arg | Default | Meaning |
| --- | --- | --- |
| `env_url` | `http://localhost:8737` | reward server address |
| `count` | 256 | tasks in the split |
| `seed` | 123 | split seed; change it for a held-out split |

The `task_pass` metric (weight 0) reports the raw pass rate next to the shaped
reward.

## Honest scope

The verifier checks structural validity, budgets, connectivity, and required
components; a grounding study in the parent repo (`training/grounding.py`)
shows verifier blockers predict PyTorch construction/forward failures. It does
not claim the resulting network trains to a good final metric.
