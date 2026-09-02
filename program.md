# program.md

Instructions for one agent, running unattended.

You are optimizing a **policy**: the prompt that turns a design spec into graph
edits. Your score is the fraction of held-out architecture-design tasks that
policy passes, judged by a deterministic verifier that runs in milliseconds on
a CPU. No GPU, no human, no LLM judge. Point a coding agent at this file and
leave; it can run a few hundred experiments while you sleep.

The loop is a ratchet: change the policy, measure, keep it if the number went
up, revert it if it did not, append one row to `results.tsv`, repeat.

---

## Setup

1. Pick a run tag (e.g. `mar5`) and create the branch `autoresearch/<tag>`.
2. Read `README.md` (what the benchmark is) and `ANTI_GAMING.md` (what counts
   as cheating). Skim `bench.mjs`: it is the grader, and it is short.
3. Confirm the harness runs at all, with no API key:
   ```bash
   node leaderboard.mjs --providers=reference    # must print 12/12 passed
   ```
4. Export the key for the model you are optimizing the policy for, e.g.
   `XAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY`.
5. Create `results.tsv` with this header:
   ```
   commit	pass_rate	avg_score	n	status	note
   ```

## What you may change

**Editable: `policy.md`, and nothing else.**

It is the system prompt the design model receives. Everything is fair game
inside it: the instruction wording, the ordering, worked examples, a checklist,
a chain-of-thought scaffold, output-format rules.

**Read-only: everything else.** In particular `bench.mjs`, `generate.mjs`,
`solutions.json`, and `tasks.json` are the grader and the exam. Editing them is
not a result, it is a broken thermometer. Do not add dependencies (this repo
has zero on purpose) and do not edit `providers.mjs`.

## The metric

```bash
NEURARCH_POLICY_FILE=policy.md node leaderboard.mjs \
  --providers=<your provider> --generate=40 --seed=<TRAIN_SEED>
```

Read `pass_rate` (tasks passed / tasks graded); `avg score` breaks ties.

Fix `TRAIN_SEED` once at the start of the run and never change it mid-search,
or you are comparing against a different exam every step. `--generate` mints a
fresh deterministic split from the seed, so the tasks you tune against are not
on the public web and cannot have been memorized.

**Every 10 kept improvements, run the held-out seed once:**

```bash
NEURARCH_POLICY_FILE=policy.md node leaderboard.mjs \
  --providers=<your provider> --generate=40 --seed=<HELDOUT_SEED>
```

Log it with `status=heldout`. If the train number keeps climbing while the
held-out number does not, you are fitting the seed, not improving the policy.
Say so in the note and change approach.

## The loop

Repeat until interrupted:

1. Pick ONE hypothesis about why the policy fails. Ground it in the failure
   rows the last run printed, not in a general theory of prompting. The
   verifier tells you exactly what broke: `embedDim 100 not divisible by
   numHeads 7`, `inFeatures=25088 but upstream last dim is 5408`, budget
   overruns. Those are the specific mistakes worth writing a rule about.
2. Edit `policy.md`. One change per step, so an improvement is attributable.
3. `git add -A && git commit -m "<one line>"`.
4. Run the metric command above (`> run.log 2>&1` if you want the raw output).
5. Append one row to `results.tsv`: commit hash, pass_rate, avg_score, n,
   `keep` / `discard` / `error`, and a short note.
6. If `pass_rate` improved, keep the commit. If it is equal or worse,
   `git reset --hard HEAD~1`. A tie reverts: keeping ties lets the policy drift
   sideways on noise forever.
7. Go to 1.

Do NOT ask "should I keep going?". Do not summarize progress and wait. The
loop runs until the human interrupts you, period. If a run errors (rate limit,
network), log the row with `status=error` and continue; if three runs error in
a row, stop and say why.

## Rules

- **One change per step.** Two changes and a win teaches you nothing.
- **A tie is a revert.**
- **Never edit the grader or the tasks.** If you believe a task is
  unsatisfiable, log it in the note and route around it; do not fix the exam.
- **Prompt only.** Do not add a retry loop, a repair pass, a second model call,
  or any scaffolding outside `policy.md`. A better harness is a different
  experiment; this one measures the policy.
- **No task-specific answers.** Encoding a solution to a named task (or a
  parameter value that only works for one generated family) is memorization,
  not policy improvement, and the held-out seed will catch it. Write rules that
  generalize.
- **Simplicity is the tiebreaker.** If two policies score the same, keep the
  shorter one.
- **Keep every row.** A failed idea is a result. `results.tsv` is the artifact
  this run produces, more than the final `policy.md` is.

## What this measures, and what it does not

It measures whether a policy reliably turns a spec into a **valid, connected,
budget-respecting architecture**: legality and structure, machine-checked.

It does not measure whether the designed network **trains well**. Those are
different questions, and the gap is measured rather than assumed: on a
grounded split where every statically-passing design was actually trained on a
GPU, the rank correlation between static score and trained outcome was 0.17
over 24 designs. The verifier is necessary (nothing it blocks trains) and not
sufficient (it cannot rank healthy designs). Do not report a pass rate as
accuracy, and do not conclude from a high score that the architectures are
good, only that they are correct.

That limit is also why this loop is worth running unattended: because the
metric is cheap and deterministic, you get hundreds of measured policy
experiments per night on a laptop, with no GPU bill and nothing to babysit.

## Shape credit

The loop shape (fixed budget, ratchet on one metric, one editable file,
`results.tsv`, do not ask permission) is taken from
[karpathy/autoresearch](https://github.com/karpathy/autoresearch), which
applies it to a 5-minute GPU training run scored by validation bits per byte.
The differences here are deliberate: the metric is a CPU verifier instead of a
training run, so an experiment costs seconds instead of five minutes; the
editable surface is a prompt instead of `train.py`; and the exam is minted
fresh from a seed instead of fixed, so a night of search cannot quietly
overfit a public split.
