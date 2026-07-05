#!/usr/bin/env node
/**
 * neurarch-arch-bench — leaderboard runner.
 *
 * Runs frontier models against the design-from-spec benchmark: each model is
 * shown a spec + a starting graph and asked to emit edit actions; we apply them
 * and grade the resulting graph. Prints a leaderboard and optionally writes JSON.
 *
 * Usage (real API calls — costs tokens):
 *   XAI_API_KEY=xai-...       node leaderboard.mjs --providers=grok
 *   ANTHROPIC_API_KEY=sk-...  node leaderboard.mjs --providers=claude
 *   GEMINI_API_KEY=AIza...    node leaderboard.mjs --providers=gemini
 *   GROQ_API_KEY=gsk-...      node leaderboard.mjs --providers=groq
 *   OPENAI_API_KEY=sk-...     node leaderboard.mjs --providers=openai
 *   node leaderboard.mjs --providers=grok,claude,gemini --only=cnn-cifar,text-encoder
 *   LEADERBOARD_OUT=board.json node leaderboard.mjs --providers=grok
 *
 * Procedural split (contamination-resistant, any size, seeded):
 *   node leaderboard.mjs --providers=grok --generate=50 --seed=7
 *
 * Pick the exact model with <PROVIDER>_MODEL, e.g. XAI_MODEL=grok-3.
 */
import fs from 'node:fs';
import path from 'node:path';
import {
  loadBenchmark, loadSolutions, buildFixture, applyActions, gradeTask, serializeModel, categorizeFailure,
} from './bench.mjs';
import { generateCases } from './generate.mjs';
import { SYSTEM_PROMPT, REGISTRY, parseActions, runnableProviders } from './providers.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const PROVIDERS = (args.providers ?? 'grok').split(',').map(s => s.trim());
const ONLY = args.only ? new Set(args.only.split(',')) : null;
const FORMAT = args.format ?? 'text'; // 'text' | 'md'
const OUT = process.env.LEADERBOARD_OUT;
// --generate=N grades a procedurally-generated split of N tasks (seeded by
// --seed) instead of the curated set. The oracle provider replays each
// generated case's own reference solution.
const GENERATE = Math.max(0, parseInt(args.generate ?? '0', 10) || 0);
const SEED = parseInt(args.seed ?? '1', 10) || 1;
// --tasks=proposed.json runs a task file minted by propose.mjs (LLM-proposed,
// satisfiability-proven) instead of the curated or generated splits.
const TASKS_FILE = args.tasks && args.tasks !== 'true' ? args.tasks : null;
// --traces=file.jsonl appends one JSON line per attempt: the full
// (spec, observation, raw reply, parsed actions, verdict) tuple. This is the
// benchmark's data exhaust: SFT-ready on passes, DPO/repair-ready on failures.
const TRACES = args.traces && args.traces !== 'true' ? args.traces : null;

const SOLUTIONS = loadSolutions();

function trace(record) {
  if (!TRACES) return;
  fs.appendFileSync(path.resolve(TRACES), JSON.stringify(record) + '\n');
}

async function run() {
  const bench = loadBenchmark();
  // A case pairs a task with its start graph and an oracle solution, unifying
  // the curated set (fixtures + solutions.json) and the generated split
  // (embedded start + reference).
  const cases = TASKS_FILE
    ? JSON.parse(fs.readFileSync(path.resolve(TASKS_FILE), 'utf8')).tasks
        .map(t => ({ task: t.task, start: t.start, solution: t.reference }))
    : GENERATE > 0
      ? generateCases(GENERATE, SEED).map(c => ({ task: c.task, start: c.start, solution: c.reference }))
      : bench.tasks
          .filter(t => !ONLY || ONLY.has(t.id))
          .map(t => ({ task: t, start: buildFixture(bench, t.start), solution: SOLUTIONS[t.id] ?? [] }));
  const providers = runnableProviders(PROVIDERS);
  if (!providers.length) { console.error('no runnable providers (set the API key env vars, or use --providers=reference)'); process.exit(2); }

  const splitNote = TASKS_FILE ? ` [task file: ${TASKS_FILE}]` : GENERATE > 0 ? ` [generated split, seed=${SEED}]` : '';
  console.log(`neurarch-arch-bench v${bench.version}: ${cases.length} tasks${splitNote} x ${providers.length} provider(s) [${providers.join(', ')}]\n`);
  const rows = [];

  for (const provider of providers) {
    const spec = REGISTRY[provider];
    for (const { task, start, solution } of cases) {
      const observation = serializeModel(start);
      const user = `SPEC:\n${task.spec}\n\nCURRENT MODEL:\n${observation}\n\nReturn the actions that fulfil the spec.`;
      const t0 = Date.now();
      let raw = null;
      let tokens = 0;
      try {
        let actions;
        if (spec.oracle) { actions = solution; }
        else {
          const reply = await spec.call(SYSTEM_PROMPT, user);
          raw = reply.text; tokens = reply.tokens;
          actions = parseActions(raw);
        }
        const { model } = applyActions(start, actions);
        const g = gradeTask(task, model, actions.length, actions.map(a => a?.type).filter(Boolean));
        const status = g.pass ? 'PASS' : 'FAIL';
        const failureCategories = g.failures.map(categorizeFailure);
        rows.push({ provider, taskId: task.id, status, score: g.score, ...g, failureCategories, tokens, ms: Date.now() - t0 });
        trace({ provider, taskId: task.id, spec: task.spec, observation, raw, actions, pass: g.pass, score: g.score, params: g.params, failures: g.failures, failureCategories, ms: Date.now() - t0 });
        console.log(`[${status}] ${provider.padEnd(8)} ${task.id.padEnd(22)} score=${String(g.score).padStart(3)} params=${g.params} (${Date.now() - t0}ms)`);
        for (const f of g.failures) console.log(`         - ${f}`);
      } catch (err) {
        const failureCategories = [categorizeFailure(err.message ?? err)];
        rows.push({ provider, taskId: task.id, status: 'ERROR', score: 0, error: String(err), failureCategories, tokens, ms: Date.now() - t0 });
        trace({ provider, taskId: task.id, spec: task.spec, observation, raw, actions: null, pass: false, score: 0, error: String(err), failureCategories, ms: Date.now() - t0 });
        console.log(`[ERR ] ${provider.padEnd(8)} ${task.id.padEnd(22)} ${err.message}`);
      }
    }
  }

  const board = providers.map(p => {
    const pr = rows.filter(r => r.provider === p);
    const passed = pr.filter(r => r.status === 'PASS').length;
    const avgScore = pr.length ? Math.round(pr.reduce((a, r) => a + r.score, 0) / pr.length) : 0;
    // Aggregate the failure taxonomy: which categories dominate this model's
    // failures. This is the legible comparison ("60% divisibility").
    const failureCategories = {};
    for (const r of pr) {
      for (const c of r.failureCategories ?? []) failureCategories[c] = (failureCategories[c] ?? 0) + 1;
    }
    // Cost of intelligence: provider-reported tokens per SOLVED task (burning
    // more tokens to fail is worse, so the denominator is passes).
    const totalTokens = pr.reduce((a, r) => a + (r.tokens ?? 0), 0);
    const tokensPerSolve = passed > 0 ? Math.round(totalTokens / passed) : null;
    return { provider: p, passed, total: pr.length, avgScore, tokensPerSolve, failureCategories };
  }).sort((a, b) => b.passed - a.passed || b.avgScore - a.avgScore);

  if (FORMAT === 'md') {
    console.log('\n| Model | Passed | Avg score |');
    console.log('| --- | --- | --- |');
    for (const b of board) console.log(`| ${b.provider} | ${b.passed}/${b.total} | ${b.avgScore} |`);
  } else {
    console.log('\n-- Leaderboard --');
    for (const b of board) {
      console.log(`  ${b.provider.padEnd(10)} ${b.passed}/${b.total} passed  avg score ${b.avgScore}${b.tokensPerSolve ? `  ${b.tokensPerSolve} tok/solve` : ''}`);
      const cats = Object.entries(b.failureCategories).sort((x, y) => y[1] - x[1]);
      if (cats.length) console.log(`             failures: ${cats.map(([c, n]) => `${c} x${n}`).join(', ')}`);
    }
  }

  if (OUT) {
    const meta = {
      benchmark: bench.version,
      split: TASKS_FILE ? { kind: 'file', path: TASKS_FILE, count: cases.length } : GENERATE > 0 ? { kind: 'generated', count: GENERATE, seed: SEED } : { kind: 'curated', count: cases.length },
      models: Object.fromEntries(providers.map(p => [p, REGISTRY[p].modelId()])),
      generatedAt: new Date().toISOString(),
    };
    fs.writeFileSync(path.resolve(OUT), JSON.stringify({ ...meta, board, rows }, null, 2));
    console.log(`\nWrote ${OUT}`);
  }
  if (TRACES) console.log(`Appended ${rows.length} trace line(s) to ${TRACES}`);
}

run().catch(err => { console.error(err); process.exit(2); });
