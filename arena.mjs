#!/usr/bin/env node
/**
 * arena — head-to-head architecture-design duels with a deterministic judge.
 *
 * Two models, the same specs, single-shot each; the verifier decides every
 * round (pass beats fail; both pass -> higher health score; still tied ->
 * fewer tokens wins the efficiency tiebreak; otherwise a draw). No human
 * votes, no LLM judge: every result is reproducible from the seed.
 *
 *   XAI_API_KEY=... ANTHROPIC_API_KEY=... \
 *     node arena.mjs --a=grok --b=claude --generate=20 --seed=7
 *   node arena.mjs --a=grok --b=gemini --curated
 *   ARENA_OUT=duel.json node arena.mjs ...
 */
import fs from 'node:fs';
import path from 'node:path';
import { loadBenchmark, buildFixture, applyActions, gradeTask, serializeModel } from './bench.mjs';
import { generateCases } from './generate.mjs';
import { SYSTEM_PROMPT, REGISTRY, parseActions, runnableProviders } from './providers.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const A = args.a ?? 'grok';
const B = args.b ?? 'claude';
const GENERATE = args.curated ? 0 : Math.max(0, parseInt(args.generate ?? '20', 10) || 20);
const SEED = parseInt(args.seed ?? '7', 10) || 7;
const OUT = process.env.ARENA_OUT;

async function attempt(provider, task, start) {
  const spec = REGISTRY[provider];
  const user = `SPEC:\n${task.spec}\n\nCURRENT MODEL:\n${serializeModel(start)}\n\nReturn the actions that fulfil the spec.`;
  try {
    const reply = await spec.call(SYSTEM_PROMPT, user);
    const actions = parseActions(reply.text);
    const grade = gradeTask(task, applyActions(start, actions).model, actions.length, actions.map(a => a?.type).filter(Boolean));
    return { pass: grade.pass, score: grade.score, tokens: reply.tokens, failures: grade.failures };
  } catch (err) {
    return { pass: false, score: 0, tokens: 0, failures: [String(err.message ?? err)] };
  }
}

function judge(ra, rb) {
  if (ra.pass !== rb.pass) return ra.pass ? 'A' : 'B';
  if (ra.score !== rb.score) return ra.score > rb.score ? 'A' : 'B';
  if (ra.tokens > 0 && rb.tokens > 0 && ra.tokens !== rb.tokens) return ra.tokens < rb.tokens ? 'A' : 'B';
  return 'draw';
}

async function run() {
  const runnable = runnableProviders([A, B]).filter(p => !REGISTRY[p].oracle);
  if (runnable.length < 2) { console.error('Need API keys for both providers.'); process.exit(2); }
  const bench = loadBenchmark();
  const cases = GENERATE > 0
    ? generateCases(GENERATE, SEED).map(c => ({ task: c.task, start: c.start }))
    : bench.tasks.map(t => ({ task: t, start: buildFixture(bench, t.start) }));

  const nameA = REGISTRY[A].modelId(), nameB = REGISTRY[B].modelId();
  console.log(`Arena: ${nameA} vs ${nameB}, ${cases.length} rounds, deterministic judge\n`);

  let winsA = 0, winsB = 0, draws = 0;
  const rows = [];
  for (const { task, start } of cases) {
    const [ra, rb] = [await attempt(A, task, start), await attempt(B, task, start)];
    const verdict = judge(ra, rb);
    if (verdict === 'A') winsA += 1; else if (verdict === 'B') winsB += 1; else draws += 1;
    rows.push({ taskId: task.id, verdict, a: ra, b: rb });
    const line = (r) => `${r.pass ? 'PASS' : 'fail'} s=${r.score}${r.tokens ? ` t=${r.tokens}` : ''}`;
    console.log(`[${verdict === 'draw' ? '=' : verdict}] ${task.id.padEnd(22)} ${nameA}: ${line(ra)}  |  ${nameB}: ${line(rb)}`);
  }

  console.log(`\n== Result ==`);
  console.log(`${nameA} ${winsA} : ${winsB} ${nameB}  (${draws} draws)`);
  const tok = (p) => rows.reduce((s, r) => s + (p === 'a' ? r.a.tokens : r.b.tokens), 0);
  console.log(`tokens spent: ${nameA} ${tok('a')}, ${nameB} ${tok('b')}`);
  console.log('Judge: pass > score > fewer tokens; deterministic and reproducible from the seed.');

  if (OUT) {
    fs.writeFileSync(path.resolve(OUT), JSON.stringify({
      a: nameA, b: nameB,
      split: GENERATE > 0 ? { kind: 'generated', count: GENERATE, seed: SEED } : { kind: 'curated', count: cases.length },
      generatedAt: new Date().toISOString(),
      winsA, winsB, draws, rows,
    }, null, 2));
    console.log(`Wrote ${OUT}`);
  }
}

run().catch(err => { console.error(err); process.exit(2); });
