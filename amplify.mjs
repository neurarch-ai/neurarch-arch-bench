#!/usr/bin/env node
/**
 * amplify — the verifier-in-the-loop lift study, across every provider.
 *
 * Question: how much better does a model get at architecture design when the
 * deterministic verifier's failure messages are fed back for repair rounds?
 * For each provider we run the SAME tasks twice-in-one-episode:
 *
 *   single-shot   = pass judged after turn 1 (no feedback)
 *   with verifier = pass judged after up to --turns rounds, where each failing
 *                   grade's failure list is fed back and the model revises the
 *                   (already mutated) graph
 *
 * The delta is the environment's value measured on that model. It is a
 * pro-model framing on purpose: the headline is "X + verifier > X alone",
 * for every X — Grok, Claude, Gemini, GPT, Llama alike.
 *
 * Usage (real API calls, costs tokens; providers without keys are skipped):
 *   XAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... \
 *     node amplify.mjs --providers=grok,claude,gemini --generate=30 --seed=7 --turns=3
 *   node amplify.mjs --providers=claude --curated --turns=3
 *   AMPLIFY_OUT=amplify.json node amplify.mjs ...   # full JSON record
 */
import fs from 'node:fs';
import path from 'node:path';
import {
  loadBenchmark, buildFixture, applyActions, gradeTask, serializeModel, categorizeFailure,
} from './bench.mjs';
import { generateCases } from './generate.mjs';
import { SYSTEM_PROMPT, REGISTRY, parseActions, runnableProviders } from './providers.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const PROVIDERS = (args.providers ?? 'grok,claude,gemini,openai,groq').split(',').map(s => s.trim());
const TURNS = Math.max(2, parseInt(args.turns ?? '3', 10) || 3);
const GENERATE = args.curated ? 0 : Math.max(0, parseInt(args.generate ?? '30', 10) || 30);
const SEED = parseInt(args.seed ?? '7', 10) || 7;
const OUT = process.env.AMPLIFY_OUT;

/** One episode: turn 1 is the single-shot measurement; later turns feed the
 *  verifier's failures back and accumulate edits on the mutated graph. */
async function episode(call, task, start) {
  let model = start;
  let actionCount = 0;
  const usedTypes = [];
  let grade = null;
  let passAtTurn1 = false;
  let turnsUsed = 0;
  let tokens = 0;

  for (let turn = 0; turn < TURNS; turn++) {
    const user = turn === 0
      ? `SPEC:\n${task.spec}\n\nCURRENT MODEL:\n${serializeModel(model)}\n\nReturn the actions that fulfil the spec.`
      : `SPEC:\n${task.spec}\n\nCURRENT MODEL:\n${serializeModel(model)}\n\n`
        + `Your previous edits did not pass. Remaining issues:\n${grade.failures.map(f => `- ${f}`).join('\n')}\n\n`
        + `Return only the additional or corrected actions that resolve these issues.`;
    const reply = await call(SYSTEM_PROMPT, user);
    tokens += reply.tokens;
    const actions = parseActions(reply.text);
    model = applyActions(model, actions).model;
    actionCount = actions.length; // per-turn: the repair loop's surgical budget applies to THIS correction, not the sum across turns (feedback must be able to recover action-constrained edit tasks)
    usedTypes.push(...actions.map(a => a?.type).filter(Boolean));
    grade = gradeTask(task, model, actionCount, usedTypes);
    turnsUsed = turn + 1;
    if (turn === 0) passAtTurn1 = grade.pass;
    if (grade.pass) break;
  }
  // Keep the graph behind a surviving failure. A blocker that ends an episode
  // is a claim about the model's output, and a reader is entitled to inspect
  // the graph rather than take the verdict on trust.
  return {
    passAtTurn1, passFinal: grade.pass, turnsUsed, tokens, finalFailures: grade.failures,
    ...(grade.pass ? {} : { finalGraph: { components: model.components, connections: model.connections } }),
  };
}

async function run() {
  const bench = loadBenchmark();
  const cases = GENERATE > 0
    ? generateCases(GENERATE, SEED).map(c => ({ task: c.task, start: c.start }))
    : bench.tasks.map(t => ({ task: t, start: buildFixture(bench, t.start) }));
  const providers = runnableProviders(PROVIDERS).filter(p => !REGISTRY[p].oracle);
  if (!providers.length) {
    console.error('No providers with API keys. Set e.g. XAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY and retry.');
    process.exit(2);
  }
  const split = GENERATE > 0 ? `generated (N=${GENERATE}, seed=${SEED})` : `curated (N=${cases.length})`;
  console.log(`Amplification study: ${split}, up to ${TURNS} turns, providers: ${providers.join(', ')}\n`);

  const results = [];
  const rows = [];
  for (const provider of providers) {
    const spec = REGISTRY[provider];
    let t1 = 0, tf = 0, errors = 0;
    const rescuedBy = {};
    for (const { task, start } of cases) {
      try {
        const r = await episode(spec.call, task, start);
        t1 += r.passAtTurn1 ? 1 : 0;
        tf += r.passFinal ? 1 : 0;
        if (!r.passAtTurn1 && r.passFinal) {
          // What kinds of failures did the feedback loop rescue? (Categorize
          // what was wrong after turn 1... we only kept final failures, so
          // approximate with turnsUsed.)
          rescuedBy[`turn${r.turnsUsed}`] = (rescuedBy[`turn${r.turnsUsed}`] ?? 0) + 1;
        }
        rows.push({ provider, taskId: task.id, ...r });
        const tag = r.passFinal ? (r.passAtTurn1 ? 'PASS ' : 'FIXED') : 'FAIL ';
        console.log(`[${tag}] ${provider.padEnd(8)} ${task.id.padEnd(22)} turns=${r.turnsUsed}${r.passFinal ? '' : ` (${r.finalFailures.map(categorizeFailure).join(',')})`}`);
      } catch (err) {
        errors += 1;
        rows.push({ provider, taskId: task.id, error: String(err) });
        console.log(`[ERR  ] ${provider.padEnd(8)} ${task.id.padEnd(22)} ${err.message}`);
      }
    }
    const n = cases.length;
    results.push({
      provider,
      model: spec.modelId(),
      n,
      singleShot: t1 / n,
      withVerifier: tf / n,
      liftPoints: ((tf - t1) / n) * 100,
      errors,
      rescuedBy,
    });
  }

  console.log('\n== Verifier-in-the-loop lift ==');
  console.log('| Model | single-shot | with verifier feedback | lift |');
  console.log('| --- | --- | --- | --- |');
  for (const r of results.sort((a, b) => b.withVerifier - a.withVerifier)) {
    console.log(`| ${r.model} | ${(r.singleShot * 100).toFixed(0)}% | ${(r.withVerifier * 100).toFixed(0)}% | +${r.liftPoints.toFixed(0)} pts |`);
  }
  console.log('\nSame tasks, same model, same prompt; the only variable is the');
  console.log('deterministic verifier feeding its failure messages back.');

  if (OUT) {
    fs.writeFileSync(path.resolve(OUT), JSON.stringify({
      split: GENERATE > 0 ? { kind: 'generated', count: GENERATE, seed: SEED } : { kind: 'curated', count: cases.length },
      turns: TURNS,
      generatedAt: new Date().toISOString(),
      results, rows,
    }, null, 2));
    console.log(`\nWrote ${OUT}`);
  }
}

// Keyless bracket: run the two-arm pipeline with the reference oracle (must
// pass both arms) and a noop policy (must fail both), so a broken grader or
// feedback loop is caught before spending on a real provider run.
async function selfCheck() {
  const cases = generateCases(12, SEED); // one wave, all families
  const policies = [
    ['reference', (c) => async () => ({ text: JSON.stringify({ actions: c.reference }), tokens: 0 })],
    ['noop',      ()  => async () => ({ text: '{"actions":[]}', tokens: 0 })],
  ];
  let ok = true;
  for (const [name, mk] of policies) {
    let t1 = 0, tf = 0;
    for (const c of cases) {
      const r = await episode(mk(c), c.task, c.start);
      if (r.passAtTurn1) t1++;
      if (r.passFinal) tf++;
    }
    const exp = name === 'reference' ? cases.length : 0;
    const pass = t1 === exp && tf === exp;
    ok = ok && pass;
    console.log(`[self-check] ${name.padEnd(9)} single-shot ${t1}/${cases.length}, with-verifier ${tf}/${cases.length}  ${pass ? 'OK' : `FAIL (expected ${exp})`}`);
  }
  console.log(ok
    ? '\nAmplify pipeline OK: reference 100% both arms, noop 0% both arms. Safe to spend on a real run.'
    : '\nAmplify pipeline BROKEN: do not spend on a real run until fixed.');
  process.exit(ok ? 0 : 1);
}

(args['self-check'] ? selfCheck() : run()).catch(err => { console.error(err); process.exit(2); });
