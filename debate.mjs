#!/usr/bin/env node
/**
 * debate — a multi-agent design panel arbitrated by the deterministic verifier.
 *
 * Frontier labs increasingly structure inference as multi-agent debate (Grok
 * 4.20 ships a four-agent debate; larger panels are on the roadmap). The open
 * question in every debate system is who judges the debate — typically another
 * LLM, with all the reliability problems the reward-model audit documents
 * (near-miss false-positive collapse). Here the judge is a pure function.
 *
 * N proposer agents independently attempt the same task (each with its own
 * verifier-feedback repair rounds, as in amplify). The arbiter then selects
 * among the finished proposals exactly the way arena judges duels:
 *
 *   pass beats fail  >  higher health score  >  fewer tokens  >  lower index
 *
 * — deterministic, reproducible from the seed, no LLM in the loop. Reported:
 *
 *   solo mean   = average per-agent pass rate (one agent working alone)
 *   panel       = arbitrated best-of-N pass rate (the debate's output)
 *   divergence  = share of tasks where agents disagreed, i.e. where the
 *                 arbiter had a real decision to make
 *
 * Usage (real API calls; a provider may be listed twice for independent
 * samples at API-default temperature):
 *   XAI_API_KEY=... node debate.mjs --panel=grok,grok,grok --generate=30 --seed=7 --turns=2
 *   XAI_API_KEY=... ANTHROPIC_API_KEY=... DEEPSEEK_API_KEY=... \
 *     node debate.mjs --panel=grok,claude,deepseek --curated
 *   DEBATE_OUT=debate.json node debate.mjs ...
 *
 * Keyless bracket (run before spending):
 *   node debate.mjs --self-check
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
const PANEL = (args.panel ?? 'grok,grok,grok').split(',').map(s => s.trim()).filter(Boolean);
const TURNS = Math.max(1, parseInt(args.turns ?? '2', 10) || 2);
const GENERATE = args.curated ? 0 : Math.max(0, parseInt(args.generate ?? '30', 10) || 30);
const SEED = parseInt(args.seed ?? '7', 10) || 7;
const OUT = process.env.DEBATE_OUT;

/** Deterministic arbitration over finished proposals: pass > score > fewer
 *  tokens > lower panel index. Returns the winning index. Exported for tests. */
export function arbitrate(proposals) {
  let win = 0;
  for (let i = 1; i < proposals.length; i++) {
    const a = proposals[win], b = proposals[i];
    if (b.pass !== a.pass) { if (b.pass) win = i; continue; }
    if (b.score !== a.score) { if (b.score > a.score) win = i; continue; }
    if (b.tokens < a.tokens) win = i;
  }
  return win;
}

/** One agent's independent episode: single-shot, then up to turns-1 repair
 *  rounds with the verifier's failures fed back (same loop as amplify). */
async function episode(call, task, start) {
  let model = start;
  const usedTypes = [];
  let grade = null;
  let tokens = 0;
  let passAtTurn1 = false;
  let firstGrade = null;
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
    usedTypes.push(...actions.map(a => a?.type).filter(Boolean));
    grade = gradeTask(task, model, actions.length, usedTypes);
    if (turn === 0) { passAtTurn1 = grade.pass; firstGrade = grade; }
    if (grade.pass) break;
  }
  return {
    pass: grade.pass, score: grade.score, tokens, passAtTurn1,
    turn1Score: firstGrade.score, failures: grade.failures,
  };
}

async function run() {
  const named = runnableProviders([...new Set(PANEL)]).filter(p => !REGISTRY[p].oracle);
  const panel = PANEL.filter(p => named.includes(p));
  if (panel.length < 2) {
    console.error('Need >= 2 panel seats with API keys, e.g. --panel=grok,grok,grok (a provider may repeat).');
    process.exit(2);
  }
  const bench = loadBenchmark();
  const cases = GENERATE > 0
    ? generateCases(GENERATE, SEED).map(c => ({ task: c.task, start: c.start }))
    : bench.tasks.map(t => ({ task: t, start: buildFixture(bench, t.start) }));
  const split = GENERATE > 0 ? `generated (N=${GENERATE}, seed=${SEED})` : `curated (N=${cases.length})`;
  console.log(`Debate panel [${panel.join(', ')}] on ${split}, up to ${TURNS} turns per agent\n`);

  const rows = [];
  let panelPass = 0, panelPassTurn1 = 0, soloPass = 0, diverged = 0, errors = 0;
  const winnerBySeat = panel.map(() => 0);
  for (const { task, start } of cases) {
    const proposals = [];
    for (const seat of panel) {
      try {
        proposals.push(await episode(REGISTRY[seat].call, task, start));
      } catch (err) {
        proposals.push({ pass: false, score: 0, tokens: 0, passAtTurn1: false, turn1Score: 0, failures: [String(err.message ?? err)], error: true });
        errors += 1;
      }
    }
    const win = arbitrate(proposals);
    const winT1 = arbitrate(proposals.map(p => ({ pass: p.passAtTurn1, score: p.turn1Score, tokens: p.tokens })));
    const passes = proposals.filter(p => p.pass).length;
    if (proposals[win].pass) { panelPass += 1; winnerBySeat[win] += 1; }
    if (proposals[winT1].pass !== undefined && proposals.map(p => p.passAtTurn1)[winT1]) panelPassTurn1 += 1;
    soloPass += passes / panel.length;
    if (passes > 0 && passes < panel.length) diverged += 1;
    rows.push({ taskId: task.id, winner: win, proposals });
    const verdict = proposals[win].pass ? 'PASS' : 'fail';
    console.log(`[${verdict}] ${task.id.padEnd(22)} agents ${proposals.map(p => (p.pass ? 'P' : 'f')).join('')} -> seat ${win} (${panel[win]})${proposals[win].pass ? '' : ` (${proposals[win].failures.map(categorizeFailure).join(',')})`}`);
  }

  const n = cases.length;
  console.log('\n== Verifier-arbitrated debate ==');
  console.log('| Panel | solo mean | panel (single-shot) | panel (with repair) |');
  console.log('| --- | --- | --- | --- |');
  console.log(`| ${panel.join('+')} | ${((soloPass / n) * 100).toFixed(0)}% | ${((panelPassTurn1 / n) * 100).toFixed(0)}% | ${((panelPass / n) * 100).toFixed(0)}% |`);
  console.log(`\nAgents diverged on ${diverged}/${n} tasks (where arbitration did real work).`);
  console.log(`Winning seat distribution: ${panel.map((p, i) => `${p}#${i}=${winnerBySeat[i]}`).join(', ')}${errors ? `; ${errors} seat-errors counted as failing proposals` : ''}.`);
  console.log('Judge: pass > score > fewer tokens; deterministic and reproducible from the seed.');

  if (OUT) {
    fs.writeFileSync(path.resolve(OUT), JSON.stringify({
      panel, turns: TURNS,
      split: GENERATE > 0 ? { kind: 'generated', count: GENERATE, seed: SEED } : { kind: 'curated', count: cases.length },
      generatedAt: new Date().toISOString(),
      summary: { n, soloMean: soloPass / n, panelSingleShot: panelPassTurn1 / n, panelWithRepair: panelPass / n, diverged, errors, winnerBySeat },
      rows,
    }, null, 2));
    console.log(`\nWrote ${OUT}`);
  }
}

/** Keyless bracket: a panel with one reference oracle among noops must pass
 *  every curated task (the arbiter must find the passing proposal); an
 *  all-noop panel must pass none. Catches a broken arbiter or feedback loop
 *  before any paid run. */
async function selfCheck() {
  const bench = loadBenchmark();
  const cases = bench.tasks.map(t => ({ task: t, start: buildFixture(bench, t.start) }));
  const solutions = (await import('./bench.mjs')).loadSolutions();
  const refCall = (task) => async () => ({ text: JSON.stringify({ actions: solutions[task.id] }), tokens: 0 });
  const noopCall = () => async () => ({ text: '{"actions":[]}', tokens: 0 });

  let ok = true;
  for (const [name, mkPanel, expected] of [
    ['ref-among-noops', (t) => [noopCall(t), refCall(t), noopCall(t)], cases.length],
    ['all-noop', (t) => [noopCall(t), noopCall(t), noopCall(t)], 0],
  ]) {
    let passed = 0;
    for (const { task, start } of cases) {
      const proposals = [];
      for (const call of mkPanel(task)) proposals.push(await episode(call, task, start));
      if (proposals[arbitrate(proposals)].pass) passed += 1;
    }
    const good = passed === expected;
    ok = ok && good;
    console.log(`[self-check] ${name.padEnd(15)} arbitrated ${passed}/${cases.length}  ${good ? 'OK' : `FAIL (expected ${expected})`}`);
  }
  console.log(ok
    ? '\nDebate pipeline OK: the arbiter finds the one passing proposal and rejects all-noop panels. Safe to spend on a real run.'
    : '\nDebate pipeline BROKEN: do not spend on a real run until fixed.');
  process.exit(ok ? 0 : 1);
}

const invokedDirectly = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname);
if (invokedDirectly) {
  (args['self-check'] ? selfCheck() : run()).catch(err => { console.error(err); process.exit(2); });
}
