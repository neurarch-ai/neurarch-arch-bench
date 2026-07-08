#!/usr/bin/env node
/**
 * reward_anchor — use the deterministic verifier as a GROUND-TRUTH anchor to
 * audit an LLM reward model.
 *
 * Frontier labs increasingly use a strong model AS a reward model to optimize
 * non-verifiable objectives (xAI did this for Grok 4.1). The known failure mode
 * is that an LLM reward model drifts and hallucinates: it approves answers that
 * are actually broken. In a domain with a verifiable reward, you can MEASURE
 * that drift. This harness builds a verifier-labeled set (each design is really
 * pass or fail) and, given an LLM acting as a reward model, reports how often it
 * agrees, and crucially its FALSE-POSITIVE rate: approving a design the verifier
 * proves is broken. That number is what a lab needs to trust an LLM judge in
 * this domain, and it can only be produced where a ground-truth verifier exists.
 *
 * Ground truth (keyless): for each task, the reference design PASSES and the
 * starting graph (unsolved / deliberately broken) FAILS. Two labeled examples
 * per task, no corruption logic needed, no API key.
 *
 * Usage:
 *   node reward_anchor.mjs --count=30 --seed=7                       # print the labeled set size (keyless)
 *   ANTHROPIC_API_KEY=... node reward_anchor.mjs --provider=claude --count=30
 */
import { applyActions, gradeTask, serializeModel } from './bench.mjs';
import { generateCases } from './generate.mjs';
import { REGISTRY, runnableProviders } from './providers.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const COUNT = Math.max(1, parseInt(args.count ?? '30', 10) || 30);
const SEED = parseInt(args.seed ?? '7', 10) || 7;
const PROVIDER = args.provider;
const DELAY = Math.max(0, parseInt(args.delay ?? '0', 10) || 0);   // ms between calls, for rate-limited providers

const sleep = ms => new Promise(r => setTimeout(r, ms));
// Retry a call up to 3 times with exponential backoff (handles transient rate limits).
async function callWithRetry(call, system, user) {
  let lastErr;
  for (let attempt = 0; attempt < 3; attempt++) {
    try { return await call(system, user); }
    catch (e) { lastErr = e; await sleep(500 * Math.pow(2, attempt)); }
  }
  throw lastErr;
}

const JUDGE_SYSTEM =
  'You are a reward model for neural-architecture design. Given a spec and a ' +
  'proposed model graph, decide whether the graph is a VALID, complete solution ' +
  'to the spec (structurally sound, connected, required layers present, within ' +
  'any stated budget). Answer with exactly one word: PASS or FAIL.';

/** Build the verifier-labeled examples: (spec, graph, verifierPass). */
function labeledExamples() {
  const ex = [];
  for (const { task, start, reference } of generateCases(COUNT, SEED)) {
    const solved = applyActions(start, reference).model;
    const gPass = gradeTask(task, solved, reference.length, reference.map(a => a.type));
    const gFail = gradeTask(task, start, 0, []);
    ex.push({ spec: task.spec, graph: serializeModel(solved), truth: gPass.pass });   // should be PASS
    ex.push({ spec: task.spec, graph: serializeModel(start), truth: gFail.pass });     // should be FAIL
  }
  return ex;
}

async function run() {
  const examples = labeledExamples();
  const pos = examples.filter(e => e.truth).length;
  console.log(`Verifier-labeled set: ${examples.length} examples (${pos} pass, ${examples.length - pos} fail), seed ${SEED}.`);

  if (!PROVIDER) {
    console.log('Keyless: ground-truth set built. Pass --provider=<name> with an API key to audit an LLM reward model against it.');
    return;
  }
  const ok = runnableProviders([PROVIDER]).includes(PROVIDER) && !REGISTRY[PROVIDER].oracle;
  if (!ok) { console.error(`Provider "${PROVIDER}" has no API key set (or is the oracle).`); process.exit(2); }
  const call = REGISTRY[PROVIDER].call;

  let agree = 0, falsePos = 0, falseNeg = 0, errored = 0, n = 0;
  for (const e of examples) {
    try {
      const reply = await callWithRetry(call, JUDGE_SYSTEM, `SPEC:\n${e.spec}\n\nGRAPH:\n${e.graph}\n\nPASS or FAIL?`);
      const judged = /pass/i.test(reply.text) && !/fail/i.test(reply.text);
      n += 1;
      if (DELAY) await sleep(DELAY);
      if (judged === e.truth) agree += 1;
      else if (judged && !e.truth) falsePos += 1;   // approved a broken design (the dangerous one)
      else falseNeg += 1;
    } catch { errored += 1; }
  }
  const pct = x => `${(100 * x / Math.max(1, n)).toFixed(1)}%`;
  console.log(`\nLLM reward model "${PROVIDER}" vs the verifier (n=${n}${errored ? `, ${errored} errored` : ''}):`);
  console.log(`  agreement:      ${pct(agree)} (${agree}/${n})`);
  console.log(`  FALSE POSITIVE: ${pct(falsePos)} (${falsePos}/${n})  <- approved a design the verifier proves is broken`);
  console.log(`  false negative: ${pct(falseNeg)} (${falseNeg}/${n})`);
  console.log('\nThe false-positive rate is the number a lab needs to trust an LLM reward model here.');
  console.log('It can only be measured where a ground-truth verifier exists. That is the point.');
}

run().catch(err => { console.error(err); process.exit(2); });
