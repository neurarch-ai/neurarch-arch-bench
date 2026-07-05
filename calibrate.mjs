#!/usr/bin/env node
/**
 * calibrate — measure per-family difficulty against the band labs actually
 * buy: an environment needs hard families near the ~2-3% pass floor (one
 * success per 64-128 rollouts) for RL signal, and must not be saturated.
 *
 * Two keyless self-test policies bracket the harness itself:
 *   --policy=reference   replays each case's known-good solution: must be 100%
 *   --policy=noop        submits an empty plan: must be ~0%
 * If either self-test is off, the harness (not the model) is broken.
 *
 *   node calibrate.mjs --policy=reference
 *   node calibrate.mjs --policy=noop
 *   XAI_API_KEY=... node calibrate.mjs --provider=grok --per-family=16 --seed=11
 *   CALIBRATE_OUT=calib.json node calibrate.mjs ...
 */
import fs from 'node:fs';
import path from 'node:path';
import { applyActions, gradeTask, serializeModel } from './bench.mjs';
import { generateCases } from './generate.mjs';
import { SYSTEM_PROMPT, REGISTRY, parseActions, runnableProviders } from './providers.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const PROVIDER = args.provider ?? null;
const POLICY = args.policy ?? (PROVIDER ? 'model' : 'reference');
const PER_FAMILY = Math.max(2, parseInt(args['per-family'] ?? '16', 10) || 16);
const SEED = parseInt(args.seed ?? '11', 10) || 11;
const OUT = process.env.CALIBRATE_OUT;

const FAMILY_COUNT = 10; // families cycle i % 10 in generate.mjs
const familyOf = (id) => id.replace(/^gen-/, '').replace(/-\d+$/, '');

/** Wilson 95% interval: honest uncertainty at small n. */
function wilson(passes, n) {
  if (n === 0) return [0, 1];
  const z = 1.96, p = passes / n;
  const denom = 1 + z * z / n;
  const center = (p + z * z / (2 * n)) / denom;
  const half = (z / denom) * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

/** Band vs the procurement floor. Boundaries are judgment calls, stated here
 *  once: below 2% has too little signal to train on; 2-15% is the hard band
 *  labs ask for; 15-70% is eval-useful; above 70% is saturated for RL. */
function band(p) {
  if (p < 0.02) return 'TOO-HARD (<2%)';
  if (p <= 0.15) return 'TARGET (2-15%)';
  if (p <= 0.70) return 'EASY (eval-useful)';
  return 'SATURATED (>70%)';
}

async function attempt(task, start, reference) {
  if (POLICY === 'reference') {
    const applied = applyActions(start, reference);
    return gradeTask(task, applied.model, reference.length, reference.map(a => a?.type).filter(Boolean)).pass;
  }
  if (POLICY === 'noop') {
    return gradeTask(task, start, 0, []).pass;
  }
  const spec = REGISTRY[PROVIDER];
  const user = `SPEC:\n${task.spec}\n\nCURRENT MODEL:\n${serializeModel(start)}\n\nReturn the actions that fulfil the spec.`;
  try {
    const reply = await spec.call(SYSTEM_PROMPT, user);
    const actions = parseActions(reply.text);
    return gradeTask(task, applyActions(start, actions).model, actions.length, actions.map(a => a?.type).filter(Boolean)).pass;
  } catch {
    return false;
  }
}

async function run() {
  if (POLICY === 'model') {
    const ok = runnableProviders([PROVIDER]).filter(p => !REGISTRY[p]?.oracle);
    if (!ok.length) { console.error(`Provider ${PROVIDER} not runnable (missing key?).`); process.exit(2); }
  }
  const total = PER_FAMILY * FAMILY_COUNT;
  const cases = generateCases(total, SEED);
  const who = POLICY === 'model' ? REGISTRY[PROVIDER].modelId() : `policy:${POLICY}`;
  console.log(`Calibration: ${who}, ${PER_FAMILY} tasks/family x ${FAMILY_COUNT} families, seed ${SEED}\n`);

  const perFamily = new Map();
  for (const { task, start, reference } of cases) {
    const fam = familyOf(task.id);
    const pass = await attempt(task, start, reference);
    const s = perFamily.get(fam) ?? { n: 0, passes: 0 };
    s.n += 1; s.passes += pass ? 1 : 0;
    perFamily.set(fam, s);
  }

  const rows = [...perFamily.entries()].map(([family, s]) => {
    const p = s.passes / s.n;
    const [lo, hi] = wilson(s.passes, s.n);
    return { family, n: s.n, passes: s.passes, rate: p, ci: [lo, hi], band: band(p) };
  }).sort((a, b) => a.rate - b.rate);

  console.log('| Family | pass | rate | 95% CI | band |');
  console.log('| --- | --- | --- | --- | --- |');
  for (const r of rows) {
    console.log(`| ${r.family} | ${r.passes}/${r.n} | ${(r.rate * 100).toFixed(0)}% | ${(r.ci[0] * 100).toFixed(0)}-${(r.ci[1] * 100).toFixed(0)}% | ${r.band} |`);
  }
  const allPasses = rows.reduce((a, r) => a + r.passes, 0);
  const allN = rows.reduce((a, r) => a + r.n, 0);
  console.log(`\nOverall: ${allPasses}/${allN} = ${(allPasses / allN * 100).toFixed(1)}%`);

  // Self-test verdicts, so CI can assert the harness brackets.
  if (POLICY === 'reference' && allPasses !== allN) {
    console.error('SELF-TEST FAILED: reference policy must pass 100%.');
    process.exit(1);
  }
  if (POLICY === 'noop' && allPasses > 0) {
    console.error('SELF-TEST FAILED: noop policy must pass 0%.');
    process.exit(1);
  }
  if (POLICY !== 'model') console.log(`Self-test OK (${POLICY}).`);

  if (OUT) {
    fs.writeFileSync(path.resolve(OUT), JSON.stringify({
      who, seed: SEED, perFamily: PER_FAMILY, generatedAt: new Date().toISOString(), rows,
    }, null, 2));
    console.log(`Wrote ${OUT}`);
  }
}

run().catch(err => { console.error(err); process.exit(2); });
