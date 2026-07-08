#!/usr/bin/env node
/**
 * reasoning_traces — a data foundry for VERIFIED architecture-design reasoning.
 *
 * The 2026 commodity for RLVR/reasoning-model training is verified reasoning
 * data: (problem -> reasoning -> answer) triples where the answer is checked,
 * not human-rated. This produces exactly that for neural architecture design:
 * (spec -> reasoning -> design), and every design's final graph is re-graded by
 * the same deterministic verifier, so a trace is in the set ONLY if its design
 * actually passes. No LLM judge anywhere.
 *
 * Two modes:
 *   --provider=<name>   REJECTION-SAMPLED, model-generated reasoning (premium).
 *                       The model is asked to reason step by step and then emit
 *                       actions; we apply + grade; only PASSING traces are kept.
 *                       Needs an API key (same registry as the leaderboard).
 *   (default, keyless)  REFERENCE-DERIVED reasoning: the reasoning is composed
 *                       programmatically from the task's own constraints and the
 *                       reference design (verified by construction). A structural
 *                       scaffold; no key needed, useful for format + smoke tests.
 *
 * Usage:
 *   node reasoning_traces.mjs --count=200 --seed=20260708 --out=arch-reasoning
 *   ANTHROPIC_API_KEY=... node reasoning_traces.mjs --provider=claude --count=200 --tries=2 --out=arch-reasoning-claude
 *
 * Output: <out>.jsonl (one verified trace per line) + <out>.stats.json.
 */
import fs from 'node:fs';
import path from 'node:path';
import { applyActions, gradeTask, serializeModel } from '../bench.mjs';
import { generateCases } from '../generate.mjs';
import { SYSTEM_PROMPT, REGISTRY, parseActions, runnableProviders } from '../providers.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const COUNT = Math.max(1, parseInt(args.count ?? '200', 10) || 200);
const SEED = parseInt(args.seed ?? '20260708', 10) || 20260708;
const OUT = args.out ?? 'arch-reasoning';
const PROVIDER = args.provider;              // undefined => keyless reference-derived
const TRIES = Math.max(1, parseInt(args.tries ?? '2', 10) || 2); // rejection-sampling attempts
const DELAY = Math.max(0, parseInt(args.delay ?? '0', 10) || 0); // ms between calls, for rate-limited providers

const sleep = ms => new Promise(r => setTimeout(r, ms));
// Retry a call up to 3 times with exponential backoff (handles transient rate limits).
async function callWithRetry(call, system, user) {
  let lastErr;
  for (let attempt = 0; attempt < 3; attempt++) {
    try { return await call(system, user); }
    catch (e) { lastErr = e; await sleep(800 * Math.pow(2, attempt)); }
  }
  throw lastErr;
}

// A reasoning prompt that asks for explicit steps THEN the JSON actions. We keep
// the model's reasoning text and its actions; the verifier decides if it counts.
const REASON_SYSTEM = SYSTEM_PROMPT +
  '\nFirst think step by step about the constraints (required layers, shapes, ' +
  'divisibility, parameter/KV budgets) inside <reasoning>...</reasoning>, then ' +
  'output the single JSON object with the actions.';

// Isolate the balanced {...} block that contains "actions" from prose (the
// reasoning text can itself contain braces, which defeats a greedy match).
function isolateActionsJson(text) {
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== '{') continue;
    let depth = 0, inStr = false, esc = false;
    for (let j = i; j < text.length; j++) {
      const ch = text[j];
      if (esc) { esc = false; continue; }
      if (ch === '\\') { esc = true; continue; }
      if (ch === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (ch === '{') depth++;
      else if (ch === '}' && --depth === 0) {
        const cand = text.slice(i, j + 1);
        if (cand.includes('"actions"')) return cand;
        break;
      }
    }
  }
  return text;
}

function splitReasoning(text) {
  const m = text.match(/<reasoning>([\s\S]*?)<\/reasoning>/i);
  const reasoning = m ? m[1].trim() : '';
  const rest = m ? text.slice(m.index + m[0].length) : text;
  return { reasoning, rest };
}

/** Keyless: compose a faithful reasoning from the task's constraints + the
 *  reference structure. Verified by construction (the reference passes). */
function referenceReasoning(task, start, reference) {
  const c = task.constraints ?? {};
  const steps = [];
  steps.push(`Goal: ${task.spec}`);
  if (c.mustContainTypes?.length) steps.push(`The spec requires these layer types: ${c.mustContainTypes.join(', ')}. I include each.`);
  if (c.mustContainTypesAny?.length) steps.push(`At least one of ${c.mustContainTypesAny.join(', ')} is required (e.g. to bound the KV cache without deleting attention).`);
  if (typeof c.maxParams === 'number') steps.push(`Parameter budget: <= ${c.maxParams.toLocaleString()}. I size widths so the total stays under it.`);
  if (typeof c.minParams === 'number') steps.push(`Two-sided band: params must also be >= ${c.minParams.toLocaleString()}, so I cannot shrink to nothing.`);
  if (typeof c.maxKvBytesPerToken === 'number') steps.push(`KV-cache budget: <= ${c.maxKvBytesPerToken} bytes/token; I reduce KV heads (grouped-query attention) to fit while keeping embedDim and the query heads.`);
  if (typeof c.maxActions === 'number') steps.push(`At most ${c.maxActions} surgical edit(s); no full rebuild.`);
  steps.push('Attention layers need embedDim divisible by numHeads (and numHeads divisible by numKVHeads for GQA); every linear’s inFeatures must match the upstream width; the input must reach the output.');
  steps.push('Design that satisfies all of the above:');
  for (const a of reference) {
    if (a.type === 'replace_model') steps.push(`  build ${a.components?.length ?? 0} components wired as specified.`);
    else steps.push(`  ${a.type} on ${a.name ?? ''} ${JSON.stringify(a.params ?? {})}`);
  }
  return steps.join('\n');
}

async function run() {
  const provider = PROVIDER ? (runnableProviders([PROVIDER]).includes(PROVIDER) ? PROVIDER : null) : null;
  if (PROVIDER && !provider) { console.error(`Provider "${PROVIDER}" has no API key set.`); process.exit(2); }
  const call = provider && !REGISTRY[provider].oracle ? REGISTRY[provider].call : null;

  const outPath = path.resolve(`${OUT}.jsonl`);
  const stream = fs.createWriteStream(outPath);
  let kept = 0, attempted = 0, rejected = 0, errored = 0;

  for (const { task, start, reference } of generateCases(COUNT, SEED)) {
    attempted += 1;
    let reasoning, actions, source;

    if (call) {
      // Rejection sampling: try up to TRIES; keep the first verified one.
      // Separate API failures (errored) from genuine solve failures (rejected),
      // so a low yield is not silently blamed on task difficulty.
      let ok = false, apiError = false;
      for (let t = 0; t < TRIES && !ok; t++) {
        const user = `SPEC:\n${task.spec}\n\nCURRENT MODEL:\n${serializeModel(start)}\n\nReturn the actions that fulfil the spec.`;
        let reply;
        try { reply = await callWithRetry(call, REASON_SYSTEM, user); }
        catch { apiError = true; break; }   // API failed after retries: not a solve failure
        if (DELAY) await sleep(DELAY);
        try {
          const parts = splitReasoning(reply.text);
          const acts = parseActions(isolateActionsJson(parts.rest || reply.text));
          const model = applyActions(start, acts).model;
          const grade = gradeTask(task, model, acts.length, acts.map(a => a?.type).filter(Boolean));
          if (grade.pass) { reasoning = parts.reasoning; actions = acts; source = `${provider}:verified`; ok = true; }
        } catch { /* parse/apply failed; try again */ }
      }
      if (!ok) { if (apiError) errored += 1; else rejected += 1; continue; }
    } else {
      // Keyless: reference-derived (verified by construction; assert it).
      const grade = gradeTask(task, applyActions(start, reference).model, reference.length, reference.map(a => a.type));
      if (!grade.pass) { errored += 1; continue; }
      reasoning = referenceReasoning(task, start, reference); actions = reference; source = 'reference-derived';
    }

    stream.write(JSON.stringify({
      task_id: task.id,
      spec: task.spec,
      observation: serializeModel(start),
      reasoning,
      actions,
      verified: true,           // the design was re-graded and passed
      source,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: task.spec },
        { role: 'assistant', content: `<reasoning>${reasoning}</reasoning>\n${JSON.stringify({ actions })}` },
      ],
    }) + '\n');
    kept += 1;
  }
  stream.end();

  const solvable = attempted - errored;   // exclude API failures from the denominator
  const stats = { out: `${OUT}.jsonl`, mode: call ? `${provider} (rejection-sampled, tries=${TRIES})` : 'reference-derived (keyless)',
    attempted, kept, rejected, errored,
    keptRate: +(kept / attempted).toFixed(3),
    solveRate: +(kept / Math.max(1, solvable)).toFixed(3),   // true verified-solve rate, API errors removed
    seed: SEED };
  fs.writeFileSync(path.resolve(`${OUT}.stats.json`), JSON.stringify(stats, null, 2));
  console.log(`Wrote ${kept} VERIFIED reasoning traces to ${OUT}.jsonl (${stats.mode}).`);
  console.log(`  attempted ${attempted}, kept ${kept}, rejected ${rejected}, errored ${errored} (API failures).`);
  if (errored) console.log(`  verified-solve rate excluding API errors: ${(100 * kept / Math.max(1, solvable)).toFixed(1)}% (${kept}/${solvable}). Re-run with --delay to cut API errors.`);
  console.log('  Every kept trace’s design was re-graded by the deterministic verifier and passed.');
}

run().catch(err => { console.error(err); process.exit(2); });
