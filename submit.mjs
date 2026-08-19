#!/usr/bin/env node
/**
 * submit: send your action plans to the public Arch-Bench ledger.
 *
 * The ledger accepts ACTION PLANS, never scores: the server re-applies your
 * plans and grades them with the same verifier this repo ships, then stores its
 * own verdict at a permanent URL (https://neurarch.com/result.html?id=...).
 * That is what makes a row citable: anyone can re-run it.
 *
 * Two-step flow, entirely self-serve:
 *
 *   1. Run the harness with traces on (any provider, or your own policy):
 *        XAI_API_KEY=... node leaderboard.mjs --providers=grok --traces=traces.jsonl
 *   2. Submit the plans:
 *        node submit.mjs --traces=traces.jsonl --model="grok-4" --submitter="Your Lab"
 *
 * Or, if your policy is not wired into providers.mjs, build the JSON yourself
 * and pipe it in:
 *        node submit.mjs --file=submission.json
 *   where submission.json is { "model": "...", "submitter": "...",
 *   "results": [ { "taskId": "...", "actions": [ ... ] }, ... ] }.
 *
 * Notes:
 *   - The public endpoint grades the curated split (tasks.json ids). Tasks you
 *     omit count as failures; the denominator is the benchmark, not your file.
 *   - No account or API key needed. Submission is open; publication on the
 *     public board happens after a human review, but your result URL is yours
 *     either way.
 *   - When traces contain several attempts for one task, the LAST one is
 *     submitted.
 */
import fs from 'node:fs';
import path from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));

const ENDPOINT = process.env.NEURARCH_BENCH_URL ?? 'https://neurarch.com/api/v1/bench';

function die(msg) { console.error(msg); process.exit(2); }

let payload;
if (args.file) {
  payload = JSON.parse(fs.readFileSync(path.resolve(args.file), 'utf8'));
} else if (args.traces) {
  const model = (args.model ?? '').trim();
  if (!model) die('--model=<label> is required with --traces (e.g. --model="grok-4")');
  const lines = fs.readFileSync(path.resolve(args.traces), 'utf8').split('\n').filter(Boolean);
  const byTask = new Map(); // last attempt per task wins
  for (const line of lines) {
    let rec;
    try { rec = JSON.parse(line); } catch { continue; }
    if (!rec.taskId || !Array.isArray(rec.actions)) continue; // errored attempts carry actions: null
    byTask.set(rec.taskId, rec.actions);
  }
  if (byTask.size === 0) die(`No usable (taskId, actions) records in ${args.traces}`);
  payload = {
    model,
    ...(args.submitter ? { submitter: args.submitter } : {}),
    results: [...byTask].map(([taskId, actions]) => ({ taskId, actions })),
  };
} else {
  die('usage: node submit.mjs --traces=traces.jsonl --model="<label>" [--submitter="Your Lab"]\n' +
      '       node submit.mjs --file=submission.json');
}

console.error(`Submitting ${payload.results.length} plan(s) as "${payload.model}" to ${ENDPOINT} ...`);
const r = await fetch(ENDPOINT, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});
const body = await r.json().catch(() => ({}));
if (!r.ok) die(`${r.status}: ${body.error ?? 'submission failed'}`);

console.log(JSON.stringify(body, null, 2));
console.error(`\nGraded ${body.graded?.passed}/${body.graded?.total} (avg ${body.graded?.avgScore}) by rubric v${body.graded?.rubricVersion}.`);
console.error(`Permanent result: ${body.url}`);
