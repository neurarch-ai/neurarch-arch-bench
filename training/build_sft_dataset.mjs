#!/usr/bin/env node
/**
 * build_sft_dataset — mint a verified architecture-design SFT dataset.
 *
 * Every generated task ships a reference solution, and every emitted row is
 * re-graded by the verifier before it is written, so the dataset carries a
 * machine-checked guarantee no scraped corpus can: 100% of targets pass the
 * benchmark's own constraints. Because tasks are procedurally generated from
 * a seed, the data is contamination-free by construction and this script can
 * mint arbitrarily large (or private, unpublished-seed) splits on demand.
 *
 *   node build_sft_dataset.mjs --count=10000 --seed=20260704 --out=arch-design-sft
 *
 * Writes <out>.raw.jsonl   — {id, family, spec, observation, actions, grade}
 *        <out>.chat.jsonl  — {messages:[system,user,assistant]} ready for
 *                            SFT trainers (assistant = the JSON action plan)
 *        and prints per-family stats. See DATASET_CARD.md for the HF card.
 */
import fs from 'node:fs';
import path from 'node:path';
import { applyActions, gradeTask } from '../bench.mjs';
import { generateCases } from '../generate.mjs';
import { SYSTEM_PROMPT } from '../providers.mjs';
import { serializeModel } from '../bench.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const COUNT = Math.max(1, parseInt(args.count ?? '10000', 10) || 10000);
const SEED = parseInt(args.seed ?? '20260704', 10) || 20260704;
const OUT = args.out ?? 'arch-design-sft';

const rawPath = path.resolve(`${OUT}.raw.jsonl`);
const chatPath = path.resolve(`${OUT}.chat.jsonl`);
const raw = fs.createWriteStream(rawPath);
const chat = fs.createWriteStream(chatPath);

const familyOf = (id) => id.replace(/^gen-/, '').replace(/-\d+$/, '');
const stats = {};
let written = 0, rejected = 0;

// Never let a training row collide with the standard eval split: exclude any
// case whose spec appears in the holdout (seed 999 x 512 covers eval splits up
// to count=512). Override with --holdout-seed / --holdout-count; 0 disables.
const HOLDOUT_SEED = parseInt(args['holdout-seed'] ?? '999', 10);
const HOLDOUT_COUNT = parseInt(args['holdout-count'] ?? '512', 10);
const holdout = new Set();
if (HOLDOUT_COUNT > 0) {
  for (const { task } of generateCases(HOLDOUT_COUNT, HOLDOUT_SEED)) holdout.add(task.spec);
}
let excluded = 0;

for (const { task, start, reference } of generateCases(COUNT, SEED)) {
  if (holdout.has(task.spec)) { excluded += 1; continue; }
  // The guarantee: re-apply and re-grade every reference before writing.
  const applied = applyActions(start, reference);
  const grade = gradeTask(task, applied.model, reference.length, reference.map(a => a.type));
  if (applied.errors.length || !grade.pass) { rejected += 1; continue; }

  const observation = serializeModel(start);
  const user = `SPEC:\n${task.spec}\n\nCURRENT MODEL:\n${observation}\n\nReturn the actions that fulfil the spec.`;

  raw.write(JSON.stringify({
    id: task.id,
    family: familyOf(task.id),
    seed: SEED,
    spec: task.spec,
    observation,
    constraints: task.constraints,
    actions: reference,
    grade: { pass: grade.pass, score: grade.score, params: grade.params },
  }) + '\n');

  chat.write(JSON.stringify({
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: user },
      { role: 'assistant', content: JSON.stringify({ actions: reference }) },
    ],
  }) + '\n');

  stats[familyOf(task.id)] = (stats[familyOf(task.id)] ?? 0) + 1;
  written += 1;
}

raw.end();
chat.end();

console.log(`Wrote ${written} verified rows (${rejected} rejected by re-grading — should be 0)`);
if (excluded) console.log(`  excluded ${excluded} row(s) colliding with the holdout split (seed ${HOLDOUT_SEED} x ${HOLDOUT_COUNT})`);
console.log(`  ${rawPath}`);
console.log(`  ${chatPath}`);
console.log('By family:');
for (const [f, n] of Object.entries(stats).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${f.padEnd(8)} ${n}`);
}
if (rejected > 0) {
  console.error('Non-zero rejects means the generator and grader disagree — investigate before publishing.');
  process.exit(1);
}
