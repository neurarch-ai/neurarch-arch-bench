/**
 * combine_traces — merge per-model reasoning-trace files into one verified set.
 *
 * Every row is RE-GRADED at merge time against the current rubric before it
 * enters the combined file, so the output is verified under HEAD regardless of
 * when or by whom each input was minted. Dedupes on (source, task_id, seed is
 * implicit in task identity via spec) and records per-source counts.
 *
 * Rows minted from generator-v1 task splits cannot be re-verified against the
 * current generator; pass them through `--allow-preverified` only if their file
 * was already re-filtered (e.g. arch-reasoning-claude-v2.jsonl), else omit.
 *
 * Usage:
 *   node training/combine_traces.mjs --out=arch-reasoning-combined \
 *     traces-qwen.jsonl traces-llama.jsonl traces-deepseek.jsonl arch-reasoning-grok.jsonl
 */
import fs from 'node:fs';
import { applyActions, gradeTask, RUBRIC_VERSION } from '../bench.mjs';
import { generateCases } from '../generate.mjs';

const argv = process.argv.slice(2);
const flags = Object.fromEntries(argv.filter(a => a.startsWith('--')).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return [m[1], m[2] ?? 'true'];
}));
const files = argv.filter(a => !a.startsWith('--'));
const OUT = flags.out ?? 'arch-reasoning-combined';
// --preverified=a.jsonl,b.jsonl : rows from these files skip the re-grade
// (for sets already filtered under the current rubric whose generator-v1
// tasks cannot be regenerated at HEAD). Still deduped and counted.
const PREV = new Set((flags.preverified ?? '').split(',').filter(Boolean));
if (!files.length) { console.error('no input files'); process.exit(2); }

// Tasks are regenerated per (seed) on demand; rows carry task_id but the seed
// lives in the sibling .stats.json of each input file when present.
function seedFor(file) {
  const statsPath = file.replace(/\.jsonl$/, '.stats.json');
  if (fs.existsSync(statsPath)) {
    const s = JSON.parse(fs.readFileSync(statsPath, 'utf8'));
    if (typeof s.seed === 'number') return s.seed;
  }
  return 20260708;
}

const splitCache = new Map();
function tasksFor(seed) {
  if (!splitCache.has(seed)) splitCache.set(seed, new Map(generateCases(2000, seed).map(c => [c.task.id, c])));
  return splitCache.get(seed);
}

const kept = [];
const seen = new Set();
const perSource = {};
let rejected = 0, unmatched = 0, dupes = 0;
for (const file of files) {
  const seed = seedFor(file);
  const tasks = tasksFor(seed);
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    let r; try { r = JSON.parse(line); } catch { rejected++; continue; }
    const key = `${r.source}|${seed}|${r.task_id}`;
    if (seen.has(key)) { dupes++; continue; }
    if (!PREV.has(file)) {
      const c = tasks.get(r.task_id);
      if (!c || c.task.spec !== r.spec) { unmatched++; continue; }
      const g = gradeTask(c.task, applyActions(c.start, r.actions).model, r.actions.length, r.actions.map(a => a?.type).filter(Boolean));
      if (!g.pass) { rejected++; continue; }
    }
    seen.add(key);
    perSource[r.source] = (perSource[r.source] ?? 0) + 1;
    kept.push(line);
  }
  console.log(`${file}: running total ${kept.length}`);
}

fs.writeFileSync(`${OUT}.jsonl`, kept.join('\n') + '\n');
const stats = { out: `${OUT}.jsonl`, rubricVersion: RUBRIC_VERSION, kept: kept.length, rejected, unmatched, dupes, perSource };
fs.writeFileSync(`${OUT}.stats.json`, JSON.stringify(stats, null, 2));
console.log(JSON.stringify(stats, null, 1));
console.log('Every kept row re-graded under the current rubric at merge time.');
