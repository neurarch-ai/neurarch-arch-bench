#!/usr/bin/env node
/**
 * rubric_delta — how much does a shape-blind rubric overstate a model?
 *
 * Rubric v1 checked attention divisibility and input-to-output reachability
 * but never propagated widths, so a graph whose linear `inFeatures` disagreed
 * with its upstream, or whose branch was severed while another path survived,
 * was scored as clean. v2 checks both. The gap between the two verdicts is a
 * measurement: it is what any verifier that does not propagate shapes will
 * report as competence it did not observe.
 *
 * This script reconstructs the v1 verdict from a v2 leaderboard artifact
 * rather than re-running the models: a task counts as "v1 would have passed"
 * when every one of its recorded failures is either a v2-only blocker or the
 * low-score line that blocker mechanically produces (scoreModel collapses any
 * blocked graph to 20 - 5*|blockers|). Tasks carrying an independent failure
 * (over budget, missing layer type, too few components, action limit) failed
 * under v1 as well and are not flipped.
 *
 * Reconstruction, not a re-run: it needs no API key, and its fidelity is
 * checkable against the v1 numbers published from live runs.
 *
 *   node rubric_delta.mjs lb-claude-grok-gen.json
 *   node rubric_delta.mjs lb-*.json
 */
import fs from 'node:fs';

const V2_ONLY = /declares input width|differing widths|no incoming connection/;
const SCORE_LINE = /^score \d+ < min \d+$/;
const familyOf = (id) => id.replace(/^gen-/, '').replace(/-\d+$/, '');

/** True when every recorded failure is v2-only, so v1 saw a clean graph. */
export function v1WouldPass(row) {
  if (row.pass) return true;
  const failures = (row.failures ?? []).map(f => String(f).trim());
  if (!failures.length) return false;
  return failures.every(f => V2_ONLY.test(f) || SCORE_LINE.test(f));
}

export function delta(artifact) {
  const out = new Map();
  for (const row of artifact.rows ?? []) {
    if (row.status === 'ERROR') continue;
    const acc = out.get(row.provider) ?? { n: 0, v2: 0, v1: 0, flipped: [] };
    acc.n += 1;
    if (row.pass) { acc.v2 += 1; acc.v1 += 1; }
    else if (v1WouldPass(row)) { acc.v1 += 1; acc.flipped.push(row.taskId); }
    out.set(row.provider, acc);
  }
  return out;
}

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: node rubric_delta.mjs <leaderboard artifact>...');
  process.exit(2);
}

for (const file of files) {
  const artifact = JSON.parse(fs.readFileSync(file, 'utf8'));
  const split = typeof artifact.split === 'string' ? artifact.split : (artifact.split?.kind ?? artifact.split?.name ?? JSON.stringify(artifact.split));
  console.log(`\n== ${file}  (${split})`);
  console.log('| model | v2 (shape-aware) | v1 (reconstructed) | overstatement | concentrated in |');
  console.log('| --- | --- | --- | --- | --- |');
  for (const [provider, a] of delta(artifact)) {
    const pct = (k) => `${k}/${a.n} (${(100 * k / a.n).toFixed(1)}%)`;
    const fams = {};
    for (const id of a.flipped) fams[familyOf(id)] = (fams[familyOf(id)] ?? 0) + 1;
    const where = Object.entries(fams).sort((x, y) => y[1] - x[1]).map(([f, n]) => `${f} x${n}`).join(', ') || 'none';
    const model = artifact.models?.[provider] ?? provider;
    console.log(`| ${model} | ${pct(a.v2)} | ${pct(a.v1)} | +${(100 * (a.v1 - a.v2) / a.n).toFixed(1)} pts | ${where} |`);
  }
}
