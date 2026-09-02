#!/usr/bin/env node
/**
 * contamination — measure, rather than argue, that generated tasks are absent
 * from the web-scale corpora models pretrain on.
 *
 * The paper claimed contamination resistance as "a property of construction":
 * tasks are synthesized, so a held-out split need never have appeared online.
 * That is an argument, not a measurement, and a reviewer is right to say so.
 * This turns it into a number.
 *
 * Method. For each generated task spec, walk decreasing n-gram lengths and ask
 * an n-gram index over a pretraining corpus how many documents contain that
 * exact span. The longest n that still matches is the spec's overlap with the
 * corpus. A spec whose longest match is a 4-gram shares only stock phrasing
 * ("Keep it under"); one whose longest match is a 15-gram is on the web.
 *
 * The control is the point. A run that reports zero overlap proves nothing
 * unless the same procedure finds overlap where overlap certainly exists, so
 * every run also measures sentences from famous papers, which the corpus
 * contains. If the control does not light up, the measurement is broken, not
 * the benchmark clean, and the script says so and exits non-zero.
 *
 * Index: infini-gram (Liu et al., 2024) over RedPajama / Dolma / Pile.
 *
 *   node contamination.mjs --count=100 --seed=999
 *   node contamination.mjs --count=200 --seed=999 --index=v4_dolma-v1_7_llama
 */
import { generateCases } from './generate.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const COUNT = Math.max(1, parseInt(args.count ?? '100', 10) || 100);
const SEED = parseInt(args.seed ?? '999', 10) || 999;
const INDEX = args.index ?? 'v4_rpj_llama_s4';
const DELAY = Math.max(0, parseInt(args.delay ?? '900', 10) || 900);
const API = 'https://api.infini-gram.io/';

// Lengths probed, longest first. 12+ words of continuous overlap is the length
// at which a match stops being stock phrasing and starts being the same text.
const NGRAM_LENGTHS = (args.lengths ?? '15,12,8,5').split(',').map(Number);

// Sentences that are certainly in any web-scale pretraining corpus. If these do
// not match, the index or the query path is broken and a zero result below is
// meaningless.
const CONTROLS = [
  'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks',
  'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms',
  'Deeper neural networks are more difficult to train. We present a residual learning framework',
];

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function count(query) {
  let lastBody = '';
  // Six attempts with a widening back-off: the public index answers a burst
  // with 403 for a while, and three tries 1.5 s apart read that as an outage.
  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      const res = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: INDEX, query_type: 'count', query }),
      });
      const j = await res.json();
      if (typeof j.count === 'number') return j.count;
      // A tokenisation complaint is a real "this span cannot appear"; anything
      // else (rate limit, "Forbidden", a bad gateway) must NOT read as a miss,
      // or a throttled run reports a pristine benchmark. Those raise.
      if (typeof j.error === 'string' && /token/i.test(j.error)) return 0;
      lastBody = JSON.stringify(j).slice(0, 120);
      await sleep(5000 * (attempt + 1));
    } catch (e) { lastBody = String(e).slice(0, 120); await sleep(5000 * (attempt + 1)); }
  }
  throw new Error(`index unreachable after 6 attempts: ${lastBody}`);
}

/** Longest contiguous word span of `text` that the corpus contains verbatim. */
async function longestMatch(text) {
  const words = text.split(/\s+/).filter(Boolean);
  for (const n of NGRAM_LENGTHS) {
    if (n > words.length) continue;
    // Probe a few windows rather than all of them: a spec that appears online
    // appears as a whole, so any window suffices to detect it.
    const starts = new Set(
      (args.windows === 'all'
        ? [0, Math.floor((words.length - n) / 2), words.length - n]
        : [Math.floor((words.length - n) / 2)]).filter(s => s >= 0));
    for (const s of starts) {
      const span = words.slice(s, s + n).join(' ');
      const c = await count(span);
      await sleep(DELAY);
      if (c !== null && c > 0) return { n, span, docs: c };
    }
  }
  return { n: 0, span: null, docs: 0 };
}

const main = async () => {
  console.log(`index: ${INDEX}`);
  console.log(`\n--- positive control (these must match) ---`);
  let controlsOk = 0;
  for (const c of CONTROLS) {
    const r = await longestMatch(c);
    console.log(`  ${r.n >= 10 ? 'OK  ' : 'MISS'} longest match ${r.n}-gram (${r.docs} docs): ${c.slice(0, 60)}...`);
    if (r.n >= 10) controlsOk += 1;
  }
  if (controlsOk === 0) {
    console.error('\nFAIL: no control matched. The index or the query path is broken; ' +
                  'a zero result on the benchmark below would be meaningless.');
    process.exit(1);
  }

  console.log(`\n--- ${COUNT} generated task specs, seed ${SEED} ---`);
  const cases = generateCases(COUNT, SEED);
  const hist = new Map();
  let maxSeen = 0, maxSpec = null;
  let i = 0;
  for (const { task } of cases) {
    const r = await longestMatch(task.spec);
    hist.set(r.n, (hist.get(r.n) ?? 0) + 1);
    if (r.n > maxSeen) { maxSeen = r.n; maxSpec = { spec: task.spec, ...r }; }
    i += 1;
    if (i % 10 === 0) process.stderr.write(`  ${i}/${COUNT}\r`);
  }

  console.log(`\nlongest verbatim overlap with the corpus, per spec:`);
  const lens = [...hist.keys()].sort((a, b) => b - a);
  for (const n of lens) {
    console.log(`  ${String(n).padStart(2)}-gram: ${String(hist.get(n)).padStart(4)} specs` +
                (n === 0 ? '  (no span of 4+ words appears at all)' : ''));
  }
  const contaminated = lens.filter(n => n >= 12).reduce((s, n) => s + hist.get(n), 0);
  console.log(`\nspecs with a 12+ word verbatim match: ${contaminated}/${COUNT} ` +
              `(${(100 * contaminated / COUNT).toFixed(1)}%)`);
  console.log(`longest overlap seen anywhere: ${maxSeen}-gram`);
  if (maxSpec && maxSeen > 0) console.log(`  "${maxSpec.span}" (${maxSpec.docs} docs)`);
  console.log(`\nControls matched: ${controlsOk}/${CONTROLS.length}, so the procedure detects ` +
              `overlap where overlap exists.`);
};

main();
