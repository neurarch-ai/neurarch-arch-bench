/**
 * mint_repair_prompts — keyless prompts for repair-conditioned RL.
 *
 * The amplification study shows one round of verifier feedback closes the gap
 * for frontier models at inference time. This mints the TRAINING version of
 * that setting: for each generated task, a deliberately flawed first attempt
 * (a corrupted reference) plus the exact failure messages the verifier returns
 * for it. A policy trained on these prompts learns to consume verifier
 * feedback and emit the corrected action list; the reward is the same /grade
 * endpoint as plain GRPO (rows carry seed/count/index).
 *
 * Corruption kinds rotate per task and every corrupted attempt is re-graded to
 * PROVE it fails before it is written; a task whose corruptions all pass is
 * skipped and counted.
 *
 * Usage:
 *   node training/mint_repair_prompts.mjs --count=256 --seed=123 --out=repair-train.jsonl
 *   node training/mint_repair_prompts.mjs --count=192 --seed=999 --out=repair-eval.jsonl
 * Then:
 *   python training/train_grpo.py --steps 100 --lora --repair-prompts repair-train.jsonl ...
 */
import fs from 'node:fs';
import { applyActions, gradeTask, serializeModel } from '../bench.mjs';
import { SYSTEM_PROMPT } from '../providers.mjs';
import { generateCases } from '../generate.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const COUNT = Math.max(1, parseInt(args.count ?? '256', 10) || 256);
const SEED = parseInt(args.seed ?? '123', 10) || 123;
const OUT = args.out ?? 'repair-prompts.jsonl';
// --chat-out=file.jsonl additionally writes SFT rows: the repair prompt as the
// user turn and the task's REFERENCE solution as the verified target, so a
// policy can be TAUGHT the feedback-consumption format before RL refines it.
// The user text must stay byte-identical to train_grpo.py::build_repair_prompt.
const CHAT_OUT = args['chat-out'] ?? null;
// --families=fix,trim,norm,grow restricts minting to those families. The
// edit-in-place families are the natural repair setting: attempts and targets
// are short surgical action lists that fit training and completion budgets,
// unlike design families whose 'repair' degenerates to regenerating the full
// graph and blows past max-len (the mixed-run failure mode).
const FAMS = args.families ? new Set(args.families.split(',').map(s => s.trim())) : null;

function corruptions(reference) {
  const variants = [];
  const base = () => structuredClone(reference);
  if (reference[0]?.type === 'replace_model') {
    { const a = base(); const rm = a[0]; if (rm.connections?.length) { rm.connections = rm.connections.slice(0, -1); variants.push(['dropped-connection', a]); } }
    { const a = base(); const rm = a[0];
      const att = rm.components?.find(c => typeof c.params?.numHeads === 'number');
      if (att) { const d = att.params.embedDim ?? att.params.hiddenDim ?? 0; att.params.numHeads = d % 7 !== 0 ? 7 : att.params.numHeads + 1; variants.push(['broken-divisibility', a]); } }
    { const a = base(); const rm = a[0]; if ((rm.components?.length ?? 0) > 4) { rm.components = [...rm.components.slice(0, -2), rm.components[rm.components.length - 1]]; rm.connections = rm.connections?.filter(cn => rm.components.some(c => c.name === cn.from) && rm.components.some(c => c.name === cn.to)); variants.push(['dropped-component', a]); } }
  } else {
    if (reference.length > 1) variants.push(['incomplete-repair', reference.slice(0, -1)]);
    // Try several candidate wrong values: 1 often trivially passes (numHeads=1
    // divides everything), so 7 / 13 / value+1 are tried too; every variant is
    // re-graded below, so only ones that provably fail survive.
    for (const bad of [7, 13, 1, null]) {
      const a = base(); const withParams = a.find(x => x.params && Object.keys(x.params).length);
      if (!withParams) break;
      const k = Object.keys(withParams.params)[0];
      const v = bad === null ? Number(withParams.params[k]) + 1 : bad;
      if (v === withParams.params[k]) continue;
      withParams.params = { ...withParams.params, [k]: v };
      variants.push(['wrong-param', a]);
    }
  }
  return variants;
}

const rows = [];
const chatRows = [];
let skipped = 0;
const kinds = {};
const cases = generateCases(COUNT, SEED);
cases.forEach((c, index) => {
  if (FAMS && !FAMS.has(c.task.id.split('-')[1])) return;
  for (const [kind, attempt] of corruptions(c.reference)) {
    const g = gradeTask(c.task, applyActions(c.start, attempt).model, attempt.length, attempt.map(a => a?.type).filter(Boolean));
    if (g.pass) continue; // corruption did not bite; try the next kind
    rows.push(JSON.stringify({
      seed: SEED, count: COUNT, index,
      task_id: c.task.id, spec: c.task.spec,
      observation: serializeModel(c.start),
      attempt, failures: g.failures, corruption: kind,
    }));
    if (CHAT_OUT) {
      const user = `SPEC:\n${c.task.spec}\n\n`
        + `CURRENT MODEL (unchanged -- the previous attempt below was NOT applied):\n${serializeModel(c.start)}\n\n`
        + `A PREVIOUS ATTEMPT failed:\n${JSON.stringify({ actions: attempt })}\n\n`
        + 'THE VERIFIER REPORTED:\n- ' + g.failures.join('\n- ') + '\n\n'
        + 'Return the COMPLETE corrected action list to apply to the CURRENT MODEL '
        + 'above. Do NOT return a patch on the previous attempt: its actions were '
        + 'never applied, and names it introduced do not exist unless your list '
        + 'creates them. Fix what the verifier reported and output one full '
        + 'actions JSON.';
      chatRows.push(JSON.stringify({ messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: user },
        { role: 'assistant', content: JSON.stringify({ actions: c.reference }) },
      ] }));
    }
    kinds[kind] = (kinds[kind] ?? 0) + 1;
    return;
  }
  skipped += 1;
});

fs.writeFileSync(OUT, rows.join('\n') + '\n');
if (CHAT_OUT) { fs.writeFileSync(CHAT_OUT, chatRows.join('\n') + '\n'); console.log(`Wrote ${chatRows.length} repair SFT rows to ${CHAT_OUT} (reference targets, verified by construction).`); }
console.log(`Wrote ${rows.length} repair prompts to ${OUT} (${skipped} tasks skipped: no corruption failed).`);
console.log('corruption mix:', JSON.stringify(kinds));
console.log('Every attempt above was re-graded and provably fails its task.');
