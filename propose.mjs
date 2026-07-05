#!/usr/bin/env node
/**
 * propose — the benchmark that grows itself, with satisfiability proofs.
 *
 * An LLM proposes new design-from-spec tasks; nothing enters the accepted set
 * unless the proposal's OWN reference solution passes the deterministic
 * grader (proof the task is solvable) under safety-net constraints
 * (forbidBlockers / minScore / mustReachOutput are always enforced). Because
 * verification is a free pure function, LLM task authorship is safe here in a
 * way it is not for benchmarks graded by humans or LLM judges. An optional
 * probe model filters for difficulty: keep only tasks the probe fails but the
 * reference solves.
 *
 *   ANTHROPIC_API_KEY=... node propose.mjs --provider=claude --count=20 --out=proposed.json
 *   ... --probe=groq            # keep only tasks the probe model fails
 *   node leaderboard.mjs --providers=grok --tasks=proposed.json
 */
import fs from 'node:fs';
import path from 'node:path';
import { applyActions, gradeTask, serializeModel } from './bench.mjs';
import { generateCases } from './generate.mjs';
import { SYSTEM_PROMPT, REGISTRY, parseActions, runnableProviders } from './providers.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const PROVIDER = args.provider ?? 'claude';
const PROBE = args.probe ?? null;
const COUNT = Math.max(1, parseInt(args.count ?? '10', 10) || 10);
const OUT = args.out ?? 'proposed.json';
const MAX_ATTEMPTS = COUNT * 4;

function stub(id, shape) {
  const components = [
    { id: 'i', type: 'input', name: 'input', params: { shape }, inputs: [], outputs: ['o'] },
    { id: 'o', type: 'output', name: 'output', params: {}, inputs: ['i'], outputs: [] },
  ];
  return { name: id, components, connections: [{ id: 'c0', from: 'i', to: 'o' }] };
}

const PROPOSE_PROMPT = (examples, accepted) => `You design NEW benchmark tasks for architecture-design agents.

A task = a natural-language spec, an input shape, machine-checkable constraints, and a REFERENCE solution (actions that build a graph satisfying the spec). Your reference will be verified by a deterministic grader; propose only tasks you can also solve.

Constraint keys you may use: maxParams (number), mustContainTypes (array of layer types), minComponents (number).
Layer types available: input, output, linear, relu, gelu, conv2d, maxpool2d, flatten, dropout, batchNorm1d, layerNorm, embedding, multiHeadAttention, groupedQueryAttention, transformerBlock, concatenate, add, lstm.
${SYSTEM_PROMPT.split('Rules:')[1] ? 'Rules for the reference actions:' + SYSTEM_PROMPT.split('Rules:')[1] : ''}

Three existing tasks, as format examples (do NOT copy them):
${examples}

Already accepted this session (propose something meaningfully different):
${accepted.length ? accepted.map(s => `- ${s}`).join('\n') : '- (none yet)'}

Respond with ONE JSON object and nothing else:
{ "spec": "<the design brief, mention any budget>", "inputShape": [1, <dims...>], "constraints": { ... }, "reference": { "actions": [ ... ] } }`;

async function run() {
  const [prov] = runnableProviders([PROVIDER]).filter(p => !REGISTRY[p].oracle);
  if (!prov) { console.error(`Provider ${PROVIDER} not runnable (missing key?).`); process.exit(2); }
  const probe = PROBE ? runnableProviders([PROBE]).filter(p => !REGISTRY[p].oracle)[0] : null;
  if (PROBE && !probe) { console.error(`Probe ${PROBE} not runnable (missing key?).`); process.exit(2); }

  const examples = generateCases(6, 1)
    .filter(c => ['gen-mlp-', 'gen-cnn-', 'gen-txf-'].some(p => c.task.id.startsWith(p)))
    .slice(0, 3)
    .map(c => JSON.stringify({
      spec: c.task.spec,
      inputShape: c.start.components.find(x => x.type === 'input')?.params?.shape,
      constraints: { maxParams: c.task.constraints.maxParams, mustContainTypes: c.task.constraints.mustContainTypes, minComponents: c.task.constraints.minComponents },
    }))
    .join('\n');

  const accepted = [];
  let attempts = 0, rejectedProof = 0, rejectedProbe = 0, rejectedParse = 0;

  while (accepted.length < COUNT && attempts < MAX_ATTEMPTS) {
    attempts += 1;
    let proposal;
    try {
      const reply = await REGISTRY[prov].call(
        'You are a benchmark task designer. Output exactly one JSON object.',
        PROPOSE_PROMPT(examples, accepted.map(a => a.task.spec)),
      );
      const m = reply.text.replace(/```json?\n?|```\n?/g, '').match(/\{[\s\S]*\}/);
      proposal = JSON.parse(m[0]);
    } catch {
      rejectedParse += 1;
      console.log(`[parse ] attempt ${attempts} rejected`);
      continue;
    }

    const shape = Array.isArray(proposal.inputShape) && proposal.inputShape.every(n => Number.isFinite(n) && n > 0)
      ? proposal.inputShape : null;
    const refActions = proposal?.reference?.actions;
    if (!shape || typeof proposal.spec !== 'string' || !Array.isArray(refActions)) {
      rejectedParse += 1;
      console.log(`[shape ] attempt ${attempts} rejected (malformed proposal)`);
      continue;
    }

    const id = `prop-${accepted.length}`;
    const task = {
      id,
      spec: proposal.spec,
      constraints: {
        // Safety net first: a proposal cannot weaken the structural floor.
        forbidBlockers: true,
        minScore: 50,
        mustReachOutput: true,
        ...(typeof proposal.constraints?.maxParams === 'number' ? { maxParams: proposal.constraints.maxParams } : {}),
        ...(Array.isArray(proposal.constraints?.mustContainTypes) ? { mustContainTypes: proposal.constraints.mustContainTypes } : {}),
        ...(typeof proposal.constraints?.minComponents === 'number' ? { minComponents: proposal.constraints.minComponents } : {}),
      },
    };
    const start = stub(id, shape);

    // The satisfiability proof: the proposer's own reference must pass.
    const applied = applyActions(start, refActions);
    const grade = gradeTask(task, applied.model, refActions.length, refActions.map(a => a?.type).filter(Boolean));
    if (applied.errors.length || !grade.pass) {
      rejectedProof += 1;
      console.log(`[proof ] attempt ${attempts} rejected: ${applied.errors[0] ?? grade.failures[0]}`);
      continue;
    }

    // Optional difficulty filter: keep only tasks the probe model fails.
    if (probe) {
      try {
        const user = `SPEC:\n${task.spec}\n\nCURRENT MODEL:\n${serializeModel(start)}\n\nReturn the actions that fulfil the spec.`;
        const probeReply = await REGISTRY[probe].call(SYSTEM_PROMPT, user);
        const probeActions = parseActions(probeReply.text);
        const probeGrade = gradeTask(task, applyActions(start, probeActions).model, probeActions.length, probeActions.map(a => a?.type).filter(Boolean));
        if (probeGrade.pass) {
          rejectedProbe += 1;
          console.log(`[probe ] attempt ${attempts} rejected: too easy (${probe} solved it)`);
          continue;
        }
      } catch { /* probe parse failure counts as probe failing: keep the task */ }
    }

    accepted.push({ task, start, reference: refActions });
    console.log(`[KEEP  ] ${id}: ${task.spec.slice(0, 90)}${task.spec.length > 90 ? '…' : ''}`);
  }

  fs.writeFileSync(path.resolve(OUT), JSON.stringify({
    version: 1,
    proposer: REGISTRY[prov].modelId(),
    probe: probe ? REGISTRY[probe].modelId() : null,
    generatedAt: new Date().toISOString(),
    tasks: accepted,
  }, null, 2));
  console.log(`\nAccepted ${accepted.length}/${attempts} attempts `
    + `(rejected: ${rejectedProof} unproven, ${rejectedProbe} too easy, ${rejectedParse} malformed)`);
  console.log(`Wrote ${OUT}. Run it: node leaderboard.mjs --providers=<p> --tasks=${OUT}`);
}

run().catch(err => { console.error(err); process.exit(2); });
