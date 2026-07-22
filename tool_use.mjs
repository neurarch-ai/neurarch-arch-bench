/**
 * tool_use — the verifier as a TOOL the model calls mid-reasoning.
 *
 * Frontier labs train reasoning models with tool-integrated reasoning: the
 * model calls tools inside its chain of thought and conditions on the results.
 * This harness measures exactly that setting for architecture design. The
 * model gets two function tools:
 *
 *   audit_architecture({actions})  -> grade the would-be graph WITHOUT
 *                                     submitting: pass, score, blockers,
 *                                     failures, params, kvBytesPerToken
 *   submit_actions({actions})      -> final answer, ends the episode
 *
 * Two arms on identical tasks, same model, same prompt contract:
 *   raw  — single shot, no tools (the leaderboard setting)
 *   tool — the model may audit as often as it likes before submitting
 *
 * The delta is the value of the verifier as an in-reasoning tool, as opposed
 * to post-hoc repair (amplify.mjs measures that). Works with any provider that
 * speaks the OpenAI tools schema (grok / openai / deepseek).
 *
 * Usage:
 *   node tool_use.mjs --self-check                     # keyless harness test
 *   XAI_API_KEY=... node tool_use.mjs --provider=grok --generate=30 --seed=7
 *   XAI_API_KEY=... node tool_use.mjs --provider=grok --generate=30 --seed=7 --tier=frontier
 *   TOOLUSE_OUT=tooluse-grok.json node tool_use.mjs ...   # save raw rows
 */
import { applyActions, gradeTask, serializeModel, kvBytesPerToken } from './bench.mjs';
import { generateCases } from './generate.mjs';
import { generateFrontierCases } from './generate-frontier.mjs';
import { generateEdgeCases } from './generate-edge.mjs';
import { SYSTEM_PROMPT, parseActions } from './providers.mjs';
import fs from 'node:fs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));

const PROVIDERS = {
  grok: { base: 'https://api.x.ai/v1', keyEnv: 'XAI_API_KEY', model: () => process.env.XAI_MODEL ?? 'grok-4' },
  openai: { base: 'https://api.openai.com/v1', keyEnv: 'OPENAI_API_KEY', model: () => process.env.OPENAI_MODEL ?? 'gpt-4o' },
  deepseek: { base: 'https://api.deepseek.com/v1', keyEnv: 'DEEPSEEK_API_KEY', model: () => process.env.DEEPSEEK_MODEL ?? 'deepseek-chat' },
};

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'audit_architecture',
      description: 'Dry-run the deterministic verifier on your proposed actions WITHOUT submitting. Returns pass, score, hard blockers, failures, params, and KV bytes/token. Call this before submitting; fix what it reports.',
      parameters: {
        type: 'object',
        properties: { actions: { type: 'array', description: 'The action list to audit, same schema as the final answer.', items: { type: 'object' } } },
        required: ['actions'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'submit_actions',
      description: 'Submit your final action list. Ends the episode.',
      parameters: {
        type: 'object',
        properties: { actions: { type: 'array', description: 'The final action list.', items: { type: 'object' } } },
        required: ['actions'],
      },
    },
  },
];

const TOOL_SYSTEM = `${SYSTEM_PROMPT}

You have a deterministic verifier available as a tool. Call audit_architecture with a candidate action list to see exactly what would fail, fix what it reports, and audit again if needed. When the audit passes, call submit_actions with the final list. Always finish by calling submit_actions.`;

function auditFor(taskCase) {
  return (actions) => {
    const applied = applyActions(taskCase.start, actions);
    const grade = gradeTask(taskCase.task, applied.model, actions.length, actions.map(a => a?.type).filter(Boolean));
    return {
      pass: grade.pass, score: grade.score, params: grade.params,
      kvBytesPerToken: kvBytesPerToken(applied.model),
      blockers: grade.blockers, failures: grade.failures, applyErrors: applied.errors,
    };
  };
}

async function chat(p, body) {
  const r = await fetch(`${p.base}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env[p.keyEnv]}` },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${p.base} ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

/** Tool arm: run the tool-calling loop for one task. Returns {actions, audits, rounds}. */
async function runToolEpisode(p, model, taskCase, maxRounds) {
  const audit = auditFor(taskCase);
  const messages = [
    { role: 'system', content: TOOL_SYSTEM },
    { role: 'user', content: `Task: ${taskCase.task.spec}\n\nCurrent model state:\n${serializeModel(taskCase.start)}` },
  ];
  let audits = 0;
  for (let round = 1; round <= maxRounds; round++) {
    const json = await chat(p, { model, temperature: 0.2, messages, tools: TOOLS });
    const msg = json.choices?.[0]?.message;
    if (!msg) throw new Error('empty completion');
    messages.push(msg);
    const calls = msg.tool_calls ?? [];
    if (!calls.length) {
      // Text-only turn: try to salvage a JSON plan, else nudge once.
      try { return { actions: parseActions(msg.content ?? ''), audits, rounds: round }; }
      catch { messages.push({ role: 'user', content: 'Call submit_actions with your final action list.' }); continue; }
    }
    for (const call of calls) {
      let parsed;
      try { parsed = JSON.parse(call.function?.arguments ?? '{}'); }
      catch { parsed = {}; }
      const actions = Array.isArray(parsed.actions) ? parsed.actions : [];
      if (call.function?.name === 'submit_actions') return { actions, audits, rounds: round };
      audits += 1;
      messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(audit(actions)) });
    }
  }
  return { actions: [], audits, rounds: maxRounds }; // never submitted
}

/** Raw arm: the leaderboard single-shot setting. */
async function runRawEpisode(p, model, taskCase) {
  const json = await chat(p, {
    model, temperature: 0.2,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: `Task: ${taskCase.task.spec}\n\nCurrent model state:\n${serializeModel(taskCase.start)}` },
    ],
  });
  return { actions: parseActions(json.choices?.[0]?.message?.content ?? '') };
}

function gradeCase(taskCase, actions) {
  const applied = applyActions(taskCase.start, actions);
  return gradeTask(taskCase.task, applied.model, actions.length, actions.map(a => a?.type).filter(Boolean));
}

// ─── Keyless self-check: the harness itself must be sound ────────────────────
function selfCheck() {
  const cases = [...generateCases(10, 7), ...generateFrontierCases(9, 7), ...generateEdgeCases(6, 7)];
  let ok = true;
  for (const c of cases) {
    const audit = auditFor(c);
    const auditRes = audit(c.reference);
    const refGrade = gradeCase(c, c.reference);
    const noopGrade = gradeCase(c, []);
    if (!auditRes.pass || !refGrade.pass) { console.error(`[self-check FAIL] ${c.task.id}: reference did not pass (audit=${auditRes.pass}, grade=${refGrade.pass}) ${refGrade.failures.join('; ')}`); ok = false; }
    if (noopGrade.pass) { console.error(`[self-check FAIL] ${c.task.id}: noop passed`); ok = false; }
    // Audit must be side-effect free: grading again after audit is identical.
    const again = gradeCase(c, c.reference);
    if (JSON.stringify(again) !== JSON.stringify(refGrade)) { console.error(`[self-check FAIL] ${c.task.id}: audit mutated state`); ok = false; }
  }
  console.log(ok ? `self-check OK: ${cases.length} cases (core + frontier), reference passes audit+grade, noop fails, audit is pure` : 'SELF-CHECK FAILED');
  process.exitCode = ok ? 0 : 1;
}

async function main() {
  if (args['self-check']) return selfCheck();

  const providerName = args.provider ?? 'grok';
  const p = PROVIDERS[providerName];
  if (!p) { console.error(`unknown provider "${providerName}" (grok|openai|deepseek)`); process.exit(1); }
  if (!process.env[p.keyEnv]) { console.error(`${p.keyEnv} not set`); process.exit(1); }
  const model = p.model();
  const count = Math.max(1, parseInt(args.generate ?? '30', 10) || 30);
  const seed = parseInt(args.seed ?? '7', 10) || 7;
  const tier = args.tier ?? 'generated';
  const maxRounds = Math.max(2, parseInt(args['max-rounds'] ?? '6', 10) || 6);
  const delay = Math.max(0, parseInt(args.delay ?? '0', 10) || 0);
  const arm = args.arm ?? 'both';

  const cases = (tier === 'frontier' ? generateFrontierCases : tier === 'edge' ? generateEdgeCases : generateCases)(count, seed);
  console.log(`tool-integrated reasoning vs single-shot | ${providerName} (${model}) | ${tier} split, count=${count}, seed=${seed}\n`);

  const rows = [];
  const tally = { raw: [0, 0], tool: [0, 0] }; // [graded, passed]
  let auditTotal = 0;
  for (const c of cases) {
    for (const a of arm === 'both' ? ['raw', 'tool'] : [arm]) {
      try {
        const out = a === 'raw' ? await runRawEpisode(p, model, c) : await runToolEpisode(p, model, c, maxRounds);
        const grade = gradeCase(c, out.actions);
        tally[a][0] += 1; tally[a][1] += grade.pass ? 1 : 0;
        if (a === 'tool') auditTotal += out.audits ?? 0;
        rows.push({ arm: a, taskId: c.task.id, pass: grade.pass, score: grade.score, audits: out.audits ?? 0, failures: grade.failures });
        console.log(`[${grade.pass ? 'PASS' : 'FAIL'}] ${a.padEnd(4)} ${c.task.id.padEnd(14)} score=${String(grade.score).padStart(3)}${a === 'tool' ? ` audits=${out.audits}` : ''}${grade.pass ? '' : `  (${grade.failures[0] ?? ''})`}`);
      } catch (err) {
        rows.push({ arm: a, taskId: c.task.id, error: String(err).slice(0, 200) });
        console.log(`[ERR ] ${a.padEnd(4)} ${c.task.id.padEnd(14)} ${String(err).slice(0, 120)}`);
      }
      if (delay) await new Promise(res => setTimeout(res, delay));
    }
  }

  console.log('\n=== summary ===');
  for (const a of ['raw', 'tool']) {
    const [graded, passed] = tally[a];
    if (!graded) continue;
    console.log(`${a.padEnd(4)}: ${passed}/${graded} = ${(100 * passed / graded).toFixed(1)}%${a === 'tool' ? `  (mean audits/task ${(auditTotal / graded).toFixed(2)})` : ''}`);
  }
  if (tally.raw[0] && tally.tool[0]) {
    const lift = 100 * (tally.tool[1] / tally.tool[0] - tally.raw[1] / tally.raw[0]);
    console.log(`tool-integrated lift: ${lift >= 0 ? '+' : ''}${lift.toFixed(1)} pts`);
  }
  if (process.env.TOOLUSE_OUT) {
    fs.writeFileSync(process.env.TOOLUSE_OUT, JSON.stringify({ provider: providerName, model, tier, count, seed, rows }, null, 1));
    console.log(`rows saved to ${process.env.TOOLUSE_OUT}`);
  }
}

main().catch(err => { console.error(err); process.exit(1); });
