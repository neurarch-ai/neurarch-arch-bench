#!/usr/bin/env node
/**
 * neurarch-arch-bench — leaderboard runner.
 *
 * Runs frontier models against the design-from-spec benchmark: each model is
 * shown a spec + a starting graph and asked to emit edit actions; we apply them
 * and grade the resulting graph. Prints a leaderboard and optionally writes JSON.
 *
 * Usage (real API calls — costs tokens):
 *   XAI_API_KEY=xai-...       node leaderboard.mjs --providers=grok
 *   ANTHROPIC_API_KEY=sk-...  node leaderboard.mjs --providers=claude
 *   GEMINI_API_KEY=AIza...    node leaderboard.mjs --providers=gemini
 *   GROQ_API_KEY=gsk-...      node leaderboard.mjs --providers=groq
 *   OPENAI_API_KEY=sk-...     node leaderboard.mjs --providers=openai
 *   node leaderboard.mjs --providers=grok,claude,gemini --only=cnn-cifar,text-encoder
 *   LEADERBOARD_OUT=board.json node leaderboard.mjs --providers=grok
 *
 * Pick the exact model with <PROVIDER>_MODEL, e.g. XAI_MODEL=grok-3.
 */
import fs from 'node:fs';
import path from 'node:path';
import {
  loadBenchmark, loadSolutions, buildFixture, applyActions, gradeTask, serializeModel,
} from './bench.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const PROVIDERS = (args.providers ?? 'grok').split(',').map(s => s.trim());
const ONLY = args.only ? new Set(args.only.split(',')) : null;
const FORMAT = args.format ?? 'text'; // 'text' | 'md'
const OUT = process.env.LEADERBOARD_OUT;

const SYSTEM_PROMPT = `You are a neural-architecture design agent. You edit a structured model graph by emitting actions.
Respond with ONE JSON object and nothing else: { "actions": [ <action> ... ] }

Action types:
- { "type": "add_component", "componentType": "<layer>", "name": "<unique name>", "afterName": "<existing node to insert after>", "params": { ... } }
- { "type": "add_connection", "fromName": "<node>", "toName": "<node>" }
- { "type": "update_params", "name": "<node>", "params": { ... } }
- { "type": "scale_params", "paramKey": "<param>", "factor": <number>, "namePattern": "<optional regex>" }
- { "type": "delete_component", "name": "<node>" }
- { "type": "replace_model", "components": [ { "componentType": "...", "name": "...", "params": {...} } ], "connections": [ { "from": "...", "to": "..." } ] }

Rules:
- Insert layers in order using afterName so the graph stays connected input->output.
- Attention: embedDim MUST be divisible by numHeads.
- Param keys: linear {inFeatures,outFeatures}; conv2d {inChannels,outChannels,kernelSize}; embedding {numEmbeddings,embeddingDim}; multiHeadAttention {embedDim,numHeads}; transformerBlock {hiddenDim,numHeads}.
- Respect any parameter budget in the spec. Output only the JSON object.`;

async function openaiCompat(baseUrl, key, model, system, user) {
  const r = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({ model, temperature: 0.2, messages: [{ role: 'system', content: system }, { role: 'user', content: user }] }),
  });
  if (!r.ok) throw new Error(`${baseUrl} ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return (await r.json()).choices?.[0]?.message?.content ?? '';
}

// 'reference' is a keyless oracle: it replays the known-good solution for each
// task. It needs no API key, proves the benchmark is runnable end-to-end out of
// the box, and gives an upper-bound row real models are measured against.
const SOLUTIONS = loadSolutions();
const REGISTRY = {
  reference: {
    envKey: null,
    oracle: true,
    solve: (taskId) => SOLUTIONS[taskId] ?? [],
  },
  grok: {
    envKey: 'XAI_API_KEY',
    call: (s, u) => openaiCompat('https://api.x.ai/v1', process.env.XAI_API_KEY, process.env.XAI_MODEL ?? 'grok-4', s, u),
  },
  groq: {
    envKey: 'GROQ_API_KEY',
    call: (s, u) => openaiCompat('https://api.groq.com/openai/v1', process.env.GROQ_API_KEY, process.env.GROQ_MODEL ?? 'llama-3.3-70b-versatile', s, u),
  },
  openai: {
    envKey: 'OPENAI_API_KEY',
    call: (s, u) => openaiCompat('https://api.openai.com/v1', process.env.OPENAI_API_KEY, process.env.OPENAI_MODEL ?? 'gpt-4o', s, u),
  },
  claude: {
    envKey: 'ANTHROPIC_API_KEY',
    call: async (s, u) => {
      const r = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': process.env.ANTHROPIC_API_KEY, 'anthropic-version': '2023-06-01' },
        body: JSON.stringify({ model: process.env.ANTHROPIC_MODEL ?? 'claude-sonnet-4-6', max_tokens: 2000, system: s, messages: [{ role: 'user', content: u }] }),
      });
      if (!r.ok) throw new Error(`claude ${r.status}: ${(await r.text()).slice(0, 200)}`);
      return (await r.json()).content?.[0]?.text ?? '';
    },
  },
  gemini: {
    envKey: 'GEMINI_API_KEY',
    call: async (s, u) => {
      const model = process.env.GEMINI_MODEL ?? 'gemini-2.5-flash';
      const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${process.env.GEMINI_API_KEY}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ systemInstruction: { parts: [{ text: s }] }, contents: [{ role: 'user', parts: [{ text: u }] }], generationConfig: { temperature: 0.2, responseMimeType: 'application/json' } }),
      });
      if (!r.ok) throw new Error(`gemini ${r.status}: ${(await r.text()).slice(0, 200)}`);
      return (await r.json()).candidates?.[0]?.content?.parts?.[0]?.text ?? '';
    },
  },
};

function parseActions(raw) {
  let s = raw.trim();
  if (s.startsWith('```')) s = s.replace(/```json?\n?/g, '').replace(/```\n?/g, '');
  const m = s.match(/\{[\s\S]*\}/);
  if (!m) throw new Error('no JSON object in reply');
  const obj = JSON.parse(m[0]);
  if (!Array.isArray(obj.actions)) throw new Error('reply has no "actions" array');
  return obj.actions;
}

async function run() {
  const bench = loadBenchmark();
  const tasks = bench.tasks.filter(t => !ONLY || ONLY.has(t.id));
  const providers = PROVIDERS.filter(p => {
    const spec = REGISTRY[p];
    if (!spec) { console.error(`unknown provider "${p}" — skipping`); return false; }
    if (spec.oracle) return true; // keyless
    if (!process.env[spec.envKey]) { console.error(`${spec.envKey} not set — skipping ${p}`); return false; }
    return true;
  });
  if (!providers.length) { console.error('no runnable providers (set the API key env vars, or use --providers=reference)'); process.exit(2); }

  console.log(`neurarch-arch-bench v${bench.version}: ${tasks.length} tasks x ${providers.length} provider(s) [${providers.join(', ')}]\n`);
  const rows = [];

  for (const provider of providers) {
    const spec = REGISTRY[provider];
    for (const task of tasks) {
      const start = buildFixture(bench, task.start);
      const user = `SPEC:\n${task.spec}\n\nCURRENT MODEL:\n${serializeModel(start)}\n\nReturn the actions that fulfil the spec.`;
      const t0 = Date.now();
      try {
        const actions = spec.oracle ? spec.solve(task.id) : parseActions(await spec.call(SYSTEM_PROMPT, user));
        const { model } = applyActions(start, actions);
        const g = gradeTask(task, model, actions.length);
        const status = g.pass ? 'PASS' : 'FAIL';
        rows.push({ provider, taskId: task.id, status, score: g.score, ...g, ms: Date.now() - t0 });
        console.log(`[${status}] ${provider.padEnd(8)} ${task.id.padEnd(22)} score=${String(g.score).padStart(3)} params=${g.params} (${Date.now() - t0}ms)`);
        for (const f of g.failures) console.log(`         - ${f}`);
      } catch (err) {
        rows.push({ provider, taskId: task.id, status: 'ERROR', score: 0, error: String(err), ms: Date.now() - t0 });
        console.log(`[ERR ] ${provider.padEnd(8)} ${task.id.padEnd(22)} ${err.message}`);
      }
    }
  }

  const board = providers.map(p => {
    const pr = rows.filter(r => r.provider === p);
    const passed = pr.filter(r => r.status === 'PASS').length;
    const avgScore = pr.length ? Math.round(pr.reduce((a, r) => a + r.score, 0) / pr.length) : 0;
    return { provider: p, passed, total: pr.length, avgScore };
  }).sort((a, b) => b.passed - a.passed || b.avgScore - a.avgScore);

  if (FORMAT === 'md') {
    console.log('\n| Model | Passed | Avg score |');
    console.log('| --- | --- | --- |');
    for (const b of board) console.log(`| ${b.provider} | ${b.passed}/${b.total} | ${b.avgScore} |`);
  } else {
    console.log('\n-- Leaderboard --');
    for (const b of board) console.log(`  ${b.provider.padEnd(10)} ${b.passed}/${b.total} passed  avg score ${b.avgScore}`);
  }

  if (OUT) { fs.writeFileSync(path.resolve(OUT), JSON.stringify({ benchmark: bench.version, board, rows }, null, 2)); console.log(`\nWrote ${OUT}`); }
}

run().catch(err => { console.error(err); process.exit(2); });
