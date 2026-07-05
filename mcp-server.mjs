#!/usr/bin/env node
/**
 * mcp-server — the verifier as a Model Context Protocol tool server.
 *
 * Give ANY agent (Claude Code, Grok agents, custom stacks) a deterministic
 * physics checker for neural architectures: audit a graph, grade a benchmark
 * attempt, mint fresh tasks. Zero dependencies; newline-delimited JSON-RPC
 * over stdio per the MCP spec.
 *
 * Register (e.g. Claude Code):
 *   claude mcp add arch-bench -- node /path/to/neurarch-arch-bench/mcp-server.mjs
 *
 * Tools:
 *   audit_architecture  graph -> blockers, 0..100 score, params, KV/token
 *   grade_task          (taskId | seed+count+index) + actions -> pass/failures/reward
 *   generate_tasks      count/seed/offset/limit -> task specs + observations
 */
import readline from 'node:readline';
import {
  loadBenchmark, buildFixture, applyActions, gradeTask, scoreModel,
  estimateParams, kvBytesPerToken, serializeModel,
} from './bench.mjs';
import { generateCases } from './generate.mjs';
import { computeReward } from './env-server.mjs';

const BENCH = loadBenchmark();

// ── Graph hydration for tool input (same convention as buildFixture) ────────
function hydrate(graph) {
  const components = (graph.components ?? []).map((c, i) => ({
    id: c.id ?? `n${i}`,
    type: c.type ?? c.componentType,
    name: c.name ?? `${c.type ?? 'node'}-${i}`,
    params: { ...(c.params ?? {}) },
    inputs: [], outputs: [],
  }));
  const byName = new Map(components.map(c => [c.name, c.id]));
  const connections = (graph.connections ?? []).map((cn, i) => ({
    id: cn.id ?? `c${i}`,
    from: byName.get(cn.from) ?? cn.from,
    to: byName.get(cn.to) ?? cn.to,
  }));
  const byId = new Map(components.map(c => [c.id, c]));
  for (const cn of connections) {
    byId.get(cn.from)?.outputs.push(cn.to);
    byId.get(cn.to)?.inputs.push(cn.from);
  }
  return { name: graph.name ?? 'audit', components, connections };
}

// ── Tool implementations ─────────────────────────────────────────────────────
const TOOLS = [
  {
    name: 'audit_architecture',
    description: 'Deterministically audit a neural-network graph: structural blockers (attention divisibility, connectivity), 0..100 health score, estimated params, and KV cache bytes per generated token (fp16). No LLM judging; pure functions. Components need {type|componentType, name, params}; connections {from, to} by name or id.',
    inputSchema: {
      type: 'object',
      properties: {
        graph: {
          type: 'object',
          properties: {
            components: { type: 'array', items: { type: 'object' } },
            connections: { type: 'array', items: { type: 'object' } },
          },
          required: ['components'],
        },
        maxParams: { type: 'number', description: 'optional parameter budget for the score' },
      },
      required: ['graph'],
    },
    run: ({ graph, maxParams }) => {
      const model = hydrate(graph);
      const { score, blockers } = scoreModel(model, maxParams ? { maxParams } : {});
      return {
        score,
        blockers,
        params: estimateParams(model),
        kvBytesPerToken: kvBytesPerToken(model, 2),
        components: model.components.length,
        summary: serializeModel(model),
      };
    },
  },
  {
    name: 'grade_task',
    description: 'Grade an action plan against a benchmark task. Pass taskId for the curated set, or seed+count+index for a generated split. Returns pass/fail, failures, score, and the shaped RL reward.',
    inputSchema: {
      type: 'object',
      properties: {
        taskId: { type: 'string' },
        seed: { type: 'number' },
        count: { type: 'number' },
        index: { type: 'number' },
        actions: { type: 'array', items: { type: 'object' } },
      },
      required: ['actions'],
    },
    run: ({ taskId, seed = 1, count = 256, index, actions }) => {
      let task, start;
      if (typeof taskId === 'string') {
        task = BENCH.tasks.find(t => t.id === taskId);
        if (!task) throw new Error(`unknown taskId "${taskId}"`);
        start = buildFixture(BENCH, task.start);
      } else {
        if (typeof index !== 'number') throw new Error('need taskId, or seed+count+index');
        const split = generateCases(count, seed);
        if (!(index >= 0 && index < split.length)) throw new Error(`index out of range for count=${count}`);
        ({ task, start } = split[index]);
      }
      const applied = applyActions(start, actions);
      const grade = gradeTask(task, applied.model, actions.length, actions.map(a => a?.type).filter(Boolean));
      return { ...grade, applyErrors: applied.errors, reward: computeReward(grade, applied.errors.length) };
    },
  },
  {
    name: 'generate_tasks',
    description: 'Mint deterministic architecture-design tasks from a seed (same seed+count = identical split, so results are reproducible). Returns spec + serialized starting graph per task.',
    inputSchema: {
      type: 'object',
      properties: {
        count: { type: 'number', description: 'split size (default 256)' },
        seed: { type: 'number', description: 'split seed (default 1)' },
        offset: { type: 'number', description: 'first index to return (default 0)' },
        limit: { type: 'number', description: 'max tasks to return (default 10, cap 50)' },
      },
    },
    run: ({ count = 256, seed = 1, offset = 0, limit = 10 }) => {
      const split = generateCases(Math.min(100000, Math.max(1, count)), seed);
      return split.slice(offset, offset + Math.min(50, limit)).map((c, i) => ({
        index: offset + i,
        id: c.task.id,
        spec: c.task.spec,
        constraints: c.task.constraints,
        observation: serializeModel(c.start),
      }));
    },
  },
];

// ── Newline-delimited JSON-RPC over stdio ────────────────────────────────────
const send = (msg) => process.stdout.write(JSON.stringify(msg) + '\n');

function handle(req) {
  const { id, method, params } = req;
  const reply = (result) => id !== undefined && send({ jsonrpc: '2.0', id, result });
  const fail = (code, message) => id !== undefined && send({ jsonrpc: '2.0', id, error: { code, message } });

  try {
    if (method === 'initialize') {
      return reply({
        protocolVersion: params?.protocolVersion ?? '2024-11-05',
        capabilities: { tools: {} },
        serverInfo: { name: 'neurarch-arch-bench', version: '0.1.0' },
      });
    }
    if (method === 'notifications/initialized' || method?.startsWith('notifications/')) return;
    if (method === 'ping') return reply({});
    if (method === 'tools/list') {
      return reply({ tools: TOOLS.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })) });
    }
    if (method === 'tools/call') {
      const tool = TOOLS.find(t => t.name === params?.name);
      if (!tool) return fail(-32602, `unknown tool "${params?.name}"`);
      try {
        const result = tool.run(params?.arguments ?? {});
        return reply({ content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] });
      } catch (err) {
        return reply({ content: [{ type: 'text', text: `Error: ${err.message}` }], isError: true });
      }
    }
    return fail(-32601, `method not found: ${method}`);
  } catch (err) {
    return fail(-32603, String(err?.message ?? err));
  }
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', (line) => {
  const s = line.trim();
  if (!s) return;
  try {
    handle(JSON.parse(s));
  } catch {
    send({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'parse error' } });
  }
});
