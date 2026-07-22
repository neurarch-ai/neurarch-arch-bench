#!/usr/bin/env node
/**
 * env-server — the benchmark as an HTTP reward service for RL training.
 *
 * A GRPO/PPO trainer (see training/) needs two things from an environment:
 * a stream of tasks (prompt material) and a reward for each completion. This
 * server exposes both over plain HTTP so the trainer can live in Python while
 * the verifier stays the single JS implementation — no port drift, no GPU on
 * this side, sub-millisecond grading.
 *
 * Zero dependencies (node:http only), node >= 18.
 *
 *   node env-server.mjs                # listens on :8737
 *   PORT=9000 node env-server.mjs
 *
 * Endpoints:
 *   GET  /health
 *     -> { ok: true }
 *
 *   GET  /tasks?count=256&seed=123&split=generated
 *     -> [{ index, id, spec, observation, constraints }]
 *     Deterministic: the same (count, seed) always yields the same split, so
 *     train/held-out splits are just different seeds.
 *     split=curated returns the hand-authored tasks.json set instead.
 *
 *   POST /grade   { "seed": 123, "count": 256, "index": 4, "actions": [...] }
 *     or          { "taskId": "cnn-cifar", "actions": [...] }        (curated)
 *     -> { pass, score, params, blockers, failures, applied, applyErrors, reward }
 *
 *   Multi-turn repair episodes (the verifier's failures are the feedback):
 *   POST /episode/start  { "seed": 123, "count": 256, "index": 4, "maxTurns": 3 }
 *     -> { episodeId, spec, observation, turn, maxTurns }
 *   POST /episode/step   { "episodeId": "...", "actions": [...] }
 *     -> { observation, grade..., reward, feedback: [failures], turn, done }
 *     The graph MUTATES across turns (edits accumulate); done on pass or when
 *     the turn budget runs out. Step reward uses the same shaping as /grade,
 *     so a single-turn episode is exactly /grade.
 *
 * Reward shaping (documented so results are interpretable):
 *   reward = (pass ? 1.0 : 0.0)            # the verifiable outcome
 *          + 0.5 * score / 100             # dense gradient on graph health
 *          - 0.1 * min(applyErrors, 3)     # malformed edits cost a little
 * Range ~ [-0.3, 1.5]. A completion that fails to parse as JSON never reaches
 * this server; the trainer assigns it a flat penalty (see training/).
 */
import http from 'node:http';
import { loadBenchmark, buildFixture, applyActions, gradeTask, serializeModel } from './bench.mjs';
import { generateCases } from './generate.mjs';
import { generateFrontierCases } from './generate-frontier.mjs';
import { generateEdgeCases } from './generate-edge.mjs';

const PORT = Number(process.env.PORT || 8737);
const BENCH = loadBenchmark();

// Splits are deterministic functions of (count, seed, tier); memoize recents.
const splitCache = new Map();
function getSplit(count, seed, tier = 'generated') {
  const key = `${tier}:${count}:${seed}`;
  if (!splitCache.has(key)) {
    if (splitCache.size > 16) splitCache.delete(splitCache.keys().next().value);
    const gen = tier === 'frontier' ? generateFrontierCases : tier === 'edge' ? generateEdgeCases : generateCases;
    splitCache.set(key, gen(count, seed));
  }
  return splitCache.get(key);
}

function curatedCase(taskId) {
  const task = BENCH.tasks.find(t => t.id === taskId);
  if (!task) return null;
  return { task, start: buildFixture(BENCH, task.start) };
}

export function computeReward(grade, applyErrors) {
  return (grade.pass ? 1.0 : 0.0) + 0.5 * (grade.score / 100) - 0.1 * Math.min(applyErrors, 3);
}

// ── Multi-turn episode store ─────────────────────────────────────────────────
// In-memory and bounded: an episode is (task, current graph, turn counter).
// Oldest episodes are evicted past the cap; ids are a counter, not a secret.
const EPISODE_CAP = 10_000;
const episodes = new Map();
let episodeCounter = 0;

function startEpisode(picked, maxTurns) {
  if (episodes.size >= EPISODE_CAP) episodes.delete(episodes.keys().next().value);
  const id = `ep-${++episodeCounter}`;
  episodes.set(id, {
    task: picked.task,
    model: structuredClone(picked.start),
    turn: 0,
    maxTurns: Math.max(1, Math.min(16, maxTurns)),
    usedTypes: [],
    actionCount: 0,
    done: false,
  });
  return id;
}

function json(res, status, body) {
  const s = JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
  });
  res.end(s);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', c => {
      data += c;
      if (data.length > 5_000_000) { reject(new Error('body too large')); req.destroy(); }
    });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  try {
    if (req.method === 'OPTIONS') return json(res, 204, {});

    if (req.method === 'GET' && url.pathname === '/health') {
      return json(res, 200, { ok: true, benchmark: BENCH.version });
    }

    if (req.method === 'GET' && url.pathname === '/tasks') {
      const split = url.searchParams.get('split') ?? 'generated';
      if (split === 'curated') {
        return json(res, 200, BENCH.tasks.map((t, index) => ({
          index, id: t.id, spec: t.spec,
          observation: serializeModel(buildFixture(BENCH, t.start)),
          constraints: t.constraints,
        })));
      }
      const count = Math.max(1, Math.min(100_000, Number(url.searchParams.get('count') ?? 256)));
      const seed = Number(url.searchParams.get('seed') ?? 1);
      return json(res, 200, getSplit(count, seed, split).map((c, index) => ({
        index, id: c.task.id, spec: c.task.spec,
        observation: serializeModel(c.start),
        constraints: c.task.constraints,
      })));
    }

    if (req.method === 'POST' && url.pathname === '/grade') {
      const body = JSON.parse(await readBody(req) || '{}');
      const actions = Array.isArray(body.actions) ? body.actions : [];
      let picked;
      if (typeof body.taskId === 'string') {
        picked = curatedCase(body.taskId);
        if (!picked) return json(res, 404, { error: `unknown taskId "${body.taskId}"` });
      } else {
        const count = Math.max(1, Math.min(100_000, Number(body.count ?? 256)));
        const seed = Number(body.seed ?? 1);
        const index = Number(body.index ?? -1);
        const split = getSplit(count, seed, body.split ?? 'generated');
        if (!(index >= 0 && index < split.length)) {
          return json(res, 400, { error: `index ${body.index} out of range for count=${count}` });
        }
        picked = split[index];
      }
      const applied = applyActions(picked.start, actions);
      const grade = gradeTask(picked.task, applied.model, actions.length, actions.map(a => a?.type).filter(Boolean));
      return json(res, 200, {
        ...grade,
        applied: applied.applied,
        applyErrors: applied.errors,
        reward: computeReward(grade, applied.errors.length),
      });
    }

    if (req.method === 'POST' && url.pathname === '/episode/start') {
      const body = JSON.parse(await readBody(req) || '{}');
      let picked;
      if (typeof body.taskId === 'string') {
        picked = curatedCase(body.taskId);
        if (!picked) return json(res, 404, { error: `unknown taskId "${body.taskId}"` });
      } else {
        const count = Math.max(1, Math.min(100_000, Number(body.count ?? 256)));
        const seed = Number(body.seed ?? 1);
        const index = Number(body.index ?? -1);
        const split = getSplit(count, seed, body.split ?? 'generated');
        if (!(index >= 0 && index < split.length)) {
          return json(res, 400, { error: `index ${body.index} out of range for count=${count}` });
        }
        picked = split[index];
      }
      const episodeId = startEpisode(picked, Number(body.maxTurns ?? 3));
      const ep = episodes.get(episodeId);
      return json(res, 200, {
        episodeId,
        taskId: ep.task.id,
        spec: ep.task.spec,
        observation: serializeModel(ep.model),
        turn: 0,
        maxTurns: ep.maxTurns,
      });
    }

    if (req.method === 'POST' && url.pathname === '/episode/step') {
      const body = JSON.parse(await readBody(req) || '{}');
      const ep = episodes.get(body.episodeId);
      if (!ep) return json(res, 404, { error: `unknown episodeId "${body.episodeId}"` });
      if (ep.done) return json(res, 400, { error: 'episode already finished' });
      const actions = Array.isArray(body.actions) ? body.actions : [];
      const applied = applyActions(ep.model, actions);
      ep.model = applied.model;
      ep.actionCount += actions.length;
      ep.usedTypes.push(...actions.map(a => a?.type).filter(Boolean));
      ep.turn += 1;
      const grade = gradeTask(ep.task, ep.model, ep.actionCount, ep.usedTypes);
      ep.done = grade.pass || ep.turn >= ep.maxTurns;
      if (ep.done) episodes.delete(body.episodeId);
      return json(res, 200, {
        ...grade,
        applied: applied.applied,
        applyErrors: applied.errors,
        reward: computeReward(grade, applied.errors.length),
        observation: serializeModel(ep.model),
        feedback: grade.failures,
        turn: ep.turn,
        done: ep.done,
      });
    }

    return json(res, 404, { error: 'not found' });
  } catch (err) {
    return json(res, 400, { error: String(err?.message ?? err) });
  }
});

// Importable for tests; listens only when run directly.
if (import.meta.url === `file://${process.argv[1]}`) {
  server.listen(PORT, () => {
    console.log(`neurarch-arch-bench env server on http://localhost:${PORT}`);
    console.log(`  GET  /tasks?count=256&seed=123`);
    console.log(`  POST /grade { seed, count, index, actions }`);
  });
}

export { server };
