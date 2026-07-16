/**
 * providers — the model registry shared by the leaderboard and the
 * amplification study. One "design" model per provider, all speaking the same
 * action-plan contract. Keys come from env vars; a provider with no key is
 * simply skipped by the callers.
 */
export const SYSTEM_PROMPT = `You are a neural-architecture design agent. You edit a structured model graph by emitting actions.
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
- Param keys: linear {inFeatures,outFeatures}; conv2d {inChannels,outChannels,kernelSize}; embedding {numEmbeddings,embeddingDim}; multiHeadAttention {embedDim,numHeads}; groupedQueryAttention {embedDim,numHeads,numKVHeads}; transformerBlock {hiddenDim,numHeads}; batchNorm1d {numFeatures}; layerNorm {normalizedShape}; concatenate {dim}.
- GQA: numHeads MUST also be divisible by numKVHeads.
- If the spec says to repair or edit in place, use surgical actions (update_params, add_component); do NOT use replace_model or clear_canvas.
- Every numeric value MUST be a single computed integer, never an arithmetic expression: write "inFeatures": 6400, NOT "inFeatures": 64 * 10 * 10.
- Respect any parameter budget in the spec. Output only the JSON object.`;

// Every call returns { text, tokens } — tokens is the provider-reported total
// so the leaderboard can price intelligence (tokens per solved task), not just
// rank it.
async function openaiCompat(baseUrl, key, model, system, user) {
  const r = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({ model, temperature: 0.2, messages: [{ role: 'system', content: system }, { role: 'user', content: user }] }),
  });
  if (!r.ok) throw new Error(`${baseUrl} ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const json = await r.json();
  return { text: json.choices?.[0]?.message?.content ?? '', tokens: json.usage?.total_tokens ?? 0 };
}

// 'reference' is a keyless oracle handled by the callers (it replays each
// case's known-good solution) — it has no `call`.
export const REGISTRY = {
  reference: {
    envKey: null,
    oracle: true,
    modelId: () => 'reference',
  },
  grok: {
    envKey: 'XAI_API_KEY',
    modelId: () => process.env.XAI_MODEL ?? 'grok-4',
    call: (s, u) => openaiCompat('https://api.x.ai/v1', process.env.XAI_API_KEY, process.env.XAI_MODEL ?? 'grok-4', s, u),
  },
  groq: {
    envKey: 'GROQ_API_KEY',
    modelId: () => process.env.GROQ_MODEL ?? 'llama-3.3-70b-versatile',
    call: (s, u) => openaiCompat('https://api.groq.com/openai/v1', process.env.GROQ_API_KEY, process.env.GROQ_MODEL ?? 'llama-3.3-70b-versatile', s, u),
  },
  openai: {
    envKey: 'OPENAI_API_KEY',
    modelId: () => process.env.OPENAI_MODEL ?? 'gpt-4o',
    call: (s, u) => openaiCompat('https://api.openai.com/v1', process.env.OPENAI_API_KEY, process.env.OPENAI_MODEL ?? 'gpt-4o', s, u),
  },
  deepseek: {
    envKey: 'DEEPSEEK_API_KEY',
    modelId: () => process.env.DEEPSEEK_MODEL ?? 'deepseek-chat',
    call: (s, u) => openaiCompat('https://api.deepseek.com/v1', process.env.DEEPSEEK_API_KEY, process.env.DEEPSEEK_MODEL ?? 'deepseek-chat', s, u),
  },
  openrouter: {
    // One key, dozens of open models. Pick any via OPENROUTER_MODEL, e.g.
    // qwen/qwen-2.5-72b-instruct, mistralai/mistral-large, meta-llama/llama-3.3-70b-instruct.
    envKey: 'OPENROUTER_API_KEY',
    modelId: () => process.env.OPENROUTER_MODEL ?? 'meta-llama/llama-3.3-70b-instruct',
    call: (s, u) => openaiCompat('https://openrouter.ai/api/v1', process.env.OPENROUTER_API_KEY, process.env.OPENROUTER_MODEL ?? 'meta-llama/llama-3.3-70b-instruct', s, u),
  },
  claude: {
    envKey: 'ANTHROPIC_API_KEY',
    modelId: () => process.env.ANTHROPIC_MODEL ?? 'claude-sonnet-4-6',
    call: async (s, u) => {
      const r = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': process.env.ANTHROPIC_API_KEY, 'anthropic-version': '2023-06-01' },
        body: JSON.stringify({ model: process.env.ANTHROPIC_MODEL ?? 'claude-sonnet-4-6', max_tokens: 2000, system: s, messages: [{ role: 'user', content: u }] }),
      });
      if (!r.ok) throw new Error(`claude ${r.status}: ${(await r.text()).slice(0, 200)}`);
      const json = await r.json();
      return { text: json.content?.[0]?.text ?? '', tokens: (json.usage?.input_tokens ?? 0) + (json.usage?.output_tokens ?? 0) };
    },
  },
  gemini: {
    envKey: 'GEMINI_API_KEY',
    modelId: () => process.env.GEMINI_MODEL ?? 'gemini-2.5-flash',
    call: async (s, u) => {
      const model = process.env.GEMINI_MODEL ?? 'gemini-2.5-flash';
      const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${process.env.GEMINI_API_KEY}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ systemInstruction: { parts: [{ text: s }] }, contents: [{ role: 'user', parts: [{ text: u }] }], generationConfig: { temperature: 0.2, responseMimeType: 'application/json' } }),
      });
      if (!r.ok) throw new Error(`gemini ${r.status}: ${(await r.text()).slice(0, 200)}`);
      const json = await r.json();
      return { text: json.candidates?.[0]?.content?.parts?.[0]?.text ?? '', tokens: json.usageMetadata?.totalTokenCount ?? 0 };
    },
  },
};

// Evaluate a simple integer arithmetic expression (only * and +, e.g. the
// `64 * 10 * 10` a model writes for a flattened conv dim) WITHOUT eval.
function evalIntExpr(expr) {
  const flat = expr.replace(/\s+/g, '');
  if (!/^\d+([*+]\d+)+$/.test(flat)) return null;
  let sum = 0;
  for (const term of flat.split('+')) {
    let prod = 1;
    for (const f of term.split('*')) prod *= Number(f);
    sum += prod;
  }
  return Number.isSafeInteger(sum) ? sum : null;
}

export function parseActions(raw) {
  let s = raw.trim();
  if (s.startsWith('```')) s = s.replace(/```json?\n?/g, '').replace(/```\n?/g, '');
  // Prefer the FIRST balanced {...} block that contains "actions": models
  // sometimes append prose ("Wait, ...") or a second JSON attempt after a valid
  // object, which a greedy first-{-to-last-} match would swallow.
  let json = null;
  for (let i = 0; i < s.length && json === null; i++) {
    if (s[i] !== '{') continue;
    let depth = 0, inStr = false, esc = false;
    for (let j = i; j < s.length; j++) {
      const ch = s[j];
      if (esc) { esc = false; continue; }
      if (ch === '\\') { esc = true; continue; }
      if (ch === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (ch === '{') depth++;
      else if (ch === '}' && --depth === 0) {
        const cand = s.slice(i, j + 1);
        if (cand.includes('"actions"')) json = cand;
        break;
      }
    }
  }
  if (json === null) {
    const m = s.match(/\{[\s\S]*\}/);
    if (!m) throw new Error('no JSON object in reply');
    json = m[0];
  }
  let obj;
  try {
    obj = JSON.parse(json);
  } catch (first) {
    // Fallback for format hiccups that are not design errors: (a) arithmetic in
    // a value position, e.g. "inFeatures": 64 * 10 * 10 (Groq/Llama & Claude on
    // conv specs) -> evaluate to the intended integer so the grader still judges
    // the flatten dim; (b) a dropped comma between properties. Only after a parse
    // failure, never touching strings or valid JSON.
    let repaired = json.replace(
      /(?<=[:[,]\s*)\d+(?:\s*[*+]\s*\d+)+(?=\s*[,\]}])/g,
      (expr) => { const v = evalIntExpr(expr); return v === null ? expr : String(v); });
    repaired = repaired.replace(
      /("(?:[^"\\]|\\.)*")(\s+)("(?:[^"\\]|\\.)*"\s*:)/g, '$1,$2$3');
    try {
      obj = JSON.parse(repaired);
    } catch {
      const msg = first instanceof Error ? first.message : String(first);
      const pos = Number((msg.match(/position (\d+)/) || [])[1] ?? -1);
      const near = pos >= 0 ? json.slice(Math.max(0, pos - 50), pos + 50).replace(/\n/g, '\\n') : json.slice(0, 120);
      throw new Error(`bad JSON (${msg}); near >>>${near}<<<`);
    }
  }
  if (!Array.isArray(obj.actions)) throw new Error('reply has no "actions" array');
  return obj.actions;
}

/** Providers whose API key is present (oracle excluded). */
export function runnableProviders(requested) {
  return requested.filter(p => {
    const spec = REGISTRY[p];
    if (!spec) { console.error(`unknown provider "${p}" — skipping`); return false; }
    if (spec.oracle) return true;
    if (!process.env[spec.envKey]) { console.error(`${spec.envKey} not set — skipping ${p}`); return false; }
    return true;
  });
}
