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
      return (await r.json()).content?.[0]?.text ?? '';
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
      return (await r.json()).candidates?.[0]?.content?.parts?.[0]?.text ?? '';
    },
  },
};

export function parseActions(raw) {
  let s = raw.trim();
  if (s.startsWith('```')) s = s.replace(/```json?\n?/g, '').replace(/```\n?/g, '');
  const m = s.match(/\{[\s\S]*\}/);
  if (!m) throw new Error('no JSON object in reply');
  const obj = JSON.parse(m[0]);
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
