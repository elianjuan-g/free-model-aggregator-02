# free-model-aggregator

A lightweight OpenAI- and Anthropic-compatible model aggregator that serves
prompts through the strongest free models on OpenRouter, with clean failover
when a model refuses.

## What it does

- Serves `POST /v1/chat/completions` (OpenAI, JSON + streaming) and
  `POST /v1/messages` (Anthropic, JSON + streaming), translating Anthropic
  requests to OpenAI Chat Completions and the answers back.
- Serves `GET /v1/models` in OpenAI shape for a Bearer token and in
  Anthropic shape for an `x-api-key`.
- Registers its port and its 14-model catalog with the trial environment and
  polls `/work` until the trial is done, then exits 0.
- `node main.ts --list` prints the same catalog that is registered.

## Model selection

The brief's house ratings rank the 14 free models. Each prompt is served by
the strongest model that meets what the prompt needs:

- General prompt: the highest-rated free model.
- JSON-mode prompt: the highest-rated model that supports JSON mode (the
  top-rated model does not), so the answer comes back as real JSON.
- Tools / vision: the highest-rated model that supports the capability.

The `model` field always names the model that actually answered, never a
fallback labelled as the requested one. Real upstream token counts are
reported; nothing is fabricated.

## Failover

At most 3 model calls per prompt, under one shared 25-second deadline:

- 429 or a dropped connection: retry the same model once after 1.5 s.
- 5xx or 400: fail over at once to the next strongest model that meets the
  prompt.
- 401: give up immediately — every model would refuse without a key.

Every real upstream refusal is reported in the
`x-makefun-upstream-failures` header (JSON array, at most 8 rows of
`{status, code, model}`), on served-after-recovery answers and on give-ups.

## Running

Requires Node 22+. No dependencies, no build step.

```sh
# Print the catalog (needs the brief):
MAKEFUN_PROMPT_FILE=/path/to/prompt.txt node main.ts --list

# Run against a trial environment:
MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
MAKEFUN_PROMPT_FILE=/path/to/prompt.txt \
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENROUTER_API_KEY=<your key> \
node main.ts
```

Environment variables:

| Variable | Meaning |
| --- | --- |
| `MAKEFUN_ENV_URL` | Trial environment base URL (register + work polling). |
| `MAKEFUN_PROMPT_FILE` | Path to the trial brief (`prompt.txt`): the source of the 14-model catalog and the house ratings. Without it the catalog is empty. |
| `OPENAI_BASE_URL` | Gateway base URL for model calls (default `https://openrouter.ai/api/v1`). |
| `OPENROUTER_API_KEY` | Key sent as `Authorization: Bearer …` on model calls. Never commit a key. |

## Files

- `main.ts` — entrypoint: startup, `--list`, HTTP server, register, poll,
  model selection and failover.
- `lib/brief.ts` — brief parser, model table, catalog builder, candidate
  ranking.
- `lib/translate.ts` — Anthropic/OpenAI translation, SSE relay and Anthropic
  event synthesis, error and model-list shapes.
