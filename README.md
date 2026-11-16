# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/elianjuan-g/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/elianjuan-g/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A lightweight OpenAI- and Anthropic-compatible model aggregator that serves
prompts through the strongest free models on OpenRouter, with clean failover
when a model refuses. Entry for [make.fun](https://make.fun) bounty: [The best live free-model aggregator](https://make.fun/bounty/free-model-aggregator-02).

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
- Context: a model whose context is shorter than the request's estimated
  tokens cannot hold it, so it is never dialled for that request.

The `model` field always names the model that actually answered, never a
fallback labelled as the requested one. Real upstream token counts are
reported; nothing is fabricated.

## Failover

The failover is direct: a 429, a 5xx or a 400 fails over at once to the next
strongest model that meets the prompt, because the free tier's quota and
the trial's 240 seconds are too tight to spend on a model that already
refused. Only a connection that dropped before any answer is retried once
on the same model, briefly. A call that times out on its own 20 seconds
fails over at once. A 401 ends the prompt at once: without a key every
model would refuse. At most 3 model calls serve one prompt, under one
shared 25-second deadline, so one prompt never starves the other five.

A model that refused for a reason other than transport or timeout is
skipped for the rest of the trial (a circuit breaker): re-dialling it
would only burn quota.

The prompt's unique token is verified before the answer is returned: a
completion that does not carry it is treated as a refusal, and the next
strongest model is tried. The token is read off the request's own user
messages, never hard-coded.

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

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
