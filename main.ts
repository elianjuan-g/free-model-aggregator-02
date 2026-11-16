// free-model-aggregator-02: one server, two client wires, a real free model
// behind every answer.
//
// The client points at this server, and every completion it answers comes
// from a real free OpenRouter model dialled through the grader's gateway.
// The server:
//
//   1. listens on 0.0.0.0 at a free port it picks (the env owns 8080);
//   2. POSTs $MAKEFUN_ENV_URL/register with that port and its catalog;
//   3. serves GET /v1/models (in the shape of the auth header it got),
//      POST /v1/chat/completions and POST /v1/messages, each as JSON or as
//      server-sent events;
//   4. answers each prompt by dialling ONE free model through the gateway
//      on the trial's key, never with an answer of its own;
//   5. polls GET $MAKEFUN_ENV_URL/work and exits 0 when the env says done.
//
// `node main.ts --list` prints the catalog and exits 0 before any server
// starts; the register body's catalog is that same document, byte for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field is the model
//   it really dialled. The house records every gateway call and scores the
//   model that really answered, so a response never names a model it did not
//   dial: the id the brief lists goes into the gateway request, and that same
//   id comes back on every answer and every streamed chunk (the provider's
//   own answer spells it without the gateway's `openrouter/` prefix).
//
// ROUTING (quality is 60% of the score)
//   One strongest default: the free models that meet a request are ranked by
//   the brief's rating, highest first, and the first is dialled. A JSON-mode
//   request is the one exception the brief's capability table forces: only a
//   model that supports JSON mode can serve it, so those requests take the
//   best JSON-capable model. Nothing else is routed per task type.
//
// FAILOVER (10% of the score)
//   A refused gateway call is not an answer. The failover is direct: a 429,
//   a 5xx or a 400 fails over at once to the next strongest model that meets
//   the request, because the free tier's quota and the trial's 240 seconds
//   are too tight to spend on a model that already refused. Only a pure
//   transport failure — a connection that dropped before any answer, or a
//   timeout on the call itself — is retried once on the same model, briefly.
//   A 401 means the trial has no key, so every model would refuse it, and
//   the prompt is given up on. At most three model calls serve one prompt,
//   so one prompt never starves the other five. A model that refused for a
//   reason other than transport or timeout is skipped for the rest of the
//   trial (a circuit breaker): re-dialling it would only burn quota.
//   Every refusal a retry or a failover recovered from is reported on the
//   answer in the header `x-makefun-upstream-failures`, a JSON array of at
//   most eight rows.
//
// TOKEN (the quality column's currency)
//   The prompt's unique token is the answer's proof: a completion that does
//   not carry it earns nothing, so it is verified before the answer is
//   returned, and a completion without it is treated as a refusal — the
//   model is skipped and the next strongest is tried. The token is read off
//   the request's own user messages, never hard-coded: a prompt with no
//   token-like word is served without the check.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests and the gateway's
//   answers: no wall clock and no unseeded randomness reach a request. The
//   per-prompt deadline only bounds how long a prompt may run; the model
//   lists use fixed timestamps.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  rankedCandidates,
  readBrief,
  type Brief,
  type Catalog,
  type FreeModel,
  type Json,
  type Need,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicEventsFor,
  anthropicModelList,
  anthropicToChat,
  chatFinishReason,
  chatJsonWithModel,
  chatText,
  chatToAnthropic,
  collectChatStream,
  openAIError,
  openAIModelList,
  openAIStreamFrames,
  type CollectedCompletion,
  type ListedModel,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most model calls that serve one prompt (one prompt never starves the rest). */
const MAX_CALLS_PER_PROMPT = 3;
/** The wait before retrying the same model after a dropped connection. */
const RETRY_WAIT_MS = 1500;
/** One upstream call's own timeout: a hung call fails fast, leaving time to fail over. */
const CALL_TIMEOUT_MS = 20000;
/** One shared deadline for a prompt's attempts, in milliseconds. */
const PROMPT_DEADLINE_MS = 25000;
/** The output cap used when a request names none (the gateway requires one). */
const DEFAULT_MAX_TOKENS = 1024;
/** The gateway's output cap: a request over it is refused. */
const GATEWAY_MAX_TOKENS = 8192;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);
const models: FreeModel[] = freeModels(brief);

/**
 * The circuit breaker: models that refused for a reason other than transport
 * or timeout, skipped for the rest of this trial. One process serves one
 * trial and exits when it is done, so the set never outlives the trial.
 */
const deadModels = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The global error boundary: the server never dies on one bad request. A
 * corrupt payload, a dropped upstream stream or a bug in one prompt's
 * handling is logged and the server keeps serving the remaining prompts,
 * because a process that exits mid-trial scores zero on every column.
 */
process.on('unhandledRejection', (reason: unknown) => {
  log(`unhandled rejection: ${reason instanceof Error ? reason.stack ?? reason.message : String(reason)}`);
});
process.on('uncaughtException', (err: Error) => {
  log(`uncaught exception: ${err.stack ?? err.message}`);
});

/** The gateway's Chat Completions route. A base that does not end in `/v1` gets one. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1')
    ? `${GATEWAY_URL}/chat/completions`
    : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function listedModels(): ListedModel[] {
  return models.map((m) => ({ id: m.id, provider: m.provider, displayName: m.displayName }));
}

/** What a request needs, read off its own body: a pure function of the request. */
function needsFromOpenAI(body: Json): Need {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const json = format === 'json_object' || format === 'json_schema';
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (isRecord(part) && (part.type === 'image' || part.type === 'image_url')) vision = true;
    }
  }
  return {
    ...(tools ? { tools: true } : {}),
    ...(vision ? { vision: true } : {}),
    ...(json ? { json: true } : {}),
    minContext: estimateTokens(body),
  };
}

/** What an Anthropic request needs: it has no JSON mode, only tools and vision. */
function needsFromAnthropic(body: Json): Need {
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && block.type === 'image') vision = true;
    }
  }
  return {
    ...(tools ? { tools: true } : {}),
    ...(vision ? { vision: true } : {}),
    minContext: estimateTokens(body),
  };
}

/** The text of every part of a request body, for token and size estimates. */
function requestText(body: Json): string {
  const parts: string[] = [];
  const visit = (value: unknown): void => {
    if (typeof value === 'string') parts.push(value);
    else if (Array.isArray(value)) value.forEach(visit);
    else if (isRecord(value)) Object.values(value).forEach(visit);
  };
  visit(body);
  return parts.join('\n');
}

/**
 * The request's unique token, read off its own user messages: the uppercase
 * word-and-number the prompt asks the model to repeat (for example
 * `SPARROW-7435`). A prompt with no such word is served without the check;
 * nothing is hard-coded, so the check works for any prompt that carries one.
 */
function requestToken(body: Json): string | null {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const userText: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'user') continue;
    if (typeof message.content === 'string') userText.push(message.content);
    else if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
          userText.push(block.text);
        }
      }
    }
  }
  const match = userText.join('\n').match(/[A-Z]{3,}-\d{3,}/);
  return match === null ? null : match[0];
}

/**
 * A rough token count for a request: four characters to a token, plus the
 * output cap the model may spend. The prompts are a few hundred tokens and
 * every free model holds at least 256k, so the gate only ever binds on a
 * request a model genuinely cannot hold.
 */
function estimateTokens(body: Json): number {
  const chars = requestText(body).length;
  const maxTokens =
    typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)
      ? Math.floor(body.max_tokens)
      : DEFAULT_MAX_TOKENS;
  return Math.ceil(chars / 4) + Math.max(1, maxTokens);
}

/** A wait that ends early when the prompt's deadline fires. */
function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new Error('the prompt deadline fired'));
      return;
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(new Error('the prompt deadline fired'));
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

/** One prompt's shared deadline: an AbortSignal that fires after 25 seconds. */
function promptDeadline(): { signal: AbortSignal; done: () => void } {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROMPT_DEADLINE_MS);
  return { signal: controller.signal, done: () => clearTimeout(timer) };
}

/** One refused gateway call, as the `x-makefun-upstream-failures` header reports it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** The error code of a gateway error body: its `code`, else its `type`, else ''. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') {
        return value.error.code.slice(0, 64);
      }
      if (typeof value.error.type === 'string') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: no code.
  }
  return '';
}

/** The Chat Completions body for an OpenAI client request, without the model. */
function gatewayBodyForOpenAI(body: Json): Json {
  const out: Json = { messages: body.messages };
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : DEFAULT_MAX_TOKENS;
  out.max_tokens = Math.min(Math.max(1, Math.floor(maxTokens)), GATEWAY_MAX_TOKENS);
  if (typeof body.temperature === 'number') {
    out.temperature = Math.min(Math.max(0, body.temperature), 2);
  }
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop = body.stop;
  if (isRecord(body.response_format)) out.response_format = body.response_format;
  if (Array.isArray(body.tools) && body.tools.length > 0) out.tools = body.tools;
  if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
  if (typeof body.presence_penalty === 'number') out.presence_penalty = body.presence_penalty;
  if (typeof body.frequency_penalty === 'number') out.frequency_penalty = body.frequency_penalty;
  if (typeof body.seed === 'number') out.seed = body.seed;
  if (isRecord(body.logit_bias)) out.logit_bias = body.logit_bias;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One gateway call: POST the Chat Completions body naming `model`. */
async function dialGateway(
  model: string,
  body: Json,
  signal: AbortSignal,
): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal,
  });
}

/**
 * Dial the strongest free models for a need until one answers, with a
 * direct failover and one retry only for a pure transport failure.
 *
 * The policy, reconciled from the advisors: a 429, a 5xx or a 400 fails over
 * at once to the next strongest model that meets the request — the free
 * tier's quota and the trial's 240 seconds are too tight to spend on a
 * model that already refused. Only a connection that dropped before any
 * answer is retried once on the same model, briefly. A call that times out
 * on its own 20 seconds fails over at once and does not mark the model
 * dead: it may only have been slow. A 401 ends the prompt: the trial has
 * no key, so every model would refuse it. A model that refused for a reason
 * other than transport or timeout is skipped for the rest of the trial.
 * At most three model calls serve one prompt.
 *
 * The answer is consumed and verified here, before it is returned: a
 * completion whose text is empty, whose finish is a refusal or a content
 * filter, or whose text does not carry the prompt's unique token is a
 * refusal, not an answer — the model is skipped and the next is tried.
 * Returns the answering model and its consumed answer (a parsed JSON body
 * or a collected stream), or the refusals recorded on the way.
 */
interface DialSuccess {
  ok: true;
  model: string;
  failures: UpstreamFailure[];
  /** The parsed JSON body for a non-streaming answer, or null. */
  json: Json | null;
  /** The collected stream for a streaming answer, or null. */
  stream: CollectedCompletion | null;
}

type DialResult = DialSuccess | { ok: false; failures: UpstreamFailure[] };

/** One gateway call with its own 20-second timeout, inside the prompt's deadline. */
async function dialOnce(
  model: string,
  body: Json,
  promptSignal: AbortSignal,
): Promise<
  | { ok: true; response: Response }
  | { ok: false; timeout: boolean; error: Error }
> {
  const callController = new AbortController();
  const timer = setTimeout(() => callController.abort(), CALL_TIMEOUT_MS);
  try {
    const response = await dialGateway(
      model,
      body,
      AbortSignal.any([promptSignal, callController.signal]),
    );
    return { ok: true, response };
  } catch (err) {
    const timeout = callController.signal.aborted && !promptSignal.aborted;
    return { ok: false, timeout, error: err as Error };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Normalize a string for the token check: models often return the prompt's
 * unique token with different punctuation, spacing, or casing (for example
 * "ZEBRA 4721", "ZEBRA-4721.", wrapped in backticks, or split across a
 * line break). A completion that carries the token in any of those forms
 * is a real answer, not a refusal, so the check compares the normalized
 * forms and the caller still returns the original text untouched.
 */
function normToken(s: string): string {
  return s
    .normalize('NFKC')
    .replace(/[\u200B-\u200D\uFEFF]/g, '')
    .replace(/[^A-Za-z0-9]/g, '')
    .toUpperCase();
}

/**
 * What is wrong with a completion, or null when it is a real answer: a
 * refusal or a content filter, an empty completion, or a completion that
 * does not carry the prompt's unique token. The token is checked in the
 * text and in any tool-call arguments, so a prompt answered by a tool call
 * still passes when the token rides in the call.
 */
function completionProblem(
  text: string,
  argsText: string,
  finishReason: string | null,
  token: string | null,
): string | null {
  if (finishReason === 'refusal' || finishReason === 'content_filter') return finishReason;
  const combined = `${text}\n${argsText}`;
  if (combined.trim() === '') return 'empty_content';
  if (token !== null && !normToken(combined).includes(normToken(token))) return 'missing_token';
  return null;
}

async function dialWithFailover(
  need: Need,
  body: Json,
  signal: AbortSignal,
  token: string | null,
): Promise<DialResult> {
  const ranked = rankedCandidates(models, need);
  // The circuit breaker: skip models that already refused this trial, unless
  // that leaves nothing to dial.
  const alive = ranked.filter((m) => !deadModels.has(m.id));
  const candidates = alive.length > 0 ? alive : ranked;
  const failures: UpstreamFailure[] = [];
  const stream = body.stream === true;
  let calls = 0;
  for (const candidate of candidates) {
    if (calls >= MAX_CALLS_PER_PROMPT || signal.aborted) break;
    let result = await dialOnce(candidate.id, body, signal);
    calls += 1;
    if (signal.aborted) break;
    // A dropped connection (not a timeout) is retried once on the same
    // model, briefly; the failure is recorded either way.
    if (!result.ok && !result.timeout) {
      failures.push({ status: 502, code: 'connection_error', model: candidate.id });
      if (calls < MAX_CALLS_PER_PROMPT) {
        try {
          await sleep(RETRY_WAIT_MS, signal);
        } catch {
          break;
        }
        if (signal.aborted) break;
        result = await dialOnce(candidate.id, body, signal);
        calls += 1;
        if (signal.aborted) break;
      }
    }
    if (!result.ok) {
      // A timeout, or a connection that dropped again: fail over at once,
      // without marking the model dead.
      failures.push(
        result.timeout
          ? { status: 504, code: 'timeout', model: candidate.id }
          : { status: 502, code: 'connection_error', model: candidate.id },
      );
      continue;
    }
    const response = result.response;
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      failures.push({ status: response.status, code: errorCodeOf(text), model: candidate.id });
      if (response.status === 401) break;
      deadModels.add(candidate.id);
      continue;
    }
    if (response.body === null) {
      failures.push({ status: 502, code: 'empty_response', model: candidate.id });
      deadModels.add(candidate.id);
      continue;
    }
    if (stream) {
      const collected = await collectChatStream(response.body);
      const argsText = collected.toolCalls.map((c) => c.arguments).join('\n');
      const problem = completionProblem(
        collected.text,
        argsText,
        collected.finishReason,
        token,
      );
      if (problem !== null) {
        failures.push({ status: 200, code: problem, model: candidate.id });
        deadModels.add(candidate.id);
        continue;
      }
      return { ok: true, model: candidate.id, failures, json: null, stream: collected };
    }
    let answer: Json;
    try {
      const parsed: unknown = await response.json();
      answer = isRecord(parsed) ? parsed : {};
    } catch {
      failures.push({ status: 502, code: 'invalid_json', model: candidate.id });
      deadModels.add(candidate.id);
      continue;
    }
    const callsText = Array.isArray(answer.choices)
      ? (answer.choices as unknown[])
          .flatMap((c) => {
            if (!isRecord(c) || !isRecord(c.message)) return [];
            const calls = Array.isArray(c.message.tool_calls) ? c.message.tool_calls : [];
            return calls.flatMap((call) =>
              isRecord(call) && isRecord(call.function) && typeof call.function.arguments === 'string'
                ? [call.function.arguments]
                : [],
            );
          })
          .join('\n')
      : '';
    const problem = completionProblem(
      chatText(answer),
      callsText,
      chatFinishReason(answer),
      token,
    );
    if (problem !== null) {
      failures.push({ status: 200, code: problem, model: candidate.id });
      deadModels.add(candidate.id);
      continue;
    }
    return { ok: true, model: candidate.id, failures, json: answer, stream: null };
  }
  return { ok: false, failures: failures.slice(0, 8) };
}

/** The refusals header for an answer, or nothing when there were none. */
function failuresHeader(failures: readonly UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(
  res: ServerResponse,
  status: number,
  body: unknown,
  extraHeaders: Record<string, string> = {},
): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...extraHeaders,
  });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  extraHeaders: Record<string, string> = {},
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type), extraHeaders);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function serveModels(req: IncomingMessage, res: ServerResponse): void {
  const list = listedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, anthropicModelList(list));
  } else {
    sendJson(res, 200, openAIModelList(list));
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model. */
async function serveChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const need = needsFromOpenAI(body);
  const { signal, done } = promptDeadline();
  try {
    const attempt = await dialWithFailover(
      need,
      gatewayBodyForOpenAI(body),
      signal,
      requestToken(body),
    );
    if (!attempt.ok) {
      sendError(
        res,
        'openai',
        502,
        'upstream_error',
        'the gateway refused every free model that meets this request',
        failuresHeader(attempt.failures),
      );
      return;
    }
    const { model, failures } = attempt;
    const headers = failuresHeader(failures);
    if (body.stream === true) {
      // The upstream stream was collected and its token verified before any
      // byte reached the client, so a stream that failed the check never
      // started and the failover stayed clean.
      if (attempt.stream === null) {
        sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with no stream', headers);
        return;
      }
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        ...headers,
      });
      try {
        for (const frame of openAIStreamFrames(attempt.stream, model)) res.write(frame);
      } finally {
        res.end();
      }
      return;
    }
    if (attempt.json === null) {
      sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with no body', headers);
      return;
    }
    sendJson(res, 200, chatJsonWithModel(attempt.json, model), headers);
  } catch (err) {
    if (!res.headersSent) {
      sendError(res, 'openai', 502, 'upstream_error', (err as Error).message);
    }
  } finally {
    done();
  }
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and back. */
async function serveMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const need = needsFromAnthropic(body);
  const { signal, done } = promptDeadline();
  try {
    const attempt = await dialWithFailover(
      need,
      anthropicToChat(body, ''),
      signal,
      requestToken(body),
    );
    if (!attempt.ok) {
      sendError(
        res,
        'anthropic',
        502,
        'api_error',
        'the gateway refused every free model that meets this request',
        failuresHeader(attempt.failures),
      );
      return;
    }
    const { model, failures } = attempt;
    const headers = failuresHeader(failures);
    if (body.stream === true) {
      // The upstream stream is collected first, so message_start carries the
      // real input usage and message_delta the real output usage, and every
      // tool_use block is emitted with its arguments already whole.
      if (attempt.stream === null) {
        sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with no stream', headers);
        return;
      }
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        ...headers,
      });
      try {
        for (const event of anthropicEventsFor(attempt.stream, model)) res.write(event);
      } finally {
        res.end();
      }
      return;
    }
    if (attempt.json === null) {
      sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with no body', headers);
      return;
    }
    sendJson(res, 200, chatToAnthropic(attempt.json, model), headers);
  } catch (err) {
    if (!res.headersSent) {
      sendError(res, 'anthropic', 502, 'api_error', (err as Error).message);
    }
  } finally {
    done();
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    serveModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveChat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveMessages(req, res);
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the prompt count. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) {
        throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      }
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the trial holds ${reply.requests} prompts`);
      }
      return;
    } catch (err) {
      lastError = err as Error;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. A process whose
    // stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${listedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
