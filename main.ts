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
//   A refused gateway call is not an answer. A 429 (or a dropped connection)
//   is retried once on the same model after a short wait; a 5xx or a 400
//   fails over at once to the next strongest model that meets the request.
//   A 401 means the trial has no key, so every model would refuse it, and
//   the prompt is given up on. At most three model calls serve one prompt,
//   so one prompt never starves the other five. Every refusal a retry or a
//   failover recovered from is reported on the answer in the header
//   `x-makefun-upstream-failures`, a JSON array of at most eight rows.
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
  chatJsonWithModel,
  chatToAnthropic,
  collectChatStream,
  frameWithModel,
  openAIError,
  openAIModelList,
  relayChatStream,
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
/** The wait before retrying the same model after a 429 or a dropped connection. */
const RETRY_WAIT_MS = 1500;
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
  return { ...(tools ? { tools: true } : {}), ...(vision ? { vision: true } : {}), ...(json ? { json: true } : {}) };
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
  return { ...(tools ? { tools: true } : {}), ...(vision ? { vision: true } : {}) };
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
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
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
 * Dial the strongest free models for a need until one answers, with one
 * retry of the same model after a 429 or a dropped connection and a fast
 * failover to the next strongest on a 5xx or a 400. A 401 ends the prompt:
 * the trial has no key, so every model would refuse it. At most three calls
 * serve one prompt. Returns the answering model and its response, or the
 * refusals that were recorded on the way.
 */
async function dialWithFailover(
  need: Need,
  body: Json,
  signal: AbortSignal,
): Promise<
  | { ok: true; model: string; response: Response; failures: UpstreamFailure[] }
  | { ok: false; failures: UpstreamFailure[] }
> {
  const candidates = rankedCandidates(models, need);
  const failures: UpstreamFailure[] = [];
  let calls = 0;
  for (const candidate of candidates) {
    if (calls >= MAX_CALLS_PER_PROMPT) break;
    // The first try of this model.
    let response: Response | null = null;
    let thrown: Error | null = null;
    try {
      calls += 1;
      response = await dialGateway(candidate.id, body, signal);
    } catch (err) {
      thrown = err as Error;
    }
    if (signal.aborted) break;
    if (response !== null && response.ok) {
      return { ok: true, model: candidate.id, response, failures };
    }
    if (response !== null) {
      const text = await response.text().catch(() => '');
      failures.push({ status: response.status, code: errorCodeOf(text), model: candidate.id });
      if (response.status === 401) break;
      // A 429 is retried once below; a 5xx or a 400 fails over at once.
      if (response.status !== 429) continue;
    }
    // A 429, or a connection that dropped before any answer: retry the same
    // model once, briefly, then fail over if it refuses again.
    if (calls >= MAX_CALLS_PER_PROMPT) break;
    try {
      await sleep(RETRY_WAIT_MS, signal);
    } catch {
      break;
    }
    if (signal.aborted) break;
    try {
      calls += 1;
      response = await dialGateway(candidate.id, body, signal);
      thrown = null;
    } catch (err) {
      thrown = err as Error;
      response = null;
    }
    if (signal.aborted) break;
    if (response !== null && response.ok) {
      return { ok: true, model: candidate.id, response, failures };
    }
    if (response !== null) {
      const text = await response.text().catch(() => '');
      failures.push({ status: response.status, code: errorCodeOf(text), model: candidate.id });
      if (response.status === 401) break;
    }
    void thrown;
    // The retry refused too: fail over to the next strongest model.
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
    const attempt = await dialWithFailover(need, gatewayBodyForOpenAI(body), signal);
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
    const { model, response, failures } = attempt;
    const headers = failuresHeader(failures);
    if (response.body === null) {
      sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with no body', headers);
      return;
    }
    if (body.stream === true) {
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        ...headers,
      });
      try {
        for await (const frame of relayChatStream(response.body, model)) res.write(frame);
      } finally {
        res.end();
      }
      return;
    }
    const answer: unknown = await response.json();
    sendJson(res, 200, chatJsonWithModel(isRecord(answer) ? answer : {}, model), headers);
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
    const attempt = await dialWithFailover(need, anthropicToChat(body, ''), signal);
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
    const { model, response, failures } = attempt;
    const headers = failuresHeader(failures);
    if (response.body === null) {
      sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with no body', headers);
      return;
    }
    if (body.stream === true) {
      // The upstream stream is collected first, so message_start carries the
      // real input usage and message_delta the real output usage, and every
      // tool_use block is emitted with its arguments already whole.
      const done2 = await collectChatStream(response.body);
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        ...headers,
      });
      try {
        for (const event of anthropicEventsFor(done2, model)) res.write(event);
      } finally {
        res.end();
      }
      return;
    }
    const answer: unknown = await response.json();
    sendJson(res, 200, chatToAnthropic(isRecord(answer) ? answer : {}, model), headers);
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
