// The two client wires and the one upstream wire.
//
// The trial's client calls this entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. The entry dials the grader's gateway, an
// OpenAI-compatible endpoint, so every prompt goes upstream as a Chat
// Completions request and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id)
//     to the gateway and relays the gateway's JSON or SSE answer, with every
//     `model` field set to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system
//     prompt, image blocks, tool definitions, tool results and tool choice
//     included) and rewrites the gateway's answer into a Messages object, or
//     into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the
//     stop reason and the usage translated. Tool calls become tool_use blocks
//     with their arguments collected whole before the block is emitted.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really
//   answered, so a response's `model` must be the model this entry dialled.
//   OpenRouter's own answer names a free model without the gateway's
//   `openrouter/` prefix; these functions put the dialled id back on every
//   answer and every streamed chunk. They never make a model up: the id came
//   from the brief and went to the gateway in the request's `model`.
//
// Everything here is a pure function of its arguments, except the stream
// readers: no clock and no randomness, so the same body and the same gateway
// answer give the same bytes on a replay. The two model listings use fixed
// timestamps for the same reason: a `created` that moved between runs would
// be a nondeterministic document.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ListedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the OpenAI list shape. */
export function openAIModelList(models: readonly ListedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_CREATED,
      owned_by: m.provider,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request: Anthropic's list shape. */
export function anthropicModelList(models: readonly ListedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.displayName,
      created_at: FIXED_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

/** The text of a string, or of a block array's text blocks; '' when none. */
function plainText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) =>
      isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [],
    )
    .join('\n');
}

/** One Anthropic content block as an OpenAI content part, or null. */
function openAIPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { type: 'text', text: block.text };
  }
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = plainText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {}. */
function argumentsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function finiteNumber(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** An OpenAI finish_reason as an Anthropic stop_reason. */
export function finishToStop(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the top-level system prompt becomes a system message, image
 * blocks become image_url data URLs, tool_use blocks become tool_calls,
 * tool_result blocks become tool messages, tools and tool_choice are
 * rewritten, and the output cap, temperature, top_p, stop sequences and
 * streaming pass through.
 *
 * The gateway refuses a request with no output cap, so a request that names
 * none gets the default 1024 (never the 400 a bare default would imply).
 * Temperature is clamped to Anthropic's 0-1, and `top_k` is dropped: the
 * gateway's wire has no such field. Consecutive messages of one role are
 * merged, because the upstream wire wants strict user/assistant alternation,
 * and an empty content array becomes an empty string rather than a dropped
 * turn.
 */
export function anthropicToChat(body: Json, model: string): Json {
  const messages: Json[] = [];
  const pushMessage = (message: Json): void => {
    const previous = messages[messages.length - 1];
    if (
      previous !== undefined &&
      previous.role === message.role &&
      typeof previous.content === 'string' &&
      typeof message.content === 'string'
    ) {
      previous.content = previous.content === '' ? message.content : `${previous.content}\n${message.content}`;
      return;
    }
    messages.push(message);
  };
  const system = plainText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      pushMessage({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = plainText(content);
      pushMessage({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    let sawToolResult = false;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        sawToolResult = true;
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) pushMessage({ role: 'user', content: parts });
    else if (!sawToolResult) pushMessage({ role: 'user', content: '' });
  }

  const out: Json = { model, messages };
  const maxTokens =
    typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)
      ? Math.min(Math.max(1, Math.floor(body.max_tokens)), 8192)
      : 1024;
  out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') {
    out.temperature = Math.min(Math.max(0, body.temperature), 1);
  }
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    out.stop = body.stop_sequences;
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema)
              ? raw.input_schema
              : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.disable_parallel_tool_use === true) out.parallel_tool_calls = false;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function chatToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : plainText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsObject(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: finishToStop(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: finiteNumber(usage.prompt_tokens),
      output_tokens: finiteNumber(usage.completion_tokens),
    },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled. */
export function chatJsonWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

/** The assistant message's text in a Chat Completions answer, '' when none. */
export function chatText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  return plainText(message.content);
}

/** The finish reason of a Chat Completions answer, null when it has none. */
export function chatFinishReason(body: Json): string | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  return typeof choice.finish_reason === 'string' ? choice.finish_reason : null;
}

/** One server-sent event: `event: <name>` when named, then the data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * One SSE frame with every data payload's `model` set to `model`, its blank
 * line kept: comments, `event:` lines and payloads that are not JSON objects
 * pass through untouched. Chat Completions names the model on its chunks;
 * the client sees the id the entry dialled, never the provider's spelling.
 */
export function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON: relay the line as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

/** Split SSE text into frames on the blank line that ends each one. */
export function splitFrames(text: string): string[] {
  const normalized = text.replace(/\r\n/g, '\n');
  const frames: string[] = [];
  for (const part of normalized.split('\n\n')) {
    if (part.trim() !== '') frames.push(part);
  }
  return frames;
}

/** One tool call as the upstream stream delivered it, arguments collected whole. */
export interface CollectedToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** A Chat Completions stream, collected: its text, tool calls, stop reason and usage. */
export interface CollectedCompletion {
  text: string;
  toolCalls: CollectedToolCall[];
  finishReason: string | null;
  inputTokens: number;
  outputTokens: number;
  sawFinish: boolean;
  sawDone: boolean;
  /** The raw SSE frames as they arrived, each without its trailing blank line. */
  frames: string[];
}

/**
 * Read a Chat Completions SSE body to the end and collect what it carried:
 * the text deltas joined, each tool call with its fragmented arguments
 * joined whole, the finish reason, and the usage from the final chunk. The
 * arguments are collected before any Anthropic tool_use block is emitted
 * from them, so a block never carries a partial JSON string.
 */
export async function collectChatStream(
  upstream: ReadableStream<Uint8Array>,
): Promise<CollectedCompletion> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  const toolCalls: CollectedToolCall[] = [];
  let finishReason: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let sawFinish = false;
  let sawDone = false;
  const frames: string[] = [];

  const handleFrame = (frame: string): void => {
    if (frame.trim() !== '') frames.push(frame);
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        sawDone = true;
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = finiteNumber(chunk.usage.prompt_tokens);
        outputTokens = finiteNumber(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') text += delta.content;
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolCalls.length;
        let slot = toolCalls[upstreamIndex];
        if (slot === undefined) {
          slot = { id: '', name: '', arguments: '' };
          toolCalls[upstreamIndex] = slot;
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof call.id === 'string' && call.id !== '') slot.id = call.id;
        if (typeof fn.name === 'string' && fn.name !== '') slot.name = fn.name;
        if (typeof fn.arguments === 'string') slot.arguments += fn.arguments;
      }
      if (typeof choice.finish_reason === 'string') {
        finishReason = choice.finish_reason;
        sawFinish = true;
      }
    }
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        handleFrame(frame);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') handleFrame(buffer);
  } finally {
    reader.releaseLock();
  }
  return {
    text,
    toolCalls: toolCalls.filter((c) => c !== undefined),
    finishReason,
    inputTokens,
    outputTokens,
    sawFinish,
    sawDone,
    frames,
  };
}

/**
 * A collected Chat Completions answer as the Anthropic event sequence:
 * message_start (with the real input usage) first, one content block per
 * text span and tool call, then message_delta (with the real output usage
 * and the stop reason) and message_stop. The input usage rides on
 * message_start and the output usage on message_delta, as the wire expects.
 */
export function anthropicEventsFor(done: CollectedCompletion, model: string): string[] {
  const events: string[] = [];
  const stopReason = finishToStop(done.finishReason);
  const outputTokens = done.outputTokens > 0 ? done.outputTokens : done.text === '' && done.toolCalls.length === 0 ? 0 : 1;
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: done.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (done.text !== '') {
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'text', text: '' },
      }),
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'text_delta', text: done.text },
      }),
      sseEvent('content_block_stop', { type: 'content_block_stop', index }),
    );
    index += 1;
  }
  for (const call of done.toolCalls) {
    const callIndex = index;
    index += 1;
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: callIndex,
        content_block: {
          type: 'tool_use',
          id: call.id !== '' ? call.id : `toolu_${callIndex}`,
          name: call.name,
          input: {},
        },
      }),
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: callIndex,
        delta: { type: 'input_json_delta', partial_json: call.arguments },
      }),
      sseEvent('content_block_stop', { type: 'content_block_stop', index: callIndex }),
    );
  }
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return events;
}

/**
 * A collected Chat Completions stream as OpenAI SSE frames, every data
 * payload's `model` the id the entry dialled, with a synthetic finish chunk
 * and `[DONE]` appended when the upstream body lacked them. The frames were
 * collected before any byte reached the client, so a stream that failed its
 * token check never started: failover stays clean.
 */
export function openAIStreamFrames(done: CollectedCompletion, model: string): string[] {
  const out: string[] = done.frames.map((frame) => frameWithModel(frame, model));
  if (!done.sawFinish) {
    const finishChunk: Json = {
      id: 'chatcmpl-relay',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
    };
    if (done.outputTokens > 0 || done.inputTokens > 0) {
      finishChunk.usage = {
        prompt_tokens: done.inputTokens,
        completion_tokens: done.outputTokens,
        total_tokens: done.inputTokens + done.outputTokens,
      };
    }
    out.push(`data: ${JSON.stringify(finishChunk)}\n\n`);
  }
  if (!done.sawDone) out.push('data: [DONE]\n\n');
  return out;
}
