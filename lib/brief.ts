// The trial brief: the free model list and the prompt manifest it carries.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) is prose with two JSON
// documents embedded in it. This module finds those documents by their
// `schema` member, never by parsing the prose around them, and turns them
// into the tables the server routes on:
//
//   - the free models (`free-model-aggregator-02/models/v1`): one row per
//     model the gateway routes, keyed by its dialable id
//     (`openrouter/<vendor>/<model>:free`), with the house's published
//     rating in milli-points, what it can do (tools, vision, JSON mode,
//     context length) and its free flag. Only a free model may serve a
//     prompt; a paid row is never routed.
//   - the six prompts (`free-model-aggregator-02/prompts/v1`): the prompts
//     the env will send, in order, each with its id, its api, whether it
//     streams, its weight and the requirements a model must meet to serve
//     it. The server reads the requirements off each arriving request body
//     instead, so the manifest is kept only as a cross-check.
//
// Everything here is deterministic: no clock and no randomness, so the
// catalog `--list` prints and the register body carries are the same
// document, byte for byte, on a live pass and on its replay.

import { readFileSync } from 'node:fs';

/** The schema prefix every document in this contest's brief names. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';
/** The schema of the free model list document. */
export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
/** The schema of the prompt manifest document. */
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** The public base URL of the provider the free models belong to. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under. */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

/** A JSON object, the only shape this module treats as a record. */
export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface FreeModel {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  displayName: string;
  /** The published rating in milli-points: the quality column's currency. */
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs from a model: a model without it cannot serve it. */
export interface Need {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  /** The request's estimated tokens: a model whose context is shorter cannot hold it. */
  minContext?: number;
}

/** One prompt the env will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Need;
}

/** The brief as this module reads it. `error` is set when a document is missing. */
export interface Brief {
  models: FreeModel[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** The catalog document `--list` prints and the register body carries. */
export interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: CatalogModel[];
  }>;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function closingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function schemaDocuments(text: string): Json[] {
  const out: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closingBrace(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a JSON object on its own: look on.
    }
  }
  return out;
}

function textOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this contest's prefix that carries one of `keys`
 * as an array. A brief that spells its lists under a sibling's names still
 * reads, because the document is found by the arrays it carries.
 */
function documentWithArray(
  docs: readonly Json[],
  keys: readonly string[],
): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!textOf(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider
 * that names it: a row that gives the bare OpenRouter id
 * (`vendor/model:free`) and names `openrouter` is routed as the gateway
 * spells it (`openrouter/vendor/model:free`). A row is free unless it says
 * `free: false`; a paid row is never routed.
 */
function modelRow(v: unknown): FreeModel | null {
  if (!isRecord(v)) return null;
  const raw = textOf(v.id) !== '' ? textOf(v.id) : textOf(v.model);
  if (raw === '') return null;
  const named = textOf(v.provider) !== '' ? textOf(v.provider) : textOf(v.wire);
  const id =
    named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/')
      ? `openrouter/${raw}`
      : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    displayName: textOf(v.display_name) !== '' ? textOf(v.display_name) : raw,
    ratingMilli: numOrNull(v.rating_milli) ?? 0,
    contextLength: numOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptRow(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = textOf(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(req.tools === true ? { tools: true } : {}),
      ...(req.vision === true ? { vision: true } : {}),
      ...(req.json === true ? { json: true } : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's prefix. */
export function readBrief(path: string): Brief {
  if (path === '') {
    return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelsDoc = documentWithArray(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelRow(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = documentWithArray(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptRow(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in brief order. */
export function freeModels(brief: Brief): FreeModel[] {
  return brief.models.filter((m) => m.free);
}

/** Whether a model's capabilities meet one prompt's needs. Unknown never meets one. */
export function meetsNeed(need: Need, model: FreeModel): boolean {
  if (need.tools === true && model.tools !== true) return false;
  if (need.vision === true && model.vision !== true) return false;
  if (need.json === true && model.json !== true) return false;
  // A context length the brief did not publish cannot be checked, so it does
  // not exclude a model; a published length shorter than the request does.
  if (
    need.minContext !== undefined &&
    model.contextLength !== null &&
    model.contextLength < need.minContext
  ) {
    return false;
  }
  return true;
}

/**
 * The free models that can serve a need, best first: the house's rating in
 * milli-points, highest first, the id breaking a tie so the order is the
 * same on every run. The first entry is the strongest default for the need;
 * the rest are the failover order.
 */
export function rankedCandidates(models: readonly FreeModel[], need: Need): FreeModel[] {
  return models
    .filter((m) => m.free && meetsNeed(need, m))
    .sort((a, b) => {
      if (b.ratingMilli !== a.ratingMilli) return b.ratingMilli - a.ratingMilli;
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });
}

/**
 * The catalog the brief's free model list makes: one provider, the models it
 * serves, each once, in brief order.
 *
 * The base_url is the provider's public endpoint, never the gateway's
 * trial-local address: the gateway's origin differs between a live pass and
 * its replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte.
 */
export function buildCatalog(brief: Brief): Catalog {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.contextLength });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
