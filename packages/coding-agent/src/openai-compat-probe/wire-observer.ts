import type { FetchImpl, RawSseEvent } from "@oh-my-pi/pi-ai/types";
import { compareProbeObjectKeys } from "./ordering";
import type { OpenAICompatProbeWireAttempt } from "./types";

const MAX_ERROR_BODY_BYTES = 16_384;
const MAX_ERROR_STRING_CHARS = 4_096;
const SENSITIVE_PROPERTY_PATTERN =
	"api[_-]?key|access[_-]?token|auth[_-]?token|authorization|proxy[_-]?authorization|token|secret|client[_-]?secret|password|credential(?:s)?|cookie|session(?:[_-]?id)?";
const SENSITIVE_PROPERTY = new RegExp(`^(?:${SENSITIVE_PROPERTY_PATTERN})$`, "i");
const SENSITIVE_JSON_STRING_PROPERTY = new RegExp(
	`("(?:${SENSITIVE_PROPERTY_PATTERN})"\\s*:\\s*)"(?:\\\\.|[^"\\\\])*"`,
	"gi",
);
const EMBEDDED_HTTP_URL = /https?:\/\/[^\s"'<>]+/giu;
const SENSITIVE_AUTH_ASSIGNMENT = /\b(authorization|proxy-authorization)\s*(?:=|:)\s*(?:[^\s,;]+\s+)?[^\s,;]+/giu;
const SENSITIVE_ASSIGNMENT = new RegExp(
	`\\b(${SENSITIVE_PROPERTY_PATTERN})\\s*(?:=|:)\\s*(?:"(?:\\\\.|[^"\\\\])*"|'(?:\\\\.|[^'\\\\])*'|[^\\s,;]+)`,
	"gi",
);
const BEARER_CREDENTIAL = /\bBearer\s+[^\s,;]+/giu;
const SAFE_ERROR_PROPERTIES: Record<string, true> = {
	code: true,
	detail: true,
	details: true,
	error: true,
	errors: true,
	message: true,
	name: true,
	param: true,
	reason: true,
	request_id: true,
	status: true,
	title: true,
	type: true,
};
const REASONING_FIELDS: Record<string, true> = {
	reasoning: true,
	reasoning_content: true,
	reasoning_text: true,
	reasoning_effort: true,
	thinking: true,
	enable_thinking: true,
	chat_template_kwargs: true,
};
const STRUCTURAL_GENERATED_PROPERTIES: Record<string, true> = {
	role: true,
	type: true,
	name: true,
	index: true,
	status: true,
};

interface AttemptObservation {
	rawBody?: unknown;
	startedAt: number;
	lastEventAt?: number;
	firstEventAt?: number;
	maxInterEventGapMs: number;
	rawUsageObserved: boolean;
	reasoningFields: Set<string>;
	reasoningDeltas: string[];
	contentDeltas: string[];
	toolCallObserved: boolean;
	finishReason?: string;
}

export interface OpenAICompatGeneratedDeltaEvidence {
	redacted: true;
	length: number;
}

export interface OpenAICompatRawWireDeltas {
	reasoningDeltas: readonly string[];
	contentDeltas: readonly string[];
	reasoningDeltaSequences: readonly (readonly string[])[];
}

const rawWireDeltas = new WeakMap<OpenAICompatWireEvidence, OpenAICompatRawWireDeltas>();
const rawRequestBodies = new WeakMap<OpenAICompatProbeWireAttempt, unknown>();

function summarizeGeneratedDelta(value: string): OpenAICompatGeneratedDeltaEvidence {
	return { redacted: true, length: value.length };
}

export function getOpenAICompatRawWireDeltas(evidence: OpenAICompatWireEvidence): OpenAICompatRawWireDeltas {
	const raw = rawWireDeltas.get(evidence);
	if (!raw) throw new Error("Raw OpenAI compatibility wire deltas are unavailable");
	return raw;
}
export function getOpenAICompatRawRequestBody(attempt: OpenAICompatProbeWireAttempt): unknown {
	if (!rawRequestBodies.has(attempt)) throw new Error("Raw OpenAI compatibility request body is unavailable");
	return rawRequestBodies.get(attempt);
}

export interface OpenAICompatWireEvidence {
	rawUsageObserved: boolean;
	reasoningFields: string[];
	reasoningDeltas: OpenAICompatGeneratedDeltaEvidence[];
	contentDeltas: OpenAICompatGeneratedDeltaEvidence[];
	reasoningDeltaSequences: OpenAICompatGeneratedDeltaEvidence[][];
	toolCallObserved: boolean;
	timeToFirstEventMs?: number;
	maxInterEventGapMs: number;
}

export interface OpenAICompatWireObserver {
	fetch: FetchImpl;
	onPayload(payload: unknown): void;
	onSseEvent(event: RawSseEvent): void;
	attempts: OpenAICompatProbeWireAttempt[];
	evidence(): OpenAICompatWireEvidence;
}

function replaceApiKey(value: string, apiKey: string): string {
	const redacted = apiKey.length > 0 ? value.replaceAll(apiKey, "[REDACTED]") : value;
	return redacted.length <= MAX_ERROR_STRING_CHARS ? redacted : `${redacted.slice(0, MAX_ERROR_STRING_CHARS)}…`;
}

async function sha256(value: string): Promise<string> {
	const bytes = new TextEncoder().encode(value);
	const digest = await crypto.subtle.digest("SHA-256", bytes);
	return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
}

function headersRecord(headers: RequestInit["headers"] | Headers | undefined): Record<string, string> {
	const pairs = Array.from(new Headers(headers).entries()).sort(([left], [right]) =>
		compareProbeObjectKeys(left, right),
	);
	const result: Record<string, string> = {};
	for (const [key] of pairs) result[key] = "[REDACTED]";
	return result;
}

export function sanitizeProbeUrl(value: string, apiKey = ""): string {
	try {
		const url = new URL(value);
		url.username = "";
		url.password = "";
		if (url.protocol !== "http:" && url.protocol !== "https:") return "[INVALID_URL_REDACTED]";
		for (const key of Array.from(url.searchParams.keys())) url.searchParams.set(key, "[REDACTED]");
		if (url.hash.length > 0) url.hash = "[REDACTED]";
		return replaceApiKey(url.toString(), apiKey);
	} catch {
		return "[INVALID_URL_REDACTED]";
	}
}

export function sanitizeProbeErrorMessage(value: string, apiKey = ""): string {
	const sanitized = value
		.replace(EMBEDDED_HTTP_URL, url => sanitizeProbeUrl(url, apiKey))
		.replace(SENSITIVE_JSON_STRING_PROPERTY, '$1"[REDACTED]"')
		.replace(BEARER_CREDENTIAL, "Bearer [REDACTED]")
		.replace(SENSITIVE_AUTH_ASSIGNMENT, "$1=[REDACTED]")
		.replace(SENSITIVE_ASSIGNMENT, "$1=[REDACTED]");
	return replaceApiKey(sanitized, apiKey);
}

export function assertCredentialFreeProbeValue(value: unknown, apiKey: string, label: string): void {
	const seen = new WeakSet<object>();
	const visit = (current: unknown, path: string): void => {
		if (typeof current === "string") {
			if (apiKey.length > 0 && current.includes(apiKey)) {
				throw new Error(`${label} contains an API credential at ${path}`);
			}
			return;
		}
		if (typeof current !== "object" || current === null) return;
		if (seen.has(current)) return;
		seen.add(current);
		if (Array.isArray(current)) {
			for (let index = 0; index < current.length; index += 1) visit(current[index], `${path}[${index}]`);
			return;
		}
		for (const [key, child] of Object.entries(current)) {
			if (apiKey.length > 0 && key.includes(apiKey)) {
				throw new Error(`${label} contains the API credential in a property name`);
			}
			const childPath = path.length > 0 ? `${path}.${key}` : key;
			if (SENSITIVE_PROPERTY.test(key)) {
				throw new Error(`${label} contains a credential-like property at ${childPath}`);
			}
			visit(child, childPath);
		}
	};
	visit(value, "");
}

async function sanitizeValue(
	value: unknown,
	apiKey: string,
	generatedContent = false,
	privateStrings: ReadonlySet<string> = new Set(),
): Promise<unknown> {
	if (typeof value === "string") {
		const safe = privateStrings.has(value) ? "[REDACTED]" : sanitizeProbeErrorMessage(value, apiKey);
		if (!generatedContent) return safe;
		return { redacted: true, length: safe.length, sha256: await sha256(safe) };
	}
	if (Array.isArray(value)) {
		return Promise.all(value.map(item => sanitizeValue(item, apiKey, generatedContent, privateStrings)));
	}
	if (typeof value !== "object" || value === null) return value;
	const source = value as Record<string, unknown>;
	const result: Record<string, unknown> = {};
	for (const key of Object.keys(source).sort()) {
		if (SENSITIVE_PROPERTY.test(key)) {
			result[key] = "[REDACTED]";
			continue;
		}
		result[key] = await sanitizeValue(
			source[key],
			apiKey,
			generatedContent && STRUCTURAL_GENERATED_PROPERTIES[key] !== true,
			privateStrings,
		);
	}
	return result;
}

async function sanitizeRequestBody(
	value: unknown,
	apiKey: string,
	privateStrings: ReadonlySet<string>,
): Promise<unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		return sanitizeValue(value, apiKey, false, privateStrings);
	}
	const source = value as Record<string, unknown>;
	const result = (await sanitizeValue(source, apiKey, false, privateStrings)) as Record<string, unknown>;
	if (!Array.isArray(source.messages) || !Array.isArray(result.messages)) return result;

	for (let index = 0; index < source.messages.length; index += 1) {
		const rawMessage = source.messages[index];
		const safeMessage = result.messages[index];
		if (typeof rawMessage !== "object" || rawMessage === null || Array.isArray(rawMessage)) continue;
		if (typeof safeMessage !== "object" || safeMessage === null || Array.isArray(safeMessage)) continue;
		const raw = rawMessage as Record<string, unknown>;
		const safe = safeMessage as Record<string, unknown>;
		if (raw.role === "assistant") {
			for (const key of [
				"content",
				"reasoning",
				"reasoning_content",
				"reasoning_text",
				"refusal",
				"tool_calls",
				"function_call",
			] as const) {
				if (raw[key] !== undefined) safe[key] = await sanitizeValue(raw[key], apiKey, true, privateStrings);
			}
		} else if (raw.role === "tool" && raw.tool_call_id !== undefined) {
			safe.tool_call_id = await sanitizeValue(raw.tool_call_id, apiKey, true, privateStrings);
		}
	}
	return result;
}

function parseBody(body: RequestInit["body"] | null | undefined): unknown {
	if (typeof body !== "string") return body === undefined || body === null ? undefined : "[NON_JSON_BODY]";
	try {
		return JSON.parse(body) as unknown;
	} catch {
		return body;
	}
}

function countStrictFields(value: unknown): number {
	if (Array.isArray(value)) return value.reduce((total, child) => total + countStrictFields(child), 0);
	if (typeof value !== "object" || value === null) return 0;
	let count = 0;
	for (const [key, child] of Object.entries(value)) {
		if (key === "strict" && child === true) count += 1;
		count += countStrictFields(child);
	}
	return count;
}

function reasoningShape(value: unknown): string {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return "";
	const source = value as Record<string, unknown>;
	const shape: Record<string, unknown> = {};
	for (const key of Object.keys(REASONING_FIELDS)) {
		if (source[key] !== undefined) shape[key] = source[key];
	}
	return JSON.stringify(shape);
}

function classifyAttempt(
	attempts: readonly OpenAICompatProbeWireAttempt[],
	observations: readonly AttemptObservation[],
	rawBody: unknown,
): OpenAICompatProbeWireAttempt["classification"] {
	if (attempts.length === 0) return "first";
	const previousAttempt = attempts.at(-1);
	const previousObservation = observations.at(-1);
	if (!previousAttempt || !previousObservation) return "transport_retry";
	const previousStatus = previousAttempt.response?.status;
	if (previousStatus === undefined || previousStatus === 408 || previousStatus === 429 || previousStatus >= 500) {
		return "transport_retry";
	}
	if (countStrictFields(rawBody) < countStrictFields(previousObservation.rawBody)) return "strict_fallback";
	if (reasoningShape(rawBody) !== reasoningShape(previousObservation.rawBody)) return "reasoning_fallback";
	if (previousObservation.finishReason === "stop" && !previousObservation.toolCallObserved) {
		const visible = previousObservation.contentDeltas.some(delta => /\S/u.test(delta));
		if (!visible) return "empty_retry";
	}
	return "transport_retry";
}

function responseHeaders(response: Response): Record<string, string> {
	return headersRecord(response.headers);
}

async function readBoundedErrorBytes(response: Response): Promise<{ bytes: Uint8Array; truncated: boolean }> {
	const body = response.clone().body;
	if (!body) return { bytes: new Uint8Array(), truncated: false };
	const reader = body.getReader();
	const chunks: Uint8Array[] = [];
	let size = 0;
	let truncated = false;
	try {
		while (true) {
			const chunk = await reader.read();
			if (chunk.done) break;
			const remaining = MAX_ERROR_BODY_BYTES - size;
			if (chunk.value.byteLength <= remaining) {
				chunks.push(chunk.value);
				size += chunk.value.byteLength;
				continue;
			}
			if (remaining > 0) {
				chunks.push(chunk.value.subarray(0, remaining));
				size += remaining;
			}
			truncated = true;
			void reader.cancel().catch(() => {});
			break;
		}
	} finally {
		reader.releaseLock();
	}
	const bytes = new Uint8Array(size);
	let offset = 0;
	for (const chunk of chunks) {
		bytes.set(chunk, offset);
		offset += chunk.byteLength;
	}
	return { bytes, truncated };
}

function redactErrorBodyStrings(value: unknown, apiKey: string): unknown {
	if (typeof value === "string") {
		const sanitized = sanitizeProbeErrorMessage(value, apiKey);
		return { redacted: true, length: sanitized.length };
	}
	if (Array.isArray(value)) return value.map(item => redactErrorBodyStrings(item, apiKey));
	if (typeof value !== "object" || value === null) return value;
	const source = value as Record<string, unknown>;
	const result: Record<string, unknown> = {};
	let unknownKeyIndex = 0;
	for (const key of Object.keys(source).sort(compareProbeObjectKeys)) {
		let safeKey = key;
		if (SAFE_ERROR_PROPERTIES[key.toLowerCase()] !== true) {
			unknownKeyIndex += 1;
			safeKey = `[REDACTED_KEY_${unknownKeyIndex}]`;
		}
		result[safeKey] = SENSITIVE_PROPERTY.test(key) ? "[REDACTED]" : redactErrorBodyStrings(source[key], apiKey);
	}
	return result;
}

async function readErrorBody(response: Response, apiKey: string): Promise<string | undefined> {
	try {
		const { bytes, truncated } = await readBoundedErrorBytes(response);
		const decoded = new TextDecoder().decode(bytes);
		let value: unknown = decoded;
		try {
			value = JSON.parse(decoded) as unknown;
		} catch {}
		const text = JSON.stringify(redactErrorBodyStrings(value, apiKey));
		const outputTruncated = text.length > MAX_ERROR_STRING_CHARS;
		const bounded = outputTruncated ? text.slice(0, MAX_ERROR_STRING_CHARS) : text;
		return truncated || outputTruncated ? `${bounded}…` : bounded;
	} catch {
		return undefined;
	}
}

function inspectSsePayload(observation: AttemptObservation, payload: unknown): void {
	if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return;
	const source = payload as Record<string, unknown>;
	if (typeof source.usage === "object" && source.usage !== null) observation.rawUsageObserved = true;
	const choices = Array.isArray(source.choices) ? source.choices : [];
	for (const choice of choices) {
		if (typeof choice !== "object" || choice === null || Array.isArray(choice)) continue;
		const choiceObject = choice as Record<string, unknown>;
		if (typeof choiceObject.finish_reason === "string") observation.finishReason = choiceObject.finish_reason;
		const delta = choiceObject.delta;
		if (typeof delta !== "object" || delta === null || Array.isArray(delta)) continue;
		const deltaObject = delta as Record<string, unknown>;
		if (typeof deltaObject.content === "string") observation.contentDeltas.push(deltaObject.content);
		if (Array.isArray(deltaObject.tool_calls) && deltaObject.tool_calls.length > 0)
			observation.toolCallObserved = true;
		for (const key of ["reasoning", "reasoning_content", "reasoning_text"] as const) {
			if (typeof deltaObject[key] !== "string") continue;
			observation.reasoningFields.add(key);
			observation.reasoningDeltas.push(deltaObject[key]);
		}
	}
}

export function createOpenAICompatWireObserver(
	innerFetch: FetchImpl,
	apiKey: string,
	privateValues: readonly string[] = [],
): OpenAICompatWireObserver {
	const privateStrings = new Set(privateValues.filter(value => value.length > 0));
	const attempts: OpenAICompatProbeWireAttempt[] = [];
	const observations: AttemptObservation[] = [];
	let lastPayload: unknown;

	const observedFetch = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
		const startedAt = performance.now();
		const rawBody = parseBody(init?.body ?? (input instanceof Request ? await input.clone().text() : undefined));
		const inputHeaders = input instanceof Request ? input.headers : undefined;
		const mergedHeaders = new Headers(inputHeaders);
		for (const [key, value] of new Headers(init?.headers)) mergedHeaders.set(key, value);
		const requestUrl = input instanceof Request ? input.url : String(input);
		const observation: AttemptObservation = {
			startedAt,
			rawBody,
			maxInterEventGapMs: 0,
			rawUsageObserved: false,
			reasoningFields: new Set<string>(),
			reasoningDeltas: [],
			contentDeltas: [],
			toolCallObserved: false,
		};
		const attempt: OpenAICompatProbeWireAttempt = {
			ordinal: attempts.length + 1,
			request: {
				method: init?.method ?? (input instanceof Request ? input.method : "GET"),
				url: sanitizeProbeUrl(requestUrl, apiKey),
				headers: headersRecord(mergedHeaders),
				...(rawBody !== undefined ? { body: await sanitizeRequestBody(rawBody, apiKey, privateStrings) } : {}),
			},
			latencyMs: 0,
			classification: classifyAttempt(attempts, observations, rawBody ?? lastPayload),
		};
		rawRequestBodies.set(attempt, rawBody);
		attempts.push(attempt);
		observations.push(observation);
		try {
			const response = await innerFetch(input, init);
			attempt.latencyMs = Math.max(0, Math.round(performance.now() - startedAt));
			attempt.response = {
				status: response.status,
				headers: responseHeaders(response),
				...(response.ok ? {} : { errorBody: await readErrorBody(response, apiKey) }),
			};
			return response;
		} catch (error) {
			attempt.latencyMs = Math.max(0, Math.round(performance.now() - startedAt));
			throw error;
		}
	};
	const fetchWithPreconnect = Object.assign(observedFetch, { preconnect: innerFetch.preconnect });

	return {
		fetch: fetchWithPreconnect,
		attempts,
		onPayload(payload) {
			lastPayload = payload;
		},
		onSseEvent(event) {
			const observation = observations.at(-1);
			if (!observation) return;
			const now = performance.now();
			if (observation.firstEventAt === undefined) observation.firstEventAt = now;
			if (observation.lastEventAt !== undefined) {
				observation.maxInterEventGapMs = Math.max(observation.maxInterEventGapMs, now - observation.lastEventAt);
			}
			observation.lastEventAt = now;
			if (event.data === "[DONE]") return;
			try {
				inspectSsePayload(observation, JSON.parse(event.data) as unknown);
			} catch {}
		},
		evidence() {
			const firstAttempt = attempts[0];
			const firstObservation = observations[0];
			const reasoningFields = new Set<string>();
			const reasoningDeltas: string[] = [];
			const contentDeltas: string[] = [];
			let rawUsageObserved = false;
			let toolCallObserved = false;
			let maxInterEventGapMs = 0;
			for (const observation of observations) {
				rawUsageObserved ||= observation.rawUsageObserved;
				toolCallObserved ||= observation.toolCallObserved;
				maxInterEventGapMs = Math.max(maxInterEventGapMs, observation.maxInterEventGapMs);
				for (const field of observation.reasoningFields) reasoningFields.add(field);
				reasoningDeltas.push(...observation.reasoningDeltas);
				contentDeltas.push(...observation.contentDeltas);
			}
			const rawDeltas: OpenAICompatRawWireDeltas = {
				reasoningDeltas,
				contentDeltas,
				reasoningDeltaSequences: observations.map(observation => [...observation.reasoningDeltas]),
			};
			const evidence: OpenAICompatWireEvidence = {
				rawUsageObserved,
				reasoningFields: Array.from(reasoningFields).sort(),
				reasoningDeltas: reasoningDeltas.map(summarizeGeneratedDelta),
				contentDeltas: contentDeltas.map(summarizeGeneratedDelta),
				reasoningDeltaSequences: rawDeltas.reasoningDeltaSequences.map(sequence =>
					sequence.map(summarizeGeneratedDelta),
				),
				toolCallObserved,
				...(firstAttempt && firstObservation?.firstEventAt !== undefined
					? {
							timeToFirstEventMs: Math.max(
								0,
								Math.round(firstObservation.firstEventAt - firstObservation.startedAt),
							),
						}
					: {}),
				maxInterEventGapMs: Math.max(0, Math.round(maxInterEventGapMs)),
			};
			rawWireDeltas.set(evidence, rawDeltas);
			return evidence;
		},
	};
}
