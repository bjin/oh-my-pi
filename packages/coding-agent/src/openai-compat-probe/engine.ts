import * as AIError from "@oh-my-pi/pi-ai/error";
import { type OpenAICompletionsOptions, streamOpenAICompletions } from "@oh-my-pi/pi-ai/providers/openai-completions";
import type { AssistantMessage, Context, ProviderSessionState, RawSseEvent, ToolChoice } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Effort, THINKING_EFFORTS } from "@oh-my-pi/pi-catalog/effort";
import type { Model, ModelSpec, OpenAICompat } from "@oh-my-pi/pi-catalog/types";
import { mergeCompat, toModelSpec } from "../config/model-compat";
import { OPENAI_COMPAT_CAPABILITY_REGISTRY, OPENAI_COMPAT_KEYS, validateStrictOpenAICompat } from "./capabilities";
import { compareProbeStrings } from "./ordering";
import {
	BASELINE_SYSTEM_PROMPT,
	BASELINE_USER_PROMPT,
	COMPAT_ECHO_NON_STRICT_TOOL,
	COMPAT_ECHO_TOOL,
	createBaselineContext,
	createHistoryContext,
	createProbeUserMessage,
	createToolContext,
	FIRST_SYSTEM_PROMPT,
	SECOND_SYSTEM_PROMPT,
	TOOL_FOLLOWUP_PROMPT,
	TOOL_RESULT_PROMPT,
	TOOL_USER_PROMPT,
} from "./scenarios";
import type {
	OpenAICompatProbeCandidate,
	OpenAICompatProbeFinding,
	OpenAICompatProbeInput,
	OpenAICompatProbeReport,
	OpenAICompatProbeScenario,
	OpenAICompatProbeVerdict,
	OpenAICompatProbeWireAttempt,
} from "./types";
import {
	assertCredentialFreeProbeValue,
	createOpenAICompatWireObserver,
	getOpenAICompatRawRequestBody,
	getOpenAICompatRawWireDeltas,
	type OpenAICompatWireEvidence,
	sanitizeProbeUrl,
} from "./wire-observer";

const DEFAULT_TIMEOUT_MS = 120_000;
const MAX_TIMEOUT_MS = 3_600_000;
const MAX_LOGICAL_CALLS_PER_SCENARIO = 64;
const SCENARIO_BUDGET_OVERHEAD_MS = 10_000;
export const OPENAI_COMPAT_PROBE_SCENARIO_IDS = [
	"connectivity",
	"store",
	"streaming-usage",
	"max-tokens",
	"always-max-tokens",
	"sampling-parameters",
	"multiple-system-messages",
	"developer-role",
	"reasoning-format",
	"reasoning-effort",
	"reasoning-disable",
	"tool-choice",
	"tool-strictness",
	"tool-schema-flavor",
	"history-tool-result-name",
	"history-assistant-after-tool-result",
	"history-thinking-as-text",
	"history-tool-call-id",
	"history-reasoning-content",
	"history-reasoning-all-assistant",
	"history-reasoning-replay",
	"history-assistant-tool-content",
	"reasoning-tool-interaction",
	"candidate-profiles",
	"fallback-latching",
	"stream-observation",
	"prevalidation",
] as const;
const PROBE_SCENARIO_TOTAL = OPENAI_COMPAT_PROBE_SCENARIO_IDS.length;
const CANONICAL_EFFORTS: readonly Effort[] = THINKING_EFFORTS;
const SCHEMA_REJECTION_STATUSES: Record<number, true> = { 400: true, 404: true, 422: true };
const RESPONSE_ONLY_KEYS: Partial<Record<keyof OpenAICompat, true>> = {
	supportsLongPromptCacheRetention: true,
	strictResponsesPairing: true,
	supportsImageDetailOriginal: true,
	includeEncryptedReasoning: true,
	filterReasoningHistory: true,
};
const REASONING_ONLY_KEYS: Partial<Record<keyof OpenAICompat, true>> = {
	supportsReasoningEffort: true,
	reasoningEffortMap: true,
	thinkingFormat: true,
	reasoningDisableMode: true,
	omitReasoningEffort: true,
	thinkingKeep: true,
	reasoningContentField: true,
	requiresReasoningContentForToolCalls: true,
	requiresReasoningContentForAllAssistantTurns: true,
	allowsSyntheticReasoningContentForToolCalls: true,
	replayReasoningContent: true,
	qwenPreserveThinking: true,
	disableReasoningOnForcedToolChoice: true,
	disableReasoningOnToolChoice: true,
	supportsReasoningParams: true,
	whenThinking: true,
};
const CONSERVATIVE_CONNECTIVITY_PROFILE: OpenAICompat = {
	supportsStore: false,
	supportsUsageInStreaming: false,
	supportsDeveloperRole: false,
	supportsMultipleSystemMessages: false,
	supportsSamplingParams: false,
	supportsReasoningParams: false,
	supportsReasoningEffort: false,
	supportsToolChoice: false,
	alwaysSendMaxTokens: false,
	qwenPreserveThinking: false,
};

interface CompatLeaf {
	path: string;
	segments: string[];
	value: unknown;
}

interface ProbeCallResult {
	message?: AssistantMessage;
	attempts: OpenAICompatProbeWireAttempt[];
	wire: OpenAICompatWireEvidence;
}

interface ProbeRequestOptions
	extends Omit<
		OpenAICompletionsOptions,
		| "apiKey"
		| "fetch"
		| "signal"
		| "providerSessionState"
		| "sessionId"
		| "promptCacheKey"
		| "onPayload"
		| "onSseEvent"
		| "streamFirstEventTimeoutMs"
		| "streamIdleTimeoutMs"
	> {}

interface ScenarioEvaluation {
	verdict: OpenAICompatProbeVerdict;
	evidence?: Record<string, unknown>;
}

interface ScenarioEnvironment {
	signal: AbortSignal;
	call(
		model: Model<"openai-completions">,
		context: Context,
		options?: ProbeRequestOptions,
		sharedState?: Map<string, ProviderSessionState>,
	): Promise<ProbeCallResult>;
	createSharedState(): Map<string, ProviderSessionState>;
}

interface PairDecision {
	verdict: OpenAICompatProbeVerdict;
	recommendedValue?: boolean;
}

interface FinalValidationResult {
	viable: boolean;
	fallbackFree: boolean;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
	const prototype = Object.getPrototypeOf(value);
	return prototype === Object.prototype || prototype === null;
}

function mergeProfiles(base: OpenAICompat, override: OpenAICompat): OpenAICompat {
	return (mergeCompat(base, override) ?? {}) as OpenAICompat;
}

function cloneCompat(value: OpenAICompat): OpenAICompat {
	return structuredClone(value);
}

function setCompatValue(profile: OpenAICompat, key: keyof OpenAICompat, value: unknown): OpenAICompat {
	const next = cloneCompat(profile) as Record<string, unknown>;
	next[key] = structuredClone(value);
	return next as OpenAICompat;
}

function deleteCompatValue(profile: OpenAICompat, key: keyof OpenAICompat): OpenAICompat {
	const next = cloneCompat(profile) as Record<string, unknown>;
	delete next[key];
	return next as OpenAICompat;
}

function encodeCompatPath(segments: readonly string[]): string {
	return `/${segments.map(segment => segment.replaceAll("~", "~0").replaceAll("/", "~1")).join("/")}`;
}

function collectLeaves(value: unknown, segments: string[] = []): CompatLeaf[] {
	if (!isPlainRecord(value)) {
		return [{ path: encodeCompatPath(segments), segments, value }];
	}
	const keys = Object.keys(value);
	if (keys.length === 0 && segments.length > 0) return [{ path: encodeCompatPath(segments), segments, value: {} }];
	const leaves: CompatLeaf[] = [];
	for (const key of keys) leaves.push(...collectLeaves(value[key], [...segments, key]));
	return leaves;
}
function collectPrivateCompatStrings(value: unknown): string[] {
	if (!isPlainRecord(value)) return [];
	const privateStrings = new Set<string>();
	const collect = (current: unknown): void => {
		if (typeof current === "string") {
			if (current.length > 0) privateStrings.add(current);
			return;
		}
		if (Array.isArray(current)) {
			for (const child of current) collect(child);
			return;
		}
		if (!isPlainRecord(current)) return;
		for (const child of Object.values(current)) collect(child);
	};
	for (const [key, child] of Object.entries(value)) {
		if (!Object.hasOwn(OPENAI_COMPAT_CAPABILITY_REGISTRY, key)) continue;
		const compatKey = key as keyof OpenAICompat;
		if (compatKey === "whenThinking" || OPENAI_COMPAT_CAPABILITY_REGISTRY[compatKey].redaction === "nested_values") {
			collect(child);
		}
	}
	return [...privateStrings];
}

function getPath(value: unknown, segments: readonly string[]): unknown {
	let current = value;
	for (const segment of segments) {
		if (!isPlainRecord(current)) return undefined;
		current = current[segment];
	}
	return current;
}

function removeLeaf(profile: OpenAICompat, segments: readonly string[]): OpenAICompat {
	const next = cloneCompat(profile) as Record<string, unknown>;
	const parents: Record<string, unknown>[] = [next];
	let current = next;
	for (const segment of segments.slice(0, -1)) {
		const child = current[segment];
		if (!isPlainRecord(child)) return next as OpenAICompat;
		current = child;
		parents.push(current);
	}
	const finalSegment = segments.at(-1);
	if (finalSegment === undefined) return next as OpenAICompat;
	delete current[finalSegment];
	for (let index = parents.length - 1; index > 0; index -= 1) {
		if (Object.keys(parents[index]).length > 0) break;
		delete parents[index - 1][segments[index - 1]];
	}
	return next as OpenAICompat;
}

function compatValuesEqual(left: unknown, right: unknown): boolean {
	return JSON.stringify(left) === JSON.stringify(right);
}

function extractKnownCompat(value: unknown): OpenAICompat {
	if (!isPlainRecord(value)) return {};
	const result: Record<string, unknown> = {};
	for (const key of OPENAI_COMPAT_KEYS) {
		const raw = value[key];
		if (raw === undefined) continue;
		if (key === "whenThinking") {
			if (isPlainRecord(raw)) {
				const nested: Record<string, unknown> = {};
				for (const nestedKey of OPENAI_COMPAT_KEYS) {
					if (nestedKey === "whenThinking" || raw[nestedKey] === undefined) continue;
					nested[nestedKey] = structuredClone(raw[nestedKey]);
				}
				result[key] = nested;
			}
			continue;
		}
		result[key] = structuredClone(raw);
	}
	return result as OpenAICompat;
}

function collectIgnoredConfiguredKeys(value: unknown): string[] {
	if (!isPlainRecord(value)) return [];
	const ignored: string[] = [];
	for (const key of Object.keys(value)) {
		if (!Object.hasOwn(OPENAI_COMPAT_CAPABILITY_REGISTRY, key)) {
			for (const leaf of collectLeaves(value[key], [key])) ignored.push(leaf.path);
			continue;
		}
		if (key !== "whenThinking" || !isPlainRecord(value[key])) continue;
		for (const nestedKey of Object.keys(value[key])) {
			if (nestedKey !== "whenThinking" && Object.hasOwn(OPENAI_COMPAT_CAPABILITY_REGISTRY, nestedKey)) {
				continue;
			}
			for (const leaf of collectLeaves(value[key][nestedKey], [key, nestedKey])) ignored.push(leaf.path);
		}
	}
	return Array.from(new Set(ignored)).sort();
}

function validateCandidates(
	candidates: readonly OpenAICompatProbeCandidate[] | undefined,
	apiKey: string,
): OpenAICompatProbeCandidate[] {
	if (!candidates) return [];
	if (candidates.length > 16) throw new Error("At most 16 OpenAI compatibility candidates are allowed");
	const names = new Set<string>();
	const validated: OpenAICompatProbeCandidate[] = [];
	for (const candidate of candidates) {
		if (!candidate || typeof candidate !== "object")
			throw new Error("Each compatibility candidate must be an object");
		const name = typeof candidate.name === "string" ? candidate.name.trim() : "";
		if (name.length < 1 || name.length > 64) {
			throw new Error("Compatibility candidate names must contain 1-64 characters after trimming");
		}
		if (apiKey.length > 0 && name.includes(apiKey)) {
			throw new Error("Compatibility candidate name contains the API credential");
		}
		if (names.has(name)) throw new Error(`Duplicate compatibility candidate name: ${name}`);
		names.add(name);
		if (candidate.source !== "built_in" && candidate.source !== "configured" && candidate.source !== "user") {
			throw new Error(`Invalid compatibility candidate source for ${name}`);
		}
		assertCredentialFreeProbeValue(candidate.compat, apiKey, `Compatibility candidate ${name}`);
		validated.push({
			name,
			source: candidate.source,
			compat: structuredClone(validateStrictOpenAICompat(candidate.compat)),
		});
	}
	return validated;
}

function sanitizeIdentifier(value: string, apiKey: string): string {
	return apiKey.length > 0 ? value.replaceAll(apiKey, "[REDACTED]") : value;
}

function callSucceeded(result: ProbeCallResult): boolean {
	const message = result.message;
	if (!message || message.stopReason === "error" || message.stopReason === "aborted") return false;
	const finalStatus = result.attempts.at(-1)?.response?.status;
	return finalStatus === undefined || (finalStatus >= 200 && finalStatus < 300);
}

function callHasCompatibilityFallback(result: ProbeCallResult): boolean {
	return result.attempts.some(
		attempt => attempt.classification === "reasoning_fallback" || attempt.classification === "strict_fallback",
	);
}

function callSchemaRejected(result: ProbeCallResult): boolean {
	if (callSucceeded(result)) return false;
	const statuses = [result.message?.errorStatus, ...result.attempts.map(attempt => attempt.response?.status)];
	return statuses.some(status => status !== undefined && SCHEMA_REJECTION_STATUSES[status] === true);
}

function callAuthenticationFailed(result: ProbeCallResult): boolean {
	const statuses = [result.message?.errorStatus, ...result.attempts.map(attempt => attempt.response?.status)];
	return statuses.some(status => status === 401 || status === 403);
}

function callTransientlyFailed(result: ProbeCallResult): boolean {
	if (AIError.is(result.message?.errorId, AIError.Flag.Transient)) return true;
	const statuses = [result.message?.errorStatus, ...result.attempts.map(attempt => attempt.response?.status)];
	if (statuses.some(status => status === 408 || status === 429 || (status !== undefined && status >= 500))) {
		return true;
	}
	if (
		result.attempts.some(
			attempt => attempt.classification === "transport_retry" || attempt.classification === "empty_retry",
		)
	) {
		return true;
	}
	return (
		!callSucceeded(result) &&
		(result.attempts.some(attempt => attempt.response === undefined) || result.message?.stopReason === "aborted")
	);
}

function callDirectlySucceeded(result: ProbeCallResult): boolean {
	return callSucceeded(result) && !callHasCompatibilityFallback(result) && !callTransientlyFailed(result);
}

function callSucceededViaCompatibilityFallback(result: ProbeCallResult): boolean {
	return callSucceeded(result) && callHasCompatibilityFallback(result) && !callTransientlyFailed(result);
}

function hasCompatEchoCall(message: AssistantMessage | undefined): boolean {
	if (!message) return false;
	return message.content.some(
		block => block.type === "toolCall" && block.name === "compat_echo" && block.arguments.value === "PROBE_OK",
	);
}

function hasThinkingBlock(message: AssistantMessage | undefined): boolean {
	return message?.content.some(block => block.type === "thinking" && block.thinking.length > 0) === true;
}

function visibleText(message: AssistantMessage | undefined): string {
	if (!message) return "";
	return message.content
		.filter((block): block is Extract<AssistantMessage["content"][number], { type: "text" }> => block.type === "text")
		.map(block => block.text)
		.join("");
}

function hasCumulativeReasoningDeltas(deltas: readonly string[]): boolean {
	return deltas.some(
		(delta, index) => index > 0 && delta.length > deltas[index - 1].length && delta.startsWith(deltas[index - 1]),
	);
}

function finalRequestBody(result: ProbeCallResult): Record<string, unknown> | undefined {
	const body = result.attempts.at(-1)?.request.body;
	return isPlainRecord(body) ? body : undefined;
}
function rawFinalRequestBody(result: ProbeCallResult): Record<string, unknown> | undefined {
	const attempt = result.attempts.at(-1);
	if (!attempt) return undefined;
	const body = getOpenAICompatRawRequestBody(attempt);
	return isPlainRecord(body) ? body : undefined;
}

function rawFinalReasoningEffort(result: ProbeCallResult): string | undefined {
	const body = rawFinalRequestBody(result);
	const nestedReasoning = isPlainRecord(body?.reasoning) ? body.reasoning : undefined;
	return typeof body?.reasoning_effort === "string"
		? body.reasoning_effort
		: typeof nestedReasoning?.effort === "string"
			? nestedReasoning.effort
			: undefined;
}

function pairDecision(enabled: ProbeCallResult, disabled: ProbeCallResult, semantic: boolean): PairDecision {
	if (callAuthenticationFailed(enabled) || callAuthenticationFailed(disabled)) return { verdict: "indeterminate" };
	if (callTransientlyFailed(enabled) || callTransientlyFailed(disabled)) return { verdict: "indeterminate" };
	if (callSucceeded(enabled) && callHasCompatibilityFallback(enabled)) {
		return { verdict: "fallback_only", recommendedValue: false };
	}
	if (callSchemaRejected(enabled) && callSucceeded(disabled)) {
		return { verdict: "unsupported", recommendedValue: false };
	}
	if (callSucceeded(enabled) && callSchemaRejected(disabled)) {
		return { verdict: "required", recommendedValue: true };
	}
	if (callSucceeded(enabled)) return { verdict: semantic ? "effective" : "accepted" };
	return { verdict: "indeterminate" };
}
function semanticPairDecision(
	enabled: ProbeCallResult,
	disabled: ProbeCallResult,
	enabledSemantic: boolean,
	disabledSemantic: boolean,
): PairDecision {
	if (callDirectlySucceeded(enabled) && callDirectlySucceeded(disabled)) {
		if (enabledSemantic && !disabledSemantic) return { verdict: "effective", recommendedValue: true };
		if (!enabledSemantic && disabledSemantic) return { verdict: "unsupported", recommendedValue: false };
		return { verdict: enabledSemantic ? "effective" : "accepted" };
	}
	return pairDecision(enabled, disabled, enabledSemantic);
}

function resolveValidationToolChoice(compat: {
	supportsToolChoice: boolean;
	supportsForcedToolChoice: boolean;
	supportsNamedToolChoice: boolean;
}): ToolChoice {
	if (!compat.supportsToolChoice || !compat.supportsForcedToolChoice) return "auto";
	return compat.supportsNamedToolChoice ? { type: "function", function: { name: "compat_echo" } } : "required";
}

function candidateRank(candidate: OpenAICompatProbeCandidate): number {
	switch (candidate.source) {
		case "built_in":
			return 0;
		case "configured":
			return 1;
		case "user":
			return 2;
	}
}

function closeProviderState(state: Map<string, ProviderSessionState>): void {
	for (const value of state.values()) value.close();
	state.clear();
}

/** Probe one configured Chat Completions model without touching user state. */
export async function probeOpenAICompatibility(input: OpenAICompatProbeInput): Promise<OpenAICompatProbeReport> {
	const runtimeModel = input.model as Model;
	if (runtimeModel.api !== "openai-completions") {
		throw new Error("OpenAI compatibility probing requires api: openai-completions");
	}
	if (runtimeModel.transport !== undefined)
		throw new Error("OpenAI compatibility probing does not support transport overrides");
	if (runtimeModel.supportsTools === false) throw new Error("OpenAI compatibility probing requires tool support");
	if (typeof input.apiKey !== "string" || input.apiKey.length === 0) throw new Error("An API key is required");
	const candidates = validateCandidates(input.candidates, input.apiKey);
	const model = input.model;
	const requestedTimeoutMs =
		input.timeoutMs ??
		(model.compat.streamIdleTimeoutMs !== undefined && model.compat.streamIdleTimeoutMs > 0
			? model.compat.streamIdleTimeoutMs
			: DEFAULT_TIMEOUT_MS);
	if (!Number.isInteger(requestedTimeoutMs) || requestedTimeoutMs <= 0 || requestedTimeoutMs > MAX_TIMEOUT_MS) {
		throw new Error(`timeoutMs must be an integer from 1 through ${MAX_TIMEOUT_MS}`);
	}
	if (input.signal?.aborted) throw input.signal.reason ?? new DOMException("Aborted", "AbortError");

	const timeoutMs = requestedTimeoutMs;
	const rawConfigured = isPlainRecord(model.compatConfig) ? structuredClone(model.compatConfig) : {};
	assertCredentialFreeProbeValue(rawConfigured, input.apiKey, "Configured compatibility");
	const configuredKnown = validateStrictOpenAICompat(extractKnownCompat(rawConfigured));
	const configuredCompat = cloneCompat(configuredKnown) as OpenAICompat & Record<string, unknown>;
	const ignoredConfiguredKeys = collectIgnoredConfiguredKeys(rawConfigured);
	const modelSpec = toModelSpec(model);
	const baselineModel = buildModel({ ...modelSpec, compat: undefined } as ModelSpec<"openai-completions">);
	const scenarios: OpenAICompatProbeScenario[] = [];
	const warnings: string[] = [];
	const findings = new Map<keyof OpenAICompat, OpenAICompatProbeFinding>();
	const provisionalKeys = new Set<keyof OpenAICompat>();
	for (const key of OPENAI_COMPAT_KEYS) {
		const notApplicable = RESPONSE_ONLY_KEYS[key] === true || (!model.reasoning && REASONING_ONLY_KEYS[key] === true);
		findings.set(key, {
			key,
			verdict: notApplicable ? "not_applicable" : "indeterminate",
			scenarioIds: [],
			persistable: true,
		});
	}

	const setFinding = (
		key: keyof OpenAICompat,
		verdict: OpenAICompatProbeVerdict,
		scenarioId: string,
		recommendedValue?: unknown,
		note?: string,
	): void => {
		const current = findings.get(key);
		if (!current || current.verdict === "not_applicable") return;
		current.verdict = verdict;
		if (!current.scenarioIds.includes(scenarioId)) current.scenarioIds.push(scenarioId);
		if (recommendedValue === undefined) delete current.recommendedValue;
		else current.recommendedValue = structuredClone(recommendedValue);
		if (note === undefined) delete current.note;
		else current.note = note;
	};

	const buildTrialModel = (profile: OpenAICompat): Model<"openai-completions"> => {
		const replacement = mergeCompat({} as OpenAICompat, profile) ?? {};
		return buildModel({ ...toModelSpec(model), compat: replacement } as ModelSpec<"openai-completions">);
	};

	let probeIndex = 0;
	let minimizeIndex = 0;
	let minimizeTotal = 0;
	let validateIndex = 0;
	let workingProfile: OpenAICompat = {};
	let authenticationFailed = false;
	let winningEffort: Effort | undefined;
	let fallbackProbe:
		| { model: Model<"openai-completions">; context: Context; options: ProbeRequestOptions }
		| undefined;
	const allWireEvidence: OpenAICompatWireEvidence[] = [];

	const runScenario = async (
		baseId: string,
		phase: "probe" | "minimize" | "validate",
		run: (environment: ScenarioEnvironment) => Promise<ScenarioEvaluation>,
		explicitId?: string,
	): Promise<OpenAICompatProbeScenario> => {
		const id = explicitId ?? `${baseId}@${phase}`;
		const index = phase === "probe" ? ++probeIndex : phase === "minimize" ? ++minimizeIndex : ++validateIndex;
		const total = phase === "probe" ? PROBE_SCENARIO_TOTAL : phase === "minimize" ? minimizeTotal : 1;
		await input.onProgress?.({ type: "scenario_start", scenarioId: id, phase, index, total });
		if (input.signal?.aborted) throw input.signal.reason ?? new DOMException("Aborted", "AbortError");
		const scenarioController = new AbortController();
		const scenarioBudgetMs = MAX_LOGICAL_CALLS_PER_SCENARIO * timeoutMs + SCENARIO_BUDGET_OVERHEAD_MS;
		const timer = setTimeout(
			() =>
				scenarioController.abort(
					new DOMException(`Scenario budget exceeded after ${scenarioBudgetMs}ms`, "TimeoutError"),
				),
			scenarioBudgetMs,
		);
		const signal = input.signal
			? AbortSignal.any([input.signal, scenarioController.signal])
			: scenarioController.signal;
		const attempts: OpenAICompatProbeWireAttempt[] = [];
		const sharedStates = new Set<Map<string, ProviderSessionState>>();
		let callOrdinal = 0;
		let lastMessage: AssistantMessage | undefined;
		let evaluation: ScenarioEvaluation;

		const call = async (
			callModel: Model<"openai-completions">,
			context: Context,
			options: ProbeRequestOptions = {},
			sharedState?: Map<string, ProviderSessionState>,
		): Promise<ProbeCallResult> => {
			if (input.signal?.aborted) throw input.signal.reason ?? new DOMException("Aborted", "AbortError");
			if (scenarioController.signal.aborted) {
				throw scenarioController.signal.reason ?? new DOMException("Scenario timed out", "TimeoutError");
			}
			const state = sharedState ?? new Map<string, ProviderSessionState>();
			if (!sharedState) sharedStates.add(state);
			const observer = createOpenAICompatWireObserver(
				input.fetch ?? fetch,
				input.apiKey,
				collectPrivateCompatStrings(callModel.compatConfig),
			);
			callOrdinal += 1;
			const requestOptions = {
				...options,
				apiKey: input.apiKey,
				fetch: observer.fetch,
				maxInFlightRequests: {},
				signal,
				providerSessionState: state,
				sessionId: `omp-openai-compat:${id}:${callOrdinal}`,
				promptCacheKey: `omp-openai-compat:${id}:${callOrdinal}`,
				streamFirstEventTimeoutMs: timeoutMs,
				streamIdleTimeoutMs: timeoutMs,
				suppressRejectedRequestDump: true,
				onPayload: (payload: unknown) => observer.onPayload(payload),
				onSseEvent: (event: RawSseEvent) => observer.onSseEvent(event),
			};
			let message: AssistantMessage | undefined;
			let abortError: unknown;
			try {
				message = await streamOpenAICompletions(callModel, context, requestOptions).result();
				lastMessage = message;
			} catch (error) {
				if (input.signal?.aborted) {
					abortError = input.signal.reason ?? error;
				} else if (scenarioController.signal.aborted) {
					abortError = scenarioController.signal.reason ?? error;
				}
			} finally {
				if (!sharedState) {
					closeProviderState(state);
					sharedStates.delete(state);
				}
			}
			for (const attempt of observer.attempts) {
				attempts.push({ ...attempt, ordinal: attempts.length + 1 });
			}
			const wire = observer.evidence();
			allWireEvidence.push(wire);
			if (abortError !== undefined) throw abortError;
			return { message, attempts: observer.attempts, wire };
		};

		const environment: ScenarioEnvironment = {
			signal,
			call,
			createSharedState() {
				const state = new Map<string, ProviderSessionState>();
				sharedStates.add(state);
				return state;
			},
		};

		try {
			evaluation = await run(environment);
			if (input.signal?.aborted) throw input.signal.reason ?? new DOMException("Aborted", "AbortError");
			if (scenarioController.signal.aborted) {
				throw scenarioController.signal.reason ?? new DOMException("Scenario timed out", "TimeoutError");
			}
		} catch (error) {
			if (input.signal?.aborted) throw input.signal.reason ?? error;
			if (!scenarioController.signal.aborted) throw error;
			evaluation = {
				verdict: "indeterminate",
				evidence: { timeout: true, timeoutMs: scenarioBudgetMs },
			};
		} finally {
			clearTimeout(timer);
			for (const state of sharedStates) closeProviderState(state);
			sharedStates.clear();
		}
		const scenario: OpenAICompatProbeScenario = {
			id,
			baseId,
			phase,
			verdict: evaluation.verdict,
			attempts,
			final: {
				...(lastMessage?.stopReason !== undefined ? { stopReason: lastMessage.stopReason } : {}),
				...(lastMessage?.errorStatus !== undefined ? { errorStatus: lastMessage.errorStatus } : {}),
				...(lastMessage?.errorId !== undefined ? { errorId: lastMessage.errorId } : {}),
			},
			evidence: evaluation.evidence ?? {},
		};
		scenarios.push(scenario);
		await input.onProgress?.({
			type: "scenario_complete",
			scenarioId: id,
			phase,
			index,
			total,
			verdict: scenario.verdict,
		});
		if (input.signal?.aborted) throw input.signal.reason ?? new DOMException("Aborted", "AbortError");
		return scenario;
	};

	const applyBooleanDecision = (key: keyof OpenAICompat, decision: PairDecision, scenarioId: string): void => {
		setFinding(key, decision.verdict, scenarioId, decision.recommendedValue);
		if (decision.recommendedValue !== undefined) {
			workingProfile = setCompatValue(workingProfile, key, decision.recommendedValue);
		}
	};

	let connectivityWinner:
		| { name: string; compat: OpenAICompat; source: OpenAICompatProbeCandidate["source"] }
		| undefined;
	let connectivityStable = false;
	await runScenario("connectivity", "probe", async environment => {
		const seeds: Array<{ name: string; compat: OpenAICompat; source: OpenAICompatProbeCandidate["source"] }> = [
			{ name: "conservative-control", compat: CONSERVATIVE_CONNECTIVITY_PROFILE, source: "built_in" },
			{
				name: "configured-under-control",
				compat: mergeProfiles(configuredKnown, CONSERVATIVE_CONNECTIVITY_PROFILE),
				source: "configured",
			},
			...candidates.map(candidate => ({
				name: `${candidate.name}-under-control`,
				compat: mergeProfiles(mergeProfiles(configuredKnown, candidate.compat), CONSERVATIVE_CONNECTIVITY_PROFILE),
				source: candidate.source,
			})),
			{ name: "configured", compat: configuredKnown, source: "configured" },
			...candidates.map(candidate => ({
				name: candidate.name,
				compat: mergeProfiles(configuredKnown, candidate.compat),
				source: candidate.source,
			})),
		];
		const results: Array<Record<string, unknown>> = [];
		let transientWinner: (typeof seeds)[number] | undefined;
		for (const seed of seeds) {
			const result = await environment.call(buildTrialModel(seed.compat), createBaselineContext());
			results.push({
				name: seed.name,
				passing: callSucceeded(result),
				stable: callDirectlySucceeded(result),
				status: result.message?.errorStatus ?? result.attempts.at(-1)?.response?.status,
			});
			if (callAuthenticationFailed(result)) {
				authenticationFailed = true;
				break;
			}
			if (callDirectlySucceeded(result)) {
				connectivityWinner = seed;
				connectivityStable = true;
				break;
			}
			if (callSucceeded(result)) transientWinner ??= seed;
			if (input.signal?.aborted) throw input.signal.reason;
		}
		if (!connectivityWinner && !authenticationFailed) connectivityWinner = transientWinner;
		return {
			verdict: connectivityStable ? "accepted" : "indeterminate",
			evidence: { seeds: results, winner: connectivityWinner?.name, stable: connectivityStable },
		};
	});

	if (connectivityWinner) workingProfile = cloneCompat(connectivityWinner.compat);
	if (connectivityStable && connectivityWinner && connectivityWinner.name !== "conservative-control") {
		const winnerLeaves = collectLeaves(connectivityWinner.compat);
		for (const leaf of winnerLeaves) {
			const key = leaf.segments[0] as keyof OpenAICompat;
			if (!Object.hasOwn(OPENAI_COMPAT_CAPABILITY_REGISTRY, key)) continue;
			provisionalKeys.add(key);
			setFinding(
				key,
				"effective",
				"connectivity@probe",
				undefined,
				`Provisionally selected connectivity seed ${connectivityWinner.name}`,
			);
		}
	}

	if (connectivityWinner && !connectivityStable) {
		warnings.push(
			"Connectivity succeeded only after an automatic retry; capability probes will require stable evidence.",
		);
	}

	const runNotApplicableScenario = async (baseId: string): Promise<void> => {
		await runScenario(baseId, "probe", async () => ({ verdict: "not_applicable" }));
	};

	if (!connectivityWinner || authenticationFailed) {
		for (const baseId of OPENAI_COMPAT_PROBE_SCENARIO_IDS.slice(1)) {
			await runNotApplicableScenario(baseId);
		}
		if (authenticationFailed) warnings.push("Authentication failed; remaining capability scenarios were skipped.");
		else warnings.push("No connectivity seed produced a successful completion.");
		const report: OpenAICompatProbeReport = {
			schemaVersion: 1,
			generatedAt: new Date().toISOString(),
			target: {
				provider: sanitizeIdentifier(model.provider, input.apiKey),
				model: sanitizeIdentifier(model.id, input.apiKey),
				api: "openai-completions",
				baseUrl: sanitizeProbeUrl(model.baseUrl, input.apiKey),
			},
			configuredCompat,
			ignoredConfiguredKeys,
			resolvedBaseline: baselineModel.compat,
			recommendedCompat: {},
			resolvedRecommended: baselineModel.compat,
			removeConfiguredPaths: [],
			viable: false,
			fallbackFree: false,
			findings: OPENAI_COMPAT_KEYS.map(key => findings.get(key)!),
			scenarios,
			warnings,
		};
		return report;
	}

	await runScenario("store", "probe", async environment => {
		const enabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "supportsStore", true)),
			createBaselineContext(),
		);
		const disabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "supportsStore", false)),
			createBaselineContext(),
		);
		const shapePresent = Object.hasOwn(finalRequestBody(enabled) ?? {}, "store");
		const decision = pairDecision(enabled, disabled, false);
		if (!shapePresent && callDirectlySucceeded(enabled)) decision.verdict = "ignored";
		applyBooleanDecision("supportsStore", decision, "store@probe");
		return {
			verdict: decision.verdict,
			evidence: {
				enabledShapePresent: shapePresent,
				enabledPassing: callSucceeded(enabled),
				disabledPassing: callSucceeded(disabled),
			},
		};
	});

	await runScenario("streaming-usage", "probe", async environment => {
		const enabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "supportsUsageInStreaming", true)),
			createBaselineContext(),
		);
		const disabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "supportsUsageInStreaming", false)),
			createBaselineContext(),
		);
		let decision = semanticPairDecision(
			enabled,
			disabled,
			enabled.wire.rawUsageObserved,
			disabled.wire.rawUsageObserved,
		);
		if (
			callDirectlySucceeded(enabled) &&
			!enabled.wire.rawUsageObserved &&
			callDirectlySucceeded(disabled) &&
			!disabled.wire.rawUsageObserved
		) {
			decision = { verdict: "unsupported", recommendedValue: false };
		}
		applyBooleanDecision("supportsUsageInStreaming", decision, "streaming-usage@probe");
		return {
			verdict: decision.verdict,
			evidence: {
				enabledRawUsageObserved: enabled.wire.rawUsageObserved,
				disabledRawUsageObserved: disabled.wire.rawUsageObserved,
				includeUsageShapePresent: isPlainRecord(finalRequestBody(enabled)?.stream_options),
			},
		};
	});

	await runScenario("max-tokens", "probe", async environment => {
		if (model.omitMaxOutputTokens) {
			setFinding(
				"maxTokensField",
				"not_applicable",
				"max-tokens@probe",
				undefined,
				"Model omits output-token fields",
			);
			return { verdict: "not_applicable" };
		}
		const completion = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "maxTokensField", "max_completion_tokens")),
			createBaselineContext(),
			{ maxTokens: 64, maxTokensExplicit: true },
		);
		const legacy = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "maxTokensField", "max_tokens")),
			createBaselineContext(),
			{ maxTokens: 64, maxTokensExplicit: true },
		);
		const omittedModel = buildModel({
			...toModelSpec(buildTrialModel(workingProfile)),
			omitMaxOutputTokens: true,
		} as ModelSpec<"openai-completions">);
		const omitted = await environment.call(omittedModel, createBaselineContext(), {
			maxTokens: 64,
			maxTokensExplicit: true,
		});
		const completionPasses = callDirectlySucceeded(completion);
		const legacyPasses = callDirectlySucceeded(legacy);
		let verdict: OpenAICompatProbeVerdict = "indeterminate";
		if (completionPasses && legacyPasses) {
			verdict = "accepted";
			setFinding("maxTokensField", verdict, "max-tokens@probe");
		} else if ((completionPasses && callSchemaRejected(legacy)) || (legacyPasses && callSchemaRejected(completion))) {
			const value = completionPasses ? "max_completion_tokens" : "max_tokens";
			workingProfile = setCompatValue(workingProfile, "maxTokensField", value);
			verdict = "required";
			setFinding("maxTokensField", verdict, "max-tokens@probe", value);
		} else if (callSchemaRejected(completion) && callSchemaRejected(legacy) && callDirectlySucceeded(omitted)) {
			warnings.push("Both max-token fields were rejected; configure omitMaxOutputTokens: true on the model.");
			setFinding(
				"maxTokensField",
				"unsupported",
				"max-tokens@probe",
				undefined,
				"Requires non-compat omitMaxOutputTokens",
			);
			verdict = "unsupported";
		}
		return {
			verdict,
			evidence: {
				maxCompletionTokens: { passing: completionPasses, body: finalRequestBody(completion) },
				maxTokens: { passing: legacyPasses, body: finalRequestBody(legacy) },
				omittedPassing: callDirectlySucceeded(omitted),
			},
		};
	});

	await runScenario("always-max-tokens", "probe", async environment => {
		if (model.omitMaxOutputTokens) {
			setFinding("alwaysSendMaxTokens", "not_applicable", "always-max-tokens@probe");
			return { verdict: "not_applicable" };
		}
		const enabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "alwaysSendMaxTokens", true)),
			createBaselineContext(),
		);
		const disabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "alwaysSendMaxTokens", false)),
			createBaselineContext(),
		);
		const decision = pairDecision(enabled, disabled, false);
		applyBooleanDecision("alwaysSendMaxTokens", decision, "always-max-tokens@probe");
		return {
			verdict: decision.verdict,
			evidence: {
				enabledBody: finalRequestBody(enabled),
				disabledBody: finalRequestBody(disabled),
			},
		};
	});

	await runScenario("sampling-parameters", "probe", async environment => {
		const sampling: ProbeRequestOptions = {
			temperature: 0.2,
			topP: 0.9,
			topK: 20,
			minP: 0.05,
			presencePenalty: 0.1,
			repetitionPenalty: 1.05,
			frequencyPenalty: 0.1,
		};
		const enabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "supportsSamplingParams", true)),
			createBaselineContext(),
			sampling,
		);
		const disabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "supportsSamplingParams", false)),
			createBaselineContext(),
			sampling,
		);
		const body = finalRequestBody(enabled) ?? {};
		const expectedFields = [
			"temperature",
			"top_p",
			"top_k",
			"min_p",
			"presence_penalty",
			"repetition_penalty",
			"frequency_penalty",
		];
		const completeShape = expectedFields.every(field => Object.hasOwn(body, field));
		const decision = pairDecision(enabled, disabled, false);
		if (!completeShape && callDirectlySucceeded(enabled)) decision.verdict = "ignored";
		applyBooleanDecision("supportsSamplingParams", decision, "sampling-parameters@probe");
		return { verdict: decision.verdict, evidence: { completeShape, enabledBody: body } };
	});

	await runScenario("multiple-system-messages", "probe", async environment => {
		const context = createBaselineContext([FIRST_SYSTEM_PROMPT, SECOND_SYSTEM_PROMPT]);
		const enabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "supportsMultipleSystemMessages", true)),
			context,
		);
		const disabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "supportsMultipleSystemMessages", false)),
			context,
		);
		const decision = pairDecision(enabled, disabled, false);
		applyBooleanDecision("supportsMultipleSystemMessages", decision, "multiple-system-messages@probe");
		return {
			verdict: decision.verdict,
			evidence: { separateBody: finalRequestBody(enabled), coalescedBody: finalRequestBody(disabled) },
		};
	});

	await runScenario("developer-role", "probe", async environment => {
		if (!model.reasoning) return { verdict: "not_applicable" };
		const enabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "supportsDeveloperRole", true)),
			createBaselineContext(),
			{ reasoning: winningEffort ?? Effort.Medium },
		);
		const disabled = await environment.call(
			buildTrialModel(setCompatValue(workingProfile, "supportsDeveloperRole", false)),
			createBaselineContext(),
			{ reasoning: winningEffort ?? Effort.Medium },
		);
		const decision = pairDecision(enabled, disabled, false);
		applyBooleanDecision("supportsDeveloperRole", decision, "developer-role@probe");
		return {
			verdict: decision.verdict,
			evidence: { developerBody: finalRequestBody(enabled), systemBody: finalRequestBody(disabled) },
		};
	});

	let winningThinkingFormat = baselineModel.compat.thinkingFormat;
	await runScenario("reasoning-format", "probe", async environment => {
		if (!model.reasoning) return { verdict: "not_applicable" };
		const configuredFormat = configuredKnown.thinkingFormat;
		const orderedFormats = [
			configuredFormat,
			baselineModel.compat.thinkingFormat,
			"openai",
			"openrouter",
			"zai",
			"qwen",
			"qwen-chat-template",
		].filter(
			(value, index, values): value is NonNullable<OpenAICompat["thinkingFormat"]> =>
				value !== undefined && values.indexOf(value) === index,
		);
		const effort = model.thinking?.efforts.includes(Effort.Medium)
			? Effort.Medium
			: (model.thinking?.efforts[0] ?? Effort.Medium);
		const results: Array<{
			format: NonNullable<OpenAICompat["thinkingFormat"]>;
			enabled: ProbeCallResult;
			disabled: ProbeCallResult;
		}> = [];
		for (const format of orderedFormats) {
			let profile = deleteCompatValue(workingProfile, "reasoningDisableMode");
			profile = mergeProfiles(profile, {
				thinkingFormat: format,
				supportsReasoningParams: true,
				supportsReasoningEffort: true,
				omitReasoningEffort: false,
			});
			const enabled = await environment.call(buildTrialModel(profile), createBaselineContext(), {
				reasoning: effort,
			});
			const disabledProfile = setCompatValue(profile, "reasoningDisableMode", "omit");
			const disabled = await environment.call(buildTrialModel(disabledProfile), createBaselineContext(), {
				disableReasoning: true,
			});
			results.push({ format, enabled, disabled });
			if (callAuthenticationFailed(enabled) || callAuthenticationFailed(disabled)) {
				authenticationFailed = true;
				break;
			}
		}
		const semanticResults = results.filter(
			result =>
				callDirectlySucceeded(result.enabled) &&
				hasThinkingBlock(result.enabled.message) &&
				((callDirectlySucceeded(result.disabled) && !hasThinkingBlock(result.disabled.message)) ||
					callSchemaRejected(result.disabled)),
		);
		const baselineFormatResult = results.find(result => result.format === baselineModel.compat.thinkingFormat);
		const semantic =
			semanticResults.find(result => result.format === baselineModel.compat.thinkingFormat) ?? semanticResults[0];
		const accepted = results.find(result => callDirectlySucceeded(result.enabled));
		const winner = semantic ?? accepted;
		let verdict: OpenAICompatProbeVerdict = "indeterminate";
		const controlsRequired = semantic !== undefined && callSchemaRejected(semantic.disabled);
		if (winner) {
			if (semantic) {
				winningThinkingFormat = winner.format;
				winningEffort = effort;
			}
			verdict = semantic ? (controlsRequired ? "required" : "effective") : "accepted";
			if (semantic && winner.format === baselineModel.compat.thinkingFormat) {
				workingProfile = deleteCompatValue(workingProfile, "thinkingFormat");
				setFinding("thinkingFormat", verdict, "reasoning-format@probe");
			} else if (
				semantic &&
				winner.format !== baselineModel.compat.thinkingFormat &&
				baselineFormatResult &&
				callSchemaRejected(baselineFormatResult.enabled)
			) {
				workingProfile = setCompatValue(workingProfile, "thinkingFormat", winner.format);
				setFinding("thinkingFormat", "required", "reasoning-format@probe", winner.format);
			} else if (semantic && winner.format !== baselineModel.compat.thinkingFormat) {
				workingProfile = setCompatValue(workingProfile, "thinkingFormat", winner.format);
				setFinding("thinkingFormat", verdict, "reasoning-format@probe", winner.format);
			} else {
				setFinding("thinkingFormat", verdict, "reasoning-format@probe");
			}
			if (semantic) {
				workingProfile = setCompatValue(workingProfile, "supportsReasoningParams", true);
				setFinding("supportsReasoningParams", verdict, "reasoning-format@probe", true);
			} else {
				setFinding("supportsReasoningParams", verdict, "reasoning-format@probe");
			}
		} else if (results.some(result => callSchemaRejected(result.enabled) && callDirectlySucceeded(result.disabled))) {
			workingProfile = setCompatValue(workingProfile, "supportsReasoningParams", false);
			setFinding("supportsReasoningParams", "unsupported", "reasoning-format@probe", false);
			verdict = "unsupported";
		}
		return {
			verdict,
			evidence: {
				formats: results.map(result => ({
					format: result.format,
					enabledPassing: callSucceeded(result.enabled),
					disabledPassing: callSucceeded(result.disabled),
					parsedThinking: hasThinkingBlock(result.enabled.message),
					disabledParsedThinking: hasThinkingBlock(result.disabled.message),
					controlRejected: callSchemaRejected(result.disabled),
					fallback: callHasCompatibilityFallback(result.enabled),
					body: finalRequestBody(result.enabled),
				})),
			},
		};
	});

	await runScenario("reasoning-effort", "probe", async environment => {
		if (!model.reasoning) return { verdict: "not_applicable" };
		const profile = mergeProfiles(workingProfile, {
			thinkingFormat: winningThinkingFormat,
			supportsReasoningParams: true,
			supportsReasoningEffort: true,
			omitReasoningEffort: false,
		});
		const noEffortProfile = setCompatValue(profile, "reasoningDisableMode", "omit");
		const noEffort = await environment.call(buildTrialModel(noEffortProfile), createBaselineContext());
		const effortResults: Array<{ effort: Effort; result: ProbeCallResult }> = [];
		for (const effort of CANONICAL_EFFORTS) {
			const result = await environment.call(buildTrialModel(profile), createBaselineContext(), {
				reasoning: effort,
			});
			effortResults.push({ effort, result });
			if (callHasCompatibilityFallback(result) && !fallbackProbe) {
				fallbackProbe = {
					model: buildTrialModel(profile),
					context: createBaselineContext(),
					options: { reasoning: effort },
				};
			}
		}
		const direct = effortResults.filter(
			({ result }) => callDirectlySucceeded(result) && rawFinalReasoningEffort(result) !== undefined,
		);
		const noEffortRejected = callSchemaRejected(noEffort);
		const semanticDirect =
			noEffortRejected || (callDirectlySucceeded(noEffort) && !hasThinkingBlock(noEffort.message))
				? direct.filter(({ result }) => hasThinkingBlock(result.message))
				: [];
		const exposed = model.thinking?.efforts ?? [];
		winningEffort =
			semanticDirect.find(({ effort }) => effort === Effort.Medium && exposed.includes(effort))?.effort ??
			semanticDirect.find(({ effort }) => exposed.includes(effort))?.effort ??
			semanticDirect[0]?.effort;
		const effortMap: NonNullable<OpenAICompat["reasoningEffortMap"]> = {
			...(profile.reasoningEffortMap ?? {}),
		};
		const discoveredEffortMap: NonNullable<OpenAICompat["reasoningEffortMap"]> = {};
		const mappingResults: Array<{ source: Effort; target: Effort; result: ProbeCallResult }> = [];
		const availableTargets = semanticDirect.map(({ effort }) => effort).filter(effort => exposed.includes(effort));
		let effortMapChanged = false;
		for (const { effort: source, result } of effortResults) {
			const reasoningFallback = result.attempts.some(attempt => attempt.classification === "reasoning_fallback");
			if (
				callDirectlySucceeded(result) ||
				callTransientlyFailed(result) ||
				(!callSchemaRejected(result) && !reasoningFallback)
			) {
				continue;
			}
			const wireTarget = rawFinalReasoningEffort(result);
			const fallbackTarget = CANONICAL_EFFORTS.find(effort => effort === wireTarget && exposed.includes(effort));
			const sourceIndex = CANONICAL_EFFORTS.indexOf(source);
			const targets = [...availableTargets].sort((left, right) => {
				if (left === fallbackTarget) return -1;
				if (right === fallbackTarget) return 1;
				return (
					Math.abs(CANONICAL_EFFORTS.indexOf(left) - sourceIndex) -
						Math.abs(CANONICAL_EFFORTS.indexOf(right) - sourceIndex) ||
					CANONICAL_EFFORTS.indexOf(left) - CANONICAL_EFFORTS.indexOf(right)
				);
			});
			for (const target of targets) {
				const candidateMap = { ...effortMap, [source]: target };
				const mappedResult = await environment.call(
					buildTrialModel(mergeProfiles(profile, { reasoningEffortMap: candidateMap })),
					createBaselineContext(),
					{ reasoning: source },
				);
				mappingResults.push({ source, target, result: mappedResult });
				if (
					!callDirectlySucceeded(mappedResult) ||
					rawFinalReasoningEffort(mappedResult) !== target ||
					!hasThinkingBlock(mappedResult.message)
				) {
					continue;
				}
				effortMap[source] = target;
				discoveredEffortMap[source] = target;
				effortMapChanged = true;
				break;
			}
		}
		let verdict: OpenAICompatProbeVerdict = "indeterminate";
		if (semanticDirect.length > 0) {
			verdict = noEffortRejected ? "required" : "effective";
			workingProfile = mergeProfiles(workingProfile, {
				supportsReasoningParams: true,
				supportsReasoningEffort: true,
				omitReasoningEffort: false,
			});
			setFinding("supportsReasoningParams", verdict, "reasoning-effort@probe", true);
			setFinding("supportsReasoningEffort", verdict, "reasoning-effort@probe", true);
			setFinding(
				"omitReasoningEffort",
				noEffortRejected || baselineModel.compat.omitReasoningEffort ? "required" : "accepted",
				"reasoning-effort@probe",
				noEffortRejected || baselineModel.compat.omitReasoningEffort ? false : undefined,
			);
		} else if (direct.length > 0) {
			verdict = "accepted";
			if (findings.get("supportsReasoningParams")?.recommendedValue === undefined) {
				setFinding("supportsReasoningParams", "accepted", "reasoning-effort@probe");
			}
			if (findings.get("supportsReasoningEffort")?.recommendedValue === undefined) {
				setFinding("supportsReasoningEffort", "accepted", "reasoning-effort@probe");
			}
			if (findings.get("omitReasoningEffort")?.recommendedValue === undefined) {
				setFinding("omitReasoningEffort", "accepted", "reasoning-effort@probe");
			}
		} else {
			const fallback = effortResults.find(({ result }) => callSucceededViaCompatibilityFallback(result));
			if (fallback) {
				workingProfile = mergeProfiles(workingProfile, {
					supportsReasoningEffort: false,
					omitReasoningEffort: true,
				});
				setFinding("supportsReasoningEffort", "fallback_only", "reasoning-effort@probe", false);
				setFinding("omitReasoningEffort", "fallback_only", "reasoning-effort@probe", true);
				verdict = "fallback_only";
			}
		}
		if (effortMapChanged) {
			workingProfile = mergeProfiles(workingProfile, {
				supportsReasoningParams: true,
				supportsReasoningEffort: true,
				omitReasoningEffort: false,
				reasoningEffortMap: effortMap,
			});
			setFinding("reasoningEffortMap", "required", "reasoning-effort@probe", discoveredEffortMap);
			verdict = "required";
		} else if (effortResults.every(({ result }) => callDirectlySucceeded(result))) {
			setFinding("reasoningEffortMap", "accepted", "reasoning-effort@probe");
		}
		return {
			verdict,
			evidence: {
				noEffort: {
					passing: callDirectlySucceeded(noEffort),
					parsedThinking: hasThinkingBlock(noEffort.message),
					rejected: noEffortRejected,
					body: finalRequestBody(noEffort),
				},
				efforts: effortResults.map(({ effort, result }) => ({
					effort,
					passing: callSucceeded(result),
					parsedThinking: hasThinkingBlock(result.message),
					fallback: callHasCompatibilityFallback(result),
					body: finalRequestBody(result),
				})),
				mappings: mappingResults.map(({ source, target, result }) => ({
					source,
					target,
					passing: callDirectlySucceeded(result),
					body: finalRequestBody(result),
				})),
				winningEffort,
			},
		};
	});

	await runScenario("reasoning-disable", "probe", async environment => {
		if (!model.reasoning) return { verdict: "not_applicable" };
		const modes: NonNullable<OpenAICompat["reasoningDisableMode"]>[] = [
			"omit",
			"lowest-effort",
			"openrouter-enabled-false",
			"zai-thinking-disabled",
			"qwen-enable-thinking-false",
			"qwen-template-false",
		];
		const results: Array<{ mode: NonNullable<OpenAICompat["reasoningDisableMode"]>; result: ProbeCallResult }> = [];
		for (const mode of modes) {
			const profile = mergeProfiles(workingProfile, {
				thinkingFormat: winningThinkingFormat,
				reasoningDisableMode: mode,
				supportsReasoningParams: true,
			});
			results.push({
				mode,
				result: await environment.call(buildTrialModel(profile), createBaselineContext(), {
					disableReasoning: true,
				}),
			});
		}
		const autoMode = baselineModel.compat.reasoningDisableMode;
		const auto = results.find(result => result.mode === autoMode);
		const winner =
			(auto && callDirectlySucceeded(auto.result) ? auto : undefined) ??
			results.find(result => callDirectlySucceeded(result.result));
		let verdict: OpenAICompatProbeVerdict = "indeterminate";
		if (winner !== undefined && winner === auto) {
			verdict = "accepted";
			setFinding("reasoningDisableMode", "accepted", "reasoning-disable@probe");
		} else if (winner && auto && callSchemaRejected(auto.result)) {
			workingProfile = setCompatValue(workingProfile, "reasoningDisableMode", winner.mode);
			setFinding("reasoningDisableMode", "required", "reasoning-disable@probe", winner.mode);
			verdict = "required";
		}
		return {
			verdict,
			evidence: {
				modes: results.map(({ mode, result }) => ({
					mode,
					passing: callSucceeded(result),
					body: finalRequestBody(result),
				})),
			},
		};
	});

	let toolChoiceForValidation: ToolChoice = "auto";
	await runScenario("tool-choice", "probe", async environment => {
		const noChoice = await environment.call(buildTrialModel(workingProfile), createToolContext());
		const autoResults: ProbeCallResult[] = [];
		for (let attempt = 0; attempt < 3; attempt += 1) {
			const result = await environment.call(
				buildTrialModel(setCompatValue(workingProfile, "supportsToolChoice", true)),
				createToolContext(),
				{ toolChoice: "auto" },
			);
			autoResults.push(result);
			if (callDirectlySucceeded(result) && hasCompatEchoCall(result.message)) break;
		}
		const auto = autoResults.at(-1)!;
		const required = await environment.call(
			buildTrialModel(mergeProfiles(workingProfile, { supportsToolChoice: true, supportsForcedToolChoice: true })),
			createToolContext(),
			{ toolChoice: "required" },
		);
		const named = await environment.call(
			buildTrialModel(
				mergeProfiles(workingProfile, {
					supportsToolChoice: true,
					supportsForcedToolChoice: true,
					supportsNamedToolChoice: true,
				}),
			),
			createToolContext(),
			{ toolChoice: { type: "function", function: { name: "compat_echo" } } },
		);
		const autoHttpOutcomes = autoResults.map(callSucceeded);
		const autoDisagrees =
			autoResults.some(callTransientlyFailed) ||
			(autoHttpOutcomes.some(Boolean) && autoHttpOutcomes.some(value => !value));
		const toolChoiceDecision = semanticPairDecision(
			auto,
			noChoice,
			hasCompatEchoCall(auto.message),
			hasCompatEchoCall(noChoice.message),
		);
		if (autoDisagrees) {
			toolChoiceDecision.verdict = "indeterminate";
			delete toolChoiceDecision.recommendedValue;
		}
		applyBooleanDecision("supportsToolChoice", toolChoiceDecision, "tool-choice@probe");
		const forcedDecision = semanticPairDecision(
			required,
			auto,
			hasCompatEchoCall(required.message),
			hasCompatEchoCall(auto.message),
		);
		applyBooleanDecision("supportsForcedToolChoice", forcedDecision, "tool-choice@probe");
		if (forcedDecision.recommendedValue === true) {
			workingProfile = setCompatValue(workingProfile, "supportsToolChoice", true);
			setFinding("supportsToolChoice", "required", "tool-choice@probe", true);
		}
		const namedDecision = semanticPairDecision(
			named,
			required,
			hasCompatEchoCall(named.message),
			hasCompatEchoCall(required.message),
		);
		applyBooleanDecision("supportsNamedToolChoice", namedDecision, "tool-choice@probe");
		if (namedDecision.recommendedValue === true) {
			workingProfile = mergeProfiles(workingProfile, {
				supportsToolChoice: true,
				supportsForcedToolChoice: true,
				supportsNamedToolChoice: true,
			});
			setFinding("supportsToolChoice", "required", "tool-choice@probe", true);
			setFinding("supportsForcedToolChoice", "required", "tool-choice@probe", true);
			setFinding("supportsNamedToolChoice", "required", "tool-choice@probe", true);
		}
		toolChoiceForValidation = resolveValidationToolChoice(buildTrialModel(workingProfile).compat);
		const decisions = [toolChoiceDecision, forcedDecision, namedDecision];
		const verdict: OpenAICompatProbeVerdict = decisions.some(decision => decision.verdict === "indeterminate")
			? "indeterminate"
			: decisions.some(decision => decision.verdict === "fallback_only")
				? "fallback_only"
				: decisions.some(decision => decision.verdict === "unsupported")
					? "unsupported"
					: decisions.some(decision => decision.verdict === "required")
						? "required"
						: [named, required, auto].some(
									result => callDirectlySucceeded(result) && hasCompatEchoCall(result.message),
								)
							? "effective"
							: [named, required, auto, noChoice].some(callDirectlySucceeded)
								? "accepted"
								: "indeterminate";
		return {
			verdict,
			evidence: {
				noChoice: {
					passing: callSucceeded(noChoice),
					semantic: hasCompatEchoCall(noChoice.message),
					body: finalRequestBody(noChoice),
				},
				auto: {
					attempts: autoResults.length,
					successCount: autoResults.filter(result => hasCompatEchoCall(result.message)).length,
					body: finalRequestBody(auto),
				},
				required: {
					passing: callSucceeded(required),
					semantic: hasCompatEchoCall(required.message),
					body: finalRequestBody(required),
				},
				named: {
					passing: callSucceeded(named),
					semantic: hasCompatEchoCall(named.message),
					body: finalRequestBody(named),
				},
			},
		};
	});

	await runScenario("tool-strictness", "probe", async environment => {
		let mixedProfile = mergeProfiles(workingProfile, { supportsStrictMode: true });
		mixedProfile = deleteCompatValue(mixedProfile, "toolStrictMode");
		const mixed = await environment.call(
			buildTrialModel(mixedProfile),
			createToolContext([COMPAT_ECHO_TOOL, COMPAT_ECHO_NON_STRICT_TOOL]),
			{ toolChoice: "auto" },
		);
		const allStrict = await environment.call(
			buildTrialModel(mergeProfiles(workingProfile, { supportsStrictMode: true, toolStrictMode: "all_strict" })),
			createToolContext([COMPAT_ECHO_TOOL, { ...COMPAT_ECHO_NON_STRICT_TOOL, strict: true }]),
			{ toolChoice: "auto" },
		);
		const none = await environment.call(
			buildTrialModel(mergeProfiles(workingProfile, { supportsStrictMode: true, toolStrictMode: "none" })),
			createToolContext([COMPAT_ECHO_TOOL, COMPAT_ECHO_NON_STRICT_TOOL]),
			{ toolChoice: "auto" },
		);
		const mixedSemantic = callDirectlySucceeded(mixed) && hasCompatEchoCall(mixed.message);
		const allStrictSemantic = callDirectlySucceeded(allStrict) && hasCompatEchoCall(allStrict.message);
		const noneSemantic = callDirectlySucceeded(none) && hasCompatEchoCall(none.message);
		const mixedFallbackSemantic = callSucceededViaCompatibilityFallback(mixed) && hasCompatEchoCall(mixed.message);
		const mixedComparable =
			callDirectlySucceeded(mixed) || callSchemaRejected(mixed) || callSucceededViaCompatibilityFallback(mixed);
		const allStrictComparable =
			callDirectlySucceeded(allStrict) ||
			callSchemaRejected(allStrict) ||
			callSucceededViaCompatibilityFallback(allStrict);
		let verdict: OpenAICompatProbeVerdict = "indeterminate";
		if (mixedSemantic) {
			workingProfile = setCompatValue(workingProfile, "supportsStrictMode", true);
			workingProfile = deleteCompatValue(workingProfile, "toolStrictMode");
			provisionalKeys.add("supportsStrictMode");
			setFinding("supportsStrictMode", "effective", "tool-strictness@probe");
			setFinding("toolStrictMode", "accepted", "tool-strictness@probe");
			verdict = "accepted";
		} else if (mixedComparable && allStrictSemantic) {
			workingProfile = setCompatValue(workingProfile, "supportsStrictMode", true);
			setFinding("supportsStrictMode", "required", "tool-strictness@probe", true);
			workingProfile = setCompatValue(workingProfile, "toolStrictMode", "all_strict");
			setFinding("toolStrictMode", "required", "tool-strictness@probe", "all_strict");
			verdict = "required";
		} else if (mixedComparable && allStrictComparable && noneSemantic) {
			workingProfile = setCompatValue(workingProfile, "toolStrictMode", "none");
			setFinding("supportsStrictMode", "accepted", "tool-strictness@probe");
			setFinding("toolStrictMode", "required", "tool-strictness@probe", "none");
			verdict = "required";
		} else if (mixedFallbackSemantic) {
			workingProfile = setCompatValue(workingProfile, "supportsStrictMode", false);
			setFinding("supportsStrictMode", "fallback_only", "tool-strictness@probe", false);
			verdict = "fallback_only";
			fallbackProbe ??= {
				model: buildTrialModel(mixedProfile),
				context: createToolContext([COMPAT_ECHO_TOOL, COMPAT_ECHO_NON_STRICT_TOOL]),
				options: { toolChoice: "auto" },
			};
		} else if (callDirectlySucceeded(mixed)) {
			setFinding("supportsStrictMode", "accepted", "tool-strictness@probe");
			setFinding("toolStrictMode", "accepted", "tool-strictness@probe");
			verdict = "accepted";
		}
		return {
			verdict,
			evidence: {
				mixed: {
					passing: callSucceeded(mixed),
					fallback: callHasCompatibilityFallback(mixed),
					semantic: hasCompatEchoCall(mixed.message),
					body: finalRequestBody(mixed),
				},
				allStrict: {
					passing: callSucceeded(allStrict),
					fallback: callHasCompatibilityFallback(allStrict),
					semantic: hasCompatEchoCall(allStrict.message),
					body: finalRequestBody(allStrict),
				},
				none: {
					passing: callSucceeded(none),
					semantic: hasCompatEchoCall(none.message),
					body: finalRequestBody(none),
				},
			},
		};
	});

	await runScenario("tool-schema-flavor", "probe", async environment => {
		const flavors: NonNullable<OpenAICompat["toolSchemaFlavor"]>[] = ["none", "moonshot-mfjs"];
		const results: Array<{ flavor: NonNullable<OpenAICompat["toolSchemaFlavor"]>; result: ProbeCallResult }> = [];
		for (const flavor of flavors) {
			const result = await environment.call(
				buildTrialModel(setCompatValue(workingProfile, "toolSchemaFlavor", flavor)),
				createToolContext(),
				{ toolChoice: toolChoiceForValidation },
			);
			results.push({ flavor, result });
		}
		const autoFlavor = baselineModel.compat.toolSchemaFlavor ?? "none";
		const auto = results.find(result => result.flavor === autoFlavor);
		const semanticWinner =
			(auto && callDirectlySucceeded(auto.result) && hasCompatEchoCall(auto.result.message) ? auto : undefined) ??
			results.find(result => callDirectlySucceeded(result.result) && hasCompatEchoCall(result.result.message));
		const acceptedWinner =
			(auto && callDirectlySucceeded(auto.result) ? auto : undefined) ??
			results.find(result => callDirectlySucceeded(result.result));
		const winner = semanticWinner ?? acceptedWinner;
		let verdict: OpenAICompatProbeVerdict = "indeterminate";
		if (semanticWinner === auto) {
			verdict = "effective";
			setFinding("toolSchemaFlavor", "effective", "tool-schema-flavor@probe");
		} else if (semanticWinner && auto) {
			workingProfile = setCompatValue(workingProfile, "toolSchemaFlavor", semanticWinner.flavor);
			verdict = callSchemaRejected(auto.result) ? "required" : "effective";
			setFinding("toolSchemaFlavor", verdict, "tool-schema-flavor@probe", semanticWinner.flavor);
		} else if (winner === auto) {
			verdict = "accepted";
			setFinding("toolSchemaFlavor", "accepted", "tool-schema-flavor@probe");
		} else if (winner && auto && callSchemaRejected(auto.result)) {
			workingProfile = setCompatValue(workingProfile, "toolSchemaFlavor", winner.flavor);
			setFinding("toolSchemaFlavor", "required", "tool-schema-flavor@probe", winner.flavor);
			verdict = "required";
		}
		return {
			verdict,
			evidence: {
				flavors: results.map(({ flavor, result }) => ({
					flavor,
					passing: callSucceeded(result),
					semantic: hasCompatEchoCall(result.message),
					body: finalRequestBody(result),
				})),
			},
		};
	});

	await runScenario("history-tool-result-name", "probe", async environment => {
		const enabledProfile = setCompatValue(workingProfile, "requiresToolResultName", true);
		const disabledProfile = setCompatValue(workingProfile, "requiresToolResultName", false);
		const enabled = await environment.call(
			buildTrialModel(enabledProfile),
			createHistoryContext(buildTrialModel(enabledProfile)),
		);
		const disabled = await environment.call(
			buildTrialModel(disabledProfile),
			createHistoryContext(buildTrialModel(disabledProfile)),
		);
		const decision = pairDecision(enabled, disabled, false);
		applyBooleanDecision("requiresToolResultName", decision, "history-tool-result-name@probe");
		return {
			verdict: decision.verdict,
			evidence: { withName: finalRequestBody(enabled), withoutName: finalRequestBody(disabled) },
		};
	});

	await runScenario("history-assistant-after-tool-result", "probe", async environment => {
		const enabledProfile = setCompatValue(workingProfile, "requiresAssistantAfterToolResult", true);
		const disabledProfile = setCompatValue(workingProfile, "requiresAssistantAfterToolResult", false);
		const enabled = await environment.call(
			buildTrialModel(enabledProfile),
			createHistoryContext(buildTrialModel(enabledProfile)),
		);
		const disabled = await environment.call(
			buildTrialModel(disabledProfile),
			createHistoryContext(buildTrialModel(disabledProfile)),
		);
		const decision = pairDecision(enabled, disabled, false);
		applyBooleanDecision("requiresAssistantAfterToolResult", decision, "history-assistant-after-tool-result@probe");
		return {
			verdict: decision.verdict,
			evidence: { inserted: finalRequestBody(enabled), direct: finalRequestBody(disabled) },
		};
	});

	await runScenario("history-thinking-as-text", "probe", async environment => {
		if (!model.reasoning) return { verdict: "not_applicable" };
		const enabledProfile = setCompatValue(workingProfile, "requiresThinkingAsText", true);
		const disabledProfile = setCompatValue(workingProfile, "requiresThinkingAsText", false);
		const enabledModel = buildTrialModel(enabledProfile);
		const disabledModel = buildTrialModel(disabledProfile);
		const enabled = await environment.call(
			enabledModel,
			createHistoryContext(enabledModel, { includeReasoning: true, includePlainReasoning: true }),
			{ reasoning: winningEffort },
		);
		const disabled = await environment.call(
			disabledModel,
			createHistoryContext(disabledModel, { includeReasoning: true, includePlainReasoning: true }),
			{ reasoning: winningEffort },
		);
		const decision = pairDecision(enabled, disabled, false);
		applyBooleanDecision("requiresThinkingAsText", decision, "history-thinking-as-text@probe");
		return {
			verdict: decision.verdict,
			evidence: { textEncoding: finalRequestBody(enabled), nativeEncoding: finalRequestBody(disabled) },
		};
	});

	await runScenario("history-tool-call-id", "probe", async environment => {
		const rawProfile = mergeProfiles(workingProfile, {
			requiresMistralToolIds: false,
			usesOpenAIToolCallIdLimit: false,
		});
		const mistralProfile = mergeProfiles(workingProfile, {
			requiresMistralToolIds: true,
			usesOpenAIToolCallIdLimit: false,
		});
		const openAIProfile = mergeProfiles(workingProfile, {
			requiresMistralToolIds: false,
			usesOpenAIToolCallIdLimit: true,
		});
		const rawModel = buildTrialModel(rawProfile);
		const mistralModel = buildTrialModel(mistralProfile);
		const openAIModel = buildTrialModel(openAIProfile);
		const raw = await environment.call(rawModel, createHistoryContext(rawModel));
		const mistral = await environment.call(mistralModel, createHistoryContext(mistralModel));
		const openAI = await environment.call(openAIModel, createHistoryContext(openAIModel));
		let verdict: OpenAICompatProbeVerdict = "indeterminate";
		if (callSchemaRejected(raw) && callDirectlySucceeded(mistral)) {
			workingProfile = mergeProfiles(workingProfile, {
				requiresMistralToolIds: true,
				usesOpenAIToolCallIdLimit: false,
			});
			setFinding("requiresMistralToolIds", "required", "history-tool-call-id@probe", true);
			setFinding("usesOpenAIToolCallIdLimit", "unsupported", "history-tool-call-id@probe", false);
			verdict = "required";
		} else if (callSchemaRejected(raw) && callDirectlySucceeded(openAI)) {
			workingProfile = mergeProfiles(workingProfile, {
				requiresMistralToolIds: false,
				usesOpenAIToolCallIdLimit: true,
			});
			setFinding("usesOpenAIToolCallIdLimit", "required", "history-tool-call-id@probe", true);
			setFinding("requiresMistralToolIds", "unsupported", "history-tool-call-id@probe", false);
			verdict = "required";
		} else {
			verdict = callDirectlySucceeded(raw) ? "accepted" : "indeterminate";
			setFinding(
				"requiresMistralToolIds",
				callDirectlySucceeded(raw) ? "accepted" : "indeterminate",
				"history-tool-call-id@probe",
			);
			setFinding(
				"usesOpenAIToolCallIdLimit",
				callDirectlySucceeded(raw) ? "accepted" : "indeterminate",
				"history-tool-call-id@probe",
			);
		}
		return {
			verdict,
			evidence: {
				raw: finalRequestBody(raw),
				mistral9: finalRequestBody(mistral),
				openai40: finalRequestBody(openAI),
			},
		};
	});

	await runScenario("history-reasoning-content", "probe", async environment => {
		if (!model.reasoning) return { verdict: "not_applicable" };
		const fields: NonNullable<OpenAICompat["reasoningContentField"]>[] = [
			"reasoning_content",
			"reasoning",
			"reasoning_text",
		];
		const results: Array<{ field: NonNullable<OpenAICompat["reasoningContentField"]>; result: ProbeCallResult }> = [];
		for (const field of fields) {
			const profile = mergeProfiles(workingProfile, {
				reasoningContentField: field,
				requiresReasoningContentForToolCalls: true,
				allowsSyntheticReasoningContentForToolCalls: true,
			});
			const trial = buildTrialModel(profile);
			results.push({
				field,
				result: await environment.call(trial, createHistoryContext(trial, { includeReasoning: true }), {
					reasoning: winningEffort,
				}),
			});
		}
		const baselineField = baselineModel.compat.reasoningContentField ?? "reasoning_content";
		const baseline = results.find(result => result.field === baselineField);
		const winner =
			(baseline && callDirectlySucceeded(baseline.result) ? baseline : undefined) ??
			results.find(result => callDirectlySucceeded(result.result));
		let fieldVerdict: OpenAICompatProbeVerdict = "indeterminate";
		if (winner !== undefined && winner === baseline) {
			fieldVerdict = "accepted";
			setFinding("reasoningContentField", "accepted", "history-reasoning-content@probe");
		} else if (winner && baseline && callSchemaRejected(baseline.result)) {
			workingProfile = setCompatValue(workingProfile, "reasoningContentField", winner.field);
			setFinding("reasoningContentField", "required", "history-reasoning-content@probe", winner.field);
			fieldVerdict = "required";
		}

		const selectedField = winner?.field ?? baselineField;
		const requiredProfile = mergeProfiles(workingProfile, {
			reasoningContentField: selectedField,
			requiresReasoningContentForToolCalls: true,
			allowsSyntheticReasoningContentForToolCalls: true,
		});
		const omittedProfile = mergeProfiles(workingProfile, {
			reasoningContentField: selectedField,
			requiresReasoningContentForToolCalls: false,
			allowsSyntheticReasoningContentForToolCalls: true,
		});
		const requiredModel = buildTrialModel(requiredProfile);
		const omittedModel = buildTrialModel(omittedProfile);
		const required = await environment.call(requiredModel, createHistoryContext(requiredModel), {
			reasoning: winningEffort,
		});
		const omitted = await environment.call(omittedModel, createHistoryContext(omittedModel), {
			reasoning: winningEffort,
		});
		const requiredDecision = pairDecision(required, omitted, false);
		applyBooleanDecision("requiresReasoningContentForToolCalls", requiredDecision, "history-reasoning-content@probe");

		const syntheticProfile = mergeProfiles(requiredProfile, {
			allowsSyntheticReasoningContentForToolCalls: true,
		});
		const exactProfile = mergeProfiles(requiredProfile, {
			allowsSyntheticReasoningContentForToolCalls: false,
		});
		const syntheticModel = buildTrialModel(syntheticProfile);
		const exactModel = buildTrialModel(exactProfile);
		const synthetic = await environment.call(syntheticModel, createHistoryContext(syntheticModel), {
			reasoning: winningEffort,
		});
		const exact = await environment.call(exactModel, createHistoryContext(exactModel), {
			reasoning: winningEffort,
		});
		const syntheticDecision = pairDecision(synthetic, exact, false);
		applyBooleanDecision(
			"allowsSyntheticReasoningContentForToolCalls",
			syntheticDecision,
			"history-reasoning-content@probe",
		);

		const verdicts = [fieldVerdict, requiredDecision.verdict, syntheticDecision.verdict];
		const verdict: OpenAICompatProbeVerdict = verdicts.includes("indeterminate")
			? "indeterminate"
			: verdicts.includes("fallback_only")
				? "fallback_only"
				: verdicts.includes("unsupported")
					? "unsupported"
					: verdicts.includes("required")
						? "required"
						: "accepted";
		return {
			verdict,
			evidence: {
				fields: results.map(({ field, result }) => ({
					field,
					passing: callSucceeded(result),
					body: finalRequestBody(result),
				})),
				requirement: {
					required: finalRequestBody(required),
					omitted: finalRequestBody(omitted),
				},
				synthetic: {
					placeholder: finalRequestBody(synthetic),
					exact: finalRequestBody(exact),
				},
			},
		};
	});

	await runScenario("history-reasoning-all-assistant", "probe", async environment => {
		if (!model.reasoning) return { verdict: "not_applicable" };
		const commonProfile = {
			requiresReasoningContentForToolCalls: true,
			allowsSyntheticReasoningContentForToolCalls: false,
			replayReasoningContent: false,
		} satisfies OpenAICompat;
		const enabledProfile = mergeProfiles(workingProfile, {
			...commonProfile,
			requiresReasoningContentForAllAssistantTurns: true,
		});
		const disabledProfile = mergeProfiles(workingProfile, {
			...commonProfile,
			requiresReasoningContentForAllAssistantTurns: false,
		});
		const enabledModel = buildTrialModel(enabledProfile);
		const disabledModel = buildTrialModel(disabledProfile);
		const enabled = await environment.call(
			enabledModel,
			createHistoryContext(enabledModel, { includeReasoning: true }),
			{ reasoning: winningEffort },
		);
		const disabled = await environment.call(
			disabledModel,
			createHistoryContext(disabledModel, { includeReasoning: true }),
			{ reasoning: winningEffort },
		);
		const decision = pairDecision(enabled, disabled, false);
		applyBooleanDecision(
			"requiresReasoningContentForAllAssistantTurns",
			decision,
			"history-reasoning-all-assistant@probe",
		);
		return {
			verdict: decision.verdict,
			evidence: { allAssistant: finalRequestBody(enabled), toolOnly: finalRequestBody(disabled) },
		};
	});

	await runScenario("history-reasoning-replay", "probe", async environment => {
		if (!model.reasoning) return { verdict: "not_applicable" };
		const replayEnabledProfile = mergeProfiles(workingProfile, {
			requiresReasoningContentForToolCalls: false,
			requiresReasoningContentForAllAssistantTurns: false,
			replayReasoningContent: true,
			qwenPreserveThinking: false,
		});
		const replayDisabledProfile = mergeProfiles(workingProfile, {
			requiresReasoningContentForToolCalls: false,
			requiresReasoningContentForAllAssistantTurns: false,
			replayReasoningContent: false,
			qwenPreserveThinking: false,
		});
		const replayEnabledModel = buildTrialModel(replayEnabledProfile);
		const replayDisabledModel = buildTrialModel(replayDisabledProfile);
		const replayEnabled = await environment.call(
			replayEnabledModel,
			createHistoryContext(replayEnabledModel, { includeReasoning: true }),
			{ reasoning: winningEffort },
		);
		const replayDisabled = await environment.call(
			replayDisabledModel,
			createHistoryContext(replayDisabledModel, { includeReasoning: true }),
			{ reasoning: winningEffort },
		);
		const replayDecision = pairDecision(replayEnabled, replayDisabled, false);
		if (replayDecision.verdict === "accepted") {
			setFinding(
				"replayReasoningContent",
				"indeterminate",
				"history-reasoning-replay@probe",
				configuredKnown.replayReasoningContent,
				"Ordinary responses cannot prove cache-preserving replay",
			);
		} else {
			applyBooleanDecision("replayReasoningContent", replayDecision, "history-reasoning-replay@probe");
		}

		const qwenEnabledProfile = mergeProfiles(workingProfile, {
			replayReasoningContent: false,
			qwenPreserveThinking: true,
		});
		const qwenDisabledProfile = mergeProfiles(workingProfile, {
			replayReasoningContent: false,
			qwenPreserveThinking: false,
		});
		const qwenEnabledModel = buildTrialModel(qwenEnabledProfile);
		const qwenDisabledModel = buildTrialModel(qwenDisabledProfile);
		const qwenEnabled = await environment.call(
			qwenEnabledModel,
			createHistoryContext(qwenEnabledModel, { includeReasoning: true }),
			{ reasoning: winningEffort },
		);
		const qwenDisabled = await environment.call(
			qwenDisabledModel,
			createHistoryContext(qwenDisabledModel, { includeReasoning: true }),
			{ reasoning: winningEffort },
		);
		const qwenDecision = pairDecision(qwenEnabled, qwenDisabled, false);
		if (qwenDecision.verdict === "accepted") {
			setFinding(
				"qwenPreserveThinking",
				"indeterminate",
				"history-reasoning-replay@probe",
				configuredKnown.qwenPreserveThinking,
				"Ordinary responses cannot prove cache-preserving replay",
			);
		} else {
			applyBooleanDecision("qwenPreserveThinking", qwenDecision, "history-reasoning-replay@probe");
		}

		const verdicts = [
			replayDecision.verdict === "accepted" ? "indeterminate" : replayDecision.verdict,
			qwenDecision.verdict === "accepted" ? "indeterminate" : qwenDecision.verdict,
		];
		const verdict: OpenAICompatProbeVerdict = verdicts.includes("indeterminate")
			? "indeterminate"
			: verdicts.includes("fallback_only")
				? "fallback_only"
				: verdicts.includes("unsupported")
					? "unsupported"
					: verdicts.includes("required")
						? "required"
						: "accepted";
		return {
			verdict,
			evidence: {
				replay: {
					enabled: finalRequestBody(replayEnabled),
					disabled: finalRequestBody(replayDisabled),
				},
				qwenPreserve: {
					enabled: finalRequestBody(qwenEnabled),
					disabled: finalRequestBody(qwenDisabled),
				},
			},
		};
	});

	await runScenario("history-assistant-tool-content", "probe", async environment => {
		const enabledProfile = setCompatValue(workingProfile, "requiresAssistantContentForToolCalls", true);
		const disabledProfile = setCompatValue(workingProfile, "requiresAssistantContentForToolCalls", false);
		const enabledModel = buildTrialModel(enabledProfile);
		const disabledModel = buildTrialModel(disabledProfile);
		const enabled = await environment.call(
			enabledModel,
			createHistoryContext(enabledModel, { assistantToolContent: true }),
		);
		const disabled = await environment.call(disabledModel, createHistoryContext(disabledModel));
		const decision = pairDecision(enabled, disabled, false);
		applyBooleanDecision("requiresAssistantContentForToolCalls", decision, "history-assistant-tool-content@probe");
		return {
			verdict: decision.verdict,
			evidence: { nonempty: finalRequestBody(enabled), omitted: finalRequestBody(disabled) },
		};
	});

	await runScenario("reasoning-tool-interaction", "probe", async environment => {
		if (!model.reasoning) return { verdict: "not_applicable" };
		const falseProfile = mergeProfiles(workingProfile, {
			disableReasoningOnForcedToolChoice: false,
			disableReasoningOnToolChoice: false,
		});
		const forcedProfile = mergeProfiles(workingProfile, {
			disableReasoningOnForcedToolChoice: true,
			disableReasoningOnToolChoice: false,
		});
		const allProfile = mergeProfiles(workingProfile, {
			disableReasoningOnForcedToolChoice: false,
			disableReasoningOnToolChoice: true,
		});
		const direct = await environment.call(buildTrialModel(falseProfile), createToolContext(), {
			reasoning: winningEffort,
			toolChoice: toolChoiceForValidation,
		});
		const forced = await environment.call(buildTrialModel(forcedProfile), createToolContext(), {
			reasoning: winningEffort,
			toolChoice: toolChoiceForValidation,
		});
		const all = await environment.call(buildTrialModel(allProfile), createToolContext(), {
			reasoning: winningEffort,
			toolChoice: "auto",
		});
		let verdict: OpenAICompatProbeVerdict = "indeterminate";
		if (callSchemaRejected(direct) && callDirectlySucceeded(forced)) {
			workingProfile = setCompatValue(workingProfile, "disableReasoningOnForcedToolChoice", true);
			setFinding("disableReasoningOnForcedToolChoice", "required", "reasoning-tool-interaction@probe", true);
			verdict = "required";
		} else if (callSchemaRejected(direct) && callDirectlySucceeded(all)) {
			workingProfile = setCompatValue(workingProfile, "disableReasoningOnToolChoice", true);
			setFinding("disableReasoningOnToolChoice", "required", "reasoning-tool-interaction@probe", true);
			verdict = "required";
		} else {
			verdict = callDirectlySucceeded(direct) ? "accepted" : "indeterminate";
			setFinding(
				"disableReasoningOnForcedToolChoice",
				callDirectlySucceeded(direct) ? "accepted" : "indeterminate",
				"reasoning-tool-interaction@probe",
			);
			setFinding(
				"disableReasoningOnToolChoice",
				callDirectlySucceeded(direct) ? "accepted" : "indeterminate",
				"reasoning-tool-interaction@probe",
			);
		}
		return {
			verdict,
			evidence: {
				direct: { passing: callSucceeded(direct), body: finalRequestBody(direct) },
				forcedDisabled: { passing: callSucceeded(forced), body: finalRequestBody(forced) },
				allChoiceDisabled: { passing: callSucceeded(all), body: finalRequestBody(all) },
			},
		};
	});

	await runScenario("candidate-profiles", "probe", async environment => {
		if (candidates.length === 0) return { verdict: "not_applicable" };
		const baseSnapshot = cloneCompat(workingProfile);
		const baseResult = await environment.call(buildTrialModel(baseSnapshot), createToolContext(), {
			reasoning: model.reasoning ? winningEffort : undefined,
			toolChoice: toolChoiceForValidation,
		});
		const results: Array<{ candidate: OpenAICompatProbeCandidate; result: ProbeCallResult }> = [];
		for (const candidate of candidates) {
			const profile = mergeProfiles(baseSnapshot, candidate.compat);
			results.push({
				candidate,
				result: await environment.call(buildTrialModel(profile), createToolContext(), {
					reasoning: model.reasoning ? winningEffort : undefined,
					toolChoice: toolChoiceForValidation,
				}),
			});
		}
		const passing = results
			.filter(({ result }) => callDirectlySucceeded(result) && hasCompatEchoCall(result.message))
			.sort(
				(left, right) =>
					candidateRank(left.candidate) - candidateRank(right.candidate) ||
					compareProbeStrings(left.candidate.name, right.candidate.name),
			);
		const winner = passing[0];
		let verdict: OpenAICompatProbeVerdict = "indeterminate";
		const baseConclusiveFailure =
			(callDirectlySucceeded(baseResult) && !hasCompatEchoCall(baseResult.message)) ||
			callSchemaRejected(baseResult);
		if (winner && baseConclusiveFailure) {
			workingProfile = mergeProfiles(baseSnapshot, winner.candidate.compat);
			for (const leaf of collectLeaves(winner.candidate.compat)) {
				const key = leaf.segments[0] as keyof OpenAICompat;
				if (!Object.hasOwn(OPENAI_COMPAT_CAPABILITY_REGISTRY, key)) continue;
				provisionalKeys.add(key);
				setFinding(
					key,
					"effective",
					"candidate-profiles@probe",
					undefined,
					`Provisionally selected candidate profile ${winner.candidate.name}`,
				);
			}
			const rerun = await environment.call(buildTrialModel(workingProfile), createToolContext(), {
				reasoning: model.reasoning ? winningEffort : undefined,
				toolChoice: toolChoiceForValidation,
			});
			verdict = callDirectlySucceeded(rerun) && hasCompatEchoCall(rerun.message) ? "effective" : "indeterminate";
		} else if (callDirectlySucceeded(baseResult) && hasCompatEchoCall(baseResult.message)) {
			verdict = "effective";
		}
		return {
			verdict,
			evidence: {
				base: {
					passing: callSucceeded(baseResult),
					semantic: hasCompatEchoCall(baseResult.message),
					body: finalRequestBody(baseResult),
				},
				profiles: results.map(({ candidate, result }) => ({
					name: candidate.name,
					source: candidate.source,
					passing: callSucceeded(result),
					semantic: hasCompatEchoCall(result.message),
					fallback: callHasCompatibilityFallback(result),
					body: finalRequestBody(result),
				})),
				selected: winner?.candidate.name,
			},
		};
	});

	await runScenario("fallback-latching", "probe", async environment => {
		if (!fallbackProbe) return { verdict: "not_applicable" };
		const state = environment.createSharedState();
		const first = await environment.call(fallbackProbe.model, fallbackProbe.context, fallbackProbe.options, state);
		const second = await environment.call(fallbackProbe.model, fallbackProbe.context, fallbackProbe.options, state);
		const firstFallback = callHasCompatibilityFallback(first);
		const secondFallback = callHasCompatibilityFallback(second);
		return {
			verdict: firstFallback && callDirectlySucceeded(second) && !secondFallback ? "effective" : "indeterminate",
			evidence: {
				first: { fallback: firstFallback, attempts: first.attempts.length, body: finalRequestBody(first) },
				second: { fallback: secondFallback, attempts: second.attempts.length, body: finalRequestBody(second) },
			},
		};
	});

	await runScenario("stream-observation", "probe", async environment => {
		const rawWireEvidence = allWireEvidence.map(getOpenAICompatRawWireDeltas);
		const cumulativeObserved = rawWireEvidence.some(evidence =>
			evidence.reasoningDeltaSequences.some(hasCumulativeReasoningDeltas),
		);
		const contentDeltas = rawWireEvidence.flatMap(evidence => evidence.contentDeltas);
		const deepseekObserved = contentDeltas.some(delta => /<(?:｜|\|)[A-Za-z0-9_.｜|▁]{1,64}(?:｜|\|)>/u.test(delta));
		const markupPattern = contentDeltas.some(delta => delta.includes("<|tool_calls_section_begin|>"))
			? "kimi"
			: contentDeltas.some(delta => delta.includes("<｜DSML｜"))
				? "dsml"
				: contentDeltas.some(delta => delta.includes("<thinking>"))
					? "thinking"
					: undefined;
		let verdict: OpenAICompatProbeVerdict = "indeterminate";
		if (cumulativeObserved && model.reasoning) {
			const profile = setCompatValue(workingProfile, "reasoningDeltasMayBeCumulative", true);
			const result = await environment.call(buildTrialModel(profile), createBaselineContext(), {
				reasoning: winningEffort,
			});
			if (callDirectlySucceeded(result) && hasThinkingBlock(result.message)) {
				workingProfile = profile;
				setFinding("reasoningDeltasMayBeCumulative", "effective", "stream-observation@probe", true);
				verdict = "effective";
			}
		}
		if (deepseekObserved) {
			const profile = setCompatValue(workingProfile, "stripDeepseekSpecialTokens", true);
			const result = await environment.call(buildTrialModel(profile), createBaselineContext(), {
				reasoning: model.reasoning ? winningEffort : undefined,
			});
			if (
				callDirectlySucceeded(result) &&
				!/<(?:｜|\|)[A-Za-z0-9_.｜|▁]{1,64}(?:｜|\|)>/u.test(visibleText(result.message))
			) {
				workingProfile = profile;
				setFinding("stripDeepseekSpecialTokens", "effective", "stream-observation@probe", true);
				verdict = "effective";
			}
		}
		if (markupPattern) {
			const profile = setCompatValue(workingProfile, "streamMarkupHealingPattern", markupPattern);
			const result = await environment.call(buildTrialModel(profile), createToolContext(), {
				toolChoice: toolChoiceForValidation,
			});
			if (callDirectlySucceeded(result)) {
				workingProfile = profile;
				setFinding("streamMarkupHealingPattern", "effective", "stream-observation@probe", markupPattern);
				verdict = "effective";
			}
		}
		const maximumGap = Math.max(0, ...allWireEvidence.map(evidence => evidence.maxInterEventGapMs));
		const firstEventSamples = allWireEvidence
			.map(evidence => evidence.timeToFirstEventMs)
			.filter((value): value is number => value !== undefined);
		for (const key of [
			"enableGeminiThinkingLoopGuard",
			"emptyLengthFinishIsContextError",
			"streamIdleTimeoutMs",
		] as const) {
			const current = findings.get(key);
			if (current?.verdict === "indeterminate") {
				setFinding(
					key,
					"indeterminate",
					"stream-observation@probe",
					configuredKnown[key],
					"Trigger condition was not observed",
				);
			}
		}
		return {
			verdict,
			evidence: {
				cumulativeReasoningObserved: cumulativeObserved,
				deepseekSpecialTokensObserved: deepseekObserved,
				markupPatternObserved: markupPattern,
				timeToFirstEventMs: firstEventSamples,
				maximumInterEventGapMs: maximumGap,
			},
		};
	});

	const runFinalValidation = async (
		profile: OpenAICompat,
		phase: "probe" | "minimize" | "validate",
		id?: string,
	): Promise<FinalValidationResult> => {
		let result: FinalValidationResult = { viable: false, fallbackFree: false };
		await runScenario(
			"final",
			phase,
			async environment => {
				const trial = buildTrialModel(profile);
				const state = environment.createSharedState();
				const validationEffort = trial.reasoning ? (winningEffort ?? Effort.Medium) : undefined;
				const commonOptions: ProbeRequestOptions = {
					maxTokens: 64,
					maxTokensExplicit: true,
					...(validationEffort ? { reasoning: validationEffort } : {}),
				};
				const baseline = await environment.call(trial, createBaselineContext(), commonOptions, state);
				if (!callDirectlySucceeded(baseline) || !/PONG/u.test(visibleText(baseline.message))) {
					return {
						verdict: "indeterminate",
						evidence: { stage: "baseline", passing: callDirectlySucceeded(baseline) },
					};
				}
				const toolMessages = [
					createProbeUserMessage(BASELINE_USER_PROMPT),
					baseline.message!,
					createProbeUserMessage(TOOL_USER_PROMPT),
				];
				const toolContext: Context = {
					systemPrompt: [BASELINE_SYSTEM_PROMPT],
					messages: toolMessages,
					tools: [COMPAT_ECHO_TOOL],
				};
				const resolvedChoice = resolveValidationToolChoice(trial.compat);
				let toolResult: ProbeCallResult | undefined;
				for (let attempt = 0; attempt < (resolvedChoice === "auto" ? 3 : 1); attempt += 1) {
					const callResult = await environment.call(
						trial,
						toolContext,
						{ ...commonOptions, toolChoice: resolvedChoice },
						state,
					);
					toolResult = callResult;
					if (callDirectlySucceeded(callResult) && hasCompatEchoCall(callResult.message)) break;
				}
				if (!toolResult || !callDirectlySucceeded(toolResult) || !hasCompatEchoCall(toolResult.message)) {
					return {
						verdict: "indeterminate",
						evidence: { stage: "tool", passing: toolResult ? callDirectlySucceeded(toolResult) : false },
					};
				}
				const toolCall = toolResult.message!.content.find(
					block =>
						block.type === "toolCall" && block.name === "compat_echo" && block.arguments.value === "PROBE_OK",
				);
				if (toolCall?.type !== "toolCall") {
					return { verdict: "indeterminate", evidence: { stage: "tool-parse" } };
				}
				const followupContext: Context = {
					systemPrompt: [BASELINE_SYSTEM_PROMPT],
					messages: [
						...toolMessages,
						toolResult.message!,
						{
							role: "toolResult",
							toolCallId: toolCall.id,
							toolName: toolCall.name,
							content: [{ type: "text", text: TOOL_RESULT_PROMPT }],
							isError: false,
							timestamp: 0,
						},
						createProbeUserMessage(TOOL_FOLLOWUP_PROMPT),
					],
					tools: [COMPAT_ECHO_TOOL],
				};
				const followup = await environment.call(trial, followupContext, commonOptions, state);
				const fallbackFree = [baseline, toolResult, followup].every(
					callResult => !callHasCompatibilityFallback(callResult),
				);
				const stable = [baseline, toolResult, followup].every(callResult => !callTransientlyFailed(callResult));
				const completed = callDirectlySucceeded(followup) && /TOOL_RESULT_OK/u.test(visibleText(followup.message));
				result = { viable: completed && fallbackFree && stable, fallbackFree: fallbackFree && stable };
				return {
					verdict: result.viable ? "effective" : "indeterminate",
					evidence: {
						stage: completed ? "complete" : "followup",
						toolChoice: resolvedChoice,
						fallbackFree,
						stable,
						reasoningEffort: validationEffort,
					},
				};
			},
			id,
		);
		return result;
	};

	for (const key of OPENAI_COMPAT_KEYS) {
		const finding = findings.get(key);
		const configuredValue = configuredKnown[key];
		if (finding?.verdict === "indeterminate" && configuredValue !== undefined) {
			workingProfile = setCompatValue(workingProfile, key, configuredValue);
			finding.recommendedValue = structuredClone(configuredValue);
		}
	}

	const prevalidation = await runFinalValidation(workingProfile, "probe", "prevalidation@probe");
	let recommendedCompat = cloneCompat(workingProfile);
	let viable = prevalidation.viable;
	let fallbackFree = prevalidation.fallbackFree;

	if (viable) {
		const initialLeaves = collectLeaves(recommendedCompat);
		minimizeTotal = initialLeaves.length;
		for (const leaf of initialLeaves) {
			const candidate = removeLeaf(recommendedCompat, leaf.segments);
			const scenarioId = `final@minimize:${encodeURIComponent(leaf.path)}`;
			const validation = await runFinalValidation(candidate, "minimize", scenarioId);
			const key = leaf.segments[0] as keyof OpenAICompat;
			const finding = findings.get(key);
			const resolvedCandidate = buildTrialModel(candidate).compat;
			const requiredValue =
				finding?.recommendedValue === undefined
					? undefined
					: leaf.segments.length === 1
						? finding.recommendedValue
						: getPath(finding.recommendedValue, leaf.segments.slice(1));
			const requiredValueStillResolved =
				requiredValue === undefined ||
				finding?.verdict === "indeterminate" ||
				compatValuesEqual(getPath(resolvedCandidate, leaf.segments), requiredValue);
			const preservesConfiguredIndeterminate =
				finding?.verdict !== "indeterminate" || getPath(configuredKnown, leaf.segments) === undefined;
			if (
				validation.viable &&
				validation.fallbackFree &&
				requiredValueStillResolved &&
				preservesConfiguredIndeterminate
			) {
				recommendedCompat = candidate;
			} else if (finding && !finding.scenarioIds.includes(scenarioId)) {
				finding.scenarioIds.push(scenarioId);
			}
		}
		const finalValidation = await runFinalValidation(recommendedCompat, "validate");
		viable = finalValidation.viable;
		fallbackFree = finalValidation.fallbackFree;
	}
	if (viable) {
		for (const key of OPENAI_COMPAT_KEYS) {
			const finding = findings.get(key);
			const value = recommendedCompat[key];
			if (finding && value !== undefined && finding.verdict === "effective" && provisionalKeys.has(key)) {
				finding.verdict = "required";
				finding.recommendedValue = structuredClone(value);
				finding.note = finding.note?.replace(/^Provisionally selected /u, "Selected ");
			}
		}
	}
	if (!viable) {
		recommendedCompat = {};
		fallbackFree = false;
	}
	validateStrictOpenAICompat(recommendedCompat);
	const resolvedRecommendedModel = viable ? buildTrialModel(recommendedCompat) : baselineModel;
	const recommendedLeafPaths = new Set(collectLeaves(recommendedCompat).map(leaf => leaf.path));
	const configuredLeafPaths = collectLeaves(rawConfigured).map(leaf => leaf.path);
	const removeConfiguredPaths = viable
		? configuredLeafPaths.filter(path => !recommendedLeafPaths.has(path)).sort()
		: [];
	const report: OpenAICompatProbeReport = {
		schemaVersion: 1,
		generatedAt: new Date().toISOString(),
		target: {
			provider: sanitizeIdentifier(model.provider, input.apiKey),
			model: sanitizeIdentifier(model.id, input.apiKey),
			api: "openai-completions",
			baseUrl: sanitizeProbeUrl(model.baseUrl, input.apiKey),
		},
		configuredCompat,
		ignoredConfiguredKeys,
		resolvedBaseline: baselineModel.compat,
		recommendedCompat,
		resolvedRecommended: resolvedRecommendedModel.compat,
		removeConfiguredPaths,
		viable,
		fallbackFree,
		findings: OPENAI_COMPAT_KEYS.map(key => findings.get(key)!),
		scenarios,
		warnings,
	};
	return report;
}
