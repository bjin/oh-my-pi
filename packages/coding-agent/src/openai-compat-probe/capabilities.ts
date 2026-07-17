import type { OpenAICompat } from "@oh-my-pi/pi-catalog/types";
import { OpenAICompatSchema } from "../config/models-config-schema";

export type OpenAICompatProbeReducer = "paired" | "semantic" | "observation" | "candidate" | "not_applicable";

export type OpenAICompatPersistenceRule =
	| "evidence_forced"
	| "preference_only"
	| "preserve_indeterminate"
	| "not_applicable";

export interface OpenAICompatCapability {
	scenarios: readonly string[];
	candidateValues: readonly unknown[];
	reducer: OpenAICompatProbeReducer;
	persistence: OpenAICompatPersistenceRule;
	redaction: "public" | "nested_values";
}

const pairedBoolean = (scenario: string): OpenAICompatCapability => ({
	scenarios: [scenario],
	candidateValues: [true, false],
	reducer: "paired",
	persistence: "evidence_forced",
	redaction: "public",
});

const semanticBoolean = (scenario: string): OpenAICompatCapability => ({
	scenarios: [scenario],
	candidateValues: [true, false],
	reducer: "semantic",
	persistence: "evidence_forced",
	redaction: "public",
});

const observedBoolean = (scenario: string): OpenAICompatCapability => ({
	scenarios: [scenario],
	candidateValues: [true, false],
	reducer: "observation",
	persistence: "preserve_indeterminate",
	redaction: "public",
});

const candidateOnly = (
	scenarios: readonly string[],
	candidateValues: readonly unknown[] = [],
	redaction: OpenAICompatCapability["redaction"] = "public",
): OpenAICompatCapability => ({
	scenarios,
	candidateValues,
	reducer: "candidate",
	persistence: "preserve_indeterminate",
	redaction,
});

const notApplicable = (): OpenAICompatCapability => ({
	scenarios: [],
	candidateValues: [],
	reducer: "not_applicable",
	persistence: "not_applicable",
	redaction: "public",
});

/**
 * Canonical compatibility order. The `satisfies` clause makes adding an
 * `OpenAICompat` field a compile error until the probe explicitly classifies it.
 */
export const OPENAI_COMPAT_CAPABILITY_REGISTRY = {
	supportsStore: pairedBoolean("store"),
	supportsDeveloperRole: pairedBoolean("developer-role"),
	supportsMultipleSystemMessages: pairedBoolean("multiple-system-messages"),
	supportsReasoningEffort: semanticBoolean("reasoning-effort"),
	reasoningEffortMap: candidateOnly(["reasoning-effort"], [], "nested_values"),
	supportsUsageInStreaming: semanticBoolean("streaming-usage"),
	enableGeminiThinkingLoopGuard: observedBoolean("stream-observation"),
	maxTokensField: {
		scenarios: ["max-tokens"],
		candidateValues: ["max_completion_tokens", "max_tokens"],
		reducer: "paired",
		persistence: "evidence_forced",
		redaction: "public",
	},
	requiresToolResultName: pairedBoolean("history-tool-result-name"),
	requiresAssistantAfterToolResult: pairedBoolean("history-assistant-after-tool-result"),
	requiresThinkingAsText: semanticBoolean("history-thinking-as-text"),
	requiresMistralToolIds: pairedBoolean("history-tool-call-id"),
	thinkingFormat: {
		scenarios: ["reasoning-format"],
		candidateValues: ["openai", "openrouter", "zai", "qwen", "qwen-chat-template"],
		reducer: "semantic",
		persistence: "evidence_forced",
		redaction: "public",
	},
	reasoningDisableMode: {
		scenarios: ["reasoning-disable"],
		candidateValues: [
			"omit",
			"lowest-effort",
			"openrouter-enabled-false",
			"zai-thinking-disabled",
			"qwen-enable-thinking-false",
			"qwen-template-false",
		],
		reducer: "paired",
		persistence: "evidence_forced",
		redaction: "public",
	},
	omitReasoningEffort: pairedBoolean("reasoning-effort"),
	includeEncryptedReasoning: notApplicable(),
	filterReasoningHistory: notApplicable(),
	thinkingKeep: candidateOnly(["reasoning-format"], ["all", false]),
	reasoningContentField: {
		scenarios: ["history-reasoning-content"],
		candidateValues: ["reasoning_content", "reasoning", "reasoning_text"],
		reducer: "semantic",
		persistence: "evidence_forced",
		redaction: "public",
	},
	requiresReasoningContentForToolCalls: pairedBoolean("history-reasoning-content"),
	requiresReasoningContentForAllAssistantTurns: pairedBoolean("history-reasoning-all-assistant"),
	allowsSyntheticReasoningContentForToolCalls: pairedBoolean("history-reasoning-content"),
	replayReasoningContent: observedBoolean("history-reasoning-replay"),
	qwenPreserveThinking: observedBoolean("history-reasoning-replay"),
	requiresAssistantContentForToolCalls: pairedBoolean("history-assistant-tool-content"),
	supportsToolChoice: semanticBoolean("tool-choice"),
	supportsForcedToolChoice: semanticBoolean("tool-choice"),
	supportsNamedToolChoice: semanticBoolean("tool-choice"),
	disableReasoningOnForcedToolChoice: pairedBoolean("reasoning-tool-interaction"),
	disableReasoningOnToolChoice: pairedBoolean("reasoning-tool-interaction"),
	openRouterRouting: candidateOnly(["candidate-profiles"], [], "nested_values"),
	vercelGatewayRouting: candidateOnly(["candidate-profiles"], [], "nested_values"),
	extraBody: candidateOnly(["candidate-profiles"], [], "nested_values"),
	promptCacheSessionHeader: candidateOnly(["candidate-profiles"], ["x-grok-conv-id"]),
	cacheControlFormat: candidateOnly(["candidate-profiles"], ["anthropic"]),
	supportsStrictMode: semanticBoolean("tool-strictness"),
	toolSchemaFlavor: {
		scenarios: ["tool-schema-flavor"],
		candidateValues: ["moonshot-mfjs", "none"],
		reducer: "semantic",
		persistence: "evidence_forced",
		redaction: "public",
	},
	streamIdleTimeoutMs: candidateOnly(["stream-observation"]),
	supportsLongPromptCacheRetention: notApplicable(),
	toolStrictMode: {
		scenarios: ["tool-strictness"],
		candidateValues: ["all_strict", "none"],
		reducer: "semantic",
		persistence: "evidence_forced",
		redaction: "public",
	},
	supportsReasoningParams: pairedBoolean("reasoning-format"),
	supportsSamplingParams: pairedBoolean("sampling-parameters"),
	alwaysSendMaxTokens: pairedBoolean("always-max-tokens"),
	strictResponsesPairing: notApplicable(),
	supportsImageDetailOriginal: notApplicable(),
	reasoningDeltasMayBeCumulative: observedBoolean("stream-observation"),
	stripDeepseekSpecialTokens: observedBoolean("stream-observation"),
	streamMarkupHealingPattern: {
		scenarios: ["stream-observation"],
		candidateValues: ["kimi", "dsml", "thinking"],
		reducer: "observation",
		persistence: "preserve_indeterminate",
		redaction: "public",
	},
	emptyLengthFinishIsContextError: observedBoolean("stream-observation"),
	usesOpenAIToolCallIdLimit: pairedBoolean("history-tool-call-id"),
	whenThinking: candidateOnly(["reasoning-tool-interaction"], [], "nested_values"),
} satisfies Record<keyof OpenAICompat, OpenAICompatCapability>;

export const OPENAI_COMPAT_KEYS = Object.freeze(
	Object.keys(OPENAI_COMPAT_CAPABILITY_REGISTRY) as Array<keyof OpenAICompat>,
);

function isPlainRecord(value: unknown): value is Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
	const prototype = Object.getPrototypeOf(value);
	return prototype === Object.prototype || prototype === null;
}

function validatePersistableValue(value: unknown, path: string, ancestors: WeakSet<object>): void {
	if (value === null || typeof value === "string" || typeof value === "boolean") return;
	if (typeof value === "number") {
		if (!Number.isFinite(value)) throw new Error(`${path} must contain only finite numbers`);
		return;
	}
	if (typeof value !== "object") {
		throw new Error(`${path} contains non-persistable ${typeof value}`);
	}
	if (Object.getOwnPropertySymbols(value).length > 0) {
		throw new Error(`${path} contains non-persistable symbol keys`);
	}
	if (ancestors.has(value)) throw new Error(`${path} contains a cycle`);
	ancestors.add(value);
	if (Array.isArray(value)) {
		for (let index = 0; index < value.length; index += 1) {
			validatePersistableValue(value[index], `${path}[${index}]`, ancestors);
		}
	} else {
		if (!isPlainRecord(value)) throw new Error(`${path} must contain only plain objects`);
		for (const [key, child] of Object.entries(value)) {
			validatePersistableValue(child, `${path}.${key}`, ancestors);
		}
	}
	ancestors.delete(value);
}

function validateKnownKeys(value: Record<string, unknown>, path: string, allowWhenThinking: boolean): void {
	for (const key of Object.keys(value)) {
		if (!Object.hasOwn(OPENAI_COMPAT_CAPABILITY_REGISTRY, key)) {
			throw new Error(`Unknown OpenAI compatibility key: ${path}${key}`);
		}
		if (key !== "whenThinking") continue;
		if (!allowWhenThinking) {
			throw new Error(`Nested whenThinking is not allowed: ${path}${key}`);
		}
		const nested = value[key];
		if (!isPlainRecord(nested)) {
			throw new Error(`${path}${key} must be an object`);
		}
		validateKnownKeys(nested, `${path}${key}.`, false);
	}
}

/** Validate and return a persistable OpenAI compatibility object. */
export function validateStrictOpenAICompat(value: unknown): OpenAICompat {
	if (!isPlainRecord(value)) throw new Error("OpenAI compatibility must be an object");
	validatePersistableValue(value, "compat", new WeakSet<object>());
	validateKnownKeys(value, "", true);
	try {
		return OpenAICompatSchema.assert(value) as OpenAICompat;
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(`Invalid OpenAI compatibility: ${message}`, { cause: error });
	}
}
