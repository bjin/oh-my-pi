import type { FetchImpl, Model, OpenAICompat, ResolvedOpenAICompat } from "@oh-my-pi/pi-catalog/types";

export type OpenAICompatProbeVerdict =
	| "effective"
	| "accepted"
	| "unsupported"
	| "required"
	| "fallback_only"
	| "indeterminate"
	| "not_applicable"
	| "ignored";

export interface OpenAICompatProbeCandidate {
	name: string;
	compat: OpenAICompat;
	source: "built_in" | "configured" | "user";
}

export interface OpenAICompatProbeInput {
	model: Model<"openai-completions">;
	apiKey: string;
	candidates?: readonly OpenAICompatProbeCandidate[];
	timeoutMs?: number;
	signal?: AbortSignal;
	fetch?: FetchImpl;
	onProgress?: (event: OpenAICompatProbeProgress) => void | Promise<void>;
}

export interface OpenAICompatProbeProgress {
	type: "scenario_start" | "scenario_complete";
	scenarioId: string;
	phase: "probe" | "minimize" | "validate";
	index: number;
	total: number;
	verdict?: OpenAICompatProbeVerdict;
}

export interface OpenAICompatProbeFinding {
	key: keyof OpenAICompat;
	verdict: OpenAICompatProbeVerdict;
	recommendedValue?: unknown;
	scenarioIds: string[];
	persistable: boolean;
	note?: string;
}

export interface OpenAICompatProbeWireAttempt {
	ordinal: number;
	request: {
		method: string;
		url: string;
		headers: Record<string, string>;
		body?: unknown;
	};
	response?: {
		status: number;
		headers: Record<string, string>;
		errorBody?: string;
	};
	latencyMs: number;
	classification: "first" | "transport_retry" | "reasoning_fallback" | "strict_fallback" | "empty_retry";
}

export interface OpenAICompatProbeScenario {
	id: string;
	baseId: string;
	phase: "probe" | "minimize" | "validate";
	verdict: OpenAICompatProbeVerdict;
	attempts: OpenAICompatProbeWireAttempt[];
	final: {
		stopReason?: string;
		errorStatus?: number;
		errorId?: number;
	};
	evidence: Record<string, unknown>;
}

export interface OpenAICompatProbeReport {
	schemaVersion: 1;
	generatedAt: string;
	target: {
		provider: string;
		model: string;
		api: "openai-completions";
		baseUrl: string;
	};
	configuredCompat: OpenAICompat & Record<string, unknown>;
	ignoredConfiguredKeys: string[];
	resolvedBaseline: ResolvedOpenAICompat;
	recommendedCompat: OpenAICompat;
	resolvedRecommended: ResolvedOpenAICompat;
	removeConfiguredPaths: string[];
	viable: boolean;
	fallbackFree: boolean;
	findings: OpenAICompatProbeFinding[];
	scenarios: OpenAICompatProbeScenario[];
	warnings: string[];
}

export interface OpenAICompatCandidateFileV1 {
	schemaVersion: 1;
	profiles: Array<{ name: string; compat: OpenAICompat }>;
}
