import { afterEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { streamOpenAICompletions } from "@oh-my-pi/pi-ai/providers/openai-completions";
import type { Context, FetchImpl } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Effort, THINKING_EFFORTS } from "@oh-my-pi/pi-catalog/effort";
import type { ModelSpec, OpenAICompat } from "@oh-my-pi/pi-catalog/types";
import type { OpenAICompatProbeReport } from "@oh-my-pi/pi-coding-agent/openai-compat-probe";
import {
	formatOpenAICompatibilityReport,
	formatOpenAICompatibilityYaml,
	probeOpenAICompatibility,
	validateStrictOpenAICompat,
} from "@oh-my-pi/pi-coding-agent/openai-compat-probe";
import { createBaselineContext } from "../src/openai-compat-probe/scenarios";
import {
	createOpenAICompatWireObserver,
	sanitizeProbeErrorMessage,
	sanitizeProbeUrl,
} from "../src/openai-compat-probe/wire-observer";

const API_KEY = "probe-secret-key";
const runningServers: Bun.Server<unknown>[] = [];

function modelSpec(
	baseUrl: string,
	compat: OpenAICompat = {},
	options: { provider?: string; reasoning?: boolean } = {},
): ModelSpec<"openai-completions"> {
	const reasoning = options.reasoning ?? false;
	return {
		id: reasoning ? "reasoning-probe-model" : "probe-model",
		name: "Probe Model",
		api: "openai-completions",
		provider: options.provider ?? "compat-probe",
		baseUrl,
		reasoning,
		...(reasoning ? { thinking: { mode: "effort", efforts: THINKING_EFFORTS } } : {}),
		input: ["text"],
		supportsTools: true,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 16_384,
		maxTokens: 1_024,
		compat,
	};
}

function requestBody(init: RequestInit | undefined): Record<string, unknown> {
	if (typeof init?.body !== "string") throw new Error("Expected a JSON request body");
	const parsed = JSON.parse(init.body) as unknown;
	if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
		throw new Error("Expected a JSON object request body");
	}
	return parsed as Record<string, unknown>;
}

function hasStrictTrue(value: unknown): boolean {
	if (Array.isArray(value)) return value.some(hasStrictTrue);
	if (typeof value !== "object" || value === null) return false;
	return Object.entries(value).some(([key, child]) => (key === "strict" && child === true) || hasStrictTrue(child));
}

function countProperty(value: unknown, property: string): number {
	if (Array.isArray(value)) return value.reduce((total, child) => total + countProperty(child, property), 0);
	if (typeof value !== "object" || value === null) return 0;
	return Object.entries(value).reduce(
		(total, [key, child]) => total + (key === property ? 1 : 0) + countProperty(child, property),
		0,
	);
}

function propertyValues(value: unknown, property: string): unknown[] {
	if (Array.isArray(value)) return value.flatMap(child => propertyValues(child, property));
	if (typeof value !== "object" || value === null) return [];
	const entries = Object.entries(value);
	return [
		...entries.filter(([key]) => key === property).map(([, child]) => child),
		...entries.flatMap(([, child]) => propertyValues(child, property)),
	];
}

function messageText(value: unknown): string {
	if (typeof value === "string") return value;
	if (!Array.isArray(value)) return "";
	return value
		.map(part => {
			if (typeof part !== "object" || part === null || Array.isArray(part)) return "";
			const text = (part as Record<string, unknown>).text;
			return typeof text === "string" ? text : "";
		})
		.join("");
}

function lastUserText(body: Record<string, unknown>): string {
	if (!Array.isArray(body.messages)) return "";
	for (let index = body.messages.length - 1; index >= 0; index -= 1) {
		const message = body.messages[index];
		if (typeof message !== "object" || message === null || Array.isArray(message)) continue;
		const object = message as Record<string, unknown>;
		if (object.role === "user") return messageText(object.content);
	}
	return "";
}

function hasToolResult(body: Record<string, unknown>): boolean {
	return (
		Array.isArray(body.messages) &&
		body.messages.some(
			message =>
				typeof message === "object" &&
				message !== null &&
				!Array.isArray(message) &&
				(message as Record<string, unknown>).role === "tool",
		)
	);
}

function sseResponse(
	body: Record<string, unknown>,
	includeUsage: boolean,
	reasoningDeltas: readonly string[] = [],
): Response {
	const model = typeof body.model === "string" ? body.model : "probe-model";
	const frames: unknown[] = [];
	for (const reasoning of reasoningDeltas) {
		frames.push({
			id: "chatcmpl-probe",
			object: "chat.completion.chunk",
			created: 0,
			model,
			choices: [{ index: 0, delta: { role: "assistant", reasoning_content: reasoning }, finish_reason: null }],
		});
	}
	if (
		Array.isArray(body.tools) &&
		body.tools.length > 0 &&
		!hasToolResult(body) &&
		/compat_echo/u.test(lastUserText(body))
	) {
		frames.push(
			{
				id: "chatcmpl-probe",
				object: "chat.completion.chunk",
				created: 0,
				model,
				choices: [
					{
						index: 0,
						delta: {
							role: "assistant",
							tool_calls: [
								{
									index: 0,
									id: "call_probe_123",
									type: "function",
									function: { name: "compat_echo", arguments: '{"value":"PROBE_OK"}' },
								},
							],
						},
						finish_reason: null,
					},
				],
			},
			{
				id: "chatcmpl-probe",
				object: "chat.completion.chunk",
				created: 0,
				model,
				choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
			},
		);
	} else {
		const text = hasToolResult(body) || /TOOL_RESULT_OK/u.test(lastUserText(body)) ? "TOOL_RESULT_OK" : "PONG";
		frames.push(
			{
				id: "chatcmpl-probe",
				object: "chat.completion.chunk",
				created: 0,
				model,
				choices: [{ index: 0, delta: { role: "assistant", content: text }, finish_reason: null }],
			},
			{
				id: "chatcmpl-probe",
				object: "chat.completion.chunk",
				created: 0,
				model,
				choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
			},
		);
	}
	if (includeUsage) {
		frames.push({
			id: "chatcmpl-probe",
			object: "chat.completion.chunk",
			created: 0,
			model,
			choices: [],
			usage: { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 },
		});
	}
	const payload = `${frames.map(frame => `data: ${JSON.stringify(frame)}`).join("\n\n")}\n\ndata: [DONE]\n\n`;
	return new Response(payload, { status: 200, headers: { "content-type": "text/event-stream" } });
}
function textSseResponse(body: Record<string, unknown>, text = "PONG"): Response {
	const model = typeof body.model === "string" ? body.model : "probe-model";
	const frames = [
		{
			id: "chatcmpl-text",
			object: "chat.completion.chunk",
			created: 0,
			model,
			choices: [{ index: 0, delta: { role: "assistant", content: text }, finish_reason: null }],
		},
		{
			id: "chatcmpl-text",
			object: "chat.completion.chunk",
			created: 0,
			model,
			choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
		},
	];
	return new Response(`${frames.map(frame => `data: ${JSON.stringify(frame)}`).join("\n\n")}\n\ndata: [DONE]\n\n`, {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});
}

function emptySseResponse(body: Record<string, unknown>): Response {
	const model = typeof body.model === "string" ? body.model : "probe-model";
	const payload = [
		{
			id: "chatcmpl-empty",
			object: "chat.completion.chunk",
			created: 0,
			model,
			choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
		},
		{
			id: "chatcmpl-empty",
			object: "chat.completion.chunk",
			created: 0,
			model,
			choices: [],
			usage: { prompt_tokens: 4, completion_tokens: 0, total_tokens: 4 },
		},
	];
	return new Response(`${payload.map(frame => `data: ${JSON.stringify(frame)}`).join("\n\n")}\n\ndata: [DONE]\n\n`, {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});
}

function stalledSseResponse(): Response {
	return new Response(new ReadableStream<Uint8Array>({}), {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});
}

function fetchImpl(handler: (body: Record<string, unknown>) => Response | Promise<Response>): FetchImpl {
	const implementation = async (_input: string | URL | Request, init?: RequestInit): Promise<Response> =>
		handler(requestBody(init));
	return Object.assign(implementation, { preconnect: fetch.preconnect });
}

function errorResponse(status: number, message: string, param?: string, headers?: Record<string, string>): Response {
	return Response.json(
		{ error: { message, type: "invalid_request_error", ...(param ? { param } : {}) } },
		{ status, headers },
	);
}

function finding(report: OpenAICompatProbeReport, key: keyof OpenAICompat) {
	const result = report.findings.find(candidate => candidate.key === key);
	if (!result) throw new Error(`Missing finding ${key}`);
	return result;
}

function encodePointerSegment(value: string): string {
	return value.replaceAll("~", "~0").replaceAll("/", "~1");
}

function leafPaths(value: unknown, segments: string[] = []): string[] {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		return [`/${segments.map(encodePointerSegment).join("/")}`];
	}
	const entries = Object.entries(value);
	if (entries.length === 0) return segments.length > 0 ? [`/${segments.map(encodePointerSegment).join("/")}`] : [];
	return entries.flatMap(([key, child]) => leafPaths(child, [...segments, key]));
}

afterEach(() => {
	for (const server of runningServers.splice(0)) server.stop(true);
});

describe("OpenAI compatibility probe", () => {
	it("rejects duplicate, unknown, and invalid candidates before fetch", async () => {
		let calls = 0;
		const fetch = fetchImpl(body => {
			calls += 1;
			return sseResponse(body, true);
		});
		const model = buildModel(modelSpec("https://probe.invalid/v1"));
		await expect(
			probeOpenAICompatibility({
				model,
				apiKey: API_KEY,
				fetch,
				candidates: [
					{ name: "same", source: "user", compat: {} },
					{ name: " same ", source: "user", compat: {} },
				],
			}),
		).rejects.toThrow("Duplicate compatibility candidate name");
		await expect(
			probeOpenAICompatibility({
				model,
				apiKey: API_KEY,
				fetch,
				candidates: [
					{
						name: "unknown",
						source: "user",
						compat: { unknownCompatibilityFlag: true } as OpenAICompat,
					},
				],
			}),
		).rejects.toThrow("Unknown OpenAI compatibility key");
		await expect(
			probeOpenAICompatibility({
				model,
				apiKey: API_KEY,
				fetch,
				candidates: [
					{
						name: "credential",
						source: "user",
						compat: { extraBody: { api_key: "must-not-be-reported" } },
					},
				],
			}),
		).rejects.toThrow("credential-like property");
		await expect(
			probeOpenAICompatibility({
				model,
				apiKey: API_KEY,
				fetch,
				candidates: [
					{
						name: "credential-key",
						source: "user",
						compat: { extraBody: { [API_KEY]: "value" } },
					},
				],
			}),
		).rejects.toThrow("API credential in a property name");
		expect(() => validateStrictOpenAICompat({ thinkingFormat: "invalid" })).toThrow("Invalid OpenAI compatibility");
		expect(() => validateStrictOpenAICompat({ extraBody: { value: 1n } })).toThrow("non-persistable bigint");
		expect(() => validateStrictOpenAICompat({ extraBody: { value: Number.NaN } })).toThrow("finite numbers");
		expect(() => validateStrictOpenAICompat({ extraBody: { [Symbol("key")]: true } })).toThrow("symbol keys");
		await expect(probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 3_600_001 })).rejects.toThrow(
			"timeoutMs must be an integer from 1 through 3600000",
		);
		expect(calls).toBe(0);
	});

	it("honors cancellation during awaited progress callbacks", async () => {
		const controller = new AbortController();
		let calls = 0;
		let completed = 0;
		const fetch = fetchImpl(body => {
			calls += 1;
			return sseResponse(body, true);
		});
		const model = buildModel(modelSpec("https://progress-cancel.invalid/v1"));
		const run = probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch,
			signal: controller.signal,
			async onProgress(event) {
				await Promise.resolve();
				if (event.type === "scenario_complete") {
					completed += 1;
					controller.abort(new DOMException("cancelled during progress", "AbortError"));
				}
			},
		});

		await expect(run).rejects.toThrow("cancelled during progress");
		expect(completed).toBe(1);
		expect(calls).toBe(1);
	});

	it("records strict fallback, transient sampling, silent usage, and request shapes", async () => {
		const fetch = fetchImpl(body => {
			if (hasStrictTrue(body)) return errorResponse(400, "tools strict unsupported", "tools");
			if (Object.hasOwn(body, "temperature")) {
				return errorResponse(429, "sampling temporarily unavailable", undefined, { "retry-after": "0" });
			}
			return sseResponse(body, false);
		});
		const model = buildModel(
			modelSpec(
				"https://api.openai.com/v1",
				{ supportsSamplingParams: true, supportsStrictMode: true },
				{ provider: "openai" },
			),
		);
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });
		expect(report.viable).toBe(true);
		expect(finding(report, "supportsStrictMode").recommendedValue).toBeUndefined();
		expect(finding(report, "toolStrictMode")).toMatchObject({
			verdict: "required",
			recommendedValue: "none",
		});
		expect(finding(report, "supportsSamplingParams")).toMatchObject({
			verdict: "indeterminate",
			recommendedValue: true,
		});
		expect(report.recommendedCompat.supportsSamplingParams).toBe(true);
		expect(finding(report, "supportsUsageInStreaming")).toMatchObject({
			verdict: "unsupported",
			recommendedValue: false,
		});
		const strictScenario = report.scenarios.find(scenario => scenario.baseId === "tool-strictness");
		expect(strictScenario?.attempts.some(attempt => attempt.classification === "strict_fallback")).toBe(true);
		const sampling = report.scenarios.find(scenario => scenario.baseId === "sampling-parameters");
		const enabledBody = (sampling?.evidence.enabledBody ?? {}) as Record<string, unknown>;
		for (const field of [
			"temperature",
			"top_p",
			"top_k",
			"min_p",
			"presence_penalty",
			"repetition_penalty",
			"frequency_penalty",
		]) {
			expect(enabledBody).toHaveProperty(field);
		}
		const maxTokens = report.scenarios.find(scenario => scenario.baseId === "max-tokens");
		expect(JSON.stringify(maxTokens?.evidence)).toContain("max_completion_tokens");
		expect(JSON.stringify(maxTokens?.evidence)).toContain("max_tokens");
		const systems = report.scenarios.find(scenario => scenario.baseId === "multiple-system-messages");
		expect(JSON.stringify(systems?.evidence)).toContain("FIRST");
		expect(JSON.stringify(systems?.evidence)).toContain("SECOND");
	});

	it("recommends streaming usage only when include_usage produces usage", async () => {
		const fetch = fetchImpl(body => {
			const streamOptions =
				typeof body.stream_options === "object" &&
				body.stream_options !== null &&
				!Array.isArray(body.stream_options)
					? (body.stream_options as Record<string, unknown>)
					: undefined;
			return sseResponse(body, streamOptions?.include_usage === true);
		});
		const model = buildModel(modelSpec("https://streaming-usage.invalid/v1"));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(finding(report, "supportsUsageInStreaming")).toMatchObject({
			verdict: "effective",
			recommendedValue: true,
		});
		expect(report.resolvedRecommended.supportsUsageInStreaming).toBe(true);
		const scenario = report.scenarios.find(candidate => candidate.baseId === "streaming-usage");
		expect(scenario?.evidence).toMatchObject({
			enabledRawUsageObserved: true,
			disabledRawUsageObserved: false,
		});
	});

	it("treats a post-header stream watchdog as transient evidence", async () => {
		const fetch = fetchImpl(body => (Object.hasOwn(body, "store") ? stalledSseResponse() : sseResponse(body, true)));
		const model = buildModel(modelSpec("https://post-header-timeout.invalid/v1"));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 25 });

		expect(report.viable).toBe(true);
		expect(finding(report, "supportsStore").verdict).toBe("indeterminate");
		expect(Object.hasOwn(finding(report, "supportsStore"), "recommendedValue")).toBe(false);
		const scenario = report.scenarios.find(candidate => candidate.baseId === "store");
		expect(scenario?.attempts.some(attempt => attempt.response?.status === 200)).toBe(true);
	});

	it("tests candidate profiles independently and leaf-minimizes a combined extraBody profile", async () => {
		const fetch = fetchImpl(body => {
			if (
				Array.isArray(body.tools) &&
				body.tools.length > 0 &&
				(body.probe_flag !== "ok" || body["a.b"] !== "required")
			) {
				return errorResponse(400, "probe flags are required for tools", "probe_flag");
			}
			return sseResponse(body, true);
		});
		const model = buildModel(modelSpec("https://candidate.invalid/v1"));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch,
			timeoutMs: 5_000,
			candidates: [
				{ name: "incomplete", source: "user", compat: { extraBody: { secondary_flag: "keep" } } },
				{
					name: "combined",
					source: "user",
					compat: {
						extraBody: {
							probe_flag: "ok",
							"a.b": "required",
							a: { b: "remove-me" },
							secondary_flag: "remove-me",
						},
					},
				},
			],
		});
		expect(report.viable).toBe(true);
		expect(report.recommendedCompat.extraBody).toEqual({ probe_flag: "ok", "a.b": "required" });
		const scenario = report.scenarios.find(candidate => candidate.baseId === "candidate-profiles");
		expect(scenario?.evidence.selected).toBe("combined");
		const profiles = scenario?.evidence.profiles as Array<Record<string, unknown>>;
		expect(profiles.find(profile => profile.name === "incomplete")?.semantic).toBe(false);
		expect(profiles.find(profile => profile.name === "combined")?.semantic).toBe(true);
		expect(JSON.stringify(profiles.find(profile => profile.name === "combined")?.body)).toContain("secondary_flag");
		expect(report.scenarios.some(candidate => candidate.id === "final@minimize:%2FextraBody%2Fsecondary_flag")).toBe(
			true,
		);
		expect(report.scenarios.some(candidate => candidate.id === "final@minimize:%2FextraBody%2Fa.b")).toBe(true);
		expect(report.scenarios.some(candidate => candidate.id === "final@minimize:%2FextraBody%2Fa%2Fb")).toBe(true);
	});

	it("selects equally ranked Unicode candidate names deterministically", async () => {
		const fetch = fetchImpl(body => {
			if (Array.isArray(body.tools) && body.tools.length > 0 && body.candidate_pick === undefined) {
				return errorResponse(400, "candidate_pick is required", "candidate_pick");
			}
			return sseResponse(body, true);
		});
		const model = buildModel(modelSpec("https://candidate-order.invalid/v1"));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch,
			timeoutMs: 5_000,
			candidates: [
				{ name: "ä-profile", source: "user", compat: { extraBody: { candidate_pick: "umlaut" } } },
				{ name: "z-profile", source: "user", compat: { extraBody: { candidate_pick: "ascii" } } },
			],
		});

		expect(report.viable).toBe(true);
		const scenario = report.scenarios.find(candidate => candidate.baseId === "candidate-profiles");
		expect(scenario?.evidence.selected).toBe("z-profile");
		expect(report.recommendedCompat.extraBody).toEqual({ candidate_pick: "ascii" });
	});

	it("proves store, named choice, and tool-result name corrections on loopback", async () => {
		const received: Record<string, unknown>[] = [];
		const server = Bun.serve({
			hostname: "127.0.0.1",
			port: 0,
			async fetch(request) {
				if (new URL(request.url).pathname !== "/v1/chat/completions")
					return new Response("not found", { status: 404 });
				const body = (await request.json()) as Record<string, unknown>;
				received.push(body);
				if (Object.hasOwn(body, "store")) return errorResponse(400, "store unsupported", "store");
				if (typeof body.tool_choice === "object" && body.tool_choice !== null) {
					return errorResponse(400, "named tool_choice unsupported", "tool_choice");
				}
				if (
					Array.isArray(body.messages) &&
					body.messages.some(message => {
						if (typeof message !== "object" || message === null || Array.isArray(message)) return false;
						const object = message as Record<string, unknown>;
						return object.role === "tool" && typeof object.name !== "string";
					})
				) {
					return errorResponse(400, "tool result name required", "messages.name");
				}
				return sseResponse(body, true);
			},
		});
		runningServers.push(server);
		const model = buildModel(
			modelSpec(`http://127.0.0.1:${server.port}/v1`, {
				supportsStore: true,
				supportsNamedToolChoice: true,
				requiresToolResultName: false,
			}),
		);
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, timeoutMs: 5_000 });
		expect(report.viable).toBe(true);
		expect(report.fallbackFree).toBe(true);
		expect(finding(report, "supportsStore")).toMatchObject({ verdict: "unsupported", recommendedValue: false });
		expect(finding(report, "supportsNamedToolChoice")).toMatchObject({
			verdict: "unsupported",
			recommendedValue: false,
		});
		expect(finding(report, "requiresToolResultName")).toMatchObject({ verdict: "required", recommendedValue: true });
		expect(finding(report, "supportsUsageInStreaming").verdict).toBe("effective");
		const changedPaths = new Set([...leafPaths(report.recommendedCompat), ...report.removeConfiguredPaths]);
		expect(changedPaths).toEqual(new Set(["/supportsStore", "/supportsNamedToolChoice", "/requiresToolResultName"]));
		const validation = report.scenarios.find(scenario => scenario.id === "final@validate");
		expect(validation?.attempts).toHaveLength(3);
		expect(validation?.attempts.every(attempt => attempt.classification === "first")).toBe(true);
		expect(received.length).toBeGreaterThan(0);
		const yaml = formatOpenAICompatibilityYaml(report);
		const parsedYaml = Bun.YAML.parse(yaml) as { compat?: OpenAICompat };
		expect(parsedYaml.compat).toEqual(report.recommendedCompat);
		expect(validateStrictOpenAICompat(parsedYaml.compat)).toEqual(report.recommendedCompat);
		const json = formatOpenAICompatibilityReport(report);
		expect(json.endsWith("\n")).toBe(true);
		const ids = report.scenarios.map(scenario => scenario.id);
		expect(new Set(ids).size).toBe(ids.length);
	});

	it("keeps retry-noisy alternatives indeterminate instead of recommending clean controls", async () => {
		let maxTokenNoiseInjected = false;
		let strictnessNoiseInjected = false;
		const fetch = fetchImpl(body => {
			const tools = Array.isArray(body.tools) ? body.tools : [];
			const strictCount = tools.filter(tool => {
				if (typeof tool !== "object" || tool === null || Array.isArray(tool)) return false;
				const fn = (tool as Record<string, unknown>).function;
				return (
					typeof fn === "object" &&
					fn !== null &&
					!Array.isArray(fn) &&
					(fn as Record<string, unknown>).strict === true
				);
			}).length;
			if (!maxTokenNoiseInjected && tools.length === 0 && body.max_completion_tokens === 64) {
				maxTokenNoiseInjected = true;
				return emptySseResponse(body);
			}
			if (!strictnessNoiseInjected && tools.length === 2 && strictCount === 1) {
				strictnessNoiseInjected = true;
				return emptySseResponse(body);
			}
			return sseResponse(body, true);
		});
		const model = buildModel(modelSpec("https://retry-noise.invalid/v1"));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(maxTokenNoiseInjected).toBe(true);
		expect(strictnessNoiseInjected).toBe(true);
		expect(report.viable).toBe(true);
		expect(finding(report, "maxTokensField").verdict).toBe("indeterminate");
		expect(finding(report, "toolStrictMode").verdict).toBe("indeterminate");
		expect(Object.hasOwn(report.recommendedCompat, "maxTokensField")).toBe(false);
		expect(Object.hasOwn(report.recommendedCompat, "toolStrictMode")).toBe(false);
		const maxTokens = report.scenarios.find(scenario => scenario.baseId === "max-tokens");
		const strictness = report.scenarios.find(scenario => scenario.baseId === "tool-strictness");
		expect(maxTokens?.attempts.some(attempt => attempt.classification === "empty_retry")).toBe(true);
		expect(strictness?.attempts.some(attempt => attempt.classification === "empty_retry")).toBe(true);
	});

	it("derives fallback-free effort maps and isolates reasoning-history encodings", async () => {
		const spec = modelSpec("https://effort-map.invalid/v1", {}, { reasoning: true });
		spec.thinking = { mode: "effort", efforts: [Effort.Low, Effort.Medium, Effort.High, Effort.XHigh] };
		const model = buildModel(spec);
		const acceptedEfforts = new Set(["low", "medium", "high"]);
		const fetch = fetchImpl(body => {
			if (typeof body.reasoning_effort === "string" && !acceptedEfforts.has(body.reasoning_effort)) {
				return errorResponse(400, "reasoning_effort must be one of 'low', 'medium', or 'high'", "reasoning_effort");
			}
			return sseResponse(body, true, typeof body.reasoning_effort === "string" ? ["reasoning"] : []);
		});
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(report.fallbackFree).toBe(true);
		expect(report.resolvedRecommended.supportsReasoningParams).toBe(true);
		expect(report.resolvedRecommended.supportsReasoningEffort).toBe(true);
		expect(report.resolvedRecommended.omitReasoningEffort).toBe(false);
		expect(report.recommendedCompat.reasoningEffortMap).toEqual({
			minimal: "low",
			xhigh: "high",
			max: "high",
		});
		expect(finding(report, "reasoningEffortMap")).toMatchObject({
			verdict: "required",
			recommendedValue: { minimal: "low", xhigh: "high", max: "high" },
		});

		const contentScenario = report.scenarios.find(scenario => scenario.baseId === "history-reasoning-content");
		const requirement = contentScenario?.evidence.requirement as Record<string, unknown>;
		const requiredBody = requirement.required;
		const omittedBody = requirement.omitted;
		expect(countProperty(requiredBody, "reasoning_content")).toBeGreaterThan(
			countProperty(omittedBody, "reasoning_content"),
		);
		const synthetic = contentScenario?.evidence.synthetic as Record<string, unknown>;
		const placeholderValues = propertyValues(synthetic.placeholder, "reasoning_content");
		const exactValues = propertyValues(synthetic.exact, "reasoning_content");
		expect(
			placeholderValues.some(
				value =>
					typeof value === "object" &&
					value !== null &&
					!Array.isArray(value) &&
					(value as Record<string, unknown>).length === 1,
			),
		).toBe(true);
		expect(
			exactValues.some(
				value =>
					typeof value === "object" &&
					value !== null &&
					!Array.isArray(value) &&
					(value as Record<string, unknown>).length === 0,
			),
		).toBe(true);

		const allAssistant = report.scenarios.find(scenario => scenario.baseId === "history-reasoning-all-assistant");
		expect(countProperty(allAssistant?.evidence.allAssistant, "reasoning_content")).toBeGreaterThan(
			countProperty(allAssistant?.evidence.toolOnly, "reasoning_content"),
		);
		const replay = report.scenarios.find(scenario => scenario.baseId === "history-reasoning-replay");
		const replayEvidence = replay?.evidence.replay as Record<string, unknown>;
		expect(countProperty(replayEvidence.enabled, "reasoning_content")).toBeGreaterThan(
			countProperty(replayEvidence.disabled, "reasoning_content"),
		);
		const qwenEvidence = replay?.evidence.qwenPreserve as Record<string, Record<string, unknown>>;
		const enabledKwargs = qwenEvidence.enabled.chat_template_kwargs as Record<string, unknown>;
		const disabledKwargs = qwenEvidence.disabled.chat_template_kwargs as Record<string, unknown> | undefined;
		expect(enabledKwargs.preserve_thinking).toBe(true);
		expect(disabledKwargs?.preserve_thinking).not.toBe(true);
	});

	it("validates a reasoning model whose clean profile omits unsupported effort", async () => {
		const fetch = fetchImpl(body => {
			const nestedReasoning =
				typeof body.reasoning === "object" && body.reasoning !== null && !Array.isArray(body.reasoning)
					? (body.reasoning as Record<string, unknown>)
					: undefined;
			if (
				body.reasoning_effort !== undefined ||
				nestedReasoning?.effort !== undefined ||
				body.reasoning !== undefined ||
				body.thinking !== undefined ||
				body.enable_thinking !== undefined ||
				body.chat_template_kwargs !== undefined
			) {
				return errorResponse(400, "reasoning controls are unsupported", "reasoning_effort");
			}
			return sseResponse(body, true, ["reasoning"]);
		});
		const model = buildModel(modelSpec("https://reasoning-no-effort.invalid/v1", {}, { reasoning: true }));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(report.fallbackFree).toBe(true);
		expect(finding(report, "supportsReasoningEffort").verdict).toBe("fallback_only");
		const final = report.scenarios.find(scenario => scenario.id === "final@validate");
		expect(final?.verdict).toBe("effective");
		expect(
			final?.attempts.every(
				attempt =>
					countProperty(attempt.request.body, "reasoning_effort") === 0 &&
					countProperty(attempt.request.body, "effort") === 0,
			),
		).toBe(true);
	});

	it("persists the only reasoning format with direct semantic evidence", async () => {
		const fetch = fetchImpl(body => {
			const openRouterReasoning =
				typeof body.reasoning === "object" && body.reasoning !== null && !Array.isArray(body.reasoning)
					? (body.reasoning as Record<string, unknown>)
					: undefined;
			const semanticallyEnabled = openRouterReasoning !== undefined && openRouterReasoning.enabled !== false;
			return sseResponse(body, true, semanticallyEnabled ? ["semantic reasoning"] : []);
		});
		const model = buildModel(modelSpec("https://reasoning-format.invalid/v1", {}, { reasoning: true }));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(report.recommendedCompat.thinkingFormat).toBe("openrouter");
		expect(finding(report, "thinkingFormat")).toMatchObject({
			verdict: "effective",
			recommendedValue: "openrouter",
		});
		expect(finding(report, "supportsReasoningParams")).toMatchObject({
			verdict: "effective",
			recommendedValue: true,
		});
		expect(report.resolvedRecommended.supportsReasoningEffort).toBe(true);
		expect(report.resolvedRecommended.omitReasoningEffort).toBe(false);
	});
	it("prefers a clean baseline format over an equally semantic configured format", async () => {
		const fetch = fetchImpl(body => {
			const nestedReasoning =
				typeof body.reasoning === "object" && body.reasoning !== null && !Array.isArray(body.reasoning)
					? (body.reasoning as Record<string, unknown>)
					: undefined;
			const thinking =
				typeof body.thinking === "object" && body.thinking !== null && !Array.isArray(body.thinking)
					? (body.thinking as Record<string, unknown>)
					: undefined;
			const template =
				typeof body.chat_template_kwargs === "object" &&
				body.chat_template_kwargs !== null &&
				!Array.isArray(body.chat_template_kwargs)
					? (body.chat_template_kwargs as Record<string, unknown>)
					: undefined;
			const semanticallyEnabled =
				typeof body.reasoning_effort === "string" ||
				(nestedReasoning !== undefined && nestedReasoning.enabled !== false) ||
				thinking?.type === "enabled" ||
				body.enable_thinking === true ||
				template?.enable_thinking === true;
			return sseResponse(body, true, semanticallyEnabled ? ["semantic reasoning"] : []);
		});
		const model = buildModel(
			modelSpec("https://reasoning-baseline.invalid/v1", { thinkingFormat: "openrouter" }, { reasoning: true }),
		);
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(Object.hasOwn(report.recommendedCompat, "thinkingFormat")).toBe(false);
		expect(report.resolvedRecommended.thinkingFormat).toBe("openai");
		expect(finding(report, "thinkingFormat").recommendedValue).toBeUndefined();
		expect(report.removeConfiguredPaths).toContain("/thinkingFormat");
	});

	it("does not recommend reasoning controls that produce no semantic difference", async () => {
		const model = buildModel(modelSpec("https://reasoning-tolerated.invalid/v1", {}, { reasoning: true }));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch: fetchImpl(body => sseResponse(body, true)),
			timeoutMs: 5_000,
		});

		expect(report.viable).toBe(true);
		expect(finding(report, "supportsReasoningParams").recommendedValue).toBeUndefined();
		expect(finding(report, "supportsReasoningEffort").recommendedValue).toBeUndefined();
		expect(Object.hasOwn(report.recommendedCompat, "supportsReasoningParams")).toBe(false);
		expect(Object.hasOwn(report.recommendedCompat, "supportsReasoningEffort")).toBe(false);
		expect(Object.hasOwn(report.recommendedCompat, "omitReasoningEffort")).toBe(false);
		const effort = report.scenarios.find(scenario => scenario.baseId === "reasoning-effort");
		expect(effort?.verdict).toBe("accepted");
		expect((effort?.evidence.noEffort as Record<string, unknown> | undefined)?.parsedThinking).toBe(false);
	});

	it("does not treat thinking present in enabled and disabled arms as reasoning control evidence", async () => {
		const model = buildModel(modelSpec("https://reasoning-always.invalid/v1", {}, { reasoning: true }));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch: fetchImpl(body => sseResponse(body, true, ["unconditional reasoning"])),
			timeoutMs: 5_000,
		});

		expect(report.viable).toBe(true);
		expect(finding(report, "supportsReasoningParams").recommendedValue).toBeUndefined();
		expect(Object.hasOwn(report.recommendedCompat, "supportsReasoningParams")).toBe(false);
		const format = report.scenarios.find(scenario => scenario.baseId === "reasoning-format");
		expect(format?.verdict).toBe("accepted");
		const formats = format?.evidence.formats as Array<Record<string, unknown>>;
		expect(formats.every(candidate => candidate.parsedThinking && candidate.disabledParsedThinking)).toBe(true);
	});
	it("persists reasoning format controls when omission is rejected", async () => {
		let activeScenario = "";
		const fetch = fetchImpl(body => {
			if (activeScenario !== "reasoning-format@probe") return sseResponse(body, true);
			const nestedReasoning =
				typeof body.reasoning === "object" && body.reasoning !== null && !Array.isArray(body.reasoning)
					? (body.reasoning as Record<string, unknown>)
					: undefined;
			const hasControl =
				body.reasoning_effort !== undefined ||
				nestedReasoning !== undefined ||
				body.thinking !== undefined ||
				body.enable_thinking !== undefined ||
				body.chat_template_kwargs !== undefined;
			if (!hasControl) return errorResponse(400, "reasoning control is required", "reasoning");
			return sseResponse(
				body,
				true,
				nestedReasoning?.enabled !== false && nestedReasoning ? ["required reasoning"] : [],
			);
		});
		const model = buildModel(modelSpec("https://reasoning-format-required.invalid/v1", {}, { reasoning: true }));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch,
			timeoutMs: 5_000,
			onProgress(event) {
				if (event.type === "scenario_start") activeScenario = event.scenarioId;
			},
		});

		expect(report.viable).toBe(true);
		expect(report.recommendedCompat.thinkingFormat).toBe("openrouter");
		expect(finding(report, "thinkingFormat")).toMatchObject({
			verdict: "required",
			recommendedValue: "openrouter",
		});
		expect(finding(report, "supportsReasoningParams")).toMatchObject({
			verdict: "required",
			recommendedValue: true,
		});
		expect(report.resolvedRecommended.supportsReasoningParams).toBe(true);
	});

	it("persists reasoning effort controls when omission is rejected", async () => {
		let activeScenario = "";
		const fetch = fetchImpl(body => {
			if (activeScenario !== "reasoning-effort@probe") return sseResponse(body, true);
			const nestedReasoning =
				typeof body.reasoning === "object" && body.reasoning !== null && !Array.isArray(body.reasoning)
					? (body.reasoning as Record<string, unknown>)
					: undefined;
			const hasEffort = typeof body.reasoning_effort === "string" || typeof nestedReasoning?.effort === "string";
			if (!hasEffort) return errorResponse(400, "reasoning effort is required", "reasoning_effort");
			return sseResponse(body, true, ["required effort reasoning"]);
		});
		const model = buildModel(modelSpec("https://reasoning-effort-required.invalid/v1", {}, { reasoning: true }));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch,
			timeoutMs: 5_000,
			onProgress(event) {
				if (event.type === "scenario_start") activeScenario = event.scenarioId;
			},
		});

		expect(report.viable).toBe(true);
		expect(finding(report, "supportsReasoningParams")).toMatchObject({
			verdict: "required",
			recommendedValue: true,
		});
		expect(finding(report, "supportsReasoningEffort")).toMatchObject({
			verdict: "required",
			recommendedValue: true,
		});
		expect(finding(report, "omitReasoningEffort")).toMatchObject({
			verdict: "required",
			recommendedValue: false,
		});
		expect(report.resolvedRecommended.supportsReasoningParams).toBe(true);
		expect(report.resolvedRecommended.supportsReasoningEffort).toBe(true);
		expect(report.resolvedRecommended.omitReasoningEffort).toBe(false);
	});

	it("persists tool-choice controls that alone produce the requested call", async () => {
		const fetch = fetchImpl(body => {
			const isToolTurn =
				Array.isArray(body.tools) &&
				body.tools.length > 0 &&
				!hasToolResult(body) &&
				/compat_echo/u.test(lastUserText(body));
			if (isToolTurn && !Object.hasOwn(body, "tool_choice")) return textSseResponse(body);
			return sseResponse(body, true);
		});
		const model = buildModel(modelSpec("https://tool-choice-semantic.invalid/v1"));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(finding(report, "supportsToolChoice")).toMatchObject({
			verdict: "effective",
			recommendedValue: true,
		});
		const prevalidation = report.scenarios.find(scenario => scenario.id === "prevalidation@probe");
		expect(prevalidation?.verdict).toBe("effective");
	});

	it("activates prerequisites for forced and named tool-choice winners", async () => {
		for (const mode of ["required", "named"] as const) {
			const fetch = fetchImpl(body => {
				const isToolTurn =
					Array.isArray(body.tools) &&
					body.tools.length > 0 &&
					!hasToolResult(body) &&
					/compat_echo/u.test(lastUserText(body));
				const choice = body.tool_choice;
				const namedChoice =
					typeof choice === "object" &&
					choice !== null &&
					!Array.isArray(choice) &&
					typeof (choice as Record<string, unknown>).function === "object";
				const selected = mode === "required" ? choice === "required" : namedChoice;
				if (isToolTurn && !selected) return textSseResponse(body);
				return sseResponse(body, true);
			});
			const model = buildModel(modelSpec(`https://tool-choice-${mode}.invalid/v1`));
			const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

			expect(report.viable).toBe(true);
			expect(report.resolvedRecommended.supportsToolChoice).toBe(true);
			expect(report.resolvedRecommended.supportsForcedToolChoice).toBe(true);
			expect(report.resolvedRecommended.supportsNamedToolChoice).toBe(mode === "named");
			expect(report.scenarios.find(scenario => scenario.id === "prevalidation@probe")?.verdict).toBe("effective");
		}
	});
	it("persists strict-mode support required by an all-strict winner", async () => {
		let strictWinnerSeen = false;
		const fetch = fetchImpl(body => {
			const tools = Array.isArray(body.tools)
				? body.tools.filter(
						(tool): tool is Record<string, unknown> =>
							typeof tool === "object" && tool !== null && !Array.isArray(tool),
					)
				: [];
			const hasRelaxedTool = tools.some(tool => JSON.stringify(tool).includes("compat_echo_relaxed"));
			const allStrict = tools.length > 0 && tools.every(tool => hasStrictTrue(tool));
			const anyStrict = tools.some(tool => hasStrictTrue(tool));
			if (hasRelaxedTool) {
				if (tools.length !== 2) return errorResponse(400, "expected two tools", "tools");
				if (allStrict) strictWinnerSeen = true;
				else if (anyStrict) return errorResponse(400, "mixed values for 'strict'", "tools");
			} else if (strictWinnerSeen && tools.length > 0 && !allStrict) {
				return errorResponse(400, "strict marker required", "tools");
			}
			return sseResponse(body, true);
		});
		const model = buildModel(modelSpec("https://all-strict.invalid/v1"));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(report.recommendedCompat.supportsStrictMode).toBe(true);
		expect(report.recommendedCompat.toolStrictMode).toBe("all_strict");
		expect(finding(report, "supportsStrictMode")).toMatchObject({
			verdict: "required",
			recommendedValue: true,
		});
		const strictness = report.scenarios.find(scenario => scenario.baseId === "tool-strictness");
		expect(propertyValues(strictness?.evidence.allStrict, "tools")[0]).toHaveLength(2);
		const final = report.scenarios.find(scenario => scenario.id === "final@validate");
		const finalToolBodies = final?.attempts
			.map(attempt => attempt.request.body)
			.filter(body => Array.isArray((body as Record<string, unknown>).tools)) as Record<string, unknown>[];
		expect(finalToolBodies.length).toBeGreaterThan(0);
		expect(finalToolBodies.every(body => hasStrictTrue(body.tools))).toBe(true);
	});
	it("persists mixed strict support when only normal strict markers produce the tool call", async () => {
		const fetch = fetchImpl(body => {
			const tools = Array.isArray(body.tools)
				? body.tools.filter(
						(tool): tool is Record<string, unknown> =>
							typeof tool === "object" && tool !== null && !Array.isArray(tool),
					)
				: [];
			if (tools.length === 0) return sseResponse(body, true);
			const strictCount = tools.filter(tool => hasStrictTrue(tool)).length;
			const mixedStrict = (tools.length === 2 && strictCount === 1) || (tools.length === 1 && strictCount === 1);
			return mixedStrict ? sseResponse(body, true) : textSseResponse(body);
		});
		const model = buildModel(modelSpec("https://mixed-strict.invalid/v1"));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(report.recommendedCompat.supportsStrictMode).toBe(true);
		expect(Object.hasOwn(report.recommendedCompat, "toolStrictMode")).toBe(false);
		expect(finding(report, "supportsStrictMode")).toMatchObject({
			verdict: "required",
			recommendedValue: true,
		});
	});

	it("keeps strictness variants' tool sets identical", async () => {
		const fetch = fetchImpl(body =>
			Array.isArray(body.tools) && body.tools.length > 1
				? errorResponse(400, "only one tool is supported", "tools")
				: sseResponse(body, true),
		);
		const model = buildModel(modelSpec("https://single-tool.invalid/v1"));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(finding(report, "toolStrictMode").recommendedValue).toBeUndefined();
		const strictness = report.scenarios.find(scenario => scenario.baseId === "tool-strictness");
		const strictEvidence = strictness?.evidence as Record<string, unknown> | undefined;
		for (const variant of ["mixed", "allStrict", "none"]) {
			expect(propertyValues(strictEvidence?.[variant], "tools")[0]).toHaveLength(2);
		}
	});
	it("prefers an all-strict semantic winner over mixed direct text", async () => {
		const fetch = fetchImpl(body => {
			const tools = Array.isArray(body.tools) ? body.tools : [];
			const hasRelaxedTool = JSON.stringify(tools).includes("compat_echo_relaxed");
			if (!hasRelaxedTool) return sseResponse(body, true);
			const allStrict =
				tools.length > 0 &&
				tools.every(
					tool => typeof tool === "object" && tool !== null && !Array.isArray(tool) && hasStrictTrue(tool),
				);
			return allStrict ? sseResponse(body, true) : textSseResponse(body);
		});
		const model = buildModel(modelSpec("https://strict-semantic.invalid/v1"));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(report.recommendedCompat.supportsStrictMode).toBe(true);
		expect(report.recommendedCompat.toolStrictMode).toBe("all_strict");
		const strictness = report.scenarios.find(scenario => scenario.baseId === "tool-strictness");
		expect((strictness?.evidence.mixed as Record<string, unknown> | undefined)?.semantic).toBe(false);
		expect((strictness?.evidence.allStrict as Record<string, unknown> | undefined)?.semantic).toBe(true);
	});

	it("prefers direct no-strict semantics over mixed fallback and all-strict text", async () => {
		const fetch = fetchImpl(body => {
			const tools = Array.isArray(body.tools) ? body.tools : [];
			const hasRelaxedTool = JSON.stringify(tools).includes("compat_echo_relaxed");
			if (!hasRelaxedTool) return sseResponse(body, true);
			const allStrict =
				tools.length > 0 &&
				tools.every(
					tool => typeof tool === "object" && tool !== null && !Array.isArray(tool) && hasStrictTrue(tool),
				);
			const anyStrict = tools.some(
				tool => typeof tool === "object" && tool !== null && !Array.isArray(tool) && hasStrictTrue(tool),
			);
			if (allStrict) return textSseResponse(body);
			if (anyStrict) return errorResponse(400, "mixed values for 'strict'", "tools");
			return sseResponse(body, true);
		});
		const model = buildModel(modelSpec("https://no-strict-semantic.invalid/v1"));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(report.recommendedCompat.toolStrictMode).toBe("none");
		expect(finding(report, "supportsStrictMode").recommendedValue).toBeUndefined();
		expect(finding(report, "toolStrictMode")).toMatchObject({
			verdict: "required",
			recommendedValue: "none",
		});
		const strictness = report.scenarios.find(scenario => scenario.baseId === "tool-strictness");
		expect((strictness?.evidence.mixed as Record<string, unknown> | undefined)?.fallback).toBe(true);
		expect((strictness?.evidence.allStrict as Record<string, unknown> | undefined)?.semantic).toBe(false);
		expect((strictness?.evidence.none as Record<string, unknown> | undefined)?.semantic).toBe(true);
	});

	it("prefers the tool schema flavor with structured-call evidence", async () => {
		let activeScenario = "";
		const fetch = fetchImpl(body => {
			const isToolTurn =
				Array.isArray(body.tools) &&
				body.tools.length > 0 &&
				!hasToolResult(body) &&
				/compat_echo/u.test(lastUserText(body));
			if (activeScenario === "tool-strictness@probe" && isToolTurn) return textSseResponse(body);
			if (activeScenario === "tool-schema-flavor@probe" && isToolTurn && countProperty(body, "const") > 0) {
				return textSseResponse(body);
			}
			return sseResponse(body, true);
		});
		const model = buildModel(modelSpec("https://tool-flavor-semantic.invalid/v1"));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch,
			timeoutMs: 5_000,
			onProgress(event) {
				if (event.type === "scenario_start") activeScenario = event.scenarioId;
			},
		});

		expect(report.viable).toBe(true);
		expect(report.recommendedCompat.toolSchemaFlavor).toBe("moonshot-mfjs");
		expect(finding(report, "toolSchemaFlavor")).toMatchObject({
			verdict: "effective",
			recommendedValue: "moonshot-mfjs",
		});
	});

	it("keeps final validation on the selected Chat Completions endpoint", async () => {
		const urls: string[] = [];
		const implementation = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
			const url = input instanceof Request ? input.url : String(input);
			urls.push(url);
			if (!new URL(url).pathname.endsWith("/chat/completions"))
				return new Response("wrong endpoint", { status: 404 });
			return sseResponse(requestBody(init), true);
		};
		const fetch = Object.assign(implementation, { preconnect: globalThis.fetch.preconnect });
		const model = buildModel(modelSpec("https://kimi-path.invalid/v1", {}, { provider: "kimi-code" }));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(urls.length).toBeGreaterThan(0);
		expect(urls.every(url => new URL(url).pathname.endsWith("/chat/completions"))).toBe(true);
	});

	it("does not infer cumulative reasoning deltas across separate wire attempts", async () => {
		let requestOrdinal = 0;
		const fetch = fetchImpl(body => {
			requestOrdinal += 1;
			return sseResponse(body, true, ["r".repeat(requestOrdinal)]);
		});
		const model = buildModel(modelSpec("https://delta-sequences.invalid/v1", {}, { reasoning: true }));
		const report = await probeOpenAICompatibility({ model, apiKey: API_KEY, fetch, timeoutMs: 5_000 });

		expect(report.viable).toBe(true);
		expect(finding(report, "reasoningDeltasMayBeCumulative").verdict).toBe("indeterminate");
		expect(Object.hasOwn(report.recommendedCompat, "reasoningDeltasMayBeCumulative")).toBe(false);
		const stream = report.scenarios.find(scenario => scenario.baseId === "stream-observation");
		expect(stream?.evidence.cumulativeReasoningObserved).toBe(false);
		const firstEventSamples = stream?.evidence.timeToFirstEventMs as number[];
		expect(firstEventSamples.length).toBeGreaterThan(0);
		expect(firstEventSamples.every(sample => sample >= 0)).toBe(true);
		expect(stream?.evidence.maximumInterEventGapMs).toBeGreaterThanOrEqual(0);
	});

	it("classifies a reasoning-effort retry from final serialized bodies", async () => {
		const model = buildModel(
			modelSpec("https://reasoning.invalid/v1", { supportsReasoningEffort: true }, { reasoning: true }),
		);
		const fetch = fetchImpl(body => {
			if (body.reasoning_effort === "xhigh") {
				return errorResponse(400, "reasoning_effort must be one of 'high'", "reasoning_effort");
			}
			const response = sseResponse(body, true);
			return response;
		});
		const observer = createOpenAICompatWireObserver(fetch, API_KEY);
		const context: Context = createBaselineContext();
		const message = await streamOpenAICompletions(model, context, {
			apiKey: API_KEY,
			fetch: observer.fetch,
			reasoning: Effort.XHigh,
			onPayload: payload => observer.onPayload(payload),
			onSseEvent: event => observer.onSseEvent(event),
		}).result();
		expect(message.stopReason).toBe("stop");
		expect(observer.attempts).toHaveLength(2);
		expect(observer.attempts.map(attempt => attempt.classification)).toEqual(["first", "reasoning_fallback"]);
		expect((observer.attempts[0].request.body as Record<string, unknown>).reasoning_effort).toBe("xhigh");
		expect((observer.attempts[1].request.body as Record<string, unknown>).reasoning_effort).toBe("high");
	});

	it("redacts wire credentials, URL values, errors, and generated assistant content", async () => {
		const inner = fetchImpl(() =>
			Response.json(
				{
					error: {
						message: `password=${API_KEY}`,
						param: "password",
						token: "error-token",
						padding: "x".repeat(20_000),
						detail: "opaque-error-secret",
						endpoint: "https://error-user:error-pass@example.test/v1?token=error-body-query",
						"opaque-provider-key-secret": "value",
					},
				},
				{
					status: 400,
					headers: { "set-cookie": "server-cookie", "X-Arbitrary": "response-header-secret" },
				},
			),
		);
		const observer = createOpenAICompatWireObserver(inner, API_KEY, ["compat-private-secret"]);
		await observer.fetch("https://alice:pw@example.test/v1?token=query-value&mode=private#fragment-private", {
			method: "POST",
			headers: {
				Authorization: `Bearer ${API_KEY}`,
				Cookie: "cookie-value",
				"X-Probe": API_KEY,
				"X-Gateway-Credential": "request-header-secret",
			},
			body: JSON.stringify({
				api_key: API_KEY,
				credential: "credential-value",
				authorization: "nested-authorization",
				token: "nested-token",
				gateway: "compat-private-secret",
				messages: [
					{
						role: "assistant",
						content: [{ type: "text", text: `generated ${API_KEY}` }],
						tool_calls: [
							{
								id: "provider-tool-id-opaque",
								type: "function",
								function: {
									name: "compat_echo",
									arguments: '{"value":"secret-generated-argument"}',
								},
							},
						],
					},
					{
						role: "tool",
						tool_call_id: "provider-tool-id-opaque",
						content: "probe result",
					},
				],
			}),
		});
		observer.onSseEvent({
			event: null,
			raw: [],
			data: JSON.stringify({
				choices: [{ delta: { content: "response-generated", reasoning_content: "response-reasoning" } }],
			}),
		});
		const evidence = observer.evidence();
		const sanitizedError = sanitizeProbeErrorMessage(
			"failed https://error-user:error-pw@example.test/v1?token=error-query authorization: Bearer error-bearer password=error-password",
			API_KEY,
		);
		const plainObserver = createOpenAICompatWireObserver(
			Object.assign(
				async (): Promise<Response> =>
					new Response(
						"Authorization: Bearer opaque-bearer-secret https://plain-user:plain-pass@example.test/?token=plain-query opaque-plain-secret",
						{ status: 400, headers: { "X-Plain": "plain-header-secret" } },
					),
				{ preconnect: globalThis.fetch.preconnect },
			),
			API_KEY,
		);
		await plainObserver.fetch("https://plain-error.invalid/v1", { method: "POST" });
		const serialized = JSON.stringify({
			attempts: observer.attempts,
			plainAttempts: plainObserver.attempts,
			evidence,
			sanitizedError,
		});
		const assistantIds = propertyValues(observer.attempts[0].request.body, "id");
		const toolResultIds = propertyValues(observer.attempts[0].request.body, "tool_call_id");
		expect(assistantIds).toHaveLength(1);
		expect(toolResultIds).toHaveLength(1);
		expect(assistantIds[0]).toEqual(toolResultIds[0]);
		for (const secret of [
			API_KEY,
			"alice",
			"pw",
			"query-value",
			"private",
			"cookie-value",
			"server-cookie",
			"credential-value",
			"fragment-private",
			"error-user",
			"error-pw",
			"error-query",
			"error-bearer",
			"error-password",
			"error-token",
			"generated",
			"nested-authorization",
			"nested-token",
			"response-reasoning",
			"secret-generated-argument",
			"opaque-error-secret",
			"error-pass",
			"error-body-query",
			"response-header-secret",
			"request-header-secret",
			"compat-private-secret",
			"opaque-bearer-secret",
			"plain-user",
			"plain-pass",
			"plain-query",
			"opaque-plain-secret",
			"plain-header-secret",
			"opaque-provider-key-secret",
			"provider-tool-id-opaque",
		]) {
			expect(serialized).not.toContain(secret);
		}
		expect(serialized).toContain("[REDACTED]");
		expect(serialized).toContain("sha256");
		expect(serialized).toContain("compat_echo");
		expect(serialized).toContain('"type":"text"');
		expect(evidence.contentDeltas).toEqual([{ redacted: true, length: "response-generated".length }]);
		expect(observer.attempts[0].response?.errorBody?.length).toBeLessThanOrEqual(4_097);
		expect(sanitizeProbeUrl("malformed-url-secret")).toBe("[INVALID_URL_REDACTED]");
		expect(sanitizeProbeUrl("file:///scheme-path-secret")).toBe("[INVALID_URL_REDACTED]");
	});
	it("omits arbitrary provider error details from scenario finals", async () => {
		const providerSecret = "opaque-customer-provider-secret";
		const model = buildModel(modelSpec("https://provider-error.invalid/v1", { supportsStore: true }));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch: fetchImpl(() => errorResponse(400, providerSecret)),
			timeoutMs: 5_000,
		});
		const json = formatOpenAICompatibilityReport(report);

		expect(report.viable).toBe(false);
		expect(json).not.toContain(providerSecret);
		expect(report.scenarios.every(scenario => !Object.hasOwn(scenario.final, "errorMessage"))).toBe(true);
		expect(report.removeConfiguredPaths).toEqual([]);
	});
	it("does not suggest removing configured paths after final validation fails", async () => {
		let finalValidation = false;
		const model = buildModel(modelSpec("https://final-failure.invalid/v1", { supportsStore: true }));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch: fetchImpl(body =>
				finalValidation ? errorResponse(400, "final validation rejected") : sseResponse(body, true),
			),
			timeoutMs: 5_000,
			onProgress(event) {
				if (event.type === "scenario_start" && event.scenarioId === "final@validate") finalValidation = true;
			},
		});

		expect(report.viable).toBe(false);
		expect(report.recommendedCompat).toEqual({});
		expect(report.removeConfiguredPaths).toEqual([]);
	});

	it("redacts nested compat values and omits embedded YAML from JSON", async () => {
		const model = buildModel(modelSpec("https://report-redaction.invalid/v1"));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch: fetchImpl(body => sseResponse(body, true)),
			timeoutMs: 5_000,
		});
		const endpoint = "https://report-user:report-pass@example.test/v1?token=report-query";
		report.target.baseUrl = "not-a-url?token=malformed-url-secret";
		report.configuredCompat.extraBody = { endpoint, "ä-key": "umlaut", "z-key": "ascii" };
		report.recommendedCompat = {
			extraBody: { endpoint, "ä-key": "umlaut", "z-key": "ascii" },
			reasoningEffortMap: { low: "opaque-map-secret" },
			whenThinking: { extraBody: { gateway: "nested-private-secret" } },
		};
		finding(report, "extraBody").recommendedValue = { endpoint };

		const json = formatOpenAICompatibilityReport(report);
		for (const secret of [
			"report-user",
			"report-pass",
			"report-query",
			"opaque-map-secret",
			"nested-private-secret",
			"malformed-url-secret",
		]) {
			expect(json).not.toContain(secret);
		}
		expect(json).toContain("[REDACTED]");
		expect(JSON.parse(json)).not.toHaveProperty("yamlFragment");
		const parsed = JSON.parse(json) as {
			target: { baseUrl: string };
			configuredCompat: { extraBody: Record<string, unknown> };
		};
		expect(parsed.target.baseUrl).toBe("[INVALID_URL_REDACTED]");
		expect(Object.keys(parsed.configuredCompat.extraBody)).toEqual(["endpoint", "z-key", "ä-key"]);

		const yaml = formatOpenAICompatibilityYaml(report);
		expect(yaml).toContain(endpoint);
		expect(yaml).toContain("opaque-map-secret");
		expect(yaml).toContain("nested-private-secret");
	});
	it("normalizes YAML-forbidden controls and line breaks in manual comments", async () => {
		const model = buildModel(modelSpec("https://yaml-comments.invalid/v1"));
		const report = await probeOpenAICompatibility({
			model,
			apiKey: API_KEY,
			fetch: fetchImpl(body => sseResponse(body, true)),
			timeoutMs: 5_000,
		});
		report.target.provider = "provider\u000Binjected: true\u0085continued";
		report.target.model = "model\u0080next: true\u2028line\u2029tail\uD800end";
		report.removeConfiguredPaths = [
			"/two\u000Cactive: true",
			"/one\u009Fnext: true",
			"/three\u007Ftail: true\uFFFEend",
		];

		const yaml = formatOpenAICompatibilityYaml(report);
		expect(yaml).not.toMatch(/[\u000B\u000C\u007F\u0080\u0085\u009F\u2028\u2029\uD800\uFFFE]/u);
		expect(yaml).toContain(
			"# OpenAI compatibility probe for provider injected: true continued/model next: true line tail end",
		);
		expect(yaml).toContain(
			"# Remove these paths at their originating config blocks: /one next: true, /three tail: true end, /two active: true",
		);
		expect(Bun.YAML.parse(yaml)).toEqual({ compat: report.recommendedCompat });
	});

	it("does not persist rejected probe requests outside the report", async () => {
		const root = await fs.mkdtemp(path.join(os.tmpdir(), "omp-probe-dump-"));
		const home = path.join(root, "home");
		await fs.mkdir(home);
		const script = `
			import { buildModel } from "@oh-my-pi/pi-catalog/build";
			import { probeOpenAICompatibility } from "@oh-my-pi/pi-coding-agent/openai-compat-probe";
			const model = buildModel({
				id: "probe-model",
				name: "Probe Model",
				api: "openai-completions",
				provider: "compat-probe",
				baseUrl: "https://dump.invalid/v1",
				reasoning: false,
				input: ["text"],
				supportsTools: true,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 8192,
				maxTokens: 1024,
				compat: {},
			});
			const probeFetch = Object.assign(
				async () => Response.json({ error: { message: "rejected" } }, { status: 400 }),
				{ preconnect: globalThis.fetch.preconnect },
			);
			await probeOpenAICompatibility({ model, apiKey: "probe-key", fetch: probeFetch, timeoutMs: 100 });
		`;
		try {
			const child = Bun.spawn([process.execPath, "-e", script], {
				cwd: path.resolve(import.meta.dir, ".."),
				env: {
					...process.env,
					BUN_ENV: "production",
					NODE_ENV: "production",
					HOME: home,
					PI_CODING_AGENT_DIR: path.join(home, "agent"),
					PI_CONFIG_DIR: ".omp-probe",
					XDG_CACHE_HOME: path.join(root, "cache"),
					XDG_DATA_HOME: path.join(root, "data"),
					XDG_STATE_HOME: path.join(root, "state"),
				},
				stdout: "ignore",
				stderr: "pipe",
			});
			const [stderr, exitCode] = await Promise.all([new Response(child.stderr).text(), child.exited]);
			expect(exitCode).toBe(0);
			expect(stderr).toBe("");
			const entries = await fs.readdir(root, { recursive: true });
			expect(entries.some(entry => entry.includes("http-400-requests"))).toBe(false);
		} finally {
			await fs.rm(root, { recursive: true, force: true });
		}
	});

	it("imports the public API without initializing native addons or user state", async () => {
		// Fresh subprocesses are required to observe module-initialization side effects.
		const importStderr = async (specifier: string, envOverrides: Record<string, string> = {}): Promise<string> => {
			const child = Bun.spawn([process.execPath, "-e", `await import(${JSON.stringify(specifier)})`], {
				cwd: path.resolve(import.meta.dir, ".."),
				env: { ...process.env, ...envOverrides, PI_DEBUG_STARTUP: "1" },
				stdout: "ignore",
				stderr: "pipe",
			});
			const [stderr, exitCode] = await Promise.all([new Response(child.stderr).text(), child.exited]);
			expect(exitCode).toBe(0);
			return stderr;
		};

		const root = await fs.mkdtemp(path.join(os.tmpdir(), "omp-probe-import-"));
		const home = path.join(root, "home");
		await fs.mkdir(home);
		try {
			const [probeStderr, nativeStderr] = await Promise.all([
				importStderr("@oh-my-pi/pi-coding-agent/openai-compat-probe", {
					HOME: home,
					PI_CODING_AGENT_DIR: path.join(home, "agent"),
					PI_CONFIG_DIR: ".omp-probe",
					XDG_CACHE_HOME: path.join(root, "runtime-cache"),
					XDG_DATA_HOME: path.join(home, "data"),
					XDG_STATE_HOME: path.join(home, "state"),
				}),
				importStderr("@oh-my-pi/pi-natives"),
			]);
			expect(nativeStderr).toContain("native:loadNative");
			expect(probeStderr).not.toContain("native:loadNative");
			expect(await fs.readdir(home)).toEqual([]);
		} finally {
			await fs.rm(root, { recursive: true, force: true });
		}
	});
});
