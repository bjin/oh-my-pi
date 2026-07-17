#!/usr/bin/env bun
import * as path from "node:path";
import type { Api, Model, OpenAICompat } from "@oh-my-pi/pi-catalog/types";
import { VERSION } from "@oh-my-pi/pi-utils";
import { ModelRegistry } from "./config/model-registry";
import type {
	OpenAICompatCandidateFileV1,
	OpenAICompatProbeCandidate,
	OpenAICompatProbeInput,
	OpenAICompatProbeReport,
} from "./openai-compat-probe";
import {
	formatOpenAICompatibilityReport,
	formatOpenAICompatibilityYaml,
	OPENAI_COMPAT_PROBE_SCENARIO_IDS,
	probeOpenAICompatibility,
	validateStrictOpenAICompat,
} from "./openai-compat-probe";
import { discoverAuthStorage } from "./sdk";
import type { AuthStorage } from "./session/auth-storage";

const MAX_CANDIDATE_FILE_BYTES = 262_144;
const MAX_TIMEOUT_MS = 3_600_000;
const HELP = `Usage: omp-openai-compat <provider/model-id> [options]

Probe an OpenAI Chat Completions endpoint and emit compatibility evidence.

Options:
  --report <file>       Write the JSON evidence report to a file
  --yaml <file>         Write the validated models.yml compat fragment
  --candidates <file>   Load candidate profiles from a versioned JSON file
  --timeout-ms <ms>     Per-attempt watchdog (1-${MAX_TIMEOUT_MS})
  --help                Show this help (must be the sole argument)
  --version             Show the version (must be the sole argument)
`;

export interface OpenAICompatProbeCliWriter {
	write(data: string): unknown | Promise<unknown>;
}

export interface OpenAICompatProbeCliRegistry {
	getAll(): Model<Api>[];
	getAvailable(): Model<Api>[];
	refreshProvider(provider: string): Promise<void>;
	getApiKey(model: Model<Api>): Promise<string | undefined>;
}

export interface OpenAICompatProbeCliContext {
	stdout?: OpenAICompatProbeCliWriter;
	stderr?: OpenAICompatProbeCliWriter;
	discoverAuthStorage?: () => Promise<AuthStorage>;
	createModelRegistry?: (authStorage: AuthStorage) => OpenAICompatProbeCliRegistry;
	probeOpenAICompatibility?: (input: OpenAICompatProbeInput) => Promise<OpenAICompatProbeReport>;
	writeFile?: (filePath: string, content: string) => Promise<unknown> | unknown;
	registerInterruptHandler?: (handler: () => void) => () => void;
}

interface ParsedCliArgs {
	selector: string;
	provider: string;
	modelId: string;
	reportPath?: string;
	yamlPath?: string;
	candidatesPath?: string;
	timeoutMs?: number;
}

class CliInputError extends Error {}

async function stdout(context: OpenAICompatProbeCliContext | undefined, text: string): Promise<void> {
	await (context?.stdout ?? Bun.stdout).write(text);
}

async function stderr(context: OpenAICompatProbeCliContext | undefined, text: string): Promise<void> {
	await (context?.stderr ?? Bun.stderr).write(text);
}

function boundedError(error: unknown): string {
	const value = error instanceof Error ? error.message : String(error);
	const flattened = value.replace(/[\r\n\t]+/gu, " ").trim();
	return flattened.length <= 4_096 ? flattened : `${flattened.slice(0, 4_096)}…`;
}

function parseArguments(args: readonly string[]): ParsedCliArgs {
	if (args.length === 0) throw new CliInputError("Missing required selector <provider/model-id>");
	if (args.includes("--help") || args.includes("--version")) {
		throw new CliInputError("--help and --version must be the sole argument");
	}
	const values: Partial<Record<"reportPath" | "yamlPath" | "candidatesPath", string>> = {};
	let timeoutMs: number | undefined;
	let selector: string | undefined;
	const seenFlags = new Set<string>();
	for (let index = 0; index < args.length; index += 1) {
		const arg = args[index];
		if (arg === "--") throw new CliInputError("The -- separator is not supported");
		if (arg.startsWith("--") && arg.includes("=")) {
			throw new CliInputError(`Use a separate value token for ${arg.slice(0, arg.indexOf("="))}`);
		}
		if (arg.startsWith("--")) {
			if (arg !== "--report" && arg !== "--yaml" && arg !== "--candidates" && arg !== "--timeout-ms") {
				throw new CliInputError(`Unknown option: ${arg}`);
			}
			if (seenFlags.has(arg)) throw new CliInputError(`Duplicate option: ${arg}`);
			seenFlags.add(arg);
			const value = args[index + 1];
			if (value === undefined || value.startsWith("--")) throw new CliInputError(`Missing value for ${arg}`);
			index += 1;
			switch (arg) {
				case "--report":
					values.reportPath = value;
					break;
				case "--yaml":
					values.yamlPath = value;
					break;
				case "--candidates":
					values.candidatesPath = value;
					break;
				case "--timeout-ms": {
					if (!/^\d+$/u.test(value)) throw new CliInputError("--timeout-ms must be a positive integer");
					const parsed = Number(value);
					if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > MAX_TIMEOUT_MS) {
						throw new CliInputError(`--timeout-ms must be between 1 and ${MAX_TIMEOUT_MS}`);
					}
					timeoutMs = parsed;
					break;
				}
			}
			continue;
		}
		if (selector !== undefined) throw new CliInputError(`Unexpected extra positional argument: ${arg}`);
		selector = arg;
	}
	if (!selector) throw new CliInputError("Missing required selector <provider/model-id>");
	const slash = selector.indexOf("/");
	if (slash <= 0 || slash === selector.length - 1) {
		throw new CliInputError("Selector must contain a nonempty provider and model id: <provider/model-id>");
	}
	if (values.reportPath && values.yamlPath && path.resolve(values.reportPath) === path.resolve(values.yamlPath)) {
		throw new CliInputError("--report and --yaml must resolve to different files");
	}
	return {
		selector,
		provider: selector.slice(0, slash),
		modelId: selector.slice(slash + 1),
		...values,
		...(timeoutMs !== undefined ? { timeoutMs } : {}),
	};
}

function validateCandidateDocument(value: unknown): OpenAICompatCandidateFileV1 {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new CliInputError("Candidate file must contain an object");
	}
	const document = value as Record<string, unknown>;
	if (document.schemaVersion !== 1) throw new CliInputError("Candidate file schemaVersion must be 1");
	if (!Array.isArray(document.profiles) || document.profiles.length < 1 || document.profiles.length > 16) {
		throw new CliInputError("Candidate file must contain 1-16 profiles");
	}
	const names = new Set<string>();
	const profiles: OpenAICompatCandidateFileV1["profiles"] = [];
	for (const rawProfile of document.profiles) {
		if (typeof rawProfile !== "object" || rawProfile === null || Array.isArray(rawProfile)) {
			throw new CliInputError("Each candidate profile must be an object");
		}
		const profile = rawProfile as Record<string, unknown>;
		const name = typeof profile.name === "string" ? profile.name.trim() : "";
		if (name.length < 1 || name.length > 64) {
			throw new CliInputError("Candidate profile names must contain 1-64 characters after trimming");
		}
		if (names.has(name)) throw new CliInputError(`Duplicate candidate profile name: ${name}`);
		names.add(name);
		let compat: OpenAICompat;
		try {
			compat = validateStrictOpenAICompat(profile.compat);
		} catch (error) {
			throw new CliInputError(`Invalid candidate profile ${name}: ${boundedError(error)}`);
		}
		profiles.push({ name, compat });
	}
	return { schemaVersion: 1, profiles };
}

async function loadCandidateProfiles(filePath: string | undefined): Promise<OpenAICompatProbeCandidate[]> {
	if (!filePath) return [];
	const file = Bun.file(filePath);
	try {
		if (file.size > MAX_CANDIDATE_FILE_BYTES) {
			throw new CliInputError(`Candidate file exceeds ${MAX_CANDIDATE_FILE_BYTES} bytes`);
		}
		const text = await file.text();
		if (new TextEncoder().encode(text).byteLength > MAX_CANDIDATE_FILE_BYTES) {
			throw new CliInputError(`Candidate file exceeds ${MAX_CANDIDATE_FILE_BYTES} bytes`);
		}
		let parsed: unknown;
		try {
			parsed = JSON.parse(text) as unknown;
		} catch (error) {
			throw new CliInputError(`Candidate file is not valid JSON: ${boundedError(error)}`);
		}
		const document = validateCandidateDocument(parsed);
		return document.profiles.map(profile => ({ ...profile, source: "user" }));
	} catch (error) {
		if (error instanceof CliInputError) throw error;
		throw new CliInputError(`Cannot read candidate file ${filePath}: ${boundedError(error)}`);
	}
}

async function writeOutput(
	context: OpenAICompatProbeCliContext | undefined,
	filePath: string,
	content: string,
): Promise<void> {
	if (context?.writeFile) {
		await context.writeFile(filePath, content);
		return;
	}
	await Bun.write(filePath, content);
}

export async function runOpenAICompatProbeCli(
	args: readonly string[] = Bun.argv.slice(2),
	context?: OpenAICompatProbeCliContext,
): Promise<number> {
	if (args.length === 1 && args[0] === "--help") {
		await stdout(context, HELP);
		return 0;
	}
	if (args.length === 1 && args[0] === "--version") {
		await stdout(context, `${VERSION}\n`);
		return 0;
	}

	let parsed: ParsedCliArgs;
	let candidates: OpenAICompatProbeCandidate[];
	try {
		parsed = parseArguments(args);
		candidates = await loadCandidateProfiles(parsed.candidatesPath);
	} catch (error) {
		await stderr(context, `Error: ${boundedError(error)}\n`);
		await stderr(context, "Run 'omp-openai-compat --help' for usage.\n");
		return 2;
	}

	const discover = context?.discoverAuthStorage ?? discoverAuthStorage;
	let authStorage: AuthStorage | undefined;
	let registry: OpenAICompatProbeCliRegistry;
	let target: Model<Api> | undefined;
	let apiKey: string | undefined;
	try {
		authStorage = await discover();
		registry = context?.createModelRegistry?.(authStorage) ?? new ModelRegistry(authStorage);
		target = registry.getAll().find(model => `${model.provider}/${model.id}` === parsed.selector);
		if (!target) {
			try {
				await registry.refreshProvider(parsed.provider);
			} catch (error) {
				throw new CliInputError(`Provider refresh failed for ${parsed.provider}: ${boundedError(error)}`);
			}
			target = registry.getAll().find(model => `${model.provider}/${model.id}` === parsed.selector);
		}
		if (!target) throw new CliInputError(`Model not found: ${parsed.selector}`);
		if (!registry.getAvailable().includes(target))
			throw new CliInputError(`Model is unavailable: ${parsed.selector}`);
		if (target.api !== "openai-completions") {
			throw new CliInputError(`Model ${parsed.selector} uses ${target.api}, not openai-completions`);
		}
		if (target.transport === "pi-native") {
			throw new CliInputError(`Model ${parsed.selector} uses unsupported pi-native transport`);
		}
		if (target.supportsTools === false) throw new CliInputError(`Model ${parsed.selector} explicitly disables tools`);
		apiKey = await registry.getApiKey(target);
		if (!apiKey) throw new CliInputError(`No credential is available for ${parsed.selector}`);
	} catch (error) {
		authStorage?.close();
		await stderr(context, `Error: ${boundedError(error)}\n`);
		return 2;
	}

	const controller = new AbortController();
	let interrupted = false;
	const onInterrupt = (): void => {
		interrupted = true;
		controller.abort(new DOMException("Interrupted", "AbortError"));
	};
	const removeInterruptHandler =
		context?.registerInterruptHandler?.(onInterrupt) ??
		(() => {
			process.on("SIGINT", onInterrupt);
			return () => process.off("SIGINT", onInterrupt);
		})();
	let report: OpenAICompatProbeReport;
	try {
		await stderr(
			context,
			`Planned probe scenarios: ${OPENAI_COMPAT_PROBE_SCENARIO_IDS.length}; minimization reruns are added after probing.\n`,
		);
		const probe = context?.probeOpenAICompatibility ?? probeOpenAICompatibility;
		report = await probe({
			model: target as Model<"openai-completions">,
			apiKey,
			...(candidates.length > 0 ? { candidates } : {}),
			...(parsed.timeoutMs !== undefined ? { timeoutMs: parsed.timeoutMs } : {}),
			signal: controller.signal,
			onProgress: async event => {
				if (event.type === "scenario_start") {
					await stderr(context, `[${event.phase} ${event.index}/${event.total}] ${event.scenarioId}\n`);
				}
			},
		});
	} catch (error) {
		if (interrupted) {
			await stderr(context, "Interrupted.\n");
			return 130;
		}
		await stderr(context, `Probe failed: ${boundedError(error)}\n`);
		return 1;
	} finally {
		removeInterruptHandler();
		authStorage.close();
	}
	if (interrupted) {
		await stderr(context, "Interrupted.\n");
		return 130;
	}

	const json = formatOpenAICompatibilityReport(report);
	try {
		if (parsed.reportPath) {
			await writeOutput(context, parsed.reportPath, json);
			await stderr(context, `JSON report: ${parsed.reportPath}\n`);
		} else {
			await stdout(context, json);
		}
	} catch (error) {
		await stderr(context, `Error writing JSON report: ${boundedError(error)}\n`);
		return 2;
	}

	for (const warning of report.warnings) await stderr(context, `Warning: ${warning}\n`);
	if (parsed.yamlPath && report.viable) {
		try {
			const yaml = formatOpenAICompatibilityYaml(report);
			await writeOutput(context, parsed.yamlPath, yaml);
			await stderr(context, `YAML fragment: ${parsed.yamlPath}\n`);
		} catch (error) {
			await stderr(context, `Error writing YAML fragment: ${boundedError(error)}\n`);
			return 2;
		}
	} else if (parsed.yamlPath) {
		await stderr(context, "YAML fragment not written because no fallback-free viable profile was found.\n");
	}
	return report.viable ? 0 : 1;
}

if (import.meta.main) {
	const code = await runOpenAICompatProbeCli();
	process.exit(code);
}
