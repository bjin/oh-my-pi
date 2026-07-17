import { afterEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import type { Api, Model, ModelSpec } from "@oh-my-pi/pi-catalog/types";
import {
	formatOpenAICompatibilityReport,
	formatOpenAICompatibilityYaml,
	type OpenAICompatProbeInput,
	type OpenAICompatProbeReport,
} from "@oh-my-pi/pi-coding-agent/openai-compat-probe";
import {
	type OpenAICompatProbeCliContext,
	type OpenAICompatProbeCliRegistry,
	type OpenAICompatProbeCliWriter,
	runOpenAICompatProbeCli,
} from "@oh-my-pi/pi-coding-agent/openai-compat-probe-cli";
import type { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";

const tempRoots: string[] = [];

interface CaptureWriter extends OpenAICompatProbeCliWriter {
	text(): string;
}

function captureWriter(): CaptureWriter {
	let content = "";
	return {
		write(data) {
			content += data;
		},
		text() {
			return content;
		},
	};
}

function probeModel(overrides: Partial<ModelSpec<"openai-completions">> = {}): Model<"openai-completions"> {
	return buildModel({
		id: "model/path",
		name: "Probe Model",
		api: "openai-completions",
		provider: "probe",
		baseUrl: "https://probe.invalid/v1",
		reasoning: false,
		input: ["text"],
		supportsTools: true,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8_192,
		maxTokens: 1_024,
		compat: {},
		...overrides,
	});
}

function probeReport(model: Model<"openai-completions">, viable = true): OpenAICompatProbeReport {
	return {
		schemaVersion: 1,
		generatedAt: "2026-07-17T00:00:00.000Z",
		target: {
			provider: model.provider,
			model: model.id,
			api: "openai-completions",
			baseUrl: model.baseUrl,
		},
		configuredCompat: {},
		ignoredConfiguredKeys: [],
		resolvedBaseline: model.compat,
		recommendedCompat: viable ? { supportsStore: false } : {},
		resolvedRecommended: model.compat,
		removeConfiguredPaths: [],
		viable,
		fallbackFree: viable,
		findings: [],
		scenarios: [],
		warnings: ["fixture warning"],
	};
}

function fakeAuth(onClose?: () => void): AuthStorage {
	return { close: onClose ?? (() => {}) } as AuthStorage;
}

function registryFor(
	model: Model<Api>,
	options: {
		available?: boolean;
		apiKey?: string;
		initiallyMissing?: boolean;
		onRefresh?: (provider: string) => void;
	} = {},
): OpenAICompatProbeCliRegistry {
	let refreshed = !options.initiallyMissing;
	return {
		getAll: () => (refreshed ? [model] : []),
		getAvailable: () => (options.available === false ? [] : [model]),
		async refreshProvider(provider) {
			options.onRefresh?.(provider);
			refreshed = true;
		},
		async getApiKey() {
			return options.apiKey === undefined ? "fixture-key" : options.apiKey;
		},
	};
}

async function tempFile(name: string, content: string): Promise<string> {
	const root = await fs.mkdtemp(path.join(os.tmpdir(), "omp-openai-compat-cli-"));
	tempRoots.push(root);
	const filePath = path.join(root, name);
	await Bun.write(filePath, content);
	return filePath;
}

afterEach(async () => {
	await Promise.all(tempRoots.splice(0).map(root => fs.rm(root, { recursive: true, force: true })));
});

describe("omp-openai-compat CLI", () => {
	it("rejects malformed arguments and candidate files before auth discovery", async () => {
		const candidatePath = await tempFile("candidates.json", JSON.stringify({ schemaVersion: 2, profiles: [] }));
		const cases: readonly string[][] = [
			[],
			["--help", "probe/model"],
			["probe"],
			["probe/model", "--timeout-ms=1"],
			["probe/model", "--report", "one", "--report", "two"],
			["probe/model", "--report", "same", "--yaml", "same"],
			["probe/model", "extra"],
			["probe/model", "--candidates", candidatePath],
		];
		let discoveries = 0;
		for (const args of cases) {
			const error = captureWriter();
			const code = await runOpenAICompatProbeCli(args, {
				stderr: error,
				discoverAuthStorage: async () => {
					discoveries += 1;
					throw new Error("must not run");
				},
			});
			expect(code).toBe(2);
			expect(error.text()).toContain("Error:");
			expect(error.text()).toContain("--help");
		}
		expect(discoveries).toBe(0);
	});

	it("refreshes the exact target, loads candidates, reports progress, and writes machine outputs", async () => {
		const model = probeModel();
		const report = probeReport(model);
		const candidatePath = await tempFile(
			"candidates.json",
			JSON.stringify({
				schemaVersion: 1,
				profiles: [{ name: " candidate ", compat: { supportsNamedToolChoice: false } }],
			}),
		);
		const output = captureWriter();
		const error = captureWriter();
		const writes = new Map<string, string>();
		let closed = 0;
		let removedHandler = 0;
		let refreshedProvider: string | undefined;
		let probeInput: OpenAICompatProbeInput | undefined;
		const code = await runOpenAICompatProbeCli(
			[
				"--timeout-ms",
				"321",
				"probe/model/path",
				"--candidates",
				candidatePath,
				"--report",
				"report.json",
				"--yaml",
				"compat.yml",
			],
			{
				stdout: output,
				stderr: error,
				discoverAuthStorage: async () => fakeAuth(() => (closed += 1)),
				createModelRegistry: () =>
					registryFor(model, {
						initiallyMissing: true,
						onRefresh: provider => {
							refreshedProvider = provider;
						},
					}),
				probeOpenAICompatibility: async input => {
					probeInput = input;
					input.onProgress?.({
						type: "scenario_start",
						scenarioId: "connectivity@probe",
						phase: "probe",
						index: 1,
						total: 27,
					});
					return report;
				},
				writeFile: (filePath, content) => writes.set(filePath, content),
				registerInterruptHandler: () => () => {
					removedHandler += 1;
				},
			},
		);
		expect(code).toBe(0);
		expect(refreshedProvider).toBe("probe");
		expect(closed).toBe(1);
		expect(removedHandler).toBe(1);
		expect(probeInput?.timeoutMs).toBe(321);
		expect(probeInput?.candidates).toEqual([
			{ name: "candidate", source: "user", compat: { supportsNamedToolChoice: false } },
		]);
		expect(writes.get("report.json")).toBe(formatOpenAICompatibilityReport(report));
		expect(writes.get("compat.yml")).toBe(formatOpenAICompatibilityYaml(report));
		expect(output.text()).toBe("");
		expect(error.text()).toContain("Planned probe scenarios: 27");
		expect(error.text()).toContain("[probe 1/27] connectivity@probe");
		expect(error.text()).toContain("Warning: fixture warning");
	});

	it("rejects unavailable, incompatible, transport, tool, and credential targets", async () => {
		const base = probeModel();
		const cases: Array<{
			model: Model<Api>;
			registry?: { available?: boolean; apiKey?: string };
			message: string;
		}> = [
			{ model: base, registry: { available: false }, message: "unavailable" },
			{
				model: { ...base, api: "openai-responses" } as unknown as Model<Api>,
				message: "not openai-completions",
			},
			{
				model: { ...base, transport: "pi-native" } as Model<Api>,
				message: "unsupported pi-native transport",
			},
			{ model: { ...base, supportsTools: false } as Model<Api>, message: "explicitly disables tools" },
			{ model: base, registry: { apiKey: "" }, message: "No credential" },
		];
		for (const candidate of cases) {
			const error = captureWriter();
			let closed = 0;
			const code = await runOpenAICompatProbeCli(["probe/model/path"], {
				stderr: error,
				discoverAuthStorage: async () => fakeAuth(() => (closed += 1)),
				createModelRegistry: () => registryFor(candidate.model, candidate.registry),
			});
			expect(code).toBe(2);
			expect(error.text()).toContain(candidate.message);
			expect(closed).toBe(1);
		}
	});

	it("returns nonviable, probe failure, output failure, and interrupt exit codes", async () => {
		const model = probeModel();
		const nonviable = probeReport(model, false);
		const output = captureWriter();
		const nonviableError = captureWriter();
		const writes = new Map<string, string>();
		const baseContext: OpenAICompatProbeCliContext = {
			stdout: output,
			stderr: nonviableError,
			discoverAuthStorage: async () => fakeAuth(),
			createModelRegistry: () => registryFor(model),
			probeOpenAICompatibility: async () => nonviable,
			writeFile: (filePath, content) => writes.set(filePath, content),
			registerInterruptHandler: () => () => {},
		};
		expect(await runOpenAICompatProbeCli(["probe/model/path", "--yaml", "compat.yml"], baseContext)).toBe(1);
		expect(JSON.parse(output.text())).toMatchObject({ schemaVersion: 1, viable: false });
		expect(writes.has("compat.yml")).toBe(false);
		expect(nonviableError.text()).toContain("YAML fragment not written");

		const probeError = captureWriter();
		expect(
			await runOpenAICompatProbeCli(["probe/model/path"], {
				...baseContext,
				stdout: captureWriter(),
				stderr: probeError,
				probeOpenAICompatibility: async () => {
					throw new Error("provider exploded");
				},
			}),
		).toBe(1);
		expect(probeError.text()).toContain("Probe failed: provider exploded");

		const writeError = captureWriter();
		expect(
			await runOpenAICompatProbeCli(["probe/model/path", "--report", "report.json"], {
				...baseContext,
				stderr: writeError,
				probeOpenAICompatibility: async () => probeReport(model),
				writeFile: () => {
					throw new Error("disk full");
				},
			}),
		).toBe(2);
		expect(writeError.text()).toContain("Error writing JSON report: disk full");

		let interrupt: (() => void) | undefined;
		const interruptError = captureWriter();
		expect(
			await runOpenAICompatProbeCli(["probe/model/path"], {
				...baseContext,
				stderr: interruptError,
				registerInterruptHandler: handler => {
					interrupt = handler;
					return () => {
						interrupt = undefined;
					};
				},
				probeOpenAICompatibility: async input => {
					interrupt?.();
					throw input.signal?.reason ?? new Error("not interrupted");
				},
			}),
		).toBe(130);
		expect(interrupt).toBeUndefined();
		expect(interruptError.text()).toContain("Interrupted.");
	});

	it("keeps help and version as clean one-line-or-document stdout modes", async () => {
		const help = captureWriter();
		const helpError = captureWriter();
		expect(await runOpenAICompatProbeCli(["--help"], { stdout: help, stderr: helpError })).toBe(0);
		expect(help.text()).toStartWith("Usage: omp-openai-compat");
		expect(helpError.text()).toBe("");

		const version = captureWriter();
		const versionError = captureWriter();
		expect(await runOpenAICompatProbeCli(["--version"], { stdout: version, stderr: versionError })).toBe(0);
		expect(version.text()).toMatch(/^\d+\.\d+\.\d+\n$/u);
		expect(versionError.text()).toBe("");
	});

	it("cleans up auth and signal handlers when progress output fails", async () => {
		const model = probeModel();
		let writes = 0;
		let closed = 0;
		let removed = 0;
		let probes = 0;
		const stderr: OpenAICompatProbeCliWriter = {
			write() {
				writes += 1;
				if (writes === 1) throw new Error("broken progress pipe");
			},
		};
		const code = await runOpenAICompatProbeCli(["probe/model/path"], {
			stderr,
			discoverAuthStorage: async () => fakeAuth(() => (closed += 1)),
			createModelRegistry: () => registryFor(model),
			probeOpenAICompatibility: async () => {
				probes += 1;
				return probeReport(model);
			},
			registerInterruptHandler: () => () => {
				removed += 1;
			},
		});

		expect(code).toBe(1);
		expect(probes).toBe(0);
		expect(closed).toBe(1);
		expect(removed).toBe(1);
	});

	it("waits for asynchronous output writers before returning", async () => {
		const gate = Promise.withResolvers<void>();
		let writeStarted = false;
		let settled = false;
		const writer: OpenAICompatProbeCliWriter = {
			async write() {
				writeStarted = true;
				await gate.promise;
			},
		};
		const run = runOpenAICompatProbeCli(["--version"], { stdout: writer });
		void run.then(() => {
			settled = true;
		});
		await Promise.resolve();
		expect(writeStarted).toBe(true);
		expect(settled).toBe(false);
		gate.resolve();
		expect(await run).toBe(0);
		expect(settled).toBe(true);
	});
});
