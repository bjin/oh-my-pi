import type { OpenAICompat } from "@oh-my-pi/pi-catalog/types";
import { OPENAI_COMPAT_CAPABILITY_REGISTRY, OPENAI_COMPAT_KEYS, validateStrictOpenAICompat } from "./capabilities";
import { compareProbeObjectKeys } from "./ordering";
import type { OpenAICompatProbeReport } from "./types";
import { sanitizeProbeUrl } from "./wire-observer";

function canonicalize(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(canonicalize);
	if (typeof value !== "object" || value === null) return value;
	const source = value as Record<string, unknown>;
	const result: Record<string, unknown> = {};
	for (const key of Object.keys(source).sort(compareProbeObjectKeys)) {
		if (source[key] !== undefined) result[key] = canonicalize(source[key]);
	}
	return result;
}

function redactNestedValues(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(redactNestedValues);
	if (typeof value !== "object" || value === null) return "[REDACTED]";
	const source = value as Record<string, unknown>;
	const result: Record<string, unknown> = {};
	for (const key of Object.keys(source).sort(compareProbeObjectKeys)) {
		if (source[key] !== undefined) result[key] = redactNestedValues(source[key]);
	}
	return result;
}

function canonicalizeCompat(value: unknown, redactNested = false): Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return {};
	const source = value as Record<string, unknown>;
	const result: Record<string, unknown> = {};
	for (const key of OPENAI_COMPAT_KEYS) {
		if (source[key] === undefined) continue;
		if (key === "whenThinking") {
			result[key] = canonicalizeCompat(source[key], redactNested);
		} else if (redactNested && OPENAI_COMPAT_CAPABILITY_REGISTRY[key].redaction === "nested_values") {
			result[key] = redactNestedValues(source[key]);
		} else {
			result[key] = canonicalize(source[key]);
		}
	}
	for (const key of Object.keys(source).sort(compareProbeObjectKeys)) {
		if (result[key] !== undefined || source[key] === undefined) continue;
		result[key] = canonicalize(source[key]);
	}
	return result;
}

function canonicalReport(report: OpenAICompatProbeReport): Record<string, unknown> {
	return {
		schemaVersion: report.schemaVersion,
		generatedAt: report.generatedAt,
		target: {
			provider: report.target.provider,
			model: report.target.model,
			api: report.target.api,
			baseUrl: sanitizeProbeUrl(report.target.baseUrl),
		},
		configuredCompat: canonicalizeCompat(report.configuredCompat, true),
		ignoredConfiguredKeys: [...report.ignoredConfiguredKeys].sort(),
		resolvedBaseline: canonicalizeCompat(report.resolvedBaseline, true),
		recommendedCompat: canonicalizeCompat(report.recommendedCompat, true),
		resolvedRecommended: canonicalizeCompat(report.resolvedRecommended, true),
		removeConfiguredPaths: [...report.removeConfiguredPaths].sort(),
		viable: report.viable,
		fallbackFree: report.fallbackFree,
		findings: report.findings.map(finding => ({
			key: finding.key,
			verdict: finding.verdict,
			...(finding.recommendedValue !== undefined
				? {
						recommendedValue:
							OPENAI_COMPAT_CAPABILITY_REGISTRY[finding.key].redaction === "nested_values"
								? redactNestedValues(finding.recommendedValue)
								: canonicalize(finding.recommendedValue),
					}
				: {}),
			scenarioIds: [...finding.scenarioIds],
			persistable: finding.persistable,
			...(finding.note !== undefined ? { note: finding.note } : {}),
		})),
		scenarios: report.scenarios.map(scenario => ({
			id: scenario.id,
			baseId: scenario.baseId,
			phase: scenario.phase,
			verdict: scenario.verdict,
			attempts: scenario.attempts.map(attempt => ({
				ordinal: attempt.ordinal,
				request: {
					method: attempt.request.method,
					url: sanitizeProbeUrl(attempt.request.url),
					headers: canonicalize(attempt.request.headers),
					...(attempt.request.body !== undefined ? { body: canonicalize(attempt.request.body) } : {}),
				},
				...(attempt.response
					? {
							response: {
								status: attempt.response.status,
								headers: canonicalize(attempt.response.headers),
								...(attempt.response.errorBody !== undefined ? { errorBody: attempt.response.errorBody } : {}),
							},
						}
					: {}),
				latencyMs: attempt.latencyMs,
				classification: attempt.classification,
			})),
			final: canonicalize(scenario.final),
			evidence: canonicalize(scenario.evidence),
		})),
		warnings: [...report.warnings],
	};
}

/** Pretty, deterministic JSON with one trailing newline. */
export function formatOpenAICompatibilityReport(report: OpenAICompatProbeReport): string {
	return `${JSON.stringify(canonicalReport(report), null, 2)}\n`;
}

function safeCommentValue(value: string): string {
	return value
		.replace(/[\u0000-\u0008\u000A-\u001F\u007F-\u009F\u2028\u2029\uD800-\uDFFF\uFFFE\uFFFF]+/gu, " ")
		.trim();
}

/** Render a complete replacement `compat:` block for a viable report. */
export function formatOpenAICompatibilityYaml(report: OpenAICompatProbeReport): string {
	if (!report.viable) throw new Error("Cannot format compatibility YAML for a non-viable report");
	const validated = validateStrictOpenAICompat(report.recommendedCompat);
	const ordered = canonicalizeCompat(validated) as OpenAICompat;
	const target = `${safeCommentValue(report.target.provider)}/${safeCommentValue(report.target.model)}`;
	const comments = [
		`# OpenAI compatibility probe for ${target}`,
		"# Replace the target model's own compat block with this complete block.",
		"# Provider and modelOverride compat values still merge into the model.",
	];
	if (report.removeConfiguredPaths.length > 0) {
		comments.push(
			`# Remove these paths at their originating config blocks: ${[...report.removeConfiguredPaths].sort().map(safeCommentValue).join(", ")}`,
		);
	} else if (Object.keys(validated).length === 0) {
		comments.push("# OMP's auto-detected compatibility profile passed without overrides.");
	}
	return `${comments.join("\n")}\n${Bun.YAML.stringify({ compat: ordered })}`;
}
