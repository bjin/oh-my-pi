import type { AssistantMessage, Context, Message, Model, Tool, ToolResultMessage, Usage } from "@oh-my-pi/pi-ai/types";
import baselineSystemPrompt from "./prompts/baseline-system.md" with { type: "text" };
import baselineUserPrompt from "./prompts/baseline-user.md" with { type: "text" };
import historyAssistantPrompt from "./prompts/history-assistant.md" with { type: "text" };
import historyUserPrompt from "./prompts/history-user.md" with { type: "text" };
import systemFirstPrompt from "./prompts/system-first.md" with { type: "text" };
import systemSecondPrompt from "./prompts/system-second.md" with { type: "text" };
import toolDescriptionPrompt from "./prompts/tool-description.md" with { type: "text" };
import toolFollowupPrompt from "./prompts/tool-followup.md" with { type: "text" };
import toolResultPrompt from "./prompts/tool-result.md" with { type: "text" };
import toolUserPrompt from "./prompts/tool-user.md" with { type: "text" };

export const BASELINE_SYSTEM_PROMPT = baselineSystemPrompt.trim();
export const BASELINE_USER_PROMPT = baselineUserPrompt.trim();
export const FIRST_SYSTEM_PROMPT = systemFirstPrompt.trim();
export const SECOND_SYSTEM_PROMPT = systemSecondPrompt.trim();
export const TOOL_USER_PROMPT = toolUserPrompt.trim();
export const TOOL_FOLLOWUP_PROMPT = toolFollowupPrompt.trim();
export const TOOL_RESULT_PROMPT = toolResultPrompt.trim();

const HISTORY_USER_PROMPT = historyUserPrompt.trim();
const HISTORY_ASSISTANT_PROMPT = historyAssistantPrompt.trim();
const TOOL_DESCRIPTION = toolDescriptionPrompt.trim();

const EMPTY_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const COMPAT_ECHO_PARAMETERS = {
	type: "object",
	properties: {
		value: { const: "PROBE_OK", type: "string" },
	},
	required: ["value"],
	additionalProperties: false,
} as const;

export const COMPAT_ECHO_TOOL: Tool = {
	name: "compat_echo",
	description: TOOL_DESCRIPTION,
	parameters: COMPAT_ECHO_PARAMETERS,
	strict: true,
};

export const COMPAT_ECHO_NON_STRICT_TOOL: Tool = {
	...COMPAT_ECHO_TOOL,
	name: "compat_echo_relaxed",
	strict: false,
};

export function createProbeUserMessage(content: string): Message {
	return { role: "user", content, timestamp: 0 };
}

export function createBaselineContext(systemPrompt: readonly string[] = [BASELINE_SYSTEM_PROMPT]): Context {
	return {
		systemPrompt: [...systemPrompt],
		messages: [createProbeUserMessage(BASELINE_USER_PROMPT)],
	};
}

export function createToolContext(tools: readonly Tool[] = [COMPAT_ECHO_TOOL]): Context {
	return {
		systemPrompt: [BASELINE_SYSTEM_PROMPT],
		messages: [createProbeUserMessage(TOOL_USER_PROMPT)],
		tools: [...tools],
	};
}

export function createAssistantMessage(
	model: Model<"openai-completions">,
	content: AssistantMessage["content"],
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: { ...EMPTY_USAGE, cost: { ...EMPTY_USAGE.cost } },
		stopReason: content.some(block => block.type === "toolCall") ? "toolUse" : "stop",
		timestamp: 0,
	};
}

export interface ProbeHistoryOptions {
	toolCallId?: string;
	includeReasoning?: boolean;
	includePlainReasoning?: boolean;
	toolResultName?: string;
	assistantAfterToolResult?: boolean;
	assistantToolContent?: boolean;
}

export function createHistoryContext(model: Model<"openai-completions">, options: ProbeHistoryOptions = {}): Context {
	const toolCallId = options.toolCallId ?? "call_probe_raw_identifier_123456789012345678901234567890";
	const plainContent: AssistantMessage["content"] = [];
	if (options.includePlainReasoning) plainContent.push({ type: "thinking", thinking: "." });
	plainContent.push({ type: "text", text: HISTORY_ASSISTANT_PROMPT });

	const toolContent: AssistantMessage["content"] = [];
	if (options.includeReasoning) toolContent.push({ type: "thinking", thinking: "." });
	if (options.assistantToolContent) toolContent.push({ type: "text", text: "." });
	toolContent.push({ type: "toolCall", id: toolCallId, name: "compat_echo", arguments: { value: "PROBE_OK" } });

	const toolResult: ToolResultMessage = {
		role: "toolResult",
		toolCallId,
		toolName: options.toolResultName ?? "compat_echo",
		content: [{ type: "text", text: TOOL_RESULT_PROMPT }],
		isError: false,
		timestamp: 0,
	};
	const messages: Message[] = [
		createProbeUserMessage(HISTORY_USER_PROMPT),
		createAssistantMessage(model, plainContent),
		createAssistantMessage(model, toolContent),
		toolResult,
	];
	if (options.assistantAfterToolResult) {
		messages.push(createAssistantMessage(model, [{ type: "text", text: TOOL_RESULT_PROMPT }]));
	}
	messages.push(createProbeUserMessage(TOOL_FOLLOWUP_PROMPT));
	return { systemPrompt: [BASELINE_SYSTEM_PROMPT], messages, tools: [COMPAT_ECHO_TOOL] };
}
