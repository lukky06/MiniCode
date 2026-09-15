export type CommandCatalogItem = {
  name: string;
  description: string;
  argument_hint?: string | null;
  argument_choices: string[];
  availability: "idle" | "active" | "both";
};

export type ServerMessage =
  | { type: "session_started"; session_id: string }
  | { type: "command_catalog"; commands: CommandCatalogItem[] }
  | {
      type: "session_settings";
      permission_mode: "read-only" | "workspace-write" | "full-access";
      approval_policy: "on-request" | "never";
      collaboration_mode: "default" | "plan";
    }
  | { type: "run_started"; run_id: string }
  | {
      type: "context";
      used: number;
      window: number;
      prompt_budget: number;
      reserved_output: number;
    }
  | {
      type: "tool_started";
      id: string;
      step: number;
      tool: string;
      target?: string | null;
    }
  | {
      type: "tool_finished";
      id: string;
      step: number;
      tool: string;
      status: string;
      summary?: string | null;
      command_status?: string | null;
      returncode?: number | null;
      duration_ms?: number | null;
      runtime_task_id?: string | null;
      diff_preview?: string | null;
      diff_truncated?: boolean;
    }
  | { type: "assistant_delta"; text: string }
  | { type: "reasoning_delta"; text: string }
  | {
      type: "approval_required";
      id: string;
      tool_call_id: string;
      tool: string;
      summary?: string | null;
      details?: string | null;
      can_approve_session?: boolean;
    }
  | {
      type: "user_input_required";
      id: string;
      question: string;
      options: Array<{ label: string; description?: string | null }>;
    }
  | {
      type: "run_finished";
      status: string;
      run_id?: string | null;
      stop_reason?: string | null;
    }
  | { type: "error"; message: string; fatal?: boolean }
  | {
      type: "panel";
      name: string;
      title: string;
      content: string;
    }
  | { type: "exit_requested" };

export type ClientMessage =
  | { type: "task"; text: string }
  | { type: "steer"; text: string }
  | { type: "command"; text: string }
  | {
      type: "approval_response";
      id: string;
      decision: "approve" | "approve_session" | "reject" | "skip" | "abort";
    }
  | {
      type: "user_input_response";
      id: string;
      selected_index: number;
    }
  | { type: "cancel" };

const SERVER_TYPES = new Set<ServerMessage["type"]>([
  "session_started",
  "command_catalog",
  "session_settings",
  "run_started",
  "context",
  "tool_started",
  "tool_finished",
  "assistant_delta",
  "reasoning_delta",
  "approval_required",
  "user_input_required",
  "run_finished",
  "error",
  "panel",
  "exit_requested",
]);

export function parseServerMessage(line: string): ServerMessage {
  if (line.includes("\n") || line.includes("\r")) {
    throw new Error("JSONL record must occupy one physical line");
  }
  const value: unknown = JSON.parse(line);
  if (!isRecord(value) || typeof value.type !== "string" || !SERVER_TYPES.has(value.type as ServerMessage["type"])) {
    throw new Error("Unknown server message");
  }
  validateRequiredFields(value);
  return value as ServerMessage;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateRequiredFields(value: Record<string, unknown>): void {
  switch (value.type) {
    case "session_started":
      exactKeys(value, ["type", "session_id"]);
      requireString(value, "session_id");
      return;
    case "command_catalog":
      exactKeys(value, ["type", "commands"]);
      if (!Array.isArray(value.commands)) {
        throw new Error("Expected command catalog array");
      }
      for (const command of value.commands) {
        if (!isRecord(command)) {
          throw new Error("Expected command catalog item object");
        }
        exactKeys(command, [
          "name",
          "description",
          "argument_hint",
          "argument_choices",
          "availability",
        ]);
        requireString(command, "name");
        requireString(command, "description");
        requireOptionalString(command, "argument_hint");
        if (
          !Array.isArray(command.argument_choices) ||
          command.argument_choices.some((item) => typeof item !== "string")
        ) {
          throw new Error("Expected command argument choice strings");
        }
        if (!["idle", "active", "both"].includes(String(command.availability))) {
          throw new Error("Expected command availability");
        }
      }
      return;
    case "session_settings":
      exactKeys(value, [
        "type",
        "permission_mode",
        "approval_policy",
        "collaboration_mode",
      ]);
      if (!["read-only", "workspace-write", "full-access"].includes(String(value.permission_mode))) {
        throw new Error("Expected permission mode");
      }
      if (!["on-request", "never"].includes(String(value.approval_policy))) {
        throw new Error("Expected approval policy");
      }
      if (!["default", "plan"].includes(String(value.collaboration_mode))) {
        throw new Error("Expected collaboration mode");
      }
      return;
    case "run_started":
      exactKeys(value, ["type", "run_id"]);
      requireString(value, "run_id");
      return;
    case "context":
      exactKeys(value, ["type", "used", "window", "prompt_budget", "reserved_output"]);
      requireNonNegative(value, "used");
      requirePositive(value, "window");
      requirePositive(value, "prompt_budget");
      requireNonNegative(value, "reserved_output");
      return;
    case "tool_started":
      exactKeys(value, ["type", "id", "step", "tool", "target"]);
      requireString(value, "id");
      requireNonNegative(value, "step");
      requireString(value, "tool");
      requireOptionalString(value, "target");
      return;
    case "tool_finished":
      exactKeys(value, [
        "type",
        "id",
        "step",
        "tool",
        "status",
        "summary",
        "command_status",
        "returncode",
        "duration_ms",
        "runtime_task_id",
        "diff_preview",
        "diff_truncated",
      ]);
      requireString(value, "id");
      requireNonNegative(value, "step");
      requireString(value, "tool");
      requireString(value, "status");
      requireOptionalString(value, "summary");
      requireOptionalString(value, "command_status");
      requireOptionalInteger(value, "returncode");
      requireOptionalNonNegativeInteger(value, "duration_ms");
      requireOptionalString(value, "runtime_task_id");
      requireOptionalString(value, "diff_preview");
      if (
        value.diff_truncated !== undefined &&
        typeof value.diff_truncated !== "boolean"
      ) {
        throw new Error("Expected boolean field: diff_truncated");
      }
      return;
    case "assistant_delta":
    case "reasoning_delta":
      exactKeys(value, ["type", "text"]);
      requireString(value, "text");
      return;
    case "approval_required":
      exactKeys(value, [
        "type",
        "id",
        "tool_call_id",
        "tool",
        "summary",
        "details",
        "can_approve_session",
      ]);
      requireString(value, "id");
      requireString(value, "tool_call_id");
      requireString(value, "tool");
      requireOptionalString(value, "summary");
      requireOptionalString(value, "details");
      if (
        value.can_approve_session !== undefined &&
        typeof value.can_approve_session !== "boolean"
      ) {
        throw new Error("Expected boolean field: can_approve_session");
      }
      return;
    case "user_input_required":
      exactKeys(value, ["type", "id", "question", "options"]);
      requireString(value, "id");
      requireString(value, "question");
      if (!Array.isArray(value.options) || value.options.length < 2 || value.options.length > 4) {
        throw new Error("Expected 2-4 user input options");
      }
      for (const option of value.options) {
        if (!isRecord(option)) {
          throw new Error("Expected user input option object");
        }
        exactKeys(option, ["label", "description"]);
        requireString(option, "label");
        requireOptionalString(option, "description");
      }
      return;
    case "run_finished":
      exactKeys(value, ["type", "status", "run_id", "stop_reason"]);
      requireString(value, "status");
      requireOptionalString(value, "run_id");
      requireOptionalString(value, "stop_reason");
      return;
    case "error":
      exactKeys(value, ["type", "message", "fatal"]);
      requireString(value, "message");
      if (value.fatal !== undefined && typeof value.fatal !== "boolean") {
        throw new Error("Expected boolean field: fatal");
      }
      return;
    case "panel":
      exactKeys(value, ["type", "name", "title", "content"]);
      requireString(value, "name");
      requireString(value, "title");
      requireString(value, "content");
      return;
    case "exit_requested":
      exactKeys(value, ["type"]);
      return;
    default:
      throw new Error("Unknown server message");
  }
}

function exactKeys(value: Record<string, unknown>, allowed: string[]): void {
  const allowedKeys = new Set(allowed);
  const extra = Object.keys(value).find((key) => !allowedKeys.has(key));
  if (extra !== undefined) {
    throw new Error(`Unexpected field: ${extra}`);
  }
}

function requireString(value: Record<string, unknown>, key: string): void {
  if (typeof value[key] !== "string") {
    throw new Error(`Expected string field: ${key}`);
  }
}

function requireOptionalString(value: Record<string, unknown>, key: string): void {
  const field = value[key];
  if (field !== undefined && field !== null && typeof field !== "string") {
    throw new Error(`Expected optional string field: ${key}`);
  }
}

function requireOptionalInteger(value: Record<string, unknown>, key: string): void {
  const field = value[key];
  if (
    field !== undefined &&
    field !== null &&
    (typeof field !== "number" || !Number.isInteger(field))
  ) {
    throw new Error(`Expected optional integer field: ${key}`);
  }
}

function requireOptionalNonNegativeInteger(
  value: Record<string, unknown>,
  key: string,
): void {
  requireOptionalInteger(value, key);
  const field = value[key];
  if (typeof field === "number" && field < 0) {
    throw new Error(`Expected optional non-negative integer field: ${key}`);
  }
}

function requireNonNegative(value: Record<string, unknown>, key: string): void {
  requireNumber(value, key);
  if ((value[key] as number) < 0) {
    throw new Error(`Expected non-negative field: ${key}`);
  }
}

function requirePositive(value: Record<string, unknown>, key: string): void {
  requireNumber(value, key);
  if ((value[key] as number) <= 0) {
    throw new Error(`Expected positive field: ${key}`);
  }
}

function requireNumber(value: Record<string, unknown>, key: string): void {
  if (typeof value[key] !== "number" || !Number.isFinite(value[key])) {
    throw new Error(`Expected numeric field: ${key}`);
  }
}
