import {
  Key,
  Text,
  matchesKey,
  truncateToWidth,
  type Component,
} from "@earendil-works/pi-tui";

import type { ServerMessage } from "../protocol.js";
import { ui } from "../theme.js";

type ApprovalEvent = Extract<ServerMessage, { type: "approval_required" }>;
export type ApprovalDecision =
  | "approve"
  | "approve_session"
  | "reject"
  | "skip"
  | "abort";

export class ApprovalOverlay implements Component {
  private readonly body = new Text();
  private showDetails = false;

  constructor(
    private readonly event: ApprovalEvent,
    private readonly onDecision: (decision: ApprovalDecision) => void,
  ) {}

  invalidate(): void {
    this.body.invalidate();
  }

  handleInput(data: string): void {
    if (matchesKey(data, "y")) {
      this.onDecision("approve");
      return;
    }
    if (this.event.can_approve_session && matchesKey(data, "g")) {
      this.onDecision("approve_session");
      return;
    }
    if (matchesKey(data, "n")) {
      this.onDecision("reject");
      return;
    }
    if (matchesKey(data, "s")) {
      this.onDecision("skip");
      return;
    }
    if (matchesKey(data, "a")) {
      this.onDecision("abort");
      return;
    }
    if (matchesKey(data, "v")) {
      this.showDetails = !this.showDetails;
      return;
    }
    if (matchesKey(data, Key.escape)) {
      this.onDecision("abort");
    }
  }

  render(width: number): string[] {
    const preview = approvalPreview(this.event.details);
    const lines = [
      ui.warning("! ACTION REQUIRED"),
      ui.bold(approvalAction(this.event.tool)),
      "",
    ];
    if (preview.command) {
      lines.push(ui.code(`$ ${preview.command}`), "");
    } else if (preview.path) {
      lines.push(ui.code(preview.path), "");
    } else if (this.event.summary) {
      lines.push(this.event.summary, "");
    }
    if (preview.riskLevel) {
      lines.push(`${ui.muted("Risk")}    ${ui.warning(preview.riskLevel.toUpperCase())}`);
    }
    if (preview.reason) {
      lines.push(`${ui.muted("Reason")}  ${preview.reason}`);
    }
    if (preview.effects.length > 0) {
      lines.push(ui.muted("Effects"));
      lines.push(...preview.effects.slice(0, 3).map((effect) => `  - ${effect}`));
    }
    if (preview.riskLevel || preview.reason || preview.effects.length > 0) {
      lines.push("");
    }
    if (this.showDetails && this.event.details) {
      lines.push(this.event.details, "");
    }
    const grant = this.event.can_approve_session
      ? `  ${ui.success("[G] Allow for this session")}`
      : "";
    lines.push(
      `${ui.success("[Y] Allow once")}${grant}  ${ui.error("[N] Reject")}`,
      ui.muted("[S] Skip · [A] Abort · [V] Details"),
    );
    this.body.setText(lines.join("\n"));
    return this.body
      .render(width)
      .map((line) => truncateToWidth(line, Math.max(0, width)));
  }
}

type ApprovalPreview = {
  riskLevel?: string;
  command?: string;
  path?: string;
  reason?: string;
  effects: string[];
};

function approvalPreview(details?: string | null): ApprovalPreview {
  if (!details) return { effects: [] };
  try {
    const raw = JSON.parse(details);
    if (!isRecord(raw)) return { effects: [] };
    const preview = isRecord(raw.preview) ? raw.preview : {};
    return {
      riskLevel: stringValue(raw.risk_level),
      command: stringValue(preview.command),
      path: stringValue(preview.path),
      reason: stringValue(preview.reason),
      effects: Array.isArray(preview.effects)
        ? preview.effects.filter((value): value is string => typeof value === "string")
        : [],
    };
  } catch {
    return { effects: [] };
  }
}

function approvalAction(tool: string): string {
  if (tool === "run_command") return "MiniCode wants to run a command";
  if (["edit", "write", "apply_patch"].includes(tool)) {
    return "MiniCode wants to change workspace files";
  }
  return "MiniCode requests permission";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}
