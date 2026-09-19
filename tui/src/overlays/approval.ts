import {
  Text,
  matchesKey,
  truncateToWidth,
  type Component,
} from "@earendil-works/pi-tui";

import type { ServerMessage } from "../protocol.js";
import { ui } from "../theme.js";

type ApprovalEvent = Extract<ServerMessage, { type: "approval_required" }>;
export type ApprovalDecision = "approve" | "approve_session" | "reject";

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
    if (this.event.session_scope && matchesKey(data, "g")) {
      this.onDecision("approve_session");
      return;
    }
    if (matchesKey(data, "n")) {
      this.onDecision("reject");
      return;
    }
    if (matchesKey(data, "v")) {
      this.showDetails = !this.showDetails;
    }
  }

  render(width: number): string[] {
    const preview = approvalPreview(this.event.details);
    const lines = [
      ui.warning("! PERMISSION REQUIRED"),
      ui.bold(approvalQuestion(this.event.tool)),
      "",
    ];

    if (preview.command) {
      lines.push(ui.code(`$ ${preview.command}`), "");
    } else if (preview.path) {
      lines.push(ui.code(preview.path), "");
    } else if (this.event.summary) {
      lines.push(this.event.summary, "");
    }

    if (preview.reason) {
      lines.push(`${ui.muted("Reason")}  ${preview.reason}`);
    }
    if (preview.effects.length > 0) {
      lines.push(ui.muted("Effects"));
      lines.push(...preview.effects.slice(0, 3).map((effect) => `  - ${effect}`));
    }
    if (this.event.session_scope) {
      lines.push(
        `${ui.muted("Session scope")}  ${ui.code(this.event.session_scope)}`,
      );
    }
    if (preview.reason || preview.effects.length > 0 || this.event.session_scope) {
      lines.push("");
    }

    if (this.showDetails && this.event.details) {
      lines.push(this.event.details, "");
    }

    const grant = this.event.session_scope
      ? `  ${ui.success("[G] Allow similar this session")}`
      : "";
    lines.push(
      `${ui.success("[Y] Allow once")}${grant}  ${ui.error("[N] Reject")}`,
      ui.muted("[V] Details"),
    );

    this.body.setText(lines.join("\n"));
    return this.body
      .render(width)
      .map((line) => truncateToWidth(line, Math.max(0, width)));
  }
}

type ApprovalPreview = {
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

function approvalQuestion(tool: string): string {
  if (tool === "run_command") {
    return "Run this command?";
  }
  if (["edit", "write", "apply_patch"].includes(tool)) {
    return "Change workspace files?";
  }
  return "Allow this action?";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}
