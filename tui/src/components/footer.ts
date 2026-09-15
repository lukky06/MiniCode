import {
  truncateToWidth,
  visibleWidth,
  type Component,
} from "@earendil-works/pi-tui";

import { ui } from "../theme.js";

type FooterTone = "muted" | "working" | "success" | "warning" | "error";

export class Footer implements Component {
  private text = "Ready";
  private tone: FooterTone = "muted";
  private hint = "Ctrl+C exit";
  private contextUsed: number | null = null;
  private promptBudget: number | null = null;
  private permissionMode = "read-only";
  private approvalPolicy = "on-request";
  private collaborationMode = "default";

  setText(text: string): void {
    this.setStatus(text);
  }

  setStatus(text: string, tone: FooterTone = "muted", hint = ""): void {
    this.text = text;
    this.tone = tone;
    this.hint = hint;
  }

  setContext(used: number, promptBudget: number): void {
    this.contextUsed = Math.max(0, used);
    this.promptBudget = Math.max(1, promptBudget);
  }

  setSessionSettings(
    permissionMode: string,
    approvalPolicy: string,
    collaborationMode: string,
  ): void {
    this.permissionMode = permissionMode;
    this.approvalPolicy = approvalPolicy;
    this.collaborationMode = collaborationMode;
  }

  invalidate(): void {}

  render(width: number): string[] {
    if (width <= 0) return [""];

    const status = paintTone(this.tone, this.text);
    const context = this.contextLabel();

    if (width < 28) {
      return [truncateToWidth(status, width)];
    }

    const mode = this.modeLabel();
    const leftParts = [status, mode];
    if (this.hint) leftParts.push(ui.dim(this.hint));
    const left = leftParts.join("  ");

    if (!context || width < 52) {
      return [truncateToWidth(left, width)];
    }

    const right = ui.dim(context);
    const rightWidth = visibleWidth(right);
    const leftBudget = Math.max(1, width - rightWidth - 1);
    const fittedLeft = truncateToWidth(left, leftBudget);
    const spaces = Math.max(1, width - visibleWidth(fittedLeft) - rightWidth);
    return [
      truncateToWidth(
        `${fittedLeft}${" ".repeat(spaces)}${right}`,
        width,
      ),
    ];
  }

  private modeLabel(): string {
    const prefix = this.collaborationMode === "plan" ? "plan · " : "";
    return ui.dim(`${prefix}${this.permissionMode} · ${this.approvalPolicy}`);
  }

  private contextLabel(): string {
    if (this.contextUsed === null || this.promptBudget === null) return "";
    const percent = Math.min(
      999,
      Math.round((this.contextUsed / this.promptBudget) * 100),
    );
    return `context ${percent}% · ${compact(this.contextUsed)}/${compact(this.promptBudget)}`;
  }
}

function paintTone(tone: FooterTone, text: string): string {
  switch (tone) {
    case "working":
      return ui.accent(`● ${text}`);
    case "success":
      return ui.success(`✓ ${text}`);
    case "warning":
      return ui.warning(`! ${text}`);
    case "error":
      return ui.error(`✕ ${text}`);
    default:
      return ui.muted(text);
  }
}

function compact(value: number): string {
  return value >= 1000
    ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k`
    : String(value);
}
