import {
  truncateToWidth,
  visibleWidth,
  type Component,
} from "@earendil-works/pi-tui";

import { ui } from "../theme.js";

export interface HeaderOptions {
  workspace?: string;
  provider?: string;
  model?: string;
}

export class Header implements Component {
  private session = "new";
  private readonly workspace: string;
  private readonly provider?: string;
  private readonly model?: string;

  constructor(options: HeaderOptions = {}) {
    this.workspace = options.workspace ?? "";
    this.provider = options.provider;
    this.model = options.model;
  }

  setSession(sessionId: string): void {
    this.session = sessionId;
  }

  invalidate(): void {}

  render(width: number): string[] {
    if (width <= 0) return [""];

    const project = projectName(this.workspace);
    const modelLabel = this.model || this.provider || "";
    const sessionLabel = this.session === "new" ? "new session" : shortSession(this.session);

    if (width < 44) {
      const compact = [
        ui.accentStrong("MiniCode"),
        project ? ui.muted(` · ${project}`) : "",
      ].join("");
      return [truncateToWidth(compact, width)];
    }

    const left = `${ui.accentStrong("MiniCode")}  ${ui.dim(sessionLabel)}`;
    const right = modelLabel ? ui.muted(modelLabel) : "";
    const first = pair(left, right, width);
    const second = this.workspace
      ? truncateToWidth(ui.dim(this.workspace), width)
      : "";
    return second ? [first, second] : [first];
  }
}

function shortSession(value: string): string {
  const compact = value.replace(/^session_/, "");
  return `session ${compact.slice(0, 8)}`;
}

function projectName(value: string): string {
  const normalized = value.replace(/[\\/]+$/, "");
  const parts = normalized.split(/[\\/]/);
  return parts.at(-1) ?? "";
}

function pair(left: string, right: string, width: number): string {
  if (!right) return truncateToWidth(left, width);
  const rightWidth = visibleWidth(right);
  if (rightWidth >= width - 8) return truncateToWidth(left, width);
  const leftBudget = Math.max(1, width - rightWidth - 1);
  const fittedLeft = truncateToWidth(left, leftBudget);
  const spaces = Math.max(1, width - visibleWidth(fittedLeft) - rightWidth);
  return truncateToWidth(`${fittedLeft}${" ".repeat(spaces)}${right}`, width);
}
