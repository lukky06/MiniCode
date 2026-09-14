import {
  truncateToWidth,
  visibleWidth,
  type Component,
} from "@earendil-works/pi-tui";

import { ui } from "../theme.js";

type ActivityStatus = "working" | "worked";
type ToolStatus = "running" | "ok" | "warning" | "cancelled" | "error";

export interface ToolFinishDetails {
  commandStatus?: string | null;
  returncode?: number | null;
  durationMs?: number | null;
  runtimeTaskId?: string | null;
  diffPreview?: string | null;
  diffTruncated?: boolean;
}

interface ActivityEntry {
  id: string;
  action: string;
  target: string | null;
  detail: string | null;
  diffPreview: string | null;
  diffTruncated: boolean;
  status: ToolStatus;
}

const MAX_VISIBLE_ENTRIES = 6;
const SUCCESSFUL_TOOL_STATUSES = new Set([
  "ok",
  "duplicate_reused",
  "background_started",
]);

export class Activity implements Component {
  private status: ActivityStatus = "working";
  private readonly entries: ActivityEntry[] = [];

  start(id: string, action: string, target?: string | null): void {
    this.status = "working";
    const existing = this.entries.find((entry) => entry.id === id);
    if (existing) {
      existing.action = action || existing.action;
      existing.target = target ?? existing.target;
      existing.status = "running";
      existing.detail = null;
      existing.diffPreview = null;
      existing.diffTruncated = false;
      return;
    }
    this.entries.push({
      id,
      action: action || "Working",
      target: target ?? null,
      detail: null,
      diffPreview: null,
      diffTruncated: false,
      status: "running",
    });
  }

  finish(
    id: string,
    status: string,
    summary?: string | null,
    details: ToolFinishDetails = {},
  ): void {
    const entry = this.entries.find((candidate) => candidate.id === id);
    if (!entry) return;
    const successful = SUCCESSFUL_TOOL_STATUSES.has(status);
    entry.status =
      status === "command_timed_out"
        ? "warning"
        : status === "command_cancelled"
          ? "cancelled"
          : successful
            ? "ok"
            : "error";
    const commandDetail = formatCommandDetail(details);
    if (commandDetail) {
      entry.detail = commandDetail;
    } else if (!successful && summary) {
      entry.target = summary;
    }
    if (successful && details.diffPreview) {
      entry.diffPreview = details.diffPreview;
      entry.diffTruncated = details.diffTruncated === true;
    }
  }

  complete(): void {
    this.status = "worked";
    for (const entry of this.entries) {
      if (entry.status === "running") entry.status = "ok";
    }
  }

  invalidate(): void {}

  render(width: number): string[] {
    if (width <= 0) return [""];

    const title =
      this.status === "working"
        ? `${ui.accent("> Working")}`
        : `${ui.success("+ Worked")}${
            this.entries.length
              ? ui.muted(
                  ` · ${this.entries.length} action${this.entries.length === 1 ? "" : "s"}`,
                )
              : ""
          }`;
    const lines = [truncateToWidth(title, width)];
    const visible = this.entries.slice(-MAX_VISIBLE_ENTRIES);
    const hidden = Math.max(0, this.entries.length - visible.length);

    if (hidden > 0) {
      lines.push(
        truncateToWidth(ui.dim(`  … ${hidden} earlier action${hidden === 1 ? "" : "s"}`), width),
      );
    }

    visible.forEach((entry, index) => {
      const branch = index === visible.length - 1 ? "└" : "├";
      const icon =
        entry.status === "running"
          ? ui.accent(">")
          : entry.status === "ok"
            ? ui.success("+")
            : entry.status === "warning"
              ? ui.warning("!")
              : entry.status === "cancelled"
                ? ui.muted("-")
                : ui.error("x");
      const action = ui.accent(entry.action);
      const target = entry.target ? `  ${ui.text(entry.target)}` : "";
      const detail = entry.detail ? `  ${ui.dim(`· ${entry.detail}`)}` : "";
      const prefix = `  ${ui.dim(branch)} ${icon} `;
      const available = Math.max(0, width - visibleWidth(prefix));
      lines.push(
        truncateToWidth(
          prefix + truncateToWidth(`${action}${target}${detail}`, available),
          width,
        ),
      );
      if (entry.diffPreview) {
        for (const diffLine of entry.diffPreview.split("\n")) {
          lines.push(renderDiffLine(diffLine, width));
        }
        if (entry.diffTruncated) {
          lines.push(truncateToWidth(ui.dim("      … diff preview truncated"), width));
        }
      }
    });

    return lines;
  }
}

function renderDiffLine(line: string, width: number): string {
  const prefix = "      ";
  const content =
    line.startsWith("+++") || line.startsWith("---")
      ? ui.dim(line)
      : line.startsWith("+")
        ? ui.success(line)
        : line.startsWith("-")
          ? ui.error(line)
          : line.startsWith("@@")
            ? ui.accent(line)
            : ui.dim(line);
  return truncateToWidth(`${prefix}${content}`, width);
}

function formatCommandDetail(details: ToolFinishDetails): string | null {
  const status = details.commandStatus;
  if (!status) return null;
  const duration =
    typeof details.durationMs === "number"
      ? formatDuration(details.durationMs)
      : null;
  if (status === "background_started") {
    return details.runtimeTaskId
      ? `background ${details.runtimeTaskId}`
      : "background started";
  }
  if (status === "timed_out") {
    return ["timed out", duration].filter(Boolean).join(" · ");
  }
  if (status === "cancelled") {
    return ["cancelled", duration].filter(Boolean).join(" · ");
  }
  const exit =
    typeof details.returncode === "number" ? `exit ${details.returncode}` : null;
  return [status === "completed" ? null : status, exit, duration]
    .filter(Boolean)
    .join(" · ");
}

function formatDuration(durationMs: number): string {
  if (durationMs < 1000) return `${durationMs}ms`;
  return `${(durationMs / 1000).toFixed(durationMs < 10_000 ? 1 : 0)}s`;
}
