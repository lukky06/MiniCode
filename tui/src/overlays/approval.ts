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
    const lines = [
      `${ui.warning("! Approval required")}  ${ui.muted(this.event.tool)}`,
      "",
      this.event.summary || this.event.tool,
      "",
    ];
    if (this.showDetails && this.event.details) {
      lines.push(this.event.details, "");
    }
    const grant = this.event.can_approve_session
      ? `  ${ui.success("g session")}`
      : "";
    lines.push(
      `${ui.success("y once")}${grant}  ${ui.error("n reject")}  ${ui.muted("s skip · a abort · v details")}`,
    );
    this.body.setText(lines.join("\n"));
    return this.body
      .render(width)
      .map((line) => truncateToWidth(line, Math.max(0, width)));
  }
}
