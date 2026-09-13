import {
  Markdown,
  truncateToWidth,
  type Component,
} from "@earendil-works/pi-tui";

import { markdownTheme, ui } from "../theme.js";

export class ReasoningMessage implements Component {
  private value = "";
  private expanded = false;
  private complete = false;
  private readonly markdown = new Markdown("", 0, 0, markdownTheme);

  appendDelta(text: string): void {
    this.value += text;
    this.markdown.setText(this.value);
  }

  finish(): void {
    this.complete = true;
  }

  setExpanded(expanded: boolean): void {
    this.expanded = expanded;
  }

  get text(): string {
    return this.value;
  }

  invalidate(): void {
    this.markdown.invalidate();
  }

  render(width: number): string[] {
    if (width <= 0) return [""];

    const state = this.complete ? "Thought" : "Thinking";
    const hint = this.expanded ? "Ctrl+O collapse" : "Ctrl+O expand";
    const label = truncateToWidth(
      `${ui.accent(">")} ${ui.muted(state)} ${ui.dim(`[${hint}]`)}`,
      width,
    );
    if (!this.value.trim()) return [label];

    if (!this.expanded) {
      const preview = this.value.replace(/\s+/g, " ").trim();
      return [
        label,
        truncateToWidth(`  ${ui.dim(preview)}`, width),
      ];
    }

    const bodyWidth = Math.max(1, width - 2);
    const body = this.markdown
      .render(bodyWidth)
      .map((line) => truncateToWidth(`  ${ui.dim(line)}`, width));
    return [label, ...body];
  }
}
