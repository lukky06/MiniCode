import {
  Markdown,
  truncateToWidth,
  type Component,
} from "@earendil-works/pi-tui";

import { markdownTheme, ui } from "../theme.js";

export class AssistantMessage implements Component {
  private value = "";
  private readonly markdown = new Markdown("", 0, 0, markdownTheme);

  appendDelta(text: string): void {
    this.value += text;
    this.markdown.setText(this.value);
  }

  get text(): string {
    return this.value;
  }

  invalidate(): void {
    this.markdown.invalidate();
  }

  render(width: number): string[] {
    if (width <= 0) return [""];
    const label = truncateToWidth(
      `${ui.accent("●")} ${ui.muted("MiniCode")}`,
      width,
    );
    const bodyWidth = Math.max(1, width - 2);
    const body = this.markdown
      .render(bodyWidth)
      .map((line) => truncateToWidth(`  ${line}`, width));
    return [label, ...body];
  }
}
