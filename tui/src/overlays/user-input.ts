import {
  Text,
  matchesKey,
  truncateToWidth,
  type Component,
} from "@earendil-works/pi-tui";

import type { ServerMessage } from "../protocol.js";
import { ui } from "../theme.js";

type UserInputEvent = Extract<ServerMessage, { type: "user_input_required" }>;

export class UserInputOverlay implements Component {
  private readonly body = new Text();

  constructor(
    private readonly event: UserInputEvent,
    private readonly onSelect: (selectedIndex: number) => void,
  ) {}

  invalidate(): void {
    this.body.invalidate();
  }

  handleInput(data: string): void {
    const keys = ["1", "2", "3", "4"] as const;
    for (let index = 0; index < this.event.options.length; index += 1) {
      if (matchesKey(data, keys[index] ?? "1")) {
        this.onSelect(index);
        return;
      }
    }
  }

  render(width: number): string[] {
    const lines = [
      ui.warning("! Input required"),
      "",
      this.event.question,
      "",
    ];
    for (let index = 0; index < this.event.options.length; index += 1) {
      const option = this.event.options[index];
      lines.push(
        `${ui.accentStrong(String(index + 1))}  ${option.label}`,
      );
      if (option.description) {
        lines.push(`   ${ui.muted(option.description)}`);
      }
    }
    lines.push("", ui.muted("Press 1-4 to choose · Esc cancels the run"));
    this.body.setText(lines.join("\n"));
    return this.body
      .render(width)
      .map((line) => truncateToWidth(line, Math.max(0, width)));
  }
}
