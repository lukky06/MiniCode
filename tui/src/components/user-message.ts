import {
  Text,
  truncateToWidth,
  type Component,
} from "@earendil-works/pi-tui";

import { ui } from "../theme.js";

export class UserMessage implements Component {
  private readonly text: Text;

  constructor(message: string) {
    this.text = new Text(`${ui.accentStrong("›")} ${ui.bold(message)}`);
  }

  invalidate(): void {
    this.text.invalidate();
  }

  render(width: number): string[] {
    return this.text
      .render(width)
      .map((line) => truncateToWidth(line, Math.max(0, width)));
  }
}
