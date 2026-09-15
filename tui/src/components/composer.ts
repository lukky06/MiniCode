import {
  Editor,
  truncateToWidth,
  visibleWidth,
  type TUI,
} from "@earendil-works/pi-tui";

import { editorTheme, ui } from "../theme.js";

export type ComposerMode = "ask" | "steer" | "approval";

export class Composer extends Editor {
  private mode: ComposerMode = "ask";

  constructor(tui: TUI) {
    super(tui, editorTheme, { paddingX: 1 });
  }

  setMode(mode: ComposerMode): void {
    this.mode = mode;
    this.borderColor =
      mode === "approval"
        ? ui.warning
        : mode === "steer"
          ? ui.accent
          : ui.border;
  }

  protected renderTopBorder(width: number, hiddenLineCount: number): string {
    if (hiddenLineCount > 0) {
      return super.renderTopBorder(width, hiddenLineCount);
    }
    if (width <= 0) return "";

    const label =
      this.mode === "approval"
        ? " Permission required "
        : this.mode === "steer"
          ? " Steer MiniCode "
          : " Ask MiniCode ";
    const styledLabel =
      this.mode === "approval" ? ui.warning(label) : ui.accentStrong(label);
    const remaining = Math.max(0, width - visibleWidth(label));
    return truncateToWidth(
      styledLabel + this.borderColor("─".repeat(remaining)),
      width,
    );
  }

  protected renderBottomBorder(width: number, hiddenLineCount: number): string {
    if (hiddenLineCount > 0) {
      return super.renderBottomBorder(width, hiddenLineCount);
    }
    if (width <= 0) return "";

    const hint =
      this.mode === "approval"
        ? " choose in the approval panel "
        : this.mode === "steer"
          ? " Enter steer · Esc cancel "
          : " Enter send · / commands · Alt+Enter newline ";
    if (visibleWidth(hint) >= width - 2) {
      return this.borderColor("─".repeat(width));
    }
    const remaining = Math.max(0, width - visibleWidth(hint));
    const left = Math.floor(remaining / 2);
    const right = remaining - left;
    return truncateToWidth(
      this.borderColor("─".repeat(left)) +
        ui.dim(hint) +
        this.borderColor("─".repeat(right)),
      width,
    );
  }
}
