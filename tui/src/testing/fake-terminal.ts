import type { Terminal } from "@earendil-works/pi-tui";

export class FakeTerminal implements Terminal {
  private input?: (data: string) => void;
  private resizeHandler?: () => void;
  readonly writes: string[] = [];
  kittyProtocolActive = false;

  constructor(public columns = 80, public rows = 24) {}

  start(onInput: (data: string) => void, onResize: () => void): void {
    this.input = onInput;
    this.resizeHandler = onResize;
  }

  stop(): void {
    this.input = undefined;
    this.resizeHandler = undefined;
  }

  async drainInput(): Promise<void> {}

  write(data: string): void {
    this.writes.push(data);
  }

  moveBy(): void {}
  hideCursor(): void {}
  showCursor(): void {}
  clearLine(): void {}
  clearFromCursor(): void {}
  clearScreen(): void {}
  setTitle(): void {}
  setProgress(): void {}

  sendInput(data: string): void {
    this.input?.(data);
  }

  resize(columns: number, rows: number): void {
    this.columns = columns;
    this.rows = rows;
    this.resizeHandler?.();
  }
}
