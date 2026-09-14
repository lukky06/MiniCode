import {
  ProcessTerminal,
  Key,
  ScrollView,
  Text,
  TuiAltScreen,
  VStack,
  matchesKey,
  type OverlayHandle,
  type Terminal,
} from "@earendil-works/pi-tui";

import { Composer } from "./components/composer.js";
import { Footer } from "./components/footer.js";
import { Header } from "./components/header.js";
import { ApprovalOverlay } from "./overlays/approval.js";
import { UserInputOverlay } from "./overlays/user-input.js";
import type { ClientMessage, ServerMessage } from "./protocol.js";
import { ui } from "./theme.js";
import { Transcript } from "./transcript.js";

const BOTTOM_DOCK_MIN_ROWS = 4;

export interface MiniCodeTuiOptions {
  terminal?: Terminal;
  onSubmit?: (text: string) => void;
  onClientMessage?: (message: ClientMessage) => void;
  onExit?: () => void;
  workspace?: string;
  provider?: string;
  model?: string;
}

export class MiniCodeTuiApp {
  readonly transcript = new Transcript();
  readonly header: Header;
  readonly footer = new Footer();
  readonly tui: TuiAltScreen;
  readonly editor: Composer;
  private running = false;
  private approvalHandle: OverlayHandle | null = null;
  private userInputHandle: OverlayHandle | null = null;
  private panelHandle: OverlayHandle | null = null;
  private pendingApprovalId: string | null = null;
  private pendingUserInputId: string | null = null;
  private readonly onSubmit?: (text: string) => void;
  private readonly onClientMessage?: (message: ClientMessage) => void;
  private readonly onExit?: () => void;

  constructor(options: MiniCodeTuiOptions = {}) {
    const terminal = options.terminal ?? new ProcessTerminal();
    this.onSubmit = options.onSubmit;
    this.onClientMessage = options.onClientMessage;
    this.onExit = options.onExit;
    this.header = new Header({
      workspace: options.workspace,
      provider: options.provider,
      model: options.model,
    });
    this.tui = new TuiAltScreen(terminal, true);
    this.editor = new Composer(this.tui);

    this.editor.onSubmit = (text) => this.submitText(text);

    this.tui.addInputListener((data) => {
      if (
        this.panelHandle !== null &&
        matchesKey(data, Key.escape)
      ) {
        this.clearPanel();
        this.tui.requestRender();
        return { consume: true };
      }
      if (
        this.panelHandle === null &&
        this.pendingApprovalId === null &&
        this.pendingUserInputId === null &&
        matchesKey(data, Key.ctrl("o"))
      ) {
        const expanded = this.transcript.toggleReasoning();
        this.footer.setStatus(
          this.running ? "Working" : "Ready",
          this.running ? "working" : "muted",
          expanded ? "reasoning expanded" : "reasoning collapsed",
        );
        this.tui.requestRender();
        return { consume: true };
      }
      if (!this.running) {
        if (matchesKey(data, Key.ctrl("c"))) {
          if (this.editor.getText()) {
            this.editor.setText("");
            this.tui.requestRender();
          } else {
            this.onExit?.();
          }
          return { consume: true };
        }
        return undefined;
      }
      if (
        matchesKey(data, Key.escape) ||
        matchesKey(data, Key.ctrl("c"))
      ) {
        this.onClientMessage?.({ type: "cancel" });
        this.footer.setStatus("Cancelling", "warning", "waiting for active tool");
        this.tui.requestRender();
        return { consume: true };
      }
      return undefined;
    });

    const transcriptView = new ScrollView(this.transcript, {
      follow: "end",
      primary: true,
      overscroll: "contain",
      scrollbar: "auto",
    });
    const bottomDock = new VStack([
      { component: this.editor, basis: "auto" },
      { component: this.footer, basis: "auto" },
    ]);
    const root = new VStack([
      { component: this.header, basis: "auto" },
      { component: transcriptView, basis: 0, grow: 1, minSize: 1 },
      { component: bottomDock, basis: "auto", shrink: 1, minSize: BOTTOM_DOCK_MIN_ROWS },
    ]);

    this.tui.setLayoutRoot(root);
    this.tui.setFocus(this.editor);
  }

  submitText(text: string): void {
    const task = text.trim();
    if (
      !task ||
      this.pendingApprovalId !== null ||
      this.pendingUserInputId !== null ||
      this.panelHandle !== null
    ) return;
    if (!this.running && task.startsWith("/")) {
      this.editor.addToHistory(task);
      this.editor.setText("");
      this.onClientMessage?.({ type: "command", text: task });
      this.footer.setStatus(`Loading ${task}`, "working");
      this.tui.requestRender();
      return;
    }
    this.transcript.appendUser(task);
    const message: ClientMessage = this.running
      ? { type: "steer", text: task }
      : { type: "task", text: task };
    if (!this.running) {
      this.running = true;
      this.transcript.startRun();
      this.editor.setMode("steer");
      this.footer.setStatus("Working", "working");
    }
    this.editor.addToHistory(task);
    this.editor.setText("");
    this.onSubmit?.(task);
    this.onClientMessage?.(message);
    this.tui.requestRender();
  }

  start(): void {
    this.tui.start();
  }

  stop(): void {
    this.tui.stop();
  }

  handleServerEvent(event: ServerMessage): void {
    switch (event.type) {
      case "session_started":
        this.header.setSession(event.session_id);
        break;
      case "run_started":
        this.running = true;
        this.transcript.startRun();
        this.editor.setMode("steer");
        this.footer.setStatus("Working", "working");
        break;
      case "context":
        this.footer.setContext(event.used, event.prompt_budget);
        break;
      case "tool_started":
        this.transcript.startTool(
          event.id,
          toolAction(event.tool),
          event.target,
        );
        this.footer.setStatus("Working", "working");
        break;
      case "tool_finished":
        this.transcript.finishTool(event.id, event.status, event.summary, {
          commandStatus: event.command_status,
          returncode: event.returncode,
          durationMs: event.duration_ms,
          runtimeTaskId: event.runtime_task_id,
          diffPreview: event.diff_preview,
          diffTruncated: event.diff_truncated,
        });
        break;
      case "reasoning_delta":
        this.transcript.appendReasoningDelta(event.text);
        this.footer.setStatus("Thinking", "working", "Ctrl+O details");
        break;
      case "assistant_delta":
        this.transcript.completeActivity();
        this.transcript.appendAssistantDelta(event.text);
        this.footer.setStatus("Answering", "working");
        break;
      case "approval_required":
        this.transcript.startTool(
          event.id,
          "Approval required",
          event.summary ?? event.tool,
        );
        this.showApproval(event);
        this.editor.setMode("approval");
        this.footer.setStatus("Permission required", "warning");
        break;
      case "user_input_required":
        this.showUserInput(event);
        this.editor.setMode("approval");
        this.footer.setStatus("Input required", "warning");
        break;
      case "run_finished":
        this.running = false;
        this.clearApproval();
        this.clearUserInput();
        this.transcript.completeReasoning();
        this.transcript.completeActivity();
        this.editor.setMode("ask");
        this.footer.setStatus(
          event.status,
          event.status === "completed" ? "success" : "muted",
          "Ctrl+C exit",
        );
        break;
      case "error":
        if (event.fatal) {
          this.running = false;
          this.clearApproval();
          this.clearUserInput();
          this.clearPanel();
        }
        this.editor.setMode(this.running ? "steer" : "ask");
        this.footer.setStatus(`Error: ${event.message}`, "error");
        break;
      case "panel":
        this.showPanel(event);
        break;
      case "exit_requested":
        this.onExit?.();
        return;
    }
    this.tui.requestRender();
  }

  private showApproval(
    event: Extract<ServerMessage, { type: "approval_required" }>,
  ): void {
    this.clearApproval();
    this.pendingApprovalId = event.id;
    this.editor.disableSubmit = true;
    const overlay = new ApprovalOverlay(event, (decision) => {
      if (this.pendingApprovalId !== event.id) return;
      this.onClientMessage?.({
        type: "approval_response",
        id: event.id,
        decision,
      });
      this.clearApproval();
      this.tui.requestRender();
    });
    this.approvalHandle = this.tui.showOverlay(overlay, {
      anchor: "bottom-center",
      width: "80%",
      maxHeight: "60%",
      margin: { left: 1, right: 1, bottom: BOTTOM_DOCK_MIN_ROWS },
    });
  }

  private clearApproval(): void {
    this.approvalHandle?.hide();
    this.approvalHandle = null;
    this.pendingApprovalId = null;
    this.editor.disableSubmit = false;
    this.editor.setMode(this.running ? "steer" : "ask");
    if (this.running) {
      this.footer.setStatus("Working", "working");
    }
  }

  private showUserInput(
    event: Extract<ServerMessage, { type: "user_input_required" }>,
  ): void {
    this.clearUserInput();
    this.pendingUserInputId = event.id;
    this.editor.disableSubmit = true;
    const overlay = new UserInputOverlay(event, (selectedIndex) => {
      if (this.pendingUserInputId !== event.id) return;
      this.onClientMessage?.({
        type: "user_input_response",
        id: event.id,
        selected_index: selectedIndex,
      });
      this.clearUserInput();
      this.tui.requestRender();
    });
    this.userInputHandle = this.tui.showOverlay(overlay, {
      anchor: "center",
      width: "80%",
      maxHeight: "80%",
      margin: 1,
    });
  }

  private clearUserInput(): void {
    this.userInputHandle?.hide();
    this.userInputHandle = null;
    this.pendingUserInputId = null;
    this.editor.disableSubmit = false;
    this.editor.setMode(this.running ? "steer" : "ask");
    if (this.running) {
      this.footer.setStatus("Working", "working");
    } else {
      this.footer.setStatus("Ready", "muted", "Ctrl+C exit");
    }
  }

  private showPanel(event: Extract<ServerMessage, { type: "panel" }>): void {
    this.clearPanel();
    const content = new Text(
      `${ui.accentStrong(event.title)}\n\n${event.content}\n\n${ui.dim("Esc close")}`,
    );
    const scroll = new ScrollView(content, {
      follow: "none",
      overscroll: "contain",
      scrollbar: "auto",
    });
    this.panelHandle = this.tui.showOverlay(scroll, {
      anchor: "center",
      width: "90%",
      maxHeight: "80%",
      margin: 1,
    });
    this.footer.setStatus(event.title, "muted", "Esc close");
  }

  private clearPanel(): void {
    this.panelHandle?.hide();
    this.panelHandle = null;
    this.footer.setStatus(
      this.running ? "Working" : "Ready",
      this.running ? "working" : "muted",
      this.running ? "" : "Ctrl+C exit",
    );
  }
}

function toolAction(tool: string): string {
  switch (tool) {
    case "read":
    case "search":
      return "Explore";
    case "edit":
    case "write":
    case "apply_patch":
      return "Change";
    case "run_command":
      return "Run";
    case "delegate_task":
    case "delegate_worktree":
      return "Delegate";
    default:
      return tool;
  }
}
