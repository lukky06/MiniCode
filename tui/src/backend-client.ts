import {
  spawn,
  type ChildProcessWithoutNullStreams,
} from "node:child_process";
import { createInterface } from "node:readline";

import {
  parseServerMessage,
  type ClientMessage,
  type ServerMessage,
} from "./protocol.js";

export interface BackendClientOptions {
  pythonExecutable: string;
  workspace: string;
  provider?: string;
  model?: string;
  writeEnabled?: boolean;
  approvalPolicy?: "on-request" | "never";
  permissionMode?: "read-only" | "workspace-write" | "full-access";
  sandboxMode?: "local" | "docker";
  sandboxImage?: string;
  collaborationMode?: "default" | "plan";
  skills?: string[];
  skillsEnabled?: boolean;
  projectConventionsEnabled?: boolean;
  subagentsEnabled?: boolean;
  mcpConfig?: string;
  sessionMode?: "new" | "continue" | "exact";
  sessionId?: string;
}

export interface BackendHandlers {
  onEvent: (event: ServerMessage) => void;
  onProtocolError?: (message: string) => void;
  onDebug?: (text: string) => void;
  onExit?: (code: number | null, signal: NodeJS.Signals | null) => void;
}

export class BackendClient {
  private child: ChildProcessWithoutNullStreams | null = null;

  constructor(private readonly options: BackendClientOptions) {}

  start(handlers: BackendHandlers): void {
    if (this.child !== null) {
      throw new Error("MiniCode backend is already running.");
    }

    const child = spawn(
      this.options.pythonExecutable,
      buildBackendArgs(this.options),
      {
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
      },
    );
    this.child = child;

    const lines = createInterface({ input: child.stdout });
    lines.on("line", (line) => {
      try {
        handlers.onEvent(parseServerMessage(line));
      } catch (error) {
        handlers.onProtocolError?.(
          error instanceof Error ? error.message : String(error),
        );
      }
    });

    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      handlers.onDebug?.(chunk);
    });

    child.on("exit", (code, signal) => {
      lines.close();
      this.child = null;
      handlers.onExit?.(code, signal);
    });
  }

  send(message: ClientMessage): void {
    const child = this.child;
    if (child === null || child.stdin.destroyed || !child.stdin.writable) {
      throw new Error("MiniCode backend is not writable.");
    }
    child.stdin.write(encodeClientMessage(message));
  }

  close(): void {
    const child = this.child;
    if (child === null) return;
    if (!child.stdin.destroyed && child.stdin.writable) {
      child.stdin.end();
    }
  }
}

export function encodeClientMessage(message: ClientMessage): string {
  return JSON.stringify(message) + "\n";
}

export function buildBackendArgs(options: BackendClientOptions): string[] {
  const args = [
    "-m",
    "minicode_harness.tui_bridge.backend",
    "--workspace",
    options.workspace,
  ];
  if (options.provider) {
    args.push("--provider", options.provider);
  }
  if (options.model) {
    args.push("--model", options.model);
  }
  if (options.writeEnabled === false) {
    args.push("--no-write");
  }
  if (options.approvalPolicy) {
    args.push("--approval-policy", options.approvalPolicy);
  }
  if (options.permissionMode) {
    args.push("--permission-mode", options.permissionMode);
  }
  if (options.collaborationMode) {
    args.push("--mode", options.collaborationMode);
  }
  if (options.sandboxMode) {
    args.push("--sandbox-mode", options.sandboxMode);
  }
  if (options.sandboxImage) {
    args.push("--sandbox-image", options.sandboxImage);
  }
  for (const skill of options.skills ?? []) {
    args.push("--skill", skill);
  }
  if (options.skillsEnabled === false) {
    args.push("--no-skills");
  }
  if (options.projectConventionsEnabled === false) {
    args.push("--no-project-conventions");
  }
  if (options.subagentsEnabled === false) {
    args.push("--no-subagents");
  }
  if (options.mcpConfig) {
    args.push("--mcp-config", options.mcpConfig);
  }
  if (options.sessionMode) {
    args.push("--session-mode", options.sessionMode);
  }
  if (options.sessionId) {
    args.push("--session-id", options.sessionId);
  }
  return args;
}
