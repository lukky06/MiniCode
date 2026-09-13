import { BackendClient } from "./backend-client.js";
import { MiniCodeTuiApp } from "./app.js";

const sessionMode = process.env.MINICODE_TUI_SESSION_MODE as
  | "new"
  | "continue"
  | "exact"
  | undefined;
const skills = JSON.parse(process.env.MINICODE_TUI_SKILLS ?? "[]") as string[];

const backend = new BackendClient({
  pythonExecutable: process.env.MINICODE_PYTHON || "python",
  workspace: process.env.MINICODE_WORKSPACE || process.cwd(),
  provider: process.env.MINICODE_PROVIDER,
  model: process.env.MINICODE_MODEL,
  writeEnabled: process.env.MINICODE_TUI_NO_WRITE !== "1",
  approvalPolicy: process.env.MINICODE_TUI_APPROVAL_POLICY as
    | "on-request"
    | "never"
    | undefined,
  permissionMode: process.env.MINICODE_TUI_PERMISSION_MODE as
    | "read-only"
    | "workspace-write"
    | "full-access"
    | undefined,
  collaborationMode: process.env.MINICODE_TUI_COLLABORATION_MODE as
    | "default"
    | "plan"
    | undefined,
  sandboxMode: process.env.MINICODE_TUI_SANDBOX_MODE as
    | "local"
    | "docker"
    | undefined,
  sandboxImage: process.env.MINICODE_TUI_SANDBOX_IMAGE,
  skills,
  skillsEnabled: process.env.MINICODE_TUI_NO_SKILLS !== "1",
  promptCacheEnabled: process.env.MINICODE_TUI_NO_PROMPT_CACHE !== "1",
  projectConventionsEnabled:
    process.env.MINICODE_TUI_NO_PROJECT_CONVENTIONS !== "1",
  subagentsEnabled: process.env.MINICODE_TUI_NO_SUBAGENTS !== "1",
  mcpConfig: process.env.MINICODE_TUI_MCP_CONFIG,
  sessionMode,
  sessionId: process.env.MINICODE_TUI_SESSION_ID,
});

let app: MiniCodeTuiApp;
let shuttingDown = false;
let fatalBackendError: string | undefined;
const shutdown = (): void => {
  if (shuttingDown) return;
  shuttingDown = true;
  backend.close();
  app.stop();
};

app = new MiniCodeTuiApp({
  onClientMessage: (message) => backend.send(message),
  onExit: shutdown,
  workspace: process.env.MINICODE_WORKSPACE || process.cwd(),
  provider: process.env.MINICODE_PROVIDER,
  model: process.env.MINICODE_MODEL,
});

app.start();
backend.start({
  onEvent: (event) => {
    if (event.type === "error" && event.fatal) {
      fatalBackendError = event.message;
    }
    app.handleServerEvent(event);
  },
  onProtocolError: (message) =>
    app.handleServerEvent({
      type: "error",
      message: "Protocol error: " + message,
      fatal: true,
    }),
  onExit: (code, signal) => {
    if (shuttingDown) return;
    if (code === 0 && signal === null) {
      app.stop();
      return;
    }
    const message = fatalBackendError ??
      "Backend exited (" + (code ?? signal ?? "unknown") + ").";
    app.stop();
    process.stderr.write("MiniCode backend error: " + message + "\n");
    process.exitCode = typeof code === "number" && code !== 0 ? code : 1;
  },
});

const initialTask = process.env.MINICODE_TUI_INITIAL_TASK?.trim();
if (initialTask) {
  app.submitText(initialTask);
}

process.once("SIGTERM", shutdown);
