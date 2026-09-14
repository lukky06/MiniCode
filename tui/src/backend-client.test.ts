import assert from "node:assert/strict";
import test from "node:test";

import {
  buildBackendArgs,
  encodeClientMessage,
} from "./backend-client.js";

test("client messages are one JSONL record with escaped embedded newlines", () => {
  const encoded = encodeClientMessage({
    type: "steer",
    text: "第一行\n第二行 😀",
  });

  assert.equal(encoded.split("\n").length, 2);
  assert.deepEqual(JSON.parse(encoded), {
    type: "steer",
    text: "第一行\n第二行 😀",
  });
});

test("backend args use module execution and preserve workspace as one argv", () => {
  const args = buildBackendArgs({
    pythonExecutable: "python",
    workspace: String.raw`C:\workspace with spaces\project`,
    provider: "deepseek",
    model: "deepseek-reasoner",
    writeEnabled: false,
    approvalPolicy: "never",
    permissionMode: "workspace-write",
    collaborationMode: "plan",
    sandboxMode: "docker",
    sandboxImage: "python:3.11-slim",
    skills: ["reviewer", "handoff"],
    subagentsEnabled: false,
    mcpConfig: String.raw`C:\workspace with spaces\mcp.json`,
    sessionMode: "exact",
    sessionId: "session_abc",
  });

  assert.deepEqual(args, [
    "-m",
    "minicode_harness.tui_bridge.backend",
    "--workspace",
    String.raw`C:\workspace with spaces\project`,
    "--provider",
    "deepseek",
    "--model",
    "deepseek-reasoner",
    "--no-write",
    "--approval-policy",
    "never",
    "--permission-mode",
    "workspace-write",
    "--mode",
    "plan",
    "--sandbox-mode",
    "docker",
    "--sandbox-image",
    "python:3.11-slim",
    "--skill",
    "reviewer",
    "--skill",
    "handoff",
    "--no-subagents",
    "--mcp-config",
    String.raw`C:\workspace with spaces\mcp.json`,
    "--session-mode",
    "exact",
    "--session-id",
    "session_abc",
  ]);
});
