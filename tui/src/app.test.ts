import assert from "node:assert/strict";
import test from "node:test";

import {
  stripTerminalSequences,
  visibleWidth,
} from "@earendil-works/pi-tui";

import { MiniCodeTuiApp } from "./app.js";
import { FakeTerminal } from "./testing/fake-terminal.js";

test("fake server events update one transcript and survive resize", () => {
  const terminal = new FakeTerminal(80, 20);
  const app = new MiniCodeTuiApp({ terminal });

  app.start();
  app.handleServerEvent({ type: "session_started", session_id: "会话_1" });
  app.handleServerEvent({ type: "run_started", run_id: "run_1" });
  app.handleServerEvent({
    type: "tool_started",
    id: "call_1",
    step: 1,
    tool: "read",
    target: String.raw`C:\workspace\project\src\terminal\layout.py`,
  });

  app.handleServerEvent({ type: "assistant_delta", text: "已经" });
  app.tui.renderNow(true);
  const writesAfterFirstDelta = terminal.writes.length;

  app.handleServerEvent({ type: "assistant_delta", text: "完成修改 😀" });
  app.tui.renderNow();
  assert.ok(terminal.writes.length > writesAfterFirstDelta);
  assert.equal(app.transcript.currentAssistantText, "已经完成修改 😀");

  for (const width of [10, 24, 80]) {
    for (const line of app.transcript.render(width)) {
      assert.ok(visibleWidth(line) <= width);
    }
  }

  const writesBeforeResize = terminal.writes.length;
  terminal.resize(18, 10);
  app.tui.renderNow(true);
  assert.ok(terminal.writes.length > writesBeforeResize);
  app.stop();
});

test("T7 product UI exposes project hierarchy, run status, and tool timeline", () => {
  const terminal = new FakeTerminal(96, 28);
  const app = new MiniCodeTuiApp({
    terminal,
    workspace: String.raw`C:\workspace\hmdp-rebuild-simplified`,
    provider: "deepseek",
  });

  app.handleServerEvent({ type: "session_started", session_id: "session_a409a4903a95" });
  const header = stripTerminalSequences(app.header.render(96).join("\n"));
  assert.match(header, /MiniCode/);
  assert.match(header, /deepseek/);
  assert.match(header, /hmdp-rebuild-simplified/);
  assert.ok(app.header.render(96).join("\n").includes("\x1b["));

  const idleComposer = stripTerminalSequences(app.editor.render(96).join("\n"));
  assert.match(idleComposer, /Ask MiniCode/);

  app.handleServerEvent({ type: "run_started", run_id: "run_1" });
  app.handleServerEvent({
    type: "context",
    used: 8_000,
    window: 100_000,
    prompt_budget: 80_000,
    reserved_output: 8_000,
  });
  app.handleServerEvent({
    type: "tool_started",
    id: "read_1",
    step: 1,
    tool: "read",
    target: "src/main/java/App.java",
  });
  app.handleServerEvent({
    type: "tool_finished",
    id: "read_1",
    step: 1,
    tool: "read",
    status: "ok",
  });
  app.handleServerEvent({
    type: "tool_started",
    id: "search_1",
    step: 1,
    tool: "search",
    target: "VoucherOrderService",
  });

  const runningComposer = stripTerminalSequences(app.editor.render(96).join("\n"));
  const runningTranscript = stripTerminalSequences(app.transcript.render(96).join("\n"));
  const runningFooter = stripTerminalSequences(app.footer.render(96).join("\n"));
  assert.match(runningComposer, /Steer MiniCode/);
  assert.match(runningTranscript, /Working/);
  assert.match(runningTranscript, /src\/main\/java\/App.java/);
  assert.match(runningTranscript, /VoucherOrderService/);
  assert.match(runningFooter, /Working/);
  assert.match(runningFooter, /context 10%/);

  app.handleServerEvent({ type: "assistant_delta", text: "完成。" });
  const answered = stripTerminalSequences(app.transcript.render(96).join("\n"));
  assert.match(answered, /Worked · 2 actions/);
  assert.match(answered, /MiniCode/);
  assert.match(answered, /完成。/);

  app.handleServerEvent({
    type: "run_finished",
    status: "completed",
    run_id: "run_1",
  });
  assert.match(stripTerminalSequences(app.editor.render(96).join("\n")), /Ask MiniCode/);
  assert.match(stripTerminalSequences(app.footer.render(96).join("\n")), /completed/);
});


test("successful mutation tool event renders inline diff in transcript", () => {
  const terminal = new FakeTerminal(90, 24);
  const app = new MiniCodeTuiApp({ terminal });

  app.handleServerEvent({ type: "run_started", run_id: "run_diff" });
  app.handleServerEvent({
    type: "tool_started",
    id: "edit_1",
    step: 1,
    tool: "edit",
    target: "src/app.py",
  });
  app.handleServerEvent({
    type: "tool_finished",
    id: "edit_1",
    step: 1,
    tool: "edit",
    status: "ok",
    diff_preview: "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new",
    diff_truncated: false,
  });

  const rendered = stripTerminalSequences(app.transcript.render(90).join("\n"));
  assert.match(rendered, /src\/app.py/);
  assert.match(rendered, /-old/);
  assert.match(rendered, /\+new/);
});

test("reasoning stays collapsed by default and Ctrl+O expands it", () => {
  const terminal = new FakeTerminal(80, 24);
  const app = new MiniCodeTuiApp({ terminal });
  app.start();

  app.handleServerEvent({ type: "run_started", run_id: "run_reasoning" });
  app.handleServerEvent({
    type: "reasoning_delta",
    text: "先检查入口。\n\n第二段包含 UNIQUE_REASONING_DETAIL。",
  });

  const collapsed = stripTerminalSequences(app.transcript.render(36).join("\n"));
  assert.match(collapsed, /Thinking/);
  assert.doesNotMatch(collapsed, /UNIQUE_REASONING_DETAIL/);
  assert.match(stripTerminalSequences(app.footer.render(80).join("\n")), /Ctrl\+O details/);

  terminal.sendInput("\x0f");
  const expanded = stripTerminalSequences(app.transcript.render(80).join("\n"));
  assert.match(expanded, /UNIQUE_REASONING_DETAIL/);

  app.handleServerEvent({
    type: "tool_started",
    id: "read_reasoning",
    step: 1,
    tool: "read",
    target: "README.md",
  });
  const completed = stripTerminalSequences(app.transcript.render(80).join("\n"));
  assert.match(completed, /Thought/);
  assert.match(completed, /Working/);

  app.stop();
});

test("editor submit routes idle task then running steering", () => {
  const terminal = new FakeTerminal();
  const submitted: string[] = [];
  const messages: unknown[] = [];
  const app = new MiniCodeTuiApp({
    terminal,
    onSubmit: (text) => submitted.push(text),
    onClientMessage: (message) => messages.push(message),
  });

  app.editor.setText("  只修改 terminal  ");
  app.editor.onSubmit?.(app.editor.getText());
  app.editor.setText("不要修改 runtime");
  app.editor.onSubmit?.(app.editor.getText());

  assert.deepEqual(submitted, ["只修改 terminal", "不要修改 runtime"]);
  assert.deepEqual(messages, [
    { type: "task", text: "只修改 terminal" },
    { type: "steer", text: "不要修改 runtime" },
  ]);
  assert.equal(app.editor.getText(), "");
  assert.match(app.transcript.render(80).join("\n"), /只修改 terminal/);

  app.handleServerEvent({
    type: "run_finished",
    status: "completed",
    run_id: "run_1",
  });
  app.editor.setText("next task");
  app.editor.onSubmit?.(app.editor.getText());
  assert.deepEqual(messages.at(-1), { type: "task", text: "next task" });
});

test("programmatic initial task uses the same task submission path", () => {
  const terminal = new FakeTerminal();
  const messages: unknown[] = [];
  const app = new MiniCodeTuiApp({
    terminal,
    onClientMessage: (message) => messages.push(message),
  });

  app.submitText("  initial task  ");

  assert.deepEqual(messages, [{ type: "task", text: "initial task" }]);
  assert.match(app.transcript.render(80).join("\n"), /initial task/);
});

test("idle ctrl-c clears draft before requesting exit", () => {
  const terminal = new FakeTerminal();
  let exits = 0;
  const app = new MiniCodeTuiApp({
    terminal,
    onExit: () => {
      exits += 1;
    },
  });
  app.start();

  app.editor.setText("draft");
  terminal.sendInput("\x03");
  assert.equal(app.editor.getText(), "");
  assert.equal(exits, 0);

  terminal.sendInput("\x03");
  assert.equal(exits, 1);
  app.stop();
});

test("escape and ctrl-c request cooperative cancel only while running", () => {
  const terminal = new FakeTerminal();
  const messages: unknown[] = [];
  const app = new MiniCodeTuiApp({
    terminal,
    onClientMessage: (message) => messages.push(message),
  });
  app.start();

  terminal.sendInput("\x1b");
  assert.deepEqual(messages, []);

  app.editor.setText("task");
  app.editor.onSubmit?.("task");
  terminal.sendInput("\x1b");
  terminal.sendInput("\x03");

  assert.deepEqual(messages, [
    { type: "task", text: "task" },
    { type: "cancel" },
    { type: "cancel" },
  ]);
  app.stop();
});

test("approval overlay owns input and returns the exact approval id", () => {
  const terminal = new FakeTerminal();
  const messages: unknown[] = [];
  const app = new MiniCodeTuiApp({
    terminal,
    onClientMessage: (message) => messages.push(message),
  });
  app.start();
  app.handleServerEvent({ type: "run_started", run_id: "run_1" });
  app.handleServerEvent({
    type: "approval_required",
    id: "approval_1",
    tool: "run_command",
    summary: "Run focused tests",
    details: "{\"risk_level\":\"high\"}",
  });

  assert.equal(app.editor.disableSubmit, true);
  terminal.sendInput("y");

  assert.deepEqual(messages, [
    {
      type: "approval_response",
      id: "approval_1",
      decision: "approve",
    },
  ]);
  assert.equal(app.editor.disableSubmit, false);
  app.stop();
});

test("user input overlay returns the exact choice index", () => {
  const terminal = new FakeTerminal();
  const messages: unknown[] = [];
  const app = new MiniCodeTuiApp({
    terminal,
    onClientMessage: (message) => messages.push(message),
  });
  app.start();
  app.handleServerEvent({ type: "run_started", run_id: "run_plan" });
  app.handleServerEvent({
    type: "user_input_required",
    id: "input_1",
    question: "Choose compatibility strategy",
    options: [
      { label: "strict", description: "Break old callers" },
      { label: "compat", description: "Keep compatibility" },
    ],
  });

  assert.equal(app.editor.disableSubmit, true);
  terminal.sendInput("2");

  assert.deepEqual(messages, [
    { type: "user_input_response", id: "input_1", selected_index: 1 },
  ]);
  assert.equal(app.editor.disableSubmit, false);
  app.stop();
});

test("post-run user input returns footer to ready after selection", () => {
  const terminal = new FakeTerminal();
  const app = new MiniCodeTuiApp({
    terminal,
    onClientMessage: () => {},
  });
  app.start();
  app.handleServerEvent({ type: "run_started", run_id: "run_plan" });
  app.handleServerEvent({
    type: "run_finished",
    status: "completed",
    run_id: "run_plan",
  });
  app.handleServerEvent({
    type: "user_input_required",
    id: "input_handoff",
    question: "The plan is ready. What should MiniCode do next?",
    options: [
      { label: "Execute plan" },
      { label: "Continue planning" },
      { label: "Finish planning" },
    ],
  });

  terminal.sendInput("2");

  assert.match(app.footer.render(80)[0], /Ready/);
  app.stop();
});

test("cancel during approval sends cancel and waits for backend resolution", () => {
  const terminal = new FakeTerminal();
  const messages: unknown[] = [];
  const app = new MiniCodeTuiApp({
    terminal,
    onClientMessage: (message) => messages.push(message),
  });
  app.start();
  app.handleServerEvent({ type: "run_started", run_id: "run_1" });
  app.handleServerEvent({
    type: "approval_required",
    id: "approval_1",
    tool: "edit",
    summary: "Change src/app.ts",
  });

  terminal.sendInput("\x1b");
  assert.deepEqual(messages, [{ type: "cancel" }]);
  assert.equal(app.editor.disableSubmit, true);

  app.handleServerEvent({
    type: "run_finished",
    status: "cancelled",
    run_id: "run_1",
  });
  assert.equal(app.editor.disableSubmit, false);
  app.stop();
});

test("idle slash command uses command protocol while running slash input stays steering", () => {
  const terminal = new FakeTerminal();
  const messages: unknown[] = [];
  const app = new MiniCodeTuiApp({
    terminal,
    onClientMessage: (message) => messages.push(message),
  });

  app.editor.setText("/help");
  app.editor.onSubmit?.("/help");
  app.editor.setText("normal task");
  app.editor.onSubmit?.("normal task");
  app.editor.setText("/diff");
  app.editor.onSubmit?.("/diff");

  assert.deepEqual(messages, [
    { type: "command", text: "/help" },
    { type: "task", text: "normal task" },
    { type: "steer", text: "/diff" },
  ]);
});

test("exit requested delegates to the app exit callback", () => {
  const terminal = new FakeTerminal();
  let exits = 0;
  const app = new MiniCodeTuiApp({
    terminal,
    onExit: () => { exits += 1; },
  });

  app.handleServerEvent({ type: "exit_requested" });

  assert.equal(exits, 1);
});

test("panel event opens one read-only overlay and escape closes it", () => {
  const terminal = new FakeTerminal();
  const app = new MiniCodeTuiApp({ terminal });
  app.start();

  app.handleServerEvent({
    type: "panel",
    name: "context",
    title: "Context",
    content: "Usage: 7.2k / 56.0k",
  });

  assert.equal(app.tui.hasOverlay(), true);
  terminal.sendInput("\x1b");
  assert.equal(app.tui.hasOverlay(), false);
  app.stop();
});
