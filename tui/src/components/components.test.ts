import assert from "node:assert/strict";
import test from "node:test";

import {
  stripTerminalSequences,
  visibleWidth,
} from "@earendil-works/pi-tui";

import { Activity } from "./activity.js";
import { AssistantMessage } from "./assistant-message.js";
import { Footer } from "./footer.js";
import { Header } from "./header.js";
import { ReasoningMessage } from "./reasoning-message.js";
import { UserMessage } from "./user-message.js";

const CASES = [
  String.raw`C:\workspace\project\very\long\windows\path\with\中文\file.py`,
  "中文输入和 Emoji 😀🚀",
  "averyveryveryveryveryveryveryveryverylongword",
  "\x1b[31mANSI command output\x1b[0m",
  "pytest tests/terminal/test_layout.py --maxfail=1 --disable-warnings --very-long-option=value",
];

test("custom components never render wider than the requested width", () => {
  for (const width of [4, 8, 12, 20, 40]) {
    const header = new Header();
    header.setSession(CASES[0]);

    const footer = new Footer();
    footer.setText(CASES[1]);

    const activity = new Activity();
    activity.start("run_1", "Run", CASES[2]);

    const assistant = new AssistantMessage();
    assistant.appendDelta(CASES.join("\n"));

    const reasoning = new ReasoningMessage();
    reasoning.appendDelta(CASES.join("\n"));

    const components = [
      header,
      footer,
      activity,
      assistant,
      reasoning,
      ...CASES.map((value) => new UserMessage(value)),
    ];

    for (const component of components) {
      for (const line of component.render(width)) {
        assert.ok(
          visibleWidth(line) <= width,
          `width=${width}, actual=${visibleWidth(line)}, line=${JSON.stringify(line)}`,
        );
      }
    }
  }
});

test("assistant message reuses one leaf while streaming deltas", () => {
  const assistant = new AssistantMessage();

  assistant.appendDelta("## 标题\n");
  assistant.appendDelta("流式 **Markdown** 😀");

  assert.equal(assistant.text, "## 标题\n流式 **Markdown** 😀");
  const rendered = assistant.render(24).join("\n");
  assert.match(rendered, /标题/);
  assert.match(rendered, /流式/);
});

test("reasoning is compact by default and expands on demand", () => {
  const reasoning = new ReasoningMessage();
  reasoning.appendDelta(
    "先检查入口和调用关系。\n\n第二段展开后可见 UNIQUE_REASONING_DETAIL。",
  );

  const collapsed = stripTerminalSequences(reasoning.render(32).join("\n"));
  assert.match(collapsed, /Thinking/);
  assert.doesNotMatch(collapsed, /UNIQUE_REASONING_DETAIL/);

  reasoning.setExpanded(true);
  const expanded = stripTerminalSequences(reasoning.render(80).join("\n"));
  assert.match(expanded, /UNIQUE_REASONING_DETAIL/);

  reasoning.finish();
  assert.match(
    stripTerminalSequences(reasoning.render(80).join("\n")),
    /Thought/,
  );
});

test("activity mutates from working to worked without creating a second component", () => {
  const activity = new Activity();

  activity.start("read_1", "Explore", "loop.py");
  activity.finish("read_1", "duplicate_reused");
  activity.start("edit_1", "Change", "app.ts");
  const running = activity.render(48).join("\n");
  assert.match(running, /Working/);
  assert.match(running, /loop.py/);
  assert.match(running, /app.ts/);

  activity.complete();
  assert.match(
    stripTerminalSequences(activity.render(48).join("\n")),
    /Worked · 2 actions/,
  );
});

test("activity renders running command elapsed time and pauses for approval", () => {
  const originalNow = Date.now;
  let now = 1_000;
  Date.now = () => now;
  try {
    const activity = new Activity();
    activity.start("cmd_live", "Run", "git branch --show-current", true);
    now = 4_500;
    let rendered = stripTerminalSequences(activity.render(100).join("\n"));
    assert.match(rendered, /3\.5s elapsed/);

    activity.pause("cmd_live", "waiting for approval");
    now = 9_500;
    rendered = stripTerminalSequences(activity.render(100).join("\n"));
    assert.match(rendered, /waiting for approval/);
    assert.doesNotMatch(rendered, /8\.5s elapsed/);

    activity.resume("cmd_live");
    now = 12_000;
    rendered = stripTerminalSequences(activity.render(100).join("\n"));
    assert.match(rendered, /2\.5s elapsed/);
  } finally {
    Date.now = originalNow;
  }
});

test("activity clears approval wait detail after tool finishes", () => {
  const activity = new Activity();
  activity.start("edit_wait", "Change", "src/app.ts");
  activity.pause("edit_wait", "waiting for approval");
  activity.finish("edit_wait", "ok");

  const rendered = stripTerminalSequences(activity.render(100).join("\n"));
  assert.doesNotMatch(rendered, /waiting for approval/);
  assert.match(rendered, /src\/app\.ts/);
});

test("activity completion clears pending approval detail", () => {
  const activity = new Activity();
  activity.start("edit_cancelled", "Change", "src/app.ts");
  activity.pause("edit_cancelled", "waiting for approval");
  activity.complete();

  const rendered = stripTerminalSequences(activity.render(100).join("\n"));
  assert.doesNotMatch(rendered, /waiting for approval/);
});

test("activity renders command lifecycle without replacing command identity", () => {
  const activity = new Activity();

  activity.start("cmd_1", "Run", "python -m pytest -q tests/test_a.py");
  activity.finish("cmd_1", "command_timed_out", "timeout details", {
    commandStatus: "timed_out",
    returncode: 124,
    durationMs: 30_000,
  });
  const timedOut = stripTerminalSequences(activity.render(100).join("\n"));
  assert.match(timedOut, /python -m pytest -q tests\/test_a.py/);
  assert.match(timedOut, /timed out · 30s/);

  activity.start("cmd_2", "Run", "python worker.py");
  activity.finish("cmd_2", "background_started", null, {
    commandStatus: "background_started",
    runtimeTaskId: "cmd_0001",
  });
  const background = stripTerminalSequences(activity.render(100).join("\n"));
  assert.match(background, /background cmd_0001/);
});

test("activity renders successful mutation diff inline and hides failed preview", () => {
  const activity = new Activity();

  activity.start("edit_1", "Change", "app.py");
  activity.finish("edit_1", "ok", null, {
    diffPreview: "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new",
    diffTruncated: true,
  });
  const rendered = stripTerminalSequences(activity.render(80).join("\n"));
  assert.match(rendered, /-old/);
  assert.match(rendered, /\+new/);
  assert.match(rendered, /diff preview truncated/);

  const failed = new Activity();
  failed.start("edit_2", "Change", "app.py");
  failed.finish("edit_2", "error", "edit failed", {
    diffPreview: "-should-not-render\n+unsafe-preview",
  });
  const failedRendered = stripTerminalSequences(failed.render(80).join("\n"));
  assert.doesNotMatch(failedRendered, /should-not-render/);
  assert.match(failedRendered, /edit failed/);
});

test("activity status markers stay ASCII-safe for Windows terminals", () => {
  const activity = new Activity();

  activity.start("read_1", "Explore", "README.md");
  const running = stripTerminalSequences(activity.render(48).join("\n"));
  assert.match(running, /> Working/);
  assert.match(running, /> Explore/);
  assert.doesNotMatch(running, /[●✓›✕]/);

  activity.finish("read_1", "ok");
  activity.complete();
  const worked = stripTerminalSequences(activity.render(48).join("\n"));
  assert.match(worked, /\+ Worked/);
  assert.match(worked, /\+ Explore/);
  assert.doesNotMatch(worked, /[●✓›✕]/);
});
