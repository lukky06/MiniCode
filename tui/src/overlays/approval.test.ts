import assert from "node:assert/strict";
import test from "node:test";

import { stripTerminalSequences, visibleWidth } from "@earendil-works/pi-tui";

import { ApprovalOverlay } from "./approval.js";

test("approval overlay stays within narrow widths and toggles Python-provided details", () => {
  const decisions: string[] = [];
  const overlay = new ApprovalOverlay(
    {
      type: "approval_required",
      id: "approval_1",
      tool: "run_command",
      summary: "运行一个非常长的聚焦测试命令 😀",
      can_approve_session: true,
      details: JSON.stringify({
        risk_level: "high",
        preview: {
          command:
            "pytest tests/terminal/test_layout.py --maxfail=1 --disable-warnings",
          reason: "Command needs explicit approval.",
          effects: ["may create workspace-local test artifacts"],
        },
      }),
    },
    (decision) => decisions.push(decision),
  );

  const compact = stripTerminalSequences(overlay.render(100).join("\n"));
  assert.match(compact, /MiniCode wants to run a command/);
  assert.match(compact, /pytest tests\/terminal\/test_layout\.py/);
  assert.match(compact, /Risk\s+HIGH/);
  assert.match(compact, /Reason\s+Command needs explicit approval/);
  assert.match(compact, /Allow once/);
  assert.match(compact, /Allow for this session/);
  assert.match(compact, /Reject/);
  assert.doesNotMatch(compact, /risk_level/);

  overlay.handleInput("v");
  const detailed = stripTerminalSequences(overlay.render(100).join("\n"));
  assert.match(detailed, /risk_level/);
  for (const width of [4, 8, 20, 40]) {
    for (const line of overlay.render(width)) {
      assert.ok(visibleWidth(line) <= width);
    }
  }

  overlay.handleInput("g");
  overlay.handleInput("s");
  assert.deepEqual(decisions, ["approve_session", "skip"]);
});
