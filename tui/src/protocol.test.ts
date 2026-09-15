import assert from "node:assert/strict";
import test from "node:test";

import { parseServerMessage } from "./protocol.js";

test("server JSONL parser preserves unicode and escaped newlines", () => {
  const event = parseServerMessage(
    JSON.stringify({ type: "assistant_delta", text: "中文\n😀" }),
  );

  assert.deepEqual(event, { type: "assistant_delta", text: "中文\n😀" });
});

test("server JSONL parser accepts command catalog metadata", () => {
  const event = parseServerMessage(
    JSON.stringify({
      type: "command_catalog",
      commands: [
        {
          name: "permissions",
          description: "View or change runtime permissions",
          argument_hint: "[mode <value>]",
          argument_choices: ["mode read-only", "mode workspace-write"],
          availability: "idle",
        },
      ],
    }),
  );

  assert.equal(event.type, "command_catalog");
  if (event.type !== "command_catalog") return;
  assert.equal(event.commands[0]?.name, "permissions");
  assert.deepEqual(event.commands[0]?.argument_choices, [
    "mode read-only",
    "mode workspace-write",
  ]);
});

test("server JSONL parser accepts reasoning deltas", () => {
  const event = parseServerMessage(
    JSON.stringify({ type: "reasoning_delta", text: "先分析调用链。\n" }),
  );

  assert.deepEqual(event, {
    type: "reasoning_delta",
    text: "先分析调用链。\n",
  });
});

test("server JSONL parser accepts command lifecycle fields", () => {
  const event = parseServerMessage(
    JSON.stringify({
      type: "tool_finished",
      id: "cmd_call",
      step: 2,
      tool: "run_command",
      status: "command_timed_out",
      summary: "focused test timed out",
      command_status: "timed_out",
      returncode: 124,
      duration_ms: 30000,
      runtime_task_id: null,
    }),
  );

  assert.equal(event.type, "tool_finished");
  if (event.type !== "tool_finished") return;
  assert.equal(event.command_status, "timed_out");
  assert.equal(event.returncode, 124);
  assert.equal(event.duration_ms, 30000);
});

test("server JSONL parser accepts bounded mutation diff fields", () => {
  const event = parseServerMessage(
    JSON.stringify({
      type: "tool_finished",
      id: "edit_1",
      step: 3,
      tool: "edit",
      status: "ok",
      diff_preview: "@@ -1 +1 @@\n-old\n+new",
      diff_truncated: true,
    }),
  );

  assert.equal(event.type, "tool_finished");
  if (event.type !== "tool_finished") return;
  assert.match(event.diff_preview ?? "", /\+new/);
  assert.equal(event.diff_truncated, true);
});

test("server JSONL parser accepts session approval capability and tool call identity", () => {
  const event = parseServerMessage(
    JSON.stringify({
      type: "approval_required",
      id: "approval_1",
      tool_call_id: "call_1",
      tool: "run_command",
      can_approve_session: true,
    }),
  );

  assert.equal(event.type, "approval_required");
  if (event.type !== "approval_required") return;
  assert.equal(event.tool_call_id, "call_1");
  assert.equal(event.can_approve_session, true);
});

test("server JSONL parser accepts bounded user input choices", () => {
  const event = parseServerMessage(
    JSON.stringify({
      type: "user_input_required",
      id: "input_1",
      question: "Choose compatibility strategy",
      options: [
        { label: "strict", description: "Break old callers" },
        { label: "compat", description: "Keep compatibility" },
      ],
    }),
  );

  assert.equal(event.type, "user_input_required");
  if (event.type !== "user_input_required") return;
  assert.equal(event.options.length, 2);
  assert.equal(event.options[1].label, "compat");
});

test("server JSONL parser accepts read-only panel events", () => {
  const event = parseServerMessage(
    JSON.stringify({
      type: "panel",
      name: "runs",
      title: "Runs",
      content: "Run History\n- run_1",
    }),
  );

  assert.deepEqual(event, {
    type: "panel",
    name: "runs",
    title: "Runs",
    content: "Run History\n- run_1",
  });
});

test("server JSONL parser accepts exit requests", () => {
  const event = parseServerMessage(JSON.stringify({ type: "exit_requested" }));
  assert.deepEqual(event, { type: "exit_requested" });
});

test("server JSONL parser rejects unknown, extra, malformed, or multiline records", () => {
  assert.throws(() => parseServerMessage('{"type":"unknown"}'));
  assert.throws(() => parseServerMessage('{"type":"assistant_delta","text":"a","extra":1}'));
  assert.throws(() => parseServerMessage('{"type":"error","message":"x","fatal":"yes"}'));
  assert.throws(() =>
    parseServerMessage('{"type":"context","used":-1,"window":64000,"prompt_budget":56000,"reserved_output":8000}'),
  );
  assert.throws(() =>
    parseServerMessage('{"type":"tool_finished","id":"x","step":1,"tool":"run_command","status":"ok","duration_ms":-1}'),
  );
  assert.throws(() =>
    parseServerMessage('{"type":"assistant_delta","text":"a"}\n{"type":"assistant_delta","text":"b"}'),
  );
});
