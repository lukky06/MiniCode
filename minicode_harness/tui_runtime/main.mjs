// src/backend-client.ts
import {
  spawn
} from "node:child_process";
import { createInterface } from "node:readline";

// src/protocol.ts
var SERVER_TYPES = /* @__PURE__ */ new Set([
  "session_started",
  "run_started",
  "context",
  "tool_started",
  "tool_finished",
  "assistant_delta",
  "reasoning_delta",
  "approval_required",
  "user_input_required",
  "run_finished",
  "error",
  "panel",
  "exit_requested"
]);
function parseServerMessage(line) {
  if (line.includes("\n") || line.includes("\r")) {
    throw new Error("JSONL record must occupy one physical line");
  }
  const value = JSON.parse(line);
  if (!isRecord(value) || typeof value.type !== "string" || !SERVER_TYPES.has(value.type)) {
    throw new Error("Unknown server message");
  }
  validateRequiredFields(value);
  return value;
}
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function validateRequiredFields(value) {
  switch (value.type) {
    case "session_started":
      exactKeys(value, ["type", "session_id"]);
      requireString(value, "session_id");
      return;
    case "run_started":
      exactKeys(value, ["type", "run_id"]);
      requireString(value, "run_id");
      return;
    case "context":
      exactKeys(value, ["type", "used", "window", "prompt_budget", "reserved_output"]);
      requireNonNegative(value, "used");
      requirePositive(value, "window");
      requirePositive(value, "prompt_budget");
      requireNonNegative(value, "reserved_output");
      return;
    case "tool_started":
      exactKeys(value, ["type", "id", "step", "tool", "target"]);
      requireString(value, "id");
      requireNonNegative(value, "step");
      requireString(value, "tool");
      requireOptionalString(value, "target");
      return;
    case "tool_finished":
      exactKeys(value, [
        "type",
        "id",
        "step",
        "tool",
        "status",
        "summary",
        "command_status",
        "returncode",
        "duration_ms",
        "runtime_task_id",
        "diff_preview",
        "diff_truncated"
      ]);
      requireString(value, "id");
      requireNonNegative(value, "step");
      requireString(value, "tool");
      requireString(value, "status");
      requireOptionalString(value, "summary");
      requireOptionalString(value, "command_status");
      requireOptionalInteger(value, "returncode");
      requireOptionalNonNegativeInteger(value, "duration_ms");
      requireOptionalString(value, "runtime_task_id");
      requireOptionalString(value, "diff_preview");
      if (value.diff_truncated !== void 0 && typeof value.diff_truncated !== "boolean") {
        throw new Error("Expected boolean field: diff_truncated");
      }
      return;
    case "assistant_delta":
    case "reasoning_delta":
      exactKeys(value, ["type", "text"]);
      requireString(value, "text");
      return;
    case "approval_required":
      exactKeys(value, [
        "type",
        "id",
        "tool",
        "summary",
        "details",
        "can_approve_session"
      ]);
      requireString(value, "id");
      requireString(value, "tool");
      requireOptionalString(value, "summary");
      requireOptionalString(value, "details");
      if (value.can_approve_session !== void 0 && typeof value.can_approve_session !== "boolean") {
        throw new Error("Expected boolean field: can_approve_session");
      }
      return;
    case "user_input_required":
      exactKeys(value, ["type", "id", "question", "options"]);
      requireString(value, "id");
      requireString(value, "question");
      if (!Array.isArray(value.options) || value.options.length < 2 || value.options.length > 4) {
        throw new Error("Expected 2-4 user input options");
      }
      for (const option of value.options) {
        if (!isRecord(option)) {
          throw new Error("Expected user input option object");
        }
        exactKeys(option, ["label", "description"]);
        requireString(option, "label");
        requireOptionalString(option, "description");
      }
      return;
    case "run_finished":
      exactKeys(value, ["type", "status", "run_id", "stop_reason"]);
      requireString(value, "status");
      requireOptionalString(value, "run_id");
      requireOptionalString(value, "stop_reason");
      return;
    case "error":
      exactKeys(value, ["type", "message", "fatal"]);
      requireString(value, "message");
      if (value.fatal !== void 0 && typeof value.fatal !== "boolean") {
        throw new Error("Expected boolean field: fatal");
      }
      return;
    case "panel":
      exactKeys(value, ["type", "name", "title", "content"]);
      requireString(value, "name");
      requireString(value, "title");
      requireString(value, "content");
      return;
    case "exit_requested":
      exactKeys(value, ["type"]);
      return;
    default:
      throw new Error("Unknown server message");
  }
}
function exactKeys(value, allowed) {
  const allowedKeys = new Set(allowed);
  const extra = Object.keys(value).find((key) => !allowedKeys.has(key));
  if (extra !== void 0) {
    throw new Error(`Unexpected field: ${extra}`);
  }
}
function requireString(value, key) {
  if (typeof value[key] !== "string") {
    throw new Error(`Expected string field: ${key}`);
  }
}
function requireOptionalString(value, key) {
  const field = value[key];
  if (field !== void 0 && field !== null && typeof field !== "string") {
    throw new Error(`Expected optional string field: ${key}`);
  }
}
function requireOptionalInteger(value, key) {
  const field = value[key];
  if (field !== void 0 && field !== null && (typeof field !== "number" || !Number.isInteger(field))) {
    throw new Error(`Expected optional integer field: ${key}`);
  }
}
function requireOptionalNonNegativeInteger(value, key) {
  requireOptionalInteger(value, key);
  const field = value[key];
  if (typeof field === "number" && field < 0) {
    throw new Error(`Expected optional non-negative integer field: ${key}`);
  }
}
function requireNonNegative(value, key) {
  requireNumber(value, key);
  if (value[key] < 0) {
    throw new Error(`Expected non-negative field: ${key}`);
  }
}
function requirePositive(value, key) {
  requireNumber(value, key);
  if (value[key] <= 0) {
    throw new Error(`Expected positive field: ${key}`);
  }
}
function requireNumber(value, key) {
  if (typeof value[key] !== "number" || !Number.isFinite(value[key])) {
    throw new Error(`Expected numeric field: ${key}`);
  }
}

// src/backend-client.ts
var BackendClient = class {
  constructor(options) {
    this.options = options;
  }
  options;
  child = null;
  start(handlers) {
    if (this.child !== null) {
      throw new Error("MiniCode backend is already running.");
    }
    const child = spawn(
      this.options.pythonExecutable,
      buildBackendArgs(this.options),
      {
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true
      }
    );
    this.child = child;
    const lines = createInterface({ input: child.stdout });
    lines.on("line", (line) => {
      try {
        handlers.onEvent(parseServerMessage(line));
      } catch (error) {
        handlers.onProtocolError?.(
          error instanceof Error ? error.message : String(error)
        );
      }
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      handlers.onDebug?.(chunk);
    });
    child.on("exit", (code, signal) => {
      lines.close();
      this.child = null;
      handlers.onExit?.(code, signal);
    });
  }
  send(message) {
    const child = this.child;
    if (child === null || child.stdin.destroyed || !child.stdin.writable) {
      throw new Error("MiniCode backend is not writable.");
    }
    child.stdin.write(encodeClientMessage(message));
  }
  close() {
    const child = this.child;
    if (child === null) return;
    if (!child.stdin.destroyed && child.stdin.writable) {
      child.stdin.end();
    }
  }
};
function encodeClientMessage(message) {
  return JSON.stringify(message) + "\n";
}
function buildBackendArgs(options) {
  const args = [
    "-m",
    "minicode_harness.tui_bridge.backend",
    "--workspace",
    options.workspace
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

// node_modules/marked/lib/marked.esm.js
function M() {
  return { async: false, breaks: false, extensions: null, gfm: true, hooks: null, pedantic: false, renderer: null, silent: false, tokenizer: null, walkTokens: null };
}
var T = M();
function N(l3) {
  T = l3;
}
var _ = { exec: () => null };
function E(l3) {
  let e = [];
  return (t) => {
    let n = Math.max(0, Math.min(3, t - 1)), s = e[n];
    return s || (s = l3(n), e[n] = s), s;
  };
}
function d(l3, e = "") {
  let t = typeof l3 == "string" ? l3 : l3.source, n = { replace: (s, r) => {
    let i = typeof r == "string" ? r : r.source;
    return i = i.replace(m.caret, "$1"), t = t.replace(s, i), n;
  }, getRegex: () => new RegExp(t, e) };
  return n;
}
var Te = ((l3 = "") => {
  try {
    return !!new RegExp("(?<=1)(?<!1)" + l3);
  } catch {
    return false;
  }
})();
var m = { codeRemoveIndent: /^(?: {1,4}| {0,3}\t)/gm, outputLinkReplace: /\\([\[\]])/g, indentCodeCompensation: /^(\s+)(?:```)/, beginningSpace: /^\s+/, endingHash: /#$/, startingSpaceChar: /^ /, endingSpaceChar: / $/, nonSpaceChar: /[^ ]/, newLineCharGlobal: /\n/g, tabCharGlobal: /\t/g, multipleSpaceGlobal: /\s+/g, blankLine: /^[ \t]*$/, doubleBlankLine: /\n[ \t]*\n[ \t]*$/, blockquoteStart: /^ {0,3}>/, blockquoteSetextReplace: /\n {0,3}((?:=+|-+) *)(?=\n|$)/g, blockquoteSetextReplace2: /^ {0,3}>[ \t]?/gm, listReplaceNesting: /^ {1,4}(?=( {4})*[^ ])/g, listIsTask: /^\[[ xX]\] +\S/, listReplaceTask: /^\[[ xX]\] +/, listTaskCheckbox: /\[[ xX]\]/, anyLine: /\n.*\n/, hrefBrackets: /^<(.*)>$/, tableDelimiter: /[:|]/, tableAlignChars: /^\||\| *$/g, tableRowBlankLine: /\n[ \t]*$/, tableAlignRight: /^ *-+: *$/, tableAlignCenter: /^ *:-+: *$/, tableAlignLeft: /^ *:-+ *$/, startATag: /^<a /i, endATag: /^<\/a>/i, startPreScriptTag: /^<(pre|code|kbd|script)(\s|>)/i, endPreScriptTag: /^<\/(pre|code|kbd|script)(\s|>)/i, startAngleBracket: /^</, endAngleBracket: />$/, pedanticHrefTitle: /^([^'"]*[^\s])\s+(['"])(.*)\2/, unicodeAlphaNumeric: /[\p{L}\p{N}]/u, escapeTest: /[&<>"']/, escapeReplace: /[&<>"']/g, escapeTestNoEncode: /[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/, escapeReplaceNoEncode: /[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/g, caret: /(^|[^\[])\^/g, percentDecode: /%25/g, findPipe: /\|/g, splitPipe: / \|/, slashPipe: /\\\|/g, carriageReturn: /\r\n|\r/g, spaceLine: /^ +$/gm, notSpaceStart: /^\S*/, endingNewline: /\n$/, listItemRegex: (l3) => new RegExp(`^( {0,3}${l3})((?:[	 ][^\\n]*)?(?:\\n|$))`), nextBulletRegex: E((l3) => new RegExp(`^ {0,${l3}}(?:[*+-]|\\d{1,9}[.)])((?:[ 	][^\\n]*)?(?:\\n|$))`)), hrRegex: E((l3) => new RegExp(`^ {0,${l3}}((?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$)`)), fencesBeginRegex: E((l3) => new RegExp(`^ {0,${l3}}(?:\`\`\`|~~~)`)), headingBeginRegex: E((l3) => new RegExp(`^ {0,${l3}}#`)), htmlBeginRegex: E((l3) => new RegExp(`^ {0,${l3}}<(?:[a-z].*>|!--)`, "i")), blockquoteBeginRegex: E((l3) => new RegExp(`^ {0,${l3}}>`)) };
var Oe = /^(?:[ \t]*(?:\n|$))+/;
var we = /^((?: {4}| {0,3}\t)[^\n]+(?:\n(?:[ \t]*(?:\n|$))*)?)+/;
var ye = /^ {0,3}(`{3,}(?=[^`\n]*(?:\n|$))|~{3,})([^\n]*)(?:\n|$)(?:|([\s\S]*?)(?:\n|$))(?: {0,3}\1[~`]* *(?=\n|$)|$)/;
var B = /^ {0,3}((?:-[\t ]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})(?:\n+|$)/;
var Pe = /^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/;
var j = / {0,3}(?:[*+-]|\d{1,9}[.)])/;
var oe = /^(?!bull |blockCode|fences|blockquote|heading|html|table)((?:.|\n(?!\s*?\n|bull |blockCode|fences|blockquote|heading|html|table))+?)\n {0,3}(=+|-+) *(?:\n+|$)/;
var ae = d(oe).replace(/bull/g, j).replace(/blockCode/g, /(?: {4}| {0,3}\t)/).replace(/fences/g, / {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g, / {0,3}>/).replace(/heading/g, / {0,3}#{1,6}/).replace(/html/g, / {0,3}<[^\n>]+>\n/).replace(/\|table/g, "").getRegex();
var Se = d(oe).replace(/bull/g, j).replace(/blockCode/g, /(?: {4}| {0,3}\t)/).replace(/fences/g, / {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g, / {0,3}>/).replace(/heading/g, / {0,3}#{1,6}/).replace(/html/g, / {0,3}<[^\n>]+>\n/).replace(/table/g, / {0,3}\|?(?:[:\- ]*\|)+[\:\- ]*\n/).getRegex();
var F = /^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/;
var $e = /^[^\n]+/;
var U = /(?!\s*\])(?:\\[\s\S]|[^\[\]\\])+/;
var Le = d(/^ {0,3}\[(label)\]: *(?:\n[ \t]*)?([^<\s][^\s]*|<.*?>)(?:(?: +(?:\n[ \t]*)?| *\n[ \t]*)(title))? *(?:\n+|$)/).replace("label", U).replace("title", /(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/).getRegex();
var _e = d(/^(bull)([ \t][^\n]*?)?(?:\n|$)/).replace(/bull/g, j).getRegex();
var H = "address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul";
var K = /<!--(?:-?>|[\s\S]*?(?:-->|$))/;
var ze = d("^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n[ 	]*)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ 	]*)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ 	]*)+\\n|$))", "i").replace("comment", K).replace("tag", H).replace("attribute", / +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/).getRegex();
var le = d(F).replace("hr", B).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("|table", "").replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)])[ \\t]+[^ \\t\\n]").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", H).getRegex();
var Me = d(/^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/).replace("paragraph", le).getRegex();
var W = { blockquote: Me, code: we, def: Le, fences: ye, heading: Pe, hr: B, html: ze, lheading: ae, list: _e, newline: Oe, paragraph: le, table: _, text: $e };
var se = d("^ *([^\\n ].*)\\n {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)").replace("hr", B).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("blockquote", " {0,3}>").replace("code", "(?: {4}| {0,3}	)[^\\n]").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)])[ \\t]").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", H).getRegex();
var Ee = { ...W, lheading: Se, table: se, paragraph: d(F).replace("hr", B).replace("heading", " {0,3}#{1,6}(?:\\s|$)").replace("|lheading", "").replace("table", se).replace("blockquote", " {0,3}>").replace("fences", " {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list", " {0,3}(?:[*+-]|1[.)])[ \\t]+[^ \\t\\n]").replace("html", "</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag", H).getRegex() };
var Ie = { ...W, html: d(`^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:"[^"]*"|'[^']*'|\\s[^'"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))`).replace("comment", K).replace(/tag/g, "(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b").getRegex(), def: /^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/, heading: /^(#{1,6})(.*)(?:\n+|$)/, fences: _, lheading: /^(.+?)\n {0,3}(=+|-+) *(?:\n+|$)/, paragraph: d(F).replace("hr", B).replace("heading", ` *#{1,6} *[^
]`).replace("lheading", ae).replace("|table", "").replace("blockquote", " {0,3}>").replace("|fences", "").replace("|list", "").replace("|html", "").replace("|tag", "").getRegex() };
var Ae = /^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/;
var Ce = /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/;
var ue = /^( {2,}|\\)\n(?!\s*$)/;
var Be = /^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/;
var I = /[\p{P}\p{S}]/u;
var Z = /[\s\p{P}\p{S}]/u;
var X = /[^\s\p{P}\p{S}]/u;
var De = d(/^((?![*_])punctSpace)/, "u").replace(/punctSpace/g, Z).getRegex();
var pe = /(?!~)[\p{P}\p{S}]/u;
var qe = /(?!~)[\s\p{P}\p{S}]/u;
var ve = /(?:[^\s\p{P}\p{S}]|~)/u;
var He = d(/link|precode-code|html/, "g").replace("link", /\[(?:[^\[\]`]|(?<a>`+)[^`]+\k<a>(?!`))*?\]\((?:\\[\s\S]|[^\\\(\)]|\((?:\\[\s\S]|[^\\\(\)])*\))*\)/).replace("precode-", Te ? "(?<!`)()" : "(^^|[^`])").replace("code", /(?<b>`+)[^`]+\k<b>(?!`)/).replace("html", /<(?! )[^<>]*?>/).getRegex();
var ce = /^(?:\*+(?:((?!\*)punct)|([^\s*]))?)|^_+(?:((?!_)punct)|([^\s_]))?/;
var Ze = d(ce, "u").replace(/punct/g, I).getRegex();
var Ge = d(ce, "u").replace(/punct/g, pe).getRegex();
var he = "^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)punct(\\*+)(?=[\\s]|$)|notPunctSpace(\\*+)(?!\\*)(?=punctSpace|$)|(?!\\*)punctSpace(\\*+)(?=notPunctSpace)|[\\s](\\*+)(?!\\*)(?=punct)|(?!\\*)punct(\\*+)(?!\\*)(?=punct)|notPunctSpace(\\*+)(?=notPunctSpace)";
var Ne = d(he, "gu").replace(/notPunctSpace/g, X).replace(/punctSpace/g, Z).replace(/punct/g, I).getRegex();
var Qe = d(he, "gu").replace(/notPunctSpace/g, ve).replace(/punctSpace/g, qe).replace(/punct/g, pe).getRegex();
var je = d("^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)punct(_+)(?=[\\s]|$)|notPunctSpace(_+)(?!_)(?=punctSpace|$)|(?!_)punctSpace(_+)(?=notPunctSpace)|[\\s](_+)(?!_)(?=punct)|(?!_)punct(_+)(?!_)(?=punct)", "gu").replace(/notPunctSpace/g, X).replace(/punctSpace/g, Z).replace(/punct/g, I).getRegex();
var Fe = d(/^~~?(?:((?!~)punct)|[^\s~])/, "u").replace(/punct/g, I).getRegex();
var Ue = "^[^~]+(?=[^~])|(?!~)punct(~~?)(?=[\\s]|$)|notPunctSpace(~~?)(?!~)(?=punctSpace|$)|(?!~)punctSpace(~~?)(?=notPunctSpace)|[\\s](~~?)(?!~)(?=punct)|(?!~)punct(~~?)(?!~)(?=punct)|notPunctSpace(~~?)(?=notPunctSpace)";
var Ke = d(Ue, "gu").replace(/notPunctSpace/g, X).replace(/punctSpace/g, Z).replace(/punct/g, I).getRegex();
var We = d(/\\(punct)/, "gu").replace(/punct/g, I).getRegex();
var Xe = d(/^<(scheme:[^\s\x00-\x1f<>]*|email)>/).replace("scheme", /[a-zA-Z][a-zA-Z0-9+.-]{1,31}/).replace("email", /[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/).getRegex();
var Je = d(K).replace("(?:-->|$)", "-->").getRegex();
var Ve = d("^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>").replace("comment", Je).replace("attribute", /\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/).getRegex();
var v = /(?:\[(?:\\[\s\S]|[^\[\]\\])*\]|\\[\s\S]|`+(?!`)[^`]*?`+(?!`)|``+(?=\])|[^\[\]\\`])*?/;
var Ye = d(/^!?\[(label)\]\(\s*(href)(?:(?:[ \t]+(?:\n[ \t]*)?|\n[ \t]*)(title))?\s*\)/).replace("label", v).replace("href", /<(?:\\.|[^\n<>\\])+>|[^ \t\n\x00-\x1f]*/).replace("title", /"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/).getRegex();
var ke = d(/^!?\[(label)\]\[(ref)\]/).replace("label", v).replace("ref", U).getRegex();
var de = d(/^!?\[(ref)\](?:\[\])?/).replace("ref", U).getRegex();
var et = d("reflink|nolink(?!\\()", "g").replace("reflink", ke).replace("nolink", de).getRegex();
var ie = /[hH][tT][tT][pP][sS]?|[fF][tT][pP]/;
var J = { _backpedal: _, anyPunctuation: We, autolink: Xe, blockSkip: He, br: ue, code: Ce, del: _, delLDelim: _, delRDelim: _, emStrongLDelim: Ze, emStrongRDelimAst: Ne, emStrongRDelimUnd: je, escape: Ae, link: Ye, nolink: de, punctuation: De, reflink: ke, reflinkSearch: et, tag: Ve, text: Be, url: _ };
var tt = { ...J, link: d(/^!?\[(label)\]\((.*?)\)/).replace("label", v).getRegex(), reflink: d(/^!?\[(label)\]\s*\[([^\]]*)\]/).replace("label", v).getRegex() };
var Q = { ...J, emStrongRDelimAst: Qe, emStrongLDelim: Ge, delLDelim: Fe, delRDelim: Ke, url: d(/^((?:protocol):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/).replace("protocol", ie).replace("email", /[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/).getRegex(), _backpedal: /(?:[^?!.,:;*_'"~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_'"~)]+(?!$))+/, del: /^(~~?)(?=[^\s~])((?:\\[\s\S]|[^\\])*?(?:\\[\s\S]|[^\s~\\]))\1(?=[^~]|$)/, text: d(/^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|protocol:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/).replace("protocol", ie).getRegex() };
var nt = { ...Q, br: d(ue).replace("{2,}", "*").getRegex(), text: d(Q.text).replace("\\b_", "\\b_| {2,}\\n").replace(/\{2,\}/g, "*").getRegex() };
var D = { normal: W, gfm: Ee, pedantic: Ie };
var A = { normal: J, gfm: Q, breaks: nt, pedantic: tt };
var rt = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
var ge = (l3) => rt[l3];
function O(l3, e) {
  if (e) {
    if (m.escapeTest.test(l3)) return l3.replace(m.escapeReplace, ge);
  } else if (m.escapeTestNoEncode.test(l3)) return l3.replace(m.escapeReplaceNoEncode, ge);
  return l3;
}
function V(l3) {
  try {
    l3 = encodeURI(l3).replace(m.percentDecode, "%");
  } catch {
    return null;
  }
  return l3;
}
function Y(l3, e) {
  let t = l3.replace(m.findPipe, (r, i, o) => {
    let u = false, a = i;
    for (; --a >= 0 && o[a] === "\\"; ) u = !u;
    return u ? "|" : " |";
  }), n = t.split(m.splitPipe), s = 0;
  if (n[0].trim() || n.shift(), n.length > 0 && !n.at(-1)?.trim() && n.pop(), e) if (n.length > e) n.splice(e);
  else for (; n.length < e; ) n.push("");
  for (; s < n.length; s++) n[s] = n[s].trim().replace(m.slashPipe, "|");
  return n;
}
function $(l3, e, t) {
  let n = l3.length;
  if (n === 0) return "";
  let s = 0;
  for (; s < n; ) {
    let r = l3.charAt(n - s - 1);
    if (r === e && !t) s++;
    else if (r !== e && t) s++;
    else break;
  }
  return l3.slice(0, n - s);
}
function ee(l3) {
  let e = l3.split(`
`), t = e.length - 1;
  for (; t >= 0 && m.blankLine.test(e[t]); ) t--;
  return e.length - t <= 2 ? l3 : e.slice(0, t + 1).join(`
`);
}
function fe(l3, e) {
  if (l3.indexOf(e[1]) === -1) return -1;
  let t = 0;
  for (let n = 0; n < l3.length; n++) if (l3[n] === "\\") n++;
  else if (l3[n] === e[0]) t++;
  else if (l3[n] === e[1] && (t--, t < 0)) return n;
  return t > 0 ? -2 : -1;
}
function me(l3, e = 0) {
  let t = e, n = "";
  for (let s of l3) if (s === "	") {
    let r = 4 - t % 4;
    n += " ".repeat(r), t += r;
  } else n += s, t++;
  return n;
}
function xe(l3, e, t, n, s) {
  let r = e.href, i = e.title || null, o = l3[1].replace(s.other.outputLinkReplace, "$1");
  n.state.inLink = true;
  let u = { type: l3[0].charAt(0) === "!" ? "image" : "link", raw: t, href: r, title: i, text: o, tokens: n.inlineTokens(o) };
  return n.state.inLink = false, u;
}
function st(l3, e, t) {
  let n = l3.match(t.other.indentCodeCompensation);
  if (n === null) return e;
  let s = n[1];
  return e.split(`
`).map((r) => {
    let i = r.match(t.other.beginningSpace);
    if (i === null) return r;
    let [o] = i;
    return o.length >= s.length ? r.slice(s.length) : r;
  }).join(`
`);
}
var w = class {
  options;
  rules;
  lexer;
  constructor(e) {
    this.options = e || T;
  }
  space(e) {
    let t = this.rules.block.newline.exec(e);
    if (t && t[0].length > 0) return { type: "space", raw: t[0] };
  }
  code(e) {
    let t = this.rules.block.code.exec(e);
    if (t) {
      let n = this.options.pedantic ? t[0] : ee(t[0]), s = n.replace(this.rules.other.codeRemoveIndent, "");
      return { type: "code", raw: n, codeBlockStyle: "indented", text: s };
    }
  }
  fences(e) {
    let t = this.rules.block.fences.exec(e);
    if (t) {
      let n = t[0], s = st(n, t[3] || "", this.rules);
      return { type: "code", raw: n, lang: t[2] ? t[2].trim().replace(this.rules.inline.anyPunctuation, "$1") : t[2], text: s };
    }
  }
  heading(e) {
    let t = this.rules.block.heading.exec(e);
    if (t) {
      let n = t[2].trim();
      if (this.rules.other.endingHash.test(n)) {
        let s = $(n, "#");
        (this.options.pedantic || !s || this.rules.other.endingSpaceChar.test(s)) && (n = s.trim());
      }
      return { type: "heading", raw: $(t[0], `
`), depth: t[1].length, text: n, tokens: this.lexer.inline(n) };
    }
  }
  hr(e) {
    let t = this.rules.block.hr.exec(e);
    if (t) return { type: "hr", raw: $(t[0], `
`) };
  }
  blockquote(e) {
    let t = this.rules.block.blockquote.exec(e);
    if (t) {
      let n = $(t[0], `
`).split(`
`), s = "", r = "", i = [];
      for (; n.length > 0; ) {
        let o = false, u = [], a;
        for (a = 0; a < n.length; a++) if (this.rules.other.blockquoteStart.test(n[a])) u.push(n[a]), o = true;
        else if (!o) u.push(n[a]);
        else break;
        n = n.slice(a);
        let c = u.join(`
`), p = c.replace(this.rules.other.blockquoteSetextReplace, `
    $1`).replace(this.rules.other.blockquoteSetextReplace2, "");
        s = s ? `${s}
${c}` : c, r = r ? `${r}
${p}` : p;
        let k = this.lexer.state.top;
        if (this.lexer.state.top = true, this.lexer.blockTokens(p, i, true), this.lexer.state.top = k, n.length === 0) break;
        let h = i.at(-1);
        if (h?.type === "code") break;
        if (h?.type === "blockquote") {
          let R = h, f = R.raw + `
` + n.join(`
`), S = this.blockquote(f);
          i[i.length - 1] = S, s = s.substring(0, s.length - R.raw.length) + S.raw, r = r.substring(0, r.length - R.text.length) + S.text;
          break;
        } else if (h?.type === "list") {
          let R = h, f = R.raw + `
` + n.join(`
`), S = this.list(f);
          i[i.length - 1] = S, s = s.substring(0, s.length - h.raw.length) + S.raw, r = r.substring(0, r.length - R.raw.length) + S.raw, n = f.substring(i.at(-1).raw.length).split(`
`);
          continue;
        }
      }
      return { type: "blockquote", raw: s, tokens: i, text: r };
    }
  }
  list(e) {
    let t = this.rules.block.list.exec(e);
    if (t) {
      let n = t[1].trim(), s = n.length > 1, r = { type: "list", raw: "", ordered: s, start: s ? +n.slice(0, -1) : "", loose: false, items: [] };
      n = s ? `\\d{1,9}\\${n.slice(-1)}` : `\\${n}`, this.options.pedantic && (n = s ? n : "[*+-]");
      let i = this.rules.other.listItemRegex(n), o = false;
      for (; e; ) {
        let a = false, c = "", p = "";
        if (!(t = i.exec(e)) || this.rules.block.hr.test(e)) break;
        c = t[0], e = e.substring(c.length);
        let k = me(t[2].split(`
`, 1)[0], t[1].length), h = e.split(`
`, 1)[0], R = !k.trim(), f = 0;
        if (this.options.pedantic ? (f = 2, p = k.trimStart()) : R ? f = t[1].length + 1 : (f = k.search(this.rules.other.nonSpaceChar), f = f > 4 ? 1 : f, p = k.slice(f), f += t[1].length), R && this.rules.other.blankLine.test(h) && (c += h + `
`, e = e.substring(h.length + 1), a = true), !a) {
          let S = this.rules.other.nextBulletRegex(f), te = this.rules.other.hrRegex(f), ne = this.rules.other.fencesBeginRegex(f), re = this.rules.other.headingBeginRegex(f), be = this.rules.other.htmlBeginRegex(f), Re = this.rules.other.blockquoteBeginRegex(f);
          for (; e; ) {
            let G = e.split(`
`, 1)[0], C;
            if (h = G, this.options.pedantic ? (h = h.replace(this.rules.other.listReplaceNesting, "  "), C = h) : C = h.replace(this.rules.other.tabCharGlobal, "    "), ne.test(h) || re.test(h) || be.test(h) || Re.test(h) || S.test(h) || te.test(h)) break;
            if (C.search(this.rules.other.nonSpaceChar) >= f || !h.trim()) p += `
` + C.slice(f);
            else {
              if (R || k.replace(this.rules.other.tabCharGlobal, "    ").search(this.rules.other.nonSpaceChar) >= 4 || ne.test(k) || re.test(k) || te.test(k)) break;
              p += `
` + h;
            }
            R = !h.trim(), c += G + `
`, e = e.substring(G.length + 1), k = C.slice(f);
          }
        }
        r.loose || (o ? r.loose = true : this.rules.other.doubleBlankLine.test(c) && (o = true)), r.items.push({ type: "list_item", raw: c, task: !!this.options.gfm && this.rules.other.listIsTask.test(p), loose: false, text: p, tokens: [] }), r.raw += c;
      }
      let u = r.items.at(-1);
      if (u) u.raw = u.raw.trimEnd(), u.text = u.text.trimEnd();
      else return;
      r.raw = r.raw.trimEnd();
      for (let a of r.items) {
        this.lexer.state.top = false, a.tokens = this.lexer.blockTokens(a.text, []);
        let c = a.tokens[0];
        if (a.task && (c?.type === "text" || c?.type === "paragraph")) {
          a.text = a.text.replace(this.rules.other.listReplaceTask, ""), c.raw = c.raw.replace(this.rules.other.listReplaceTask, ""), c.text = c.text.replace(this.rules.other.listReplaceTask, "");
          for (let k = this.lexer.inlineQueue.length - 1; k >= 0; k--) if (this.rules.other.listIsTask.test(this.lexer.inlineQueue[k].src)) {
            this.lexer.inlineQueue[k].src = this.lexer.inlineQueue[k].src.replace(this.rules.other.listReplaceTask, "");
            break;
          }
          let p = this.rules.other.listTaskCheckbox.exec(a.raw);
          if (p) {
            let k = { type: "checkbox", raw: p[0] + " ", checked: p[0] !== "[ ]" };
            a.checked = k.checked, r.loose ? a.tokens[0] && ["paragraph", "text"].includes(a.tokens[0].type) && "tokens" in a.tokens[0] && a.tokens[0].tokens ? (a.tokens[0].raw = k.raw + a.tokens[0].raw, a.tokens[0].text = k.raw + a.tokens[0].text, a.tokens[0].tokens.unshift(k)) : a.tokens.unshift({ type: "paragraph", raw: k.raw, text: k.raw, tokens: [k] }) : a.tokens.unshift(k);
          }
        } else a.task && (a.task = false);
        if (!r.loose) {
          let p = a.tokens.filter((h) => h.type === "space"), k = p.length > 0 && p.some((h) => this.rules.other.anyLine.test(h.raw));
          r.loose = k;
        }
      }
      if (r.loose) for (let a of r.items) {
        a.loose = true;
        for (let c of a.tokens) c.type === "text" && (c.type = "paragraph");
      }
      return r;
    }
  }
  html(e) {
    let t = this.rules.block.html.exec(e);
    if (t) {
      let n = ee(t[0]);
      return { type: "html", block: true, raw: n, pre: t[1] === "pre" || t[1] === "script" || t[1] === "style", text: n };
    }
  }
  def(e) {
    let t = this.rules.block.def.exec(e);
    if (t) {
      let n = t[1].toLowerCase().replace(this.rules.other.multipleSpaceGlobal, " "), s = t[2] ? t[2].replace(this.rules.other.hrefBrackets, "$1").replace(this.rules.inline.anyPunctuation, "$1") : "", r = t[3] ? t[3].substring(1, t[3].length - 1).replace(this.rules.inline.anyPunctuation, "$1") : t[3];
      return { type: "def", tag: n, raw: $(t[0], `
`), href: s, title: r };
    }
  }
  table(e) {
    let t = this.rules.block.table.exec(e);
    if (!t || !this.rules.other.tableDelimiter.test(t[2])) return;
    let n = Y(t[1]), s = t[2].replace(this.rules.other.tableAlignChars, "").split("|"), r = t[3]?.trim() ? t[3].replace(this.rules.other.tableRowBlankLine, "").split(`
`) : [], i = { type: "table", raw: $(t[0], `
`), header: [], align: [], rows: [] };
    if (n.length === s.length) {
      for (let o of s) this.rules.other.tableAlignRight.test(o) ? i.align.push("right") : this.rules.other.tableAlignCenter.test(o) ? i.align.push("center") : this.rules.other.tableAlignLeft.test(o) ? i.align.push("left") : i.align.push(null);
      for (let o = 0; o < n.length; o++) i.header.push({ text: n[o], tokens: this.lexer.inline(n[o]), header: true, align: i.align[o] });
      for (let o of r) i.rows.push(Y(o, i.header.length).map((u, a) => ({ text: u, tokens: this.lexer.inline(u), header: false, align: i.align[a] })));
      return i;
    }
  }
  lheading(e) {
    let t = this.rules.block.lheading.exec(e);
    if (t) {
      let n = t[1].trim();
      return { type: "heading", raw: $(t[0], `
`), depth: t[2].charAt(0) === "=" ? 1 : 2, text: n, tokens: this.lexer.inline(n) };
    }
  }
  paragraph(e) {
    let t = this.rules.block.paragraph.exec(e);
    if (t) {
      let n = t[1].charAt(t[1].length - 1) === `
` ? t[1].slice(0, -1) : t[1];
      return { type: "paragraph", raw: t[0], text: n, tokens: this.lexer.inline(n) };
    }
  }
  text(e) {
    let t = this.rules.block.text.exec(e);
    if (t) return { type: "text", raw: t[0], text: t[0], tokens: this.lexer.inline(t[0]) };
  }
  escape(e) {
    let t = this.rules.inline.escape.exec(e);
    if (t) return { type: "escape", raw: t[0], text: t[1] };
  }
  tag(e) {
    let t = this.rules.inline.tag.exec(e);
    if (t) return !this.lexer.state.inLink && this.rules.other.startATag.test(t[0]) ? this.lexer.state.inLink = true : this.lexer.state.inLink && this.rules.other.endATag.test(t[0]) && (this.lexer.state.inLink = false), !this.lexer.state.inRawBlock && this.rules.other.startPreScriptTag.test(t[0]) ? this.lexer.state.inRawBlock = true : this.lexer.state.inRawBlock && this.rules.other.endPreScriptTag.test(t[0]) && (this.lexer.state.inRawBlock = false), { type: "html", raw: t[0], inLink: this.lexer.state.inLink, inRawBlock: this.lexer.state.inRawBlock, block: false, text: t[0] };
  }
  link(e) {
    let t = this.rules.inline.link.exec(e);
    if (t) {
      let n = t[2].trim();
      if (!this.options.pedantic && this.rules.other.startAngleBracket.test(n)) {
        if (!this.rules.other.endAngleBracket.test(n)) return;
        let i = $(n.slice(0, -1), "\\");
        if ((n.length - i.length) % 2 === 0) return;
      } else {
        let i = fe(t[2], "()");
        if (i === -2) return;
        if (i > -1) {
          let u = (t[0].indexOf("!") === 0 ? 5 : 4) + t[1].length + i;
          t[2] = t[2].substring(0, i), t[0] = t[0].substring(0, u).trim(), t[3] = "";
        }
      }
      let s = t[2], r = "";
      if (this.options.pedantic) {
        let i = this.rules.other.pedanticHrefTitle.exec(s);
        i && (s = i[1], r = i[3]);
      } else r = t[3] ? t[3].slice(1, -1) : "";
      return s = s.trim(), this.rules.other.startAngleBracket.test(s) && (this.options.pedantic && !this.rules.other.endAngleBracket.test(n) ? s = s.slice(1) : s = s.slice(1, -1)), xe(t, { href: s && s.replace(this.rules.inline.anyPunctuation, "$1"), title: r && r.replace(this.rules.inline.anyPunctuation, "$1") }, t[0], this.lexer, this.rules);
    }
  }
  reflink(e, t) {
    let n;
    if ((n = this.rules.inline.reflink.exec(e)) || (n = this.rules.inline.nolink.exec(e))) {
      let s = (n[2] || n[1]).replace(this.rules.other.multipleSpaceGlobal, " "), r = t[s.toLowerCase()];
      if (!r) {
        let i = n[0].charAt(0);
        return { type: "text", raw: i, text: i };
      }
      return xe(n, r, n[0], this.lexer, this.rules);
    }
  }
  emStrong(e, t, n = "") {
    let s = this.rules.inline.emStrongLDelim.exec(e);
    if (!s || !s[1] && !s[2] && !s[3] && !s[4] || s[4] && n.match(this.rules.other.unicodeAlphaNumeric)) return;
    if (!(s[1] || s[3] || "") || !n || this.rules.inline.punctuation.exec(n)) {
      let i = [...s[0]].length - 1, o, u, a = i, c = 0, p = s[0][0] === "*" ? this.rules.inline.emStrongRDelimAst : this.rules.inline.emStrongRDelimUnd;
      for (p.lastIndex = 0, t = t.slice(-1 * e.length + i); (s = p.exec(t)) !== null; ) {
        if (o = s[1] || s[2] || s[3] || s[4] || s[5] || s[6], !o) continue;
        if (u = [...o].length, s[3] || s[4]) {
          a += u;
          continue;
        } else if ((s[5] || s[6]) && i % 3 && !((i + u) % 3)) {
          c += u;
          continue;
        }
        if (a -= u, a > 0) continue;
        u = Math.min(u, u + a + c);
        let k = [...s[0]][0].length, h = e.slice(0, i + s.index + k + u);
        if (Math.min(i, u) % 2) {
          let f = h.slice(1, -1);
          return { type: "em", raw: h, text: f, tokens: this.lexer.inlineTokens(f) };
        }
        let R = h.slice(2, -2);
        return { type: "strong", raw: h, text: R, tokens: this.lexer.inlineTokens(R) };
      }
    }
  }
  codespan(e) {
    let t = this.rules.inline.code.exec(e);
    if (t) {
      let n = t[2].replace(this.rules.other.newLineCharGlobal, " "), s = this.rules.other.nonSpaceChar.test(n), r = this.rules.other.startingSpaceChar.test(n) && this.rules.other.endingSpaceChar.test(n);
      return s && r && (n = n.substring(1, n.length - 1)), { type: "codespan", raw: t[0], text: n };
    }
  }
  br(e) {
    let t = this.rules.inline.br.exec(e);
    if (t) return { type: "br", raw: t[0] };
  }
  del(e, t, n = "") {
    let s = this.rules.inline.delLDelim.exec(e);
    if (!s) return;
    if (!(s[1] || "") || !n || this.rules.inline.punctuation.exec(n)) {
      let i = [...s[0]].length - 1, o, u, a = i, c = this.rules.inline.delRDelim;
      for (c.lastIndex = 0, t = t.slice(-1 * e.length + i); (s = c.exec(t)) !== null; ) {
        if (o = s[1] || s[2] || s[3] || s[4] || s[5] || s[6], !o || (u = [...o].length, u !== i)) continue;
        if (s[3] || s[4]) {
          a += u;
          continue;
        }
        if (a -= u, a > 0) continue;
        u = Math.min(u, u + a);
        let p = [...s[0]][0].length, k = e.slice(0, i + s.index + p + u), h = k.slice(i, -i);
        return { type: "del", raw: k, text: h, tokens: this.lexer.inlineTokens(h) };
      }
    }
  }
  autolink(e) {
    let t = this.rules.inline.autolink.exec(e);
    if (t) {
      let n, s;
      return t[2] === "@" ? (n = t[1], s = "mailto:" + n) : (n = t[1], s = n), { type: "link", raw: t[0], text: n, href: s, tokens: [{ type: "text", raw: n, text: n }] };
    }
  }
  url(e) {
    let t;
    if (t = this.rules.inline.url.exec(e)) {
      let n, s;
      if (t[2] === "@") n = t[0], s = "mailto:" + n;
      else {
        let r;
        do
          r = t[0], t[0] = this.rules.inline._backpedal.exec(t[0])?.[0] ?? "";
        while (r !== t[0]);
        n = t[0], t[1] === "www." ? s = "http://" + t[0] : s = t[0];
      }
      return { type: "link", raw: t[0], text: n, href: s, tokens: [{ type: "text", raw: n, text: n }] };
    }
  }
  inlineText(e) {
    let t = this.rules.inline.text.exec(e);
    if (t) {
      let n = this.lexer.state.inRawBlock;
      return { type: "text", raw: t[0], text: t[0], escaped: n };
    }
  }
};
var x = class l {
  tokens;
  options;
  state;
  inlineQueue;
  tokenizer;
  constructor(e) {
    this.tokens = [], this.tokens.links = /* @__PURE__ */ Object.create(null), this.options = e || T, this.options.tokenizer = this.options.tokenizer || new w(), this.tokenizer = this.options.tokenizer, this.tokenizer.options = this.options, this.tokenizer.lexer = this, this.inlineQueue = [], this.state = { inLink: false, inRawBlock: false, top: true };
    let t = { other: m, block: D.normal, inline: A.normal };
    this.options.pedantic ? (t.block = D.pedantic, t.inline = A.pedantic) : this.options.gfm && (t.block = D.gfm, this.options.breaks ? t.inline = A.breaks : t.inline = A.gfm), this.tokenizer.rules = t;
  }
  static get rules() {
    return { block: D, inline: A };
  }
  static lex(e, t) {
    return new l(t).lex(e);
  }
  static lexInline(e, t) {
    return new l(t).inlineTokens(e);
  }
  lex(e) {
    e = e.replace(m.carriageReturn, `
`), this.blockTokens(e, this.tokens);
    for (let t = 0; t < this.inlineQueue.length; t++) {
      let n = this.inlineQueue[t];
      this.inlineTokens(n.src, n.tokens);
    }
    return this.inlineQueue = [], this.tokens;
  }
  blockTokens(e, t = [], n = false) {
    this.tokenizer.lexer = this, this.options.pedantic && (e = e.replace(m.tabCharGlobal, "    ").replace(m.spaceLine, ""));
    let s = 1 / 0;
    for (; e; ) {
      if (e.length < s) s = e.length;
      else {
        this.infiniteLoopError(e.charCodeAt(0));
        break;
      }
      let r;
      if (this.options.extensions?.block?.some((o) => (r = o.call({ lexer: this }, e, t)) ? (e = e.substring(r.raw.length), t.push(r), true) : false)) continue;
      if (r = this.tokenizer.space(e)) {
        e = e.substring(r.raw.length);
        let o = t.at(-1);
        r.raw.length === 1 && o !== void 0 ? o.raw += `
` : t.push(r);
        continue;
      }
      if (r = this.tokenizer.code(e)) {
        e = e.substring(r.raw.length);
        let o = t.at(-1);
        o?.type === "paragraph" || o?.type === "text" ? (o.raw += (o.raw.endsWith(`
`) ? "" : `
`) + r.raw, o.text += `
` + r.text, this.inlineQueue.at(-1).src = o.text) : t.push(r);
        continue;
      }
      if (r = this.tokenizer.fences(e)) {
        e = e.substring(r.raw.length), t.push(r);
        continue;
      }
      if (r = this.tokenizer.heading(e)) {
        e = e.substring(r.raw.length), t.push(r);
        continue;
      }
      if (r = this.tokenizer.hr(e)) {
        e = e.substring(r.raw.length), t.push(r);
        continue;
      }
      if (r = this.tokenizer.blockquote(e)) {
        e = e.substring(r.raw.length), t.push(r);
        continue;
      }
      if (r = this.tokenizer.list(e)) {
        e = e.substring(r.raw.length), t.push(r);
        continue;
      }
      if (r = this.tokenizer.html(e)) {
        e = e.substring(r.raw.length), t.push(r);
        continue;
      }
      if (r = this.tokenizer.def(e)) {
        e = e.substring(r.raw.length);
        let o = t.at(-1);
        o?.type === "paragraph" || o?.type === "text" ? (o.raw += (o.raw.endsWith(`
`) ? "" : `
`) + r.raw, o.text += `
` + r.raw, this.inlineQueue.at(-1).src = o.text) : this.tokens.links[r.tag] || (this.tokens.links[r.tag] = { href: r.href, title: r.title }, t.push(r));
        continue;
      }
      if (r = this.tokenizer.table(e)) {
        e = e.substring(r.raw.length), t.push(r);
        continue;
      }
      if (r = this.tokenizer.lheading(e)) {
        e = e.substring(r.raw.length), t.push(r);
        continue;
      }
      let i = e;
      if (this.options.extensions?.startBlock) {
        let o = 1 / 0, u = e.slice(1), a;
        this.options.extensions.startBlock.forEach((c) => {
          a = c.call({ lexer: this }, u), typeof a == "number" && a >= 0 && (o = Math.min(o, a));
        }), o < 1 / 0 && o >= 0 && (i = e.substring(0, o + 1));
      }
      if (this.state.top && (r = this.tokenizer.paragraph(i))) {
        let o = t.at(-1);
        n && o?.type === "paragraph" ? (o.raw += (o.raw.endsWith(`
`) ? "" : `
`) + r.raw, o.text += `
` + r.text, this.inlineQueue.pop(), this.inlineQueue.at(-1).src = o.text) : t.push(r), n = i.length !== e.length, e = e.substring(r.raw.length);
        continue;
      }
      if (r = this.tokenizer.text(e)) {
        e = e.substring(r.raw.length);
        let o = t.at(-1);
        o?.type === "text" ? (o.raw += (o.raw.endsWith(`
`) ? "" : `
`) + r.raw, o.text += `
` + r.text, this.inlineQueue.pop(), this.inlineQueue.at(-1).src = o.text) : t.push(r);
        continue;
      }
      if (e) {
        this.infiniteLoopError(e.charCodeAt(0));
        break;
      }
    }
    return this.state.top = true, t;
  }
  inline(e, t = []) {
    return this.inlineQueue.push({ src: e, tokens: t }), t;
  }
  inlineTokens(e, t = []) {
    this.tokenizer.lexer = this;
    let n = e, s = null;
    if (this.tokens.links) {
      let a = Object.keys(this.tokens.links);
      if (a.length > 0) for (; (s = this.tokenizer.rules.inline.reflinkSearch.exec(n)) !== null; ) a.includes(s[0].slice(s[0].lastIndexOf("[") + 1, -1)) && (n = n.slice(0, s.index) + "[" + "a".repeat(s[0].length - 2) + "]" + n.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex));
    }
    for (; (s = this.tokenizer.rules.inline.anyPunctuation.exec(n)) !== null; ) n = n.slice(0, s.index) + "++" + n.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);
    let r;
    for (; (s = this.tokenizer.rules.inline.blockSkip.exec(n)) !== null; ) r = s[2] ? s[2].length : 0, n = n.slice(0, s.index + r) + "[" + "a".repeat(s[0].length - r - 2) + "]" + n.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);
    n = this.options.hooks?.emStrongMask?.call({ lexer: this }, n) ?? n;
    let i = false, o = "", u = 1 / 0;
    for (; e; ) {
      if (e.length < u) u = e.length;
      else {
        this.infiniteLoopError(e.charCodeAt(0));
        break;
      }
      i || (o = ""), i = false;
      let a;
      if (this.options.extensions?.inline?.some((p) => (a = p.call({ lexer: this }, e, t)) ? (e = e.substring(a.raw.length), t.push(a), true) : false)) continue;
      if (a = this.tokenizer.escape(e)) {
        e = e.substring(a.raw.length), t.push(a);
        continue;
      }
      if (a = this.tokenizer.tag(e)) {
        e = e.substring(a.raw.length), t.push(a);
        continue;
      }
      if (a = this.tokenizer.link(e)) {
        e = e.substring(a.raw.length), t.push(a);
        continue;
      }
      if (a = this.tokenizer.reflink(e, this.tokens.links)) {
        e = e.substring(a.raw.length);
        let p = t.at(-1);
        a.type === "text" && p?.type === "text" ? (p.raw += a.raw, p.text += a.text) : t.push(a);
        continue;
      }
      if (a = this.tokenizer.emStrong(e, n, o)) {
        e = e.substring(a.raw.length), t.push(a);
        continue;
      }
      if (a = this.tokenizer.codespan(e)) {
        e = e.substring(a.raw.length), t.push(a);
        continue;
      }
      if (a = this.tokenizer.br(e)) {
        e = e.substring(a.raw.length), t.push(a);
        continue;
      }
      if (a = this.tokenizer.del(e, n, o)) {
        e = e.substring(a.raw.length), t.push(a);
        continue;
      }
      if (a = this.tokenizer.autolink(e)) {
        e = e.substring(a.raw.length), t.push(a);
        continue;
      }
      if (!this.state.inLink && (a = this.tokenizer.url(e))) {
        e = e.substring(a.raw.length), t.push(a);
        continue;
      }
      let c = e;
      if (this.options.extensions?.startInline) {
        let p = 1 / 0, k = e.slice(1), h;
        this.options.extensions.startInline.forEach((R) => {
          h = R.call({ lexer: this }, k), typeof h == "number" && h >= 0 && (p = Math.min(p, h));
        }), p < 1 / 0 && p >= 0 && (c = e.substring(0, p + 1));
      }
      if (a = this.tokenizer.inlineText(c)) {
        e = e.substring(a.raw.length), a.raw.slice(-1) !== "_" && (o = a.raw.slice(-1)), i = true;
        let p = t.at(-1);
        p?.type === "text" ? (p.raw += a.raw, p.text += a.text) : t.push(a);
        continue;
      }
      if (e) {
        this.infiniteLoopError(e.charCodeAt(0));
        break;
      }
    }
    return t;
  }
  infiniteLoopError(e) {
    let t = "Infinite loop on byte: " + e;
    if (this.options.silent) console.error(t);
    else throw new Error(t);
  }
};
var y = class {
  options;
  parser;
  constructor(e) {
    this.options = e || T;
  }
  space(e) {
    return "";
  }
  code({ text: e, lang: t, escaped: n }) {
    let s = (t || "").match(m.notSpaceStart)?.[0], r = e.replace(m.endingNewline, "") + `
`;
    return s ? '<pre><code class="language-' + O(s) + '">' + (n ? r : O(r, true)) + `</code></pre>
` : "<pre><code>" + (n ? r : O(r, true)) + `</code></pre>
`;
  }
  blockquote({ tokens: e }) {
    return `<blockquote>
${this.parser.parse(e)}</blockquote>
`;
  }
  html({ text: e }) {
    return e;
  }
  def(e) {
    return "";
  }
  heading({ tokens: e, depth: t }) {
    return `<h${t}>${this.parser.parseInline(e)}</h${t}>
`;
  }
  hr(e) {
    return `<hr>
`;
  }
  list(e) {
    let t = e.ordered, n = e.start, s = "";
    for (let o = 0; o < e.items.length; o++) {
      let u = e.items[o];
      s += this.listitem(u);
    }
    let r = t ? "ol" : "ul", i = t && n !== 1 ? ' start="' + n + '"' : "";
    return "<" + r + i + `>
` + s + "</" + r + `>
`;
  }
  listitem(e) {
    return `<li>${this.parser.parse(e.tokens)}</li>
`;
  }
  checkbox({ checked: e }) {
    return "<input " + (e ? 'checked="" ' : "") + 'disabled="" type="checkbox"> ';
  }
  paragraph({ tokens: e }) {
    return `<p>${this.parser.parseInline(e)}</p>
`;
  }
  table(e) {
    let t = "", n = "";
    for (let r = 0; r < e.header.length; r++) n += this.tablecell(e.header[r]);
    t += this.tablerow({ text: n });
    let s = "";
    for (let r = 0; r < e.rows.length; r++) {
      let i = e.rows[r];
      n = "";
      for (let o = 0; o < i.length; o++) n += this.tablecell(i[o]);
      s += this.tablerow({ text: n });
    }
    return s && (s = `<tbody>${s}</tbody>`), `<table>
<thead>
` + t + `</thead>
` + s + `</table>
`;
  }
  tablerow({ text: e }) {
    return `<tr>
${e}</tr>
`;
  }
  tablecell(e) {
    let t = this.parser.parseInline(e.tokens), n = e.header ? "th" : "td";
    return (e.align ? `<${n} align="${e.align}">` : `<${n}>`) + t + `</${n}>
`;
  }
  strong({ tokens: e }) {
    return `<strong>${this.parser.parseInline(e)}</strong>`;
  }
  em({ tokens: e }) {
    return `<em>${this.parser.parseInline(e)}</em>`;
  }
  codespan({ text: e }) {
    return `<code>${O(e, true)}</code>`;
  }
  br(e) {
    return "<br>";
  }
  del({ tokens: e }) {
    return `<del>${this.parser.parseInline(e)}</del>`;
  }
  link({ href: e, title: t, tokens: n }) {
    let s = this.parser.parseInline(n), r = V(e);
    if (r === null) return s;
    e = r;
    let i = '<a href="' + e + '"';
    return t && (i += ' title="' + O(t) + '"'), i += ">" + s + "</a>", i;
  }
  image({ href: e, title: t, text: n, tokens: s }) {
    s && (n = this.parser.parseInline(s, this.parser.textRenderer));
    let r = V(e);
    if (r === null) return O(n);
    e = r;
    let i = `<img src="${e}" alt="${O(n)}"`;
    return t && (i += ` title="${O(t)}"`), i += ">", i;
  }
  text(e) {
    return "tokens" in e && e.tokens ? this.parser.parseInline(e.tokens) : "escaped" in e && e.escaped ? e.text : O(e.text);
  }
};
var L = class {
  strong({ text: e }) {
    return e;
  }
  em({ text: e }) {
    return e;
  }
  codespan({ text: e }) {
    return e;
  }
  del({ text: e }) {
    return e;
  }
  html({ text: e }) {
    return e;
  }
  text({ text: e }) {
    return e;
  }
  link({ text: e }) {
    return "" + e;
  }
  image({ text: e }) {
    return "" + e;
  }
  br() {
    return "";
  }
  checkbox({ raw: e }) {
    return e;
  }
};
var b = class l2 {
  options;
  renderer;
  textRenderer;
  constructor(e) {
    this.options = e || T, this.options.renderer = this.options.renderer || new y(), this.renderer = this.options.renderer, this.renderer.options = this.options, this.renderer.parser = this, this.textRenderer = new L();
  }
  static parse(e, t) {
    return new l2(t).parse(e);
  }
  static parseInline(e, t) {
    return new l2(t).parseInline(e);
  }
  parse(e) {
    this.renderer.parser = this;
    let t = "";
    for (let n = 0; n < e.length; n++) {
      let s = e[n];
      if (this.options.extensions?.renderers?.[s.type]) {
        let i = s, o = this.options.extensions.renderers[i.type].call({ parser: this }, i);
        if (o !== false || !["space", "hr", "heading", "code", "table", "blockquote", "list", "html", "def", "paragraph", "text"].includes(i.type)) {
          t += o || "";
          continue;
        }
      }
      let r = s;
      switch (r.type) {
        case "space": {
          t += this.renderer.space(r);
          break;
        }
        case "hr": {
          t += this.renderer.hr(r);
          break;
        }
        case "heading": {
          t += this.renderer.heading(r);
          break;
        }
        case "code": {
          t += this.renderer.code(r);
          break;
        }
        case "table": {
          t += this.renderer.table(r);
          break;
        }
        case "blockquote": {
          t += this.renderer.blockquote(r);
          break;
        }
        case "list": {
          t += this.renderer.list(r);
          break;
        }
        case "checkbox": {
          t += this.renderer.checkbox(r);
          break;
        }
        case "html": {
          t += this.renderer.html(r);
          break;
        }
        case "def": {
          t += this.renderer.def(r);
          break;
        }
        case "paragraph": {
          t += this.renderer.paragraph(r);
          break;
        }
        case "text": {
          t += this.renderer.text(r);
          break;
        }
        default: {
          let i = 'Token with "' + r.type + '" type was not found.';
          if (this.options.silent) return console.error(i), "";
          throw new Error(i);
        }
      }
    }
    return t;
  }
  parseInline(e, t = this.renderer) {
    this.renderer.parser = this;
    let n = "";
    for (let s = 0; s < e.length; s++) {
      let r = e[s];
      if (this.options.extensions?.renderers?.[r.type]) {
        let o = this.options.extensions.renderers[r.type].call({ parser: this }, r);
        if (o !== false || !["escape", "html", "link", "image", "strong", "em", "codespan", "br", "del", "text"].includes(r.type)) {
          n += o || "";
          continue;
        }
      }
      let i = r;
      switch (i.type) {
        case "escape": {
          n += t.text(i);
          break;
        }
        case "html": {
          n += t.html(i);
          break;
        }
        case "link": {
          n += t.link(i);
          break;
        }
        case "image": {
          n += t.image(i);
          break;
        }
        case "checkbox": {
          n += t.checkbox(i);
          break;
        }
        case "strong": {
          n += t.strong(i);
          break;
        }
        case "em": {
          n += t.em(i);
          break;
        }
        case "codespan": {
          n += t.codespan(i);
          break;
        }
        case "br": {
          n += t.br(i);
          break;
        }
        case "del": {
          n += t.del(i);
          break;
        }
        case "text": {
          n += t.text(i);
          break;
        }
        default: {
          let o = 'Token with "' + i.type + '" type was not found.';
          if (this.options.silent) return console.error(o), "";
          throw new Error(o);
        }
      }
    }
    return n;
  }
};
var P = class {
  options;
  block;
  constructor(e) {
    this.options = e || T;
  }
  static passThroughHooks = /* @__PURE__ */ new Set(["preprocess", "postprocess", "processAllTokens", "emStrongMask"]);
  static passThroughHooksRespectAsync = /* @__PURE__ */ new Set(["preprocess", "postprocess", "processAllTokens"]);
  preprocess(e) {
    return e;
  }
  postprocess(e) {
    return e;
  }
  processAllTokens(e) {
    return e;
  }
  emStrongMask(e) {
    return e;
  }
  provideLexer(e = this.block) {
    return e ? x.lex : x.lexInline;
  }
  provideParser(e = this.block) {
    return e ? b.parse : b.parseInline;
  }
};
var q = class {
  defaults = M();
  options = this.setOptions;
  parse = this.parseMarkdown(true);
  parseInline = this.parseMarkdown(false);
  Parser = b;
  Renderer = y;
  TextRenderer = L;
  Lexer = x;
  Tokenizer = w;
  Hooks = P;
  constructor(...e) {
    this.use(...e);
  }
  walkTokens(e, t) {
    let n = [];
    for (let s of e) switch (n = n.concat(t.call(this, s)), s.type) {
      case "table": {
        let r = s;
        for (let i of r.header) n = n.concat(this.walkTokens(i.tokens, t));
        for (let i of r.rows) for (let o of i) n = n.concat(this.walkTokens(o.tokens, t));
        break;
      }
      case "list": {
        let r = s;
        n = n.concat(this.walkTokens(r.items, t));
        break;
      }
      default: {
        let r = s;
        this.defaults.extensions?.childTokens?.[r.type] ? this.defaults.extensions.childTokens[r.type].forEach((i) => {
          let o = r[i].flat(1 / 0);
          n = n.concat(this.walkTokens(o, t));
        }) : r.tokens && (n = n.concat(this.walkTokens(r.tokens, t)));
      }
    }
    return n;
  }
  use(...e) {
    let t = this.defaults.extensions || { renderers: {}, childTokens: {} };
    return e.forEach((n) => {
      let s = { ...n };
      if (s.async = this.defaults.async || s.async || false, n.extensions && (n.extensions.forEach((r) => {
        if (!r.name) throw new Error("extension name required");
        if ("renderer" in r) {
          let i = t.renderers[r.name];
          i ? t.renderers[r.name] = function(...o) {
            let u = r.renderer.apply(this, o);
            return u === false && (u = i.apply(this, o)), u;
          } : t.renderers[r.name] = r.renderer;
        }
        if ("tokenizer" in r) {
          if (!r.level || r.level !== "block" && r.level !== "inline") throw new Error("extension level must be 'block' or 'inline'");
          let i = t[r.level];
          i ? i.unshift(r.tokenizer) : t[r.level] = [r.tokenizer], r.start && (r.level === "block" ? t.startBlock ? t.startBlock.push(r.start) : t.startBlock = [r.start] : r.level === "inline" && (t.startInline ? t.startInline.push(r.start) : t.startInline = [r.start]));
        }
        "childTokens" in r && r.childTokens && (t.childTokens[r.name] = r.childTokens);
      }), s.extensions = t), n.renderer) {
        let r = this.defaults.renderer || new y(this.defaults);
        for (let i in n.renderer) {
          if (!(i in r)) throw new Error(`renderer '${i}' does not exist`);
          if (["options", "parser"].includes(i)) continue;
          let o = i, u = n.renderer[o], a = r[o];
          r[o] = (...c) => {
            let p = u.apply(r, c);
            return p === false && (p = a.apply(r, c)), p || "";
          };
        }
        s.renderer = r;
      }
      if (n.tokenizer) {
        let r = this.defaults.tokenizer || new w(this.defaults);
        for (let i in n.tokenizer) {
          if (!(i in r)) throw new Error(`tokenizer '${i}' does not exist`);
          if (["options", "rules", "lexer"].includes(i)) continue;
          let o = i, u = n.tokenizer[o], a = r[o];
          r[o] = (...c) => {
            let p = u.apply(r, c);
            return p === false && (p = a.apply(r, c)), p;
          };
        }
        s.tokenizer = r;
      }
      if (n.hooks) {
        let r = this.defaults.hooks || new P();
        for (let i in n.hooks) {
          if (!(i in r)) throw new Error(`hook '${i}' does not exist`);
          if (["options", "block"].includes(i)) continue;
          let o = i, u = n.hooks[o], a = r[o];
          P.passThroughHooks.has(i) ? r[o] = (c) => {
            if (this.defaults.async && P.passThroughHooksRespectAsync.has(i)) return (async () => {
              let k = await u.call(r, c);
              return a.call(r, k);
            })();
            let p = u.call(r, c);
            return a.call(r, p);
          } : r[o] = (...c) => {
            if (this.defaults.async) return (async () => {
              let k = await u.apply(r, c);
              return k === false && (k = await a.apply(r, c)), k;
            })();
            let p = u.apply(r, c);
            return p === false && (p = a.apply(r, c)), p;
          };
        }
        s.hooks = r;
      }
      if (n.walkTokens) {
        let r = this.defaults.walkTokens, i = n.walkTokens;
        s.walkTokens = function(o) {
          let u = [];
          return u.push(i.call(this, o)), r && (u = u.concat(r.call(this, o))), u;
        };
      }
      this.defaults = { ...this.defaults, ...s };
    }), this;
  }
  setOptions(e) {
    return this.defaults = { ...this.defaults, ...e }, this;
  }
  lexer(e, t) {
    return x.lex(e, t ?? this.defaults);
  }
  parser(e, t) {
    return b.parse(e, t ?? this.defaults);
  }
  parseMarkdown(e) {
    return (n, s) => {
      let r = { ...s }, i = { ...this.defaults, ...r }, o = this.onError(!!i.silent, !!i.async);
      if (this.defaults.async === true && r.async === false) return o(new Error("marked(): The async option was set to true by an extension. Remove async: false from the parse options object to return a Promise."));
      if (typeof n > "u" || n === null) return o(new Error("marked(): input parameter is undefined or null"));
      if (typeof n != "string") return o(new Error("marked(): input parameter is of type " + Object.prototype.toString.call(n) + ", string expected"));
      if (i.hooks && (i.hooks.options = i, i.hooks.block = e), i.async) return (async () => {
        let u = i.hooks ? await i.hooks.preprocess(n) : n, c = await (i.hooks ? await i.hooks.provideLexer(e) : e ? x.lex : x.lexInline)(u, i), p = i.hooks ? await i.hooks.processAllTokens(c) : c;
        i.walkTokens && await Promise.all(this.walkTokens(p, i.walkTokens));
        let h = await (i.hooks ? await i.hooks.provideParser(e) : e ? b.parse : b.parseInline)(p, i);
        return i.hooks ? await i.hooks.postprocess(h) : h;
      })().catch(o);
      try {
        i.hooks && (n = i.hooks.preprocess(n));
        let a = (i.hooks ? i.hooks.provideLexer(e) : e ? x.lex : x.lexInline)(n, i);
        i.hooks && (a = i.hooks.processAllTokens(a)), i.walkTokens && this.walkTokens(a, i.walkTokens);
        let p = (i.hooks ? i.hooks.provideParser(e) : e ? b.parse : b.parseInline)(a, i);
        return i.hooks && (p = i.hooks.postprocess(p)), p;
      } catch (u) {
        return o(u);
      }
    };
  }
  onError(e, t) {
    return (n) => {
      if (n.message += `
Please report this to https://github.com/markedjs/marked.`, e) {
        let s = "<p>An error occurred:</p><pre>" + O(n.message + "", true) + "</pre>";
        return t ? Promise.resolve(s) : s;
      }
      if (t) return Promise.reject(n);
      throw n;
    };
  }
};
var z = new q();
function g(l3, e) {
  return z.parse(l3, e);
}
g.options = g.setOptions = function(l3) {
  return z.setOptions(l3), g.defaults = z.defaults, N(g.defaults), g;
};
g.getDefaults = M;
g.defaults = T;
g.use = function(...l3) {
  return z.use(...l3), g.defaults = z.defaults, N(g.defaults), g;
};
g.walkTokens = function(l3, e) {
  return z.walkTokens(l3, e);
};
g.parseInline = z.parseInline;
g.Parser = b;
g.parser = b.parse;
g.Renderer = y;
g.TextRenderer = L;
g.Lexer = x;
g.lexer = x.lex;
g.Tokenizer = w;
g.Hooks = P;
g.parse = g;
var Ft = g.options;
var Ut = g.setOptions;
var Kt = g.use;
var Wt = g.walkTokens;
var Xt = g.parseInline;
var Vt = b.parse;
var Yt = x.lex;

// node_modules/@earendil-works/pi-tui/dist/tui.js
import { performance } from "node:perf_hooks";

// node_modules/@earendil-works/pi-tui/dist/keys.js
var _kittyProtocolActive = false;
function setKittyProtocolActive(active) {
  _kittyProtocolActive = active;
}
var Key = {
  // Special keys
  escape: "escape",
  esc: "esc",
  enter: "enter",
  return: "return",
  tab: "tab",
  space: "space",
  backspace: "backspace",
  delete: "delete",
  insert: "insert",
  clear: "clear",
  home: "home",
  end: "end",
  pageUp: "pageUp",
  pageDown: "pageDown",
  up: "up",
  down: "down",
  left: "left",
  right: "right",
  f1: "f1",
  f2: "f2",
  f3: "f3",
  f4: "f4",
  f5: "f5",
  f6: "f6",
  f7: "f7",
  f8: "f8",
  f9: "f9",
  f10: "f10",
  f11: "f11",
  f12: "f12",
  // Symbol keys
  backtick: "`",
  hyphen: "-",
  equals: "=",
  leftbracket: "[",
  rightbracket: "]",
  backslash: "\\",
  semicolon: ";",
  quote: "'",
  comma: ",",
  period: ".",
  slash: "/",
  exclamation: "!",
  at: "@",
  hash: "#",
  dollar: "$",
  percent: "%",
  caret: "^",
  ampersand: "&",
  asterisk: "*",
  leftparen: "(",
  rightparen: ")",
  underscore: "_",
  plus: "+",
  pipe: "|",
  tilde: "~",
  leftbrace: "{",
  rightbrace: "}",
  colon: ":",
  lessthan: "<",
  greaterthan: ">",
  question: "?",
  // Single modifiers
  ctrl: (key) => `ctrl+${key}`,
  shift: (key) => `shift+${key}`,
  alt: (key) => `alt+${key}`,
  super: (key) => `super+${key}`,
  // Combined modifiers
  ctrlShift: (key) => `ctrl+shift+${key}`,
  shiftCtrl: (key) => `shift+ctrl+${key}`,
  ctrlAlt: (key) => `ctrl+alt+${key}`,
  altCtrl: (key) => `alt+ctrl+${key}`,
  shiftAlt: (key) => `shift+alt+${key}`,
  altShift: (key) => `alt+shift+${key}`,
  ctrlSuper: (key) => `ctrl+super+${key}`,
  superCtrl: (key) => `super+ctrl+${key}`,
  shiftSuper: (key) => `shift+super+${key}`,
  superShift: (key) => `super+shift+${key}`,
  altSuper: (key) => `alt+super+${key}`,
  superAlt: (key) => `super+alt+${key}`,
  // Triple modifiers
  ctrlShiftAlt: (key) => `ctrl+shift+alt+${key}`,
  ctrlShiftSuper: (key) => `ctrl+shift+super+${key}`
};
var SYMBOL_KEYS = /* @__PURE__ */ new Set([
  "`",
  "-",
  "=",
  "[",
  "]",
  "\\",
  ";",
  "'",
  ",",
  ".",
  "/",
  "!",
  "@",
  "#",
  "$",
  "%",
  "^",
  "&",
  "*",
  "(",
  ")",
  "_",
  "+",
  "|",
  "~",
  "{",
  "}",
  ":",
  "<",
  ">",
  "?"
]);
var MODIFIERS = {
  shift: 1,
  alt: 2,
  ctrl: 4,
  super: 8
};
var LOCK_MASK = 64 + 128;
var CODEPOINTS = {
  escape: 27,
  tab: 9,
  enter: 13,
  space: 32,
  backspace: 127,
  kpEnter: 57414
  // Numpad Enter (Kitty protocol)
};
var ARROW_CODEPOINTS = {
  up: -1,
  down: -2,
  right: -3,
  left: -4
};
var FUNCTIONAL_CODEPOINTS = {
  delete: -10,
  insert: -11,
  pageUp: -12,
  pageDown: -13,
  home: -14,
  end: -15
};
var KITTY_FUNCTIONAL_KEY_EQUIVALENTS = /* @__PURE__ */ new Map([
  [57399, 48],
  // KP_0 -> 0
  [57400, 49],
  // KP_1 -> 1
  [57401, 50],
  // KP_2 -> 2
  [57402, 51],
  // KP_3 -> 3
  [57403, 52],
  // KP_4 -> 4
  [57404, 53],
  // KP_5 -> 5
  [57405, 54],
  // KP_6 -> 6
  [57406, 55],
  // KP_7 -> 7
  [57407, 56],
  // KP_8 -> 8
  [57408, 57],
  // KP_9 -> 9
  [57409, 46],
  // KP_DECIMAL -> .
  [57410, 47],
  // KP_DIVIDE -> /
  [57411, 42],
  // KP_MULTIPLY -> *
  [57412, 45],
  // KP_SUBTRACT -> -
  [57413, 43],
  // KP_ADD -> +
  [57415, 61],
  // KP_EQUAL -> =
  [57416, 44],
  // KP_SEPARATOR -> ,
  [57417, ARROW_CODEPOINTS.left],
  [57418, ARROW_CODEPOINTS.right],
  [57419, ARROW_CODEPOINTS.up],
  [57420, ARROW_CODEPOINTS.down],
  [57421, FUNCTIONAL_CODEPOINTS.pageUp],
  [57422, FUNCTIONAL_CODEPOINTS.pageDown],
  [57423, FUNCTIONAL_CODEPOINTS.home],
  [57424, FUNCTIONAL_CODEPOINTS.end],
  [57425, FUNCTIONAL_CODEPOINTS.insert],
  [57426, FUNCTIONAL_CODEPOINTS.delete]
]);
function normalizeKittyFunctionalCodepoint(codepoint) {
  return KITTY_FUNCTIONAL_KEY_EQUIVALENTS.get(codepoint) ?? codepoint;
}
function normalizeShiftedLetterIdentityCodepoint(codepoint, modifier) {
  const effectiveModifier = modifier & ~LOCK_MASK;
  if ((effectiveModifier & MODIFIERS.shift) !== 0 && codepoint >= 65 && codepoint <= 90) {
    return codepoint + 32;
  }
  return codepoint;
}
var LEGACY_KEY_SEQUENCES = {
  up: ["\x1B[A", "\x1BOA"],
  down: ["\x1B[B", "\x1BOB"],
  right: ["\x1B[C", "\x1BOC"],
  left: ["\x1B[D", "\x1BOD"],
  home: ["\x1B[H", "\x1BOH", "\x1B[1~", "\x1B[7~"],
  end: ["\x1B[F", "\x1BOF", "\x1B[4~", "\x1B[8~"],
  insert: ["\x1B[2~"],
  delete: ["\x1B[3~"],
  pageUp: ["\x1B[5~", "\x1B[[5~"],
  pageDown: ["\x1B[6~", "\x1B[[6~"],
  clear: ["\x1B[E", "\x1BOE"],
  f1: ["\x1BOP", "\x1B[11~", "\x1B[[A"],
  f2: ["\x1BOQ", "\x1B[12~", "\x1B[[B"],
  f3: ["\x1BOR", "\x1B[13~", "\x1B[[C"],
  f4: ["\x1BOS", "\x1B[14~", "\x1B[[D"],
  f5: ["\x1B[15~", "\x1B[[E"],
  f6: ["\x1B[17~"],
  f7: ["\x1B[18~"],
  f8: ["\x1B[19~"],
  f9: ["\x1B[20~"],
  f10: ["\x1B[21~"],
  f11: ["\x1B[23~"],
  f12: ["\x1B[24~"]
};
var LEGACY_SHIFT_SEQUENCES = {
  up: ["\x1B[a"],
  down: ["\x1B[b"],
  right: ["\x1B[c"],
  left: ["\x1B[d"],
  clear: ["\x1B[e"],
  insert: ["\x1B[2$"],
  delete: ["\x1B[3$"],
  pageUp: ["\x1B[5$"],
  pageDown: ["\x1B[6$"],
  home: ["\x1B[7$"],
  end: ["\x1B[8$"]
};
var LEGACY_CTRL_SEQUENCES = {
  up: ["\x1BOa"],
  down: ["\x1BOb"],
  right: ["\x1BOc"],
  left: ["\x1BOd"],
  clear: ["\x1BOe"],
  insert: ["\x1B[2^"],
  delete: ["\x1B[3^"],
  pageUp: ["\x1B[5^"],
  pageDown: ["\x1B[6^"],
  home: ["\x1B[7^"],
  end: ["\x1B[8^"]
};
var matchesLegacySequence = (data, sequences) => sequences.includes(data);
var matchesLegacyModifierSequence = (data, key, modifier) => {
  if (modifier === MODIFIERS.shift) {
    return matchesLegacySequence(data, LEGACY_SHIFT_SEQUENCES[key]);
  }
  if (modifier === MODIFIERS.ctrl) {
    return matchesLegacySequence(data, LEGACY_CTRL_SEQUENCES[key]);
  }
  return false;
};
var _lastEventType = "press";
function isKeyRelease(data) {
  if (data.includes("\x1B[200~")) {
    return false;
  }
  if (data.includes(":3u") || data.includes(":3~") || data.includes(":3A") || data.includes(":3B") || data.includes(":3C") || data.includes(":3D") || data.includes(":3H") || data.includes(":3F")) {
    return true;
  }
  return false;
}
function parseEventType(eventTypeStr) {
  if (!eventTypeStr)
    return "press";
  const eventType = parseInt(eventTypeStr, 10);
  if (eventType === 2)
    return "repeat";
  if (eventType === 3)
    return "release";
  return "press";
}
function parseKittySequence(data) {
  const csiUMatch = data.match(/^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$/);
  if (csiUMatch) {
    const codepoint = parseInt(csiUMatch[1], 10);
    const shiftedKey = csiUMatch[2] && csiUMatch[2].length > 0 ? parseInt(csiUMatch[2], 10) : void 0;
    const baseLayoutKey = csiUMatch[3] ? parseInt(csiUMatch[3], 10) : void 0;
    const modValue = csiUMatch[4] ? parseInt(csiUMatch[4], 10) : 1;
    const eventType = parseEventType(csiUMatch[5]);
    _lastEventType = eventType;
    return { codepoint, shiftedKey, baseLayoutKey, modifier: modValue - 1, eventType };
  }
  const arrowMatch = data.match(/^\x1b\[1;(\d+)(?::(\d+))?([ABCD])$/);
  if (arrowMatch) {
    const modValue = parseInt(arrowMatch[1], 10);
    const eventType = parseEventType(arrowMatch[2]);
    const arrowCodes = { A: -1, B: -2, C: -3, D: -4 };
    _lastEventType = eventType;
    return { codepoint: arrowCodes[arrowMatch[3]], modifier: modValue - 1, eventType };
  }
  const funcMatch = data.match(/^\x1b\[(\d+)(?:;(\d+))?(?::(\d+))?~$/);
  if (funcMatch) {
    const keyNum = parseInt(funcMatch[1], 10);
    const modValue = funcMatch[2] ? parseInt(funcMatch[2], 10) : 1;
    const eventType = parseEventType(funcMatch[3]);
    const funcCodes = {
      2: FUNCTIONAL_CODEPOINTS.insert,
      3: FUNCTIONAL_CODEPOINTS.delete,
      5: FUNCTIONAL_CODEPOINTS.pageUp,
      6: FUNCTIONAL_CODEPOINTS.pageDown,
      7: FUNCTIONAL_CODEPOINTS.home,
      8: FUNCTIONAL_CODEPOINTS.end
    };
    const codepoint = funcCodes[keyNum];
    if (codepoint !== void 0) {
      _lastEventType = eventType;
      return { codepoint, modifier: modValue - 1, eventType };
    }
  }
  const homeEndMatch = data.match(/^\x1b\[1;(\d+)(?::(\d+))?([HF])$/);
  if (homeEndMatch) {
    const modValue = parseInt(homeEndMatch[1], 10);
    const eventType = parseEventType(homeEndMatch[2]);
    const codepoint = homeEndMatch[3] === "H" ? FUNCTIONAL_CODEPOINTS.home : FUNCTIONAL_CODEPOINTS.end;
    _lastEventType = eventType;
    return { codepoint, modifier: modValue - 1, eventType };
  }
  return null;
}
function matchesKittySequence(data, expectedCodepoint, expectedModifier) {
  const parsed = parseKittySequence(data);
  if (!parsed)
    return false;
  const actualMod = parsed.modifier & ~LOCK_MASK;
  const expectedMod = expectedModifier & ~LOCK_MASK;
  if (actualMod !== expectedMod)
    return false;
  const normalizedCodepoint = normalizeShiftedLetterIdentityCodepoint(normalizeKittyFunctionalCodepoint(parsed.codepoint), parsed.modifier);
  const normalizedExpectedCodepoint = normalizeShiftedLetterIdentityCodepoint(normalizeKittyFunctionalCodepoint(expectedCodepoint), expectedModifier);
  if (normalizedCodepoint === normalizedExpectedCodepoint)
    return true;
  if (parsed.baseLayoutKey !== void 0 && parsed.baseLayoutKey === expectedCodepoint) {
    const cp = normalizedCodepoint;
    const isLatinLetter = cp >= 97 && cp <= 122;
    const isKnownSymbol = SYMBOL_KEYS.has(String.fromCharCode(cp));
    if (!isLatinLetter && !isKnownSymbol)
      return true;
  }
  return false;
}
function parseModifyOtherKeysSequence(data) {
  const match = data.match(/^\x1b\[27;(\d+);(\d+)~$/);
  if (!match)
    return null;
  const modValue = parseInt(match[1], 10);
  const codepoint = parseInt(match[2], 10);
  return { codepoint, modifier: modValue - 1 };
}
function matchesModifyOtherKeys(data, expectedKeycode, expectedModifier) {
  const parsed = parseModifyOtherKeysSequence(data);
  if (!parsed)
    return false;
  return parsed.codepoint === expectedKeycode && parsed.modifier === expectedModifier;
}
function isWindowsTerminalSession() {
  return Boolean(process.env.WT_SESSION) && !process.env.SSH_CONNECTION && !process.env.SSH_CLIENT && !process.env.SSH_TTY;
}
function matchesRawBackspace(data, expectedModifier) {
  if (data === "\x7F")
    return expectedModifier === 0;
  if (data !== "\b")
    return false;
  return isWindowsTerminalSession() ? expectedModifier === MODIFIERS.ctrl : expectedModifier === 0;
}
function rawCtrlChar(key) {
  const char = key.toLowerCase();
  const code = char.charCodeAt(0);
  if (code >= 97 && code <= 122 || char === "[" || char === "\\" || char === "]" || char === "_") {
    return String.fromCharCode(code & 31);
  }
  if (char === "-") {
    return String.fromCharCode(31);
  }
  return null;
}
function isDigitKey(key) {
  return key >= "0" && key <= "9";
}
function matchesPrintableModifyOtherKeys(data, expectedKeycode, expectedModifier) {
  if (expectedModifier === 0)
    return false;
  const parsed = parseModifyOtherKeysSequence(data);
  if (!parsed || parsed.modifier !== expectedModifier)
    return false;
  return normalizeShiftedLetterIdentityCodepoint(parsed.codepoint, parsed.modifier) === normalizeShiftedLetterIdentityCodepoint(expectedKeycode, expectedModifier);
}
function parseKeyId(keyId) {
  const parts = keyId.toLowerCase().split("+");
  const key = parts[parts.length - 1];
  if (!key)
    return null;
  return {
    key,
    ctrl: parts.includes("ctrl"),
    shift: parts.includes("shift"),
    alt: parts.includes("alt"),
    super: parts.includes("super")
  };
}
function matchesKey(data, keyId) {
  const parsed = parseKeyId(keyId);
  if (!parsed)
    return false;
  const { key, ctrl, shift, alt, super: superModifier } = parsed;
  let modifier = 0;
  if (shift)
    modifier |= MODIFIERS.shift;
  if (alt)
    modifier |= MODIFIERS.alt;
  if (ctrl)
    modifier |= MODIFIERS.ctrl;
  if (superModifier)
    modifier |= MODIFIERS.super;
  switch (key) {
    case "escape":
    case "esc":
      if (modifier !== 0)
        return false;
      return data === "\x1B" || matchesKittySequence(data, CODEPOINTS.escape, 0) || matchesModifyOtherKeys(data, CODEPOINTS.escape, 0);
    case "space":
      if (!_kittyProtocolActive) {
        if (modifier === MODIFIERS.ctrl && data === "\0") {
          return true;
        }
        if (modifier === MODIFIERS.alt && data === "\x1B ") {
          return true;
        }
      }
      if (modifier === 0) {
        return data === " " || matchesKittySequence(data, CODEPOINTS.space, 0) || matchesModifyOtherKeys(data, CODEPOINTS.space, 0);
      }
      return matchesKittySequence(data, CODEPOINTS.space, modifier) || matchesModifyOtherKeys(data, CODEPOINTS.space, modifier);
    case "tab":
      if (modifier === MODIFIERS.shift) {
        return data === "\x1B[Z" || matchesKittySequence(data, CODEPOINTS.tab, MODIFIERS.shift) || matchesModifyOtherKeys(data, CODEPOINTS.tab, MODIFIERS.shift);
      }
      if (modifier === 0) {
        return data === "	" || matchesKittySequence(data, CODEPOINTS.tab, 0);
      }
      return matchesKittySequence(data, CODEPOINTS.tab, modifier) || matchesModifyOtherKeys(data, CODEPOINTS.tab, modifier);
    case "enter":
    case "return":
      if (modifier === MODIFIERS.shift) {
        if (matchesKittySequence(data, CODEPOINTS.enter, MODIFIERS.shift) || matchesKittySequence(data, CODEPOINTS.kpEnter, MODIFIERS.shift)) {
          return true;
        }
        if (matchesModifyOtherKeys(data, CODEPOINTS.enter, MODIFIERS.shift)) {
          return true;
        }
        if (_kittyProtocolActive) {
          return data === "\x1B\r" || data === "\n";
        }
        return false;
      }
      if (modifier === MODIFIERS.alt) {
        if (matchesKittySequence(data, CODEPOINTS.enter, MODIFIERS.alt) || matchesKittySequence(data, CODEPOINTS.kpEnter, MODIFIERS.alt)) {
          return true;
        }
        if (matchesModifyOtherKeys(data, CODEPOINTS.enter, MODIFIERS.alt)) {
          return true;
        }
        if (!_kittyProtocolActive) {
          return data === "\x1B\r";
        }
        return false;
      }
      if (modifier === 0) {
        return data === "\r" || !_kittyProtocolActive && data === "\n" || data === "\x1BOM" || // SS3 M (numpad enter in some terminals)
        matchesKittySequence(data, CODEPOINTS.enter, 0) || matchesKittySequence(data, CODEPOINTS.kpEnter, 0);
      }
      return matchesKittySequence(data, CODEPOINTS.enter, modifier) || matchesKittySequence(data, CODEPOINTS.kpEnter, modifier) || matchesModifyOtherKeys(data, CODEPOINTS.enter, modifier);
    case "backspace":
      if (modifier === MODIFIERS.alt) {
        if (data === "\x1B\x7F" || data === "\x1B\b") {
          return true;
        }
        return matchesKittySequence(data, CODEPOINTS.backspace, MODIFIERS.alt) || matchesModifyOtherKeys(data, CODEPOINTS.backspace, MODIFIERS.alt);
      }
      if (modifier === MODIFIERS.ctrl) {
        if (matchesRawBackspace(data, MODIFIERS.ctrl))
          return true;
        return matchesKittySequence(data, CODEPOINTS.backspace, MODIFIERS.ctrl) || matchesModifyOtherKeys(data, CODEPOINTS.backspace, MODIFIERS.ctrl);
      }
      if (modifier === 0) {
        return matchesRawBackspace(data, 0) || matchesKittySequence(data, CODEPOINTS.backspace, 0) || matchesModifyOtherKeys(data, CODEPOINTS.backspace, 0);
      }
      return matchesKittySequence(data, CODEPOINTS.backspace, modifier) || matchesModifyOtherKeys(data, CODEPOINTS.backspace, modifier);
    case "insert":
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.insert) || matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.insert, 0);
      }
      if (matchesLegacyModifierSequence(data, "insert", modifier)) {
        return true;
      }
      return matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.insert, modifier);
    case "delete":
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.delete) || matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.delete, 0);
      }
      if (matchesLegacyModifierSequence(data, "delete", modifier)) {
        return true;
      }
      return matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.delete, modifier);
    case "clear":
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.clear);
      }
      return matchesLegacyModifierSequence(data, "clear", modifier);
    case "home":
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.home) || matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.home, 0);
      }
      if (matchesLegacyModifierSequence(data, "home", modifier)) {
        return true;
      }
      return matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.home, modifier);
    case "end":
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.end) || matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.end, 0);
      }
      if (matchesLegacyModifierSequence(data, "end", modifier)) {
        return true;
      }
      return matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.end, modifier);
    case "pageup":
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.pageUp) || matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.pageUp, 0);
      }
      if (matchesLegacyModifierSequence(data, "pageUp", modifier)) {
        return true;
      }
      return matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.pageUp, modifier);
    case "pagedown":
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.pageDown) || matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.pageDown, 0);
      }
      if (matchesLegacyModifierSequence(data, "pageDown", modifier)) {
        return true;
      }
      return matchesKittySequence(data, FUNCTIONAL_CODEPOINTS.pageDown, modifier);
    case "up":
      if (modifier === MODIFIERS.alt) {
        return data === "\x1Bp" || matchesKittySequence(data, ARROW_CODEPOINTS.up, MODIFIERS.alt);
      }
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.up) || matchesKittySequence(data, ARROW_CODEPOINTS.up, 0);
      }
      if (matchesLegacyModifierSequence(data, "up", modifier)) {
        return true;
      }
      return matchesKittySequence(data, ARROW_CODEPOINTS.up, modifier);
    case "down":
      if (modifier === MODIFIERS.alt) {
        return data === "\x1Bn" || matchesKittySequence(data, ARROW_CODEPOINTS.down, MODIFIERS.alt);
      }
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.down) || matchesKittySequence(data, ARROW_CODEPOINTS.down, 0);
      }
      if (matchesLegacyModifierSequence(data, "down", modifier)) {
        return true;
      }
      return matchesKittySequence(data, ARROW_CODEPOINTS.down, modifier);
    case "left":
      if (modifier === MODIFIERS.alt) {
        return data === "\x1B[1;3D" || !_kittyProtocolActive && data === "\x1BB" || data === "\x1Bb" || matchesKittySequence(data, ARROW_CODEPOINTS.left, MODIFIERS.alt);
      }
      if (modifier === MODIFIERS.ctrl) {
        return data === "\x1B[1;5D" || matchesLegacyModifierSequence(data, "left", MODIFIERS.ctrl) || matchesKittySequence(data, ARROW_CODEPOINTS.left, MODIFIERS.ctrl);
      }
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.left) || matchesKittySequence(data, ARROW_CODEPOINTS.left, 0);
      }
      if (matchesLegacyModifierSequence(data, "left", modifier)) {
        return true;
      }
      return matchesKittySequence(data, ARROW_CODEPOINTS.left, modifier);
    case "right":
      if (modifier === MODIFIERS.alt) {
        return data === "\x1B[1;3C" || !_kittyProtocolActive && data === "\x1BF" || data === "\x1Bf" || matchesKittySequence(data, ARROW_CODEPOINTS.right, MODIFIERS.alt);
      }
      if (modifier === MODIFIERS.ctrl) {
        return data === "\x1B[1;5C" || matchesLegacyModifierSequence(data, "right", MODIFIERS.ctrl) || matchesKittySequence(data, ARROW_CODEPOINTS.right, MODIFIERS.ctrl);
      }
      if (modifier === 0) {
        return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES.right) || matchesKittySequence(data, ARROW_CODEPOINTS.right, 0);
      }
      if (matchesLegacyModifierSequence(data, "right", modifier)) {
        return true;
      }
      return matchesKittySequence(data, ARROW_CODEPOINTS.right, modifier);
    case "f1":
    case "f2":
    case "f3":
    case "f4":
    case "f5":
    case "f6":
    case "f7":
    case "f8":
    case "f9":
    case "f10":
    case "f11":
    case "f12": {
      if (modifier !== 0) {
        return false;
      }
      const functionKey = key;
      return matchesLegacySequence(data, LEGACY_KEY_SEQUENCES[functionKey]);
    }
  }
  if (key.length === 1 && (key >= "a" && key <= "z" || isDigitKey(key) || SYMBOL_KEYS.has(key))) {
    const codepoint = key.charCodeAt(0);
    const rawCtrl = rawCtrlChar(key);
    const isLetter = key >= "a" && key <= "z";
    const isDigit = isDigitKey(key);
    if (modifier === MODIFIERS.ctrl + MODIFIERS.alt && !_kittyProtocolActive && rawCtrl) {
      if (data === `\x1B${rawCtrl}`)
        return true;
    }
    if (modifier === MODIFIERS.alt && !_kittyProtocolActive && (isLetter || isDigit || SYMBOL_KEYS.has(key))) {
      if (data === `\x1B${key}`)
        return true;
    }
    if (modifier === MODIFIERS.ctrl) {
      if (rawCtrl && data === rawCtrl)
        return true;
      return matchesKittySequence(data, codepoint, MODIFIERS.ctrl) || matchesPrintableModifyOtherKeys(data, codepoint, MODIFIERS.ctrl);
    }
    if (modifier === MODIFIERS.shift + MODIFIERS.ctrl) {
      return matchesKittySequence(data, codepoint, MODIFIERS.shift + MODIFIERS.ctrl) || matchesPrintableModifyOtherKeys(data, codepoint, MODIFIERS.shift + MODIFIERS.ctrl);
    }
    if (modifier === MODIFIERS.shift) {
      if (isLetter && data === key.toUpperCase())
        return true;
      return matchesKittySequence(data, codepoint, MODIFIERS.shift) || matchesPrintableModifyOtherKeys(data, codepoint, MODIFIERS.shift);
    }
    if (modifier !== 0) {
      return matchesKittySequence(data, codepoint, modifier) || matchesPrintableModifyOtherKeys(data, codepoint, modifier);
    }
    return data === key || matchesKittySequence(data, codepoint, 0);
  }
  return false;
}
var KITTY_CSI_U_REGEX = /^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$/;
var KITTY_PRINTABLE_ALLOWED_MODIFIERS = MODIFIERS.shift | LOCK_MASK;
function decodeKittyPrintable(data) {
  const match = data.match(KITTY_CSI_U_REGEX);
  if (!match)
    return void 0;
  const codepoint = Number.parseInt(match[1] ?? "", 10);
  if (!Number.isFinite(codepoint))
    return void 0;
  const shiftedKey = match[2] && match[2].length > 0 ? Number.parseInt(match[2], 10) : void 0;
  const modValue = match[4] ? Number.parseInt(match[4], 10) : 1;
  const modifier = Number.isFinite(modValue) ? modValue - 1 : 0;
  if ((modifier & ~KITTY_PRINTABLE_ALLOWED_MODIFIERS) !== 0)
    return void 0;
  if (modifier & (MODIFIERS.alt | MODIFIERS.ctrl))
    return void 0;
  let effectiveCodepoint = codepoint;
  if (modifier & MODIFIERS.shift && typeof shiftedKey === "number") {
    effectiveCodepoint = shiftedKey;
  }
  effectiveCodepoint = normalizeKittyFunctionalCodepoint(effectiveCodepoint);
  if (!Number.isFinite(effectiveCodepoint) || effectiveCodepoint < 32)
    return void 0;
  try {
    return String.fromCodePoint(effectiveCodepoint);
  } catch {
    return void 0;
  }
}
function decodeModifyOtherKeysPrintable(data) {
  const parsed = parseModifyOtherKeysSequence(data);
  if (!parsed)
    return void 0;
  const modifier = parsed.modifier & ~LOCK_MASK;
  if ((modifier & ~MODIFIERS.shift) !== 0)
    return void 0;
  if (!Number.isFinite(parsed.codepoint) || parsed.codepoint < 32)
    return void 0;
  try {
    return String.fromCodePoint(parsed.codepoint);
  } catch {
    return void 0;
  }
}
function decodePrintableKey(data) {
  return decodeKittyPrintable(data) ?? decodeModifyOtherKeysPrintable(data);
}

// node_modules/@earendil-works/pi-tui/dist/terminal-colors.js
function hexToRgb(hex) {
  const normalized = hex.startsWith("#") ? hex.slice(1) : hex;
  const r = parseInt(normalized.slice(0, 2), 16);
  const g2 = parseInt(normalized.slice(2, 4), 16);
  const b2 = parseInt(normalized.slice(4, 6), 16);
  return { r, g: g2, b: b2 };
}
function parseOscHexChannel(channel) {
  if (!/^[0-9a-f]+$/i.test(channel)) {
    return void 0;
  }
  const max = 16 ** channel.length - 1;
  if (max <= 0) {
    return void 0;
  }
  return Math.round(parseInt(channel, 16) / max * 255);
}
var OSC11_BACKGROUND_COLOR_RESPONSE_PATTERN = /^\x1b\]11;([^\x07\x1b]*)(?:\x07|\x1b\\)$/i;
var COLOR_SCHEME_REPORT_PATTERN = /^(?:\x1b\[\?997;(1|2)n)+$/;
function isOsc11BackgroundColorResponse(data) {
  return OSC11_BACKGROUND_COLOR_RESPONSE_PATTERN.test(data);
}
function parseOsc11BackgroundColor(data) {
  const match = data.match(OSC11_BACKGROUND_COLOR_RESPONSE_PATTERN);
  if (!match) {
    return void 0;
  }
  const value = match[1].trim();
  if (value.startsWith("#")) {
    const hex = value.slice(1);
    if (/^[0-9a-f]{6}$/i.test(hex)) {
      return hexToRgb(value);
    }
    if (/^[0-9a-f]{12}$/i.test(hex)) {
      const r2 = parseOscHexChannel(hex.slice(0, 4));
      const g3 = parseOscHexChannel(hex.slice(4, 8));
      const b3 = parseOscHexChannel(hex.slice(8, 12));
      return r2 !== void 0 && g3 !== void 0 && b3 !== void 0 ? { r: r2, g: g3, b: b3 } : void 0;
    }
    return void 0;
  }
  const rgbValue = value.replace(/^rgba?:/i, "");
  const [red, green, blue] = rgbValue.split("/");
  if (red === void 0 || green === void 0 || blue === void 0) {
    return void 0;
  }
  const r = parseOscHexChannel(red);
  const g2 = parseOscHexChannel(green);
  const b2 = parseOscHexChannel(blue);
  return r !== void 0 && g2 !== void 0 && b2 !== void 0 ? { r, g: g2, b: b2 } : void 0;
}
function parseTerminalColorSchemeReport(data) {
  const match = data.match(COLOR_SCHEME_REPORT_PATTERN);
  if (!match) {
    return void 0;
  }
  return match[1] === "2" ? "light" : "dark";
}

// node_modules/@earendil-works/pi-tui/dist/terminal-image.js
import { execSync } from "node:child_process";
var cachedCapabilities = null;
var capabilityOverrides = {};
var cellDimensions = { widthPx: 9, heightPx: 18 };
function setCellDimensions(dims) {
  cellDimensions = dims;
}
function probeTmuxHyperlinks() {
  try {
    const termfeatures = execSync("tmux display-message -p '#{client_termfeatures}'", {
      encoding: "utf8",
      timeout: 250,
      stdio: ["ignore", "pipe", "ignore"]
    });
    return termfeatures.split(",").map((feature) => feature.trim()).includes("hyperlinks");
  } catch {
    return false;
  }
}
function detectCapabilitiesFromEnvironment(tmuxForwardsHyperlink) {
  const termProgram = process.env.TERM_PROGRAM?.toLowerCase() || "";
  const terminalEmulator = process.env.TERMINAL_EMULATOR?.toLowerCase() || "";
  const term = process.env.TERM?.toLowerCase() || "";
  const colorTerm = process.env.COLORTERM?.toLowerCase() || "";
  const hasTrueColorHint = colorTerm === "truecolor" || colorTerm === "24bit";
  const isWindowsConsole = process.platform === "win32";
  if (process.env.TMUX || term.startsWith("tmux")) {
    return { images: null, trueColor: hasTrueColorHint, hyperlinks: tmuxForwardsHyperlink() };
  }
  if (term.startsWith("screen")) {
    return { images: null, trueColor: hasTrueColorHint, hyperlinks: false };
  }
  if (process.env.KITTY_WINDOW_ID || termProgram === "kitty") {
    return { images: "kitty", trueColor: true, hyperlinks: true };
  }
  if (termProgram === "ghostty" || term.includes("ghostty") || process.env.GHOSTTY_RESOURCES_DIR) {
    return { images: "kitty", trueColor: true, hyperlinks: true };
  }
  if (process.env.WEZTERM_PANE || termProgram === "wezterm") {
    return { images: "kitty", trueColor: true, hyperlinks: true };
  }
  if (termProgram === "warpterminal" || process.env.WARP_SESSION_ID || process.env.WARP_TERMINAL_SESSION_UUID) {
    return { images: "kitty", trueColor: true, hyperlinks: true };
  }
  if (process.env.ITERM_SESSION_ID || termProgram === "iterm.app") {
    return { images: "iterm2", trueColor: true, hyperlinks: true };
  }
  if (process.env.WT_SESSION) {
    return { images: null, trueColor: true, hyperlinks: true };
  }
  if (termProgram === "alacritty" || termProgram === "vscode" || termProgram === "zed") {
    return { images: null, trueColor: true, hyperlinks: true };
  }
  if (terminalEmulator === "jetbrains-jediterm") {
    return { images: null, trueColor: true, hyperlinks: false };
  }
  if (isWindowsConsole) {
    return { images: null, trueColor: true, hyperlinks: false };
  }
  return { images: null, trueColor: hasTrueColorHint, hyperlinks: false };
}
function parseBooleanCapabilityOverride(value) {
  return value === "1" ? true : value === "0" ? false : void 0;
}
function detectCapabilities(tmuxForwardsHyperlink = probeTmuxHyperlinks) {
  const hyperlinks = parseBooleanCapabilityOverride(process.env.PI_HYPERLINKS);
  const detected = detectCapabilitiesFromEnvironment(hyperlinks === void 0 ? tmuxForwardsHyperlink : () => hyperlinks);
  const imageProtocol = process.env.PI_IMAGE_PROTOCOL?.toLowerCase();
  const images = imageProtocol === "kitty" || imageProtocol === "iterm2" ? imageProtocol : imageProtocol === "none" || imageProtocol === "0" ? null : void 0;
  const trueColor = parseBooleanCapabilityOverride(process.env.PI_TRUE_COLOR);
  return {
    ...detected,
    ...images !== void 0 ? { images } : {},
    ...trueColor !== void 0 ? { trueColor } : {},
    ...hyperlinks !== void 0 ? { hyperlinks } : {}
  };
}
function getCapabilities() {
  if (!cachedCapabilities) {
    const hyperlinks = capabilityOverrides.hyperlinks;
    cachedCapabilities = {
      ...detectCapabilities(hyperlinks === void 0 ? void 0 : () => hyperlinks),
      ...capabilityOverrides
    };
  }
  return cachedCapabilities;
}
function setCapabilities(caps) {
  cachedCapabilities = caps;
}
var KITTY_PREFIX = "\x1B_G";
var ITERM2_PREFIX = "\x1B]1337;File=";
function isImageLine(line) {
  if (line.startsWith(KITTY_PREFIX) || line.startsWith(ITERM2_PREFIX)) {
    return true;
  }
  return line.includes(KITTY_PREFIX) || line.includes(ITERM2_PREFIX);
}
function deleteKittyImage(imageId) {
  return `\x1B_Ga=d,d=I,i=${imageId},q=2\x1B\\`;
}
function deleteAllKittyImages() {
  return "\x1B_Ga=d,d=A,q=2\x1B\\";
}
function deleteAllKittyPlacements() {
  return "\x1B_Ga=d,d=a,q=2\x1B\\";
}
var kittyImageMetadata = /* @__PURE__ */ new Map();
function getRegisteredKittyImageMetadata(line) {
  const controls = /\x1b_G([^;]*);/.exec(line)?.[1];
  if (!controls)
    return void 0;
  const imageId = /(?:^|,)i=(\d+)(?:,|$)/.exec(controls)?.[1];
  return imageId === void 0 ? void 0 : kittyImageMetadata.get(Number.parseInt(imageId, 10));
}
function getKittyImageMetadata(line) {
  const metadata = getRegisteredKittyImageMetadata(line);
  if (!metadata)
    return void 0;
  return {
    imageId: metadata.imageId,
    columns: metadata.columns,
    rows: metadata.rows,
    widthPx: metadata.widthPx,
    heightPx: metadata.heightPx
  };
}
var KITTY_PLACEMENT_CONTROL_KEYS = /* @__PURE__ */ new Set([
  "i",
  "p",
  "x",
  "y",
  "w",
  "h",
  "X",
  "Y",
  "c",
  "r",
  "C",
  "U",
  "z",
  "P",
  "Q",
  "H",
  "V"
]);
function getKittyImagePlacement(line) {
  const match = /\x1b_G([^;]*);/.exec(line);
  const metadata = getRegisteredKittyImageMetadata(line);
  if (!match || !metadata)
    return void 0;
  let commandStart = match.index;
  let commandControls = match[1];
  let transmissionEnd;
  while (true) {
    const terminator = line.indexOf("\x1B\\", commandStart + KITTY_PREFIX.length);
    if (terminator === -1)
      return void 0;
    transmissionEnd = terminator + 2;
    if (!/(?:^|,)m=1(?:,|$)/.test(commandControls))
      break;
    commandStart = transmissionEnd;
    if (!line.startsWith(KITTY_PREFIX, commandStart))
      return void 0;
    const controlsEnd = line.indexOf(";", commandStart + KITTY_PREFIX.length);
    if (controlsEnd === -1)
      return void 0;
    commandControls = line.slice(commandStart + KITTY_PREFIX.length, controlsEnd);
  }
  const controls = match[1].split(",").filter((control) => KITTY_PLACEMENT_CONTROL_KEYS.has(control.split("=", 1)[0] ?? ""));
  const sequence = `\x1B_Ga=p,q=2,${controls.join(",")}\x1B\\`;
  return {
    imageId: metadata.imageId,
    transmissionGeneration: metadata.transmissionGeneration,
    transmissionBytes: transmissionEnd - match.index,
    estimatedDecodedBytes: metadata.widthPx * metadata.heightPx * 4,
    sequence,
    replacementLine: `${line.slice(0, match.index)}${sequence}${line.slice(transmissionEnd)}`
  };
}
function cropKittyImageLine(line, hiddenRows, visibleRows) {
  const metadata = getKittyImageMetadata(line);
  const match = /\x1b_G([^;]*);/.exec(line);
  if (!metadata || !match || hiddenRows < 0 || hiddenRows >= metadata.rows || visibleRows <= 0)
    return line;
  const croppedRows = Math.min(visibleRows, metadata.rows - hiddenRows);
  if (hiddenRows === 0 && croppedRows === metadata.rows)
    return line;
  const sourceY = Math.floor(metadata.heightPx * hiddenRows / metadata.rows);
  const sourceEnd = Math.ceil(metadata.heightPx * (hiddenRows + croppedRows) / metadata.rows);
  const sourceHeight = Math.max(1, Math.min(metadata.heightPx, sourceEnd) - sourceY);
  const controls = match[1].split(",").filter((control) => !/^[yhr]=/.test(control));
  controls.push(`y=${sourceY}`, `h=${sourceHeight}`, `r=${croppedRows}`);
  return `${line.slice(0, match.index)}\x1B_G${controls.join(",")};${line.slice(match.index + match[0].length)}`;
}
function hyperlink(text, url) {
  return `\x1B]8;;${url}\x1B\\${text}\x1B]8;;\x1B\\`;
}

// node_modules/get-east-asian-width/lookup-data.js
var ambiguousMinimalCodePoint = 161;
var ambiguousMaximumCodePoint = 1114109;
var ambiguousRanges = [161, 161, 164, 164, 167, 168, 170, 170, 173, 174, 176, 180, 182, 186, 188, 191, 198, 198, 208, 208, 215, 216, 222, 225, 230, 230, 232, 234, 236, 237, 240, 240, 242, 243, 247, 250, 252, 252, 254, 254, 257, 257, 273, 273, 275, 275, 283, 283, 294, 295, 299, 299, 305, 307, 312, 312, 319, 322, 324, 324, 328, 331, 333, 333, 338, 339, 358, 359, 363, 363, 462, 462, 464, 464, 466, 466, 468, 468, 470, 470, 472, 472, 474, 474, 476, 476, 593, 593, 609, 609, 708, 708, 711, 711, 713, 715, 717, 717, 720, 720, 728, 731, 733, 733, 735, 735, 768, 879, 913, 929, 931, 937, 945, 961, 963, 969, 1025, 1025, 1040, 1103, 1105, 1105, 8208, 8208, 8211, 8214, 8216, 8217, 8220, 8221, 8224, 8226, 8228, 8231, 8240, 8240, 8242, 8243, 8245, 8245, 8251, 8251, 8254, 8254, 8308, 8308, 8319, 8319, 8321, 8324, 8364, 8364, 8451, 8451, 8453, 8453, 8457, 8457, 8467, 8467, 8470, 8470, 8481, 8482, 8486, 8486, 8491, 8491, 8531, 8532, 8539, 8542, 8544, 8555, 8560, 8569, 8585, 8585, 8592, 8601, 8632, 8633, 8658, 8658, 8660, 8660, 8679, 8679, 8704, 8704, 8706, 8707, 8711, 8712, 8715, 8715, 8719, 8719, 8721, 8721, 8725, 8725, 8730, 8730, 8733, 8736, 8739, 8739, 8741, 8741, 8743, 8748, 8750, 8750, 8756, 8759, 8764, 8765, 8776, 8776, 8780, 8780, 8786, 8786, 8800, 8801, 8804, 8807, 8810, 8811, 8814, 8815, 8834, 8835, 8838, 8839, 8853, 8853, 8857, 8857, 8869, 8869, 8895, 8895, 8978, 8978, 9312, 9449, 9451, 9547, 9552, 9587, 9600, 9615, 9618, 9621, 9632, 9633, 9635, 9641, 9650, 9651, 9654, 9655, 9660, 9661, 9664, 9665, 9670, 9672, 9675, 9675, 9678, 9681, 9698, 9701, 9711, 9711, 9733, 9734, 9737, 9737, 9742, 9743, 9756, 9756, 9758, 9758, 9792, 9792, 9794, 9794, 9824, 9825, 9827, 9829, 9831, 9834, 9836, 9837, 9839, 9839, 9886, 9887, 9919, 9919, 9926, 9933, 9935, 9939, 9941, 9953, 9955, 9955, 9960, 9961, 9963, 9969, 9972, 9972, 9974, 9977, 9979, 9980, 9982, 9983, 10045, 10045, 10102, 10111, 11094, 11097, 12872, 12879, 57344, 63743, 65024, 65039, 65533, 65533, 127232, 127242, 127248, 127277, 127280, 127337, 127344, 127373, 127375, 127376, 127387, 127404, 917760, 917999, 983040, 1048573, 1048576, 1114109];
var fullwidthMinimalCodePoint = 12288;
var fullwidthMaximumCodePoint = 65510;
var fullwidthRanges = [12288, 12288, 65281, 65376, 65504, 65510];
var wideMinimalCodePoint = 4352;
var wideMaximumCodePoint = 262141;
var wideRanges = [4352, 4447, 8986, 8987, 9001, 9002, 9193, 9196, 9200, 9200, 9203, 9203, 9725, 9726, 9748, 9749, 9776, 9783, 9800, 9811, 9855, 9855, 9866, 9871, 9875, 9875, 9889, 9889, 9898, 9899, 9917, 9918, 9924, 9925, 9934, 9934, 9940, 9940, 9962, 9962, 9970, 9971, 9973, 9973, 9978, 9978, 9981, 9981, 9989, 9989, 9994, 9995, 10024, 10024, 10060, 10060, 10062, 10062, 10067, 10069, 10071, 10071, 10133, 10135, 10160, 10160, 10175, 10175, 11035, 11036, 11088, 11088, 11093, 11093, 11904, 11929, 11931, 12019, 12032, 12245, 12272, 12287, 12289, 12350, 12353, 12438, 12441, 12543, 12549, 12591, 12593, 12686, 12688, 12773, 12783, 12830, 12832, 12871, 12880, 42124, 42128, 42182, 43360, 43388, 44032, 55203, 63744, 64255, 65040, 65049, 65072, 65106, 65108, 65126, 65128, 65131, 94176, 94180, 94192, 94198, 94208, 101589, 101631, 101662, 101760, 101874, 110576, 110579, 110581, 110587, 110589, 110590, 110592, 110882, 110898, 110898, 110928, 110930, 110933, 110933, 110948, 110951, 110960, 111355, 119552, 119638, 119648, 119670, 126980, 126980, 127183, 127183, 127374, 127374, 127377, 127386, 127488, 127490, 127504, 127547, 127552, 127560, 127568, 127569, 127584, 127589, 127744, 127776, 127789, 127797, 127799, 127868, 127870, 127891, 127904, 127946, 127951, 127955, 127968, 127984, 127988, 127988, 127992, 128062, 128064, 128064, 128066, 128252, 128255, 128317, 128331, 128334, 128336, 128359, 128378, 128378, 128405, 128406, 128420, 128420, 128507, 128591, 128640, 128709, 128716, 128716, 128720, 128722, 128725, 128728, 128732, 128735, 128747, 128748, 128756, 128764, 128992, 129003, 129008, 129008, 129292, 129338, 129340, 129349, 129351, 129535, 129648, 129660, 129664, 129674, 129678, 129734, 129736, 129736, 129741, 129756, 129759, 129770, 129775, 129784, 131072, 196605, 196608, 262141];

// node_modules/get-east-asian-width/utilities.js
var isInRange = (ranges, codePoint) => {
  let low = 0;
  let high = Math.floor(ranges.length / 2) - 1;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const i = mid * 2;
    if (codePoint < ranges[i]) {
      high = mid - 1;
    } else if (codePoint > ranges[i + 1]) {
      low = mid + 1;
    } else {
      return true;
    }
  }
  return false;
};

// node_modules/get-east-asian-width/lookup.js
var commonCjkCodePoint = 19968;
var [wideFastPathStart, wideFastPathEnd] = /* @__PURE__ */ findWideFastPathRange(wideRanges);
function findWideFastPathRange(ranges) {
  let fastPathStart = ranges[0];
  let fastPathEnd = ranges[1];
  for (let index = 0; index < ranges.length; index += 2) {
    const start = ranges[index];
    const end = ranges[index + 1];
    if (commonCjkCodePoint >= start && commonCjkCodePoint <= end) {
      return [start, end];
    }
    if (end - start > fastPathEnd - fastPathStart) {
      fastPathStart = start;
      fastPathEnd = end;
    }
  }
  return [fastPathStart, fastPathEnd];
}
var isAmbiguous = (codePoint) => {
  if (codePoint < ambiguousMinimalCodePoint || codePoint > ambiguousMaximumCodePoint) {
    return false;
  }
  return isInRange(ambiguousRanges, codePoint);
};
var isFullWidth = (codePoint) => {
  if (codePoint < fullwidthMinimalCodePoint || codePoint > fullwidthMaximumCodePoint) {
    return false;
  }
  return isInRange(fullwidthRanges, codePoint);
};
var isWide = (codePoint) => {
  if (codePoint >= wideFastPathStart && codePoint <= wideFastPathEnd) {
    return true;
  }
  if (codePoint < wideMinimalCodePoint || codePoint > wideMaximumCodePoint) {
    return false;
  }
  return isInRange(wideRanges, codePoint);
};

// node_modules/get-east-asian-width/index.js
function validate(codePoint) {
  if (!Number.isSafeInteger(codePoint)) {
    throw new TypeError(`Expected a code point, got \`${typeof codePoint}\`.`);
  }
}
function eastAsianWidth(codePoint, { ambiguousAsWide = false } = {}) {
  validate(codePoint);
  if (isFullWidth(codePoint) || isWide(codePoint) || ambiguousAsWide && isAmbiguous(codePoint)) {
    return 2;
  }
  return 1;
}

// node_modules/@earendil-works/pi-tui/dist/utils.js
var graphemeSegmenter = new Intl.Segmenter(void 0, { granularity: "grapheme" });
var wordSegmenter = new Intl.Segmenter(void 0, { granularity: "word" });
function getGraphemeSegmenter() {
  return graphemeSegmenter;
}
function getWordSegmenter() {
  return wordSegmenter;
}
function couldBeEmoji(segment) {
  const cp = segment.codePointAt(0);
  return cp >= 126976 && cp <= 130047 || // Emoji and Pictograph
  cp >= 8960 && cp <= 9215 || // Misc technical
  cp >= 9728 && cp <= 10175 || // Misc symbols, dingbats
  cp >= 11088 && cp <= 11093 || // Specific stars/circles
  segment.includes("\uFE0F") || // Contains VS16 (emoji presentation selector)
  segment.length > 2;
}
var zeroWidthRegex = new RegExp("^(?:\\p{Default_Ignorable_Code_Point}|\\p{Control}|\\p{Mark}|\\p{Surrogate})+$", "v");
var leadingNonPrintingRegex = new RegExp("^[\\p{Default_Ignorable_Code_Point}\\p{Control}\\p{Format}\\p{Mark}\\p{Surrogate}]+", "v");
var nonPrintingCharRegex = new RegExp("^(?:\\p{Default_Ignorable_Code_Point}|\\p{Control}|\\p{Format}|\\p{Mark}|\\p{Surrogate})$", "v");
var markCharRegex = new RegExp("^\\p{Mark}$", "v");
var terminalSpacingMarkRegex = new RegExp("^(?:[\\p{Spacing_Mark}--[\\u1734\\u302E\\u302F]]|[\\u065F\\u0F7F\\u102B\\u102C\\u1031\\u1033-\\u1035\\u1038\\u103A-\\u103E])+$", "v");
var rgiEmojiRegex = new RegExp("^\\p{RGI_Emoji}$", "v");
var WIDTH_CACHE_SIZE = 512;
var widthCache = /* @__PURE__ */ new Map();
var cjkBreakRegex = /[\p{Script_Extensions=Han}\p{Script_Extensions=Hiragana}\p{Script_Extensions=Katakana}\p{Script_Extensions=Hangul}\p{Script_Extensions=Bopomofo}]/u;
function isPrintableAscii(str) {
  for (let i = 0; i < str.length; i++) {
    const code = str.charCodeAt(i);
    if (code < 32 || code > 126) {
      return false;
    }
  }
  return true;
}
function truncateFragmentToWidth(text, maxWidth) {
  if (maxWidth <= 0 || text.length === 0) {
    return { text: "", width: 0 };
  }
  if (isPrintableAscii(text)) {
    const clipped = text.slice(0, maxWidth);
    return { text: clipped, width: clipped.length };
  }
  const hasAnsi = text.includes("\x1B");
  const hasTabs = text.includes("	");
  if (!hasAnsi && !hasTabs) {
    let result2 = "";
    let width2 = 0;
    for (const { segment } of graphemeSegmenter.segment(text)) {
      const w2 = graphemeWidth(segment);
      if (width2 + w2 > maxWidth) {
        break;
      }
      result2 += segment;
      width2 += w2;
    }
    return { text: result2, width: width2 };
  }
  let result = "";
  let width = 0;
  let i = 0;
  let pendingAnsi = "";
  while (i < text.length) {
    const ansi = extractAnsiCode(text, i);
    if (ansi) {
      pendingAnsi += ansi.code;
      i += ansi.length;
      continue;
    }
    if (text[i] === "	") {
      if (width + 3 > maxWidth) {
        break;
      }
      if (pendingAnsi) {
        result += pendingAnsi;
        pendingAnsi = "";
      }
      result += "	";
      width += 3;
      i++;
      continue;
    }
    let end = i;
    while (end < text.length && text[end] !== "	") {
      const nextAnsi = extractAnsiCode(text, end);
      if (nextAnsi) {
        break;
      }
      end++;
    }
    for (const { segment } of graphemeSegmenter.segment(text.slice(i, end))) {
      const w2 = graphemeWidth(segment);
      if (width + w2 > maxWidth) {
        return { text: result, width };
      }
      if (pendingAnsi) {
        result += pendingAnsi;
        pendingAnsi = "";
      }
      result += segment;
      width += w2;
    }
    i = end;
  }
  return { text: result, width };
}
function finalizeTruncatedResult(prefix, prefixWidth, ellipsis, ellipsisWidth, maxWidth, pad) {
  const reset = "\x1B[0m";
  const hyperlinkClose = getActiveOsc8Close(prefix);
  const visibleWidth2 = prefixWidth + ellipsisWidth;
  let result;
  if (ellipsis.length > 0) {
    result = `${prefix}${hyperlinkClose}${reset}${ellipsis}${reset}`;
  } else {
    result = `${prefix}${hyperlinkClose}${reset}`;
  }
  return pad ? result + " ".repeat(Math.max(0, maxWidth - visibleWidth2)) : result;
}
function graphemeWidth(segment) {
  if (segment === "	") {
    return 3;
  }
  if (terminalSpacingMarkRegex.test(segment)) {
    return [...segment].length;
  }
  if (zeroWidthRegex.test(segment)) {
    return 0;
  }
  if (couldBeEmoji(segment) && rgiEmojiRegex.test(segment)) {
    return 2;
  }
  const base = segment.replace(leadingNonPrintingRegex, "");
  const cp = base.codePointAt(0);
  if (cp === void 0) {
    return 0;
  }
  if (cp >= 127462 && cp <= 127487) {
    return 2;
  }
  let width = eastAsianWidth(cp);
  let followsMark = false;
  const chars = [...base];
  for (const char of chars.slice(1)) {
    if (terminalSpacingMarkRegex.test(char)) {
      width += 1;
      followsMark = false;
    } else if (markCharRegex.test(char)) {
      followsMark = true;
    } else if (!nonPrintingCharRegex.test(char)) {
      const c = char.codePointAt(0);
      if (followsMark || c >= 65280 && c <= 65519) {
        width += eastAsianWidth(c);
      } else if (c === 3635 || c === 3763) {
        width += 1;
      }
      followsMark = false;
    }
  }
  return width;
}
function visibleWidth(str) {
  if (str.length === 0) {
    return 0;
  }
  if (isPrintableAscii(str)) {
    return str.length;
  }
  const cached = widthCache.get(str);
  if (cached !== void 0) {
    return cached;
  }
  let clean = str;
  if (str.includes("	")) {
    clean = clean.replace(/\t/g, "   ");
  }
  if (clean.includes("\x1B")) {
    let stripped = "";
    let i = 0;
    while (i < clean.length) {
      const ansi = extractAnsiCode(clean, i);
      if (ansi) {
        i += ansi.length;
        continue;
      }
      stripped += clean[i];
      i++;
    }
    clean = stripped;
  }
  let width = 0;
  for (const { segment } of graphemeSegmenter.segment(clean)) {
    width += graphemeWidth(segment);
  }
  if (widthCache.size >= WIDTH_CACHE_SIZE) {
    const firstKey = widthCache.keys().next().value;
    if (firstKey !== void 0) {
      widthCache.delete(firstKey);
    }
  }
  widthCache.set(str, width);
  return width;
}
function stripTerminalSequences(str) {
  if (!str.includes("\x1B"))
    return str;
  let result = "";
  let i = 0;
  while (i < str.length) {
    const ansi = extractAnsiCode(str, i);
    if (ansi) {
      i += ansi.length;
      continue;
    }
    result += str[i];
    i++;
  }
  return result;
}
function getGraphemeCellRange(line, column) {
  let currentCol = 0;
  let i = 0;
  while (i < line.length) {
    const ansi = extractAnsiCode(line, i);
    if (ansi) {
      i += ansi.length;
      continue;
    }
    let textEnd = i;
    while (textEnd < line.length && !extractAnsiCode(line, textEnd))
      textEnd++;
    for (const { segment } of graphemeSegmenter.segment(line.slice(i, textEnd))) {
      const width = graphemeWidth(segment);
      if (width > 0 && column >= currentCol && column < currentCol + width) {
        return { start: currentCol, end: currentCol + width };
      }
      currentCol += width;
    }
    i = textEnd;
  }
  return void 0;
}
function getOsc8LinkAtColumn(line, column) {
  let activeUrl;
  let currentCol = 0;
  let i = 0;
  while (i < line.length) {
    const ansi = extractAnsiCode(line, i);
    if (ansi) {
      const hyperlink2 = /^\x1b\]8;[^;]*;([^\x07\x1b]*)(?:\x07|\x1b\\)$/.exec(ansi.code);
      if (hyperlink2)
        activeUrl = hyperlink2[1] || void 0;
      i += ansi.length;
      continue;
    }
    let textEnd = i;
    while (textEnd < line.length && !extractAnsiCode(line, textEnd))
      textEnd++;
    for (const { segment } of graphemeSegmenter.segment(line.slice(i, textEnd))) {
      const width = segment === "	" ? 3 : graphemeWidth(segment);
      if (column >= currentCol && column < currentCol + width)
        return activeUrl;
      currentCol += width;
    }
    i = textEnd;
  }
  return void 0;
}
var THAI_LAO_AM_REGEX = /[\u0e33\u0eb3]/;
var THAI_LAO_AM_GLOBAL_REGEX = /[\u0e33\u0eb3]/g;
function normalizeTerminalOutput(str) {
  let normalized = str;
  if (THAI_LAO_AM_REGEX.test(normalized)) {
    normalized = normalized.replace(THAI_LAO_AM_GLOBAL_REGEX, (char) => char === "\u0E33" ? "\u0E4D\u0E32" : "\u0ECD\u0EB2");
  }
  if (!normalized.includes("	"))
    return normalized;
  let result = "";
  let i = 0;
  while (i < normalized.length) {
    const ansi = extractAnsiCode(normalized, i);
    if (ansi) {
      result += ansi.code;
      i += ansi.length;
      continue;
    }
    result += normalized[i] === "	" ? "   " : normalized[i];
    i++;
  }
  return result;
}
function extractAnsiCode(str, pos) {
  if (pos >= str.length || str[pos] !== "\x1B")
    return null;
  const next = str[pos + 1];
  if (next === "[") {
    let j2 = pos + 2;
    while (j2 < str.length && !/[mGKHJ]/.test(str[j2]))
      j2++;
    if (j2 < str.length)
      return { code: str.substring(pos, j2 + 1), length: j2 + 1 - pos };
    return null;
  }
  if (next === "]") {
    let j2 = pos + 2;
    while (j2 < str.length) {
      if (str[j2] === "\x07")
        return { code: str.substring(pos, j2 + 1), length: j2 + 1 - pos };
      if (str[j2] === "\x1B" && str[j2 + 1] === "\\")
        return { code: str.substring(pos, j2 + 2), length: j2 + 2 - pos };
      j2++;
    }
    return null;
  }
  if (next === "_") {
    let j2 = pos + 2;
    while (j2 < str.length) {
      if (str[j2] === "\x07")
        return { code: str.substring(pos, j2 + 1), length: j2 + 1 - pos };
      if (str[j2] === "\x1B" && str[j2 + 1] === "\\")
        return { code: str.substring(pos, j2 + 2), length: j2 + 2 - pos };
      j2++;
    }
    return null;
  }
  return null;
}
function parseOsc8Hyperlink(ansiCode) {
  if (!ansiCode.startsWith("\x1B]8;")) {
    return void 0;
  }
  const terminator = ansiCode.endsWith("\x07") ? "\x07" : "\x1B\\";
  const body = ansiCode.slice(4, terminator === "\x07" ? -1 : -2);
  const separatorIndex = body.indexOf(";");
  if (separatorIndex === -1) {
    return void 0;
  }
  const params = body.slice(0, separatorIndex);
  const url = body.slice(separatorIndex + 1);
  if (!url) {
    return null;
  }
  return { params, url, terminator };
}
function formatOsc8Hyperlink(hyperlink2) {
  return `\x1B]8;${hyperlink2.params};${hyperlink2.url}${hyperlink2.terminator}`;
}
function formatOsc8Close(terminator) {
  return `\x1B]8;;${terminator}`;
}
function getActiveOsc8Close(prefix) {
  if (!prefix.includes("\x1B]8;")) {
    return "";
  }
  let activeHyperlink = null;
  let i = 0;
  while (i < prefix.length) {
    const ansi = extractAnsiCode(prefix, i);
    if (ansi) {
      const hyperlink2 = parseOsc8Hyperlink(ansi.code);
      if (hyperlink2 !== void 0) {
        activeHyperlink = hyperlink2;
      }
      i += ansi.length;
    } else {
      i++;
    }
  }
  return activeHyperlink ? formatOsc8Close(activeHyperlink.terminator) : "";
}
var AnsiCodeTracker = class {
  // Track individual attributes separately so we can reset them specifically
  bold = false;
  dim = false;
  italic = false;
  underline = false;
  blink = false;
  inverse = false;
  hidden = false;
  strikethrough = false;
  fgColor = null;
  // Stores the full code like "31" or "38;5;240"
  bgColor = null;
  // Stores the full code like "41" or "48;5;240"
  activeHyperlink = null;
  process(ansiCode) {
    const hyperlink2 = parseOsc8Hyperlink(ansiCode);
    if (hyperlink2 !== void 0) {
      this.activeHyperlink = hyperlink2;
      return;
    }
    if (!ansiCode.endsWith("m")) {
      return;
    }
    const match = ansiCode.match(/\x1b\[([\d;]*)m/);
    if (!match)
      return;
    const params = match[1];
    if (params === "" || params === "0") {
      this.reset();
      return;
    }
    const parts = params.split(";");
    let i = 0;
    while (i < parts.length) {
      const code = Number.parseInt(parts[i], 10);
      if (code === 38 || code === 48) {
        if (parts[i + 1] === "5" && parts[i + 2] !== void 0) {
          const colorCode = `${parts[i]};${parts[i + 1]};${parts[i + 2]}`;
          if (code === 38) {
            this.fgColor = colorCode;
          } else {
            this.bgColor = colorCode;
          }
          i += 3;
          continue;
        } else if (parts[i + 1] === "2" && parts[i + 4] !== void 0) {
          const colorCode = `${parts[i]};${parts[i + 1]};${parts[i + 2]};${parts[i + 3]};${parts[i + 4]}`;
          if (code === 38) {
            this.fgColor = colorCode;
          } else {
            this.bgColor = colorCode;
          }
          i += 5;
          continue;
        }
      }
      switch (code) {
        case 0:
          this.reset();
          break;
        case 1:
          this.bold = true;
          break;
        case 2:
          this.dim = true;
          break;
        case 3:
          this.italic = true;
          break;
        case 4:
          this.underline = true;
          break;
        case 5:
          this.blink = true;
          break;
        case 7:
          this.inverse = true;
          break;
        case 8:
          this.hidden = true;
          break;
        case 9:
          this.strikethrough = true;
          break;
        case 21:
          this.bold = false;
          break;
        // Some terminals
        case 22:
          this.bold = false;
          this.dim = false;
          break;
        case 23:
          this.italic = false;
          break;
        case 24:
          this.underline = false;
          break;
        case 25:
          this.blink = false;
          break;
        case 27:
          this.inverse = false;
          break;
        case 28:
          this.hidden = false;
          break;
        case 29:
          this.strikethrough = false;
          break;
        case 39:
          this.fgColor = null;
          break;
        // Default fg
        case 49:
          this.bgColor = null;
          break;
        // Default bg
        default:
          if (code >= 30 && code <= 37 || code >= 90 && code <= 97) {
            this.fgColor = String(code);
          } else if (code >= 40 && code <= 47 || code >= 100 && code <= 107) {
            this.bgColor = String(code);
          }
          break;
      }
      i++;
    }
  }
  reset() {
    this.bold = false;
    this.dim = false;
    this.italic = false;
    this.underline = false;
    this.blink = false;
    this.inverse = false;
    this.hidden = false;
    this.strikethrough = false;
    this.fgColor = null;
    this.bgColor = null;
  }
  /** Clear all state for reuse. */
  clear() {
    this.reset();
    this.activeHyperlink = null;
  }
  getActiveCodes() {
    const codes = [];
    if (this.bold)
      codes.push("1");
    if (this.dim)
      codes.push("2");
    if (this.italic)
      codes.push("3");
    if (this.underline)
      codes.push("4");
    if (this.blink)
      codes.push("5");
    if (this.inverse)
      codes.push("7");
    if (this.hidden)
      codes.push("8");
    if (this.strikethrough)
      codes.push("9");
    if (this.fgColor)
      codes.push(this.fgColor);
    if (this.bgColor)
      codes.push(this.bgColor);
    let result = codes.length > 0 ? `\x1B[${codes.join(";")}m` : "";
    if (this.activeHyperlink) {
      result += formatOsc8Hyperlink(this.activeHyperlink);
    }
    return result;
  }
  getActiveBackgroundCode() {
    return this.bgColor ? `\x1B[${this.bgColor}m` : "";
  }
  hasActiveCodes() {
    return this.bold || this.dim || this.italic || this.underline || this.blink || this.inverse || this.hidden || this.strikethrough || this.fgColor !== null || this.bgColor !== null || this.activeHyperlink !== null;
  }
  /**
   * Get reset codes for attributes that need to be turned off at line end.
   * Underline must be closed to prevent bleeding into padding.
   * Active OSC 8 hyperlinks must be closed and re-opened on the next line.
   * Returns empty string if no attributes need closing.
   */
  getLineEndReset() {
    let result = "";
    if (this.underline) {
      result += "\x1B[24m";
    }
    if (this.activeHyperlink) {
      result += formatOsc8Close(this.activeHyperlink.terminator);
    }
    return result;
  }
};
function updateTrackerFromText(text, tracker) {
  let i = 0;
  while (i < text.length) {
    const ansiResult = extractAnsiCode(text, i);
    if (ansiResult) {
      tracker.process(ansiResult.code);
      i += ansiResult.length;
    } else {
      i++;
    }
  }
}
function getActiveBackgroundAnsi(text) {
  const tracker = new AnsiCodeTracker();
  updateTrackerFromText(text, tracker);
  return tracker.getActiveBackgroundCode();
}
function splitIntoTokensWithAnsi(text) {
  const tokens = [];
  let current = "";
  let pendingAnsi = "";
  let currentKind = null;
  let i = 0;
  const flushCurrent = () => {
    if (!current) {
      return;
    }
    tokens.push(current);
    current = "";
    currentKind = null;
  };
  while (i < text.length) {
    const ansiResult = extractAnsiCode(text, i);
    if (ansiResult) {
      pendingAnsi += ansiResult.code;
      i += ansiResult.length;
      continue;
    }
    let end = i;
    while (end < text.length && !extractAnsiCode(text, end)) {
      end++;
    }
    for (const { segment } of graphemeSegmenter.segment(text.slice(i, end))) {
      const segmentIsSpace = segment === " ";
      if (!segmentIsSpace && cjkBreakRegex.test(segment)) {
        flushCurrent();
        const token = pendingAnsi + segment;
        pendingAnsi = "";
        tokens.push(token);
        continue;
      }
      const segmentKind = segmentIsSpace ? "space" : "word";
      if (current && currentKind !== segmentKind) {
        flushCurrent();
      }
      if (pendingAnsi) {
        current += pendingAnsi;
        pendingAnsi = "";
      }
      currentKind = segmentKind;
      current += segment;
    }
    i = end;
  }
  if (pendingAnsi) {
    if (current) {
      current += pendingAnsi;
    } else if (tokens.length > 0) {
      tokens[tokens.length - 1] += pendingAnsi;
    } else {
      current = pendingAnsi;
    }
  }
  if (current) {
    tokens.push(current);
  }
  return tokens;
}
function wrapTextWithAnsi(text, width) {
  if (!text) {
    return [""];
  }
  const inputLines = text.split(/\r\n|\r|\n/);
  const result = [];
  const tracker = new AnsiCodeTracker();
  for (const inputLine of inputLines) {
    const prefix = result.length > 0 ? tracker.getActiveCodes() : "";
    const wrappedLines = wrapSingleLine(prefix + inputLine, width);
    for (const wrappedLine of wrappedLines) {
      result.push(wrappedLine);
    }
    updateTrackerFromText(inputLine, tracker);
  }
  return result.length > 0 ? result : [""];
}
function wrapSingleLine(line, width) {
  if (!line) {
    return [""];
  }
  const visibleLength = visibleWidth(line);
  if (visibleLength <= width) {
    return [line];
  }
  const wrapped = [];
  const tracker = new AnsiCodeTracker();
  const tokens = splitIntoTokensWithAnsi(line);
  let currentLine = "";
  let currentVisibleLength = 0;
  for (const token of tokens) {
    const tokenVisibleLength = visibleWidth(token);
    const isWhitespace = token.trim() === "";
    if (tokenVisibleLength > width && !isWhitespace) {
      if (currentLine) {
        const lineEndReset = tracker.getLineEndReset();
        if (lineEndReset) {
          currentLine += lineEndReset;
        }
        wrapped.push(currentLine);
        currentLine = "";
        currentVisibleLength = 0;
      }
      const broken = breakLongWord(token, width, tracker);
      for (let i = 0; i < broken.length - 1; i++) {
        wrapped.push(broken[i]);
      }
      currentLine = broken[broken.length - 1];
      currentVisibleLength = visibleWidth(currentLine);
      continue;
    }
    const totalNeeded = currentVisibleLength + tokenVisibleLength;
    if (totalNeeded > width && currentVisibleLength > 0) {
      let lineToWrap = currentLine.trimEnd();
      const lineEndReset = tracker.getLineEndReset();
      if (lineEndReset) {
        lineToWrap += lineEndReset;
      }
      wrapped.push(lineToWrap);
      if (isWhitespace) {
        currentLine = tracker.getActiveCodes();
        currentVisibleLength = 0;
      } else {
        currentLine = tracker.getActiveCodes() + token;
        currentVisibleLength = tokenVisibleLength;
      }
    } else {
      currentLine += token;
      currentVisibleLength += tokenVisibleLength;
    }
    updateTrackerFromText(token, tracker);
  }
  if (currentLine) {
    wrapped.push(currentLine);
  }
  return wrapped.length > 0 ? wrapped.map((line2) => line2.trimEnd()) : [""];
}
var PUNCTUATION_REGEX = /[(){}[\]<>.,;:'"!?+\-=*/\\|&%^$#@~`]/;
function isWhitespaceChar(char) {
  return /\s/.test(char);
}
function breakLongWord(word, width, tracker) {
  const lines = [];
  let currentLine = tracker.getActiveCodes();
  let currentWidth = 0;
  let i = 0;
  const segments = [];
  while (i < word.length) {
    const ansiResult = extractAnsiCode(word, i);
    if (ansiResult) {
      segments.push({ type: "ansi", value: ansiResult.code });
      i += ansiResult.length;
    } else {
      let end = i;
      while (end < word.length) {
        const nextAnsi = extractAnsiCode(word, end);
        if (nextAnsi)
          break;
        end++;
      }
      const textPortion = word.slice(i, end);
      for (const seg of graphemeSegmenter.segment(textPortion)) {
        segments.push({ type: "grapheme", value: seg.segment });
      }
      i = end;
    }
  }
  for (const seg of segments) {
    if (seg.type === "ansi") {
      currentLine += seg.value;
      tracker.process(seg.value);
      continue;
    }
    const grapheme = seg.value;
    if (!grapheme)
      continue;
    const graphemeWidth2 = visibleWidth(grapheme);
    if (currentWidth + graphemeWidth2 > width) {
      const lineEndReset = tracker.getLineEndReset();
      if (lineEndReset) {
        currentLine += lineEndReset;
      }
      lines.push(currentLine);
      currentLine = tracker.getActiveCodes();
      currentWidth = 0;
    }
    currentLine += grapheme;
    currentWidth += graphemeWidth2;
  }
  if (currentLine) {
    lines.push(currentLine);
  }
  return lines.length > 0 ? lines : [""];
}
function applyBackgroundToLine(line, width, bgFn) {
  const visibleLen = visibleWidth(line);
  const paddingNeeded = Math.max(0, width - visibleLen);
  const padding = " ".repeat(paddingNeeded);
  const withPadding = line + padding;
  return bgFn(withPadding);
}
function truncateToWidth(text, maxWidth, ellipsis = "...", pad = false) {
  if (maxWidth <= 0) {
    return "";
  }
  if (text.length === 0) {
    return pad ? " ".repeat(maxWidth) : "";
  }
  const ellipsisWidth = visibleWidth(ellipsis);
  if (ellipsisWidth >= maxWidth) {
    const textWidth = visibleWidth(text);
    if (textWidth <= maxWidth) {
      return pad ? text + " ".repeat(maxWidth - textWidth) : text;
    }
    const clippedEllipsis = truncateFragmentToWidth(ellipsis, maxWidth);
    if (clippedEllipsis.width === 0) {
      return pad ? " ".repeat(maxWidth) : "";
    }
    return finalizeTruncatedResult("", 0, clippedEllipsis.text, clippedEllipsis.width, maxWidth, pad);
  }
  if (isPrintableAscii(text)) {
    if (text.length <= maxWidth) {
      return pad ? text + " ".repeat(maxWidth - text.length) : text;
    }
    const targetWidth2 = maxWidth - ellipsisWidth;
    return finalizeTruncatedResult(text.slice(0, targetWidth2), targetWidth2, ellipsis, ellipsisWidth, maxWidth, pad);
  }
  const targetWidth = maxWidth - ellipsisWidth;
  let result = "";
  let pendingAnsi = "";
  let visibleSoFar = 0;
  let keptWidth = 0;
  let keepContiguousPrefix = true;
  let overflowed = false;
  let exhaustedInput = false;
  const hasAnsi = text.includes("\x1B");
  const hasTabs = text.includes("	");
  if (!hasAnsi && !hasTabs) {
    for (const { segment } of graphemeSegmenter.segment(text)) {
      const width = graphemeWidth(segment);
      if (keepContiguousPrefix && keptWidth + width <= targetWidth) {
        result += segment;
        keptWidth += width;
      } else {
        keepContiguousPrefix = false;
      }
      visibleSoFar += width;
      if (visibleSoFar > maxWidth) {
        overflowed = true;
        break;
      }
    }
    exhaustedInput = !overflowed;
  } else {
    let i = 0;
    while (i < text.length) {
      const ansi = extractAnsiCode(text, i);
      if (ansi) {
        pendingAnsi += ansi.code;
        i += ansi.length;
        continue;
      }
      if (text[i] === "	") {
        if (keepContiguousPrefix && keptWidth + 3 <= targetWidth) {
          if (pendingAnsi) {
            result += pendingAnsi;
            pendingAnsi = "";
          }
          result += "	";
          keptWidth += 3;
        } else {
          keepContiguousPrefix = false;
          pendingAnsi = "";
        }
        visibleSoFar += 3;
        if (visibleSoFar > maxWidth) {
          overflowed = true;
          break;
        }
        i++;
        continue;
      }
      let end = i;
      while (end < text.length && text[end] !== "	") {
        const nextAnsi = extractAnsiCode(text, end);
        if (nextAnsi) {
          break;
        }
        end++;
      }
      for (const { segment } of graphemeSegmenter.segment(text.slice(i, end))) {
        const width = graphemeWidth(segment);
        if (keepContiguousPrefix && keptWidth + width <= targetWidth) {
          if (pendingAnsi) {
            result += pendingAnsi;
            pendingAnsi = "";
          }
          result += segment;
          keptWidth += width;
        } else {
          keepContiguousPrefix = false;
          pendingAnsi = "";
        }
        visibleSoFar += width;
        if (visibleSoFar > maxWidth) {
          overflowed = true;
          break;
        }
      }
      if (overflowed) {
        break;
      }
      i = end;
    }
    exhaustedInput = i >= text.length;
  }
  if (!overflowed && exhaustedInput) {
    return pad ? text + " ".repeat(Math.max(0, maxWidth - visibleSoFar)) : text;
  }
  return finalizeTruncatedResult(result, keptWidth, ellipsis, ellipsisWidth, maxWidth, pad);
}
function sliceByColumn(line, startCol, length, strict = false) {
  return sliceWithWidth(line, startCol, length, strict).text;
}
function sliceWithWidth(line, startCol, length, strict = false) {
  if (length <= 0)
    return { text: "", width: 0 };
  const endCol = startCol + length;
  let result = "", resultWidth = 0, currentCol = 0, i = 0, pendingAnsi = "";
  while (i < line.length) {
    const ansi = extractAnsiCode(line, i);
    if (ansi) {
      if (currentCol >= startCol && currentCol < endCol)
        result += ansi.code;
      else if (currentCol < startCol)
        pendingAnsi += ansi.code;
      i += ansi.length;
      continue;
    }
    let textEnd = i;
    while (textEnd < line.length && !extractAnsiCode(line, textEnd))
      textEnd++;
    for (const { segment } of graphemeSegmenter.segment(line.slice(i, textEnd))) {
      const w2 = graphemeWidth(segment);
      const inRange = currentCol >= startCol && currentCol < endCol;
      const fits = !strict || currentCol + w2 <= endCol;
      if (inRange && fits) {
        if (pendingAnsi) {
          result += pendingAnsi;
          pendingAnsi = "";
        }
        result += segment;
        resultWidth += w2;
      }
      currentCol += w2;
      if (currentCol >= endCol)
        break;
    }
    i = textEnd;
    if (currentCol >= endCol)
      break;
  }
  return { text: result, width: resultWidth };
}
var pooledStyleTracker = new AnsiCodeTracker();
function extractSegments(line, beforeEnd, afterStart, afterLen, strictAfter = false) {
  let before = "", beforeWidth = 0, after = "", afterWidth = 0;
  let currentCol = 0, i = 0;
  let pendingAnsiBefore = "";
  let afterStarted = false;
  const afterEnd = afterStart + afterLen;
  pooledStyleTracker.clear();
  while (i < line.length) {
    const ansi = extractAnsiCode(line, i);
    if (ansi) {
      pooledStyleTracker.process(ansi.code);
      if (currentCol < beforeEnd) {
        pendingAnsiBefore += ansi.code;
      } else if (currentCol >= afterStart && currentCol < afterEnd && afterStarted) {
        after += ansi.code;
      }
      i += ansi.length;
      continue;
    }
    let textEnd = i;
    while (textEnd < line.length && !extractAnsiCode(line, textEnd))
      textEnd++;
    for (const { segment } of graphemeSegmenter.segment(line.slice(i, textEnd))) {
      const w2 = graphemeWidth(segment);
      if (currentCol < beforeEnd && currentCol + w2 <= beforeEnd) {
        if (pendingAnsiBefore) {
          before += pendingAnsiBefore;
          pendingAnsiBefore = "";
        }
        before += segment;
        beforeWidth += w2;
      } else if (currentCol >= afterStart && currentCol < afterEnd) {
        const fits = !strictAfter || currentCol + w2 <= afterEnd;
        if (fits) {
          if (!afterStarted) {
            after += pooledStyleTracker.getActiveCodes();
            afterStarted = true;
          }
          after += segment;
          afterWidth += w2;
        }
      }
      currentCol += w2;
      if (afterLen <= 0 ? currentCol >= beforeEnd : currentCol >= afterEnd)
        break;
    }
    i = textEnd;
    if (afterLen <= 0 ? currentCol >= beforeEnd : currentCol >= afterEnd)
      break;
  }
  return { before, beforeWidth, after, afterWidth };
}

// node_modules/@earendil-works/pi-tui/dist/tui.js
function dispatchMouseEvent(component, event) {
  const result = component.handleMouse?.(event);
  if (!result)
    return void 0;
  if ("target" in result)
    return result;
  if (!result.handled && !result.capture && !result.focus)
    return void 0;
  return {
    ...result,
    handled: true,
    ...result.focus ? { focusTarget: component } : {},
    target: {
      component,
      originX: event.screenX - event.x,
      originY: event.screenY - event.y,
      width: event.width,
      height: event.height
    }
  };
}
function retargetMouseEvent(event, target) {
  return {
    ...event,
    x: event.screenX - target.originX,
    y: event.screenY - target.originY,
    width: target.width,
    height: target.height
  };
}
function isFocusable(component) {
  return component !== null && "focused" in component;
}
var CURSOR_MARKER = "\x1B_pi:c\x07";
function parseSizeValue(value, referenceSize) {
  if (value === void 0)
    return void 0;
  if (typeof value === "number")
    return value;
  const match = value.match(/^(\d+(?:\.\d+)?)%$/);
  if (match) {
    return Math.floor(referenceSize * parseFloat(match[1]) / 100);
  }
  return void 0;
}
var Container = class {
  children = [];
  mouseLayout;
  addChild(component) {
    this.children.push(component);
  }
  removeChild(component) {
    const index = this.children.indexOf(component);
    if (index !== -1) {
      this.children.splice(index, 1);
    }
  }
  clear() {
    this.children = [];
  }
  invalidate() {
    for (const child of this.children) {
      child.invalidate?.();
    }
  }
  handleMouse(event) {
    if (event.y < 0 || event.y >= event.height)
      return void 0;
    const mouseChildren = this.mouseLayout?.width === event.width ? this.mouseLayout.children : this.children.map((component) => ({ component, height: component.render(event.width).length }));
    let childY = 0;
    for (const { component: child, height: childHeight } of mouseChildren) {
      if (event.y >= childY && event.y < childY + childHeight) {
        const result = dispatchMouseEvent(child, {
          ...event,
          y: event.y - childY,
          height: childHeight
        });
        if (result?.focus && this.handleInput)
          return { ...result, focusTarget: this };
        return result;
      }
      childY += childHeight;
    }
    return void 0;
  }
  render(width) {
    const lines = [];
    const mouseChildren = [];
    for (const child of this.children) {
      const childLines = child.render(width);
      mouseChildren.push({ component: child, height: childLines.length });
      for (const line of childLines) {
        lines.push(line);
      }
    }
    this.mouseLayout = { width, children: mouseChildren };
    return lines;
  }
};
var SEGMENT_RESET = "\x1B[0m\x1B]8;;\x07";
function compositeTuiLine(baseLine, overlayLine, startCol, overlayWidth, totalWidth) {
  if (isImageLine(baseLine))
    return baseLine;
  const afterStart = startCol + overlayWidth;
  const base = extractSegments(baseLine, startCol, afterStart, totalWidth - afterStart, true);
  const overlay = sliceWithWidth(overlayLine, 0, overlayWidth, true);
  const beforePad = Math.max(0, startCol - base.beforeWidth);
  const overlayPad = Math.max(0, overlayWidth - overlay.width);
  const actualBeforeWidth = Math.max(startCol, base.beforeWidth);
  const actualOverlayWidth = Math.max(overlayWidth, overlay.width);
  const afterTarget = Math.max(0, totalWidth - actualBeforeWidth - actualOverlayWidth);
  const afterPad = Math.max(0, afterTarget - base.afterWidth);
  const result = base.before + " ".repeat(beforePad) + SEGMENT_RESET + overlay.text + " ".repeat(overlayPad) + SEGMENT_RESET + base.after + " ".repeat(afterPad);
  return visibleWidth(result) <= totalWidth ? result : sliceByColumn(result, 0, totalWidth, true);
}
var VIEWPORT_TUI = /* @__PURE__ */ Symbol.for("@earendil-works/pi-tui/viewport");
var TuiBase = class _TuiBase extends Container {
  terminal;
  focusedComponent = null;
  inputListeners = /* @__PURE__ */ new Set();
  /** Global callback for debug key (Shift+Ctrl+D). Called before input is forwarded to focused component. */
  onDebug;
  renderRequested = false;
  immediateRenderScheduled = false;
  renderTimer;
  lastRenderAt = 0;
  static MIN_RENDER_INTERVAL_MS = 16;
  showHardwareCursor = false;
  clearOnShrink = false;
  fullRedrawCount = 0;
  stopped = false;
  pendingOsc11BackgroundReplies = 0;
  pendingOsc11BackgroundQueries = [];
  terminalColorSchemeListeners = /* @__PURE__ */ new Set();
  terminalColorSchemeNotificationsEnabled = false;
  /** Directory for debug/crash logs. When undefined, debug logging is disabled and crash dumps fall back to the OS temp directory. */
  logDirectory;
  // Overlay stack for modal components rendered on top of base content
  focusOrderCounter = 0;
  overlayStack = [];
  renderedOverlayLayouts = [];
  get hasOverlayEntries() {
    return this.overlayStack.length > 0;
  }
  overlayFocusRestore = { status: "inactive" };
  constructor(terminal, showHardwareCursor, logDirectory) {
    super();
    this.terminal = terminal;
    this.logDirectory = logDirectory;
    if (showHardwareCursor !== void 0) {
      this.showHardwareCursor = showHardwareCursor;
    }
  }
  resetRenderState() {
  }
  beforeTerminalStart() {
  }
  afterTerminalStart() {
  }
  beforeTerminalStop(_options) {
  }
  afterTerminalStop(_options) {
  }
  get fullRedraws() {
    return this.fullRedrawCount;
  }
  getShowHardwareCursor() {
    return this.showHardwareCursor;
  }
  setShowHardwareCursor(enabled) {
    if (this.showHardwareCursor === enabled)
      return;
    this.showHardwareCursor = enabled;
    if (!enabled) {
      this.terminal.hideCursor();
    }
    this.requestRender();
  }
  getClearOnShrink() {
    return this.clearOnShrink;
  }
  /**
   * Set whether to trigger full re-render when content shrinks.
   * When true, empty rows are cleared when content shrinks.
   * When false (default), empty rows remain (reduces redraws on slower terminals).
   */
  setClearOnShrink(enabled) {
    this.clearOnShrink = enabled;
  }
  getFocusedComponent() {
    return this.focusedComponent;
  }
  setFocus(component) {
    this.setFocusInternal({ component, overlayFocusRestore: "clear" });
  }
  setFocusInternal({ component, overlayFocusRestore }) {
    const previousFocus = this.focusedComponent;
    let nextFocus = component;
    const previousFocusedOverlay = previousFocus ? this.overlayStack.find((entry) => entry.component === previousFocus && this.isOverlayVisible(entry)) : void 0;
    const nextFocusIsOverlay = nextFocus ? this.overlayStack.some((entry) => entry.component === nextFocus) : false;
    const restoreState = this.getVisibleOverlayFocusRestore();
    if (nextFocus && !nextFocusIsOverlay) {
      if (restoreState.status === "blocked" && restoreState.blockedBy === previousFocus) {
        if (restoreState.resume.status === "focus-target" || !this.isComponentMounted(restoreState.blockedBy)) {
          nextFocus = this.resolveBlockedOverlayFocusResume(restoreState);
        } else {
          this.overlayFocusRestore = {
            status: "blocked",
            overlay: restoreState.overlay,
            blockedBy: nextFocus,
            resume: restoreState.resume
          };
        }
      } else if (previousFocusedOverlay && restoreState.status !== "inactive" && restoreState.overlay === previousFocusedOverlay && !this.isOverlayFocusAncestor(previousFocusedOverlay, nextFocus)) {
        this.overlayFocusRestore = {
          status: "blocked",
          overlay: previousFocusedOverlay,
          blockedBy: nextFocus,
          resume: { status: "restore-overlay" }
        };
      }
    } else if (nextFocus === null) {
      if (restoreState.status === "blocked" && restoreState.blockedBy === previousFocus) {
        nextFocus = this.resolveBlockedOverlayFocusResume(restoreState);
      } else if (overlayFocusRestore === "clear") {
        this.clearOverlayFocusRestore();
      }
    }
    if (isFocusable(this.focusedComponent)) {
      this.focusedComponent.focused = false;
    }
    this.focusedComponent = nextFocus;
    if (isFocusable(nextFocus)) {
      nextFocus.focused = true;
    }
    const focusedOverlay = nextFocus ? this.overlayStack.find((entry) => entry.component === nextFocus && this.isOverlayVisible(entry)) : void 0;
    if (focusedOverlay) {
      this.overlayFocusRestore = { status: "eligible", overlay: focusedOverlay };
    }
  }
  clearOverlayFocusRestore() {
    this.overlayFocusRestore = { status: "inactive" };
  }
  clearOverlayFocusRestoreFor(overlay) {
    if (this.overlayFocusRestore.status !== "inactive" && this.overlayFocusRestore.overlay === overlay) {
      this.clearOverlayFocusRestore();
    }
  }
  resolveBlockedOverlayFocusResume(restoreState) {
    if (restoreState.resume.status === "restore-overlay")
      return restoreState.overlay.component;
    this.clearOverlayFocusRestore();
    return restoreState.resume.target;
  }
  getVisibleOverlayFocusRestore() {
    const restoreState = this.overlayFocusRestore;
    if (restoreState.status === "inactive")
      return restoreState;
    if (!this.overlayStack.includes(restoreState.overlay) || !this.isOverlayVisible(restoreState.overlay)) {
      return { status: "inactive" };
    }
    return restoreState;
  }
  isOverlayFocusAncestor(entry, component) {
    const visited = /* @__PURE__ */ new Set();
    let current = entry.preFocus;
    while (current && !visited.has(current)) {
      visited.add(current);
      if (current === component)
        return true;
      current = this.overlayStack.find((overlay) => overlay.component === current)?.preFocus ?? null;
    }
    return false;
  }
  retargetOverlayPreFocus(removed) {
    for (const overlay of this.overlayStack) {
      if (overlay !== removed && overlay.preFocus === removed.component) {
        overlay.preFocus = removed.preFocus;
      }
    }
  }
  getMountedRoots() {
    return this.children;
  }
  isComponentMounted(component) {
    return this.getMountedRoots().some((child) => this.containsComponent(child, component));
  }
  containsComponent(root, target) {
    if (root === target)
      return true;
    if (!(root instanceof Container))
      return false;
    return root.children.some((child) => this.containsComponent(child, target));
  }
  /**
   * Show an overlay component with configurable positioning and sizing.
   * Returns a handle to control the overlay's visibility.
   */
  showOverlay(component, options) {
    const entry = {
      component,
      ...options === void 0 ? {} : { options },
      preFocus: this.focusedComponent,
      hidden: false,
      focusOrder: ++this.focusOrderCounter
    };
    this.overlayStack.push(entry);
    if (!options?.nonCapturing && this.isOverlayVisible(entry)) {
      this.setFocus(component);
    }
    this.terminal.hideCursor();
    this.requestRender();
    return {
      hide: () => {
        const index = this.overlayStack.indexOf(entry);
        if (index !== -1) {
          this.clearOverlayFocusRestoreFor(entry);
          this.retargetOverlayPreFocus(entry);
          this.overlayStack.splice(index, 1);
          if (this.focusedComponent === component) {
            const topVisible = this.getTopmostVisibleOverlay();
            this.setFocus(topVisible?.component ?? entry.preFocus);
          }
          if (this.overlayStack.length === 0)
            this.terminal.hideCursor();
          this.requestRender();
        }
      },
      setHidden: (hidden) => {
        if (entry.hidden === hidden)
          return;
        entry.hidden = hidden;
        if (hidden) {
          this.clearOverlayFocusRestoreFor(entry);
          if (this.focusedComponent === component) {
            const topVisible = this.getTopmostVisibleOverlay();
            this.setFocus(topVisible?.component ?? entry.preFocus);
          }
        } else {
          if (!options?.nonCapturing && this.isOverlayVisible(entry)) {
            entry.focusOrder = ++this.focusOrderCounter;
            this.setFocus(component);
          }
        }
        this.requestRender();
      },
      isHidden: () => entry.hidden,
      focus: () => {
        if (!this.overlayStack.includes(entry) || !this.isOverlayVisible(entry))
          return;
        entry.focusOrder = ++this.focusOrderCounter;
        this.setFocus(component);
        this.requestRender();
      },
      unfocus: (unfocusOptions) => {
        const isFocused = this.focusedComponent === component;
        const restoreState = this.overlayFocusRestore;
        const hasPendingRestore = restoreState.status !== "inactive" && restoreState.overlay === entry;
        if (!isFocused && !hasPendingRestore)
          return;
        if (restoreState.status === "blocked" && restoreState.overlay === entry && this.focusedComponent === restoreState.blockedBy) {
          if (unfocusOptions) {
            this.overlayFocusRestore = {
              status: "blocked",
              overlay: entry,
              blockedBy: restoreState.blockedBy,
              resume: { status: "focus-target", target: unfocusOptions.target }
            };
          } else {
            this.clearOverlayFocusRestore();
          }
          this.requestRender();
          return;
        }
        this.clearOverlayFocusRestoreFor(entry);
        if (isFocused || unfocusOptions) {
          const topVisible = this.getTopmostVisibleOverlay();
          const fallbackTarget = topVisible && topVisible !== entry ? topVisible.component : entry.preFocus;
          this.setFocus(unfocusOptions ? unfocusOptions.target : fallbackTarget);
        }
        this.requestRender();
      },
      isFocused: () => this.focusedComponent === component,
      getBounds: () => {
        if (!this.overlayStack.includes(entry) || !this.isOverlayVisible(entry) || !entry.bounds)
          return void 0;
        return { ...entry.bounds };
      }
    };
  }
  /** Hide the topmost overlay and restore previous focus. */
  hideOverlay() {
    const overlay = this.overlayStack[this.overlayStack.length - 1];
    if (!overlay)
      return;
    this.clearOverlayFocusRestoreFor(overlay);
    this.retargetOverlayPreFocus(overlay);
    this.overlayStack.pop();
    if (this.focusedComponent === overlay.component) {
      const topVisible = this.getTopmostVisibleOverlay();
      this.setFocus(topVisible?.component ?? overlay.preFocus);
    }
    if (this.overlayStack.length === 0)
      this.terminal.hideCursor();
    this.requestRender();
  }
  /** Check if there are any visible overlays */
  hasOverlay() {
    return this.overlayStack.some((o) => this.isOverlayVisible(o));
  }
  /** Check if the focused component is a visible overlay */
  isOverlayFocused() {
    return this.overlayStack.some((entry) => entry.component === this.focusedComponent && this.isOverlayVisible(entry));
  }
  /** Keep overlay containers as keyboard focus owners when a nested control is clicked. */
  resolveMouseFocusTarget(component) {
    for (let index = this.overlayStack.length - 1; index >= 0; index--) {
      const overlay = this.overlayStack[index];
      if (this.isOverlayVisible(overlay) && this.containsComponent(overlay.component, component)) {
        return overlay.component;
      }
    }
    return component;
  }
  /** Dispatch to the visually topmost overlay under the pointer. */
  dispatchMouseToOverlay(event) {
    for (let index = this.renderedOverlayLayouts.length - 1; index >= 0; index--) {
      const layout = this.renderedOverlayLayouts[index];
      if (event.screenX < layout.col || event.screenX >= layout.col + layout.width || event.screenY < layout.row || event.screenY >= layout.row + layout.height) {
        continue;
      }
      const result = dispatchMouseEvent(layout.entry.component, {
        ...event,
        x: event.screenX - layout.col,
        y: event.screenY - layout.row,
        width: layout.width,
        height: layout.height
      });
      return result ? {
        hit: true,
        result: result.focus ? { ...result, focusTarget: layout.entry.component } : result
      } : { hit: true };
    }
    return { hit: false };
  }
  /** Check if an overlay entry is currently visible */
  isOverlayVisible(entry) {
    if (entry.hidden)
      return false;
    if (entry.options?.visible) {
      return entry.options.visible(this.terminal.columns, this.terminal.rows);
    }
    return true;
  }
  /** Find the visual-frontmost visible capturing overlay, if any */
  getTopmostVisibleOverlay() {
    let topmost;
    for (const overlay of this.overlayStack) {
      if (overlay.options?.nonCapturing || !this.isOverlayVisible(overlay))
        continue;
      if (!topmost || overlay.focusOrder > topmost.focusOrder) {
        topmost = overlay;
      }
    }
    return topmost;
  }
  invalidate() {
    for (const root of this.getMountedRoots())
      root.invalidate();
    for (const overlay of this.overlayStack)
      overlay.component.invalidate();
  }
  start() {
    this.stopped = false;
    this.beforeTerminalStart();
    this.terminal.start((data) => this.handleTerminalInput(data), () => this.requestRender());
    this.afterTerminalStart();
    this.terminal.hideCursor();
    if (this.terminalColorSchemeNotificationsEnabled) {
      this.terminal.write("\x1B[?2031h");
    }
    this.queryCellSize();
    this.requestRender();
  }
  addInputListener(listener) {
    this.inputListeners.add(listener);
    return () => {
      this.inputListeners.delete(listener);
    };
  }
  removeInputListener(listener) {
    this.inputListeners.delete(listener);
  }
  onTerminalColorSchemeChange(listener) {
    this.terminalColorSchemeListeners.add(listener);
    return () => {
      this.terminalColorSchemeListeners.delete(listener);
    };
  }
  setTerminalColorSchemeNotifications(enabled) {
    if (this.terminalColorSchemeNotificationsEnabled === enabled) {
      return;
    }
    this.terminalColorSchemeNotificationsEnabled = enabled;
    if (!this.stopped) {
      this.terminal.write(enabled ? "\x1B[?2031h" : "\x1B[?2031l");
    }
  }
  queryCellSize() {
    if (!getCapabilities().images) {
      return;
    }
    this.terminal.write("\x1B[16t");
  }
  stop(options = {}) {
    this.stopped = true;
    this.cancelRenderTimer();
    if (this.terminalColorSchemeNotificationsEnabled) {
      this.terminal.write("\x1B[?2031l");
    }
    this.beforeTerminalStop(options);
    this.terminal.showCursor();
    this.terminal.stop();
    this.afterTerminalStop(options);
  }
  renderNow(force = false) {
    if (force)
      this.resetRenderState();
    this.renderRequested = false;
    this.cancelRenderTimer();
    this.lastRenderAt = performance.now();
    this.doRender();
  }
  requestRender(force = false) {
    if (force) {
      this.resetRenderState();
      this.requestImmediateRender();
      return;
    }
    if (this.renderRequested)
      return;
    this.renderRequested = true;
    process.nextTick(() => this.scheduleRender());
  }
  requestImmediateRender() {
    this.cancelRenderTimer();
    this.renderRequested = true;
    if (this.immediateRenderScheduled)
      return;
    this.immediateRenderScheduled = true;
    process.nextTick(() => {
      this.immediateRenderScheduled = false;
      if (this.stopped || !this.renderRequested)
        return;
      this.cancelRenderTimer();
      this.renderRequested = false;
      this.lastRenderAt = performance.now();
      this.doRender();
    });
  }
  cancelRenderTimer() {
    if (!this.renderTimer)
      return;
    clearTimeout(this.renderTimer);
    this.renderTimer = void 0;
  }
  scheduleRender() {
    if (this.stopped || this.renderTimer || !this.renderRequested) {
      return;
    }
    const elapsed = performance.now() - this.lastRenderAt;
    const delay = Math.max(0, _TuiBase.MIN_RENDER_INTERVAL_MS - elapsed);
    this.renderTimer = setTimeout(() => {
      this.renderTimer = void 0;
      if (this.stopped || !this.renderRequested) {
        return;
      }
      this.renderRequested = false;
      this.lastRenderAt = performance.now();
      this.doRender();
      if (this.renderRequested) {
        this.scheduleRender();
      }
    }, delay);
  }
  handleTerminalInput(data) {
    if (this.consumeOsc11BackgroundResponse(data)) {
      return;
    }
    if (this.consumeTerminalColorSchemeReport(data)) {
      return;
    }
    if (this.inputListeners.size > 0) {
      let current = data;
      for (const listener of this.inputListeners) {
        const result = listener(current);
        if (result?.consume) {
          return;
        }
        if (result?.data !== void 0) {
          current = result.data;
        }
      }
      if (current.length === 0) {
        return;
      }
      data = current;
    }
    if (this.consumeCellSizeResponse(data)) {
      return;
    }
    if (matchesKey(data, "shift+ctrl+d") && this.onDebug) {
      this.onDebug();
      return;
    }
    const focusedOverlay = this.overlayStack.find((o) => o.component === this.focusedComponent);
    if (focusedOverlay && !this.isOverlayVisible(focusedOverlay)) {
      const topVisible = this.getTopmostVisibleOverlay();
      if (topVisible) {
        this.setFocus(topVisible.component);
      } else {
        this.setFocusInternal({ component: focusedOverlay.preFocus, overlayFocusRestore: "preserve" });
      }
    }
    const focusIsOverlay = this.overlayStack.some((o) => o.component === this.focusedComponent);
    if (!focusIsOverlay) {
      const restoreState = this.getVisibleOverlayFocusRestore();
      if (restoreState.status === "eligible") {
        this.setFocus(restoreState.overlay.component);
      } else if (restoreState.status === "blocked" && restoreState.blockedBy !== this.focusedComponent) {
        if (restoreState.resume.status === "restore-overlay") {
          this.setFocus(restoreState.overlay.component);
        } else {
          this.clearOverlayFocusRestore();
          this.setFocus(restoreState.resume.target);
        }
      }
    }
    if (this.focusedComponent?.handleInput) {
      if (isKeyRelease(data) && !this.focusedComponent.wantsKeyRelease) {
        return;
      }
      this.focusedComponent.handleInput(data);
      this.requestImmediateRender();
    }
  }
  consumeOsc11BackgroundResponse(data) {
    if (this.pendingOsc11BackgroundReplies <= 0) {
      return false;
    }
    if (!isOsc11BackgroundColorResponse(data)) {
      return false;
    }
    const rgb = parseOsc11BackgroundColor(data);
    this.pendingOsc11BackgroundReplies -= 1;
    const query = this.pendingOsc11BackgroundQueries.shift();
    if (query && !query.settled) {
      query.settled = true;
      if (query.timer) {
        clearTimeout(query.timer);
        query.timer = void 0;
      }
      query.resolve?.(rgb);
      query.resolve = void 0;
    }
    return true;
  }
  consumeTerminalColorSchemeReport(data) {
    const scheme = parseTerminalColorSchemeReport(data);
    if (!scheme) {
      return false;
    }
    for (const listener of this.terminalColorSchemeListeners) {
      listener(scheme);
    }
    return true;
  }
  consumeCellSizeResponse(data) {
    const match = data.match(/^\x1b\[6;(\d+);(\d+)t$/);
    if (!match) {
      return false;
    }
    const heightPx = parseInt(match[1], 10);
    const widthPx = parseInt(match[2], 10);
    if (heightPx <= 0 || widthPx <= 0) {
      return true;
    }
    setCellDimensions({ widthPx, heightPx });
    this.invalidate();
    this.requestRender();
    return true;
  }
  /**
   * Resolve overlay layout from options.
   * Returns { width, row, col, maxHeight } for rendering.
   */
  resolveOverlayLayout(options, overlayHeight, termWidth, termHeight) {
    const opt = options ?? {};
    const margin = typeof opt.margin === "number" ? { top: opt.margin, right: opt.margin, bottom: opt.margin, left: opt.margin } : opt.margin ?? {};
    const marginTop = Math.max(0, margin.top ?? 0);
    const marginRight = Math.max(0, margin.right ?? 0);
    const marginBottom = Math.max(0, margin.bottom ?? 0);
    const marginLeft = Math.max(0, margin.left ?? 0);
    const availWidth = Math.max(1, termWidth - marginLeft - marginRight);
    const availHeight = Math.max(1, termHeight - marginTop - marginBottom);
    let width = parseSizeValue(opt.width, termWidth) ?? Math.min(80, availWidth);
    if (opt.minWidth !== void 0) {
      width = Math.max(width, opt.minWidth);
    }
    width = Math.max(1, Math.min(width, availWidth));
    let maxHeight = parseSizeValue(opt.maxHeight, termHeight);
    if (maxHeight !== void 0) {
      maxHeight = Math.max(1, Math.min(maxHeight, availHeight));
    }
    const effectiveHeight = maxHeight !== void 0 ? Math.min(overlayHeight, maxHeight) : overlayHeight;
    let row;
    let col;
    if (opt.row !== void 0) {
      if (typeof opt.row === "string") {
        const match = opt.row.match(/^(\d+(?:\.\d+)?)%$/);
        if (match) {
          const maxRow = Math.max(0, availHeight - effectiveHeight);
          const percent = parseFloat(match[1]) / 100;
          row = marginTop + Math.floor(maxRow * percent);
        } else {
          row = this.resolveAnchorRow("center", effectiveHeight, availHeight, marginTop);
        }
      } else {
        row = opt.row;
      }
    } else {
      const anchor = opt.anchor ?? "center";
      row = this.resolveAnchorRow(anchor, effectiveHeight, availHeight, marginTop);
    }
    if (opt.col !== void 0) {
      if (typeof opt.col === "string") {
        const match = opt.col.match(/^(\d+(?:\.\d+)?)%$/);
        if (match) {
          const maxCol = Math.max(0, availWidth - width);
          const percent = parseFloat(match[1]) / 100;
          col = marginLeft + Math.floor(maxCol * percent);
        } else {
          col = this.resolveAnchorCol("center", width, availWidth, marginLeft);
        }
      } else {
        col = opt.col;
      }
    } else {
      const anchor = opt.anchor ?? "center";
      col = this.resolveAnchorCol(anchor, width, availWidth, marginLeft);
    }
    if (opt.offsetY !== void 0)
      row += opt.offsetY;
    if (opt.offsetX !== void 0)
      col += opt.offsetX;
    row = Math.max(marginTop, Math.min(row, termHeight - marginBottom - effectiveHeight));
    col = Math.max(marginLeft, Math.min(col, termWidth - marginRight - width));
    return { width, row, col, maxHeight };
  }
  resolveAnchorRow(anchor, height, availHeight, marginTop) {
    switch (anchor) {
      case "top-left":
      case "top-center":
      case "top-right":
        return marginTop;
      case "bottom-left":
      case "bottom-center":
      case "bottom-right":
        return marginTop + availHeight - height;
      case "left-center":
      case "center":
      case "right-center":
        return marginTop + Math.floor((availHeight - height) / 2);
    }
  }
  resolveAnchorCol(anchor, width, availWidth, marginLeft) {
    switch (anchor) {
      case "top-left":
      case "left-center":
      case "bottom-left":
        return marginLeft;
      case "top-right":
      case "right-center":
      case "bottom-right":
        return marginLeft + availWidth - width;
      case "top-center":
      case "center":
      case "bottom-center":
        return marginLeft + Math.floor((availWidth - width) / 2);
    }
  }
  /** Composite all overlays into content lines (sorted by focusOrder, higher = on top). */
  compositeOverlays(lines, termWidth, termHeight) {
    if (this.overlayStack.length === 0) {
      this.renderedOverlayLayouts = [];
      return lines;
    }
    const result = [...lines];
    for (const entry of this.overlayStack)
      entry.bounds = void 0;
    const rendered = [];
    let minLinesNeeded = result.length;
    const visibleEntries = this.overlayStack.filter((e) => this.isOverlayVisible(e));
    visibleEntries.sort((a, b2) => a.focusOrder - b2.focusOrder);
    for (const entry of visibleEntries) {
      const { component, options } = entry;
      const { width, maxHeight } = this.resolveOverlayLayout(options, 0, termWidth, termHeight);
      let overlayLines = component.render(width);
      if (maxHeight !== void 0 && overlayLines.length > maxHeight) {
        overlayLines = overlayLines.slice(0, maxHeight);
      }
      const { row, col } = this.resolveOverlayLayout(options, overlayLines.length, termWidth, termHeight);
      entry.bounds = { row, col, width, height: overlayLines.length };
      rendered.push({ entry, overlayLines, row, col, w: width });
      minLinesNeeded = Math.max(minLinesNeeded, row + overlayLines.length);
    }
    this.renderedOverlayLayouts = rendered.map(({ entry, row, col, w: w2, overlayLines }) => ({
      entry,
      row,
      col,
      width: w2,
      height: overlayLines.length
    }));
    const workingHeight = Math.max(result.length, termHeight, minLinesNeeded);
    while (result.length < workingHeight) {
      result.push("");
    }
    const viewportStart = Math.max(0, workingHeight - termHeight);
    for (const { overlayLines, row, col, w: w2 } of rendered) {
      for (let i = 0; i < overlayLines.length; i++) {
        const idx = viewportStart + row + i;
        if (idx >= 0 && idx < result.length) {
          const truncatedOverlayLine = visibleWidth(overlayLines[i]) > w2 ? sliceByColumn(overlayLines[i], 0, w2, true) : overlayLines[i];
          result[idx] = this.compositeLineAt(result[idx], truncatedOverlayLine, col, w2, termWidth);
        }
      }
    }
    return result;
  }
  applyLineResets(lines) {
    const reset = SEGMENT_RESET;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!isImageLine(line)) {
        lines[i] = normalizeTerminalOutput(line) + reset;
      }
    }
    return lines;
  }
  compositeLineAt(baseLine, overlayLine, startCol, overlayWidth, totalWidth) {
    return compositeTuiLine(baseLine, overlayLine, startCol, overlayWidth, totalWidth);
  }
  /**
   * Find and extract cursor position from rendered lines.
   * Searches for CURSOR_MARKER, calculates its position, and strips it from the output.
   * Only scans the bottom terminal height lines (visible viewport).
   * @param lines - Rendered lines to search
   * @param height - Terminal height (visible viewport size)
   * @returns Cursor position { row, col } or null if no marker found
   */
  extractCursorPosition(lines, height) {
    const viewportTop = Math.max(0, lines.length - height);
    for (let row = lines.length - 1; row >= viewportTop; row--) {
      const line = lines[row];
      const markerIndex = line.indexOf(CURSOR_MARKER);
      if (markerIndex !== -1) {
        const beforeMarker = line.slice(0, markerIndex);
        const col = visibleWidth(beforeMarker);
        lines[row] = line.slice(0, markerIndex) + line.slice(markerIndex + CURSOR_MARKER.length);
        return { row, col };
      }
    }
    return null;
  }
  /**
   * Query the terminal's default background color with OSC 11 (`ESC ] 11 ; ? BEL`).
   * @param timeoutMs Query timeout in milliseconds.
   * @returns Promise containing the parsed RGB color, or undefined if it times out or fails to parse.
   */
  queryTerminalBackgroundColor({ timeoutMs }) {
    return new Promise((resolve) => {
      const query = {
        settled: false,
        resolve,
        timer: void 0
      };
      query.timer = setTimeout(() => {
        if (query.settled) {
          return;
        }
        query.settled = true;
        query.timer = void 0;
        query.resolve?.(void 0);
        query.resolve = void 0;
      }, timeoutMs);
      this.pendingOsc11BackgroundQueries.push(query);
      this.pendingOsc11BackgroundReplies += 1;
      this.terminal.write("\x1B]11;?\x07");
    });
  }
  /**
   * Query the terminal's color-scheme preference with DSR (`CSI ? 996 n`).
   * Terminals that support the color palette notification protocol reply with
   * `CSI ? 997 ; 1 n` for dark or `CSI ? 997 ; 2 n` for light.
   */
  queryTerminalColorScheme({ timeoutMs }) {
    return new Promise((resolve) => {
      let settled = false;
      let timer;
      let unsubscribe = () => {
      };
      const settle = (scheme) => {
        if (settled)
          return;
        settled = true;
        if (timer) {
          clearTimeout(timer);
          timer = void 0;
        }
        unsubscribe();
        resolve(scheme);
      };
      unsubscribe = this.onTerminalColorSchemeChange(settle);
      timer = setTimeout(() => settle(void 0), timeoutMs);
      this.terminal.write("\x1B[?996n");
    });
  }
};

// node_modules/@earendil-works/pi-tui/dist/keybindings.js
var TUI_KEYBINDINGS = {
  "tui.editor.cursorUp": { defaultKeys: "up", description: "Move cursor up" },
  "tui.editor.cursorDown": { defaultKeys: "down", description: "Move cursor down" },
  "tui.editor.historyPrevious": {
    defaultKeys: [],
    description: "Select previous prompt history entry"
  },
  "tui.editor.historyNext": {
    defaultKeys: [],
    description: "Select next prompt history entry"
  },
  "tui.editor.cursorLeft": {
    defaultKeys: ["left", "ctrl+b"],
    description: "Move cursor left"
  },
  "tui.editor.cursorRight": {
    defaultKeys: ["right", "ctrl+f"],
    description: "Move cursor right"
  },
  "tui.editor.cursorWordLeft": {
    defaultKeys: ["alt+left", "ctrl+left", "alt+b"],
    description: "Move cursor word left"
  },
  "tui.editor.cursorWordRight": {
    defaultKeys: ["alt+right", "ctrl+right", "alt+f"],
    description: "Move cursor word right"
  },
  "tui.editor.cursorLineStart": {
    defaultKeys: ["home", "ctrl+home", "ctrl+a"],
    description: "Move to line start"
  },
  "tui.editor.cursorLineEnd": {
    defaultKeys: ["end", "ctrl+end", "ctrl+e"],
    description: "Move to line end"
  },
  "tui.editor.jumpForward": {
    defaultKeys: "ctrl+]",
    description: "Jump forward to character"
  },
  "tui.editor.jumpBackward": {
    defaultKeys: "ctrl+alt+]",
    description: "Jump backward to character"
  },
  "tui.editor.pageUp": { defaultKeys: ["pageUp", "ctrl+pageUp"], description: "Page up" },
  "tui.editor.pageDown": { defaultKeys: ["pageDown", "ctrl+pageDown"], description: "Page down" },
  "tui.editor.deleteCharBackward": {
    defaultKeys: "backspace",
    description: "Delete character backward"
  },
  "tui.editor.deleteCharForward": {
    defaultKeys: ["delete", "ctrl+d"],
    description: "Delete character forward"
  },
  "tui.editor.deleteWordBackward": {
    defaultKeys: ["ctrl+w", "alt+backspace"],
    description: "Delete word backward"
  },
  "tui.editor.deleteWordForward": {
    defaultKeys: ["alt+d", "alt+delete"],
    description: "Delete word forward"
  },
  "tui.editor.deleteToLineStart": {
    defaultKeys: "ctrl+u",
    description: "Delete to line start"
  },
  "tui.editor.deleteToLineEnd": {
    defaultKeys: "ctrl+k",
    description: "Delete to line end"
  },
  "tui.editor.yank": { defaultKeys: "ctrl+y", description: "Yank" },
  "tui.editor.yankPop": { defaultKeys: "alt+y", description: "Yank pop" },
  "tui.editor.undo": { defaultKeys: "ctrl+-", description: "Undo" },
  "tui.input.newLine": { defaultKeys: ["shift+enter", "ctrl+j"], description: "Insert newline" },
  "tui.input.submit": { defaultKeys: "enter", description: "Submit input" },
  "tui.input.tab": { defaultKeys: "tab", description: "Tab / autocomplete" },
  "tui.input.copy": { defaultKeys: "ctrl+c", description: "Copy selection" },
  "tui.select.up": { defaultKeys: "up", description: "Move selection up" },
  "tui.select.down": { defaultKeys: "down", description: "Move selection down" },
  "tui.select.pageUp": { defaultKeys: "pageUp", description: "Selection page up" },
  "tui.select.pageDown": {
    defaultKeys: "pageDown",
    description: "Selection page down"
  },
  "tui.select.confirm": { defaultKeys: "enter", description: "Confirm selection" },
  "tui.select.cancel": {
    defaultKeys: ["escape", "ctrl+c"],
    description: "Cancel selection"
  },
  // These intentionally shadow the unmodified editor bindings in fullscreen mode.
  "tui.altScreen.pageUp": {
    defaultKeys: "pageUp",
    description: "Scroll viewport up one page"
  },
  "tui.altScreen.pageDown": {
    defaultKeys: "pageDown",
    description: "Scroll viewport down one page"
  },
  "tui.altScreen.halfPageUp": {
    defaultKeys: [],
    description: "Scroll viewport up half a page"
  },
  "tui.altScreen.halfPageDown": {
    defaultKeys: [],
    description: "Scroll viewport down half a page"
  },
  "tui.altScreen.lineUp": {
    defaultKeys: [],
    description: "Scroll viewport up one line"
  },
  "tui.altScreen.lineDown": {
    defaultKeys: [],
    description: "Scroll viewport down one line"
  },
  "tui.altScreen.previousPrompt": {
    defaultKeys: ["ctrl+shift+up", "ctrl+up"],
    description: "Jump to previous semantic prompt"
  },
  "tui.altScreen.nextPrompt": {
    defaultKeys: ["ctrl+shift+down", "ctrl+down"],
    description: "Jump to next semantic prompt"
  },
  "tui.altScreen.search": {
    defaultKeys: "ctrl+shift+f",
    description: "Search the primary scroll view"
  },
  "tui.altScreen.searchNext": {
    defaultKeys: ["enter", "ctrl+g"],
    description: "Select the next search match"
  },
  "tui.altScreen.searchPrevious": {
    defaultKeys: ["shift+enter", "ctrl+shift+g"],
    description: "Select the previous search match"
  },
  "tui.altScreen.searchClose": {
    defaultKeys: "escape",
    description: "Close transcript search"
  },
  "tui.altScreen.top": { defaultKeys: "home", description: "Scroll viewport to top" },
  "tui.altScreen.bottom": { defaultKeys: "end", description: "Scroll viewport to bottom" }
};
function normalizeKeys(keys) {
  if (keys === void 0)
    return [];
  const keyList = Array.isArray(keys) ? keys : [keys];
  const seen = /* @__PURE__ */ new Set();
  const result = [];
  for (const key of keyList) {
    if (!seen.has(key)) {
      seen.add(key);
      result.push(key);
    }
  }
  return result;
}
var KeybindingsManager = class {
  definitions;
  userBindings;
  keysById = /* @__PURE__ */ new Map();
  conflicts = [];
  constructor(definitions, userBindings = {}) {
    this.definitions = definitions;
    this.userBindings = userBindings;
    this.rebuild();
  }
  rebuild() {
    this.keysById.clear();
    this.conflicts = [];
    const userClaims = /* @__PURE__ */ new Map();
    for (const [keybinding, keys] of Object.entries(this.userBindings)) {
      if (!(keybinding in this.definitions))
        continue;
      for (const key of normalizeKeys(keys)) {
        const claimants = userClaims.get(key) ?? /* @__PURE__ */ new Set();
        claimants.add(keybinding);
        userClaims.set(key, claimants);
      }
    }
    for (const [key, keybindings] of userClaims) {
      if (keybindings.size > 1) {
        this.conflicts.push({ key, keybindings: [...keybindings] });
      }
    }
    for (const [id, definition] of Object.entries(this.definitions)) {
      const userKeys = this.userBindings[id];
      const keys = userKeys === void 0 ? normalizeKeys(definition.defaultKeys) : normalizeKeys(userKeys);
      this.keysById.set(id, keys);
    }
  }
  matches(data, keybinding) {
    const keys = this.keysById.get(keybinding) ?? [];
    for (const key of keys) {
      if (matchesKey(data, key))
        return true;
    }
    return false;
  }
  getKeys(keybinding) {
    return [...this.keysById.get(keybinding) ?? []];
  }
  getDefinition(keybinding) {
    return this.definitions[keybinding];
  }
  getConflicts() {
    return this.conflicts.map((conflict) => ({ ...conflict, keybindings: [...conflict.keybindings] }));
  }
  setUserBindings(userBindings) {
    this.userBindings = userBindings;
    this.rebuild();
  }
  getUserBindings() {
    return { ...this.userBindings };
  }
  getResolvedBindings() {
    const resolved = {};
    for (const id of Object.keys(this.definitions)) {
      const keys = this.keysById.get(id) ?? [];
      resolved[id] = keys.length === 1 ? keys[0] : [...keys];
    }
    return resolved;
  }
};
var globalKeybindings = null;
function getKeybindings() {
  if (!globalKeybindings) {
    globalKeybindings = new KeybindingsManager(TUI_KEYBINDINGS);
  }
  return globalKeybindings;
}

// node_modules/@earendil-works/pi-tui/dist/components/text.js
var Text = class {
  text;
  paddingX;
  // Left/right padding
  paddingY;
  // Top/bottom padding
  customBgFn;
  // Cache for rendered output
  cachedText;
  cachedWidth;
  cachedLines;
  constructor(text = "", paddingX = 1, paddingY = 1, customBgFn) {
    this.text = text;
    this.paddingX = paddingX;
    this.paddingY = paddingY;
    this.customBgFn = customBgFn;
  }
  setText(text) {
    this.text = text;
    this.cachedText = void 0;
    this.cachedWidth = void 0;
    this.cachedLines = void 0;
  }
  setCustomBgFn(customBgFn) {
    this.customBgFn = customBgFn;
    this.cachedText = void 0;
    this.cachedWidth = void 0;
    this.cachedLines = void 0;
  }
  invalidate() {
    this.cachedText = void 0;
    this.cachedWidth = void 0;
    this.cachedLines = void 0;
  }
  render(width) {
    if (this.cachedLines && this.cachedText === this.text && this.cachedWidth === width) {
      return this.cachedLines;
    }
    if (!this.text || this.text.trim() === "") {
      const result2 = [];
      this.cachedText = this.text;
      this.cachedWidth = width;
      this.cachedLines = result2;
      return result2;
    }
    const normalizedText = this.text.replace(/\t/g, "   ");
    const paddingX = Math.min(this.paddingX, Math.max(0, Math.floor((width - 1) / 2)));
    const contentWidth = Math.max(1, width - paddingX * 2);
    const wrappedLines = wrapTextWithAnsi(normalizedText, contentWidth);
    const leftMargin = " ".repeat(paddingX);
    const rightMargin = " ".repeat(paddingX);
    const contentLines = [];
    for (const line of wrappedLines) {
      const lineWithMargins = leftMargin + line + rightMargin;
      if (this.customBgFn) {
        contentLines.push(applyBackgroundToLine(lineWithMargins, width, this.customBgFn));
      } else {
        const visibleLen = visibleWidth(lineWithMargins);
        const paddingNeeded = Math.max(0, width - visibleLen);
        contentLines.push(lineWithMargins + " ".repeat(paddingNeeded));
      }
    }
    const emptyLine = " ".repeat(width);
    const emptyLines = [];
    for (let i = 0; i < this.paddingY; i++) {
      const line = this.customBgFn ? applyBackgroundToLine(emptyLine, width, this.customBgFn) : emptyLine;
      emptyLines.push(line);
    }
    const result = [...emptyLines, ...contentLines, ...emptyLines];
    this.cachedText = this.text;
    this.cachedWidth = width;
    this.cachedLines = result;
    return result.length > 0 ? result : [""];
  }
};

// node_modules/@earendil-works/pi-tui/dist/kill-ring.js
var KillRing = class {
  ring = [];
  /**
   * Add text to the kill ring.
   *
   * @param text - The killed text to add
   * @param opts - Push options
   * @param opts.prepend - If accumulating, prepend (backward deletion) or append (forward deletion)
   * @param opts.accumulate - Merge with the most recent entry instead of creating a new one
   */
  push(text, opts) {
    if (!text)
      return;
    if (opts.accumulate && this.ring.length > 0) {
      const last = this.ring.pop();
      this.ring.push(opts.prepend ? text + last : last + text);
    } else {
      this.ring.push(text);
    }
  }
  /** Get most recent entry without modifying the ring. */
  peek() {
    return this.ring.length > 0 ? this.ring[this.ring.length - 1] : void 0;
  }
  /** Move last entry to front (for yank-pop cycling). */
  rotate() {
    if (this.ring.length > 1) {
      const last = this.ring.pop();
      this.ring.unshift(last);
    }
  }
  get length() {
    return this.ring.length;
  }
};

// node_modules/@earendil-works/pi-tui/dist/undo-stack.js
var UndoStack = class {
  stack = [];
  /** Push a deep clone of the given state onto the stack. */
  push(state) {
    this.stack.push(structuredClone(state));
  }
  /** Pop and return the most recent snapshot, or undefined if empty. */
  pop() {
    return this.stack.pop();
  }
  /** Remove all snapshots. */
  clear() {
    this.stack.length = 0;
  }
  get length() {
    return this.stack.length;
  }
};

// node_modules/@earendil-works/pi-tui/dist/word-navigation.js
var wordSegmenter2 = getWordSegmenter();
function findWordBackward(text, cursor, options) {
  if (cursor <= 0)
    return 0;
  const textBeforeCursor = text.slice(0, cursor);
  const segmentFn = options?.segment;
  const isAtomic = options?.isAtomicSegment;
  const segments = segmentFn ? [...segmentFn(textBeforeCursor)] : [...wordSegmenter2.segment(textBeforeCursor)];
  let newCursor = cursor;
  while (segments.length > 0 && !isAtomic?.(segments[segments.length - 1]?.segment || "") && isWhitespaceChar(segments[segments.length - 1]?.segment || "")) {
    newCursor -= segments.pop()?.segment.length || 0;
  }
  if (segments.length === 0)
    return newCursor;
  const last = segments[segments.length - 1];
  if (isAtomic?.(last.segment)) {
    newCursor -= last.segment.length;
  } else if (last.isWordLike) {
    const segment = last.segment;
    const matches = [...segment.matchAll(new RegExp(PUNCTUATION_REGEX, "g"))];
    if (matches.length <= 0) {
      newCursor -= segment.length;
    } else {
      const lastMatch = matches[matches.length - 1];
      newCursor -= segment.length - (lastMatch.index + lastMatch[0].length);
    }
  } else {
    while (segments.length > 0 && !isAtomic?.(segments[segments.length - 1]?.segment || "") && !segments[segments.length - 1]?.isWordLike && !isWhitespaceChar(segments[segments.length - 1]?.segment || "")) {
      newCursor -= segments.pop()?.segment.length || 0;
    }
  }
  return newCursor;
}
function findWordForward(text, cursor, options) {
  if (cursor >= text.length)
    return text.length;
  const textAfterCursor = text.slice(cursor);
  const segmentFn = options?.segment;
  const isAtomic = options?.isAtomicSegment;
  const segments = segmentFn ? segmentFn(textAfterCursor) : wordSegmenter2.segment(textAfterCursor);
  const iterator = segments[Symbol.iterator]();
  let next = iterator.next();
  let newCursor = cursor;
  while (!next.done && !isAtomic?.(next.value.segment) && isWhitespaceChar(next.value.segment)) {
    newCursor += next.value.segment.length;
    next = iterator.next();
  }
  if (next.done)
    return newCursor;
  if (isAtomic?.(next.value.segment)) {
    newCursor += next.value.segment.length;
  } else if (next.value.isWordLike) {
    newCursor += PUNCTUATION_REGEX.exec(next.value.segment)?.index ?? next.value.segment.length;
  } else {
    while (!next.done && !isAtomic?.(next.value.segment) && !next.value.isWordLike && !isWhitespaceChar(next.value.segment)) {
      newCursor += next.value.segment.length;
      next = iterator.next();
    }
  }
  return newCursor;
}

// node_modules/@earendil-works/pi-tui/dist/components/select-list.js
var DEFAULT_PRIMARY_COLUMN_WIDTH = 32;
var PRIMARY_COLUMN_GAP = 2;
var MIN_DESCRIPTION_WIDTH = 10;
var normalizeToSingleLine = (text) => text.replace(/[\r\n]+/g, " ").trim();
var clamp = (value, min, max) => Math.max(min, Math.min(value, max));
var SelectList = class {
  items = [];
  filteredItems = [];
  selectedIndex = 0;
  mousePressedIndex;
  maxVisible = 5;
  theme;
  layout;
  onSelect;
  onCancel;
  onSelectionChange;
  constructor(items, maxVisible, theme, layout = {}) {
    this.items = items;
    this.filteredItems = items;
    this.maxVisible = maxVisible;
    this.theme = theme;
    this.layout = layout;
  }
  setFilter(filter) {
    this.filteredItems = this.items.filter((item) => item.value.toLowerCase().startsWith(filter.toLowerCase()));
    this.selectedIndex = 0;
  }
  setSelectedIndex(index) {
    this.selectedIndex = Math.max(0, Math.min(index, this.filteredItems.length - 1));
  }
  invalidate() {
  }
  render(width) {
    const lines = [];
    if (this.filteredItems.length === 0) {
      lines.push(this.theme.noMatch("  No matching commands"));
      return lines;
    }
    const primaryColumnWidth = this.getPrimaryColumnWidth();
    const { startIndex, endIndex } = this.getVisibleRange();
    for (let i = startIndex; i < endIndex; i++) {
      const item = this.filteredItems[i];
      if (!item)
        continue;
      const isSelected = i === this.selectedIndex;
      const descriptionSingleLine = item.description ? normalizeToSingleLine(item.description) : void 0;
      lines.push(this.renderItem(item, isSelected, width, descriptionSingleLine, primaryColumnWidth));
    }
    if (startIndex > 0 || endIndex < this.filteredItems.length) {
      const scrollText = `  (${this.selectedIndex + 1}/${this.filteredItems.length})`;
      lines.push(this.theme.scrollInfo(truncateToWidth(scrollText, width - 2, "")));
    }
    return lines;
  }
  handleMouse(event) {
    if (this.filteredItems.length === 0)
      return void 0;
    if (event.type === "wheel" && event.wheelDelta) {
      const delta = event.wheelDelta < 0 ? -1 : 1;
      const previousIndex = this.selectedIndex;
      this.selectedIndex = Math.max(0, Math.min(this.filteredItems.length - 1, this.selectedIndex + delta));
      if (this.selectedIndex !== previousIndex)
        this.notifySelectionChange();
      return { handled: true, render: this.selectedIndex !== previousIndex };
    }
    if (event.type !== "move" && event.button !== "left")
      return void 0;
    const { startIndex, endIndex } = this.getVisibleRange();
    const itemIndex = startIndex + event.y;
    if (itemIndex < startIndex || itemIndex >= endIndex)
      return void 0;
    if (event.type === "move" || event.type === "press") {
      if (event.type === "press")
        this.mousePressedIndex = itemIndex;
      const changed = this.selectedIndex !== itemIndex;
      if (changed) {
        this.selectedIndex = itemIndex;
        this.notifySelectionChange();
      }
      return {
        handled: true,
        focus: event.type === "press",
        ...event.type === "move" ? { render: changed } : {}
      };
    }
    if (event.type === "click") {
      const clickedIndex = this.mousePressedIndex ?? itemIndex;
      this.mousePressedIndex = void 0;
      const changed = this.selectedIndex !== clickedIndex;
      this.selectedIndex = clickedIndex;
      if (changed)
        this.notifySelectionChange();
      const selectedItem = this.filteredItems[this.selectedIndex];
      if (selectedItem)
        this.onSelect?.(selectedItem);
      return { handled: true };
    }
    return void 0;
  }
  handleInput(keyData) {
    const kb = getKeybindings();
    if (kb.matches(keyData, "tui.select.up")) {
      this.selectedIndex = this.selectedIndex === 0 ? this.filteredItems.length - 1 : this.selectedIndex - 1;
      this.notifySelectionChange();
    } else if (kb.matches(keyData, "tui.select.down")) {
      this.selectedIndex = this.selectedIndex === this.filteredItems.length - 1 ? 0 : this.selectedIndex + 1;
      this.notifySelectionChange();
    } else if (kb.matches(keyData, "tui.select.confirm")) {
      const selectedItem = this.filteredItems[this.selectedIndex];
      if (selectedItem && this.onSelect) {
        this.onSelect(selectedItem);
      }
    } else if (kb.matches(keyData, "tui.select.cancel")) {
      if (this.onCancel) {
        this.onCancel();
      }
    }
  }
  getVisibleRange() {
    const startIndex = Math.max(0, Math.min(this.selectedIndex - Math.floor(this.maxVisible / 2), this.filteredItems.length - this.maxVisible));
    return {
      startIndex,
      endIndex: Math.min(startIndex + this.maxVisible, this.filteredItems.length)
    };
  }
  renderItem(item, isSelected, width, descriptionSingleLine, primaryColumnWidth) {
    const prefix = isSelected ? "\u2192 " : "  ";
    const prefixWidth = visibleWidth(prefix);
    if (descriptionSingleLine && width > 40) {
      const effectivePrimaryColumnWidth = Math.max(1, Math.min(primaryColumnWidth, width - prefixWidth - 4));
      const maxPrimaryWidth = Math.max(1, effectivePrimaryColumnWidth - PRIMARY_COLUMN_GAP);
      const truncatedValue2 = this.truncatePrimary(item, isSelected, maxPrimaryWidth, effectivePrimaryColumnWidth);
      const truncatedValueWidth = visibleWidth(truncatedValue2);
      const spacing = " ".repeat(Math.max(1, effectivePrimaryColumnWidth - truncatedValueWidth));
      const descriptionStart = prefixWidth + truncatedValueWidth + spacing.length;
      const remainingWidth = width - descriptionStart - 2;
      if (remainingWidth > MIN_DESCRIPTION_WIDTH) {
        const truncatedDesc = truncateToWidth(descriptionSingleLine, remainingWidth, "");
        if (isSelected) {
          return this.theme.selectedText(`${prefix}${truncatedValue2}${spacing}${truncatedDesc}`);
        }
        const descText = this.theme.description(spacing + truncatedDesc);
        return prefix + truncatedValue2 + descText;
      }
    }
    const maxWidth = width - prefixWidth - 2;
    const truncatedValue = this.truncatePrimary(item, isSelected, maxWidth, maxWidth);
    if (isSelected) {
      return this.theme.selectedText(`${prefix}${truncatedValue}`);
    }
    return prefix + truncatedValue;
  }
  getPrimaryColumnWidth() {
    const { min, max } = this.getPrimaryColumnBounds();
    const widestPrimary = this.filteredItems.reduce((widest, item) => {
      return Math.max(widest, visibleWidth(this.getDisplayValue(item)) + PRIMARY_COLUMN_GAP);
    }, 0);
    return clamp(widestPrimary, min, max);
  }
  getPrimaryColumnBounds() {
    const rawMin = this.layout.minPrimaryColumnWidth ?? this.layout.maxPrimaryColumnWidth ?? DEFAULT_PRIMARY_COLUMN_WIDTH;
    const rawMax = this.layout.maxPrimaryColumnWidth ?? this.layout.minPrimaryColumnWidth ?? DEFAULT_PRIMARY_COLUMN_WIDTH;
    return {
      min: Math.max(1, Math.min(rawMin, rawMax)),
      max: Math.max(1, Math.max(rawMin, rawMax))
    };
  }
  truncatePrimary(item, isSelected, maxWidth, columnWidth) {
    const displayValue = this.getDisplayValue(item);
    const truncatedValue = this.layout.truncatePrimary ? this.layout.truncatePrimary({
      text: displayValue,
      maxWidth,
      columnWidth,
      item,
      isSelected
    }) : truncateToWidth(displayValue, maxWidth, "");
    return truncateToWidth(truncatedValue, maxWidth, "");
  }
  getDisplayValue(item) {
    return item.label || item.value;
  }
  notifySelectionChange() {
    const selectedItem = this.filteredItems[this.selectedIndex];
    if (selectedItem && this.onSelectionChange) {
      this.onSelectionChange(selectedItem);
    }
  }
  getSelectedItem() {
    const item = this.filteredItems[this.selectedIndex];
    return item || null;
  }
};

// node_modules/@earendil-works/pi-tui/dist/components/editor.js
var graphemeSegmenter2 = getGraphemeSegmenter();
var wordSegmenter3 = getWordSegmenter();
var PASTE_MARKER_REGEX = /\[paste #(\d+)( (\+\d+ lines|\d+ chars))?\]/g;
var PASTE_MARKER_SINGLE = /^\[paste #(\d+)( (\+\d+ lines|\d+ chars))?\]$/;
function isPasteMarker(segment) {
  return segment.length >= 10 && PASTE_MARKER_SINGLE.test(segment);
}
function segmentWithMarkers(text, baseSegmenter, validIds) {
  if (validIds.size === 0 || !text.includes("[paste #")) {
    return baseSegmenter.segment(text);
  }
  const markers = [];
  for (const m2 of text.matchAll(PASTE_MARKER_REGEX)) {
    const id = Number.parseInt(m2[1], 10);
    if (!validIds.has(id))
      continue;
    markers.push({ start: m2.index, end: m2.index + m2[0].length });
  }
  if (markers.length === 0) {
    return baseSegmenter.segment(text);
  }
  const baseSegments = baseSegmenter.segment(text);
  const result = [];
  let markerIdx = 0;
  for (const seg of baseSegments) {
    while (markerIdx < markers.length && markers[markerIdx].end <= seg.index) {
      markerIdx++;
    }
    const marker = markerIdx < markers.length ? markers[markerIdx] : null;
    if (marker && seg.index >= marker.start && seg.index < marker.end) {
      if (seg.index === marker.start) {
        const markerText = text.slice(marker.start, marker.end);
        result.push({
          segment: markerText,
          index: marker.start,
          input: text
        });
      }
    } else {
      result.push(seg);
    }
  }
  return result;
}
function wordWrapLine(line, maxWidth, preSegmented) {
  if (!line || maxWidth <= 0) {
    return [{ text: "", startIndex: 0, endIndex: 0 }];
  }
  const lineWidth = visibleWidth(line);
  if (lineWidth <= maxWidth) {
    return [{ text: line, startIndex: 0, endIndex: line.length }];
  }
  const chunks = [];
  const segments = preSegmented ?? [...graphemeSegmenter2.segment(line)];
  let currentWidth = 0;
  let chunkStart = 0;
  let wrapOppIndex = -1;
  let wrapOppWidth = 0;
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const grapheme = seg.segment;
    const gWidth = visibleWidth(grapheme);
    const charIndex = seg.index;
    const isWs = !isPasteMarker(grapheme) && isWhitespaceChar(grapheme);
    if (currentWidth + gWidth > maxWidth) {
      if (wrapOppIndex >= 0 && currentWidth - wrapOppWidth + gWidth <= maxWidth) {
        chunks.push({ text: line.slice(chunkStart, wrapOppIndex), startIndex: chunkStart, endIndex: wrapOppIndex });
        chunkStart = wrapOppIndex;
        currentWidth -= wrapOppWidth;
      } else if (chunkStart < charIndex) {
        chunks.push({ text: line.slice(chunkStart, charIndex), startIndex: chunkStart, endIndex: charIndex });
        chunkStart = charIndex;
        currentWidth = 0;
      }
      wrapOppIndex = -1;
    }
    if (gWidth > maxWidth) {
      const subChunks = wordWrapLine(grapheme, maxWidth);
      for (let j2 = 0; j2 < subChunks.length - 1; j2++) {
        const sc = subChunks[j2];
        chunks.push({ text: sc.text, startIndex: charIndex + sc.startIndex, endIndex: charIndex + sc.endIndex });
      }
      const last = subChunks[subChunks.length - 1];
      chunkStart = charIndex + last.startIndex;
      currentWidth = visibleWidth(last.text);
      wrapOppIndex = -1;
      continue;
    }
    currentWidth += gWidth;
    const next = segments[i + 1];
    if (isWs && next && (isPasteMarker(next.segment) || !isWhitespaceChar(next.segment))) {
      wrapOppIndex = next.index;
      wrapOppWidth = currentWidth;
    } else if (!isWs && next && !isWhitespaceChar(next.segment)) {
      const isCjk = !isPasteMarker(grapheme) && cjkBreakRegex.test(grapheme);
      const nextIsCjk = !isPasteMarker(next.segment) && cjkBreakRegex.test(next.segment);
      if (isCjk || nextIsCjk) {
        wrapOppIndex = next.index;
        wrapOppWidth = currentWidth;
      }
    }
  }
  chunks.push({ text: line.slice(chunkStart), startIndex: chunkStart, endIndex: line.length });
  return chunks;
}
var SLASH_COMMAND_SELECT_LIST_LAYOUT = {
  minPrimaryColumnWidth: 12,
  maxPrimaryColumnWidth: 32
};
var ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS = 20;
var DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS = ["@", "#"];
function escapeCharacterClass(value) {
  return value.replace(/[\\^$.*+?()[\]{}|-]/g, "\\$&");
}
function buildTriggerPattern(triggerCharacters) {
  return new RegExp(`(?:^|[\\s])[${triggerCharacters.map(escapeCharacterClass).join("")}][^\\s]*$`);
}
function buildDebouncePattern(triggerCharacters) {
  const escapedWithoutAt = triggerCharacters.filter((character) => character !== "@").map(escapeCharacterClass);
  return new RegExp(`(?:^|[ \\t])(?:@(?:"[^"]*|[^\\s]*)|[${escapedWithoutAt.join("")}][^\\s]*)$`);
}
function createScrollBorder(direction, hiddenLineCount, width) {
  const availableWidth = Math.max(0, width);
  const label = ` ${direction} ${hiddenLineCount} more `;
  const labelWidth = visibleWidth(label);
  if (labelWidth + 2 <= availableWidth) {
    const leftWidth = Math.floor((availableWidth - labelWidth) / 2);
    return "\u2500".repeat(leftWidth) + label + "\u2500".repeat(availableWidth - leftWidth - labelWidth);
  }
  const indicator = `\u2500\u2500\u2500 ${direction} ${hiddenLineCount} more `;
  const remaining = availableWidth - visibleWidth(indicator);
  if (remaining >= 0)
    return indicator + "\u2500".repeat(remaining);
  const ellipsis = "...".slice(0, availableWidth);
  const indicatorWidth = availableWidth - visibleWidth(ellipsis);
  return sliceByColumn(indicator, 0, indicatorWidth, true) + ellipsis;
}
var Editor = class {
  state = {
    lines: [""],
    cursorLine: 0,
    cursorCol: 0
  };
  /** Focusable interface - set by TUI when focus changes */
  focused = false;
  tui;
  theme;
  paddingX = 0;
  // Store last render geometry for cursor navigation and mouse hit-testing.
  lastWidth = 80;
  renderedVisibleLineCount = 1;
  renderedAutocompleteHeight = 0;
  // Vertical scrolling support
  scrollOffset = 0;
  // Border color (can be changed dynamically)
  borderColor;
  // Autocomplete support
  autocompleteProvider;
  autocompleteTriggerCharacters = [...DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS];
  autocompleteTriggerPattern = buildTriggerPattern(this.autocompleteTriggerCharacters);
  autocompleteDebouncePattern = buildDebouncePattern(this.autocompleteTriggerCharacters);
  autocompleteList;
  autocompleteState = null;
  autocompletePrefix = "";
  autocompleteMaxVisible = 5;
  autocompleteAbort;
  autocompleteDebounceTimer;
  autocompleteRequestTask = Promise.resolve();
  autocompleteStartToken = 0;
  autocompleteRequestId = 0;
  // Paste tracking for large pastes
  pastes = /* @__PURE__ */ new Map();
  pasteCounter = 0;
  // Bracketed paste mode buffering
  pasteBuffer = "";
  isInPaste = false;
  // Prompt history for up/down navigation
  history = [];
  historyIndex = -1;
  // -1 = not browsing, 0 = most recent, 1 = older, etc.
  historyDraft = null;
  // Kill ring for Emacs-style kill/yank operations
  killRing = new KillRing();
  lastAction = null;
  // Character jump mode
  jumpMode = null;
  // Preferred visual column for vertical cursor movement (sticky column)
  preferredVisualCol = null;
  // When the cursor is snapped to the start of an atomic segment, e.g. a
  // paste marker, cursorCol no longer reflects where the cursor would have
  // landed. This field stores the pre-snap cursorCol so that the next
  // vertical move can resolve it to a visual column on whatever VL it belongs
  // to.
  snappedFromCursorCol = null;
  // Undo support
  undoStack = new UndoStack();
  onSubmit;
  onChange;
  disableSubmit = false;
  constructor(tui, theme, options = {}) {
    this.tui = tui;
    this.theme = theme;
    this.borderColor = theme.borderColor;
    const paddingX = options.paddingX ?? 0;
    this.paddingX = Number.isFinite(paddingX) ? Math.max(0, Math.floor(paddingX)) : 0;
    const maxVisible = options.autocompleteMaxVisible ?? 5;
    this.autocompleteMaxVisible = Number.isFinite(maxVisible) ? Math.max(3, Math.min(20, Math.floor(maxVisible))) : 5;
  }
  /** Set of currently valid paste IDs, for marker-aware segmentation. */
  validPasteIds() {
    return new Set(this.pastes.keys());
  }
  /** Segment text with paste-marker awareness, only merging markers with valid IDs. */
  segment(text, mode) {
    return segmentWithMarkers(text, mode === "word" ? wordSegmenter3 : graphemeSegmenter2, this.validPasteIds());
  }
  getPaddingX() {
    return this.paddingX;
  }
  setPaddingX(padding) {
    const newPadding = Number.isFinite(padding) ? Math.max(0, Math.floor(padding)) : 0;
    if (this.paddingX !== newPadding) {
      this.paddingX = newPadding;
      this.tui.requestRender();
    }
  }
  getAutocompleteMaxVisible() {
    return this.autocompleteMaxVisible;
  }
  setAutocompleteMaxVisible(maxVisible) {
    const newMaxVisible = Number.isFinite(maxVisible) ? Math.max(3, Math.min(20, Math.floor(maxVisible))) : 5;
    if (this.autocompleteMaxVisible !== newMaxVisible) {
      this.autocompleteMaxVisible = newMaxVisible;
      this.tui.requestRender();
    }
  }
  setAutocompleteProvider(provider) {
    this.cancelAutocomplete();
    this.autocompleteProvider = provider;
    this.setAutocompleteTriggerCharacters(provider.triggerCharacters ?? []);
  }
  /**
   * Add a prompt to history for up/down arrow navigation.
   * Called after successful submission.
   */
  addToHistory(text) {
    const trimmed = text.trim();
    if (!trimmed)
      return;
    if (this.history.length > 0 && this.history[0] === trimmed)
      return;
    this.history.unshift(trimmed);
    if (this.history.length > 100) {
      this.history.pop();
    }
  }
  isEditorEmpty() {
    return this.state.lines.length === 1 && this.state.lines[0] === "";
  }
  isOnFirstVisualLine() {
    const visualLines = this.buildVisualLineMap(this.lastWidth);
    const currentVisualLine = this.findCurrentVisualLine(visualLines);
    return currentVisualLine === 0;
  }
  isOnLastVisualLine() {
    const visualLines = this.buildVisualLineMap(this.lastWidth);
    const currentVisualLine = this.findCurrentVisualLine(visualLines);
    return currentVisualLine === visualLines.length - 1;
  }
  navigateHistory(direction) {
    this.lastAction = null;
    if (this.history.length === 0)
      return;
    const newIndex = this.historyIndex - direction;
    if (newIndex < -1 || newIndex >= this.history.length)
      return;
    if (this.historyIndex === -1 && newIndex >= 0) {
      this.pushUndoSnapshot();
      this.historyDraft = structuredClone(this.state);
    }
    this.historyIndex = newIndex;
    if (this.historyIndex === -1) {
      const draft = this.historyDraft;
      this.historyDraft = null;
      if (draft) {
        this.state = draft;
        this.preferredVisualCol = null;
        this.snappedFromCursorCol = null;
        this.scrollOffset = 0;
        if (this.onChange)
          this.onChange(this.getText());
      } else {
        this.setTextInternal("");
      }
    } else {
      this.setTextInternal(this.history[this.historyIndex] || "", direction === -1 ? "start" : "end");
    }
  }
  exitHistoryBrowsing() {
    this.historyIndex = -1;
    this.historyDraft = null;
  }
  /** Internal setText that doesn't reset history state - used by navigateHistory */
  setTextInternal(text, cursorPlacement = "end") {
    const lines = text.split("\n");
    this.state.lines = lines.length === 0 ? [""] : lines;
    this.state.cursorLine = cursorPlacement === "start" ? 0 : this.state.lines.length - 1;
    this.setCursorCol(cursorPlacement === "start" ? 0 : this.state.lines[this.state.cursorLine]?.length || 0);
    this.scrollOffset = 0;
    if (this.onChange) {
      this.onChange(this.getText());
    }
  }
  invalidate() {
  }
  renderTopBorder(width, hiddenLineCount) {
    const border = hiddenLineCount > 0 ? createScrollBorder("\u2191", hiddenLineCount, width) : "\u2500".repeat(width);
    return this.borderColor(border);
  }
  renderBottomBorder(width, hiddenLineCount) {
    const border = hiddenLineCount > 0 ? createScrollBorder("\u2193", hiddenLineCount, width) : "\u2500".repeat(width);
    return this.borderColor(border);
  }
  render(width) {
    const maxPadding = Math.max(0, Math.floor((width - 1) / 2));
    const paddingX = Math.min(this.paddingX, maxPadding);
    const contentWidth = Math.max(1, width - paddingX * 2);
    const layoutWidth = Math.max(1, contentWidth - (paddingX ? 0 : 1));
    this.lastWidth = layoutWidth;
    const layoutLines = this.layoutText(layoutWidth);
    const terminalRows = this.tui.terminal.rows;
    const maxVisibleLines = Math.max(5, Math.floor(terminalRows * 0.3));
    let cursorLineIndex = layoutLines.findIndex((line) => line.hasCursor);
    if (cursorLineIndex === -1)
      cursorLineIndex = 0;
    if (cursorLineIndex < this.scrollOffset) {
      this.scrollOffset = cursorLineIndex;
    } else if (cursorLineIndex >= this.scrollOffset + maxVisibleLines) {
      this.scrollOffset = cursorLineIndex - maxVisibleLines + 1;
    }
    const maxScrollOffset = Math.max(0, layoutLines.length - maxVisibleLines);
    this.scrollOffset = Math.max(0, Math.min(this.scrollOffset, maxScrollOffset));
    const visibleLines = layoutLines.slice(this.scrollOffset, this.scrollOffset + maxVisibleLines);
    this.renderedVisibleLineCount = visibleLines.length;
    const result = [];
    const leftPadding = " ".repeat(paddingX);
    const rightPadding = leftPadding;
    result.push(this.renderTopBorder(width, this.scrollOffset));
    const emitCursorMarker = this.focused;
    for (const layoutLine of visibleLines) {
      let displayText = layoutLine.text;
      let lineVisibleWidth = visibleWidth(layoutLine.text);
      let cursorInPadding = false;
      if (layoutLine.hasCursor && layoutLine.cursorPos !== void 0) {
        const before = displayText.slice(0, layoutLine.cursorPos);
        const after = displayText.slice(layoutLine.cursorPos);
        const marker = emitCursorMarker ? CURSOR_MARKER : "";
        if (after.length > 0) {
          const afterGraphemes = [...this.segment(after, "grapheme")];
          const firstGrapheme = afterGraphemes[0]?.segment || "";
          const restAfter = after.slice(firstGrapheme.length);
          const cursor = `\x1B[7m${firstGrapheme}\x1B[0m`;
          displayText = before + marker + cursor + restAfter;
        } else {
          const cursor = "\x1B[7m \x1B[0m";
          displayText = before + marker + cursor;
          lineVisibleWidth = lineVisibleWidth + 1;
          if (lineVisibleWidth > contentWidth && paddingX > 0) {
            cursorInPadding = true;
          }
        }
      }
      const padding = " ".repeat(Math.max(0, contentWidth - lineVisibleWidth));
      const lineRightPadding = cursorInPadding ? rightPadding.slice(1) : rightPadding;
      result.push(`${leftPadding}${displayText}${padding}${lineRightPadding}`);
    }
    const linesBelow = layoutLines.length - (this.scrollOffset + visibleLines.length);
    result.push(this.renderBottomBorder(width, linesBelow));
    this.renderedAutocompleteHeight = 0;
    if (this.autocompleteState && this.autocompleteList) {
      const autocompleteResult = this.autocompleteList.render(contentWidth);
      this.renderedAutocompleteHeight = autocompleteResult.length;
      for (const line of autocompleteResult) {
        const lineWidth = visibleWidth(line);
        const linePadding = " ".repeat(Math.max(0, contentWidth - lineWidth));
        result.push(`${leftPadding}${line}${linePadding}${rightPadding}`);
      }
    }
    return result;
  }
  handleMouse(event) {
    const autocompleteStartRow = this.renderedVisibleLineCount + 2;
    if (this.autocompleteState && this.autocompleteList && event.y >= autocompleteStartRow && event.y < autocompleteStartRow + this.renderedAutocompleteHeight) {
      const maxPadding2 = Math.max(0, Math.floor((event.width - 1) / 2));
      const paddingX2 = Math.min(this.paddingX, maxPadding2);
      const contentWidth = Math.max(1, event.width - paddingX2 * 2);
      const result = this.autocompleteList.handleMouse?.({
        ...event,
        x: event.x - paddingX2,
        y: event.y - autocompleteStartRow,
        width: contentWidth,
        height: this.renderedAutocompleteHeight
      });
      return result ? { ...result, focus: true } : void 0;
    }
    if (event.type !== "click" || event.button !== "left")
      return void 0;
    if (event.y <= 0 || event.y > this.renderedVisibleLineCount)
      return { handled: true, focus: true };
    const visualLines = this.buildVisualLineMap(this.lastWidth);
    const visualLineIndex = this.scrollOffset + event.y - 1;
    const visualLine = visualLines[visualLineIndex];
    if (!visualLine)
      return { handled: true, focus: true };
    const logicalLine = this.state.lines[visualLine.logicalLine] ?? "";
    const chunkEnd = visualLine.startCol + visualLine.length;
    const chunk = logicalLine.slice(visualLine.startCol, chunkEnd);
    const maxPadding = Math.max(0, Math.floor((event.width - 1) / 2));
    const paddingX = Math.min(this.paddingX, maxPadding);
    const targetColumn = Math.max(0, event.x - paddingX);
    let visibleColumn = 0;
    let targetIndex = chunk.length;
    let lastGraphemeIndex = 0;
    for (const grapheme of this.segment(chunk, "grapheme")) {
      const nextColumn = visibleColumn + visibleWidth(grapheme.segment);
      lastGraphemeIndex = grapheme.index;
      if (targetColumn < nextColumn) {
        targetIndex = grapheme.index;
        break;
      }
      visibleColumn = nextColumn;
    }
    const isLastSegment = visualLineIndex === visualLines.length - 1 || visualLines[visualLineIndex + 1]?.logicalLine !== visualLine.logicalLine;
    if (!isLastSegment && targetIndex === chunk.length && chunk.length > 0)
      targetIndex = lastGraphemeIndex;
    this.state.cursorLine = visualLine.logicalLine;
    this.setCursorCol(visualLine.startCol + targetIndex);
    this.lastAction = null;
    this.exitHistoryBrowsing();
    if (this.autocompleteState)
      this.updateAutocomplete();
    return { handled: true, focus: true };
  }
  handleInput(data) {
    const kb = getKeybindings();
    if (this.jumpMode !== null) {
      if (kb.matches(data, "tui.editor.jumpForward") || kb.matches(data, "tui.editor.jumpBackward")) {
        this.jumpMode = null;
        return;
      }
      const printable2 = decodePrintableKey(data) ?? (data.charCodeAt(0) >= 32 ? data : void 0);
      if (printable2 !== void 0) {
        const direction = this.jumpMode;
        this.jumpMode = null;
        this.jumpToChar(printable2, direction);
        return;
      }
      this.jumpMode = null;
    }
    if (data.includes("\x1B[200~")) {
      this.isInPaste = true;
      this.pasteBuffer = "";
      data = data.replace("\x1B[200~", "");
    }
    if (this.isInPaste) {
      this.pasteBuffer += data;
      const endIndex = this.pasteBuffer.indexOf("\x1B[201~");
      if (endIndex !== -1) {
        const pasteContent = this.pasteBuffer.substring(0, endIndex);
        if (pasteContent.length > 0) {
          this.handlePaste(pasteContent);
        }
        this.isInPaste = false;
        const remaining = this.pasteBuffer.substring(endIndex + 6);
        this.pasteBuffer = "";
        if (remaining.length > 0) {
          this.handleInput(remaining);
        }
        return;
      }
      return;
    }
    if (kb.matches(data, "tui.input.copy")) {
      return;
    }
    if (kb.matches(data, "tui.editor.undo")) {
      this.undo();
      return;
    }
    if (this.autocompleteState && this.autocompleteList) {
      if (kb.matches(data, "tui.select.cancel")) {
        this.cancelAutocomplete();
        return;
      }
      if (kb.matches(data, "tui.select.up") || kb.matches(data, "tui.select.down")) {
        this.autocompleteList.handleInput(data);
        return;
      }
      if (kb.matches(data, "tui.input.tab")) {
        const selected = this.autocompleteList.getSelectedItem();
        if (selected && this.autocompleteProvider) {
          this.pushUndoSnapshot();
          this.lastAction = null;
          const result = this.autocompleteProvider.applyCompletion(this.state.lines, this.state.cursorLine, this.state.cursorCol, selected, this.autocompletePrefix);
          this.state.lines = result.lines;
          this.state.cursorLine = result.cursorLine;
          this.setCursorCol(result.cursorCol);
          this.cancelAutocomplete();
          if (this.onChange)
            this.onChange(this.getText());
        }
        return;
      }
      if (kb.matches(data, "tui.select.confirm")) {
        const selected = this.autocompleteList.getSelectedItem();
        if (selected && this.autocompleteProvider) {
          this.pushUndoSnapshot();
          this.lastAction = null;
          const result = this.autocompleteProvider.applyCompletion(this.state.lines, this.state.cursorLine, this.state.cursorCol, selected, this.autocompletePrefix);
          this.state.lines = result.lines;
          this.state.cursorLine = result.cursorLine;
          this.setCursorCol(result.cursorCol);
          if (this.autocompletePrefix.startsWith("/")) {
            this.cancelAutocomplete();
          } else {
            this.cancelAutocomplete();
            if (this.onChange)
              this.onChange(this.getText());
            return;
          }
        }
      }
    }
    if (kb.matches(data, "tui.input.tab") && !this.autocompleteState) {
      this.handleTabCompletion();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteToLineEnd")) {
      this.deleteToEndOfLine();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteToLineStart")) {
      this.deleteToStartOfLine();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteWordBackward")) {
      this.deleteWordBackwards();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteWordForward")) {
      this.deleteWordForward();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteCharBackward") || matchesKey(data, "shift+backspace")) {
      this.handleBackspace();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteCharForward") || matchesKey(data, "shift+delete")) {
      this.handleForwardDelete();
      return;
    }
    if (kb.matches(data, "tui.editor.yank")) {
      this.yank();
      return;
    }
    if (kb.matches(data, "tui.editor.yankPop")) {
      this.yankPop();
      return;
    }
    if (kb.matches(data, "tui.editor.historyPrevious")) {
      this.cancelAutocomplete();
      this.navigateHistory(-1);
      return;
    }
    if (kb.matches(data, "tui.editor.historyNext")) {
      this.cancelAutocomplete();
      this.navigateHistory(1);
      return;
    }
    if (kb.matches(data, "tui.editor.cursorLineStart")) {
      this.moveToLineStart();
      return;
    }
    if (kb.matches(data, "tui.editor.cursorLineEnd")) {
      this.moveToLineEnd();
      return;
    }
    if (kb.matches(data, "tui.editor.cursorWordLeft")) {
      this.moveWordBackwards();
      return;
    }
    if (kb.matches(data, "tui.editor.cursorWordRight")) {
      this.moveWordForwards();
      return;
    }
    if (kb.matches(data, "tui.input.newLine") || data.charCodeAt(0) === 10 && data.length > 1 || data === "\x1B\r" || data === "\x1B[13;2~" || data.length > 1 && data.includes("\x1B") && data.includes("\r") || data === "\n" && data.length === 1) {
      if (this.shouldSubmitOnBackslashEnter(data, kb)) {
        this.handleBackspace();
        this.submitValue();
        return;
      }
      this.addNewLine();
      return;
    }
    if (kb.matches(data, "tui.input.submit")) {
      if (this.disableSubmit)
        return;
      const currentLine = this.state.lines[this.state.cursorLine] || "";
      if (this.state.cursorCol > 0 && currentLine[this.state.cursorCol - 1] === "\\") {
        this.handleBackspace();
        this.addNewLine();
        return;
      }
      this.submitValue();
      return;
    }
    if (kb.matches(data, "tui.editor.cursorUp")) {
      if (this.isOnFirstVisualLine() && (this.isEditorEmpty() || this.historyIndex > -1 || this.state.cursorCol === 0)) {
        this.navigateHistory(-1);
      } else if (this.isOnFirstVisualLine()) {
        this.moveToLineStart();
      } else {
        this.moveCursor(-1, 0);
      }
      return;
    }
    if (kb.matches(data, "tui.editor.cursorDown")) {
      if (this.historyIndex > -1 && this.isOnLastVisualLine()) {
        this.navigateHistory(1);
      } else if (this.isOnLastVisualLine()) {
        this.moveToLineEnd();
      } else {
        this.moveCursor(1, 0);
      }
      return;
    }
    if (kb.matches(data, "tui.editor.cursorRight")) {
      this.moveCursor(0, 1);
      return;
    }
    if (kb.matches(data, "tui.editor.cursorLeft")) {
      this.moveCursor(0, -1);
      return;
    }
    if (kb.matches(data, "tui.editor.pageUp")) {
      this.pageScroll(-1);
      return;
    }
    if (kb.matches(data, "tui.editor.pageDown")) {
      this.pageScroll(1);
      return;
    }
    if (kb.matches(data, "tui.editor.jumpForward")) {
      this.jumpMode = "forward";
      return;
    }
    if (kb.matches(data, "tui.editor.jumpBackward")) {
      this.jumpMode = "backward";
      return;
    }
    if (matchesKey(data, "shift+space")) {
      this.insertCharacter(" ");
      return;
    }
    const printable = decodePrintableKey(data);
    if (printable !== void 0) {
      this.insertCharacter(printable);
      return;
    }
    if (data.charCodeAt(0) >= 32) {
      this.insertCharacter(data);
    }
  }
  layoutText(contentWidth) {
    const layoutLines = [];
    if (this.state.lines.length === 0 || this.state.lines.length === 1 && this.state.lines[0] === "") {
      layoutLines.push({
        text: "",
        hasCursor: true,
        cursorPos: 0
      });
      return layoutLines;
    }
    for (let i = 0; i < this.state.lines.length; i++) {
      const line = this.state.lines[i] || "";
      const isCurrentLine = i === this.state.cursorLine;
      const lineVisibleWidth = visibleWidth(line);
      if (lineVisibleWidth <= contentWidth) {
        if (isCurrentLine) {
          layoutLines.push({
            text: line,
            hasCursor: true,
            cursorPos: this.state.cursorCol
          });
        } else {
          layoutLines.push({
            text: line,
            hasCursor: false
          });
        }
      } else {
        const chunks = wordWrapLine(line, contentWidth, [...this.segment(line, "grapheme")]);
        for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
          const chunk = chunks[chunkIndex];
          if (!chunk)
            continue;
          const cursorPos = this.state.cursorCol;
          const isLastChunk = chunkIndex === chunks.length - 1;
          let hasCursorInChunk = false;
          let adjustedCursorPos = 0;
          if (isCurrentLine) {
            if (isLastChunk) {
              hasCursorInChunk = cursorPos >= chunk.startIndex;
              adjustedCursorPos = cursorPos - chunk.startIndex;
            } else {
              hasCursorInChunk = cursorPos >= chunk.startIndex && cursorPos < chunk.endIndex;
              if (hasCursorInChunk) {
                adjustedCursorPos = cursorPos - chunk.startIndex;
                if (adjustedCursorPos > chunk.text.length) {
                  adjustedCursorPos = chunk.text.length;
                }
              }
            }
          }
          if (hasCursorInChunk) {
            layoutLines.push({
              text: chunk.text,
              hasCursor: true,
              cursorPos: adjustedCursorPos
            });
          } else {
            layoutLines.push({
              text: chunk.text,
              hasCursor: false
            });
          }
        }
      }
    }
    return layoutLines;
  }
  getText() {
    return this.state.lines.join("\n");
  }
  expandPasteMarkers(text) {
    let result = text;
    for (const [pasteId, pasteContent] of this.pastes) {
      const markerRegex = new RegExp(`\\[paste #${pasteId}( (\\+\\d+ lines|\\d+ chars))?\\]`, "g");
      result = result.replace(markerRegex, () => pasteContent);
    }
    return result;
  }
  /**
   * Get text with paste markers expanded to their actual content.
   * Use this when you need the full content (e.g., for external editor).
   */
  getExpandedText() {
    return this.expandPasteMarkers(this.state.lines.join("\n"));
  }
  getLines() {
    return [...this.state.lines];
  }
  getCursor() {
    return { line: this.state.cursorLine, col: this.state.cursorCol };
  }
  setText(text) {
    this.cancelAutocomplete();
    this.lastAction = null;
    this.exitHistoryBrowsing();
    const normalized = this.normalizeText(text);
    if (this.getText() !== normalized) {
      this.pushUndoSnapshot();
    }
    this.pastes.clear();
    this.pasteCounter = 0;
    this.setTextInternal(normalized);
  }
  /**
   * Insert text at the current cursor position.
   * Used for programmatic insertion (e.g., clipboard image markers).
   * This is atomic for undo - single undo restores entire pre-insert state.
   */
  insertTextAtCursor(text) {
    if (!text)
      return;
    this.cancelAutocomplete();
    this.pushUndoSnapshot();
    this.lastAction = null;
    this.exitHistoryBrowsing();
    this.insertTextAtCursorInternal(text);
  }
  /**
   * Normalize text for editor storage:
   * - Normalize line endings (\r\n and \r -> \n)
   * - Expand tabs to 4 spaces
   */
  normalizeText(text) {
    return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").replace(/\t/g, "    ");
  }
  /**
   * Internal text insertion at cursor. Handles single and multi-line text.
   * Does not push undo snapshots or trigger autocomplete - caller is responsible.
   * Normalizes line endings and calls onChange once at the end.
   */
  insertTextAtCursorInternal(text) {
    if (!text)
      return;
    const normalized = this.normalizeText(text);
    const insertedLines = normalized.split("\n");
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    const beforeCursor = currentLine.slice(0, this.state.cursorCol);
    const afterCursor = currentLine.slice(this.state.cursorCol);
    if (insertedLines.length === 1) {
      this.state.lines[this.state.cursorLine] = beforeCursor + normalized + afterCursor;
      this.setCursorCol(this.state.cursorCol + normalized.length);
    } else {
      this.state.lines = [
        // All lines before current line
        ...this.state.lines.slice(0, this.state.cursorLine),
        // The first inserted line merged with text before cursor
        beforeCursor + insertedLines[0],
        // All middle inserted lines
        ...insertedLines.slice(1, -1),
        // The last inserted line with text after cursor
        insertedLines[insertedLines.length - 1] + afterCursor,
        // All lines after current line
        ...this.state.lines.slice(this.state.cursorLine + 1)
      ];
      this.state.cursorLine += insertedLines.length - 1;
      this.setCursorCol((insertedLines[insertedLines.length - 1] || "").length);
    }
    if (this.onChange) {
      this.onChange(this.getText());
    }
  }
  // All the editor methods from before...
  insertCharacter(char, skipUndoCoalescing) {
    this.exitHistoryBrowsing();
    if (!skipUndoCoalescing) {
      if (isWhitespaceChar(char) || this.lastAction !== "type-word") {
        this.pushUndoSnapshot();
      }
      this.lastAction = "type-word";
    }
    const line = this.state.lines[this.state.cursorLine] || "";
    const before = line.slice(0, this.state.cursorCol);
    const after = line.slice(this.state.cursorCol);
    this.state.lines[this.state.cursorLine] = before + char + after;
    this.setCursorCol(this.state.cursorCol + char.length);
    if (this.onChange) {
      this.onChange(this.getText());
    }
    if (!this.autocompleteState) {
      if (char === "/" && this.isAtStartOfMessage()) {
        this.tryTriggerAutocomplete();
      } else if (this.autocompleteTriggerCharacters.includes(char)) {
        const currentLine = this.state.lines[this.state.cursorLine] || "";
        const textBeforeCursor = currentLine.slice(0, this.state.cursorCol);
        const charBeforeSymbol = textBeforeCursor[textBeforeCursor.length - 2];
        if (textBeforeCursor.length === 1 || charBeforeSymbol === " " || charBeforeSymbol === "	") {
          this.tryTriggerAutocomplete();
        }
      } else if (/[a-zA-Z0-9.\-_]/.test(char)) {
        const currentLine = this.state.lines[this.state.cursorLine] || "";
        const textBeforeCursor = currentLine.slice(0, this.state.cursorCol);
        if (this.isInSlashCommandContext(textBeforeCursor)) {
          this.tryTriggerAutocomplete();
        } else if (this.autocompleteTriggerPattern.test(textBeforeCursor)) {
          this.tryTriggerAutocomplete();
        }
      }
    } else {
      this.updateAutocomplete();
    }
  }
  handlePaste(pastedText) {
    this.cancelAutocomplete();
    this.exitHistoryBrowsing();
    this.lastAction = null;
    this.pushUndoSnapshot();
    const decodedText = pastedText.replace(/\x1b\[(\d+);5u/g, (match, code) => {
      const cp = Number(code);
      if (cp >= 97 && cp <= 122)
        return String.fromCharCode(cp - 96);
      if (cp >= 65 && cp <= 90)
        return String.fromCharCode(cp - 64);
      return match;
    });
    const cleanText = this.normalizeText(decodedText);
    let filteredText = cleanText.split("").filter((char) => char === "\n" || char.charCodeAt(0) >= 32).join("");
    if (/^[/~.]/.test(filteredText)) {
      const currentLine = this.state.lines[this.state.cursorLine] || "";
      const charBeforeCursor = this.state.cursorCol > 0 ? currentLine[this.state.cursorCol - 1] : "";
      if (charBeforeCursor && /\w/.test(charBeforeCursor)) {
        filteredText = ` ${filteredText}`;
      }
    }
    const pastedLines = filteredText.split("\n");
    const totalChars = filteredText.length;
    if (pastedLines.length > 10 || totalChars > 1e3) {
      this.pasteCounter++;
      const pasteId = this.pasteCounter;
      this.pastes.set(pasteId, filteredText);
      const marker = pastedLines.length > 10 ? `[paste #${pasteId} +${pastedLines.length} lines]` : `[paste #${pasteId} ${totalChars} chars]`;
      this.insertTextAtCursorInternal(marker);
      return;
    }
    if (pastedLines.length === 1) {
      this.insertTextAtCursorInternal(filteredText);
      return;
    }
    this.insertTextAtCursorInternal(filteredText);
  }
  addNewLine() {
    this.cancelAutocomplete();
    this.exitHistoryBrowsing();
    this.lastAction = null;
    this.pushUndoSnapshot();
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    const before = currentLine.slice(0, this.state.cursorCol);
    const after = currentLine.slice(this.state.cursorCol);
    this.state.lines[this.state.cursorLine] = before;
    this.state.lines.splice(this.state.cursorLine + 1, 0, after);
    this.state.cursorLine++;
    this.setCursorCol(0);
    if (this.onChange) {
      this.onChange(this.getText());
    }
  }
  shouldSubmitOnBackslashEnter(data, kb) {
    if (this.disableSubmit)
      return false;
    if (!matchesKey(data, "enter"))
      return false;
    const submitKeys = kb.getKeys("tui.input.submit");
    const hasShiftEnter = submitKeys.includes("shift+enter") || submitKeys.includes("shift+return");
    if (!hasShiftEnter)
      return false;
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    return this.state.cursorCol > 0 && currentLine[this.state.cursorCol - 1] === "\\";
  }
  submitValue() {
    this.cancelAutocomplete();
    const result = this.expandPasteMarkers(this.state.lines.join("\n")).trim();
    this.state = { lines: [""], cursorLine: 0, cursorCol: 0 };
    this.pastes.clear();
    this.pasteCounter = 0;
    this.exitHistoryBrowsing();
    this.scrollOffset = 0;
    this.undoStack.clear();
    this.lastAction = null;
    if (this.onChange)
      this.onChange("");
    if (this.onSubmit)
      this.onSubmit(result);
  }
  handleBackspace() {
    this.exitHistoryBrowsing();
    this.lastAction = null;
    if (this.state.cursorCol > 0) {
      this.pushUndoSnapshot();
      let line = this.state.lines[this.state.cursorLine] || "";
      const beforeCursor = line.slice(0, this.state.cursorCol);
      const graphemes = [...this.segment(beforeCursor, "grapheme")];
      const lastGrapheme = graphemes[graphemes.length - 1];
      const graphemeLength = lastGrapheme ? lastGrapheme.segment.length : 1;
      const isPastedSegmented = PASTE_MARKER_SINGLE.exec(lastGrapheme.segment);
      if (isPastedSegmented) {
        const targetId = Number(isPastedSegmented[1]);
        this.pastes.delete(targetId);
        this.pasteCounter--;
        const higherIds = [...this.pastes.keys()].filter((id) => id > targetId).sort((a, b2) => a - b2);
        for (const id of higherIds) {
          this.pastes.set(id - 1, this.pastes.get(id));
          this.pastes.delete(id);
        }
        this.state.lines = this.state.lines.map((line2) => line2.replace(PASTE_MARKER_REGEX, (fullMatch, idGroup, suffixGroup) => {
          const x2 = Number(idGroup);
          if (x2 <= targetId)
            return fullMatch;
          return `[paste #${x2 - 1}${suffixGroup}]`;
        }));
      }
      line = this.state.lines[this.state.cursorLine] || "";
      const before = line.slice(0, this.state.cursorCol - graphemeLength);
      const after = line.slice(this.state.cursorCol);
      this.state.lines[this.state.cursorLine] = before + after;
      this.setCursorCol(this.state.cursorCol - graphemeLength);
    } else if (this.state.cursorLine > 0) {
      this.pushUndoSnapshot();
      const currentLine = this.state.lines[this.state.cursorLine] || "";
      const previousLine = this.state.lines[this.state.cursorLine - 1] || "";
      this.state.lines[this.state.cursorLine - 1] = previousLine + currentLine;
      this.state.lines.splice(this.state.cursorLine, 1);
      this.state.cursorLine--;
      this.setCursorCol(previousLine.length);
    }
    if (this.onChange) {
      this.onChange(this.getText());
    }
    if (this.autocompleteState) {
      this.updateAutocomplete();
    } else {
      const currentLine = this.state.lines[this.state.cursorLine] || "";
      const textBeforeCursor = currentLine.slice(0, this.state.cursorCol);
      if (this.isInSlashCommandContext(textBeforeCursor)) {
        this.tryTriggerAutocomplete();
      } else if (this.autocompleteTriggerPattern.test(textBeforeCursor)) {
        this.tryTriggerAutocomplete();
      }
    }
  }
  /**
   * Set cursor column and clear preferredVisualCol.
   * Use this for all non-vertical cursor movements to reset sticky column behavior.
   */
  setCursorCol(col) {
    this.state.cursorCol = col;
    this.preferredVisualCol = null;
    this.snappedFromCursorCol = null;
  }
  /**
   * Move cursor to a target visual line, applying sticky column logic.
   * Shared by moveCursor() and pageScroll().
   */
  moveToVisualLine(visualLines, currentVisualLine, targetVisualLine) {
    const currentVL = visualLines[currentVisualLine];
    const targetVL = visualLines[targetVisualLine];
    if (!(currentVL && targetVL))
      return;
    let currentVisualCol;
    if (this.snappedFromCursorCol !== null) {
      const vlIndex = this.findVisualLineAt(visualLines, currentVL.logicalLine, this.snappedFromCursorCol);
      currentVisualCol = this.snappedFromCursorCol - visualLines[vlIndex].startCol;
    } else {
      currentVisualCol = this.state.cursorCol - currentVL.startCol;
    }
    const isLastSourceSegment = currentVisualLine === visualLines.length - 1 || visualLines[currentVisualLine + 1]?.logicalLine !== currentVL.logicalLine;
    const sourceMaxVisualCol = isLastSourceSegment ? currentVL.length : Math.max(0, currentVL.length - 1);
    const isLastTargetSegment = targetVisualLine === visualLines.length - 1 || visualLines[targetVisualLine + 1]?.logicalLine !== targetVL.logicalLine;
    const targetMaxVisualCol = isLastTargetSegment ? targetVL.length : Math.max(0, targetVL.length - 1);
    const moveToVisualCol = this.computeVerticalMoveColumn(currentVisualCol, sourceMaxVisualCol, targetMaxVisualCol);
    this.state.cursorLine = targetVL.logicalLine;
    const targetCol = targetVL.startCol + moveToVisualCol;
    const logicalLine = this.state.lines[targetVL.logicalLine] || "";
    this.state.cursorCol = Math.min(targetCol, logicalLine.length);
    const segments = [...this.segment(logicalLine, "grapheme")];
    for (const seg of segments) {
      if (seg.index > this.state.cursorCol)
        break;
      if (seg.segment.length <= 1)
        continue;
      if (this.state.cursorCol < seg.index + seg.segment.length) {
        const isContinuation = seg.index < targetVL.startCol;
        const isMovingDown = targetVisualLine > currentVisualLine;
        if (isContinuation && isMovingDown) {
          const segEnd = seg.index + seg.segment.length;
          let next = targetVisualLine + 1;
          while (next < visualLines.length && visualLines[next].logicalLine === targetVL.logicalLine && visualLines[next].startCol < segEnd) {
            next++;
          }
          if (next < visualLines.length) {
            this.moveToVisualLine(visualLines, currentVisualLine, next);
            return;
          }
        }
        this.snappedFromCursorCol = this.state.cursorCol;
        this.state.cursorCol = seg.index;
        return;
      }
    }
    this.snappedFromCursorCol = null;
  }
  /**
   * Compute the target visual column for vertical cursor movement.
   * Implements the sticky column decision table:
   *
   * | P | S | T | U | Scenario                                             | Set Preferred | Move To     |
   * |---|---|---|---| ---------------------------------------------------- |---------------|-------------|
   * | 0 | * | 0 | - | Start nav, target fits                               | null          | current     |
   * | 0 | * | 1 | - | Start nav, target shorter                            | current       | target end  |
   * | 1 | 0 | 0 | 0 | Clamped, target fits preferred                       | null          | preferred   |
   * | 1 | 0 | 0 | 1 | Clamped, target longer but still can't fit preferred | keep          | target end  |
   * | 1 | 0 | 1 | - | Clamped, target even shorter                         | keep          | target end  |
   * | 1 | 1 | 0 | - | Rewrapped, target fits current                       | null          | current     |
   * | 1 | 1 | 1 | - | Rewrapped, target shorter than current               | current       | target end  |
   *
   * Where:
   * - P = preferred col is set
   * - S = cursor in middle of source line (not clamped to end)
   * - T = target line shorter than current visual col
   * - U = target line shorter than preferred col
   */
  computeVerticalMoveColumn(currentVisualCol, sourceMaxVisualCol, targetMaxVisualCol) {
    const hasPreferred = this.preferredVisualCol !== null;
    const cursorInMiddle = currentVisualCol < sourceMaxVisualCol;
    const targetTooShort = targetMaxVisualCol < currentVisualCol;
    if (!hasPreferred || cursorInMiddle) {
      if (targetTooShort) {
        this.preferredVisualCol = currentVisualCol;
        return targetMaxVisualCol;
      }
      this.preferredVisualCol = null;
      return currentVisualCol;
    }
    const targetCantFitPreferred = targetMaxVisualCol < this.preferredVisualCol;
    if (targetTooShort || targetCantFitPreferred) {
      return targetMaxVisualCol;
    }
    const result = this.preferredVisualCol;
    this.preferredVisualCol = null;
    return result;
  }
  moveToLineStart() {
    this.lastAction = null;
    this.setCursorCol(0);
  }
  moveToLineEnd() {
    this.lastAction = null;
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    this.setCursorCol(currentLine.length);
  }
  deleteToStartOfLine() {
    this.exitHistoryBrowsing();
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    if (this.state.cursorCol > 0) {
      this.pushUndoSnapshot();
      const deletedText = currentLine.slice(0, this.state.cursorCol);
      this.killRing.push(deletedText, { prepend: true, accumulate: this.lastAction === "kill" });
      this.lastAction = "kill";
      this.state.lines[this.state.cursorLine] = currentLine.slice(this.state.cursorCol);
      this.setCursorCol(0);
    } else if (this.state.cursorLine > 0) {
      this.pushUndoSnapshot();
      this.killRing.push("\n", { prepend: true, accumulate: this.lastAction === "kill" });
      this.lastAction = "kill";
      const previousLine = this.state.lines[this.state.cursorLine - 1] || "";
      this.state.lines[this.state.cursorLine - 1] = previousLine + currentLine;
      this.state.lines.splice(this.state.cursorLine, 1);
      this.state.cursorLine--;
      this.setCursorCol(previousLine.length);
    }
    if (this.onChange) {
      this.onChange(this.getText());
    }
  }
  deleteToEndOfLine() {
    this.exitHistoryBrowsing();
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    if (this.state.cursorCol < currentLine.length) {
      this.pushUndoSnapshot();
      const deletedText = currentLine.slice(this.state.cursorCol);
      this.killRing.push(deletedText, { prepend: false, accumulate: this.lastAction === "kill" });
      this.lastAction = "kill";
      this.state.lines[this.state.cursorLine] = currentLine.slice(0, this.state.cursorCol);
    } else if (this.state.cursorLine < this.state.lines.length - 1) {
      this.pushUndoSnapshot();
      this.killRing.push("\n", { prepend: false, accumulate: this.lastAction === "kill" });
      this.lastAction = "kill";
      const nextLine = this.state.lines[this.state.cursorLine + 1] || "";
      this.state.lines[this.state.cursorLine] = currentLine + nextLine;
      this.state.lines.splice(this.state.cursorLine + 1, 1);
    }
    if (this.onChange) {
      this.onChange(this.getText());
    }
  }
  deleteWordBackwards() {
    this.exitHistoryBrowsing();
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    if (this.state.cursorCol === 0) {
      if (this.state.cursorLine > 0) {
        this.pushUndoSnapshot();
        this.killRing.push("\n", { prepend: true, accumulate: this.lastAction === "kill" });
        this.lastAction = "kill";
        const previousLine = this.state.lines[this.state.cursorLine - 1] || "";
        this.state.lines[this.state.cursorLine - 1] = previousLine + currentLine;
        this.state.lines.splice(this.state.cursorLine, 1);
        this.state.cursorLine--;
        this.setCursorCol(previousLine.length);
      }
    } else {
      this.pushUndoSnapshot();
      const wasKill = this.lastAction === "kill";
      const oldCursorCol = this.state.cursorCol;
      this.moveWordBackwards();
      const deleteFrom = this.state.cursorCol;
      this.setCursorCol(oldCursorCol);
      const deletedText = currentLine.slice(deleteFrom, this.state.cursorCol);
      this.killRing.push(deletedText, { prepend: true, accumulate: wasKill });
      this.lastAction = "kill";
      this.state.lines[this.state.cursorLine] = currentLine.slice(0, deleteFrom) + currentLine.slice(this.state.cursorCol);
      this.setCursorCol(deleteFrom);
    }
    if (this.onChange) {
      this.onChange(this.getText());
    }
  }
  deleteWordForward() {
    this.exitHistoryBrowsing();
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    if (this.state.cursorCol >= currentLine.length) {
      if (this.state.cursorLine < this.state.lines.length - 1) {
        this.pushUndoSnapshot();
        this.killRing.push("\n", { prepend: false, accumulate: this.lastAction === "kill" });
        this.lastAction = "kill";
        const nextLine = this.state.lines[this.state.cursorLine + 1] || "";
        this.state.lines[this.state.cursorLine] = currentLine + nextLine;
        this.state.lines.splice(this.state.cursorLine + 1, 1);
      }
    } else {
      this.pushUndoSnapshot();
      const wasKill = this.lastAction === "kill";
      const oldCursorCol = this.state.cursorCol;
      this.moveWordForwards();
      const deleteTo = this.state.cursorCol;
      this.setCursorCol(oldCursorCol);
      const deletedText = currentLine.slice(this.state.cursorCol, deleteTo);
      this.killRing.push(deletedText, { prepend: false, accumulate: wasKill });
      this.lastAction = "kill";
      this.state.lines[this.state.cursorLine] = currentLine.slice(0, this.state.cursorCol) + currentLine.slice(deleteTo);
    }
    if (this.onChange) {
      this.onChange(this.getText());
    }
  }
  handleForwardDelete() {
    this.exitHistoryBrowsing();
    this.lastAction = null;
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    if (this.state.cursorCol < currentLine.length) {
      this.pushUndoSnapshot();
      const afterCursor = currentLine.slice(this.state.cursorCol);
      const graphemes = [...this.segment(afterCursor, "grapheme")];
      const firstGrapheme = graphemes[0];
      const graphemeLength = firstGrapheme ? firstGrapheme.segment.length : 1;
      const before = currentLine.slice(0, this.state.cursorCol);
      const after = currentLine.slice(this.state.cursorCol + graphemeLength);
      this.state.lines[this.state.cursorLine] = before + after;
    } else if (this.state.cursorLine < this.state.lines.length - 1) {
      this.pushUndoSnapshot();
      const nextLine = this.state.lines[this.state.cursorLine + 1] || "";
      this.state.lines[this.state.cursorLine] = currentLine + nextLine;
      this.state.lines.splice(this.state.cursorLine + 1, 1);
    }
    if (this.onChange) {
      this.onChange(this.getText());
    }
    if (this.autocompleteState) {
      this.updateAutocomplete();
    } else {
      const currentLine2 = this.state.lines[this.state.cursorLine] || "";
      const textBeforeCursor = currentLine2.slice(0, this.state.cursorCol);
      if (this.isInSlashCommandContext(textBeforeCursor)) {
        this.tryTriggerAutocomplete();
      } else if (this.autocompleteTriggerPattern.test(textBeforeCursor)) {
        this.tryTriggerAutocomplete();
      }
    }
  }
  /**
   * Build a mapping from visual lines to logical positions.
   * Returns an array where each element represents a visual line with:
   * - logicalLine: index into this.state.lines
   * - startCol: starting column in the logical line
   * - length: length of this visual line segment
   */
  buildVisualLineMap(width) {
    const visualLines = [];
    for (let i = 0; i < this.state.lines.length; i++) {
      const line = this.state.lines[i] || "";
      const lineVisWidth = visibleWidth(line);
      if (line.length === 0) {
        visualLines.push({ logicalLine: i, startCol: 0, length: 0 });
      } else if (lineVisWidth <= width) {
        visualLines.push({ logicalLine: i, startCol: 0, length: line.length });
      } else {
        const chunks = wordWrapLine(line, width, [...this.segment(line, "grapheme")]);
        for (const chunk of chunks) {
          visualLines.push({
            logicalLine: i,
            startCol: chunk.startIndex,
            length: chunk.endIndex - chunk.startIndex
          });
        }
      }
    }
    return visualLines;
  }
  /**
   * Find the visual line index that contains the given logical position.
   */
  findVisualLineAt(visualLines, line, col) {
    for (let i = 0; i < visualLines.length; i++) {
      const vl = visualLines[i];
      if (!vl || vl.logicalLine !== line)
        continue;
      const offset = col - vl.startCol;
      const isLastSegmentOfLine = i === visualLines.length - 1 || visualLines[i + 1]?.logicalLine !== vl.logicalLine;
      if (offset >= 0 && (offset < vl.length || isLastSegmentOfLine && offset === vl.length)) {
        return i;
      }
    }
    return visualLines.length - 1;
  }
  /**
   * Find the visual line index for the current cursor position.
   */
  findCurrentVisualLine(visualLines) {
    return this.findVisualLineAt(visualLines, this.state.cursorLine, this.state.cursorCol);
  }
  moveCursor(deltaLine, deltaCol) {
    this.lastAction = null;
    const visualLines = this.buildVisualLineMap(this.lastWidth);
    const currentVisualLine = this.findCurrentVisualLine(visualLines);
    if (deltaLine !== 0) {
      const targetVisualLine = currentVisualLine + deltaLine;
      if (targetVisualLine >= 0 && targetVisualLine < visualLines.length) {
        this.moveToVisualLine(visualLines, currentVisualLine, targetVisualLine);
      }
    }
    if (deltaCol !== 0) {
      const currentLine = this.state.lines[this.state.cursorLine] || "";
      if (deltaCol > 0) {
        if (this.state.cursorCol < currentLine.length) {
          const afterCursor = currentLine.slice(this.state.cursorCol);
          const graphemes = [...this.segment(afterCursor, "grapheme")];
          const firstGrapheme = graphemes[0];
          this.setCursorCol(this.state.cursorCol + (firstGrapheme ? firstGrapheme.segment.length : 1));
        } else if (this.state.cursorLine < this.state.lines.length - 1) {
          this.state.cursorLine++;
          this.setCursorCol(0);
        } else {
          const currentVL = visualLines[currentVisualLine];
          if (currentVL) {
            this.preferredVisualCol = this.state.cursorCol - currentVL.startCol;
          }
        }
      } else {
        if (this.state.cursorCol > 0) {
          const beforeCursor = currentLine.slice(0, this.state.cursorCol);
          const graphemes = [...this.segment(beforeCursor, "grapheme")];
          const lastGrapheme = graphemes[graphemes.length - 1];
          this.setCursorCol(this.state.cursorCol - (lastGrapheme ? lastGrapheme.segment.length : 1));
        } else if (this.state.cursorLine > 0) {
          this.state.cursorLine--;
          const prevLine = this.state.lines[this.state.cursorLine] || "";
          this.setCursorCol(prevLine.length);
        }
      }
    }
    if (this.autocompleteState) {
      this.updateAutocomplete();
    }
  }
  /**
   * Scroll by a page (direction: -1 for up, 1 for down).
   * Moves cursor by the page size while keeping it in bounds.
   */
  pageScroll(direction) {
    this.lastAction = null;
    const terminalRows = this.tui.terminal.rows;
    const pageSize = Math.max(5, Math.floor(terminalRows * 0.3));
    const visualLines = this.buildVisualLineMap(this.lastWidth);
    const currentVisualLine = this.findCurrentVisualLine(visualLines);
    const targetVisualLine = Math.max(0, Math.min(visualLines.length - 1, currentVisualLine + direction * pageSize));
    this.moveToVisualLine(visualLines, currentVisualLine, targetVisualLine);
  }
  moveWordBackwards() {
    this.lastAction = null;
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    if (this.state.cursorCol === 0) {
      if (this.state.cursorLine > 0) {
        this.state.cursorLine--;
        const prevLine = this.state.lines[this.state.cursorLine] || "";
        this.setCursorCol(prevLine.length);
      }
      return;
    }
    this.setCursorCol(findWordBackward(currentLine, this.state.cursorCol, {
      segment: (text) => this.segment(text, "word"),
      isAtomicSegment: isPasteMarker
    }));
  }
  /**
   * Yank (paste) the most recent kill ring entry at cursor position.
   */
  yank() {
    if (this.killRing.length === 0)
      return;
    this.pushUndoSnapshot();
    const text = this.killRing.peek();
    this.insertYankedText(text);
    this.lastAction = "yank";
  }
  /**
   * Cycle through kill ring (only works immediately after yank or yank-pop).
   * Replaces the last yanked text with the previous entry in the ring.
   */
  yankPop() {
    if (this.lastAction !== "yank" || this.killRing.length <= 1)
      return;
    this.pushUndoSnapshot();
    this.deleteYankedText();
    this.killRing.rotate();
    const text = this.killRing.peek();
    this.insertYankedText(text);
    this.lastAction = "yank";
  }
  /**
   * Insert text at cursor position (used by yank operations).
   */
  insertYankedText(text) {
    this.exitHistoryBrowsing();
    const lines = text.split("\n");
    if (lines.length === 1) {
      const currentLine = this.state.lines[this.state.cursorLine] || "";
      const before = currentLine.slice(0, this.state.cursorCol);
      const after = currentLine.slice(this.state.cursorCol);
      this.state.lines[this.state.cursorLine] = before + text + after;
      this.setCursorCol(this.state.cursorCol + text.length);
    } else {
      const currentLine = this.state.lines[this.state.cursorLine] || "";
      const before = currentLine.slice(0, this.state.cursorCol);
      const after = currentLine.slice(this.state.cursorCol);
      this.state.lines[this.state.cursorLine] = before + (lines[0] || "");
      for (let i = 1; i < lines.length - 1; i++) {
        this.state.lines.splice(this.state.cursorLine + i, 0, lines[i] || "");
      }
      const lastLineIndex = this.state.cursorLine + lines.length - 1;
      this.state.lines.splice(lastLineIndex, 0, (lines[lines.length - 1] || "") + after);
      this.state.cursorLine = lastLineIndex;
      this.setCursorCol((lines[lines.length - 1] || "").length);
    }
    if (this.onChange) {
      this.onChange(this.getText());
    }
  }
  /**
   * Delete the previously yanked text (used by yank-pop).
   * The yanked text is derived from killRing[end] since it hasn't been rotated yet.
   */
  deleteYankedText() {
    const yankedText = this.killRing.peek();
    if (!yankedText)
      return;
    const yankLines = yankedText.split("\n");
    if (yankLines.length === 1) {
      const currentLine = this.state.lines[this.state.cursorLine] || "";
      const deleteLen = yankedText.length;
      const before = currentLine.slice(0, this.state.cursorCol - deleteLen);
      const after = currentLine.slice(this.state.cursorCol);
      this.state.lines[this.state.cursorLine] = before + after;
      this.setCursorCol(this.state.cursorCol - deleteLen);
    } else {
      const startLine = this.state.cursorLine - (yankLines.length - 1);
      const startCol = (this.state.lines[startLine] || "").length - (yankLines[0] || "").length;
      const afterCursor = (this.state.lines[this.state.cursorLine] || "").slice(this.state.cursorCol);
      const beforeYank = (this.state.lines[startLine] || "").slice(0, startCol);
      this.state.lines.splice(startLine, yankLines.length, beforeYank + afterCursor);
      this.state.cursorLine = startLine;
      this.setCursorCol(startCol);
    }
    if (this.onChange) {
      this.onChange(this.getText());
    }
  }
  pushUndoSnapshot() {
    this.undoStack.push({ state: this.state, pastes: this.pastes, pasteCounter: this.pasteCounter });
  }
  undo() {
    this.exitHistoryBrowsing();
    const snapshot = this.undoStack.pop();
    if (!snapshot)
      return;
    Object.assign(this.state, snapshot.state);
    this.pastes = snapshot.pastes;
    this.pasteCounter = snapshot.pasteCounter;
    this.lastAction = null;
    this.preferredVisualCol = null;
    if (this.onChange) {
      this.onChange(this.getText());
    }
  }
  /**
   * Jump to the first occurrence of a character in the specified direction.
   * Multi-line search. Case-sensitive. Skips the current cursor position.
   */
  jumpToChar(char, direction) {
    this.lastAction = null;
    const isForward = direction === "forward";
    const lines = this.state.lines;
    const end = isForward ? lines.length : -1;
    const step = isForward ? 1 : -1;
    for (let lineIdx = this.state.cursorLine; lineIdx !== end; lineIdx += step) {
      const line = lines[lineIdx] || "";
      const isCurrentLine = lineIdx === this.state.cursorLine;
      const searchFrom = isCurrentLine ? isForward ? this.state.cursorCol + 1 : this.state.cursorCol - 1 : void 0;
      const idx = isForward ? line.indexOf(char, searchFrom) : line.lastIndexOf(char, searchFrom);
      if (idx !== -1) {
        this.state.cursorLine = lineIdx;
        this.setCursorCol(idx);
        return;
      }
    }
  }
  moveWordForwards() {
    this.lastAction = null;
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    if (this.state.cursorCol >= currentLine.length) {
      if (this.state.cursorLine < this.state.lines.length - 1) {
        this.state.cursorLine++;
        this.setCursorCol(0);
      }
      return;
    }
    this.setCursorCol(findWordForward(currentLine, this.state.cursorCol, {
      segment: (text) => this.segment(text, "word"),
      isAtomicSegment: isPasteMarker
    }));
  }
  // Slash menu only allowed on the first line of the editor
  isSlashMenuAllowed() {
    return this.state.cursorLine === 0;
  }
  // Helper method to check if cursor is at start of message (for slash command detection)
  isAtStartOfMessage() {
    if (!this.isSlashMenuAllowed())
      return false;
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    const beforeCursor = currentLine.slice(0, this.state.cursorCol);
    return beforeCursor.trim() === "" || beforeCursor.trim() === "/";
  }
  isInSlashCommandContext(textBeforeCursor) {
    return this.isSlashMenuAllowed() && textBeforeCursor.trimStart().startsWith("/");
  }
  // Autocomplete methods
  /**
   * Find the best autocomplete item index for the given prefix.
   * Returns -1 if no match is found.
   *
   * Match priority:
   * 1. Exact match (prefix === item.value) -> always selected
   * 2. Prefix match -> first item whose value starts with prefix
   * 3. No match -> -1 (keep default highlight)
   *
   * Matching is case-sensitive and checks item.value only.
   */
  getBestAutocompleteMatchIndex(items, prefix) {
    if (!prefix)
      return -1;
    let firstPrefixIndex = -1;
    for (let i = 0; i < items.length; i++) {
      const value = items[i].value;
      if (value === prefix) {
        return i;
      }
      if (firstPrefixIndex === -1 && value.startsWith(prefix)) {
        firstPrefixIndex = i;
      }
    }
    return firstPrefixIndex;
  }
  createAutocompleteList(prefix, items) {
    const layout = prefix.startsWith("/") ? SLASH_COMMAND_SELECT_LIST_LAYOUT : void 0;
    const list = new SelectList(items, this.autocompleteMaxVisible, this.theme.selectList, layout);
    list.onSelect = (selected) => {
      if (!this.autocompleteProvider)
        return;
      this.pushUndoSnapshot();
      this.lastAction = null;
      const result = this.autocompleteProvider.applyCompletion(this.state.lines, this.state.cursorLine, this.state.cursorCol, selected, this.autocompletePrefix);
      this.state.lines = result.lines;
      this.state.cursorLine = result.cursorLine;
      this.setCursorCol(result.cursorCol);
      this.cancelAutocomplete();
      this.onChange?.(this.getText());
    };
    return list;
  }
  tryTriggerAutocomplete(explicitTab = false) {
    this.requestAutocomplete({ force: false, explicitTab });
  }
  handleTabCompletion() {
    if (!this.autocompleteProvider)
      return;
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    const beforeCursor = currentLine.slice(0, this.state.cursorCol);
    if (this.isInSlashCommandContext(beforeCursor) && !beforeCursor.trimStart().includes(" ")) {
      this.handleSlashCommandCompletion();
    } else {
      this.forceFileAutocomplete(true);
    }
  }
  handleSlashCommandCompletion() {
    this.requestAutocomplete({ force: false, explicitTab: true });
  }
  forceFileAutocomplete(explicitTab = false) {
    this.requestAutocomplete({ force: true, explicitTab });
  }
  requestAutocomplete(options) {
    if (!this.autocompleteProvider)
      return;
    if (options.force) {
      const shouldTrigger = !this.autocompleteProvider.shouldTriggerFileCompletion || this.autocompleteProvider.shouldTriggerFileCompletion(this.state.lines, this.state.cursorLine, this.state.cursorCol);
      if (!shouldTrigger) {
        return;
      }
    }
    this.cancelAutocompleteRequest();
    const startToken = ++this.autocompleteStartToken;
    const debounceMs = this.getAutocompleteDebounceMs(options);
    if (debounceMs > 0) {
      this.autocompleteDebounceTimer = setTimeout(() => {
        this.autocompleteDebounceTimer = void 0;
        void this.startAutocompleteRequest(startToken, options);
      }, debounceMs);
      return;
    }
    void this.startAutocompleteRequest(startToken, options);
  }
  async startAutocompleteRequest(startToken, options) {
    const previousTask = this.autocompleteRequestTask;
    this.autocompleteRequestTask = (async () => {
      await previousTask;
      if (startToken !== this.autocompleteStartToken || !this.autocompleteProvider) {
        return;
      }
      const controller = new AbortController();
      this.autocompleteAbort = controller;
      const requestId = ++this.autocompleteRequestId;
      const snapshotText = this.getText();
      const snapshotLine = this.state.cursorLine;
      const snapshotCol = this.state.cursorCol;
      await this.runAutocompleteRequest(requestId, controller, snapshotText, snapshotLine, snapshotCol, options);
    })();
    await this.autocompleteRequestTask;
  }
  setAutocompleteTriggerCharacters(triggerCharacters) {
    const next = [...DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS];
    for (const character of triggerCharacters) {
      if (character.length !== 1 || character === "/" || isWhitespaceChar(character) || next.includes(character)) {
        continue;
      }
      next.push(character);
    }
    this.autocompleteTriggerCharacters = next;
    this.autocompleteTriggerPattern = buildTriggerPattern(next);
    this.autocompleteDebouncePattern = buildDebouncePattern(next);
  }
  getAutocompleteDebounceMs(options) {
    if (options.explicitTab || options.force) {
      return 0;
    }
    const currentLine = this.state.lines[this.state.cursorLine] || "";
    const textBeforeCursor = currentLine.slice(0, this.state.cursorCol);
    return this.autocompleteDebouncePattern.test(textBeforeCursor) ? ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS : 0;
  }
  async runAutocompleteRequest(requestId, controller, snapshotText, snapshotLine, snapshotCol, options) {
    if (!this.autocompleteProvider)
      return;
    const suggestions = await this.autocompleteProvider.getSuggestions(this.state.lines, this.state.cursorLine, this.state.cursorCol, { signal: controller.signal, force: options.force });
    if (!this.isAutocompleteRequestCurrent(requestId, controller, snapshotText, snapshotLine, snapshotCol)) {
      return;
    }
    this.autocompleteAbort = void 0;
    if (!suggestions || !Array.isArray(suggestions.items) || suggestions.items.length === 0) {
      this.cancelAutocomplete();
      this.tui.requestRender();
      return;
    }
    if (options.force && options.explicitTab && suggestions.items.length === 1) {
      const item = suggestions.items[0];
      this.pushUndoSnapshot();
      this.lastAction = null;
      const result = this.autocompleteProvider.applyCompletion(this.state.lines, this.state.cursorLine, this.state.cursorCol, item, suggestions.prefix);
      this.state.lines = result.lines;
      this.state.cursorLine = result.cursorLine;
      this.setCursorCol(result.cursorCol);
      if (this.onChange)
        this.onChange(this.getText());
      this.tui.requestRender();
      return;
    }
    this.applyAutocompleteSuggestions(suggestions, options.force ? "force" : "regular");
    this.tui.requestRender();
  }
  isAutocompleteRequestCurrent(requestId, controller, snapshotText, snapshotLine, snapshotCol) {
    return !controller.signal.aborted && requestId === this.autocompleteRequestId && this.getText() === snapshotText && this.state.cursorLine === snapshotLine && this.state.cursorCol === snapshotCol;
  }
  applyAutocompleteSuggestions(suggestions, state) {
    this.autocompletePrefix = suggestions.prefix;
    this.autocompleteList = this.createAutocompleteList(suggestions.prefix, suggestions.items);
    const bestMatchIndex = this.getBestAutocompleteMatchIndex(suggestions.items, suggestions.prefix);
    if (bestMatchIndex >= 0) {
      this.autocompleteList.setSelectedIndex(bestMatchIndex);
    }
    this.autocompleteState = state;
  }
  cancelAutocompleteRequest() {
    this.autocompleteStartToken += 1;
    if (this.autocompleteDebounceTimer) {
      clearTimeout(this.autocompleteDebounceTimer);
      this.autocompleteDebounceTimer = void 0;
    }
    this.autocompleteAbort?.abort();
    this.autocompleteAbort = void 0;
  }
  clearAutocompleteUi() {
    this.autocompleteState = null;
    this.autocompleteList = void 0;
    this.autocompletePrefix = "";
  }
  cancelAutocomplete() {
    this.cancelAutocompleteRequest();
    this.clearAutocompleteUi();
  }
  isShowingAutocomplete() {
    return this.autocompleteState !== null;
  }
  updateAutocomplete() {
    if (!this.autocompleteState || !this.autocompleteProvider)
      return;
    this.requestAutocomplete({ force: this.autocompleteState === "force", explicitTab: false });
  }
};

// node_modules/@earendil-works/pi-tui/dist/layout-node.js
var LAYOUT_NODE = /* @__PURE__ */ Symbol.for("@earendil-works/pi-tui/layout-node");
function getLayoutNode(component) {
  const candidate = component;
  return typeof candidate[LAYOUT_NODE] === "function" ? candidate[LAYOUT_NODE]() : void 0;
}

// node_modules/@earendil-works/pi-tui/dist/components/stack.js
function isStackEntry(child) {
  return !("render" in child);
}
function normalizeSize(value, fallback) {
  return value === void 0 || !Number.isFinite(value) ? fallback : Math.max(0, Math.floor(value));
}
var Stack = class extends Container {
  entries = [];
  gap;
  align;
  constructor(children = [], options = {}) {
    super();
    this.gap = normalizeSize(options.gap, 0);
    this.align = options.align ?? "stretch";
    for (const child of children) {
      if (isStackEntry(child))
        this.addChild(child.component, child);
      else
        this.addChild(child);
    }
  }
  addChild(component, options = {}) {
    super.addChild(component);
    this.entries.push({
      component,
      ...options.basis === void 0 ? {} : { basis: options.basis },
      ...options.grow === void 0 ? {} : { grow: normalizeSize(options.grow, 0) },
      ...options.shrink === void 0 ? {} : { shrink: normalizeSize(options.shrink, 1) },
      ...options.minSize === void 0 ? {} : { minSize: normalizeSize(options.minSize, 0) },
      ...options.maxSize === void 0 ? {} : { maxSize: normalizeSize(options.maxSize, Number.MAX_SAFE_INTEGER) },
      ...options.visible === void 0 ? {} : { visible: options.visible }
    });
  }
  removeChild(component) {
    super.removeChild(component);
    const index = this.entries.findIndex((entry) => entry.component === component);
    if (index !== -1)
      this.entries.splice(index, 1);
  }
  clear() {
    super.clear();
    this.entries.length = 0;
  }
  [LAYOUT_NODE]() {
    return {
      type: this.layoutType,
      entries: this.entries,
      gap: this.gap,
      align: this.align
    };
  }
};
function visibleStackEntries(entries, viewport) {
  return entries.filter((entry) => entry.visible?.(viewport) ?? true);
}
function clampSize(size, entry) {
  const min = Math.max(0, Math.floor(entry.minSize ?? 0));
  const max = Math.max(min, Math.floor(entry.maxSize ?? Number.MAX_SAFE_INTEGER));
  return Math.max(min, Math.min(max, Math.max(0, Math.floor(size))));
}
function distribute(sizes, entries, amount, mode) {
  let remaining = amount;
  while (remaining > 0) {
    const candidates = entries.map((entry, index) => ({ entry, index })).filter(({ entry, index }) => {
      if (mode === "grow") {
        return (entry.grow ?? 0) > 0 && sizes[index] < (entry.maxSize ?? Number.MAX_SAFE_INTEGER);
      }
      return (entry.shrink ?? 1) > 0 && sizes[index] > (entry.minSize ?? 0);
    });
    if (candidates.length === 0)
      return;
    const totalWeight = candidates.reduce((sum, { entry, index }) => {
      return sum + (mode === "grow" ? entry.grow ?? 0 : (entry.shrink ?? 1) * Math.max(1, sizes[index]));
    }, 0);
    let distributed = 0;
    for (const { entry, index } of candidates) {
      if (remaining <= 0)
        break;
      const weight = mode === "grow" ? entry.grow ?? 0 : (entry.shrink ?? 1) * Math.max(1, sizes[index]);
      const proposed = Math.max(1, Math.floor(remaining * weight / totalWeight));
      const capacity = mode === "grow" ? (entry.maxSize ?? Number.MAX_SAFE_INTEGER) - sizes[index] : sizes[index] - (entry.minSize ?? 0);
      const delta = Math.min(remaining, proposed, capacity);
      if (delta <= 0)
        continue;
      sizes[index] = sizes[index] + (mode === "grow" ? delta : -delta);
      remaining -= delta;
      distributed += delta;
    }
    if (distributed === 0)
      return;
  }
}
function allocateStackSizes(entries, intrinsicSizes, availableSize, gap) {
  const sizes = entries.map((entry, index) => clampSize(entry.basis === void 0 || entry.basis === "auto" ? intrinsicSizes[index] ?? 0 : entry.basis, entry));
  if (availableSize === void 0)
    return sizes;
  const contentSize = Math.max(0, Math.floor(availableSize) - Math.max(0, entries.length - 1) * gap);
  const total = sizes.reduce((sum, size) => sum + size, 0);
  if (total < contentSize)
    distribute(sizes, entries, contentSize - total, "grow");
  else if (total > contentSize)
    distribute(sizes, entries, total - contentSize, "shrink");
  return sizes;
}

// node_modules/@earendil-works/pi-tui/dist/components/input.js
var segmenter = getGraphemeSegmenter();
var Input = class {
  value = "";
  cursor = 0;
  // Cursor position in the value
  prompt;
  placeholder;
  placeholderStyle;
  renderedStartColumn = 0;
  onSubmit;
  onEscape;
  /** Focusable interface - set by TUI when focus changes */
  focused = false;
  // Bracketed paste mode buffering
  pasteBuffer = "";
  isInPaste = false;
  // Kill ring for Emacs-style kill/yank operations
  killRing = new KillRing();
  lastAction = null;
  // Undo support
  undoStack = new UndoStack();
  constructor(options = {}) {
    this.prompt = options.prompt ?? "> ";
    this.placeholder = options.placeholder ?? "";
    this.placeholderStyle = options.placeholderStyle ?? ((text) => text);
  }
  getValue() {
    return this.value;
  }
  setValue(value) {
    this.value = value;
    this.cursor = Math.min(this.cursor, value.length);
  }
  handleInput(data) {
    if (data.includes("\x1B[200~")) {
      this.isInPaste = true;
      this.pasteBuffer = "";
      data = data.replace("\x1B[200~", "");
    }
    if (this.isInPaste) {
      this.pasteBuffer += data;
      const endIndex = this.pasteBuffer.indexOf("\x1B[201~");
      if (endIndex !== -1) {
        const pasteContent = this.pasteBuffer.substring(0, endIndex);
        this.handlePaste(pasteContent);
        this.isInPaste = false;
        const remaining = this.pasteBuffer.substring(endIndex + 6);
        this.pasteBuffer = "";
        if (remaining) {
          this.handleInput(remaining);
        }
      }
      return;
    }
    const kb = getKeybindings();
    if (kb.matches(data, "tui.select.cancel")) {
      if (this.onEscape)
        this.onEscape();
      return;
    }
    if (kb.matches(data, "tui.editor.undo")) {
      this.undo();
      return;
    }
    if (kb.matches(data, "tui.input.submit") || data === "\n") {
      if (this.onSubmit)
        this.onSubmit(this.value);
      return;
    }
    if (kb.matches(data, "tui.editor.deleteCharBackward")) {
      this.handleBackspace();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteCharForward")) {
      this.handleForwardDelete();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteWordBackward")) {
      this.deleteWordBackwards();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteWordForward")) {
      this.deleteWordForward();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteToLineStart")) {
      this.deleteToLineStart();
      return;
    }
    if (kb.matches(data, "tui.editor.deleteToLineEnd")) {
      this.deleteToLineEnd();
      return;
    }
    if (kb.matches(data, "tui.editor.yank")) {
      this.yank();
      return;
    }
    if (kb.matches(data, "tui.editor.yankPop")) {
      this.yankPop();
      return;
    }
    if (kb.matches(data, "tui.editor.cursorLeft")) {
      this.lastAction = null;
      if (this.cursor > 0) {
        const beforeCursor = this.value.slice(0, this.cursor);
        const graphemes = [...segmenter.segment(beforeCursor)];
        const lastGrapheme = graphemes[graphemes.length - 1];
        this.cursor -= lastGrapheme ? lastGrapheme.segment.length : 1;
      }
      return;
    }
    if (kb.matches(data, "tui.editor.cursorRight")) {
      this.lastAction = null;
      if (this.cursor < this.value.length) {
        const afterCursor = this.value.slice(this.cursor);
        const graphemes = [...segmenter.segment(afterCursor)];
        const firstGrapheme = graphemes[0];
        this.cursor += firstGrapheme ? firstGrapheme.segment.length : 1;
      }
      return;
    }
    if (kb.matches(data, "tui.editor.cursorLineStart")) {
      this.lastAction = null;
      this.cursor = 0;
      return;
    }
    if (kb.matches(data, "tui.editor.cursorLineEnd")) {
      this.lastAction = null;
      this.cursor = this.value.length;
      return;
    }
    if (kb.matches(data, "tui.editor.cursorWordLeft")) {
      this.moveWordBackwards();
      return;
    }
    if (kb.matches(data, "tui.editor.cursorWordRight")) {
      this.moveWordForwards();
      return;
    }
    const kittyPrintable = decodeKittyPrintable(data);
    if (kittyPrintable !== void 0) {
      this.insertCharacter(kittyPrintable);
      return;
    }
    const hasControlChars = [...data].some((ch) => {
      const code = ch.charCodeAt(0);
      return code < 32 || code === 127 || code >= 128 && code <= 159;
    });
    if (!hasControlChars) {
      this.insertCharacter(data);
    }
  }
  handleMouse(event) {
    if (event.type !== "press" || event.button !== "left" || event.y !== 0)
      return void 0;
    const visibleColumn = Math.max(0, event.x - 2);
    const targetColumn = this.renderedStartColumn + visibleColumn;
    let currentColumn = 0;
    this.cursor = this.value.length;
    for (const grapheme of segmenter.segment(this.value)) {
      const nextColumn = currentColumn + visibleWidth(grapheme.segment);
      if (targetColumn < nextColumn) {
        this.cursor = grapheme.index;
        break;
      }
      currentColumn = nextColumn;
    }
    this.lastAction = null;
    return { handled: true, focus: true };
  }
  insertCharacter(char) {
    if (isWhitespaceChar(char) || this.lastAction !== "type-word") {
      this.pushUndo();
    }
    this.lastAction = "type-word";
    this.value = this.value.slice(0, this.cursor) + char + this.value.slice(this.cursor);
    this.cursor += char.length;
  }
  handleBackspace() {
    this.lastAction = null;
    if (this.cursor > 0) {
      this.pushUndo();
      const beforeCursor = this.value.slice(0, this.cursor);
      const graphemes = [...segmenter.segment(beforeCursor)];
      const lastGrapheme = graphemes[graphemes.length - 1];
      const graphemeLength = lastGrapheme ? lastGrapheme.segment.length : 1;
      this.value = this.value.slice(0, this.cursor - graphemeLength) + this.value.slice(this.cursor);
      this.cursor -= graphemeLength;
    }
  }
  handleForwardDelete() {
    this.lastAction = null;
    if (this.cursor < this.value.length) {
      this.pushUndo();
      const afterCursor = this.value.slice(this.cursor);
      const graphemes = [...segmenter.segment(afterCursor)];
      const firstGrapheme = graphemes[0];
      const graphemeLength = firstGrapheme ? firstGrapheme.segment.length : 1;
      this.value = this.value.slice(0, this.cursor) + this.value.slice(this.cursor + graphemeLength);
    }
  }
  deleteToLineStart() {
    if (this.cursor === 0)
      return;
    this.pushUndo();
    const deletedText = this.value.slice(0, this.cursor);
    this.killRing.push(deletedText, { prepend: true, accumulate: this.lastAction === "kill" });
    this.lastAction = "kill";
    this.value = this.value.slice(this.cursor);
    this.cursor = 0;
  }
  deleteToLineEnd() {
    if (this.cursor >= this.value.length)
      return;
    this.pushUndo();
    const deletedText = this.value.slice(this.cursor);
    this.killRing.push(deletedText, { prepend: false, accumulate: this.lastAction === "kill" });
    this.lastAction = "kill";
    this.value = this.value.slice(0, this.cursor);
  }
  deleteWordBackwards() {
    if (this.cursor === 0)
      return;
    const wasKill = this.lastAction === "kill";
    this.pushUndo();
    const oldCursor = this.cursor;
    this.moveWordBackwards();
    const deleteFrom = this.cursor;
    this.cursor = oldCursor;
    const deletedText = this.value.slice(deleteFrom, this.cursor);
    this.killRing.push(deletedText, { prepend: true, accumulate: wasKill });
    this.lastAction = "kill";
    this.value = this.value.slice(0, deleteFrom) + this.value.slice(this.cursor);
    this.cursor = deleteFrom;
  }
  deleteWordForward() {
    if (this.cursor >= this.value.length)
      return;
    const wasKill = this.lastAction === "kill";
    this.pushUndo();
    const oldCursor = this.cursor;
    this.moveWordForwards();
    const deleteTo = this.cursor;
    this.cursor = oldCursor;
    const deletedText = this.value.slice(this.cursor, deleteTo);
    this.killRing.push(deletedText, { prepend: false, accumulate: wasKill });
    this.lastAction = "kill";
    this.value = this.value.slice(0, this.cursor) + this.value.slice(deleteTo);
  }
  yank() {
    const text = this.killRing.peek();
    if (!text)
      return;
    this.pushUndo();
    this.value = this.value.slice(0, this.cursor) + text + this.value.slice(this.cursor);
    this.cursor += text.length;
    this.lastAction = "yank";
  }
  yankPop() {
    if (this.lastAction !== "yank" || this.killRing.length <= 1)
      return;
    this.pushUndo();
    const prevText = this.killRing.peek() || "";
    this.value = this.value.slice(0, this.cursor - prevText.length) + this.value.slice(this.cursor);
    this.cursor -= prevText.length;
    this.killRing.rotate();
    const text = this.killRing.peek() || "";
    this.value = this.value.slice(0, this.cursor) + text + this.value.slice(this.cursor);
    this.cursor += text.length;
    this.lastAction = "yank";
  }
  pushUndo() {
    this.undoStack.push({ value: this.value, cursor: this.cursor });
  }
  undo() {
    const snapshot = this.undoStack.pop();
    if (!snapshot)
      return;
    this.value = snapshot.value;
    this.cursor = snapshot.cursor;
    this.lastAction = null;
  }
  moveWordBackwards() {
    if (this.cursor === 0)
      return;
    this.lastAction = null;
    this.cursor = findWordBackward(this.value, this.cursor);
  }
  moveWordForwards() {
    if (this.cursor >= this.value.length)
      return;
    this.lastAction = null;
    this.cursor = findWordForward(this.value, this.cursor);
  }
  handlePaste(pastedText) {
    this.lastAction = null;
    this.pushUndo();
    const cleanText = pastedText.replace(/\r\n/g, "").replace(/\r/g, "").replace(/\n/g, "").replace(/\t/g, "    ");
    this.value = this.value.slice(0, this.cursor) + cleanText + this.value.slice(this.cursor);
    this.cursor += cleanText.length;
  }
  invalidate() {
  }
  render(width) {
    const availableWidth = width - visibleWidth(this.prompt);
    if (availableWidth <= 0) {
      return [truncateToWidth(this.prompt, width, "")];
    }
    if (this.value.length === 0 && this.placeholder) {
      const placeholder = truncateToWidth(this.placeholder, availableWidth, "");
      const graphemes2 = [...segmenter.segment(placeholder)];
      const atCursor2 = graphemes2[0]?.segment ?? " ";
      const afterCursor2 = placeholder.slice(atCursor2.length);
      const marker2 = this.focused ? CURSOR_MARKER : "";
      const cursorChar2 = `\x1B[7m${this.placeholderStyle(atCursor2)}\x1B[27m`;
      const textWithCursor2 = marker2 + cursorChar2 + this.placeholderStyle(afterCursor2);
      const padding2 = " ".repeat(Math.max(0, availableWidth - visibleWidth(textWithCursor2)));
      return [this.prompt + textWithCursor2 + padding2];
    }
    let visibleText = "";
    let cursorDisplay = this.cursor;
    this.renderedStartColumn = 0;
    const totalWidth = visibleWidth(this.value);
    if (totalWidth < availableWidth) {
      visibleText = this.value;
    } else {
      const scrollWidth = this.cursor === this.value.length ? availableWidth - 1 : availableWidth;
      const cursorCol = visibleWidth(this.value.slice(0, this.cursor));
      if (scrollWidth > 0) {
        const halfWidth = Math.floor(scrollWidth / 2);
        let startCol = 0;
        if (cursorCol < halfWidth) {
          startCol = 0;
        } else if (cursorCol > totalWidth - halfWidth) {
          startCol = Math.max(0, totalWidth - scrollWidth);
        } else {
          startCol = Math.max(0, cursorCol - halfWidth);
        }
        this.renderedStartColumn = startCol;
        visibleText = sliceByColumn(this.value, startCol, scrollWidth, true);
        const beforeCursor2 = sliceByColumn(this.value, startCol, Math.max(0, cursorCol - startCol), true);
        cursorDisplay = beforeCursor2.length;
      } else {
        visibleText = "";
        cursorDisplay = 0;
      }
    }
    const graphemes = [...segmenter.segment(visibleText.slice(cursorDisplay))];
    const cursorGrapheme = graphemes[0];
    const beforeCursor = visibleText.slice(0, cursorDisplay);
    const atCursor = cursorGrapheme?.segment ?? " ";
    const afterCursor = visibleText.slice(cursorDisplay + atCursor.length);
    const marker = this.focused ? CURSOR_MARKER : "";
    const cursorChar = `\x1B[7m${atCursor}\x1B[27m`;
    const textWithCursor = beforeCursor + marker + cursorChar + afterCursor;
    const visualLength = visibleWidth(textWithCursor);
    const padding = " ".repeat(Math.max(0, availableWidth - visualLength));
    const line = this.prompt + textWithCursor + padding;
    return [line];
  }
};

// node_modules/@earendil-works/pi-tui/dist/latex.js
var SYMBOLS = {
  alpha: "\u03B1",
  beta: "\u03B2",
  gamma: "\u03B3",
  delta: "\u03B4",
  epsilon: "\u03F5",
  varepsilon: "\u03B5",
  zeta: "\u03B6",
  eta: "\u03B7",
  theta: "\u03B8",
  vartheta: "\u03D1",
  iota: "\u03B9",
  kappa: "\u03BA",
  varkappa: "\u03F0",
  lambda: "\u03BB",
  mu: "\u03BC",
  nu: "\u03BD",
  xi: "\u03BE",
  pi: "\u03C0",
  varpi: "\u03D6",
  rho: "\u03C1",
  varrho: "\u03F1",
  sigma: "\u03C3",
  varsigma: "\u03C2",
  tau: "\u03C4",
  upsilon: "\u03C5",
  phi: "\u03D5",
  varphi: "\u03C6",
  chi: "\u03C7",
  psi: "\u03C8",
  omega: "\u03C9",
  Gamma: "\u0393",
  Delta: "\u0394",
  Theta: "\u0398",
  Lambda: "\u039B",
  Xi: "\u039E",
  Pi: "\u03A0",
  Sigma: "\u03A3",
  Upsilon: "\u03A5",
  Phi: "\u03A6",
  Psi: "\u03A8",
  Omega: "\u03A9",
  pm: "\xB1",
  mp: "\u2213",
  times: "\xD7",
  div: "\xF7",
  cdot: "\xB7",
  ast: "\u2217",
  star: "\u22C6",
  circ: "\u2218",
  bullet: "\u2022",
  oplus: "\u2295",
  ominus: "\u2296",
  otimes: "\u2297",
  oslash: "\u2298",
  odot: "\u2299",
  bigcirc: "\u25CB",
  dagger: "\u2020",
  ddagger: "\u2021",
  amalg: "\u2A3F",
  uplus: "\u228E",
  sqcap: "\u2293",
  sqcup: "\u2294",
  bowtie: "\u22C8",
  Join: "\u22C8",
  ltimes: "\u22C9",
  rtimes: "\u22CA",
  leftouterjoin: "\u27D5",
  rightouterjoin: "\u27D6",
  fullouterjoin: "\u27D7",
  triangleleft: "\u25C1",
  triangleright: "\u25B7",
  wr: "\u2240",
  cap: "\u2229",
  cup: "\u222A",
  bigcap: "\u22C2",
  bigcup: "\u22C3",
  bigwedge: "\u22C0",
  bigvee: "\u22C1",
  bigsqcup: "\u2A06",
  biguplus: "\u2A04",
  bigoplus: "\u2A01",
  bigotimes: "\u2A02",
  bigodot: "\u2A00",
  setminus: "\u2216",
  in: "\u2208",
  notin: "\u2209",
  ni: "\u220B",
  subset: "\u2282",
  supset: "\u2283",
  subseteq: "\u2286",
  supseteq: "\u2287",
  sqsubset: "\u228F",
  sqsupset: "\u2290",
  sqsubseteq: "\u2291",
  sqsupseteq: "\u2292",
  prec: "\u227A",
  preceq: "\u227C",
  succ: "\u227B",
  succeq: "\u227D",
  ll: "\u226A",
  gg: "\u226B",
  le: "\u2264",
  leq: "\u2264",
  leqslant: "\u2264",
  ge: "\u2265",
  geq: "\u2265",
  geqslant: "\u2265",
  ne: "\u2260",
  neq: "\u2260",
  equiv: "\u2261",
  approx: "\u2248",
  sim: "\u223C",
  simeq: "\u2243",
  cong: "\u2245",
  asymp: "\u224D",
  doteq: "\u2250",
  propto: "\u221D",
  parallel: "\u2225",
  perp: "\u22A5",
  mid: "\u2223",
  vdash: "\u22A2",
  dashv: "\u22A3",
  models: "\u22A8",
  Vdash: "\u22A9",
  Vvdash: "\u22AA",
  nvdash: "\u22AC",
  nvDash: "\u22AD",
  forall: "\u2200",
  exists: "\u2203",
  nexists: "\u2204",
  neg: "\xAC",
  land: "\u2227",
  wedge: "\u2227",
  lor: "\u2228",
  vee: "\u2228",
  to: "\u2192",
  rightarrow: "\u2192",
  longrightarrow: "\u2192",
  leftarrow: "\u2190",
  longleftarrow: "\u2190",
  gets: "\u2190",
  leftrightarrow: "\u2194",
  longleftrightarrow: "\u2194",
  hookleftarrow: "\u21A9",
  hookrightarrow: "\u21AA",
  twoheadleftarrow: "\u219E",
  twoheadrightarrow: "\u21A0",
  leftharpoonup: "\u21BC",
  leftharpoondown: "\u21BD",
  rightharpoonup: "\u21C0",
  rightharpoondown: "\u21C1",
  rightleftharpoons: "\u21CC",
  leftrightharpoons: "\u21CB",
  nearrow: "\u2197",
  searrow: "\u2198",
  swarrow: "\u2199",
  nwarrow: "\u2196",
  rightsquigarrow: "\u21DD",
  leadsto: "\u21DD",
  Rightarrow: "\u21D2",
  Longrightarrow: "\u21D2",
  Leftarrow: "\u21D0",
  Longleftarrow: "\u21D0",
  Leftrightarrow: "\u21D4",
  Longleftrightarrow: "\u21D4",
  implies: "\u21D2",
  iff: "\u21D4",
  mapsto: "\u21A6",
  longmapsto: "\u21A6",
  uparrow: "\u2191",
  downarrow: "\u2193",
  partial: "\u2202",
  nabla: "\u2207",
  int: "\u222B",
  iint: "\u222C",
  iiint: "\u222D",
  oint: "\u222E",
  sum: "\u2211",
  prod: "\u220F",
  coprod: "\u2210",
  infty: "\u221E",
  emptyset: "\u2205",
  varnothing: "\u2205",
  angle: "\u2220",
  therefore: "\u2234",
  because: "\u2235",
  aleph: "\u2135",
  beth: "\u2136",
  gimel: "\u2137",
  daleth: "\u2138",
  top: "\u22A4",
  bot: "\u22A5",
  triangle: "\u25B3",
  square: "\u25A1",
  lozenge: "\u25CA",
  checkmark: "\u2713",
  complement: "\u2201",
  wp: "\u2118",
  prime: "\u2032",
  ldots: "\u2026",
  dots: "\u2026",
  cdots: "\u22EF",
  vdots: "\u22EE",
  ddots: "\u22F1",
  ell: "\u2113",
  hbar: "\u210F",
  Im: "\u2111",
  Re: "\u211C",
  langle: "\u27E8",
  rangle: "\u27E9",
  vert: "|",
  lvert: "|",
  rvert: "|",
  Vert: "\u2016",
  lVert: "\u2016",
  rVert: "\u2016",
  lbrace: "{",
  rbrace: "}",
  backslash: "\\",
  lfloor: "\u230A",
  rfloor: "\u230B",
  lceil: "\u2308",
  rceil: "\u2309",
  colon: ":"
};
var NAMED_OPERATORS = /* @__PURE__ */ new Set([
  "arccos",
  "arcsin",
  "arctan",
  "arg",
  "cos",
  "cosh",
  "cot",
  "coth",
  "csc",
  "deg",
  "det",
  "dim",
  "exp",
  "gcd",
  "hom",
  "inf",
  "ker",
  "lg",
  "lim",
  "liminf",
  "limsup",
  "ln",
  "log",
  "max",
  "min",
  "Pr",
  "sec",
  "sin",
  "sinh",
  "sup",
  "tan",
  "tanh"
]);
var LIMIT_OPERATORS = /* @__PURE__ */ new Set([
  "argmax",
  "argmin",
  "inf",
  "injlim",
  "lim",
  "liminf",
  "limsup",
  "max",
  "min",
  "projlim",
  "sup"
]);
var DISPLAY_LIMIT_SYMBOLS = /* @__PURE__ */ new Set([
  "bigcap",
  "bigcup",
  "bigodot",
  "bigoplus",
  "bigotimes",
  "bigsqcup",
  "biguplus",
  "bigvee",
  "bigwedge",
  "coprod",
  "int",
  "iint",
  "iiint",
  "oint",
  "prod",
  "sum"
]);
var RELATION_COMMANDS = /* @__PURE__ */ new Set([
  "Leftarrow",
  "Leftrightarrow",
  "Longleftarrow",
  "Longleftrightarrow",
  "Longrightarrow",
  "Rightarrow",
  "Join",
  "Vdash",
  "Vvdash",
  "approx",
  "asymp",
  "bowtie",
  "cong",
  "dashv",
  "fullouterjoin",
  "doteq",
  "downarrow",
  "equiv",
  "ge",
  "geq",
  "geqslant",
  "gets",
  "gg",
  "hookleftarrow",
  "hookrightarrow",
  "iff",
  "implies",
  "in",
  "leadsto",
  "le",
  "leftarrow",
  "leftharpoondown",
  "leftharpoonup",
  "leftrightarrow",
  "leftrightharpoons",
  "leftouterjoin",
  "leq",
  "leqslant",
  "ll",
  "longleftarrow",
  "longleftrightarrow",
  "longmapsto",
  "longrightarrow",
  "ltimes",
  "mapsto",
  "mid",
  "models",
  "ne",
  "nearrow",
  "neq",
  "ni",
  "notin",
  "nvdash",
  "nvDash",
  "nwarrow",
  "parallel",
  "perp",
  "prec",
  "preceq",
  "propto",
  "rightharpoondown",
  "rightharpoonup",
  "rightleftharpoons",
  "rightouterjoin",
  "rightarrow",
  "rightsquigarrow",
  "rtimes",
  "searrow",
  "sim",
  "simeq",
  "sqsubset",
  "sqsubseteq",
  "sqsupset",
  "sqsupseteq",
  "subset",
  "subseteq",
  "succ",
  "succeq",
  "supset",
  "supseteq",
  "swarrow",
  "to",
  "triangleleft",
  "triangleright",
  "twoheadleftarrow",
  "twoheadrightarrow",
  "uparrow",
  "vdash"
]);
var NEGATED_SYMBOLS = {
  "<": "\u226E",
  ">": "\u226F",
  "=": "\u2260",
  "\u2208": "\u2209",
  "\u220B": "\u220C",
  "\u2223": "\u2224",
  "\u2225": "\u2226",
  "\u223C": "\u2241",
  "\u2243": "\u2244",
  "\u2245": "\u2247",
  "\u2248": "\u2249",
  "\u2261": "\u2262",
  "\u2264": "\u2270",
  "\u2265": "\u2271",
  "\u227A": "\u2280",
  "\u227B": "\u2281",
  "\u2282": "\u2284",
  "\u2283": "\u2285",
  "\u2286": "\u2288",
  "\u2287": "\u2289",
  "\u22A2": "\u22AC",
  "\u22A8": "\u22AD",
  "\u2194": "\u21AE",
  "\u2190": "\u219A",
  "\u2192": "\u219B",
  "\u21D2": "\u21CF",
  "\u21D0": "\u21CD",
  "\u21D4": "\u21CE",
  "\u227C": "\u22E0",
  "\u227D": "\u22E1"
};
var BLACKBOARD = {
  C: "\u2102",
  H: "\u210D",
  N: "\u2115",
  P: "\u2119",
  Q: "\u211A",
  R: "\u211D",
  Z: "\u2124"
};
var SUPERSCRIPTS = {
  "0": "\u2070",
  "1": "\xB9",
  "2": "\xB2",
  "3": "\xB3",
  "4": "\u2074",
  "5": "\u2075",
  "6": "\u2076",
  "7": "\u2077",
  "8": "\u2078",
  "9": "\u2079",
  "+": "\u207A",
  "-": "\u207B",
  "=": "\u207C",
  "(": "\u207D",
  ")": "\u207E",
  a: "\u1D43",
  b: "\u1D47",
  c: "\u1D9C",
  d: "\u1D48",
  e: "\u1D49",
  f: "\u1DA0",
  g: "\u1D4D",
  h: "\u02B0",
  i: "\u2071",
  j: "\u02B2",
  k: "\u1D4F",
  l: "\u02E1",
  m: "\u1D50",
  n: "\u207F",
  o: "\u1D52",
  p: "\u1D56",
  r: "\u02B3",
  s: "\u02E2",
  t: "\u1D57",
  u: "\u1D58",
  v: "\u1D5B",
  w: "\u02B7",
  x: "\u02E3",
  y: "\u02B8",
  z: "\u1DBB"
};
var SUBSCRIPTS = {
  "0": "\u2080",
  "1": "\u2081",
  "2": "\u2082",
  "3": "\u2083",
  "4": "\u2084",
  "5": "\u2085",
  "6": "\u2086",
  "7": "\u2087",
  "8": "\u2088",
  "9": "\u2089",
  "+": "\u208A",
  "-": "\u208B",
  "=": "\u208C",
  "(": "\u208D",
  ")": "\u208E",
  a: "\u2090",
  e: "\u2091",
  h: "\u2095",
  i: "\u1D62",
  j: "\u2C7C",
  k: "\u2096",
  l: "\u2097",
  m: "\u2098",
  n: "\u2099",
  o: "\u2092",
  p: "\u209A",
  r: "\u1D63",
  s: "\u209B",
  t: "\u209C",
  u: "\u1D64",
  v: "\u1D65",
  x: "\u2093"
};
var SPACING_COMMANDS = /* @__PURE__ */ new Set([
  ",",
  ":",
  ";",
  " ",
  ">",
  "enspace",
  "enskip",
  "medspace",
  "quad",
  "qquad",
  "thickspace",
  "thinspace"
]);
var NEGATIVE_SPACING_COMMANDS = /* @__PURE__ */ new Set(["!", "negmedspace", "negthickspace", "negthinspace"]);
var NEGATIVE_SPACE = "\0";
var IGNORED_COMMANDS = /* @__PURE__ */ new Set([
  "displaystyle",
  "limits",
  "nolimits",
  "scriptstyle",
  "scriptscriptstyle",
  "textstyle"
]);
var SIZE_COMMANDS = /* @__PURE__ */ new Set([
  "big",
  "Big",
  "bigg",
  "Bigg",
  "bigl",
  "Bigl",
  "biggl",
  "Biggl",
  "bigr",
  "Bigr",
  "biggr",
  "Biggr"
]);
var PLAIN_WRAPPERS = /* @__PURE__ */ new Set([
  "emph",
  "mathcal",
  "mathbf",
  "mathfrak",
  "mathit",
  "mathrm",
  "mathnormal",
  "mathscr",
  "mathsf",
  "mathtt",
  "mathup",
  "mbox",
  "overbrace",
  "pmb",
  "smash",
  "substack",
  "text",
  "textbf",
  "textit",
  "textmd",
  "textnormal",
  "textrm",
  "textsc",
  "textsf",
  "textsl",
  "texttt",
  "textup",
  "underbrace",
  "bm",
  "boldsymbol"
]);
var ACCENTS = {
  acute: "\u0301",
  bar: "\u0305",
  breve: "\u0306",
  check: "\u030C",
  ddot: "\u0308",
  dot: "\u0307",
  grave: "\u0300",
  hat: "\u0302",
  mathring: "\u030A",
  overleftarrow: "\u20D6",
  overleftrightarrow: "\u20E1",
  overline: "\u0305",
  overrightarrow: "\u20D7",
  tilde: "\u0303",
  underline: "\u0332",
  vec: "\u20D7",
  widehat: "\u0302",
  widetilde: "\u0303"
};
function replaceCharacters(value, replacements) {
  let result = "";
  for (const character of value) {
    const replacement = replacements[character];
    if (replacement === void 0) {
      return void 0;
    }
    result += replacement;
  }
  return result;
}
function formatScript(value, kind) {
  value = value.trim();
  const replacements = kind === "sub" ? SUBSCRIPTS : SUPERSCRIPTS;
  const unicode = replaceCharacters(value.replace(/\s*([=+-])\s*/g, "$1"), replacements);
  if (unicode !== void 0) {
    return unicode;
  }
  const prefix = kind === "sub" ? "_" : "^";
  if (Array.from(value).length === 1 || kind === "sub" && /^[A-Za-z]+$/.test(value)) {
    return `${prefix}${value}`;
  }
  return `${prefix}(${value})`;
}
function formatFraction(numerator, denominator) {
  numerator = numerator.trim();
  denominator = denominator.trim();
  const simpleNumerator = /^[\p{L}\p{N}.]+$/u.test(numerator);
  const simpleDenominator = /^[\p{N}.]+$/u.test(denominator) || Array.from(denominator).length === 1;
  return `${simpleNumerator ? numerator : `(${numerator})`}/${simpleDenominator ? denominator : `(${denominator})`}`;
}
function formatRoot(value, symbol = "\u221A") {
  value = value.trim();
  return /^[\p{L}\p{N}.]+$/u.test(value) ? `${symbol}${value}` : `${symbol}(${value})`;
}
var NAMED_OPERATOR_START = "\u{F0004}";
var NAMED_OPERATOR_END = "\u{F0005}";
var NAMED_OPERATOR_LEFT_SPACING_PATTERN = /(?<=[\p{L}\p{N})\]}\u{f0001}])\u{f0004}/gu;
var NAMED_OPERATOR_RIGHT_SPACING_PATTERN = /\u{f0005}(?=[\p{L}\p{N}√\u{f0000}])/gu;
function normalizeOutput(value) {
  return value.replace(NAMED_OPERATOR_LEFT_SPACING_PATTERN, " ").replaceAll(NAMED_OPERATOR_START, "").replace(NAMED_OPERATOR_RIGHT_SPACING_PATTERN, " ").replaceAll(NAMED_OPERATOR_END, "").split("\n").map((line) => line.replace(/[ \t]+/g, " ").trim()).filter((line, index, lines) => line.length > 0 || index > 0 && index < lines.length - 1).join("\n").trim();
}
var LAYOUT_MARKER_START = "\u{F0000}";
var LAYOUT_MARKER_END = "\u{F0001}";
var LAYOUT_MARKER_PATTERN = /\u{f0000}(\d+)\u{f0001}/gu;
var TRAILING_LAYOUT_MARKER_PATTERN = /\u{f0000}(\d+)\u{f0001}$/u;
var PROTECTED_SPACE = "\u{F0002}";
function padLayoutLine(line, width, centered = false) {
  const padding = Math.max(0, width - visibleWidth(line));
  const left = centered ? Math.floor(padding / 2) : 0;
  return `${" ".repeat(left)}${line}${" ".repeat(padding - left)}`;
}
function joinLayouts(layouts) {
  if (layouts.length === 0) {
    return { lines: [""], width: 0, baseline: 0 };
  }
  const baseline = Math.max(...layouts.map((layout) => layout.baseline));
  const below = Math.max(...layouts.map((layout) => layout.lines.length - layout.baseline - 1));
  const lines = [];
  for (let row = 0; row <= baseline + below; row++) {
    let line = "";
    for (const layout of layouts) {
      const sourceRow = row - baseline + layout.baseline;
      line += sourceRow >= 0 && sourceRow < layout.lines.length ? padLayoutLine(layout.lines[sourceRow] ?? "", layout.width) : " ".repeat(layout.width);
    }
    lines.push(line.trimEnd());
  }
  return {
    lines,
    width: layouts.reduce((width, layout) => width + layout.width, 0),
    baseline
  };
}
function renderLayout(source, nodes) {
  const renderedLines = [];
  let firstBaseline = 0;
  for (const sourceLine of source.split("\n")) {
    const layouts = [];
    let position = 0;
    let previousNode;
    for (const match of sourceLine.matchAll(LAYOUT_MARKER_PATTERN)) {
      const index = match.index;
      const node = nodes[Number(match[1])];
      if (!node) {
        continue;
      }
      if (index > position) {
        const sliced = sourceLine.slice(position, index);
        const trimmed = (previousNode ? sliced.trimStart() : sliced).trimEnd();
        const preserveLeadingSpace = previousNode?.type === "matrix" && /^\s/.test(sliced);
        const preserveTrailingSpace = node.type === "matrix" && /\s$/.test(sliced);
        const text = trimmed ? `${preserveLeadingSpace ? " " : ""}${trimmed}${preserveTrailingSpace ? " " : ""}` : preserveLeadingSpace || preserveTrailingSpace ? " " : "";
        layouts.push({ lines: [text], width: visibleWidth(text), baseline: 0 });
      }
      if (node.type === "fraction") {
        const numerator = renderLayout(node.numerator, nodes);
        const denominator = renderLayout(node.denominator, nodes);
        const contentWidth = Math.max(numerator.width, denominator.width, 1);
        const width = contentWidth + 2;
        layouts.push({
          lines: [
            ...numerator.lines.map((line) => padLayoutLine(line, width, true)),
            ` ${"\u2500".repeat(contentWidth)} `,
            ...denominator.lines.map((line) => padLayoutLine(line, width, true))
          ],
          width,
          baseline: numerator.lines.length
        });
      } else if (node.type === "operator") {
        const contentWidth = Math.max(visibleWidth(node.operator), node.lower === void 0 ? 0 : visibleWidth(node.lower), node.upper === void 0 ? 0 : visibleWidth(node.upper));
        const lines = [];
        if (node.upper !== void 0) {
          lines.push(`${padLayoutLine(node.upper, contentWidth, true)} `);
        }
        lines.push(`${padLayoutLine(node.operator, contentWidth, true)} `);
        if (node.lower !== void 0) {
          lines.push(`${padLayoutLine(node.lower, contentWidth, true)} `);
        }
        layouts.push({
          lines,
          width: contentWidth + 1,
          baseline: node.upper === void 0 ? 0 : 1
        });
      } else {
        const width = Math.max(0, ...node.lines.map((line) => visibleWidth(line)));
        layouts.push({
          lines: node.lines.map((line) => padLayoutLine(line, width)),
          width,
          baseline: node.baseline
        });
      }
      position = index + match[0].length;
      previousNode = node;
    }
    if (position < sourceLine.length) {
      const sliced = sourceLine.slice(position);
      const trimmed = previousNode ? sliced.trimStart() : sliced;
      const text = previousNode?.type === "matrix" && /^\s/.test(sliced) ? ` ${trimmed}` : trimmed;
      layouts.push({ lines: [text], width: visibleWidth(text), baseline: 0 });
    }
    const lineLayout = joinLayouts(layouts);
    if (renderedLines.length === 0) {
      firstBaseline = lineLayout.baseline;
    }
    renderedLines.push(...lineLayout.lines);
  }
  return {
    lines: renderedLines,
    width: Math.max(0, ...renderedLines.map((line) => visibleWidth(line))),
    baseline: firstBaseline
  };
}
var LatexParser = class _LatexParser {
  source;
  layoutNodes;
  display;
  position = 0;
  supported = true;
  stackFractions = true;
  constructor(source, layoutNodes, display) {
    this.source = source;
    this.layoutNodes = layoutNodes;
    this.display = display;
  }
  render() {
    const rendered = this.parseSequence();
    if (!this.supported || this.position !== this.source.length) {
      return void 0;
    }
    return normalizeOutput(rendered);
  }
  parseSequence(endCharacter) {
    let result = "";
    while (this.position < this.source.length) {
      const character = this.source[this.position];
      if (endCharacter && character === endCharacter) {
        this.position++;
        return result;
      }
      if (character === "}") {
        this.supported = false;
        return result;
      }
      if (character === "{") {
        this.position++;
        result += this.parseSequence("}");
        continue;
      }
      if (character === "\\") {
        const command = this.parseCommand();
        if (command === NEGATIVE_SPACE) {
          result = result.trimEnd();
          if (result.endsWith(NAMED_OPERATOR_END)) {
            result = result.slice(0, -NAMED_OPERATOR_END.length);
          }
        } else {
          result += command;
        }
        continue;
      }
      if (character === "^" || character === "_") {
        this.position++;
        result = result.trimEnd();
        const script = formatScript(this.parseRequiredArgument(false), character === "_" ? "sub" : "sup");
        if (result.endsWith(NAMED_OPERATOR_END)) {
          result = `${result.slice(0, -NAMED_OPERATOR_END.length)}${script}${NAMED_OPERATOR_END}`;
        } else {
          result += script;
        }
        continue;
      }
      if (/\s/.test(character)) {
        result += this.parseWhitespace();
        continue;
      }
      if (character === "=" || character === "<" || character === ">") {
        result = `${result.trimEnd()} ${character} `;
        this.position++;
        continue;
      }
      if (character === "&") {
        this.position++;
        continue;
      }
      if (character === "~") {
        this.position++;
        result += " ";
        continue;
      }
      if (character === ".") {
        const marker = TRAILING_LAYOUT_MARKER_PATTERN.exec(result);
        const node = marker ? this.layoutNodes[Number(marker[1])] : void 0;
        if (node?.type === "matrix") {
          const lastLine = node.lines.length - 1;
          node.lines[lastLine] = `${node.lines[lastLine] ?? ""}${character}`;
          this.position++;
          continue;
        }
      }
      result += character;
      this.position++;
    }
    if (endCharacter) {
      this.supported = false;
    }
    return result;
  }
  parseWhitespace() {
    while (this.position < this.source.length && /\s/.test(this.source[this.position] ?? "")) {
      this.position++;
    }
    return " ";
  }
  parseCommand() {
    this.position++;
    if (this.position >= this.source.length) {
      this.supported = false;
      return "";
    }
    let command = "";
    const first = this.source[this.position] ?? "";
    if (first === "\n" || first === "\r") {
      this.position++;
      if (first === "\r" && this.source[this.position] === "\n") {
        this.position++;
      }
      return " ";
    }
    if (/[A-Za-z]/.test(first)) {
      const start = this.position;
      while (this.position < this.source.length && /[A-Za-z]/.test(this.source[this.position] ?? "")) {
        this.position++;
      }
      command = this.source.slice(start, this.position);
    } else {
      command = first;
      this.position++;
    }
    if (command === "\\") {
      return "\n";
    }
    if (SPACING_COMMANDS.has(command)) {
      return " ";
    }
    if (NEGATIVE_SPACING_COMMANDS.has(command)) {
      return NEGATIVE_SPACE;
    }
    if (IGNORED_COMMANDS.has(command)) {
      return "";
    }
    if (command === "{" || command === "}" || command === "$" || command === "%" || command === "#" || command === "_" || command === "&") {
      return command;
    }
    if (command === "|") {
      return "\u2016";
    }
    if (command === "not") {
      const value = this.parseRequiredArgument(false).trim();
      const negated = NEGATED_SYMBOLS[value];
      if (negated !== void 0) {
        return ` ${negated} `;
      }
      const characters = Array.from(value);
      if (characters.length === 0) {
        this.supported = false;
        return "";
      }
      return ` ${characters[0]}\u0338${characters.slice(1).join("")} `;
    }
    if (LIMIT_OPERATORS.has(command)) {
      return this.parseOperator(command, "bracket", true, true);
    }
    const symbol = SYMBOLS[command];
    if (symbol !== void 0) {
      if (DISPLAY_LIMIT_SYMBOLS.has(command)) {
        return this.parseOperator(symbol, "script", true);
      }
      return command === "cdot" || command === "times" || RELATION_COMMANDS.has(command) ? ` ${symbol} ` : symbol;
    }
    if (NAMED_OPERATORS.has(command)) {
      return `${NAMED_OPERATOR_START}${command}${NAMED_OPERATOR_END}`;
    }
    if (SIZE_COMMANDS.has(command)) {
      return "";
    }
    if (command === "left" || command === "middle" || command === "right") {
      if (this.source[this.position] === ".") {
        this.position++;
      }
      return "";
    }
    if (command === "frac" || command === "dfrac" || command === "tfrac") {
      const shouldStack = this.display && this.stackFractions && command !== "tfrac";
      const numerator = this.parseRequiredArgument(!shouldStack);
      const denominator = this.parseRequiredArgument(!shouldStack);
      if (shouldStack) {
        const index = this.layoutNodes.push({
          type: "fraction",
          numerator: normalizeOutput(numerator),
          denominator: normalizeOutput(denominator)
        }) - 1;
        return `${LAYOUT_MARKER_START}${index}${LAYOUT_MARKER_END}`;
      }
      return formatFraction(numerator, denominator);
    }
    if (command === "sqrt") {
      const degree = this.parseOptionalArgument()?.trim();
      const value = this.parseRequiredArgument();
      if (degree === void 0 || degree === "2") {
        return formatRoot(value);
      }
      if (degree === "3") {
        return formatRoot(value, "\u221B");
      }
      if (degree === "4") {
        return formatRoot(value, "\u221C");
      }
      return `${formatScript(degree, "sup")}${formatRoot(value)}`;
    }
    if (command === "boxed" || command === "fbox") {
      return `[${this.parseRequiredArgument().trim()}]`;
    }
    if (command === "binom" || command === "dbinom" || command === "tbinom") {
      return `(${this.parseRequiredArgument()} choose ${this.parseRequiredArgument()})`;
    }
    const accent = ACCENTS[command];
    if (accent !== void 0) {
      const value = this.parseRequiredArgument();
      return Array.from(value).length === 1 ? `${value}${accent}` : `${command}(${value})`;
    }
    if (command === "mathbb") {
      const value = this.parseRequiredArgument();
      return Array.from(value, (character) => BLACKBOARD[character] ?? character).join("");
    }
    if (command === "operatorname") {
      const starred = this.source[this.position] === "*";
      if (starred) {
        this.position++;
      }
      const operator = normalizeOutput(this.parseRequiredArgument()).trim();
      return this.parseOperator(operator, "bracket", starred, true);
    }
    if (command === "mod" || command === "bmod") {
      return " mod ";
    }
    if (command === "pmod" || command === "pod") {
      const value = this.parseRequiredArgument().trim();
      return command === "pmod" ? ` (mod ${value})` : ` (${value})`;
    }
    if (command === "overset" || command === "stackrel") {
      const upper = this.parseRequiredArgument();
      const value = this.parseRequiredArgument().trim();
      return `${value}${formatScript(upper, "sup")}`;
    }
    if (command === "underset") {
      const lower = this.parseRequiredArgument();
      const value = this.parseRequiredArgument().trim();
      return `${value}${formatScript(lower, "sub")}`;
    }
    if (PLAIN_WRAPPERS.has(command)) {
      const value = this.parseRequiredArgument();
      return command.startsWith("text") || command === "mbox" ? value : value.trim();
    }
    if (command === "begin") {
      return this.parseEnvironment();
    }
    if (command === "end") {
      this.supported = false;
      return "";
    }
    this.supported = false;
    return `\\${command}`;
  }
  parseOperator(operator, inlineLowerStyle, displayLimits, spaced = false) {
    let useDisplayLimits = displayLimits;
    let modifierPosition = this.position;
    while (modifierPosition < this.source.length && /[ \t]/.test(this.source[modifierPosition] ?? "")) {
      modifierPosition++;
    }
    const modifier = /^\\(limits|nolimits)(?![A-Za-z])/.exec(this.source.slice(modifierPosition));
    if (modifier) {
      useDisplayLimits = modifier[1] === "limits";
      this.position = modifierPosition + modifier[0].length;
    }
    let lower;
    let upper;
    while (true) {
      let scriptPosition = this.position;
      while (scriptPosition < this.source.length && /[ \t]/.test(this.source[scriptPosition] ?? "")) {
        scriptPosition++;
      }
      const kind = this.source[scriptPosition];
      if (kind !== "_" && kind !== "^") {
        break;
      }
      this.position = scriptPosition + 1;
      const value = normalizeOutput(this.parseRequiredArgument(false)).replaceAll(" ", "");
      if (kind === "_") {
        if (lower !== void 0) {
          this.supported = false;
        }
        lower = value;
      } else {
        if (upper !== void 0) {
          this.supported = false;
        }
        upper = value;
      }
    }
    if (this.display && useDisplayLimits && (lower !== void 0 || upper !== void 0)) {
      const index = this.layoutNodes.push({ type: "operator", operator, lower, upper }) - 1;
      return `${LAYOUT_MARKER_START}${index}${LAYOUT_MARKER_END}`;
    }
    let rendered = operator;
    if (lower !== void 0) {
      rendered += inlineLowerStyle === "bracket" ? `[${lower}]` : formatScript(lower, "sub");
    }
    if (upper !== void 0) {
      rendered += formatScript(upper, "sup");
    }
    return spaced ? ` ${rendered} ` : rendered;
  }
  parseRequiredArgument(stackFractions = true) {
    const previousStackFractions = this.stackFractions;
    this.stackFractions = previousStackFractions && stackFractions;
    const value = this.parseRequiredArgumentValue();
    this.stackFractions = previousStackFractions;
    return value;
  }
  parseRequiredArgumentValue() {
    while (this.position < this.source.length && /\s/.test(this.source[this.position] ?? "")) {
      this.position++;
    }
    if (this.position >= this.source.length) {
      this.supported = false;
      return "";
    }
    if (this.source[this.position] === "{") {
      this.position++;
      return this.parseSequence("}");
    }
    if (this.source[this.position] === "\\") {
      return this.parseCommand();
    }
    const value = this.source[this.position] ?? "";
    this.position++;
    return value;
  }
  parseOptionalArgument() {
    while (this.position < this.source.length && /[ \t]/.test(this.source[this.position] ?? "")) {
      this.position++;
    }
    if (this.source[this.position] !== "[") {
      return void 0;
    }
    const end = this.source.indexOf("]", this.position + 1);
    if (end < 0) {
      this.supported = false;
      return void 0;
    }
    const value = this.source.slice(this.position + 1, end);
    this.position = end + 1;
    return this.renderNested(value);
  }
  readRawGroup() {
    while (this.position < this.source.length && /[ \t]/.test(this.source[this.position] ?? "")) {
      this.position++;
    }
    if (this.source[this.position] !== "{") {
      this.supported = false;
      return void 0;
    }
    const start = ++this.position;
    let depth = 1;
    while (this.position < this.source.length) {
      const character = this.source[this.position];
      if (character === "\\") {
        this.position += 2;
        continue;
      }
      if (character === "{")
        depth++;
      if (character === "}")
        depth--;
      if (depth === 0) {
        const value = this.source.slice(start, this.position);
        this.position++;
        return value;
      }
      this.position++;
    }
    this.supported = false;
    return void 0;
  }
  splitEnvironmentRows(body) {
    return body.split(/\\\\(?:\[[^\]\n]*\])?/);
  }
  parseEnvironment() {
    const environment = this.readRawGroup();
    if (!environment) {
      return "";
    }
    const endMarker = `\\end{${environment}}`;
    const end = this.source.indexOf(endMarker, this.position);
    if (end < 0) {
      this.supported = false;
      return "";
    }
    const body = this.source.slice(this.position, end);
    this.position = end + endMarker.length;
    if (environment === "equation" || environment === "equation*" || environment === "displaymath") {
      return this.renderNested(body).trim();
    }
    if (environment === "aligned" || environment === "align" || environment === "align*" || environment === "alignedat" || environment === "alignat" || environment === "alignat*" || environment === "gather" || environment === "gathered" || environment === "multline" || environment === "multline*" || environment === "split") {
      const alignedAt = ["alignedat", "alignat", "alignat*"].includes(environment);
      const alignedBody = alignedAt ? body.replace(/^\s*\{[^}]*\}/, "") : body;
      return this.splitEnvironmentRows(alignedBody).map((row) => {
        const cells = row.split("&");
        const source = alignedAt ? Array.from({ length: Math.ceil(cells.length / 2) }, (_2, index) => cells.slice(index * 2, index * 2 + 2).join("")).join(" ") : cells.join("");
        return this.renderNested(source).trim();
      }).filter(Boolean).join("\n");
    }
    if (environment === "cases" || environment === "cases*") {
      const rows = this.splitEnvironmentRows(body).map((row) => row.split("&").map((cell) => this.renderNested(cell, false).trim())).filter((row) => row.some(Boolean));
      return rows.map((row, index) => {
        const value = (row[0] ?? "").replace(/,\s*$/, "");
        const condition = row[1] ?? "";
        const delimiter = index === 0 ? "\u23A7" : index === rows.length - 1 ? "\u23A9" : "\u23A8";
        const conditionPrefix = /^(?:if|when|for|otherwise)\b/i.test(condition) ? " " : " if ";
        return `${delimiter} ${value}${condition ? `${conditionPrefix}${condition}` : ""}`;
      }).join("\n");
    }
    if (["array", "matrix", "smallmatrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix"].includes(environment)) {
      const matrixBody = environment === "array" ? body.replace(/^\s*\{[^}]*\}/, "") : body;
      return this.renderMatrix(environment, matrixBody);
    }
    this.supported = false;
    return body;
  }
  renderMatrix(environment, body) {
    const matrix = this.splitEnvironmentRows(body).map((row) => row.split("&").map((cell) => this.renderNested(cell, false).trim())).filter((row) => row.some(Boolean));
    const columnCount = Math.max(0, ...matrix.map((row) => row.length));
    const columnWidths = Array.from({ length: columnCount }, (_2, column) => Math.max(0, ...matrix.map((row) => visibleWidth(row[column] ?? ""))));
    const rows = matrix.map((row) => Array.from({ length: columnCount }, (_2, column) => {
      const cell = row[column] ?? "";
      return `${cell}${PROTECTED_SPACE.repeat(Math.max(0, (columnWidths[column] ?? 0) - visibleWidth(cell)))}`;
    }).join(" \u2502 "));
    let lines;
    if (environment === "array" || environment === "matrix" || environment === "smallmatrix") {
      lines = rows;
    } else {
      const delimiters = {
        pmatrix: ["\u239B", "\u239E", "\u239C", "\u239F", "\u239D", "\u23A0"],
        bmatrix: ["\u23A1", "\u23A4", "\u23A2", "\u23A5", "\u23A3", "\u23A6"],
        Bmatrix: ["\u23A7", "\u23AB", "\u23A8", "\u23AC", "\u23A9", "\u23AD"],
        vmatrix: ["\u2502", "\u2502", "\u2502", "\u2502", "\u2502", "\u2502"],
        Vmatrix: ["\u2551", "\u2551", "\u2551", "\u2551", "\u2551", "\u2551"]
      };
      const delimiter = delimiters[environment];
      if (!delimiter) {
        this.supported = false;
        return rows.join("\n");
      }
      lines = rows.map((row, index2) => {
        const left = index2 === 0 ? delimiter[0] : index2 === rows.length - 1 ? delimiter[4] : delimiter[2];
        const right = index2 === 0 ? delimiter[1] : index2 === rows.length - 1 ? delimiter[5] : delimiter[3];
        return `${left} ${row} ${right}`;
      });
    }
    if (lines.length <= 1) {
      return lines[0] ?? "";
    }
    const index = this.layoutNodes.push({ type: "matrix", lines, baseline: 0 }) - 1;
    return `${LAYOUT_MARKER_START}${index}${LAYOUT_MARKER_END}`;
  }
  renderNested(source, stackFractions = true) {
    const rendered = new _LatexParser(source, this.layoutNodes, this.display && stackFractions).render();
    if (rendered === void 0) {
      this.supported = false;
      return source;
    }
    return rendered;
  }
};
function renderLatex(source, options = {}) {
  const layoutNodes = [];
  const rendered = new LatexParser(source, layoutNodes, options.display === true).render();
  if (rendered === void 0) {
    return void 0;
  }
  if (layoutNodes.length === 0) {
    return rendered.replaceAll(PROTECTED_SPACE, " ");
  }
  const lines = renderLayout(rendered, layoutNodes).lines;
  const indentation = Math.min(...lines.filter((line) => line.trim()).map((line) => line.length - line.trimStart().length));
  return lines.map((line) => line.slice(indentation).trimEnd()).join("\n").trimEnd().replaceAll(PROTECTED_SPACE, " ");
}

// node_modules/@earendil-works/pi-tui/dist/components/markdown.js
var STRICT_STRIKETHROUGH_REGEX = /^(~~)(?=[^\s~])((?:\\.|[^\\])*?(?:\\.|[^\s~\\]))\1(?=[^~]|$)/;
var StrictStrikethroughTokenizer = class extends w {
  del(src) {
    const match = STRICT_STRIKETHROUGH_REGEX.exec(src);
    if (!match) {
      return void 0;
    }
    const text = match[2];
    return {
      type: "del",
      raw: match[0],
      text,
      tokens: this.lexer.inlineTokens(text)
    };
  }
};
function isEscaped(source, index) {
  let backslashes = 0;
  for (let position = index - 1; position >= 0 && source[position] === "\\"; position--) {
    backslashes++;
  }
  return backslashes % 2 === 1;
}
function findClosingDelimiter(source, closing, start) {
  let index = source.indexOf(closing, start);
  while (index >= 0 && isEscaped(source, index)) {
    index = source.indexOf(closing, index + closing.length);
  }
  return index;
}
function looksLikePendingDollarMath(source) {
  return /\\[A-Za-z]+|[_^=+*/<>()[\]|±≤≥≠≈∈→⇒∞∫∑√-]/.test(source);
}
function tokenizeInlineLatex(source) {
  let opening = "";
  let closing = "";
  if (source.startsWith("$$")) {
    opening = "$$";
    closing = "$$";
  } else if (source.startsWith("\\(")) {
    opening = "\\(";
    closing = "\\)";
  } else if (source.startsWith("\\[")) {
    opening = "\\[";
    closing = "\\]";
  } else if (source.startsWith("$") && !/^\$\s/.test(source)) {
    opening = "$";
    closing = "$";
  } else {
    return void 0;
  }
  const closingIndex = findClosingDelimiter(source, closing, opening.length);
  if (closingIndex >= 0 && opening === "$" && (/\s$/.test(source.slice(opening.length, closingIndex)) || /^\d/.test(source.slice(closingIndex + 1)) || /^[A-Z_][A-Z0-9_]*(?:[^A-Za-z0-9_\s])?$/.test(source.slice(opening.length, closingIndex)) && /^[A-Za-z_][A-Za-z0-9_]*/.test(source.slice(closingIndex + 1)) || source.slice(opening.length, closingIndex).includes("`"))) {
    return void 0;
  }
  if (closingIndex < 0) {
    const pendingSource = source.slice(opening.length);
    if (opening.startsWith("\\") || looksLikePendingDollarMath(pendingSource)) {
      return { type: "latex", raw: source, text: pendingSource, pending: true };
    }
    return void 0;
  }
  const text = source.slice(opening.length, closingIndex);
  if (!text || text.includes("\n")) {
    return void 0;
  }
  const raw = source.slice(0, closingIndex + closing.length);
  return { type: "latex", raw, text };
}
function tokenizeBlockLatex(source) {
  const dollarMatch = /^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*?)\$\$[ \t]*(?:\n|$)/.exec(source);
  if (dollarMatch?.[1]) {
    return { type: "latexBlock", raw: dollarMatch[0], text: dollarMatch[1].trim() };
  }
  const bracketMatch = /^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*?)\\\][ \t]*(?:\n|$)/.exec(source);
  if (bracketMatch?.[1]) {
    return { type: "latexBlock", raw: bracketMatch[0], text: bracketMatch[1].trim() };
  }
  const pendingBracket = /^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*)$/.exec(source);
  if (pendingBracket) {
    return { type: "latexBlock", raw: pendingBracket[0], text: pendingBracket[1], pending: true };
  }
  const pendingDollar = /^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*)$/.exec(source);
  if (pendingDollar?.[1] && looksLikePendingDollarMath(pendingDollar[1])) {
    return { type: "latexBlock", raw: pendingDollar[0], text: pendingDollar[1], pending: true };
  }
  return void 0;
}
var LATEX_MARKDOWN_EXTENSIONS = [
  {
    name: "latexBlock",
    level: "block",
    start(source) {
      const match = /(?:^|\n) {0,3}(?:\$\$|\\\[)/.exec(source);
      return match ? match.index + (match[0].startsWith("\n") ? 1 : 0) : void 0;
    },
    tokenizer: tokenizeBlockLatex
  },
  {
    name: "latex",
    level: "inline",
    start(source) {
      const indices = [source.indexOf("$"), source.indexOf("\\("), source.indexOf("\\[")].filter((index) => index >= 0);
      return indices.length > 0 ? Math.min(...indices) : void 0;
    },
    tokenizer: tokenizeInlineLatex
  }
];
function trimPartialClosingFences(tokens) {
  const token = tokens[tokens.length - 1];
  if (token?.type === "list") {
    trimPartialClosingFences(token.items[token.items.length - 1]?.tokens ?? []);
    return;
  }
  if (token?.type === "blockquote") {
    trimPartialClosingFences(token.tokens ?? []);
    return;
  }
  if (token?.type !== "code") {
    return;
  }
  const marker = /^(`{3,}|~{3,})/.exec(token.raw)?.[1];
  const lastLine = token.raw.split("\n").pop();
  if (!marker || !lastLine || lastLine.length >= marker.length || lastLine !== marker[0]?.repeat(lastLine.length)) {
    return;
  }
  token.text = token.text.slice(0, -lastLine.length).replace(/\n$/, "");
}
var markdownParser = new q();
markdownParser.setOptions({
  tokenizer: new StrictStrikethroughTokenizer()
});
markdownParser.use({ extensions: [...LATEX_MARKDOWN_EXTENSIONS] });
var Markdown = class {
  text;
  paddingX;
  // Left/right padding
  paddingY;
  // Top/bottom padding
  defaultTextStyle;
  theme;
  options;
  defaultStylePrefix;
  // Cache for rendered output
  cachedText;
  cachedWidth;
  cachedLines;
  constructor(text, paddingX, paddingY, theme, defaultTextStyle, options) {
    this.text = text;
    this.paddingX = paddingX;
    this.paddingY = paddingY;
    this.theme = theme;
    this.defaultTextStyle = defaultTextStyle;
    this.options = options ? { ...options } : {};
  }
  setText(text) {
    this.text = text;
    this.invalidate();
  }
  invalidate() {
    this.cachedText = void 0;
    this.cachedWidth = void 0;
    this.cachedLines = void 0;
  }
  render(width) {
    if (this.cachedLines && this.cachedText === this.text && this.cachedWidth === width) {
      return this.cachedLines;
    }
    const contentWidth = Math.max(1, width - this.paddingX * 2);
    const text = this.options.transform?.(this.text, contentWidth) ?? this.text;
    if (!text || text.trim() === "") {
      const result2 = [];
      this.cachedText = this.text;
      this.cachedWidth = width;
      this.cachedLines = result2;
      return result2;
    }
    const normalizedText = text.replace(/\t/g, "   ");
    const tokens = markdownParser.lexer(normalizedText);
    trimPartialClosingFences(tokens);
    const renderedLines = [];
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      const nextToken = tokens[i + 1];
      const tokenLines = this.renderToken(token, contentWidth, nextToken?.type);
      for (const tokenLine of tokenLines) {
        renderedLines.push(tokenLine);
      }
    }
    const wrappedLines = [];
    for (const line of renderedLines) {
      if (isImageLine(line)) {
        wrappedLines.push(line);
      } else {
        for (const wrappedLine of wrapTextWithAnsi(line, contentWidth)) {
          wrappedLines.push(wrappedLine);
        }
      }
    }
    const leftMargin = " ".repeat(this.paddingX);
    const rightMargin = " ".repeat(this.paddingX);
    const bgFn = this.defaultTextStyle?.bgColor;
    const contentLines = [];
    for (const line of wrappedLines) {
      if (isImageLine(line)) {
        contentLines.push(line);
        continue;
      }
      const lineWithMargins = leftMargin + line + rightMargin;
      if (bgFn) {
        contentLines.push(applyBackgroundToLine(lineWithMargins, width, bgFn));
      } else {
        const visibleLen = visibleWidth(lineWithMargins);
        const paddingNeeded = Math.max(0, width - visibleLen);
        contentLines.push(lineWithMargins + " ".repeat(paddingNeeded));
      }
    }
    const emptyLine = " ".repeat(width);
    const emptyLines = [];
    for (let i = 0; i < this.paddingY; i++) {
      const line = bgFn ? applyBackgroundToLine(emptyLine, width, bgFn) : emptyLine;
      emptyLines.push(line);
    }
    const result = emptyLines.concat(contentLines, emptyLines);
    this.cachedText = this.text;
    this.cachedWidth = width;
    this.cachedLines = result;
    return result.length > 0 ? result : [""];
  }
  /**
   * Apply default text style to a string.
   * This is the base styling applied to all text content.
   * NOTE: Background color is NOT applied here - it's applied at the padding stage
   * to ensure it extends to the full line width.
   */
  applyDefaultStyle(text) {
    if (!this.defaultTextStyle) {
      return text;
    }
    let styled = text;
    if (this.defaultTextStyle.color) {
      styled = this.defaultTextStyle.color(styled);
    }
    if (this.defaultTextStyle.bold) {
      styled = this.theme.bold(styled);
    }
    if (this.defaultTextStyle.italic) {
      styled = this.theme.italic(styled);
    }
    if (this.defaultTextStyle.strikethrough) {
      styled = this.theme.strikethrough(styled);
    }
    if (this.defaultTextStyle.underline) {
      styled = this.theme.underline(styled);
    }
    return styled;
  }
  getDefaultStylePrefix() {
    if (!this.defaultTextStyle) {
      return "";
    }
    if (this.defaultStylePrefix !== void 0) {
      return this.defaultStylePrefix;
    }
    const sentinel = "\0";
    let styled = sentinel;
    if (this.defaultTextStyle.color) {
      styled = this.defaultTextStyle.color(styled);
    }
    if (this.defaultTextStyle.bold) {
      styled = this.theme.bold(styled);
    }
    if (this.defaultTextStyle.italic) {
      styled = this.theme.italic(styled);
    }
    if (this.defaultTextStyle.strikethrough) {
      styled = this.theme.strikethrough(styled);
    }
    if (this.defaultTextStyle.underline) {
      styled = this.theme.underline(styled);
    }
    const sentinelIndex = styled.indexOf(sentinel);
    this.defaultStylePrefix = sentinelIndex >= 0 ? styled.slice(0, sentinelIndex) : "";
    return this.defaultStylePrefix;
  }
  getStylePrefix(styleFn) {
    const sentinel = "\0";
    const styled = styleFn(sentinel);
    const sentinelIndex = styled.indexOf(sentinel);
    return sentinelIndex >= 0 ? styled.slice(0, sentinelIndex) : "";
  }
  getDefaultInlineStyleContext() {
    return {
      applyText: (text) => this.applyDefaultStyle(text),
      stylePrefix: this.getDefaultStylePrefix()
    };
  }
  renderToken(token, width, nextTokenType, styleContext) {
    const lines = [];
    switch (token.type) {
      case "heading": {
        const headingLevel = token.depth;
        const headingPrefix = `${"#".repeat(headingLevel)} `;
        let headingStyleFn;
        if (headingLevel === 1) {
          headingStyleFn = (text) => this.theme.heading(this.theme.bold(this.theme.underline(text)));
        } else {
          headingStyleFn = (text) => this.theme.heading(this.theme.bold(text));
        }
        const headingStyleContext = {
          applyText: headingStyleFn,
          stylePrefix: this.getStylePrefix(headingStyleFn)
        };
        const headingText = this.renderInlineTokens(token.tokens || [], headingStyleContext);
        const styledHeading = headingLevel >= 3 ? headingStyleFn(headingPrefix) + headingText : headingText;
        lines.push(styledHeading);
        if (nextTokenType && nextTokenType !== "space") {
          lines.push("");
        }
        break;
      }
      case "paragraph": {
        const paragraphText = this.renderInlineTokens(token.tokens || [], styleContext);
        lines.push(paragraphText);
        if (nextTokenType && nextTokenType !== "list" && nextTokenType !== "space") {
          lines.push("");
        }
        break;
      }
      case "text":
        lines.push(this.renderInlineTokens([token], styleContext));
        break;
      case "latexBlock": {
        const latexToken = token;
        const rendered = !latexToken.pending && this.options.renderLatex !== false ? renderLatex(latexToken.text, { display: true }) ?? latexToken.raw.trim() : latexToken.raw.trim();
        for (const line of rendered.split("\n")) {
          lines.push(this.applyDefaultStyle(line));
        }
        if (nextTokenType && nextTokenType !== "space") {
          lines.push("");
        }
        break;
      }
      case "code": {
        const indent = this.theme.codeBlockIndent ?? "  ";
        lines.push(this.theme.codeBlockBorder(`\`\`\`${token.lang || ""}`));
        if (this.theme.highlightCode) {
          const highlightedLines = this.theme.highlightCode(token.text, token.lang);
          for (const hlLine of highlightedLines) {
            lines.push(`${indent}${hlLine}`);
          }
        } else {
          const codeLines = token.text.split("\n");
          for (const codeLine of codeLines) {
            lines.push(`${indent}${this.theme.codeBlock(codeLine)}`);
          }
        }
        lines.push(this.theme.codeBlockBorder("```"));
        if (nextTokenType && nextTokenType !== "space") {
          lines.push("");
        }
        break;
      }
      case "list": {
        const listLines = this.renderList(token, 0, width, styleContext);
        lines.push(...listLines);
        break;
      }
      case "table": {
        const tableLines = this.renderTable(token, width, nextTokenType, styleContext);
        lines.push(...tableLines);
        break;
      }
      case "blockquote": {
        const quoteStyle = (text) => this.theme.quote(this.theme.italic(text));
        const quoteStylePrefix = this.getStylePrefix(quoteStyle);
        const applyQuoteStyle = (line) => {
          if (!quoteStylePrefix) {
            return quoteStyle(line);
          }
          const lineWithReappliedStyle = line.replace(/\x1b\[0m/g, `\x1B[0m${quoteStylePrefix}`);
          return quoteStyle(lineWithReappliedStyle);
        };
        const quoteContentWidth = Math.max(1, width - 2);
        const quoteInlineStyleContext = {
          applyText: (text) => text,
          stylePrefix: quoteStylePrefix
        };
        const quoteTokens = token.tokens || [];
        const renderedQuoteLines = [];
        for (let i = 0; i < quoteTokens.length; i++) {
          const quoteToken = quoteTokens[i];
          const nextQuoteToken = quoteTokens[i + 1];
          renderedQuoteLines.push(...this.renderToken(quoteToken, quoteContentWidth, nextQuoteToken?.type, quoteInlineStyleContext));
        }
        while (renderedQuoteLines.length > 0 && renderedQuoteLines[renderedQuoteLines.length - 1] === "") {
          renderedQuoteLines.pop();
        }
        for (const quoteLine of renderedQuoteLines) {
          const styledLine = applyQuoteStyle(quoteLine);
          const wrappedLines = wrapTextWithAnsi(styledLine, quoteContentWidth);
          for (const wrappedLine of wrappedLines) {
            lines.push(this.theme.quoteBorder("\u2502 ") + wrappedLine);
          }
        }
        if (nextTokenType && nextTokenType !== "space") {
          lines.push("");
        }
        break;
      }
      case "hr":
        lines.push(this.theme.hr("\u2500".repeat(Math.min(width, 80))));
        if (nextTokenType && nextTokenType !== "space") {
          lines.push("");
        }
        break;
      case "html":
        if ("raw" in token && typeof token.raw === "string") {
          lines.push(this.applyDefaultStyle(token.raw.trim()));
        }
        break;
      case "space":
        lines.push("");
        break;
      default:
        if ("text" in token && typeof token.text === "string") {
          lines.push(token.text);
        }
    }
    return lines;
  }
  renderInlineTokens(tokens, styleContext) {
    let result = "";
    const resolvedStyleContext = styleContext ?? this.getDefaultInlineStyleContext();
    const { applyText, stylePrefix } = resolvedStyleContext;
    const applyTextWithNewlines = (text) => {
      const segments = text.split("\n");
      return segments.map((segment) => applyText(segment)).join("\n");
    };
    for (const token of tokens) {
      switch (token.type) {
        case "latex": {
          const latexToken = token;
          const rendered = !latexToken.pending && this.options.renderLatex !== false ? renderLatex(latexToken.text) ?? latexToken.raw : latexToken.raw;
          result += applyTextWithNewlines(rendered);
          break;
        }
        case "escape":
          result += applyTextWithNewlines(this.options.preserveBackslashEscapes ? token.raw : token.text);
          break;
        case "text":
          if (token.tokens && token.tokens.length > 0) {
            result += this.renderInlineTokens(token.tokens, resolvedStyleContext);
          } else {
            result += applyTextWithNewlines(token.text);
          }
          break;
        case "paragraph":
          result += this.renderInlineTokens(token.tokens || [], resolvedStyleContext);
          break;
        case "strong": {
          const boldContent = this.renderInlineTokens(token.tokens || [], resolvedStyleContext);
          result += this.theme.bold(boldContent) + stylePrefix;
          break;
        }
        case "em": {
          const italicContent = this.renderInlineTokens(token.tokens || [], resolvedStyleContext);
          result += this.theme.italic(italicContent) + stylePrefix;
          break;
        }
        case "codespan":
          result += this.theme.code(token.text) + stylePrefix;
          break;
        case "link": {
          const linkText = this.renderInlineTokens(token.tokens || [], resolvedStyleContext);
          const styledLink = this.theme.link(this.theme.underline(linkText));
          if (getCapabilities().hyperlinks) {
            result += hyperlink(styledLink, token.href) + stylePrefix;
          } else {
            const hrefForComparison = token.href.startsWith("mailto:") ? token.href.slice(7) : token.href;
            if (token.text === token.href || token.text === hrefForComparison) {
              result += styledLink + stylePrefix;
            } else {
              result += styledLink + this.theme.linkUrl(` (${token.href})`) + stylePrefix;
            }
          }
          break;
        }
        case "br":
          result += "\n";
          break;
        case "del": {
          const delContent = this.renderInlineTokens(token.tokens || [], resolvedStyleContext);
          result += this.theme.strikethrough(delContent) + stylePrefix;
          break;
        }
        case "html":
          if ("raw" in token && typeof token.raw === "string") {
            result += applyTextWithNewlines(token.raw);
          }
          break;
        default:
          if ("text" in token && typeof token.text === "string") {
            result += applyTextWithNewlines(token.text);
          }
      }
    }
    while (stylePrefix && result.endsWith(stylePrefix)) {
      result = result.slice(0, -stylePrefix.length);
    }
    return result;
  }
  getOrderedListMarker(item) {
    const match = /^(?: {0,3})(\d{1,9}[.)])[ \t]+/.exec(item.raw);
    return match ? `${match[1]} ` : void 0;
  }
  getUnorderedListMarker(item) {
    const match = /^(?: {0,3})([-+*])(?:[ \t]+|(?=\r?\n|$))/.exec(item.raw);
    return match ? `${match[1]} ` : void 0;
  }
  /**
   * Render a list with proper nesting support
   */
  renderList(token, depth, width, styleContext) {
    const lines = [];
    const indent = "    ".repeat(depth);
    const startNumber = typeof token.start === "number" ? token.start : 1;
    for (let i = 0; i < token.items.length; i++) {
      const item = token.items[i];
      const isLastItem = i === token.items.length - 1;
      const bullet = token.ordered ? this.options.preserveOrderedListMarkers ? this.getOrderedListMarker(item) ?? `${startNumber + i}. ` : `${startNumber + i}. ` : this.options.preserveOrderedListMarkers ? this.getUnorderedListMarker(item) ?? "- " : "- ";
      const taskMarker = item.task ? `[${item.checked ? "x" : " "}] ` : "";
      const marker = bullet + taskMarker;
      const firstPrefix = indent + this.theme.listBullet(marker);
      const continuationPrefix = indent + " ".repeat(visibleWidth(marker));
      const itemWidth = Math.max(1, width - visibleWidth(firstPrefix));
      let renderedAnyLine = false;
      for (const itemToken of item.tokens) {
        if (itemToken.type === "list") {
          lines.push(...this.renderList(itemToken, depth + 1, width, styleContext));
          renderedAnyLine = true;
          continue;
        }
        const itemLines = this.renderToken(itemToken, itemWidth, void 0, styleContext);
        for (const line of itemLines) {
          for (const wrappedLine of wrapTextWithAnsi(line, itemWidth)) {
            const linePrefix = renderedAnyLine ? continuationPrefix : firstPrefix;
            lines.push(linePrefix + wrappedLine);
            renderedAnyLine = true;
          }
        }
      }
      if (!renderedAnyLine) {
        lines.push(firstPrefix);
      }
      if (token.loose && !isLastItem) {
        lines.push("");
      }
    }
    return lines;
  }
  /**
   * Get the visible width of the longest word in a string.
   */
  getLongestWordWidth(text, maxWidth) {
    const words = text.split(/\s+/).filter((word) => word.length > 0);
    let longest = 0;
    for (const word of words) {
      longest = Math.max(longest, visibleWidth(word));
    }
    if (maxWidth === void 0) {
      return longest;
    }
    return Math.min(longest, maxWidth);
  }
  /**
   * Wrap a table cell to fit into a column.
   *
   * Delegates to wrapTextWithAnsi() so ANSI codes + long tokens are handled
   * consistently with the rest of the renderer.
   */
  wrapCellText(text, maxWidth, stylePrefix = "") {
    const lines = wrapTextWithAnsi(text, Math.max(1, maxWidth));
    return lines.map((line, index) => {
      const styleReset = index < lines.length - 1 ? "\x1B[22;23;24;25;27;28;29;39m" : "";
      return `${line}${styleReset}${stylePrefix}`;
    });
  }
  /**
   * Render a table with width-aware cell wrapping.
   * Cells that don't fit are wrapped to multiple lines.
   */
  renderTable(token, availableWidth, nextTokenType, styleContext) {
    const lines = [];
    const numCols = token.header.length;
    if (numCols === 0) {
      return lines;
    }
    const borderOverhead = 3 * numCols + 1;
    const availableForCells = availableWidth - borderOverhead;
    if (availableForCells < numCols) {
      const fallbackLines = token.raw ? wrapTextWithAnsi(token.raw, availableWidth) : [];
      if (nextTokenType && nextTokenType !== "space") {
        fallbackLines.push("");
      }
      return fallbackLines;
    }
    const maxUnbrokenWordWidth = 30;
    const naturalWidths = [];
    const minWordWidths = [];
    for (let i = 0; i < numCols; i++) {
      const headerText = this.renderInlineTokens(token.header[i].tokens || [], styleContext);
      naturalWidths[i] = visibleWidth(headerText);
      minWordWidths[i] = Math.max(1, this.getLongestWordWidth(headerText, maxUnbrokenWordWidth));
    }
    for (const row of token.rows) {
      for (let i = 0; i < row.length; i++) {
        const cellText = this.renderInlineTokens(row[i].tokens || [], styleContext);
        naturalWidths[i] = Math.max(naturalWidths[i] || 0, visibleWidth(cellText));
        minWordWidths[i] = Math.max(minWordWidths[i] || 1, this.getLongestWordWidth(cellText, maxUnbrokenWordWidth));
      }
    }
    let minColumnWidths = minWordWidths;
    let minCellsWidth = minColumnWidths.reduce((a, b2) => a + b2, 0);
    if (minCellsWidth > availableForCells) {
      minColumnWidths = new Array(numCols).fill(1);
      const remaining = availableForCells - numCols;
      if (remaining > 0) {
        const totalWeight = minWordWidths.reduce((total, width) => total + Math.max(0, width - 1), 0);
        const growth = minWordWidths.map((width) => {
          const weight = Math.max(0, width - 1);
          return totalWeight > 0 ? Math.floor(weight / totalWeight * remaining) : 0;
        });
        for (let i = 0; i < numCols; i++) {
          minColumnWidths[i] += growth[i] ?? 0;
        }
        const allocated = growth.reduce((total, width) => total + width, 0);
        let leftover = remaining - allocated;
        for (let i = 0; leftover > 0 && i < numCols; i++) {
          minColumnWidths[i]++;
          leftover--;
        }
      }
      minCellsWidth = minColumnWidths.reduce((a, b2) => a + b2, 0);
    }
    const totalNaturalWidth = naturalWidths.reduce((a, b2) => a + b2, 0) + borderOverhead;
    let columnWidths;
    if (totalNaturalWidth <= availableWidth) {
      columnWidths = naturalWidths.map((width, index) => Math.max(width, minColumnWidths[index]));
    } else {
      const totalGrowPotential = naturalWidths.reduce((total, width, index) => {
        return total + Math.max(0, width - minColumnWidths[index]);
      }, 0);
      const extraWidth = Math.max(0, availableForCells - minCellsWidth);
      columnWidths = minColumnWidths.map((minWidth, index) => {
        const naturalWidth = naturalWidths[index];
        const minWidthDelta = Math.max(0, naturalWidth - minWidth);
        let grow = 0;
        if (totalGrowPotential > 0) {
          grow = Math.floor(minWidthDelta / totalGrowPotential * extraWidth);
        }
        return minWidth + grow;
      });
      const allocated = columnWidths.reduce((a, b2) => a + b2, 0);
      let remaining = availableForCells - allocated;
      while (remaining > 0) {
        let grew = false;
        for (let i = 0; i < numCols && remaining > 0; i++) {
          if (columnWidths[i] < naturalWidths[i]) {
            columnWidths[i]++;
            remaining--;
            grew = true;
          }
        }
        if (!grew) {
          break;
        }
      }
    }
    const topBorderCells = columnWidths.map((w2) => "\u2500".repeat(w2));
    lines.push(`\u250C\u2500${topBorderCells.join("\u2500\u252C\u2500")}\u2500\u2510`);
    const headerCellLines = token.header.map((cell, i) => {
      const text = this.renderInlineTokens(cell.tokens || [], styleContext);
      return this.wrapCellText(text, columnWidths[i], styleContext?.stylePrefix);
    });
    const headerLineCount = Math.max(...headerCellLines.map((c) => c.length));
    for (let lineIdx = 0; lineIdx < headerLineCount; lineIdx++) {
      const rowParts = headerCellLines.map((cellLines, colIdx) => {
        const text = cellLines[lineIdx] || "";
        const padded = text + " ".repeat(Math.max(0, columnWidths[colIdx] - visibleWidth(text)));
        return this.theme.bold(padded);
      });
      lines.push(`\u2502 ${rowParts.join(" \u2502 ")} \u2502`);
    }
    const separatorCells = columnWidths.map((w2) => "\u2500".repeat(w2));
    const separatorLine = `\u251C\u2500${separatorCells.join("\u2500\u253C\u2500")}\u2500\u2524`;
    lines.push(separatorLine);
    for (let rowIndex = 0; rowIndex < token.rows.length; rowIndex++) {
      const row = token.rows[rowIndex];
      const rowCellLines = row.map((cell, i) => {
        const text = this.renderInlineTokens(cell.tokens || [], styleContext);
        return this.wrapCellText(text, columnWidths[i], styleContext?.stylePrefix);
      });
      const rowLineCount = Math.max(...rowCellLines.map((c) => c.length));
      for (let lineIdx = 0; lineIdx < rowLineCount; lineIdx++) {
        const rowParts = rowCellLines.map((cellLines, colIdx) => {
          const text = cellLines[lineIdx] || "";
          return text + " ".repeat(Math.max(0, columnWidths[colIdx] - visibleWidth(text)));
        });
        lines.push(`\u2502 ${rowParts.join(" \u2502 ")} \u2502`);
      }
      if (rowIndex < token.rows.length - 1) {
        lines.push(separatorLine);
      }
    }
    const bottomBorderCells = columnWidths.map((w2) => "\u2500".repeat(w2));
    lines.push(`\u2514\u2500${bottomBorderCells.join("\u2500\u2534\u2500")}\u2500\u2518`);
    if (nextTokenType && nextTokenType !== "space") {
      lines.push("");
    }
    return lines;
  }
};

// node_modules/@earendil-works/pi-tui/dist/components/scroll-view.js
var ScrollView = class extends Container {
  child;
  followEnd;
  primary;
  overscroll;
  scrollbarTrackStyle;
  scrollbarThumbStyle;
  currentScrollbar;
  scrollbarHideDelayMs;
  currentScrollTop = 0;
  contentHeight = 0;
  currentViewportHeight = 0;
  followingEnd;
  followSuppressedAtEnd = false;
  requestRenderCallback;
  transientScrollbarVisible = false;
  scrollbarActive = false;
  scrollbarHideTimer;
  constructor(component, options = {}) {
    super();
    if (options.axis !== void 0 && options.axis !== "vertical") {
      throw new Error(`Unsupported ScrollView axis: ${options.axis}`);
    }
    this.child = component;
    this.children.push(component);
    this.followEnd = (options.follow ?? "none") === "end";
    this.followingEnd = this.followEnd;
    this.primary = options.primary ?? false;
    this.overscroll = options.overscroll ?? "chain";
    this.currentScrollbar = options.scrollbar ?? "hidden";
    this.scrollbarTrackStyle = options.scrollbarTrackStyle ?? ((text) => `\x1B[90m${text}\x1B[39m`);
    this.scrollbarThumbStyle = options.scrollbarThumbStyle ?? ((text) => `\x1B[37m${text}\x1B[39m`);
    this.scrollbarHideDelayMs = Math.max(0, Math.floor(options.scrollbarHideDelayMs ?? 1e3));
  }
  get scrollTop() {
    return this.currentScrollTop;
  }
  get isFollowingEnd() {
    return this.followingEnd;
  }
  get viewportHeight() {
    return this.currentViewportHeight;
  }
  get scrollbar() {
    return this.currentScrollbar;
  }
  get isScrollbarVisible() {
    if (this.scrollbar === "always")
      return this.currentViewportHeight > 0;
    return this.scrollbar === "auto" && this.contentHeight > this.currentViewportHeight && this.transientScrollbarVisible;
  }
  get isScrollbarActive() {
    return this.scrollbarActive;
  }
  setScrollbar(scrollbar) {
    if (scrollbar === this.currentScrollbar)
      return;
    this.currentScrollbar = scrollbar;
    if (scrollbar !== "auto")
      this.hideTransientScrollbar();
    else if (this.scrollbarActive)
      this.markScrollbarActivity();
    this.requestRenderCallback?.();
  }
  getContentWidth(width) {
    return this.scrollbar === "always" && width > 1 ? width - 1 : width;
  }
  markScrollbarActivity() {
    if (this.scrollbar !== "auto" || this.contentHeight <= this.currentViewportHeight)
      return;
    this.transientScrollbarVisible = true;
    if (this.scrollbarHideTimer) {
      clearTimeout(this.scrollbarHideTimer);
      this.scrollbarHideTimer = void 0;
    }
    if (this.scrollbarActive)
      return;
    this.scrollbarHideTimer = setTimeout(() => {
      this.scrollbarHideTimer = void 0;
      this.transientScrollbarVisible = false;
      this.requestRenderCallback?.();
    }, this.scrollbarHideDelayMs);
    this.scrollbarHideTimer.unref();
  }
  hideTransientScrollbar() {
    this.transientScrollbarVisible = false;
    if (!this.scrollbarHideTimer)
      return;
    clearTimeout(this.scrollbarHideTimer);
    this.scrollbarHideTimer = void 0;
  }
  setScrollbarActive(active) {
    if (active === this.scrollbarActive)
      return;
    this.scrollbarActive = active;
    this.markScrollbarActivity();
    this.requestRenderCallback?.();
  }
  scrollTo(scrollTop, options = {}) {
    const requested = Number.isFinite(scrollTop) ? Math.trunc(scrollTop) : this.currentScrollTop;
    const maxScrollTop = Math.max(0, this.contentHeight - this.currentViewportHeight);
    const next = Math.max(0, Math.min(maxScrollTop, requested));
    const nextFollowSuppressedAtEnd = options.disableFollow === true && next === maxScrollTop;
    const nextFollowingEnd = !nextFollowSuppressedAtEnd && this.followEnd && next === maxScrollTop;
    if (next === this.currentScrollTop && nextFollowingEnd === this.followingEnd && nextFollowSuppressedAtEnd === this.followSuppressedAtEnd) {
      return;
    }
    const moved = next !== this.currentScrollTop;
    this.currentScrollTop = next;
    this.followingEnd = nextFollowingEnd;
    this.followSuppressedAtEnd = nextFollowSuppressedAtEnd;
    if (moved)
      this.markScrollbarActivity();
    this.requestRenderCallback?.();
  }
  scrollBy(lines) {
    const requested = Number.isFinite(lines) ? Math.trunc(lines) : 0;
    if (requested === 0)
      return 0;
    const maxScrollTop = Math.max(0, this.contentHeight - this.currentViewportHeight);
    const start = this.followingEnd ? maxScrollTop : this.currentScrollTop;
    const next = Math.max(0, Math.min(maxScrollTop, start + requested));
    const moved = next - start;
    const wasFollowingEnd = this.followingEnd;
    this.currentScrollTop = next;
    this.followingEnd = this.followEnd && next === maxScrollTop;
    this.followSuppressedAtEnd = false;
    if (moved !== 0)
      this.markScrollbarActivity();
    if (moved !== 0 || this.followingEnd !== wasFollowingEnd)
      this.requestRenderCallback?.();
    return requested - moved;
  }
  scrollToStart() {
    const changed = this.currentScrollTop !== 0 || this.followingEnd !== (this.followEnd && this.contentHeight <= this.currentViewportHeight);
    this.currentScrollTop = 0;
    this.followingEnd = this.followEnd && this.contentHeight <= this.currentViewportHeight;
    this.followSuppressedAtEnd = false;
    if (changed) {
      this.markScrollbarActivity();
      this.requestRenderCallback?.();
    }
  }
  scrollToEnd() {
    const next = Math.max(0, this.contentHeight - this.currentViewportHeight);
    const changed = this.currentScrollTop !== next || this.followingEnd !== this.followEnd;
    this.currentScrollTop = next;
    this.followingEnd = this.followEnd;
    this.followSuppressedAtEnd = false;
    if (changed) {
      this.markScrollbarActivity();
      this.requestRenderCallback?.();
    }
  }
  updateLayout(contentHeight, viewportHeight, requestRender) {
    this.contentHeight = Math.max(0, Math.floor(contentHeight));
    this.currentViewportHeight = Math.max(0, Math.floor(viewportHeight));
    this.requestRenderCallback = requestRender;
    const maxScrollTop = Math.max(0, this.contentHeight - this.currentViewportHeight);
    if (this.followingEnd)
      this.currentScrollTop = maxScrollTop;
    else
      this.currentScrollTop = Math.max(0, Math.min(this.currentScrollTop, maxScrollTop));
    if (this.currentScrollTop < maxScrollTop)
      this.followSuppressedAtEnd = false;
    if (this.followEnd && this.currentScrollTop === maxScrollTop && !this.followSuppressedAtEnd) {
      this.followingEnd = true;
    }
    if (this.contentHeight <= this.currentViewportHeight)
      this.hideTransientScrollbar();
  }
  addChild(_component) {
    throw new Error("ScrollView has exactly one child");
  }
  removeChild(_component) {
    throw new Error("ScrollView child cannot be removed");
  }
  clear() {
    throw new Error("ScrollView child cannot be cleared");
  }
  render(width) {
    const contentWidth = this.getContentWidth(width);
    const lines = this.child.render(contentWidth);
    return contentWidth === width ? lines : lines.map((line) => `${line} `);
  }
  [LAYOUT_NODE]() {
    return { type: "scroll", component: this.child, state: this };
  }
};

// node_modules/@earendil-works/pi-tui/dist/components/v-stack.js
var VStack = class extends Stack {
  layoutType = "vstack";
  constructor(children = [], options = {}) {
    super(children, options);
  }
  render(width) {
    const viewport = { width: Math.max(1, width), height: Number.MAX_SAFE_INTEGER };
    const entries = visibleStackEntries(this.entries, viewport);
    const rendered = entries.map((entry) => entry.component.render(viewport.width));
    const sizes = allocateStackSizes(entries, rendered.map((lines2) => lines2.length), void 0, this.gap);
    const lines = [];
    for (let index = 0; index < entries.length; index++) {
      if (index > 0) {
        for (let gap = 0; gap < this.gap; gap++)
          lines.push("");
      }
      const childLines = rendered[index].slice(0, sizes[index]);
      lines.push(...childLines);
      for (let padding = childLines.length; padding < sizes[index]; padding++)
        lines.push("");
    }
    return lines;
  }
};

// node_modules/@earendil-works/pi-tui/dist/stdin-buffer.js
import { EventEmitter } from "events";
var ESC = "\x1B";
var DEFAULT_SEQUENCE_TIMEOUT_MS = 50;
var DEFAULT_ESCAPE_TIMEOUT_MS = 10;
var BRACKETED_PASTE_START = "\x1B[200~";
var BRACKETED_PASTE_END = "\x1B[201~";
function isCompleteSequence(data) {
  if (!data.startsWith(ESC)) {
    return "not-escape";
  }
  if (data.length === 1) {
    return "incomplete";
  }
  const afterEsc = data.slice(1);
  if (afterEsc.startsWith("[")) {
    if (afterEsc.startsWith("[M")) {
      return data.length >= 6 ? "complete" : "incomplete";
    }
    return isCompleteCsiSequence(data);
  }
  if (afterEsc.startsWith("]")) {
    return isCompleteOscSequence(data);
  }
  if (afterEsc.startsWith("P")) {
    return isCompleteDcsSequence(data);
  }
  if (afterEsc.startsWith("_")) {
    return isCompleteApcSequence(data);
  }
  if (afterEsc.startsWith("O")) {
    return afterEsc.length >= 2 ? "complete" : "incomplete";
  }
  if (afterEsc.length === 1) {
    return "complete";
  }
  return "complete";
}
function isCompleteCsiSequence(data) {
  if (!data.startsWith(`${ESC}[`)) {
    return "complete";
  }
  if (data.length < 3) {
    return "incomplete";
  }
  const payload = data.slice(2);
  const lastChar = payload[payload.length - 1];
  const lastCharCode = lastChar.charCodeAt(0);
  if (lastCharCode >= 64 && lastCharCode <= 126) {
    if (payload.startsWith("<")) {
      const mouseMatch = /^<\d+;\d+;\d+[Mm]$/.test(payload);
      if (mouseMatch) {
        return "complete";
      }
      if (lastChar === "M" || lastChar === "m") {
        const parts = payload.slice(1, -1).split(";");
        if (parts.length === 3 && parts.every((p) => /^\d+$/.test(p))) {
          return "complete";
        }
      }
      return "incomplete";
    }
    return "complete";
  }
  return "incomplete";
}
function isCompleteOscSequence(data) {
  if (!data.startsWith(`${ESC}]`)) {
    return "complete";
  }
  if (data.endsWith(`${ESC}\\`) || data.endsWith("\x07")) {
    return "complete";
  }
  return "incomplete";
}
function isCompleteDcsSequence(data) {
  if (!data.startsWith(`${ESC}P`)) {
    return "complete";
  }
  if (data.endsWith(`${ESC}\\`)) {
    return "complete";
  }
  return "incomplete";
}
function isCompleteApcSequence(data) {
  if (!data.startsWith(`${ESC}_`)) {
    return "complete";
  }
  if (data.endsWith(`${ESC}\\`)) {
    return "complete";
  }
  return "incomplete";
}
function parseUnmodifiedKittyPrintableCodepoint(sequence) {
  const match = sequence.match(/^\x1b\[(\d+)(?::\d*)?(?::\d+)?u$/);
  if (!match)
    return void 0;
  const codepoint = parseInt(match[1], 10);
  return codepoint >= 32 ? codepoint : void 0;
}
function extractCompleteSequences(buffer) {
  const sequences = [];
  let pos = 0;
  while (pos < buffer.length) {
    const remaining = buffer.slice(pos);
    if (remaining.startsWith(ESC)) {
      let seqEnd = 1;
      while (seqEnd <= remaining.length) {
        const candidate = remaining.slice(0, seqEnd);
        const status = isCompleteSequence(candidate);
        if (status === "complete") {
          if (candidate === "\x1B\x1B") {
            const nextChar = remaining[seqEnd];
            if (nextChar === "[" || // CSI
            nextChar === "]" || // OSC
            nextChar === "O" || // SS3
            nextChar === "P" || // DCS
            nextChar === "_") {
              sequences.push(ESC);
              pos += 1;
              break;
            }
          }
          sequences.push(candidate);
          pos += seqEnd;
          break;
        } else if (status === "incomplete") {
          seqEnd++;
        } else {
          sequences.push(candidate);
          pos += seqEnd;
          break;
        }
      }
      if (seqEnd > remaining.length) {
        return { sequences, remainder: remaining };
      }
    } else {
      sequences.push(remaining[0]);
      pos++;
    }
  }
  return { sequences, remainder: "" };
}
var StdinBuffer = class extends EventEmitter {
  buffer = "";
  timeout = null;
  timeoutMs;
  escapeTimeoutMs;
  pasteMode = false;
  pasteBuffer = "";
  pendingKittyPrintableCodepoint;
  constructor(options = {}) {
    super();
    this.timeoutMs = options.timeout ?? DEFAULT_SEQUENCE_TIMEOUT_MS;
    this.escapeTimeoutMs = options.escapeTimeout ?? DEFAULT_ESCAPE_TIMEOUT_MS;
  }
  process(data) {
    if (this.timeout) {
      clearTimeout(this.timeout);
      this.timeout = null;
    }
    let str;
    if (Buffer.isBuffer(data)) {
      if (data.length === 1 && data[0] > 127) {
        const byte = data[0] - 128;
        str = `\x1B${String.fromCharCode(byte)}`;
      } else {
        str = data.toString();
      }
    } else {
      str = data;
    }
    if (str.length === 0 && this.buffer.length === 0) {
      this.emitDataSequence("");
      return;
    }
    this.buffer += str;
    if (this.pasteMode) {
      this.pasteBuffer += this.buffer;
      this.buffer = "";
      const endIndex = this.pasteBuffer.indexOf(BRACKETED_PASTE_END);
      if (endIndex !== -1) {
        const pastedContent = this.pasteBuffer.slice(0, endIndex);
        const remaining = this.pasteBuffer.slice(endIndex + BRACKETED_PASTE_END.length);
        this.pasteMode = false;
        this.pasteBuffer = "";
        this.pendingKittyPrintableCodepoint = void 0;
        this.emit("paste", pastedContent);
        if (remaining.length > 0) {
          this.process(remaining);
        }
      }
      return;
    }
    const startIndex = this.buffer.indexOf(BRACKETED_PASTE_START);
    if (startIndex !== -1) {
      if (startIndex > 0) {
        const beforePaste = this.buffer.slice(0, startIndex);
        const result2 = extractCompleteSequences(beforePaste);
        for (const sequence of result2.sequences) {
          this.emitDataSequence(sequence);
        }
      }
      this.pendingKittyPrintableCodepoint = void 0;
      this.buffer = this.buffer.slice(startIndex + BRACKETED_PASTE_START.length);
      this.pasteMode = true;
      this.pasteBuffer = this.buffer;
      this.buffer = "";
      const endIndex = this.pasteBuffer.indexOf(BRACKETED_PASTE_END);
      if (endIndex !== -1) {
        const pastedContent = this.pasteBuffer.slice(0, endIndex);
        const remaining = this.pasteBuffer.slice(endIndex + BRACKETED_PASTE_END.length);
        this.pasteMode = false;
        this.pasteBuffer = "";
        this.pendingKittyPrintableCodepoint = void 0;
        this.emit("paste", pastedContent);
        if (remaining.length > 0) {
          this.process(remaining);
        }
      }
      return;
    }
    const result = extractCompleteSequences(this.buffer);
    this.buffer = result.remainder;
    for (const sequence of result.sequences) {
      this.emitDataSequence(sequence);
    }
    if (this.buffer.length > 0) {
      const timeoutMs = this.buffer === ESC ? this.escapeTimeoutMs : this.timeoutMs;
      this.timeout = setTimeout(() => {
        const flushed = this.flush();
        for (const sequence of flushed) {
          this.emitDataSequence(sequence);
        }
      }, timeoutMs);
    }
  }
  emitDataSequence(sequence) {
    const rawCodepoint = sequence.length === 1 ? sequence.codePointAt(0) : void 0;
    if (rawCodepoint !== void 0 && rawCodepoint === this.pendingKittyPrintableCodepoint) {
      this.pendingKittyPrintableCodepoint = void 0;
      return;
    }
    this.pendingKittyPrintableCodepoint = parseUnmodifiedKittyPrintableCodepoint(sequence);
    this.emit("data", sequence);
  }
  flush() {
    if (this.timeout) {
      clearTimeout(this.timeout);
      this.timeout = null;
    }
    if (this.buffer.length === 0) {
      return [];
    }
    const sequences = [this.buffer];
    this.buffer = "";
    this.pendingKittyPrintableCodepoint = void 0;
    return sequences;
  }
  clear() {
    if (this.timeout) {
      clearTimeout(this.timeout);
      this.timeout = null;
    }
    this.buffer = "";
    this.pasteMode = false;
    this.pasteBuffer = "";
    this.pendingKittyPrintableCodepoint = void 0;
  }
  getBuffer() {
    return this.buffer;
  }
  destroy() {
    this.clear();
  }
};

// node_modules/@earendil-works/pi-tui/dist/terminal.js
import * as fs from "node:fs";
import { createRequire as createRequire3 } from "node:module";
import * as path2 from "node:path";

// node_modules/@earendil-works/pi-tui/dist/native-modifiers.js
import { createRequire as createRequire2 } from "node:module";
import * as path from "node:path";

// node_modules/@earendil-works/pi-tui/dist/native-module-path.js
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
var moduleRequire = createRequire(import.meta.url);
var TUI_PACKAGE_NAME = "@earendil-works/pi-tui";
function getNativeModuleCandidates(nativePath, options = {}) {
  const moduleDir = dirname(fileURLToPath(options.moduleUrl ?? import.meta.url));
  const candidates = [];
  try {
    const packageEntry = (options.resolvePackage ?? moduleRequire.resolve)(TUI_PACKAGE_NAME);
    candidates.push(join(dirname(packageEntry), "..", nativePath));
  } catch {
  }
  candidates.push(join(moduleDir, "..", nativePath), join(moduleDir, nativePath), join(dirname(options.execPath ?? process.execPath), nativePath));
  return Array.from(new Set(candidates));
}

// node_modules/@earendil-works/pi-tui/dist/native-modifiers.js
var cjsRequire = createRequire2(import.meta.url);
var nativeModifiersHelper;
function isNativeModifiersHelper(value) {
  if (typeof value !== "object" || value === null)
    return false;
  const candidate = value.isModifierPressed;
  return typeof candidate === "function";
}
function loadNativeModifiersHelper() {
  if (nativeModifiersHelper !== void 0)
    return nativeModifiersHelper ?? void 0;
  nativeModifiersHelper = null;
  const arch = process.arch;
  if (arch !== "x64" && arch !== "arm64")
    return void 0;
  let nativePath;
  if (process.platform === "darwin") {
    nativePath = path.join("native", "darwin", "prebuilds", `darwin-${arch}`, "darwin-modifiers.node");
  } else if (process.platform === "win32") {
    nativePath = path.join("native", "win32", "prebuilds", `win32-${arch}`, "win32-console-mode.node");
  } else {
    return void 0;
  }
  for (const modulePath of getNativeModuleCandidates(nativePath)) {
    try {
      const helper = cjsRequire(modulePath);
      if (isNativeModifiersHelper(helper)) {
        nativeModifiersHelper = helper;
        return helper;
      }
    } catch {
    }
  }
  return void 0;
}
function isNativeModifierPressed(key) {
  const helper = loadNativeModifiersHelper();
  if (!helper)
    return false;
  try {
    return helper.isModifierPressed(key) === true;
  } catch {
    return false;
  }
}

// node_modules/@earendil-works/pi-tui/dist/terminal.js
var cjsRequire2 = createRequire3(import.meta.url);
var TERMINAL_PROGRESS_KEEPALIVE_MS = 1e3;
var TERMINAL_PROGRESS_ACTIVE_SEQUENCE = "\x1B]9;4;3\x07";
var TERMINAL_PROGRESS_CLEAR_SEQUENCE = "\x1B]9;4;0\x07";
var NATIVE_SHIFT_ENTER_SEQUENCE = "\x1B[13;2u";
var DESIRED_KITTY_KEYBOARD_PROTOCOL_FLAGS = 7;
var KEYBOARD_PROTOCOL_RESPONSE_FRAGMENT_TIMEOUT_MS = 150;
var KITTY_KEYBOARD_PROTOCOL_QUERY = `\x1B[>${DESIRED_KITTY_KEYBOARD_PROTOCOL_FLAGS}u\x1B[?u\x1B[c`;
function parseKeyboardProtocolNegotiationSequence(sequence) {
  const kittyFlags = sequence.match(/^\x1b\[\?(\d+)u$/);
  if (kittyFlags) {
    return { type: "kitty-flags", flags: Number.parseInt(kittyFlags[1], 10) };
  }
  if (/^\x1b\[\?[\d;]*c$/.test(sequence)) {
    return { type: "device-attributes" };
  }
  return void 0;
}
function isKeyboardProtocolNegotiationSequencePrefix(sequence) {
  return sequence === "\x1B[" || /^\x1b\[\?[\d;]*$/.test(sequence);
}
function isAppleTerminalSession() {
  return process.platform === "darwin" && process.env.TERM_PROGRAM === "Apple_Terminal";
}
function refreshTerminalDimensions() {
  if (process.platform === "win32" || process.pid <= 0)
    return;
  try {
    process.kill(process.pid, "SIGWINCH");
  } catch {
  }
}
function normalizeNativeShiftEnterInput(data, shouldDetectNativeShiftEnter, isShiftPressed) {
  if (shouldDetectNativeShiftEnter && data === "\r" && isShiftPressed)
    return NATIVE_SHIFT_ENTER_SEQUENCE;
  return data;
}
var DEFAULT_ESCAPE_TIMEOUT_MS2 = 10;
var DEFAULT_SSH_ESCAPE_TIMEOUT_MS = 100;
function resolveEscapeTimeoutMs(env = process.env) {
  const configured = Number(env.PI_TUI_ESC_TIMEOUT);
  if (Number.isFinite(configured) && configured > 0) {
    return configured;
  }
  if (env.SSH_CONNECTION || env.SSH_TTY) {
    return DEFAULT_SSH_ESCAPE_TIMEOUT_MS;
  }
  return DEFAULT_ESCAPE_TIMEOUT_MS2;
}
var ProcessTerminal = class {
  wasRaw = false;
  inputHandler;
  resizeHandler;
  _kittyProtocolActive = false;
  _modifyOtherKeysActive = false;
  keyboardProtocolPushed = false;
  keyboardProtocolNegotiationBuffer = "";
  keyboardProtocolBufferFlushTimer;
  stdinBuffer;
  stdinDataHandler;
  progressInterval;
  writeLogPath = (() => {
    const env = process.env.PI_TUI_WRITE_LOG || "";
    if (!env)
      return "";
    try {
      if (fs.statSync(env).isDirectory()) {
        const now = /* @__PURE__ */ new Date();
        const ts = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}_${String(now.getHours()).padStart(2, "0")}-${String(now.getMinutes()).padStart(2, "0")}-${String(now.getSeconds()).padStart(2, "0")}`;
        return path2.join(env, `tui-${ts}-${process.pid}.log`);
      }
    } catch {
    }
    return env;
  })();
  get kittyProtocolActive() {
    return this._kittyProtocolActive;
  }
  get modifyOtherKeysActive() {
    return this._modifyOtherKeysActive;
  }
  start(onInput, onResize) {
    this.inputHandler = onInput;
    this.resizeHandler = onResize;
    this.wasRaw = process.stdin.isRaw || false;
    if (process.stdin.setRawMode) {
      process.stdin.setRawMode(true);
    }
    process.stdin.setEncoding("utf8");
    process.stdin.resume();
    process.stdout.write("\x1B[?2004h");
    process.stdout.on("resize", this.resizeHandler);
    refreshTerminalDimensions();
    this.enableWindowsVTInput();
    this.queryAndEnableKittyProtocol();
  }
  /**
   * Set up StdinBuffer to split batched input into individual sequences.
   * This ensures components receive single events, making matchesKey/isKeyRelease work correctly.
   *
   * Also watches for Kitty protocol response and enables it when detected.
   * This is done here (after stdinBuffer parsing) rather than on raw stdin
   * to handle the case where the response arrives split across multiple events.
   */
  setupStdinBuffer() {
    this.stdinBuffer = new StdinBuffer({ escapeTimeout: resolveEscapeTimeoutMs() });
    this.stdinBuffer.on("data", (sequence) => {
      const negotiationSequence = this.readKeyboardProtocolNegotiationSequence(sequence);
      if (negotiationSequence === "pending") {
        this.scheduleKeyboardProtocolNegotiationBufferFlush();
        return;
      }
      if (this.handleKeyboardProtocolNegotiationSequence(negotiationSequence)) {
        return;
      }
      this.forwardInputSequence(sequence);
    });
    this.stdinBuffer.on("paste", (content) => {
      if (this.inputHandler) {
        this.inputHandler(`\x1B[200~${content}\x1B[201~`);
      }
    });
    this.stdinDataHandler = (data) => {
      this.stdinBuffer.process(data);
    };
  }
  /**
   * Query terminal for Kitty keyboard protocol support and enable it if available.
   *
   * Kitty's progressive enhancement detection requires requesting the desired
   * flags before querying them. The trailing DA query is a sentinel supported by
   * terminals that do not know Kitty keyboard protocol; receiving DA before a
   * Kitty response enables modifyOtherKeys fallback without a startup timeout.
   *
   * The requested flags are:
   * - 1 = disambiguate escape codes
   * - 2 = report event types (press/repeat/release)
   * - 4 = report alternate keys (shifted key, base layout key)
   */
  queryAndEnableKittyProtocol() {
    this.setupStdinBuffer();
    process.stdin.on("data", this.stdinDataHandler);
    this.keyboardProtocolPushed = true;
    this.clearKeyboardProtocolNegotiationBuffer();
    process.stdout.write(KITTY_KEYBOARD_PROTOCOL_QUERY);
  }
  handleKeyboardProtocolNegotiationSequence(negotiationSequence) {
    if (!negotiationSequence)
      return false;
    this.clearKeyboardProtocolNegotiationBuffer();
    if (negotiationSequence.type === "kitty-flags") {
      if (negotiationSequence.flags !== 0) {
        this.disableModifyOtherKeys();
        if (!this._kittyProtocolActive) {
          this._kittyProtocolActive = true;
          setKittyProtocolActive(true);
        }
      } else {
        this.enableModifyOtherKeys();
      }
      return true;
    }
    if (!this._kittyProtocolActive) {
      this.enableModifyOtherKeys();
    }
    return true;
  }
  readKeyboardProtocolNegotiationSequence(sequence) {
    if (this.keyboardProtocolNegotiationBuffer) {
      const bufferedSequence = this.keyboardProtocolNegotiationBuffer + sequence;
      const negotiationSequence2 = parseKeyboardProtocolNegotiationSequence(bufferedSequence);
      if (negotiationSequence2) {
        this.clearKeyboardProtocolNegotiationBuffer();
        return negotiationSequence2;
      }
      if (isKeyboardProtocolNegotiationSequencePrefix(bufferedSequence)) {
        this.setKeyboardProtocolNegotiationBuffer(bufferedSequence);
        return "pending";
      }
      this.flushKeyboardProtocolNegotiationBufferAsInput();
    }
    const negotiationSequence = parseKeyboardProtocolNegotiationSequence(sequence);
    if (negotiationSequence)
      return negotiationSequence;
    if (isKeyboardProtocolNegotiationSequencePrefix(sequence)) {
      this.setKeyboardProtocolNegotiationBuffer(sequence);
      return "pending";
    }
    return void 0;
  }
  setKeyboardProtocolNegotiationBuffer(sequence) {
    this.clearKeyboardProtocolNegotiationBufferFlushTimer();
    this.keyboardProtocolNegotiationBuffer = sequence;
  }
  clearKeyboardProtocolNegotiationBuffer() {
    this.clearKeyboardProtocolNegotiationBufferFlushTimer();
    this.keyboardProtocolNegotiationBuffer = "";
  }
  flushKeyboardProtocolNegotiationBufferAsInput() {
    if (!this.keyboardProtocolNegotiationBuffer)
      return;
    const sequence = this.keyboardProtocolNegotiationBuffer;
    this.clearKeyboardProtocolNegotiationBuffer();
    this.forwardInputSequence(sequence);
  }
  scheduleKeyboardProtocolNegotiationBufferFlush() {
    if (!this.keyboardProtocolNegotiationBuffer || this.keyboardProtocolBufferFlushTimer)
      return;
    this.keyboardProtocolBufferFlushTimer = setTimeout(() => {
      this.keyboardProtocolBufferFlushTimer = void 0;
      this.flushKeyboardProtocolNegotiationBufferAsInput();
    }, KEYBOARD_PROTOCOL_RESPONSE_FRAGMENT_TIMEOUT_MS);
  }
  clearKeyboardProtocolNegotiationBufferFlushTimer() {
    if (!this.keyboardProtocolBufferFlushTimer)
      return;
    clearTimeout(this.keyboardProtocolBufferFlushTimer);
    this.keyboardProtocolBufferFlushTimer = void 0;
  }
  forwardInputSequence(sequence) {
    if (!this.inputHandler)
      return;
    const shouldDetectNativeShiftEnter = sequence === "\r" && (isAppleTerminalSession() || process.platform === "win32");
    const input = normalizeNativeShiftEnterInput(sequence, shouldDetectNativeShiftEnter, shouldDetectNativeShiftEnter && isNativeModifierPressed("shift"));
    this.inputHandler(input);
  }
  enableModifyOtherKeys() {
    if (this._kittyProtocolActive || this._modifyOtherKeysActive)
      return;
    process.stdout.write("\x1B[>4;2m");
    this._modifyOtherKeysActive = true;
  }
  disableModifyOtherKeys() {
    if (!this._modifyOtherKeysActive)
      return;
    process.stdout.write("\x1B[>4;0m");
    this._modifyOtherKeysActive = false;
  }
  /**
   * On Windows, add ENABLE_VIRTUAL_TERMINAL_INPUT (0x0200) to the stdin
   * console handle so the terminal sends VT sequences for modified keys
   * (e.g. \x1b[Z for Shift+Tab). Without this, libuv's ReadConsoleInputW
   * discards modifier state and Shift+Tab arrives as plain \t.
   */
  enableWindowsVTInput() {
    if (process.platform !== "win32")
      return;
    try {
      const arch = process.arch;
      if (arch !== "x64" && arch !== "arm64")
        return;
      const nativePath = path2.join("native", "win32", "prebuilds", `win32-${arch}`, "win32-console-mode.node");
      for (const modulePath of getNativeModuleCandidates(nativePath)) {
        try {
          const helper = cjsRequire2(modulePath);
          helper.enableVirtualTerminalInput?.();
          return;
        } catch {
        }
      }
    } catch {
    }
  }
  async drainInput(maxMs = 1e3, idleMs = 50) {
    const shouldDisableKittyProtocol = this.keyboardProtocolPushed || this._kittyProtocolActive;
    this.clearKeyboardProtocolNegotiationBuffer();
    if (shouldDisableKittyProtocol) {
      process.stdout.write("\x1B[<u");
      this.keyboardProtocolPushed = false;
      this._kittyProtocolActive = false;
      setKittyProtocolActive(false);
    }
    this.disableModifyOtherKeys();
    const previousHandler = this.inputHandler;
    this.inputHandler = void 0;
    let lastDataTime = Date.now();
    const onData = () => {
      lastDataTime = Date.now();
    };
    process.stdin.on("data", onData);
    const endTime = Date.now() + maxMs;
    try {
      while (true) {
        const now = Date.now();
        const timeLeft = endTime - now;
        if (timeLeft <= 0)
          break;
        if (now - lastDataTime >= idleMs)
          break;
        await new Promise((resolve) => setTimeout(resolve, Math.min(idleMs, timeLeft)));
      }
    } finally {
      process.stdin.removeListener("data", onData);
      this.inputHandler = previousHandler;
    }
  }
  stop() {
    if (this.clearProgressInterval()) {
      process.stdout.write(TERMINAL_PROGRESS_CLEAR_SEQUENCE);
    }
    process.stdout.write("\x1B[?2004l");
    const shouldDisableKittyProtocol = this.keyboardProtocolPushed || this._kittyProtocolActive;
    this.clearKeyboardProtocolNegotiationBuffer();
    if (shouldDisableKittyProtocol) {
      process.stdout.write("\x1B[<u");
      this.keyboardProtocolPushed = false;
      this._kittyProtocolActive = false;
      setKittyProtocolActive(false);
    }
    this.disableModifyOtherKeys();
    if (this.stdinBuffer) {
      this.stdinBuffer.destroy();
      this.stdinBuffer = void 0;
    }
    if (this.stdinDataHandler) {
      process.stdin.removeListener("data", this.stdinDataHandler);
      this.stdinDataHandler = void 0;
    }
    this.inputHandler = void 0;
    if (this.resizeHandler) {
      process.stdout.removeListener("resize", this.resizeHandler);
      this.resizeHandler = void 0;
    }
    process.stdin.pause();
    if (process.stdin.setRawMode) {
      process.stdin.setRawMode(this.wasRaw);
    }
  }
  write(data) {
    process.stdout.write(data);
    if (this.writeLogPath) {
      try {
        fs.appendFileSync(this.writeLogPath, data, { encoding: "utf8" });
      } catch {
      }
    }
  }
  get columns() {
    return process.stdout.columns || Number(process.env.COLUMNS) || 80;
  }
  get rows() {
    return process.stdout.rows || Number(process.env.LINES) || 24;
  }
  moveBy(lines) {
    if (lines > 0) {
      process.stdout.write(`\x1B[${lines}B`);
    } else if (lines < 0) {
      process.stdout.write(`\x1B[${-lines}A`);
    }
  }
  hideCursor() {
    process.stdout.write("\x1B[?25l");
  }
  showCursor() {
    process.stdout.write("\x1B[?25h");
  }
  clearLine() {
    process.stdout.write("\x1B[K");
  }
  clearFromCursor() {
    process.stdout.write("\x1B[J");
  }
  clearScreen() {
    process.stdout.write("\x1B[2J\x1B[H");
  }
  setTitle(title) {
    process.stdout.write(`\x1B]0;${title}\x07`);
  }
  setProgress(active) {
    if (active) {
      process.stdout.write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE);
      if (!this.progressInterval) {
        this.progressInterval = setInterval(() => {
          process.stdout.write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE);
        }, TERMINAL_PROGRESS_KEEPALIVE_MS);
      }
    } else {
      this.clearProgressInterval();
      process.stdout.write(TERMINAL_PROGRESS_CLEAR_SEQUENCE);
    }
  }
  clearProgressInterval() {
    if (!this.progressInterval)
      return false;
    clearInterval(this.progressInterval);
    this.progressInterval = void 0;
    return true;
  }
};

// node_modules/@earendil-works/pi-tui/dist/alt-screen-search.js
var segmenter2 = getGraphemeSegmenter();
var PRINTABLE_ASCII = /^[\x20-\x7e]*$/;
function buildSearchCorpus(lines) {
  const chunks = [];
  const spans = [];
  let textLength = 0;
  let pendingSeparator = false;
  const appendSeparator = () => {
    if (!pendingSeparator)
      return;
    chunks.push(" ");
    textLength += 1;
    pendingSeparator = false;
  };
  for (let row = 0; row < lines.length; row++) {
    const line = stripTerminalSequences(lines[row] ?? "");
    let column = 0;
    if (PRINTABLE_ASCII.test(line)) {
      let index = 0;
      while (index < line.length) {
        if (line.charCodeAt(index) === 32) {
          if (textLength > 0)
            pendingSeparator = true;
          column += 1;
          index += 1;
          continue;
        }
        let end = index + 1;
        while (end < line.length && line.charCodeAt(end) !== 32)
          end += 1;
        appendSeparator();
        const text = line.slice(index, end);
        chunks.push(text);
        spans.push({
          textStart: textLength,
          textEnd: textLength + text.length,
          row,
          startCol: column,
          endCol: column + text.length,
          linearColumns: true
        });
        textLength += text.length;
        column += text.length;
        index = end;
      }
    } else {
      for (const grapheme of segmenter2.segment(line)) {
        const text = grapheme.segment;
        const width = visibleWidth(text);
        if (/^\s+$/u.test(text)) {
          if (textLength > 0)
            pendingSeparator = true;
          column += width;
          continue;
        }
        appendSeparator();
        chunks.push(text);
        spans.push({
          textStart: textLength,
          textEnd: textLength + text.length,
          row,
          startCol: column,
          endCol: column + width,
          linearColumns: false
        });
        textLength += text.length;
        column += width;
      }
    }
    if (textLength > 0)
      pendingSeparator = true;
  }
  return { text: chunks.join(""), spans };
}
function normalizeQuery(query) {
  return query.replace(/\s+/gu, " ").trim();
}
function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function findSearchCorpusMatches(corpus, normalizedQuery) {
  if (!normalizedQuery)
    return [];
  const expression = new RegExp(escapeRegExp(normalizedQuery), "giu");
  const matches = [];
  let spanIndex = 0;
  for (const match of corpus.text.matchAll(expression)) {
    const start = match.index;
    const end = start + match[0].length;
    while (spanIndex < corpus.spans.length && corpus.spans[spanIndex].textEnd <= start)
      spanIndex += 1;
    const segments = [];
    for (let index = spanIndex; index < corpus.spans.length; index++) {
      const span = corpus.spans[index];
      if (span.textStart >= end)
        break;
      if (span.textEnd <= start)
        continue;
      const startCol = span.linearColumns ? span.startCol + Math.max(start, span.textStart) - span.textStart : span.startCol;
      const endCol = span.linearColumns ? span.startCol + Math.min(end, span.textEnd) - span.textStart : span.endCol;
      const previous = segments[segments.length - 1];
      if (previous && previous.row === span.row && startCol <= previous.endCol) {
        previous.endCol = Math.max(previous.endCol, endCol);
      } else {
        segments.push({ row: span.row, startCol, endCol });
      }
    }
    while (spanIndex < corpus.spans.length && corpus.spans[spanIndex].textEnd <= end)
      spanIndex += 1;
    if (segments.length > 0)
      matches.push({ segments });
  }
  return matches;
}
var AltScreenSearchIndex = class {
  sourceLines;
  corpus;
  normalizedQuery;
  matches = [];
  search(lines, query) {
    let sourceChanged = this.sourceLines?.length !== lines.length;
    if (!sourceChanged && this.sourceLines) {
      for (let index = 0; index < lines.length; index++) {
        if (this.sourceLines[index] === lines[index])
          continue;
        sourceChanged = true;
        break;
      }
    }
    if (sourceChanged || !this.corpus) {
      this.sourceLines = Array.from(lines);
      this.corpus = buildSearchCorpus(lines);
    }
    const normalizedQuery = normalizeQuery(query);
    const changed = sourceChanged || normalizedQuery !== this.normalizedQuery;
    if (changed) {
      this.normalizedQuery = normalizedQuery;
      this.matches = findSearchCorpusMatches(this.corpus, normalizedQuery);
    }
    return { matches: this.matches, changed };
  }
};
function getAltScreenSearchMatchKey(match) {
  const first = match.segments[0];
  const last = match.segments[match.segments.length - 1];
  return first && last ? `${first.row}:${first.startCol}:${last.row}:${last.endCol}` : "";
}
var AltScreenSearchComponent = class {
  input = new Input({
    prompt: " ",
    placeholder: "Find in transcript",
    placeholderStyle: (text) => `\x1B[2m${text}\x1B[22m`
  });
  onQueryChange;
  navigationButtonStyle;
  resultCount = 0;
  resultIndex = -1;
  previousButtonStart = -1;
  previousButtonEnd = -1;
  nextButtonStart = -1;
  nextButtonEnd = -1;
  hoveredNavigationDirection;
  _focused = false;
  constructor(onQueryChange, navigationButtonStyle = (text) => text) {
    this.onQueryChange = onQueryChange;
    this.navigationButtonStyle = navigationButtonStyle;
  }
  get focused() {
    return this._focused;
  }
  set focused(value) {
    this._focused = value;
    this.input.focused = value;
  }
  setResult(index, count) {
    this.resultIndex = index;
    this.resultCount = count;
  }
  getNavigationDirectionAt(row, column) {
    if (row !== 2)
      return void 0;
    if (column >= this.previousButtonStart && column < this.previousButtonEnd)
      return -1;
    if (column >= this.nextButtonStart && column < this.nextButtonEnd)
      return 1;
    return void 0;
  }
  setHoveredNavigationDirection(direction) {
    if (direction === this.hoveredNavigationDirection)
      return false;
    this.hoveredNavigationDirection = direction;
    return true;
  }
  handleInput(data) {
    const previous = this.input.getValue();
    this.input.handleInput(data);
    const query = this.input.getValue();
    if (query !== previous)
      this.onQueryChange(query);
  }
  invalidate() {
    this.input.invalidate();
  }
  render(width) {
    const safeWidth = Math.max(1, width);
    const innerWidth = Math.max(0, safeWidth - 2);
    const formatKey = (key) => key ? key.split("+").map((part) => {
      if (process.platform === "darwin" && part.toLowerCase() === "alt")
        return "Option";
      return part.charAt(0).toUpperCase() + part.slice(1);
    }).join("+") : "Unbound";
    const keybindings = getKeybindings();
    const previousKey = formatKey(keybindings.getKeys("tui.altScreen.searchPrevious")[0]);
    const nextKey = formatKey(keybindings.getKeys("tui.altScreen.searchNext")[0]);
    const query = this.input.getValue();
    const result = !query ? "" : this.resultCount === 0 ? "No matches" : `${this.resultIndex + 1}/${this.resultCount}`;
    const resultSpace = Math.max(0, innerWidth - 3);
    const visibleResult = truncateToWidth(result, resultSpace, "");
    const resultText = visibleResult ? `\x1B[2m ${visibleResult} \x1B[22m` : "";
    const inputWidth = Math.max(0, innerWidth - visibleWidth(resultText));
    const inputLine = truncateToWidth(this.input.render(Math.max(1, inputWidth))[0] ?? "", inputWidth, "");
    const inputPadding = " ".repeat(Math.max(0, inputWidth - visibleWidth(inputLine)));
    const content = `${inputLine}${inputPadding}${resultText}`;
    let previousButton = `\u2191 ${previousKey}`;
    let nextButton = `\u2193 ${nextKey}`;
    let separator = " \xB7 ";
    const outerGapWidth = 1;
    const availableControlsWidth = Math.max(0, innerWidth - outerGapWidth * 2 - 1);
    let controlsWidth = visibleWidth(previousButton) + visibleWidth(separator) + visibleWidth(nextButton);
    if (controlsWidth > availableControlsWidth) {
      previousButton = "\u2191";
      nextButton = "\u2193";
      separator = " ";
      controlsWidth = visibleWidth(previousButton) + visibleWidth(separator) + visibleWidth(nextButton);
    }
    const showButtons = controlsWidth <= availableControlsWidth;
    const renderedButtons = showButtons ? this.navigationButtonStyle(previousButton, this.hoveredNavigationDirection === -1) + separator + this.navigationButtonStyle(nextButton, this.hoveredNavigationDirection === 1) : "";
    const outerGapsWidth = showButtons ? outerGapWidth * 2 : 0;
    const rightRuleWidth = renderedButtons && innerWidth > controlsWidth + outerGapsWidth ? 1 : 0;
    const leftRuleWidth = Math.max(0, innerWidth - (showButtons ? controlsWidth : 0) - outerGapsWidth - rightRuleWidth);
    const previousStart = 1 + leftRuleWidth + outerGapWidth;
    this.previousButtonStart = showButtons ? previousStart : -1;
    this.previousButtonEnd = showButtons ? previousStart + visibleWidth(previousButton) : -1;
    this.nextButtonStart = showButtons ? this.previousButtonEnd + visibleWidth(separator) : -1;
    this.nextButtonEnd = showButtons ? this.nextButtonStart + visibleWidth(nextButton) : -1;
    if (safeWidth === 1)
      return ["\u250C", "\u2502", "\u2514"];
    return [
      `\u250C${"\u2500".repeat(innerWidth)}\u2510`,
      `\u2502${content}\u2502`,
      `\u2514${"\u2500".repeat(leftRuleWidth)}${renderedButtons ? " " : ""}${renderedButtons}${renderedButtons ? " " : ""}${"\u2500".repeat(rightRuleWidth)}\u2518`
    ];
  }
};

// node_modules/@earendil-works/pi-tui/dist/components/alt-screen-flash.js
var DEFAULT_DURATION_MS = 1e3;
var AltScreenFlashContainer = class {
  entries = [];
  nextId = 0;
  requestRender;
  constructor(requestRender) {
    this.requestRender = requestRender;
  }
  flash(message, durationMs = DEFAULT_DURATION_MS) {
    const id = this.nextId++;
    const timer = setTimeout(() => {
      const index = this.entries.findIndex((entry) => entry.id === id);
      if (index === -1)
        return;
      this.entries.splice(index, 1);
      this.requestRender();
    }, Math.max(0, durationMs));
    timer.unref();
    this.entries.push({ id, message, timer });
    this.requestRender();
  }
  dispose() {
    for (const entry of this.entries)
      clearTimeout(entry.timer);
    this.entries.length = 0;
  }
  invalidate() {
  }
  render(width) {
    return this.entries.map((entry) => {
      const message = truncateToWidth(` ${entry.message} `, width, "");
      return `\x1B[7m${message}\x1B[27m`;
    });
  }
};

// node_modules/@earendil-works/pi-tui/dist/layout.js
var OSC133_ZONE_PREFIX = /^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+/;
function intersect(a, b2) {
  const x2 = Math.max(a.x, b2.x);
  const y2 = Math.max(a.y, b2.y);
  const right = Math.min(a.x + a.width, b2.x + b2.width);
  const bottom = Math.min(a.y + a.height, b2.y + b2.height);
  return { x: x2, y: y2, width: Math.max(0, right - x2), height: Math.max(0, bottom - y2) };
}
function renderCached(context, component, width) {
  const safeWidth = Math.max(1, Math.floor(width));
  let widths = context.renderCache.get(component);
  if (!widths) {
    widths = /* @__PURE__ */ new Map();
    context.renderCache.set(component, widths);
  }
  let lines = widths.get(safeWidth);
  if (!lines) {
    lines = component.render(safeWidth);
    widths.set(safeWidth, lines);
  }
  return lines;
}
function measureHeight(context, component, width) {
  return renderCached(context, component, width).length;
}
function measureWidth(context, component, width) {
  return renderCached(context, component, width).reduce((max, line) => Math.max(max, visibleWidth(line)), 0);
}
function withParent(box, parent) {
  box.parent = parent;
  return box;
}
function translateBox(box, deltaY) {
  box.rect.y += deltaY;
  for (const child of box.children)
    translateBox(child, deltaY);
}
function updateClips(box, parentClip) {
  box.clip = intersect(parentClip, box.rect);
  for (const child of box.children)
    updateClips(child, box.clip);
}
function layoutComponent(context, component, x2, y2, width, height, clip) {
  const safeWidth = Math.max(1, Math.floor(width));
  const node = getLayoutNode(component);
  if (!node) {
    const lines = renderCached(context, component, safeWidth);
    const allocatedHeight2 = height === void 0 ? lines.length : Math.max(0, Math.floor(height));
    let lineOffset = 0;
    if (lines.length > allocatedHeight2 && allocatedHeight2 > 0) {
      const cursorLine = lines.findIndex((line) => line.includes(CURSOR_MARKER));
      if (cursorLine >= allocatedHeight2)
        lineOffset = cursorLine - allocatedHeight2 + 1;
    }
    return {
      component,
      rect: { x: x2, y: y2, width: safeWidth, height: allocatedHeight2 },
      clip: intersect(clip, { x: x2, y: y2, width: safeWidth, height: allocatedHeight2 }),
      children: [],
      lines,
      lineOffset,
      layer: 0
    };
  }
  if (node.type === "scroll") {
    const previousScrollTop = node.state.scrollTop;
    const contentWidth = node.state.getContentWidth(safeWidth);
    const childBox = layoutComponent(context, node.component, x2, y2 - previousScrollTop, contentWidth, void 0, clip);
    const contentHeight = childBox.rect.height;
    const viewportHeight = height === void 0 ? contentHeight : Math.max(0, Math.floor(height));
    node.state.updateLayout(contentHeight, viewportHeight, context.requestRender);
    translateBox(childBox, previousScrollTop - node.state.scrollTop);
    const scrollView = node.state;
    if (node.state.primary || !context.primaryScrollView)
      context.primaryScrollView = scrollView;
    const rect2 = { x: x2, y: y2, width: safeWidth, height: viewportHeight };
    const childClip = intersect(clip, rect2);
    const box2 = {
      component,
      rect: rect2,
      clip: childClip,
      children: [childBox],
      scrollView,
      scrollContentLines: renderCached(context, node.component, contentWidth),
      layer: 0
    };
    childBox.parent = box2;
    updateClips(childBox, childClip);
    return box2;
  }
  const entries = visibleStackEntries(node.entries, context.viewport);
  const gapTotal = Math.max(0, entries.length - 1) * node.gap;
  if (node.type === "vstack") {
    const intrinsicHeights2 = entries.map((entry) => typeof entry.basis === "number" ? entry.basis : measureHeight(context, entry.component, safeWidth));
    const sizes = allocateStackSizes(entries, intrinsicHeights2, height, node.gap);
    const naturalHeight = sizes.reduce((sum, size) => sum + size, 0) + gapTotal;
    const allocatedHeight2 = height === void 0 ? naturalHeight : Math.max(0, Math.floor(height));
    const rect2 = { x: x2, y: y2, width: safeWidth, height: allocatedHeight2 };
    const box2 = {
      component,
      rect: rect2,
      clip: intersect(clip, rect2),
      children: [],
      layer: 0
    };
    let childY = y2;
    for (let index = 0; index < entries.length; index++) {
      box2.children.push(withParent(layoutComponent(context, entries[index].component, x2, childY, safeWidth, sizes[index], box2.clip), box2));
      childY += sizes[index] + node.gap;
    }
    return box2;
  }
  const intrinsicWidths = entries.map((entry) => typeof entry.basis === "number" ? entry.basis : measureWidth(context, entry.component, safeWidth));
  const widths = allocateStackSizes(entries, intrinsicWidths, safeWidth, node.gap);
  const intrinsicHeights = entries.map((entry, index) => measureHeight(context, entry.component, Math.max(1, widths[index])));
  const allocatedHeight = height === void 0 ? intrinsicHeights.reduce((max, childHeight) => Math.max(max, childHeight), 0) : Math.max(0, height);
  const rect = { x: x2, y: y2, width: safeWidth, height: allocatedHeight };
  const box = {
    component,
    rect,
    clip: intersect(clip, rect),
    children: [],
    layer: 0
  };
  let childX = x2;
  for (let index = 0; index < entries.length; index++) {
    const naturalChildHeight = intrinsicHeights[index];
    const childHeight = node.align === "stretch" ? allocatedHeight : Math.min(allocatedHeight, naturalChildHeight);
    let childY = y2;
    if (node.align === "center")
      childY += Math.floor((allocatedHeight - childHeight) / 2);
    else if (node.align === "end")
      childY += allocatedHeight - childHeight;
    const childWidth = widths[index];
    if (childWidth === 0) {
      box.children.push({
        component: entries[index].component,
        rect: { x: childX, y: childY, width: 0, height: childHeight },
        clip: { x: childX, y: childY, width: 0, height: 0 },
        children: [],
        parent: box,
        layer: 0
      });
    } else {
      box.children.push(withParent(layoutComponent(context, entries[index].component, childX, childY, childWidth, childHeight, box.clip), box));
    }
    childX += childWidth + node.gap;
  }
  return box;
}
function replaceScrollbarCell(line, column, totalWidth, replacement, preserveTargetBackground) {
  if (isImageLine(line))
    return line;
  const graphemeRange = getGraphemeCellRange(line, column);
  const start = graphemeRange?.start ?? column;
  const end = graphemeRange?.end ?? column + 1;
  const before = sliceByColumn(line, 0, start, true);
  const target = sliceByColumn(line, start, end - start, true);
  const after = sliceByColumn(line, end, Math.max(0, totalWidth - end), true);
  let targetPrefix = "";
  let targetIndex = 0;
  while (targetIndex < target.length) {
    const ansi = extractAnsiCode(target, targetIndex);
    if (!ansi)
      break;
    targetPrefix += ansi.code;
    targetIndex += ansi.length;
  }
  const beforePadding = " ".repeat(Math.max(0, start - visibleWidth(before)));
  const cellPaddingBefore = " ".repeat(Math.max(0, column - start));
  const cellPaddingAfter = " ".repeat(Math.max(0, end - column - 1));
  const targetStyle = `\x1B[0m\x1B]8;;\x07${preserveTargetBackground ? getActiveBackgroundAnsi(targetPrefix) : ""}`;
  return `${before}${beforePadding}${targetStyle}${cellPaddingBefore}${replacement}${cellPaddingAfter}${after}`;
}
function getScrollbarGeometry(box, includeHiddenAuto = false) {
  if (!box.scrollView || box.rect.width <= 0 || box.rect.height <= 0)
    return void 0;
  const contentHeight = box.children[0]?.rect.height ?? box.scrollContentLines?.length ?? 0;
  const trackHeight = box.rect.height;
  const canRevealHiddenAuto = includeHiddenAuto && box.scrollView.scrollbar === "auto" && contentHeight > trackHeight;
  if (!box.scrollView.isScrollbarVisible && !canRevealHiddenAuto)
    return void 0;
  const minThumbHeight = Math.min(2, trackHeight);
  const thumbHeight = Math.max(minThumbHeight, Math.min(trackHeight, Math.round(trackHeight * trackHeight / contentHeight)));
  const maxScrollTop = Math.max(0, contentHeight - trackHeight);
  const maxThumbTop = trackHeight - thumbHeight;
  const thumbOffset = maxScrollTop === 0 ? 0 : Math.round(box.scrollView.scrollTop / maxScrollTop * maxThumbTop);
  const column = box.rect.x + box.rect.width - 1;
  if (column < box.clip.x || column >= box.clip.x + box.clip.width)
    return void 0;
  return {
    column,
    trackTop: box.rect.y,
    trackHeight,
    thumbTop: box.rect.y + thumbOffset,
    thumbHeight,
    maxScrollTop
  };
}
function paintScrollbar(box, screen, totalWidth) {
  const geometry = getScrollbarGeometry(box);
  if (!geometry || !box.scrollView)
    return;
  for (let offset = 0; offset < geometry.trackHeight; offset++) {
    const row = geometry.trackTop + offset;
    if (row < box.clip.y || row >= box.clip.y + box.clip.height || row < 0 || row >= screen.length)
      continue;
    const isThumb = row >= geometry.thumbTop && row < geometry.thumbTop + geometry.thumbHeight;
    const replacement = isThumb ? box.scrollView.scrollbarThumbStyle(box.scrollView.isScrollbarActive ? "\u2588" : "\u2503") : box.scrollView.scrollbarTrackStyle("\u2502");
    screen[row] = replaceScrollbarCell(screen[row] ?? "", geometry.column, totalWidth, replacement, box.scrollView.scrollbar !== "always");
  }
}
function paintBox(box, screen, totalWidth) {
  if (box.lines) {
    const offset = box.lineOffset ?? 0;
    const firstRow = Math.max(box.rect.y, box.clip.y, 0);
    const lastRow = Math.min(box.rect.y + box.rect.height, box.clip.y + box.clip.height, screen.length);
    for (let row = firstRow; row < lastRow; row++) {
      const sourceLine = box.lines[offset + row - box.rect.y];
      if (sourceLine === void 0)
        continue;
      let line = sourceLine.replace(OSC133_ZONE_PREFIX, "");
      const imageMetadata = getKittyImageMetadata(line);
      if (imageMetadata) {
        const clipBottom = Math.min(screen.length, box.clip.y + box.clip.height);
        const visibleRows = Math.min(imageMetadata.rows, clipBottom - row);
        if (visibleRows < imageMetadata.rows)
          line = cropKittyImageLine(line, 0, visibleRows);
      }
      if (box.rect.x === 0 && box.rect.width >= totalWidth && (isImageLine(line) || !screen[row])) {
        screen[row] = line;
      } else {
        screen[row] = compositeTuiLine(screen[row] ?? "", line, box.rect.x, box.rect.width, totalWidth);
      }
    }
  }
  for (const child of box.children)
    paintBox(child, screen, totalWidth);
  if (box.scrollView && box.scrollContentLines && box.scrollView.scrollTop > 0 && box.rect.height > 0) {
    for (let imageRow = box.scrollView.scrollTop - 1; imageRow >= 0; imageRow--) {
      const imageLine = box.scrollContentLines[imageRow] ?? "";
      const metadata = getKittyImageMetadata(imageLine);
      if (metadata) {
        const hiddenRows = box.scrollView.scrollTop - imageRow;
        if (hiddenRows < metadata.rows) {
          const visibleRows = Math.min(box.rect.height, metadata.rows - hiddenRows);
          const cropped = cropKittyImageLine(imageLine, hiddenRows, visibleRows);
          if (box.rect.x === 0 && box.rect.width >= totalWidth)
            screen[box.rect.y] = cropped;
        }
        break;
      }
      if (imageLine !== "")
        break;
    }
  }
  paintScrollbar(box, screen, totalWidth);
}
function renderLayoutFrame(root, width, height, requestRender) {
  const safeWidth = Math.max(1, Math.floor(width));
  const safeHeight = Math.max(1, Math.floor(height));
  const context = {
    viewport: { width: safeWidth, height: safeHeight },
    renderCache: /* @__PURE__ */ new Map(),
    requestRender,
    primaryScrollView: void 0
  };
  const rootBox = layoutComponent(context, root, 0, 0, safeWidth, safeHeight, {
    x: 0,
    y: 0,
    width: safeWidth,
    height: safeHeight
  });
  const lines = Array.from({ length: safeHeight }, () => "");
  paintBox(rootBox, lines, safeWidth);
  return {
    root: rootBox,
    width: safeWidth,
    height: safeHeight,
    lines,
    ...context.primaryScrollView === void 0 ? {} : { primaryScrollView: context.primaryScrollView }
  };
}
function containsPoint(rect, x2, y2) {
  return x2 >= rect.x && x2 < rect.x + rect.width && y2 >= rect.y && y2 < rect.y + rect.height;
}
function getLayoutBoxesAt(frame, x2, y2) {
  const result = [];
  const visit = (box, depth) => {
    if (!containsPoint(box.clip, x2, y2))
      return;
    result.push({ box, depth });
    for (const child of box.children)
      visit(child, depth + 1);
  };
  visit(frame.root, 0);
  result.sort((a, b2) => b2.box.layer - a.box.layer || b2.depth - a.depth);
  return result.map(({ box }) => box);
}
function getScrollViewBox(frame, scrollView) {
  const visit = (box) => {
    if (box.scrollView === scrollView)
      return box;
    for (const child of box.children) {
      const match = visit(child);
      if (match)
        return match;
    }
    return void 0;
  };
  return visit(frame.root);
}
function getScrollViewsAt(frame, x2, y2) {
  const result = [];
  const visit = (box, depth) => {
    if (!containsPoint(box.clip, x2, y2))
      return;
    if (box.scrollView && containsPoint(box.rect, x2, y2))
      result.push({ scrollView: box.scrollView, depth });
    for (const child of box.children)
      visit(child, depth + 1);
  };
  visit(frame.root, 0);
  result.sort((a, b2) => b2.depth - a.depth);
  return result.map((entry) => entry.scrollView);
}

// node_modules/@earendil-works/pi-tui/dist/tui-alt-screen.js
var ENTER_ALT_SCREEN = "\x1B[?1049h";
var EXIT_ALT_SCREEN = "\x1B[?1049l";
var DISABLE_AUTOWRAP = "\x1B[?7l";
var ENABLE_AUTOWRAP = "\x1B[?7h";
var ENABLE_BUTTON_MOTION_MOUSE = "\x1B[?1000h\x1B[?1002h\x1B[?1004h\x1B[?1006h";
var ENABLE_ALL_MOTION_MOUSE = "\x1B[?1000h\x1B[?1002h\x1B[?1003h\x1B[?1004h\x1B[?1006h";
var DISABLE_MOUSE = "\x1B[?1006l\x1B[?1004l\x1B[?1003l\x1B[?1002l\x1B[?1000l";
var FOCUS_IN = "\x1B[I";
var FOCUS_OUT = "\x1B[O";
var BEGIN_SYNCHRONIZED_OUTPUT = "\x1B[?2026h";
var END_SYNCHRONIZED_OUTPUT = "\x1B[?2026l";
var OSC133_ZONE_PREFIX2 = /^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+/;
var OSC133_PROMPT_START = /^\x1b\]133;A(?:\x07|\x1b\\)/;
var PAGE_SCROLL_OVERLAP = 4;
var MAX_CACHED_OFFSCREEN_KITTY_IMAGES = 16;
var MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES = 32 * 1024 * 1024;
var MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES = 64 * 1024 * 1024;
var DOUBLE_CLICK_INTERVAL_MS = 500;
var TERMINAL_WORD_SELECTION_JOINERS = /* @__PURE__ */ new Set(["/", "-"]);
var wordSegmenter4 = getWordSegmenter();
var TuiAltScreen = class extends TuiBase {
  mode = "fullscreen";
  [VIEWPORT_TUI] = true;
  previousScreen = [];
  lastDocument = [];
  previousScreenWidth = 0;
  previousScreenHeight = 0;
  layoutRoot;
  currentLayout;
  implicitDocument;
  implicitScrollView;
  flashes;
  altScreenActive = false;
  imageProtocol = null;
  savedCapabilities;
  uploadedKittyImages = /* @__PURE__ */ new Map();
  selectionAnchor;
  selectionFocus;
  selectionGranularity = "character";
  selectionInitialRange;
  lastClick;
  selectionDragPointer;
  selectionAutoScrollDirection = 0;
  selectionAutoScrollTimer;
  selectionPressActive = false;
  scrollbarDrag;
  scrollbarHover;
  scrollToEndIndicatorRect;
  activeSearch;
  pressedUrl;
  selectionDragged = false;
  mouseCapture;
  mousePressTarget;
  mousePressPoint;
  mousePressMoved = false;
  lastComponentClick;
  wheelScrollLines;
  mouseEnabled;
  searchMatchStyle;
  searchCurrentMatchStyle;
  searchNavigationButtonStyle;
  scrollToEndIndicator;
  openUrl;
  onRightClickPaste;
  copyOnSelect;
  copySelection;
  constructor(terminal, showHardwareCursor, logDirectory, options = {}) {
    super(terminal, showHardwareCursor, logDirectory);
    this.implicitDocument = {
      render: (width) => super.render(width),
      handleMouse: (event) => super.handleMouse(event),
      invalidate: () => {
        for (const child of this.children)
          child.invalidate();
      }
    };
    this.implicitScrollView = new ScrollView(this.implicitDocument, { follow: "end", primary: true });
    this.flashes = new AltScreenFlashContainer(() => this.requestRender());
    this.wheelScrollLines = Math.max(1, Math.floor(options.wheelScrollLines ?? 1));
    this.mouseEnabled = options.mouse ?? true;
    this.searchMatchStyle = options.searchMatchStyle ?? ((text) => `\x1B[4m${text}\x1B[24m`);
    this.searchCurrentMatchStyle = options.searchCurrentMatchStyle ?? ((text) => `\x1B[1;7m${text}\x1B[22;27m`);
    this.searchNavigationButtonStyle = options.searchNavigationButtonStyle ?? ((text) => text);
    this.scrollToEndIndicator = options.scrollToEndIndicator;
    this.openUrl = options.openUrl;
    this.onRightClickPaste = options.onRightClickPaste;
    this.copyOnSelect = options.copyOnSelect ?? true;
    this.copySelection = options.copySelection;
    this.addInputListener((data) => this.handleViewportInput(data));
  }
  get viewportTop() {
    return this.getPrimaryScrollView().scrollTop;
  }
  get isFollowingOutput() {
    return this.getPrimaryScrollView().isFollowingEnd;
  }
  getCopyOnSelect() {
    return this.copyOnSelect;
  }
  setCopyOnSelect(enabled) {
    this.copyOnSelect = enabled;
  }
  /** Whether the fullscreen viewport has a non-empty active text selection. */
  hasActiveSelection() {
    return this.getActiveSelectionText() !== void 0;
  }
  /** Copy the active fullscreen text selection, if any, using the configured selection clipboard path. */
  async copyActiveSelectionToClipboard() {
    const text = this.getActiveSelectionText();
    if (!text)
      return false;
    return this.copyTextToClipboard(text);
  }
  setLayoutRoot(component) {
    if (this.layoutRoot === component)
      return;
    this.layoutRoot = component;
    this.currentLayout = void 0;
    this.requestRender();
  }
  render(width) {
    return this.layoutRoot?.render(width) ?? super.render(width);
  }
  getMountedRoots() {
    return this.layoutRoot ? [this.layoutRoot] : this.children;
  }
  getPrimaryScrollView() {
    return this.currentLayout?.primaryScrollView ?? this.implicitScrollView;
  }
  beforeTerminalStart() {
    this.stopSelectionAutoScroll();
    this.selectionPressActive = false;
    this.stopScrollbarHover();
    this.stopScrollbarDrag();
    this.flashes.dispose();
    this.altScreenActive = true;
    const capabilities = getCapabilities();
    this.imageProtocol = capabilities.images;
    this.uploadedKittyImages.clear();
    if (capabilities.images === "iterm2") {
      this.savedCapabilities = capabilities;
      setCapabilities({ ...capabilities, images: null });
      this.invalidate();
    }
    this.lastDocument = [];
    this.selectionAnchor = void 0;
    this.selectionFocus = void 0;
    this.selectionGranularity = "character";
    this.selectionInitialRange = void 0;
    this.lastClick = void 0;
    this.pressedUrl = void 0;
    this.selectionDragged = false;
    this.clearComponentMouseGesture();
    this.lastComponentClick = void 0;
    this.resetRenderState();
    const term = process.env.TERM?.toLowerCase() ?? "";
    const mouseSequence = process.env.TMUX !== void 0 || process.env.ZELLIJ !== void 0 || process.env.STY !== void 0 || term.startsWith("tmux") || term.startsWith("screen") ? ENABLE_BUTTON_MOTION_MOUSE : ENABLE_ALL_MOTION_MOUSE;
    this.terminal.write(`${ENTER_ALT_SCREEN}${DISABLE_AUTOWRAP}${this.mouseEnabled ? mouseSequence : ""}\x1B[2J\x1B[H\x1B[?25l`);
  }
  beforeTerminalStop(_options) {
    this.closeSearch();
    this.stopSelectionAutoScroll();
    this.selectionPressActive = false;
    this.stopScrollbarHover();
    this.stopScrollbarDrag();
    this.clearComponentMouseGesture();
    this.flashes.dispose();
    if (!this.altScreenActive)
      return;
    this.terminal.write(`${BEGIN_SYNCHRONIZED_OUTPUT}${this.deleteKittyImages()}${this.mouseEnabled ? DISABLE_MOUSE : ""}${ENABLE_AUTOWRAP}${END_SYNCHRONIZED_OUTPUT}`);
    this.uploadedKittyImages.clear();
  }
  afterTerminalStop(options) {
    if (!this.altScreenActive)
      return;
    this.altScreenActive = false;
    if (options.preserveScreen) {
      this.terminal.write(`${BEGIN_SYNCHRONIZED_OUTPUT}${EXIT_ALT_SCREEN}\x1B[?25h${END_SYNCHRONIZED_OUTPUT}`);
    } else {
      const width = Math.max(1, this.terminal.columns);
      const documentLines = this.render(width).map((line) => line.replace(OSC133_ZONE_PREFIX2, ""));
      this.lastDocument = this.applyLineResets(documentLines.map((line) => line.replaceAll(CURSOR_MARKER, ""))).map((line) => isImageLine(line) || visibleWidth(line) <= width ? line : sliceByColumn(line, 0, width, true));
      let buffer = `${BEGIN_SYNCHRONIZED_OUTPUT}${EXIT_ALT_SCREEN}${DISABLE_AUTOWRAP}`;
      for (let row = 0; row < this.lastDocument.length; row++) {
        if (row > 0)
          buffer += "\r\n";
        buffer += `\r\x1B[2K${this.lastDocument[row] ?? ""}`;
      }
      buffer += `\x1B[0m${ENABLE_AUTOWRAP}\r
\x1B[?25h${END_SYNCHRONIZED_OUTPUT}`;
      this.terminal.write(buffer);
    }
    if (this.savedCapabilities) {
      setCapabilities(this.savedCapabilities);
      this.savedCapabilities = void 0;
    }
  }
  deleteKittyImages() {
    return this.imageProtocol === "kitty" ? deleteAllKittyImages() : "";
  }
  prepareKittyScreen(screen) {
    const visibleImageIds = /* @__PURE__ */ new Set();
    const lines = screen.map((line) => {
      const placement = getKittyImagePlacement(line);
      if (!placement)
        return line;
      visibleImageIds.add(placement.imageId);
      const cachedImage = this.uploadedKittyImages.get(placement.imageId);
      const nextCachedImage = {
        transmissionGeneration: placement.transmissionGeneration,
        transmissionBytes: placement.transmissionBytes,
        estimatedDecodedBytes: placement.estimatedDecodedBytes
      };
      if (cachedImage)
        this.uploadedKittyImages.delete(placement.imageId);
      this.uploadedKittyImages.set(placement.imageId, nextCachedImage);
      return cachedImage?.transmissionGeneration === placement.transmissionGeneration ? placement.replacementLine : line;
    });
    let cachedOffscreenImageCount = 0;
    let cachedOffscreenTransmissionBytes = 0;
    let cachedOffscreenDecodedBytes = 0;
    for (const [imageId, cachedImage] of this.uploadedKittyImages) {
      if (visibleImageIds.has(imageId))
        continue;
      cachedOffscreenImageCount += 1;
      cachedOffscreenTransmissionBytes += cachedImage.transmissionBytes;
      cachedOffscreenDecodedBytes += cachedImage.estimatedDecodedBytes;
    }
    let evictedImageDeletion = "";
    for (const [imageId, cachedImage] of this.uploadedKittyImages) {
      if (cachedOffscreenImageCount <= MAX_CACHED_OFFSCREEN_KITTY_IMAGES && cachedOffscreenTransmissionBytes <= MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES && cachedOffscreenDecodedBytes <= MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES) {
        break;
      }
      if (visibleImageIds.has(imageId))
        continue;
      evictedImageDeletion += deleteKittyImage(imageId);
      this.uploadedKittyImages.delete(imageId);
      cachedOffscreenImageCount -= 1;
      cachedOffscreenTransmissionBytes -= cachedImage.transmissionBytes;
      cachedOffscreenDecodedBytes -= cachedImage.estimatedDecodedBytes;
    }
    return { lines, evictedImageDeletion };
  }
  resetRenderState() {
    this.previousScreen = [];
    this.previousScreenWidth = 0;
    this.previousScreenHeight = 0;
    this.currentLayout = void 0;
  }
  scrollBy(lines) {
    this.getPrimaryScrollView().scrollBy(lines);
    this.requestRender();
  }
  scrollToTop() {
    this.getPrimaryScrollView().scrollToStart();
    this.requestRender();
  }
  scrollToBottom() {
    this.getPrimaryScrollView().scrollToEnd();
    this.requestRender();
  }
  scrollToPrompt(direction) {
    if (!this.currentLayout)
      return;
    const scrollView = this.getPrimaryScrollView();
    const lines = getScrollViewBox(this.currentLayout, scrollView)?.scrollContentLines;
    if (!lines)
      return;
    for (let row = scrollView.scrollTop + direction; row >= 0 && row < lines.length; row += direction) {
      if (!OSC133_PROMPT_START.test(lines[row] ?? ""))
        continue;
      scrollView.scrollTo(row);
      this.requestRender();
      return;
    }
  }
  toggleSearch() {
    if (this.activeSearch) {
      this.closeSearch();
      return;
    }
    const component = new AltScreenSearchComponent((query) => this.updateSearchQuery(query), this.searchNavigationButtonStyle);
    const search = {
      component,
      index: new AltScreenSearchIndex(),
      query: "",
      matches: [],
      selectedIndex: -1,
      anchorRow: this.getPrimaryScrollView().scrollTop,
      selectionMode: "query"
    };
    this.activeSearch = search;
    search.overlay = this.showOverlay(component, {
      anchor: "top-right",
      width: "40%",
      minWidth: 32,
      margin: 1
    });
  }
  closeSearch() {
    const search = this.activeSearch;
    if (!search)
      return;
    this.activeSearch = void 0;
    search.overlay?.hide();
    this.requestRender();
  }
  updateSearchQuery(query) {
    const search = this.activeSearch;
    if (!search || query === search.query)
      return;
    const selected = search.matches[search.selectedIndex];
    search.anchorRow = selected?.segments[0]?.row ?? this.getPrimaryScrollView().scrollTop;
    search.query = query;
    search.selectionMode = "query";
    search.component.setResult(-1, 0);
    this.requestRender();
  }
  navigateSearch(direction) {
    const search = this.activeSearch;
    if (!search?.query)
      return;
    search.selectionMode = direction < 0 ? "previous" : "next";
    this.requestRender();
  }
  getSearchNavigationDirectionAt(x2, y2) {
    const search = this.activeSearch;
    const bounds = search?.overlay?.getBounds();
    if (!search || !bounds)
      return void 0;
    if (x2 < bounds.col || x2 >= bounds.col + bounds.width || y2 < bounds.row || y2 >= bounds.row + bounds.height) {
      return void 0;
    }
    return search.component.getNavigationDirectionAt(y2 - bounds.row, x2 - bounds.col);
  }
  handleSearchMouseEvent(event) {
    const search = this.activeSearch;
    if (!search)
      return false;
    const direction = this.getSearchNavigationDirectionAt(event.x, event.y);
    if (search.component.setHoveredNavigationDirection(direction))
      this.requestRender();
    if (direction === void 0 || event.release || (event.button & 32) !== 0 || (event.button & 3) !== 0) {
      return false;
    }
    this.navigateSearch(direction);
    return true;
  }
  refreshSearch(layout) {
    const search = this.activeSearch;
    if (!search)
      return false;
    const scrollView = layout.primaryScrollView ?? this.implicitScrollView;
    const box = getScrollViewBox(layout, scrollView);
    const lines = box?.scrollContentLines;
    if (!lines || !search.query.trim()) {
      search.matches = [];
      search.selectedIndex = -1;
      search.selectedKey = void 0;
      search.selectionMode = "retain";
      search.component.setResult(-1, 0);
      return false;
    }
    const shouldRevealSelection = search.selectionMode !== "retain";
    const result = search.index.search(lines, search.query);
    const matches = result.matches;
    search.matches = matches;
    if (!result.changed && search.selectionMode === "retain")
      return false;
    const exactIndex = result.changed ? search.selectedKey ? matches.findIndex((match) => getAltScreenSearchMatchKey(match) === search.selectedKey) : -1 : search.selectedIndex;
    let selectedIndex = -1;
    if (matches.length > 0) {
      if (search.selectionMode === "query") {
        let low = 0;
        let high = matches.length;
        while (low < high) {
          const middle = low + Math.floor((high - low) / 2);
          if ((matches[middle].segments[0]?.row ?? 0) < search.anchorRow)
            low = middle + 1;
          else
            high = middle;
        }
        selectedIndex = low < matches.length ? low : 0;
      } else if (search.selectionMode === "next") {
        const baseIndex = exactIndex >= 0 ? exactIndex : Math.min(search.selectedIndex, matches.length - 1);
        selectedIndex = baseIndex < 0 ? 0 : (baseIndex + 1) % matches.length;
      } else if (search.selectionMode === "previous") {
        const baseIndex = exactIndex >= 0 ? exactIndex : Math.min(search.selectedIndex, matches.length - 1);
        selectedIndex = baseIndex < 0 ? matches.length - 1 : (baseIndex - 1 + matches.length) % matches.length;
      } else {
        selectedIndex = exactIndex >= 0 ? exactIndex : Math.min(Math.max(0, search.selectedIndex), matches.length - 1);
      }
    }
    search.selectedIndex = selectedIndex;
    search.selectedKey = selectedIndex >= 0 ? getAltScreenSearchMatchKey(matches[selectedIndex]) : void 0;
    search.selectionMode = "retain";
    search.component.setResult(selectedIndex, matches.length);
    if (!shouldRevealSelection)
      return false;
    const selected = matches[selectedIndex];
    const firstSegment = selected?.segments[0];
    const lastSegment = selected?.segments[selected.segments.length - 1];
    if (!box || !firstSegment || !lastSegment || scrollView.viewportHeight <= 0)
      return false;
    const before = scrollView.scrollTop;
    const visibleBottom = before + scrollView.viewportHeight - 1;
    let target = before;
    if (firstSegment.row < before || lastSegment.row > visibleBottom) {
      target = firstSegment.row - Math.floor(scrollView.viewportHeight / 3);
    }
    scrollView.scrollTo(target, { disableFollow: true });
    return scrollView.scrollTop !== before;
  }
  /** Show a transient message in the alternate-screen flash stack. */
  flash(message, durationMs) {
    this.flashes.flash(message, durationMs);
  }
  shouldDeferViewportInputToOverlay() {
    return this.isOverlayFocused() && this.activeSearch?.overlay?.isFocused() !== true;
  }
  clearComponentMouseGesture() {
    this.mouseCapture = void 0;
    this.mousePressTarget = void 0;
    this.mousePressPoint = void 0;
    this.mousePressMoved = false;
  }
  handleViewportInput(data) {
    if (data === FOCUS_OUT) {
      const hadActiveSelection = this.selectionPressActive;
      const hadNonEmptyActiveSelection = hadActiveSelection && this.getSelectionBounds() !== void 0;
      this.selectionPressActive = false;
      this.stopSelectionAutoScroll();
      this.stopScrollbarHover();
      if (this.activeSearch?.component.setHoveredNavigationDirection(void 0))
        this.requestRender();
      this.stopScrollbarDrag();
      this.pressedUrl = void 0;
      this.selectionDragged = false;
      this.clearComponentMouseGesture();
      this.lastComponentClick = void 0;
      if (hadActiveSelection) {
        this.selectionAnchor = void 0;
        this.selectionFocus = void 0;
        this.selectionGranularity = "character";
        this.selectionInitialRange = void 0;
        if (hadNonEmptyActiveSelection)
          this.requestRender();
      }
      this.lastClick = void 0;
      return { consume: true };
    }
    if (data === FOCUS_IN)
      return { consume: true };
    const wheelEvent = this.parseWheelEvent(data);
    if (wheelEvent) {
      const event = this.createMouseEvent("wheel", wheelEvent.button, wheelEvent.x, wheelEvent.y, {
        wheelDelta: wheelEvent.direction * this.wheelScrollLines
      });
      const overlay = this.dispatchMouseToOverlay(event);
      const result = overlay.result ?? (overlay.hit ? void 0 : this.dispatchMouseToLayout(event));
      if (result) {
        if (this.applyMouseDispatchResult(event, result))
          this.requestRender();
        return { consume: true };
      }
      if (this.shouldDeferViewportInputToOverlay())
        return void 0;
      this.routeWheel(wheelEvent);
      return { consume: true };
    }
    const mouseEvent = this.parseSgrMouseEvent(data);
    if (mouseEvent) {
      this.handleMouseEvent(mouseEvent);
      return { consume: true };
    }
    if (this.isMouseSequence(data))
      return { consume: true };
    const keybindings = getKeybindings();
    const isRelease = isKeyRelease(data);
    if (keybindings.matches(data, "tui.altScreen.search")) {
      if (!isRelease)
        this.toggleSearch();
      return { consume: true };
    }
    if (this.activeSearch?.overlay?.isFocused()) {
      if (keybindings.matches(data, "tui.altScreen.searchNext")) {
        if (!isRelease)
          this.navigateSearch(1);
        return { consume: true };
      }
      if (keybindings.matches(data, "tui.altScreen.searchPrevious")) {
        if (!isRelease)
          this.navigateSearch(-1);
        return { consume: true };
      }
      if (keybindings.matches(data, "tui.altScreen.searchClose")) {
        if (!isRelease)
          this.closeSearch();
        return { consume: true };
      }
    }
    if (this.shouldDeferViewportInputToOverlay())
      return void 0;
    if (keybindings.matches(data, "tui.altScreen.pageUp")) {
      if (!isRelease) {
        this.scrollBy(-Math.max(1, this.getPrimaryScrollView().viewportHeight - PAGE_SCROLL_OVERLAP));
      }
      return { consume: true };
    }
    if (keybindings.matches(data, "tui.altScreen.pageDown")) {
      if (!isRelease) {
        this.scrollBy(Math.max(1, this.getPrimaryScrollView().viewportHeight - PAGE_SCROLL_OVERLAP));
      }
      return { consume: true };
    }
    if (keybindings.matches(data, "tui.altScreen.halfPageUp")) {
      if (!isRelease)
        this.scrollBy(-Math.max(1, Math.floor(this.getPrimaryScrollView().viewportHeight / 2)));
      return { consume: true };
    }
    if (keybindings.matches(data, "tui.altScreen.halfPageDown")) {
      if (!isRelease)
        this.scrollBy(Math.max(1, Math.floor(this.getPrimaryScrollView().viewportHeight / 2)));
      return { consume: true };
    }
    if (keybindings.matches(data, "tui.altScreen.lineUp")) {
      if (!isRelease)
        this.scrollBy(-1);
      return { consume: true };
    }
    if (keybindings.matches(data, "tui.altScreen.lineDown")) {
      if (!isRelease)
        this.scrollBy(1);
      return { consume: true };
    }
    if (keybindings.matches(data, "tui.altScreen.previousPrompt")) {
      if (!isRelease)
        this.scrollToPrompt(-1);
      return { consume: true };
    }
    if (keybindings.matches(data, "tui.altScreen.nextPrompt")) {
      if (!isRelease)
        this.scrollToPrompt(1);
      return { consume: true };
    }
    if (keybindings.matches(data, "tui.altScreen.top")) {
      if (!isRelease)
        this.scrollToTop();
      return { consume: true };
    }
    if (keybindings.matches(data, "tui.altScreen.bottom")) {
      if (!isRelease)
        this.scrollToBottom();
      return { consume: true };
    }
    return void 0;
  }
  decodeMouseButton(button) {
    switch (button & 3) {
      case 0:
        return "left";
      case 1:
        return "middle";
      case 2:
        return "right";
      default:
        return "none";
    }
  }
  createMouseEvent(type, button, x2, y2, extra = {}) {
    return {
      type,
      button: type === "wheel" ? "none" : this.decodeMouseButton(button),
      x: x2,
      y: y2,
      screenX: x2,
      screenY: y2,
      width: Math.max(1, this.terminal.columns),
      height: Math.max(1, this.terminal.rows),
      shift: (button & 4) !== 0,
      alt: (button & 8) !== 0,
      ctrl: (button & 16) !== 0,
      ...extra.wheelDelta === void 0 ? {} : { wheelDelta: extra.wheelDelta },
      ...extra.clickCount === void 0 ? {} : { clickCount: extra.clickCount }
    };
  }
  dispatchMouseToLayout(event) {
    if (!this.currentLayout)
      return void 0;
    const visited = /* @__PURE__ */ new Set();
    const boxes = getLayoutBoxesAt(this.currentLayout, event.screenX, event.screenY);
    for (const box of boxes) {
      if (visited.has(box.component))
        continue;
      if (getLayoutNode(box.component) && box.component.handleMouse === Container.prototype.handleMouse)
        continue;
      visited.add(box.component);
      const result = dispatchMouseEvent(box.component, {
        ...event,
        x: event.screenX - box.rect.x,
        y: event.screenY - box.rect.y,
        width: box.rect.width,
        height: box.rect.height
      });
      if (result)
        return result;
    }
    return void 0;
  }
  applyMouseDispatchResult(event, result) {
    const focusTarget = this.resolveMouseFocusTarget(result.focusTarget ?? result.target.component);
    const focusChanged = result.focus === true && this.getFocusedComponent() !== focusTarget;
    if (result.focus)
      this.setFocus(focusTarget);
    if (result.capture)
      this.mouseCapture = result.target;
    return result.render ?? (focusChanged || event.type === "press" || event.type === "click" || event.type === "drag" || event.type === "wheel");
  }
  dispatchMouseToTarget(event, target) {
    return dispatchMouseEvent(target.component, retargetMouseEvent(event, target));
  }
  getComponentClickCount(target, x2, y2) {
    const now = Date.now();
    const previous = this.lastComponentClick;
    const count = previous && now - previous.timestamp <= DOUBLE_CLICK_INTERVAL_MS && previous.component === target.component && previous.x === x2 && previous.y === y2 ? previous.count % 3 + 1 : 1;
    this.lastComponentClick = { timestamp: now, count, component: target.component, x: x2, y: y2 };
    return count;
  }
  clearTextSelection() {
    this.stopSelectionAutoScroll();
    this.selectionPressActive = false;
    this.selectionAnchor = void 0;
    this.selectionFocus = void 0;
    this.selectionGranularity = "character";
    this.selectionInitialRange = void 0;
    this.pressedUrl = void 0;
    this.selectionDragged = false;
  }
  handleMouseEvent(raw) {
    const isMotion = (raw.button & 32) !== 0;
    const type = raw.release ? "release" : isMotion ? this.decodeMouseButton(raw.button) === "none" ? "move" : "drag" : "press";
    const event = this.createMouseEvent(type, raw.button, raw.x, raw.y);
    if (this.mouseCapture || this.mousePressTarget) {
      const target = this.mouseCapture ?? this.mousePressTarget;
      if (this.mousePressPoint && (raw.x !== this.mousePressPoint.x || raw.y !== this.mousePressPoint.y)) {
        this.mousePressMoved = true;
        this.lastComponentClick = void 0;
      }
      let render = false;
      const targetResult = this.dispatchMouseToTarget(event, target);
      if (targetResult)
        render = this.applyMouseDispatchResult(event, targetResult);
      if (raw.release) {
        if (!this.mousePressMoved && this.mousePressPoint?.x === raw.x && this.mousePressPoint.y === raw.y) {
          const clickEvent = this.createMouseEvent("click", raw.button, raw.x, raw.y, {
            clickCount: this.getComponentClickCount(target, raw.x, raw.y)
          });
          const clickResult = this.dispatchMouseToTarget(clickEvent, target);
          if (clickResult)
            render = this.applyMouseDispatchResult(clickEvent, clickResult) || render;
        }
        this.clearComponentMouseGesture();
      }
      if (render)
        this.requestRender();
      return;
    }
    if (this.handleSearchMouseEvent(raw))
      return;
    const overlay = this.dispatchMouseToOverlay(event);
    if (!overlay.hit) {
      if (this.handleScrollToEndIndicatorMouseEvent(raw))
        return;
      const scrollbarHandled = this.handleScrollbarMouseEvent(raw);
      if (!this.scrollbarDrag)
        this.updateScrollbarHover(raw.x, raw.y);
      if (scrollbarHandled)
        return;
    } else {
      this.stopScrollbarHover();
    }
    const result = overlay.result ?? (overlay.hit ? void 0 : this.dispatchMouseToLayout(event));
    if (result) {
      const render = this.applyMouseDispatchResult(event, result);
      if (type === "press") {
        this.clearTextSelection();
        this.mousePressTarget = result.target;
        this.mousePressPoint = { x: raw.x, y: raw.y };
        this.mousePressMoved = false;
      }
      if (render)
        this.requestRender();
      return;
    }
    if (this.handleRightClickPaste(raw))
      return;
    this.handleSelectionMouseEvent(raw);
  }
  parseWheelEvent(data) {
    const sgr2 = /^\x1b\[<(\d+);(\d+);(\d+)[Mm]$/.exec(data);
    if (sgr2) {
      const button = Number.parseInt(sgr2[1], 10);
      if ((button & 64) === 0)
        return void 0;
      const direction = button & 3;
      if (direction !== 0 && direction !== 1)
        return void 0;
      return {
        direction: direction === 0 ? -1 : 1,
        x: Number.parseInt(sgr2[2], 10) - 1,
        y: Number.parseInt(sgr2[3], 10) - 1,
        button
      };
    }
    if (data.length === 6 && data.startsWith("\x1B[M")) {
      const button = data.charCodeAt(3) - 32;
      if ((button & 64) === 0)
        return void 0;
      const direction = button & 3;
      if (direction !== 0 && direction !== 1)
        return void 0;
      return {
        direction: direction === 0 ? -1 : 1,
        x: data.charCodeAt(4) - 33,
        y: data.charCodeAt(5) - 33,
        button
      };
    }
    return void 0;
  }
  routeWheel(event) {
    let remaining = event.direction * this.wheelScrollLines;
    const seen = /* @__PURE__ */ new Set();
    for (const scrollView of this.currentLayout ? getScrollViewsAt(this.currentLayout, event.x, event.y) : []) {
      seen.add(scrollView);
      remaining = scrollView.scrollBy(remaining);
      if (remaining === 0 || scrollView.overscroll === "contain")
        break;
    }
    const primary = this.getPrimaryScrollView();
    if (remaining !== 0 && !seen.has(primary))
      primary.scrollBy(remaining);
    this.updateScrollbarHover(event.x, event.y);
    this.requestRender();
  }
  parseSgrMouseEvent(data) {
    const match = /^\x1b\[<(\d+);(\d+);(\d+)([Mm])$/.exec(data);
    if (!match)
      return void 0;
    return {
      button: Number.parseInt(match[1], 10),
      x: Number.parseInt(match[2], 10) - 1,
      y: Number.parseInt(match[3], 10) - 1,
      release: match[4] === "m"
    };
  }
  handleRightClickPaste(event) {
    if (!this.onRightClickPaste || process.platform !== "win32" || process.env.TERM_PROGRAM?.toLowerCase() === "vscode" || event.release || event.button !== 2) {
      return false;
    }
    try {
      this.onRightClickPaste();
    } catch {
    }
    return true;
  }
  handleScrollToEndIndicatorMouseEvent(event) {
    const rect = this.scrollToEndIndicatorRect;
    if (!rect || event.release || (event.button & 32) !== 0 || (event.button & 3) !== 0)
      return false;
    if (event.y !== rect.row || event.x < rect.column || event.x >= rect.column + rect.width)
      return false;
    this.scrollToBottom();
    return true;
  }
  getScrollbarTargetAt(x2, y2, includeHiddenAuto = false) {
    if (this.hasOverlay() || !this.currentLayout)
      return void 0;
    for (const scrollView of getScrollViewsAt(this.currentLayout, x2, y2)) {
      const box = getScrollViewBox(this.currentLayout, scrollView);
      const geometry = box ? getScrollbarGeometry(box, includeHiddenAuto) : void 0;
      if (geometry && x2 === geometry.column && y2 >= geometry.trackTop && y2 < geometry.trackTop + geometry.trackHeight) {
        return { scrollView, geometry };
      }
    }
    return void 0;
  }
  setScrollbarHover(scrollView) {
    if (scrollView === this.scrollbarHover)
      return;
    this.scrollbarHover?.setScrollbarActive(false);
    this.scrollbarHover = scrollView;
    this.scrollbarHover?.setScrollbarActive(true);
  }
  updateScrollbarHover(x2, y2) {
    this.setScrollbarHover(this.getScrollbarTargetAt(x2, y2, true)?.scrollView);
  }
  stopScrollbarHover() {
    this.setScrollbarHover(void 0);
  }
  scrollScrollbarToPointer(scrollView, geometry, pointerY, grabOffset) {
    const maxThumbOffset = geometry.trackHeight - geometry.thumbHeight;
    const thumbOffset = Math.max(0, Math.min(maxThumbOffset, pointerY - geometry.trackTop - grabOffset));
    const scrollTop = maxThumbOffset === 0 ? 0 : Math.round(thumbOffset / maxThumbOffset * geometry.maxScrollTop);
    scrollView.scrollTo(scrollTop);
  }
  handleScrollbarMouseEvent(event) {
    if (this.scrollbarDrag) {
      if (event.release) {
        this.stopScrollbarDrag();
        return true;
      }
      const box = this.currentLayout ? getScrollViewBox(this.currentLayout, this.scrollbarDrag.scrollView) : void 0;
      const geometry = box ? getScrollbarGeometry(box) : void 0;
      if (geometry) {
        this.scrollScrollbarToPointer(this.scrollbarDrag.scrollView, geometry, event.y, this.scrollbarDrag.grabOffset);
      }
      return true;
    }
    if (event.release || (event.button & 32) !== 0 || (event.button & 3) !== 0)
      return false;
    const target = this.getScrollbarTargetAt(event.x, event.y);
    if (!target)
      return false;
    this.stopSelectionAutoScroll();
    this.selectionPressActive = false;
    this.selectionAnchor = void 0;
    this.selectionFocus = void 0;
    this.selectionGranularity = "character";
    this.selectionInitialRange = void 0;
    this.lastClick = void 0;
    this.pressedUrl = void 0;
    this.selectionDragged = false;
    this.setScrollbarHover(target.scrollView);
    const onThumb = event.y >= target.geometry.thumbTop && event.y < target.geometry.thumbTop + target.geometry.thumbHeight;
    const grabOffset = onThumb ? event.y - target.geometry.thumbTop : Math.floor(target.geometry.thumbHeight / 2);
    if (!onThumb)
      this.scrollScrollbarToPointer(target.scrollView, target.geometry, event.y, grabOffset);
    this.scrollbarDrag = {
      scrollView: target.scrollView,
      grabOffset
    };
    return true;
  }
  stopScrollbarDrag() {
    this.scrollbarDrag = void 0;
  }
  getScrollSelectionPoint(scrollView, x2, y2) {
    if (!this.currentLayout)
      return void 0;
    const box = getScrollViewBox(this.currentLayout, scrollView);
    if (!box || box.rect.height <= 0 || box.clip.height <= 0)
      return void 0;
    const visibleTop = Math.max(0, box.rect.y, box.clip.y);
    const visibleBottom = Math.min(this.terminal.rows - 1, box.rect.y + box.rect.height - 1, box.clip.y + box.clip.height - 1);
    if (visibleBottom < visibleTop)
      return void 0;
    const pointerRow = Math.max(visibleTop, Math.min(visibleBottom, y2));
    const maxContentRow = Math.max(0, (box.scrollContentLines?.length ?? 1) - 1);
    return {
      row: Math.max(0, Math.min(maxContentRow, scrollView.scrollTop + pointerRow - box.rect.y)),
      col: Math.max(0, Math.min(box.rect.width - 1, x2 - box.rect.x)),
      scrollView
    };
  }
  getSelectionPoint(event, scrollView) {
    if (scrollView) {
      const point = this.getScrollSelectionPoint(scrollView, event.x, event.y);
      if (point)
        return point;
    }
    return {
      row: Math.max(0, Math.min(this.terminal.rows - 1, event.y)),
      col: Math.max(0, Math.min(this.terminal.columns - 1, event.x))
    };
  }
  getSelectionSourceLine(point) {
    if (point.scrollView && this.currentLayout) {
      const lines = getScrollViewBox(this.currentLayout, point.scrollView)?.scrollContentLines;
      if (lines)
        return lines[point.row] ?? "";
    }
    return this.previousScreen[point.row] ?? "";
  }
  getWordSelection(point) {
    const line = stripTerminalSequences(this.getSelectionSourceLine(point));
    const segments = [];
    let start = 0;
    for (const segment of wordSegmenter4.segment(line)) {
      const end = start + visibleWidth(segment.segment);
      const joiner = TERMINAL_WORD_SELECTION_JOINERS.has(segment.segment);
      segments.push({ start, end, selectable: segment.isWordLike === true || joiner, joiner });
      start = end;
    }
    const clickedSegmentIndex = segments.findIndex((segment) => point.col >= segment.start && point.col < segment.end);
    if (clickedSegmentIndex < 0)
      return void 0;
    const canJoin = (left, right) => left.selectable && right.selectable && (left.joiner || right.joiner);
    let selectionStart = segments[clickedSegmentIndex].start;
    let selectionEnd = segments[clickedSegmentIndex].end;
    for (let index = clickedSegmentIndex; index > 0 && canJoin(segments[index - 1], segments[index]); index--) {
      selectionStart = segments[index - 1].start;
    }
    for (let index = clickedSegmentIndex; index < segments.length - 1 && canJoin(segments[index], segments[index + 1]); index++) {
      selectionEnd = segments[index + 1].end;
    }
    return {
      start: { ...point, col: selectionStart },
      end: { ...point, col: selectionEnd, boundary: true }
    };
  }
  getLineSelection(point) {
    return {
      start: { ...point, col: 0 },
      end: { ...point, col: visibleWidth(this.getSelectionSourceLine(point)), boundary: true }
    };
  }
  updateSelectionFocus(point) {
    if (this.selectionGranularity === "character" || !this.selectionInitialRange) {
      this.selectionFocus = point;
      return;
    }
    const range = this.selectionGranularity === "word" ? this.getWordSelection(point) : this.getLineSelection(point);
    if (!range)
      return;
    const initial = this.selectionInitialRange;
    const targetBeforeInitial = range.start.row < initial.start.row || range.start.row === initial.start.row && range.start.col < initial.start.col;
    if (targetBeforeInitial) {
      this.selectionAnchor = initial.end;
      this.selectionFocus = range.start;
    } else {
      this.selectionAnchor = initial.start;
      this.selectionFocus = range.end;
    }
  }
  getClickCount(point, word) {
    const now = Date.now();
    const previous = this.lastClick;
    const count = word && previous && now - previous.timestamp <= DOUBLE_CLICK_INTERVAL_MS && previous.row === point.row && previous.scrollView === point.scrollView && previous.wordStart === word.start.col && previous.wordEnd === word.end.col ? previous.count % 3 + 1 : 1;
    this.lastClick = word ? {
      timestamp: now,
      count,
      row: point.row,
      scrollView: point.scrollView,
      wordStart: word.start.col,
      wordEnd: word.end.col
    } : void 0;
    return count;
  }
  updateSelectionAutoScroll(event) {
    const scrollView = this.selectionAnchor?.scrollView;
    if (!scrollView || !this.currentLayout) {
      this.stopSelectionAutoScroll();
      return;
    }
    const box = getScrollViewBox(this.currentLayout, scrollView);
    if (!box || box.rect.height <= 0 || box.clip.height <= 0) {
      this.stopSelectionAutoScroll();
      return;
    }
    const visibleTop = Math.max(0, box.rect.y, box.clip.y);
    const visibleBottom = Math.min(this.terminal.rows - 1, box.rect.y + box.rect.height - 1, box.clip.y + box.clip.height - 1);
    this.selectionDragPointer = { x: event.x, y: event.y };
    this.selectionAutoScrollDirection = event.y <= visibleTop ? -1 : event.y >= visibleBottom ? 1 : 0;
    if (this.selectionAutoScrollDirection === 0) {
      this.stopSelectionAutoScroll();
      return;
    }
    if (this.selectionAutoScrollTimer)
      return;
    this.selectionAutoScrollTimer = setInterval(() => this.autoScrollSelection(), 50);
    this.selectionAutoScrollTimer.unref();
  }
  autoScrollSelection() {
    const scrollView = this.selectionAnchor?.scrollView;
    const pointer = this.selectionDragPointer;
    const direction = this.selectionAutoScrollDirection;
    if (!scrollView || !pointer || direction === 0) {
      this.stopSelectionAutoScroll();
      return;
    }
    const remaining = scrollView.scrollBy(direction);
    if (remaining === direction) {
      this.stopSelectionAutoScroll();
      return;
    }
    const point = this.getScrollSelectionPoint(scrollView, pointer.x, pointer.y);
    if (point)
      this.updateSelectionFocus(point);
    this.requestRender();
  }
  stopSelectionAutoScroll() {
    if (this.selectionAutoScrollTimer) {
      clearInterval(this.selectionAutoScrollTimer);
      this.selectionAutoScrollTimer = void 0;
    }
    this.selectionAutoScrollDirection = 0;
    this.selectionDragPointer = void 0;
  }
  handleSelectionMouseEvent(event) {
    const button = event.button & 3;
    if (button !== 0 && !(event.release && button === 3))
      return;
    const anchorScrollView = this.selectionAnchor?.scrollView;
    const point = this.getSelectionPoint(event, anchorScrollView);
    if (event.release) {
      if (!this.selectionPressActive)
        return;
      this.selectionPressActive = false;
      this.stopSelectionAutoScroll();
      if (!this.selectionAnchor)
        return;
      this.updateSelectionFocus(point);
      const isClick = !this.selectionDragged && this.selectionAnchor.scrollView === point.scrollView && this.selectionAnchor.row === point.row && this.selectionAnchor.col === point.col;
      const clickedUrl = isClick ? this.pressedUrl : void 0;
      this.pressedUrl = void 0;
      if (clickedUrl && this.openUrl) {
        this.selectionAnchor = void 0;
        this.selectionFocus = void 0;
        try {
          this.openUrl(clickedUrl);
        } catch {
        }
        this.requestRender();
        return;
      }
      if (isClick) {
        const clickEvent = this.createMouseEvent("click", event.button, event.x, event.y, {
          clickCount: this.lastClick?.count ?? 1
        });
        const overlay = this.dispatchMouseToOverlay(clickEvent);
        const result = overlay.result ?? (overlay.hit ? void 0 : this.dispatchMouseToLayout(clickEvent));
        if (result) {
          const render = this.applyMouseDispatchResult(clickEvent, result);
          this.clearTextSelection();
          if (render)
            this.requestRender();
          return;
        }
      }
      if (this.copyOnSelect)
        void this.copySelectionToClipboard();
      this.requestRender();
      return;
    }
    if ((event.button & 32) !== 0) {
      if (!this.selectionPressActive || !this.selectionAnchor)
        return;
      this.selectionDragged = true;
      this.lastClick = void 0;
      this.pressedUrl = void 0;
      this.updateSelectionFocus(point);
      this.updateSelectionAutoScroll(event);
      this.requestRender();
      return;
    }
    this.stopSelectionAutoScroll();
    this.selectionPressActive = true;
    const scrollView = !this.hasOverlay() && this.currentLayout ? getScrollViewsAt(this.currentLayout, event.x, event.y)[0] : void 0;
    const anchor = this.getSelectionPoint(event, scrollView);
    const word = this.getWordSelection(anchor);
    const clickCount = this.getClickCount(anchor, word);
    const range = clickCount === 2 ? word : clickCount === 3 ? this.getLineSelection(anchor) : void 0;
    this.selectionGranularity = range ? clickCount === 2 ? "word" : "line" : "character";
    this.selectionInitialRange = range;
    this.selectionAnchor = range?.start ?? anchor;
    this.selectionFocus = range?.end ?? anchor;
    this.selectionDragged = false;
    this.pressedUrl = range ? void 0 : getOsc8LinkAtColumn(this.previousScreen[Math.max(0, Math.min(this.terminal.rows - 1, event.y))] ?? "", Math.max(0, Math.min(this.terminal.columns - 1, event.x)));
    this.requestRender();
  }
  getSelectionBounds() {
    if (!this.selectionAnchor || !this.selectionFocus)
      return void 0;
    if (this.selectionAnchor.scrollView !== this.selectionFocus.scrollView)
      return void 0;
    const anchorBeforeFocus = this.selectionAnchor.row < this.selectionFocus.row || this.selectionAnchor.row === this.selectionFocus.row && this.selectionAnchor.col < this.selectionFocus.col;
    if (this.selectionAnchor.row === this.selectionFocus.row && this.selectionAnchor.col === this.selectionFocus.col) {
      return void 0;
    }
    return anchorBeforeFocus ? { start: this.selectionAnchor, end: this.selectionFocus } : { start: this.selectionFocus, end: this.selectionAnchor };
  }
  getSelectionColumns(line, row, selection, minColumn = 0, maxColumn = visibleWidth(line)) {
    const lineWidth = visibleWidth(line);
    let start = Math.max(0, minColumn);
    let end = Math.min(lineWidth, maxColumn);
    if (row === selection.start.row) {
      start = getGraphemeCellRange(line, selection.start.col)?.start ?? Math.min(selection.start.col, lineWidth);
    }
    if (row === selection.end.row) {
      end = selection.end.boundary ? Math.min(selection.end.col, lineWidth) : getGraphemeCellRange(line, selection.end.col)?.end ?? Math.min(selection.end.col + 1, lineWidth);
    }
    return { start: Math.max(minColumn, start), end: Math.min(maxColumn, end) };
  }
  getActiveSelectionText() {
    const selection = this.getSelectionBounds();
    if (!selection)
      return void 0;
    let sourceLines = this.previousScreen;
    if (selection.start.scrollView) {
      if (!this.currentLayout)
        return void 0;
      const box = getScrollViewBox(this.currentLayout, selection.start.scrollView);
      if (!box?.scrollContentLines)
        return void 0;
      sourceLines = box.scrollContentLines;
    }
    const lines = [];
    for (let row = selection.start.row; row <= selection.end.row; row++) {
      const line = sourceLines[row] ?? "";
      const columns = this.getSelectionColumns(line, row, selection);
      lines.push(stripTerminalSequences(sliceByColumn(line, columns.start, Math.max(0, columns.end - columns.start), true)).trimEnd());
    }
    const text = lines.join("\n");
    return text.length === 0 ? void 0 : text;
  }
  async copySelectionToClipboard() {
    const text = this.getActiveSelectionText();
    if (!text)
      return false;
    return this.copyTextToClipboard(text);
  }
  async copyTextToClipboard(text) {
    if (this.copySelection) {
      const ok = await this.copySelection(text);
      this.flash(ok ? "Copied!" : "Copy failed");
      return ok;
    }
    this.terminal.write(`\x1B]52;c;${Buffer.from(text).toString("base64")}\x07`);
    this.flash("Copied!");
    return true;
  }
  applySearchTextHighlight(text, current) {
    const style = current ? this.searchCurrentMatchStyle : this.searchMatchStyle;
    let result = "";
    let plainStart = 0;
    let index = 0;
    while (index < text.length) {
      const ansi = extractAnsiCode(text, index);
      if (!ansi) {
        index += 1;
        continue;
      }
      if (index > plainStart)
        result += style(text.slice(plainStart, index));
      result += ansi.code;
      index += ansi.length;
      plainStart = index;
    }
    if (plainStart < text.length)
      result += style(text.slice(plainStart));
    return result;
  }
  applySearchHighlights(screen, layout) {
    const search = this.activeSearch;
    if (!search || search.selectedIndex < 0 || search.matches.length === 0)
      return screen;
    const scrollView = layout.primaryScrollView ?? this.implicitScrollView;
    const box = getScrollViewBox(layout, scrollView);
    if (!box)
      return screen;
    const rangesByRow = /* @__PURE__ */ new Map();
    const scrollbarColumn = getScrollbarGeometry(box)?.column;
    const minRow = Math.max(0, box.rect.y, box.clip.y);
    const maxRow = Math.min(screen.length, box.rect.y + box.rect.height, box.clip.y + box.clip.height);
    const minColumn = Math.max(0, box.rect.x, box.clip.x);
    const maxColumn = Math.min(this.terminal.columns, box.rect.x + box.rect.width, box.clip.x + box.clip.width, scrollbarColumn ?? Number.POSITIVE_INFINITY);
    const minContentRow = scrollView.scrollTop + minRow - box.rect.y;
    const maxContentRow = scrollView.scrollTop + maxRow - box.rect.y - 1;
    let low = 0;
    let high = search.matches.length;
    while (low < high) {
      const middle = low + Math.floor((high - low) / 2);
      const match = search.matches[middle];
      const lastRow = match.segments[match.segments.length - 1]?.row ?? -1;
      if (lastRow < minContentRow)
        low = middle + 1;
      else
        high = middle;
    }
    for (let matchIndex = low; matchIndex < search.matches.length; matchIndex++) {
      const match = search.matches[matchIndex];
      if ((match.segments[0]?.row ?? 0) > maxContentRow)
        break;
      for (const segment of match.segments) {
        const row = box.rect.y + segment.row - scrollView.scrollTop;
        if (row < minRow || row >= maxRow)
          continue;
        const startCol = Math.max(minColumn, box.rect.x + segment.startCol);
        const endCol = Math.min(maxColumn, box.rect.x + segment.endCol);
        if (endCol <= startCol)
          continue;
        const ranges = rangesByRow.get(row) ?? [];
        ranges.push({ startCol, endCol, current: matchIndex === search.selectedIndex });
        rangesByRow.set(row, ranges);
      }
    }
    const result = [...screen];
    for (const [row, ranges] of rangesByRow) {
      let line = result[row] ?? "";
      if (isImageLine(line))
        continue;
      const lineWidth = visibleWidth(line);
      for (const range of ranges.sort((a, b2) => b2.startCol - a.startCol)) {
        const startCol = Math.min(range.startCol, lineWidth);
        const endCol = Math.min(range.endCol, lineWidth);
        if (endCol <= startCol)
          continue;
        const before = sliceByColumn(line, 0, startCol, true);
        const highlighted = sliceByColumn(line, startCol, endCol - startCol, true);
        const after = sliceByColumn(line, endCol, Math.max(0, lineWidth - endCol), true);
        line = `${before}${this.applySearchTextHighlight(highlighted, range.current)}${after}`;
      }
      result[row] = line;
    }
    return result;
  }
  applySelectionHighlight(text) {
    let result = "\x1B[7m";
    let index = 0;
    while (index < text.length) {
      const ansi = extractAnsiCode(text, index);
      if (!ansi) {
        result += text[index];
        index += 1;
        continue;
      }
      result += ansi.code;
      if (ansi.code.endsWith("m"))
        result += "\x1B[7m";
      index += ansi.length;
    }
    return `${result}\x1B[27m`;
  }
  applySelection(screen, layout = this.currentLayout) {
    const selection = this.getSelectionBounds();
    if (!selection)
      return screen;
    let screenSelection = selection;
    let minRow = 0;
    let maxRow = screen.length - 1;
    let minColumn = 0;
    let maxColumn = this.terminal.columns;
    if (selection.start.scrollView) {
      if (!layout)
        return screen;
      const box = getScrollViewBox(layout, selection.start.scrollView);
      if (!box)
        return screen;
      minRow = Math.max(0, box.rect.y, box.clip.y);
      maxRow = Math.min(screen.length - 1, box.rect.y + box.rect.height - 1, box.clip.y + box.clip.height - 1);
      minColumn = Math.max(0, box.rect.x, box.clip.x);
      maxColumn = Math.min(this.terminal.columns, box.rect.x + box.rect.width, box.clip.x + box.clip.width);
      screenSelection = {
        start: {
          ...selection.start,
          row: box.rect.y + selection.start.row - selection.start.scrollView.scrollTop,
          col: box.rect.x + selection.start.col
        },
        end: {
          ...selection.end,
          row: box.rect.y + selection.end.row - selection.start.scrollView.scrollTop,
          col: box.rect.x + selection.end.col
        }
      };
    }
    return screen.map((line, row) => {
      if (row < minRow || row > maxRow || row < screenSelection.start.row || row > screenSelection.end.row || isImageLine(line)) {
        return line;
      }
      const lineWidth = visibleWidth(line);
      const columns = this.getSelectionColumns(line, row, screenSelection, minColumn, maxColumn);
      if (columns.end <= columns.start)
        return line;
      const before = sliceByColumn(line, 0, columns.start, true);
      const selected = sliceByColumn(line, columns.start, columns.end - columns.start, true);
      const after = sliceByColumn(line, columns.end, Math.max(0, lineWidth - columns.end), true);
      return `${before}${this.applySelectionHighlight(selected)}${after}`;
    });
  }
  isMouseSequence(data) {
    return /^\x1b\[<\d+;\d+;\d+[Mm]$/.test(data) || data.length === 6 && data.startsWith("\x1B[M");
  }
  compositeScrollToEndIndicator(screen, layout, width) {
    this.scrollToEndIndicatorRect = void 0;
    const scrollView = layout.primaryScrollView ?? this.implicitScrollView;
    if (!this.scrollToEndIndicator || !scrollView.followEnd || scrollView.isFollowingEnd)
      return screen;
    const box = getScrollViewBox(layout, scrollView);
    const clip = box?.clip;
    if (!clip || clip.width <= 0 || clip.height <= 0)
      return screen;
    const row = clip.y + clip.height - 1;
    if (row >= screen.length || isImageLine(screen[row] ?? ""))
      return screen;
    const scrollbarColumn = box ? getScrollbarGeometry(box)?.column : void 0;
    const availableWidth = Math.max(0, (scrollbarColumn ?? clip.x + clip.width) - clip.x);
    const text = truncateToWidth(this.scrollToEndIndicator(), availableWidth, "");
    const textWidth = visibleWidth(text);
    if (textWidth === 0)
      return screen;
    const column = clip.x + Math.floor((availableWidth - textWidth) / 2);
    const result = [...screen];
    result[row] = compositeTuiLine(result[row] ?? "", text, column, textWidth, width);
    this.scrollToEndIndicatorRect = { row, column, width: textWidth };
    return result;
  }
  compositeFlashes(screen, width, height) {
    const flashLines = this.flashes.render(width).slice(-height);
    if (flashLines.length === 0)
      return screen;
    const result = [...screen];
    while (result.length < height)
      result.push("");
    for (let row = 0; row < flashLines.length; row++) {
      const line = flashLines[row];
      const flashWidth = visibleWidth(line);
      if (flashWidth === 0)
        continue;
      result[row] = compositeTuiLine(result[row] ?? "", line, width - flashWidth, flashWidth, width);
    }
    return result;
  }
  doRender() {
    if (this.stopped || !this.altScreenActive)
      return;
    const width = Math.max(1, this.terminal.columns);
    const height = Math.max(1, this.terminal.rows);
    const root = this.layoutRoot ?? this.implicitScrollView;
    let nextLayout = renderLayoutFrame(root, width, height, () => this.requestRender());
    if (this.refreshSearch(nextLayout)) {
      nextLayout = renderLayoutFrame(root, width, height, () => this.requestRender());
    }
    let screen = nextLayout.lines.map((line) => line.replace(OSC133_ZONE_PREFIX2, ""));
    screen = this.applySearchHighlights(screen, nextLayout);
    screen = this.compositeScrollToEndIndicator(screen, nextLayout, width);
    screen = this.compositeOverlays(screen, width, height);
    if (screen.length > height)
      screen = screen.slice(screen.length - height);
    screen = this.applySelection(screen, nextLayout);
    screen = this.compositeFlashes(screen, width, height);
    const cursorPos = this.extractCursorPosition(screen, height);
    screen = this.applyLineResets(screen).map((line) => {
      if (isImageLine(line) || visibleWidth(line) <= width)
        return line;
      return sliceByColumn(line, 0, width, true);
    });
    const fullRedraw = this.previousScreen.length === 0 || this.previousScreenWidth !== width || this.previousScreenHeight !== height;
    const imagesNeedRedraw = screen.some((line, row) => line !== this.previousScreen[row] && (isImageLine(line) || isImageLine(this.previousScreen[row] ?? "")));
    const redrawImages = fullRedraw || imagesNeedRedraw;
    const hadUploadedKittyImages = this.uploadedKittyImages.size > 0;
    const preparedKittyScreen = redrawImages && this.imageProtocol === "kitty" ? this.prepareKittyScreen(screen) : { lines: screen, evictedImageDeletion: "" };
    let buffer = BEGIN_SYNCHRONIZED_OUTPUT;
    if (fullRedraw) {
      this.fullRedrawCount += 1;
      const clearImages = this.imageProtocol === "kitty" && hadUploadedKittyImages ? deleteAllKittyPlacements() : this.deleteKittyImages();
      buffer += `${clearImages}\x1B[2J`;
    } else if (imagesNeedRedraw) {
      if (this.imageProtocol === "iterm2")
        buffer += "\x1B[2J";
      else if (this.imageProtocol === "kitty")
        buffer += deleteAllKittyPlacements();
    }
    buffer += preparedKittyScreen.evictedImageDeletion;
    for (let row = 0; row < height; row++) {
      if (!fullRedraw && !imagesNeedRedraw && screen[row] === this.previousScreen[row])
        continue;
      buffer += `\x1B[${row + 1};1H\x1B[2K${preparedKittyScreen.lines[row] ?? ""}`;
    }
    if (cursorPos) {
      buffer += `\x1B[${cursorPos.row + 1};${Math.min(width, cursorPos.col) + 1}H`;
      buffer += this.getShowHardwareCursor() ? "\x1B[?25h" : "\x1B[?25l";
    } else {
      buffer += "\x1B[?25l";
    }
    buffer += END_SYNCHRONIZED_OUTPUT;
    this.terminal.write(buffer);
    this.previousScreen = screen;
    this.previousScreenWidth = width;
    this.previousScreenHeight = height;
    this.currentLayout = nextLayout;
  }
};

// node_modules/@earendil-works/pi-tui/dist/tui-main-screen.js
var MAX_RENDER_WRITE_CHARS = 1024 * 1024;

// src/theme.ts
function sgr(code) {
  return (text) => text ? `\x1B[${code}m${text}\x1B[0m` : "";
}
var ui = {
  accent: sgr("38;2;138;173;244"),
  accentStrong: sgr("1;38;2;138;173;244"),
  text: (text) => text,
  muted: sgr("38;2;127;140;152"),
  dim: sgr("2"),
  success: sgr("38;2;123;216;143"),
  warning: sgr("38;2;235;203;139"),
  error: sgr("38;2;243;139;168"),
  border: sgr("38;2;95;105;115"),
  code: sgr("38;2;235;203;139"),
  bold: sgr("1"),
  italic: sgr("3"),
  underline: sgr("4"),
  strike: sgr("9")
};
var selectList = {
  selectedPrefix: ui.accentStrong,
  selectedText: ui.accentStrong,
  description: ui.muted,
  scrollInfo: ui.dim,
  noMatch: ui.warning
};
var editorTheme = {
  borderColor: ui.border,
  selectList
};
var markdownTheme = {
  heading: ui.accentStrong,
  link: ui.underline,
  linkUrl: ui.dim,
  code: ui.code,
  codeBlock: ui.text,
  codeBlockBorder: ui.dim,
  quote: ui.muted,
  quoteBorder: ui.dim,
  hr: ui.dim,
  listBullet: ui.accent,
  bold: ui.bold,
  italic: ui.italic,
  strikethrough: ui.strike,
  underline: ui.underline
};

// src/components/composer.ts
var Composer = class extends Editor {
  mode = "ask";
  constructor(tui) {
    super(tui, editorTheme, { paddingX: 1 });
  }
  setMode(mode) {
    this.mode = mode;
    this.borderColor = mode === "approval" ? ui.warning : mode === "steer" ? ui.accent : ui.border;
  }
  renderTopBorder(width, hiddenLineCount) {
    if (hiddenLineCount > 0) {
      return super.renderTopBorder(width, hiddenLineCount);
    }
    if (width <= 0) return "";
    const label = this.mode === "approval" ? " Permission required " : this.mode === "steer" ? " Steer MiniCode " : " Ask MiniCode ";
    const styledLabel = this.mode === "approval" ? ui.warning(label) : ui.accentStrong(label);
    const remaining = Math.max(0, width - visibleWidth(label));
    return truncateToWidth(
      styledLabel + this.borderColor("\u2500".repeat(remaining)),
      width
    );
  }
  renderBottomBorder(width, hiddenLineCount) {
    if (hiddenLineCount > 0) {
      return super.renderBottomBorder(width, hiddenLineCount);
    }
    if (width <= 0) return "";
    const hint = this.mode === "approval" ? " choose in the approval panel " : this.mode === "steer" ? " Enter steer \xB7 Esc cancel " : " Enter send \xB7 Alt+Enter newline ";
    if (visibleWidth(hint) >= width - 2) {
      return this.borderColor("\u2500".repeat(width));
    }
    const remaining = Math.max(0, width - visibleWidth(hint));
    const left = Math.floor(remaining / 2);
    const right = remaining - left;
    return truncateToWidth(
      this.borderColor("\u2500".repeat(left)) + ui.dim(hint) + this.borderColor("\u2500".repeat(right)),
      width
    );
  }
};

// src/components/footer.ts
var Footer = class {
  text = "Ready";
  tone = "muted";
  hint = "Ctrl+C exit";
  contextUsed = null;
  promptBudget = null;
  setText(text) {
    this.setStatus(text);
  }
  setStatus(text, tone = "muted", hint = "") {
    this.text = text;
    this.tone = tone;
    this.hint = hint;
  }
  setContext(used, promptBudget) {
    this.contextUsed = Math.max(0, used);
    this.promptBudget = Math.max(1, promptBudget);
  }
  invalidate() {
  }
  render(width) {
    if (width <= 0) return [""];
    const status = paintTone(this.tone, this.text);
    const context = this.contextLabel();
    if (width < 28) {
      return [truncateToWidth(status, width)];
    }
    const left = this.hint ? `${status}  ${ui.dim(this.hint)}` : status;
    if (!context || width < 52) {
      return [truncateToWidth(left, width)];
    }
    const right = ui.dim(context);
    const rightWidth = visibleWidth(right);
    const leftBudget = Math.max(1, width - rightWidth - 1);
    const fittedLeft = truncateToWidth(left, leftBudget);
    const spaces = Math.max(1, width - visibleWidth(fittedLeft) - rightWidth);
    return [
      truncateToWidth(
        `${fittedLeft}${" ".repeat(spaces)}${right}`,
        width
      )
    ];
  }
  contextLabel() {
    if (this.contextUsed === null || this.promptBudget === null) return "";
    const percent = Math.min(
      999,
      Math.round(this.contextUsed / this.promptBudget * 100)
    );
    return `context ${percent}% \xB7 ${compact(this.contextUsed)}/${compact(this.promptBudget)}`;
  }
};
function paintTone(tone, text) {
  switch (tone) {
    case "working":
      return ui.accent(`\u25CF ${text}`);
    case "success":
      return ui.success(`\u2713 ${text}`);
    case "warning":
      return ui.warning(`! ${text}`);
    case "error":
      return ui.error(`\u2715 ${text}`);
    default:
      return ui.muted(text);
  }
}
function compact(value) {
  return value >= 1e3 ? `${(value / 1e3).toFixed(value >= 1e4 ? 0 : 1)}k` : String(value);
}

// src/components/header.ts
var Header = class {
  session = "new";
  workspace;
  provider;
  model;
  constructor(options = {}) {
    this.workspace = options.workspace ?? "";
    this.provider = options.provider;
    this.model = options.model;
  }
  setSession(sessionId) {
    this.session = sessionId;
  }
  invalidate() {
  }
  render(width) {
    if (width <= 0) return [""];
    const project = projectName(this.workspace);
    const modelLabel = this.model || this.provider || "";
    const sessionLabel = this.session === "new" ? "new session" : shortSession(this.session);
    if (width < 44) {
      const compact2 = [
        ui.accentStrong("MiniCode"),
        project ? ui.muted(` \xB7 ${project}`) : ""
      ].join("");
      return [truncateToWidth(compact2, width)];
    }
    const left = `${ui.accentStrong("MiniCode")}  ${ui.dim(sessionLabel)}`;
    const right = modelLabel ? ui.muted(modelLabel) : "";
    const first = pair(left, right, width);
    const second = this.workspace ? truncateToWidth(ui.dim(this.workspace), width) : "";
    return second ? [first, second] : [first];
  }
};
function shortSession(value) {
  const compact2 = value.replace(/^session_/, "");
  return `session ${compact2.slice(0, 8)}`;
}
function projectName(value) {
  const normalized = value.replace(/[\\/]+$/, "");
  const parts = normalized.split(/[\\/]/);
  return parts.at(-1) ?? "";
}
function pair(left, right, width) {
  if (!right) return truncateToWidth(left, width);
  const rightWidth = visibleWidth(right);
  if (rightWidth >= width - 8) return truncateToWidth(left, width);
  const leftBudget = Math.max(1, width - rightWidth - 1);
  const fittedLeft = truncateToWidth(left, leftBudget);
  const spaces = Math.max(1, width - visibleWidth(fittedLeft) - rightWidth);
  return truncateToWidth(`${fittedLeft}${" ".repeat(spaces)}${right}`, width);
}

// src/overlays/approval.ts
var ApprovalOverlay = class {
  constructor(event, onDecision) {
    this.event = event;
    this.onDecision = onDecision;
  }
  event;
  onDecision;
  body = new Text();
  showDetails = false;
  invalidate() {
    this.body.invalidate();
  }
  handleInput(data) {
    if (matchesKey(data, "y")) {
      this.onDecision("approve");
      return;
    }
    if (this.event.can_approve_session && matchesKey(data, "g")) {
      this.onDecision("approve_session");
      return;
    }
    if (matchesKey(data, "n")) {
      this.onDecision("reject");
      return;
    }
    if (matchesKey(data, "s")) {
      this.onDecision("skip");
      return;
    }
    if (matchesKey(data, "a")) {
      this.onDecision("abort");
      return;
    }
    if (matchesKey(data, "v")) {
      this.showDetails = !this.showDetails;
      return;
    }
    if (matchesKey(data, Key.escape)) {
      this.onDecision("abort");
    }
  }
  render(width) {
    const preview = approvalPreview(this.event.details);
    const lines = [
      ui.warning("! ACTION REQUIRED"),
      ui.bold(approvalAction(this.event.tool)),
      ""
    ];
    if (preview.command) {
      lines.push(ui.code(`$ ${preview.command}`), "");
    } else if (preview.path) {
      lines.push(ui.code(preview.path), "");
    } else if (this.event.summary) {
      lines.push(this.event.summary, "");
    }
    if (preview.riskLevel) {
      lines.push(`${ui.muted("Risk")}    ${ui.warning(preview.riskLevel.toUpperCase())}`);
    }
    if (preview.reason) {
      lines.push(`${ui.muted("Reason")}  ${preview.reason}`);
    }
    if (preview.effects.length > 0) {
      lines.push(ui.muted("Effects"));
      lines.push(...preview.effects.slice(0, 3).map((effect) => `  - ${effect}`));
    }
    if (preview.riskLevel || preview.reason || preview.effects.length > 0) {
      lines.push("");
    }
    if (this.showDetails && this.event.details) {
      lines.push(this.event.details, "");
    }
    const grant = this.event.can_approve_session ? `  ${ui.success("[G] Allow for this session")}` : "";
    lines.push(
      `${ui.success("[Y] Allow once")}${grant}  ${ui.error("[N] Reject")}`,
      ui.muted("[S] Skip \xB7 [A] Abort \xB7 [V] Details")
    );
    this.body.setText(lines.join("\n"));
    return this.body.render(width).map((line) => truncateToWidth(line, Math.max(0, width)));
  }
};
function approvalPreview(details) {
  if (!details) return { effects: [] };
  try {
    const raw = JSON.parse(details);
    if (!isRecord2(raw)) return { effects: [] };
    const preview = isRecord2(raw.preview) ? raw.preview : {};
    return {
      riskLevel: stringValue(raw.risk_level),
      command: stringValue(preview.command),
      path: stringValue(preview.path),
      reason: stringValue(preview.reason),
      effects: Array.isArray(preview.effects) ? preview.effects.filter((value) => typeof value === "string") : []
    };
  } catch {
    return { effects: [] };
  }
}
function approvalAction(tool) {
  if (tool === "run_command") return "MiniCode wants to run a command";
  if (["edit", "write", "apply_patch"].includes(tool)) {
    return "MiniCode wants to change workspace files";
  }
  return "MiniCode requests permission";
}
function isRecord2(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function stringValue(value) {
  return typeof value === "string" && value.trim() ? value : void 0;
}

// src/overlays/user-input.ts
var UserInputOverlay = class {
  constructor(event, onSelect) {
    this.event = event;
    this.onSelect = onSelect;
  }
  event;
  onSelect;
  body = new Text();
  invalidate() {
    this.body.invalidate();
  }
  handleInput(data) {
    const keys = ["1", "2", "3", "4"];
    for (let index = 0; index < this.event.options.length; index += 1) {
      if (matchesKey(data, keys[index] ?? "1")) {
        this.onSelect(index);
        return;
      }
    }
  }
  render(width) {
    const lines = [
      ui.warning("! Input required"),
      "",
      this.event.question,
      ""
    ];
    for (let index = 0; index < this.event.options.length; index += 1) {
      const option = this.event.options[index];
      lines.push(
        `${ui.accentStrong(String(index + 1))}  ${option.label}`
      );
      if (option.description) {
        lines.push(`   ${ui.muted(option.description)}`);
      }
    }
    lines.push("", ui.muted("Press 1-4 to choose \xB7 Esc cancels the run"));
    this.body.setText(lines.join("\n"));
    return this.body.render(width).map((line) => truncateToWidth(line, Math.max(0, width)));
  }
};

// src/components/activity.ts
var MAX_VISIBLE_ENTRIES = 6;
var SUCCESSFUL_TOOL_STATUSES = /* @__PURE__ */ new Set([
  "ok",
  "duplicate_reused",
  "background_started"
]);
var Activity = class {
  status = "working";
  entries = [];
  start(id, action, target) {
    this.status = "working";
    const existing = this.entries.find((entry) => entry.id === id);
    if (existing) {
      existing.action = action || existing.action;
      existing.target = target ?? existing.target;
      existing.status = "running";
      existing.detail = null;
      existing.diffPreview = null;
      existing.diffTruncated = false;
      return;
    }
    this.entries.push({
      id,
      action: action || "Working",
      target: target ?? null,
      detail: null,
      diffPreview: null,
      diffTruncated: false,
      status: "running"
    });
  }
  finish(id, status, summary, details = {}) {
    const entry = this.entries.find((candidate) => candidate.id === id);
    if (!entry) return;
    const successful = SUCCESSFUL_TOOL_STATUSES.has(status);
    entry.status = status === "command_timed_out" ? "warning" : status === "command_cancelled" ? "cancelled" : successful ? "ok" : "error";
    const commandDetail = formatCommandDetail(details);
    if (commandDetail) {
      entry.detail = commandDetail;
    } else if (!successful && summary) {
      entry.target = summary;
    }
    if (successful && details.diffPreview) {
      entry.diffPreview = details.diffPreview;
      entry.diffTruncated = details.diffTruncated === true;
    }
  }
  complete() {
    this.status = "worked";
    for (const entry of this.entries) {
      if (entry.status === "running") entry.status = "ok";
    }
  }
  invalidate() {
  }
  render(width) {
    if (width <= 0) return [""];
    const title = this.status === "working" ? `${ui.accent("> Working")}` : `${ui.success("+ Worked")}${this.entries.length ? ui.muted(
      ` \xB7 ${this.entries.length} action${this.entries.length === 1 ? "" : "s"}`
    ) : ""}`;
    const lines = [truncateToWidth(title, width)];
    const visible = this.entries.slice(-MAX_VISIBLE_ENTRIES);
    const hidden = Math.max(0, this.entries.length - visible.length);
    if (hidden > 0) {
      lines.push(
        truncateToWidth(ui.dim(`  \u2026 ${hidden} earlier action${hidden === 1 ? "" : "s"}`), width)
      );
    }
    visible.forEach((entry, index) => {
      const branch = index === visible.length - 1 ? "\u2514" : "\u251C";
      const icon = entry.status === "running" ? ui.accent(">") : entry.status === "ok" ? ui.success("+") : entry.status === "warning" ? ui.warning("!") : entry.status === "cancelled" ? ui.muted("-") : ui.error("x");
      const action = ui.accent(entry.action);
      const target = entry.target ? `  ${ui.text(entry.target)}` : "";
      const detail = entry.detail ? `  ${ui.dim(`\xB7 ${entry.detail}`)}` : "";
      const prefix = `  ${ui.dim(branch)} ${icon} `;
      const available = Math.max(0, width - visibleWidth(prefix));
      lines.push(
        truncateToWidth(
          prefix + truncateToWidth(`${action}${target}${detail}`, available),
          width
        )
      );
      if (entry.diffPreview) {
        for (const diffLine of entry.diffPreview.split("\n")) {
          lines.push(renderDiffLine(diffLine, width));
        }
        if (entry.diffTruncated) {
          lines.push(truncateToWidth(ui.dim("      \u2026 diff preview truncated"), width));
        }
      }
    });
    return lines;
  }
};
function renderDiffLine(line, width) {
  const prefix = "      ";
  const content = line.startsWith("+++") || line.startsWith("---") ? ui.dim(line) : line.startsWith("+") ? ui.success(line) : line.startsWith("-") ? ui.error(line) : line.startsWith("@@") ? ui.accent(line) : ui.dim(line);
  return truncateToWidth(`${prefix}${content}`, width);
}
function formatCommandDetail(details) {
  const status = details.commandStatus;
  if (!status) return null;
  const duration = typeof details.durationMs === "number" ? formatDuration(details.durationMs) : null;
  if (status === "background_started") {
    return details.runtimeTaskId ? `background ${details.runtimeTaskId}` : "background started";
  }
  if (status === "timed_out") {
    return ["timed out", duration].filter(Boolean).join(" \xB7 ");
  }
  if (status === "cancelled") {
    return ["cancelled", duration].filter(Boolean).join(" \xB7 ");
  }
  const exit = typeof details.returncode === "number" ? `exit ${details.returncode}` : null;
  return [status === "completed" ? null : status, exit, duration].filter(Boolean).join(" \xB7 ");
}
function formatDuration(durationMs) {
  if (durationMs < 1e3) return `${durationMs}ms`;
  return `${(durationMs / 1e3).toFixed(durationMs < 1e4 ? 1 : 0)}s`;
}

// src/components/assistant-message.ts
var AssistantMessage = class {
  value = "";
  markdown = new Markdown("", 0, 0, markdownTheme);
  appendDelta(text) {
    this.value += text;
    this.markdown.setText(this.value);
  }
  get text() {
    return this.value;
  }
  invalidate() {
    this.markdown.invalidate();
  }
  render(width) {
    if (width <= 0) return [""];
    const label = truncateToWidth(
      `${ui.accent("\u25CF")} ${ui.muted("MiniCode")}`,
      width
    );
    const bodyWidth = Math.max(1, width - 2);
    const body = this.markdown.render(bodyWidth).map((line) => truncateToWidth(`  ${line}`, width));
    return [label, ...body];
  }
};

// src/components/reasoning-message.ts
var ReasoningMessage = class {
  value = "";
  expanded = false;
  complete = false;
  markdown = new Markdown("", 0, 0, markdownTheme);
  appendDelta(text) {
    this.value += text;
    this.markdown.setText(this.value);
  }
  finish() {
    this.complete = true;
  }
  setExpanded(expanded) {
    this.expanded = expanded;
  }
  get text() {
    return this.value;
  }
  invalidate() {
    this.markdown.invalidate();
  }
  render(width) {
    if (width <= 0) return [""];
    const state = this.complete ? "Thought" : "Thinking";
    const hint = this.expanded ? "Ctrl+O collapse" : "Ctrl+O expand";
    const label = truncateToWidth(
      `${ui.accent(">")} ${ui.muted(state)} ${ui.dim(`[${hint}]`)}`,
      width
    );
    if (!this.value.trim()) return [label];
    if (!this.expanded) {
      const preview = this.value.replace(/\s+/g, " ").trim();
      return [
        label,
        truncateToWidth(`  ${ui.dim(preview)}`, width)
      ];
    }
    const bodyWidth = Math.max(1, width - 2);
    const body = this.markdown.render(bodyWidth).map((line) => truncateToWidth(`  ${ui.dim(line)}`, width));
    return [label, ...body];
  }
};

// src/components/user-message.ts
var UserMessage = class {
  text;
  constructor(message) {
    this.text = new Text(`${ui.accentStrong("\u203A")} ${ui.bold(message)}`);
  }
  invalidate() {
    this.text.invalidate();
  }
  render(width) {
    return this.text.render(width).map((line) => truncateToWidth(line, Math.max(0, width)));
  }
};

// src/transcript.ts
var Transcript = class {
  stack = new VStack([], { gap: 1 });
  activity = null;
  assistant = null;
  reasoning = null;
  reasoningMessages = [];
  reasoningExpanded = false;
  appendUser(text) {
    this.completeReasoning();
    this.stack.addChild(new UserMessage(text));
    this.assistant = null;
  }
  appendReasoningDelta(text) {
    if (!text) return;
    if (this.reasoning === null) {
      if (this.activity !== null) {
        this.activity.complete();
        this.activity = null;
      }
      this.reasoning = new ReasoningMessage();
      this.reasoning.setExpanded(this.reasoningExpanded);
      this.reasoningMessages.push(this.reasoning);
      this.stack.addChild(this.reasoning);
    }
    this.reasoning.appendDelta(text);
  }
  completeReasoning() {
    this.reasoning?.finish();
    this.reasoning = null;
  }
  toggleReasoning() {
    this.reasoningExpanded = !this.reasoningExpanded;
    for (const reasoning of this.reasoningMessages) {
      reasoning.setExpanded(this.reasoningExpanded);
    }
    return this.reasoningExpanded;
  }
  startTool(id, action, target) {
    this.completeReasoning();
    if (this.activity === null) {
      this.activity = new Activity();
      this.stack.addChild(this.activity);
    }
    this.activity.start(id, action, target);
  }
  finishTool(id, status, summary, details = {}) {
    this.activity?.finish(id, status, summary, details);
  }
  completeActivity() {
    this.activity?.complete();
  }
  appendAssistantDelta(text) {
    this.completeReasoning();
    if (this.assistant === null) {
      this.assistant = new AssistantMessage();
      this.stack.addChild(this.assistant);
    }
    this.assistant.appendDelta(text);
  }
  startRun() {
    this.completeReasoning();
    this.activity = null;
    this.assistant = null;
  }
  get currentAssistantText() {
    return this.assistant?.text ?? "";
  }
  invalidate() {
    this.stack.invalidate();
  }
  render(width) {
    return this.stack.render(width);
  }
};

// src/app.ts
var BOTTOM_DOCK_MIN_ROWS = 4;
var MiniCodeTuiApp = class {
  transcript = new Transcript();
  header;
  footer = new Footer();
  tui;
  editor;
  running = false;
  approvalHandle = null;
  userInputHandle = null;
  panelHandle = null;
  pendingApprovalId = null;
  pendingUserInputId = null;
  onSubmit;
  onClientMessage;
  onExit;
  constructor(options = {}) {
    const terminal = options.terminal ?? new ProcessTerminal();
    this.onSubmit = options.onSubmit;
    this.onClientMessage = options.onClientMessage;
    this.onExit = options.onExit;
    this.header = new Header({
      workspace: options.workspace,
      provider: options.provider,
      model: options.model
    });
    this.tui = new TuiAltScreen(terminal, true);
    this.editor = new Composer(this.tui);
    this.editor.onSubmit = (text) => this.submitText(text);
    this.tui.addInputListener((data) => {
      if (this.panelHandle !== null && matchesKey(data, Key.escape)) {
        this.clearPanel();
        this.tui.requestRender();
        return { consume: true };
      }
      if (this.panelHandle === null && this.pendingApprovalId === null && this.pendingUserInputId === null && matchesKey(data, Key.ctrl("o"))) {
        const expanded = this.transcript.toggleReasoning();
        this.footer.setStatus(
          this.running ? "Working" : "Ready",
          this.running ? "working" : "muted",
          expanded ? "reasoning expanded" : "reasoning collapsed"
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
        return void 0;
      }
      if (matchesKey(data, Key.escape) || matchesKey(data, Key.ctrl("c"))) {
        this.onClientMessage?.({ type: "cancel" });
        this.footer.setStatus("Cancelling", "warning", "waiting for active tool");
        this.tui.requestRender();
        return { consume: true };
      }
      return void 0;
    });
    const transcriptView = new ScrollView(this.transcript, {
      follow: "end",
      primary: true,
      overscroll: "contain",
      scrollbar: "auto"
    });
    const bottomDock = new VStack([
      { component: this.editor, basis: "auto" },
      { component: this.footer, basis: "auto" }
    ]);
    const root = new VStack([
      { component: this.header, basis: "auto" },
      { component: transcriptView, basis: 0, grow: 1, minSize: 1 },
      { component: bottomDock, basis: "auto", shrink: 1, minSize: BOTTOM_DOCK_MIN_ROWS }
    ]);
    this.tui.setLayoutRoot(root);
    this.tui.setFocus(this.editor);
  }
  submitText(text) {
    const task = text.trim();
    if (!task || this.pendingApprovalId !== null || this.pendingUserInputId !== null || this.panelHandle !== null) return;
    if (!this.running && task.startsWith("/")) {
      this.editor.addToHistory(task);
      this.editor.setText("");
      this.onClientMessage?.({ type: "command", text: task });
      this.footer.setStatus(`Loading ${task}`, "working");
      this.tui.requestRender();
      return;
    }
    this.transcript.appendUser(task);
    const message = this.running ? { type: "steer", text: task } : { type: "task", text: task };
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
  start() {
    this.tui.start();
  }
  stop() {
    this.tui.stop();
  }
  handleServerEvent(event) {
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
          event.target
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
          diffTruncated: event.diff_truncated
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
          event.summary ?? event.tool
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
          "Ctrl+C exit"
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
  showApproval(event) {
    this.clearApproval();
    this.pendingApprovalId = event.id;
    this.editor.disableSubmit = true;
    const overlay = new ApprovalOverlay(event, (decision) => {
      if (this.pendingApprovalId !== event.id) return;
      this.onClientMessage?.({
        type: "approval_response",
        id: event.id,
        decision
      });
      this.clearApproval();
      this.tui.requestRender();
    });
    this.approvalHandle = this.tui.showOverlay(overlay, {
      anchor: "bottom-center",
      width: "80%",
      maxHeight: "60%",
      margin: { left: 1, right: 1, bottom: BOTTOM_DOCK_MIN_ROWS }
    });
  }
  clearApproval() {
    this.approvalHandle?.hide();
    this.approvalHandle = null;
    this.pendingApprovalId = null;
    this.editor.disableSubmit = false;
    this.editor.setMode(this.running ? "steer" : "ask");
    if (this.running) {
      this.footer.setStatus("Working", "working");
    }
  }
  showUserInput(event) {
    this.clearUserInput();
    this.pendingUserInputId = event.id;
    this.editor.disableSubmit = true;
    const overlay = new UserInputOverlay(event, (selectedIndex) => {
      if (this.pendingUserInputId !== event.id) return;
      this.onClientMessage?.({
        type: "user_input_response",
        id: event.id,
        selected_index: selectedIndex
      });
      this.clearUserInput();
      this.tui.requestRender();
    });
    this.userInputHandle = this.tui.showOverlay(overlay, {
      anchor: "center",
      width: "80%",
      maxHeight: "80%",
      margin: 1
    });
  }
  clearUserInput() {
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
  showPanel(event) {
    this.clearPanel();
    const content = new Text(
      `${ui.accentStrong(event.title)}

${event.content}

${ui.dim("Esc close")}`
    );
    const scroll = new ScrollView(content, {
      follow: "none",
      overscroll: "contain",
      scrollbar: "auto"
    });
    this.panelHandle = this.tui.showOverlay(scroll, {
      anchor: "center",
      width: "90%",
      maxHeight: "80%",
      margin: 1
    });
    this.footer.setStatus(event.title, "muted", "Esc close");
  }
  clearPanel() {
    this.panelHandle?.hide();
    this.panelHandle = null;
    this.footer.setStatus(
      this.running ? "Working" : "Ready",
      this.running ? "working" : "muted",
      this.running ? "" : "Ctrl+C exit"
    );
  }
};
function toolAction(tool) {
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

// src/main.ts
var sessionMode = process.env.MINICODE_TUI_SESSION_MODE;
var skills = JSON.parse(process.env.MINICODE_TUI_SKILLS ?? "[]");
var backend = new BackendClient({
  pythonExecutable: process.env.MINICODE_PYTHON || "python",
  workspace: process.env.MINICODE_WORKSPACE || process.cwd(),
  provider: process.env.MINICODE_PROVIDER,
  model: process.env.MINICODE_MODEL,
  writeEnabled: process.env.MINICODE_TUI_NO_WRITE !== "1",
  approvalPolicy: process.env.MINICODE_TUI_APPROVAL_POLICY,
  permissionMode: process.env.MINICODE_TUI_PERMISSION_MODE,
  collaborationMode: process.env.MINICODE_TUI_COLLABORATION_MODE,
  sandboxMode: process.env.MINICODE_TUI_SANDBOX_MODE,
  sandboxImage: process.env.MINICODE_TUI_SANDBOX_IMAGE,
  skills,
  skillsEnabled: process.env.MINICODE_TUI_NO_SKILLS !== "1",
  projectConventionsEnabled: process.env.MINICODE_TUI_NO_PROJECT_CONVENTIONS !== "1",
  subagentsEnabled: process.env.MINICODE_TUI_NO_SUBAGENTS !== "1",
  mcpConfig: process.env.MINICODE_TUI_MCP_CONFIG,
  sessionMode,
  sessionId: process.env.MINICODE_TUI_SESSION_ID
});
var app;
var shuttingDown = false;
var fatalBackendError;
var shutdown = () => {
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
  model: process.env.MINICODE_MODEL
});
app.start();
backend.start({
  onEvent: (event) => {
    if (event.type === "error" && event.fatal) {
      fatalBackendError = event.message;
    }
    app.handleServerEvent(event);
  },
  onProtocolError: (message) => app.handleServerEvent({
    type: "error",
    message: "Protocol error: " + message,
    fatal: true
  }),
  onExit: (code, signal) => {
    if (shuttingDown) return;
    if (code === 0 && signal === null) {
      app.stop();
      return;
    }
    const message = fatalBackendError ?? "Backend exited (" + (code ?? signal ?? "unknown") + ").";
    app.stop();
    process.stderr.write("MiniCode backend error: " + message + "\n");
    process.exitCode = typeof code === "number" && code !== 0 ? code : 1;
  }
});
var initialTask = process.env.MINICODE_TUI_INITIAL_TASK?.trim();
if (initialTask) {
  app.submitText(initialTask);
}
process.once("SIGTERM", shutdown);
