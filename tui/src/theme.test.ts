import assert from "node:assert/strict";
import test from "node:test";

import { ui } from "./theme.js";

test("foreground styles preserve an enclosing modal background", () => {
  const rendered = ui.surface(
    `before ${ui.accentStrong("Sessions")} ${ui.muted("Esc close")} after`,
  );

  assert.match(rendered, /\x1b\[48;2;24;31;39m/);
  assert.doesNotMatch(rendered, /\x1b\[0m/);
  assert.match(rendered, /\x1b\[49m$/);
});
