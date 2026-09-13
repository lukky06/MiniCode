import { build } from "esbuild";
import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const tuiDir = resolve(scriptDir, "..");
const repoRoot = resolve(tuiDir, "..");
const runtimeDir = join(repoRoot, "minicode_harness", "tui_runtime");
const piTuiDir = join(tuiDir, "node_modules", "@earendil-works", "pi-tui");

await mkdir(runtimeDir, { recursive: true });

await build({
  entryPoints: [join(tuiDir, "src", "main.ts")],
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node22",
  outfile: join(runtimeDir, "main.mjs"),
  sourcemap: false,
  minify: false,
  legalComments: "none",
});

const nativeAssets = [
  ["darwin", "prebuilds", "darwin-arm64", "darwin-modifiers.node"],
  ["darwin", "prebuilds", "darwin-x64", "darwin-modifiers.node"],
  ["win32", "prebuilds", "win32-arm64", "win32-console-mode.node"],
  ["win32", "prebuilds", "win32-x64", "win32-console-mode.node"],
];

for (const parts of nativeAssets) {
  const source = join(piTuiDir, "native", ...parts);
  const destination = join(runtimeDir, "native", ...parts);
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
}
