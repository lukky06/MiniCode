import type { EditorTheme, MarkdownTheme, SelectListTheme } from "@earendil-works/pi-tui";

type Paint = (text: string) => string;

function sgr(code: string, resetCode: string): Paint {
  return (text: string): string =>
    text ? `\x1b[${code}m${text}\x1b[${resetCode}m` : "";
}

export const ui = {
  accent: sgr("38;2;138;173;244", "39"),
  accentStrong: sgr("1;38;2;138;173;244", "22;39"),
  text: (text: string): string => text,
  muted: sgr("38;2;127;140;152", "39"),
  dim: sgr("2", "22"),
  success: sgr("38;2;123;216;143", "39"),
  warning: sgr("38;2;235;203;139", "39"),
  error: sgr("38;2;243;139;168", "39"),
  border: sgr("38;2;95;105;115", "39"),
  surface: sgr("48;2;24;31;39", "49"),
  code: sgr("38;2;235;203;139", "39"),
  bold: sgr("1", "22"),
  italic: sgr("3", "23"),
  underline: sgr("4", "24"),
  strike: sgr("9", "29"),
} satisfies Record<string, Paint>;

const selectList: SelectListTheme = {
  selectedPrefix: ui.accentStrong,
  selectedText: ui.accentStrong,
  description: ui.muted,
  scrollInfo: ui.dim,
  noMatch: ui.warning,
};

export const editorTheme: EditorTheme = {
  borderColor: ui.border,
  selectList,
};

export const markdownTheme: MarkdownTheme = {
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
  underline: ui.underline,
};
