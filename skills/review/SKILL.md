---
name: review
description: Use this skill to review an existing Git diff for correctness, regressions, unsafe behavior, stale assumptions, missing validation, and focused test gaps before the change is accepted.
---

# Skill: Code Review

## When to use
Use this skill to review an existing Git diff for correctness, regressions, unsafe behavior, stale assumptions, missing validation, and focused test gaps before the change is accepted.

## Review scope
Review only the current diff and the smallest surrounding source or tests needed to judge it. Keep the review read-only.

## Evidence order
1. Inspect the current Git diff first.
2. Read changed functions, classes, configuration, or tests that directly determine the changed behavior.
3. Inspect callers or adjacent state only when the finding depends on them.
4. Use repository tests and conventions as evidence when they define intended behavior.
5. Stop expanding scope once a finding can be proven or dismissed.

## Finding standard
Report only findings that are actionable and supported by repository evidence.

Use these severities:
- `P0`: data loss, security issue, destructive behavior, or a change that makes the repository broadly unusable.
- `P1`: likely functional regression, broken recovery or persistence semantics, incorrect state transition, or a focused test that should fail.
- `P2`: meaningful maintainability or robustness issue that can cause future defects, with a concrete local fix.

Do not report style-only preferences, speculative concerns, or unrelated pre-existing issues.

## Review discipline
- Prefer one precise finding over several weak findings.
- Include the affected file and symbol or local region when known.
- Explain the concrete failure mode and the condition that triggers it.
- Check whether an existing focused test already covers the behavior before claiming a missing-test issue.
- Do not modify files, run mutation tools, start worktree workers, or suggest broad redesigns unless the diff itself makes them necessary.

## Output convention
If findings exist, list them in severity order using concise entries such as:
`P1 path/to/file.py: concrete failure mode and trigger.`

Finish with one gate recommendation:
- `Gate recommendation: FAIL` when any P0 or P1 remains.
- `Gate recommendation: PASS WITH NOTES` when only P2 findings remain.
- `Gate recommendation: PASS` when no actionable finding is supported.
