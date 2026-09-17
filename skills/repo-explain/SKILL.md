---
name: repo-explain
description: >
  Use this skill to explain project structure, architecture, module responsibilities, or code flow. For a whole-repository explanation, load this Skill before broad exploration. If the implementation location is unknown, search shallowly before reading direct source, and separate implementation evidence from document or name-based inference.
---

# Skill: Repo Explain

## When to use
Use this skill to explain project structure, architecture, module responsibilities, or code flow. For a whole-repository explanation, load this Skill before broad exploration. If the implementation location is unknown, search shallowly before reading direct source, and separate implementation evidence from document or name-based inference.

## Exploration rule
Start with a shallow file search for repository manifests and top-level modules, then narrow searches to the relevant module. Knowing candidate source or Artifact paths is not enough when the relevant definition, call site, or content location is unknown: search those exact resources before reading direct ranges. Do not replace location search with whole-file reads or begin with an unrestricted whole-repository recursive search.

A whole-repository explanation is representative by default, not exhaustive. Build a trustworthy architecture map first, then verify only the entry points, core workflows, data boundaries, and infrastructure mechanisms needed to explain the repository. Stop expanding once the evidence is sufficient to explain the project goal, major module boundaries, primary entry points, and the most important call or data flows. Inspect every module or class only when the user explicitly asks for exhaustive coverage.

Independent call chains or bounded subsystem questions may be delegated in the same response. Each child should receive one concrete scope and one expected result, such as an entry-to-core call chain, one subsystem workflow, or one infrastructure mechanism and its direct usage. Do not bundle several modules, unrelated workflows, or multiple independent questions into one child task.

Avoid overlapping delegation. The parent Agent combines concise child summaries and does not repeat the same exploration. Simple reads of known short files or known definitions should remain in the parent rather than being delegated.

## Evidence levels
Use the weakest evidence level that is sufficient for the claim:

1. Manifest, configuration, or documentation evidence may support module existence, declared responsibility, dependencies, build entry points, and configured integrations.
2. Direct implementation or call-site evidence is required for runtime behavior, concrete call relationships, data flow, side effects, and claims that A actually uses B.
3. Structural or name-based inference may orient exploration, but must be labeled as inference and must not be promoted to implemented behavior.

## Evidence checklist
A useful explanation usually cites:

1. Build or entry-point evidence when the question is project-level.
2. Representative controller, service, repository/mapper, configuration, or utility files needed for the core workflow rather than every class in the repository.
3. The data boundary involved in the behavior, such as Redis keys, database mapper calls, external clients, or DTO/entity conversion.
4. Existing tests only when they clarify observable behavior.
5. When the question assumes A uses B, inspect A's direct implementation or call site; the existence of B alone does not prove the relationship.

## Explanation shape
- State the inspected files or tool results used as evidence.
- Explain module responsibilities and call/data flow from the inspected code.
- Distinguish direct implementation findings from build/document declarations and name-based inference.
- Do not promote a README claim, class name, or module name to implemented behavior without direct implementation or call-site evidence.
- Prefer a concise architecture map plus a small number of representative verified flows over a module-by-module inventory unless the user asks for exhaustive detail.
