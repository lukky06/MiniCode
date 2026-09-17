---
name: web-controller-test
description: Use this skill when the task asks to test an HTTP route, controller, request handler, resolver, middleware boundary, or API endpoint in any web framework.
---

# Skill: Web Controller Test Generation

## When to use
Use this skill when the task asks to test an HTTP route, controller, request handler, resolver, middleware boundary, or API endpoint in any web framework.

## Boundary evidence
Before writing tests, identify:

1. Route, method, path, request fields, headers, and authentication requirements.
2. Validation and serialization behavior owned by the web layer.
3. Delegated application or domain calls and their observable arguments.
4. Response status, body, headers, and error mapping.
5. Framework-supported test client or handler invocation pattern already used by the repository.
6. Whether the behavior requires a full application container or can be isolated with a lightweight request harness.

## Test strategy
- Prefer the smallest framework-supported HTTP test harness that exercises routing, binding, validation, and response mapping.
- Mock downstream services when the test concerns only the web boundary.
- Use a full application or integration environment only when middleware, dependency injection, transaction behavior, framework plugins, or real infrastructure is part of the behavior under test.
- Follow existing repository conventions for fixtures, clients, dependency overrides, setup, and teardown.
- Test observable behavior rather than private helper methods.

## Verification
Run the narrowest allowed command that executes the new or changed test. If test discovery or framework startup fails, repair the test setup before expanding scope.
