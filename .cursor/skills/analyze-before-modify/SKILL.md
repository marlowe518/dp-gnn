---
name: analyze-before-modify
description: When modifying or implementing code, the agent must first analyze repository structure, identify relevant files, and explain the existing implementation before proposing any changes. Ensures minimal, informed edits and prevents large code changes without prior analysis. Use when the user asks to modify code, implement a feature, fix a bug, or change existing behavior.
---

# Analyze Before Modify

When asked to modify or implement code, follow this order. Do not propose or generate code changes until analysis is complete.

## 1. Analyze repository structure

- Inspect the project layout (directories, entry points, config files).
- Identify where the requested change fits (which package, module, or layer).
- Note build/test/config patterns (e.g. `pyproject.toml`, `src/`, `tests/`).

## 2. Identify relevant files

- Search or navigate to files that implement the current behavior or are the right place for the change.
- Prefer reading existing code over guessing; list the specific files that will be touched.
- If the scope is unclear, name the files you need to read before proposing edits.

## 3. Explain the existing implementation

- Summarize how the relevant code works (data flow, key functions, interfaces).
- Call out any constraints, dependencies, or patterns that affect the change.
- State what must stay the same and what the change is meant to alter.

## 4. Propose minimal changes only after analysis

- After steps 1–3, propose the smallest change that achieves the goal.
- Prefer patching existing code over rewriting; avoid new abstractions unless they reduce complexity.
- Do not generate large diffs or refactors without having done the analysis above.

## Rules

- **Never** generate large or broad code changes without prior analysis.
- **Always** complete analysis (structure → relevant files → existing behavior) before writing or suggesting code.
- If the codebase or request is ambiguous, state what you need to read or clarify before proposing changes.
