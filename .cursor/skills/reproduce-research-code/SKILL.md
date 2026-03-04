---
name: reproduce-research-code
description: Guides modifications in repositories that reproduce academic research. Preserves original implementation behavior, matches reference code and paper, and keeps changes minimal and reproducible. Use when working in a research-reproduction repo, when the user mentions the paper or reference implementation, or when correctness and reproducibility are priorities.
---

# Reproduce Research Code

When working in this repository (or any research-reproduction codebase), follow these guidelines. Correctness and fidelity to the original work take precedence over refactoring or "improvements."

## Core principles

1. **Preserve the behavior of the original implementation.** Do not change what the code does unless the user explicitly asks. Bug fixes that restore intended behavior are allowed; "cleanups" that alter behavior are not.

2. **Do not change algorithm logic unless explicitly requested.** Match the paper and reference implementation: same formulas, same hyperparameters, same ordering of operations. If something looks suboptimal, do not "fix" it without asking.

3. **Prefer matching the reference repository and paper exactly.** When in doubt, align with the official reference code and the written description in the paper. Document any intentional deviations and why they exist.

4. **Any modification must maintain reproducibility.** Changes must not break determinism (e.g. fixed seeds), environment (dependencies, versions), or documented run instructions. If you change dependencies or behavior, update docs and ensure results can be reproduced.

5. **When uncertain, ask for clarification instead of redesigning.** Do not guess the author's intent or substitute a different algorithm. Ask which behavior or reference to follow before changing logic or structure.

## In practice

- **Fixes and patches:** Prefer the smallest edit that restores correct behavior. Avoid rewriting functions or modules.
- **New features or experiments:** Implement to match the paper/reference first; optimize or generalize only if the user requests it.
- **Dependencies:** Prefer versions that match the reference repo or paper. If you must upgrade, note it and check that outputs remain reproducible.
- **Code style:** Consistency with the existing codebase is secondary to preserving correctness and matching the reference.

## Summary

Focus on **correctness** and **minimal changes**. Preserve original behavior; match the paper and reference; keep everything reproducible; ask when unsure.
