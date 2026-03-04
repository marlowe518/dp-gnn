# doit

IMPORTANT: Re-scan the current dev project before making any changes.

Please do the following BEFORE writing any code:
1) Read ENVIRONMENT.md and follow it strictly (no new dependencies).
2) Re-scan the entire current dev project workspace to get the latest structure and implementations.
   - Identify existing utilities/modules we can reuse (no duplication).
3) Inspect the reference repository at: /reference_repo/differentially_private_gnns
   - Locate the reference implementation relevant to the upcoming task.
4) Produce:
   A) A brief current-project inventory (key files/modules related to the task)
   B) A reuse plan (what to reuse vs what to add/patch)
   C) A reference-side dependency closure list for the target functionality

After you present (A–D), proceed to implement the changes using minimal engineering:
- Full functional parity for the target behavior
- Provide an independent functional test script under tests/functional/