# `/audit` — Adversarial Review Skill: Design

**Date:** 2026-08-17 · **Status:** approved by owner (brainstorming session)
**Origin:** distilled from the AUDIT-2/AUDIT-3 hardening epic and its slice-4 meta-review,
which found three live defects in the epic's own fixes plus eight unproven robustness claims.
The skill packages the methodology that found them so it can be invoked on any diff.

## Purpose

On-demand, audit-grade adversarial review of code changes, with prevention mechanisms that
stop masked-defect classes from recurring. The skill's premise, proven repeatedly this epic:
**green gates are not evidence of absence — the review hunts what tests cannot see** (paths
routing never reaches, claims no leg exercises, tolerances that swallow regressions, names
that resolve to the wrong thing).

## Deliverables

1. `~/.claude/commands/audit.md` — the skill (global, single file, matching the user's
   existing command style: metal.md, pee.md, bench.md).
2. `docs/audits/REVIEW-PATTERNS.md` in VectorAccelerate — the per-repo masking-pattern
   library, seeded from this epic (see §5).
3. A ~5-line addition to the user's global `~/.claude/CLAUDE.md` — the claim-coverage and
   class-closure disciplines applied during implementation, not just review (verbatim text
   in §6).

Nothing is committed by the implementation; the user decides commits. (The skill file lives
outside any repo regardless.)

## 1. Invocation and tiers

```
/audit [scope]          # light tier (default)
/audit deep [scope]     # panel tier
```

**Scope resolution** (both tiers): no scope → the working diff (`git diff HEAD` plus
untracked source files — excluding anything .gitignored and obvious non-source artifacts)
when the tree is dirty, else the branch diff vs the merge-base
with the default branch. A scope argument is classified: `#?\d+` → GitHub PR via `gh`;
contains `..` → git range; otherwise → path filter(s) on the working diff.

**Light tier:** the assistant applies the §3 methodology inline — no subagents. Output: the
§4 report (triaged findings + claim-coverage table + class-closure notes). No fixes are
applied; the report ends by offering a remediation slice.

**Deep tier:** dispatch 2–3 adversarial subagent reviewers (§4), then verify-then-triage:
every Critical and Important finding is reproduced/verified first-hand by the assistant
before it reaches the report (Minor findings spot-checked); the report groups findings for
owner decision in the AUDIT ledger format. Remediation is always a separate slice after the
owner picks groups — the deep tier never edits production code.

**Cost note in the skill:** deep tier ≈ hundreds of thousands of subagent tokens and
~20 minutes on a 45-file diff (measured: slice 4). Invoke it for epics, merges, and
"make sure" moments, not routine diffs.

## 2. Partitioning rule (deep tier)

Partition the diff into disjoint reviewer scopes by concern, 3 when the diff spans three or
more concerns, else 2. Domain-aware default for GPU/Metal repos: (A) kernel/shader corpus,
(B) host/infra (dispatch, resolution, caches, routing), (C) tests. Generic default:
(A) production logic, (B) interfaces/integration/config, (C) tests. **The test-honesty
reviewer is never dropped** — it was the highest-yield perspective in slice 4.

## 3. Methodology core — the six Iron Questions

Asked of every reviewed change; the light tier applies them directly, the deep-tier
templates carry them verbatim.

1. **Liveness.** What actually reaches this code? Trace dispatch geometry, routing
   thresholds, feature/config gates, fallback rescues. **Re-derive reachability whenever
   any gate changed in the diff** — a correct gate change elsewhere can arm a dormant defect
   (VA2-003's k-gate exemption arming the VA3-032 phantom). Liveness greps use suffix
   wildcards, never exact-quoted full names (the `warp_select_small_k_*` lesson).
2. **Claim-coverage.** Enumerate every correctness/robustness claim the diff adds or touches
   (comments, doc strings, ledger text — "works for ANY width", "thread-safe", "handles
   NaN"). Each maps to a covering test leg (file:line) or is flagged **UNCOVERED — a
   finding, not a note**. The claim-coverage table is a mandatory report section.
3. **Revert test.** For every new or changed test: name the smallest production regression
   (ideally one token) that flips it red, and at which parameter. No answer = the test is
   theater. Controls must assert the same properties as red legs.
4. **Tolerance honesty.** For every threshold/accuracy value: estimate the smallest
   plausible regression (one lane, one element, one slot — not the full defect) against the
   tolerance. Prefer structural assertions that make regressions integer-visible: exact
   integers below 2²⁴, poisoned buffers/sentinel tails, all-equal inputs, snapshot deltas
   instead of cumulative shared counters.
5. **Masking patterns.** Hunt the universal list — correctness-by-geometry/config (right
   only under the caller's exact shape); comments claiming robustness the code lacks;
   silent fallback swallowing errors into plausible results; vacuous assertions against
   shared/cumulative state; happy-path-only routing in tests (thresholds pinning GPU/CPU
   legs); phantom names (identifiers resolving to nothing, or to the wrong thing); divergent
   dual-build paths (per-file vs combined compile) — **plus the per-repo
   `docs/audits/REVIEW-PATTERNS.md` when present.** New patterns discovered by a review are
   appended to that file.
6. **Class closure.** Every accepted finding answers: *is this an instance of a class?*
   When a class has ≥2 instances, propose the mechanical test that makes the whole class
   impossible (name round-trips, literal-resolution closure, directory-tiling completeness),
   with shrink-only allowlists for recorded open items.

**Finding format** (both tiers, the AUDIT ledger row): severity (P1 wrong results/UB/data
loss on live paths · P2 latent or one-gate-away · P3 hygiene) + liveness (LIVE / LIVE-cond /
LATENT / DEAD) + mechanism + **concrete failure scenario** (inputs/state → wrong output).

## 4. Deep-tier reviewer templates

Three parameterized prompts baked into the skill ({SCOPE_FILES}, {DIFF_CMD}, {LEDGER_PATH},
{PATTERN_FILE}, {GATE_STATUS}), generalized from slice 4's. Rules baked into each:

- Read-only: no file mutations, no builds, no test runs, no git-state changes; untracked
  files must be Read directly (diffs won't show them).
- Read the diff hunk-by-hunk AND the full post-change unit around every hunk.
- Verify against the plan/ledger **in both directions**: does code deliver the claims; do
  the claims overstate the code.
- Framing: "the gates are already green ({GATE_STATUS}) — passing tests are not evidence;
  hunt what the gates cannot see."
- Output: Strengths / Critical / Important / Minor with file:line + mechanism + concrete
  failure scenario + fix; the claim-coverage table; a clear verdict.

**Post-panel rule:** reviewer findings are hypotheses. The assistant verifies each
Critical/Important first-hand (reproduce the mechanism, or refute it with evidence) before
the report; refuted findings are recorded as refuted, not dropped. Reviewers can also be
wrong about *shielding* — slice 4's Swift reviewer corrected the ledger's own reachability
claims twice.

## 5. `REVIEW-PATTERNS.md` — per-repo pattern library

Format: one pattern per entry — name, the mechanism, the tell (what to grep/look for), the
epic incident that proved it. The skill loads it when present and appends new patterns.
VectorAccelerate's seed list (from AUDIT-2/3 + slices 1–4):

1. Dispatch-geometry shielding — kernel correct only under the host's exact threadgroup
   shape (VA3-002/-013/-014).
2. Dual-compile divergence — debug per-file metallib vs release combined-source TU: symbol
   visibility, EPSILON macro shadowing, stale-metallib header edits (VA2-013, VA3-012).
3. Routing-gate masking — GPU legs invisible to tests because thresholds route small test
   inputs to CPU; reachability must be re-derived when gates change (VA3-031/-032).
4. Phantom names — literals resolving to no kernel, or through rewriting derivations to the
   WRONG kernel; closed mechanically by the name-closure tests (5 instances).
5. Pool stale-bytes — unwritten result buffers read back as answers; poison before dispatch
   (VA3-031/-034, VA3-008).
6. Pow2-only test data — non-pow2 dims/widths never exercised; "robustness" comments on
   orphaning trees (VA3-014).
7. Silent CPU fallback — provider rescues convert GPU breakage into correct-looking results;
   assert provenance/telemetry (AUDIT-2 foundational).
8. Fast-math idiom fragility — sqrt(a)·sqrt(b) reassociation, folded isinf/isfinite,
   subnormal-reciprocal flush; use the established idioms (VA2-008, VA3-015).
9. Cumulative shared-state assertions — process-shared pools/counters make `>= N` vacuous;
   assert deltas (Priority2 flake, slice 4).
10. Value-policy divergence across legs of one API — NaN/overflow answers differing by batch
    size or routing (VA3-033, VA3-016).

## 6. Global CLAUDE.md addition (verbatim)

> ## Review & claim discipline
> - Any correctness or robustness claim written in a comment ("works for any N",
>   "thread-safe", "handles NaN") must name the test that exercises it, be derivable in one
>   step from adjacent code, or be written as an explicit unverified contract — never as
>   fact. Unproven claims are defects waiting to be believed.
> - When a defect is an instance of a class (second occurrence anywhere), add the mechanical
>   test that closes the class, not just the instance fix.
> - When a routing/config gate changes, re-derive reachability for everything the gate
>   shields — a correct gate change can arm a dormant defect elsewhere.

## Ledger convention (deep tier)

Deep-tier findings are recorded in the repo's `docs/audits/` ledger in the AUDIT format
(summary index rows + narrative with red evidence and gate numbers). Corrections to earlier
entries are recorded as corrections — never silently rewritten. If no ledger exists, the
report offers to start one.

## Non-goals / notes

- The skill does not replace red-first TDD (superpowers:test-driven-development governs
  fixing); it produces the findings that fixes then close red→green.
- CI two-leg gating (debug + release `swift test`) is a standing recommendation the skill
  may note, not a feature — infra decision deferred by owner (AUDIT-2 decision 5).
- `/code-review` and `superpowers:requesting-code-review` remain available; `/audit` is the
  audit-grade tier above them, not a replacement.

## Open items

None — all five design decisions resolved in the brainstorming session (tiered/light
default; global + per-repo patterns; class-closure + claim-coverage + ledger prevention;
verify-then-triage endpoint; single-file packaging).
