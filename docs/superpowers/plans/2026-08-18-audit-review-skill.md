# /audit Adversarial Review Skill — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `/audit` skill (global command), seed VectorAccelerate's `REVIEW-PATTERNS.md`, and add the claim-discipline block to the global CLAUDE.md — per `docs/superpowers/specs/2026-08-17-adversarial-review-skill-design.md`.

**Architecture:** Three standalone markdown artifacts. The skill is a single self-contained command file in the user's existing `~/.claude/commands/` style, holding tier logic, the Six Iron Questions, three parameterized deep-tier reviewer templates, output contracts, and prevention duties. The pattern file and CLAUDE.md block are its per-repo and per-implementation companions.

**Tech Stack:** Markdown only. Verification is structural (grep for required sections/tokens) plus a live light-tier smoke test.

## Global Constraints

- **NO COMMITS.** The owner decides all commits (standing rule; overrides this plan template's usual commit steps). Every task ends with verification instead.
- Deliverable paths are exact: `~/.claude/commands/audit.md`, `<repo>/docs/audits/REVIEW-PATTERNS.md`, append-only edit to `~/.claude/CLAUDE.md`.
- The CLAUDE.md edit must be idempotent: skip if a `## Review & claim discipline` heading already exists.
- Content below is verbatim-final: fix typos if found, but semantic edits require owner sign-off.
- Never stage this repo's untracked junk (`.antigravitycli/`, `GEMINI.md`, `consolidated_library.md`, `generate_consolidation.py`).

---

### Task 1: Write `~/.claude/commands/audit.md`

**Files:**
- Create: `~/.claude/commands/audit.md`

**Interfaces:**
- Produces: the `/audit` skill. Referenced tokens later tasks rely on: the pattern-file path `docs/audits/REVIEW-PATTERNS.md` (Task 2 creates it for this repo) and the CLAUDE.md discipline block (Task 3) which the skill's "Prevention duties" section cross-references.

- [ ] **Step 1: Write the file with exactly this content**

````markdown
# /audit - Adversarial Audit-Grade Code Review

Review changes the way an auditor hunts, not the way a colleague skims. Operating premise,
proven across the AUDIT-2/3 hardening epics: **green gates are not evidence of absence.**
Tests prove the paths they reach with the data they use; this review hunts what they cannot
see — paths routing never takes, claims no test leg exercises, tolerances that swallow
regressions, names that resolve to nothing or to the wrong thing.

## Usage

```
/audit [scope]          # light tier (default): inline review, no subagents
/audit deep [scope]     # panel tier: 2-3 adversarial subagent reviewers, then verify-then-triage
```

**Scope resolution** (both tiers):
- No scope → the working diff: `git diff HEAD` plus untracked source files (respect
  .gitignore; skip obvious non-source artifacts) when the tree is dirty; else the branch
  diff vs `git merge-base` with the default branch.
- `#N` or bare `N` → GitHub PR N (`gh pr diff N`, `gh pr view N`).
- Contains `..` → git range, reviewed via `git diff A..B`.
- Anything else → path filter(s) applied to the working diff.

**Cost calibration:** the deep tier runs multiple subagents — on a 45-file diff it measured
in the hundreds of thousands of subagent tokens and ~20 minutes. Use it for epics, merges,
and "make sure" moments. The light tier is for routine diffs and costs only inline effort.

## Non-negotiable rules (both tiers)

1. **The review applies no fixes.** Output is findings; remediation is a separate slice the
   owner scopes after triage. (Fixing then follows red-first TDD: reproduce the mechanism
   failing before changing production code.)
2. **Finding format** — every finding carries all four, or it is not a finding:
   - Severity: **P1** wrong results / UB / data loss reachable on live paths · **P2** latent
     or one config/gate away, policy inconsistencies · **P3** hygiene, dead code, perf, docs.
   - Liveness: **LIVE** / **LIVE-cond** (behind one config or API step) / **LATENT**
     (shielded by the current caller's exact behavior) / **DEAD** (no call path).
   - Mechanism: why the code is wrong, at the level of the actual operations.
   - Concrete failure scenario: inputs/state → wrong output. "Could be a problem" is not a
     scenario.
3. **Read the full unit, not the hunk.** A hunk can be locally sound and wrong in its
   surrounding control flow. Read the complete post-change function/kernel/type around every
   hunk reviewed.
4. **Load the repo's pattern library.** If `docs/audits/REVIEW-PATTERNS.md` exists, its
   entries are mandatory checklist items. When a review discovers a NEW masking pattern,
   append it there (name, mechanism, tell, incident) and say so in the report.
5. **Liveness greps use suffix wildcards, never exact-quoted full names** — prefix-named
   variants have been misclassified dead by exact-name greps before.
6. **Corrections are recorded, never rewritten.** If the review finds an earlier record
   (ledger, comment, report) was wrong, the correction is itself a documented finding.

## The Six Iron Questions

Ask all six of every reviewed change. They are ordered by yield.

**Q1 — Liveness: what actually reaches this code?**
Trace the real callers: dispatch geometry, routing thresholds, feature flags, fallback
rescues, decision engines. Then check the diff for changes to ANY gate — and re-derive
reachability for everything that gate shields. A correct gate change elsewhere can arm a
dormant defect: the measured case was a k-gate exemption (its own valid fix) arming a
phantom-kernel throw on a public API two modules away. Shielding claims from earlier
records are re-derived, not trusted.

**Q2 — Claim-coverage: is every claim proven?**
Enumerate every correctness/robustness claim the diff adds or touches — comments, doc
strings, commit/ledger text: "works for ANY width", "thread-safe", "handles NaN",
"dispatch-robust". For each, find the covering test leg (file:line) or the one-step
derivation from adjacent code. Anything else is **UNCOVERED — a finding, not a note**. The
claim-coverage table is a mandatory report section. (This question found a rewritten
multi-threadgroup code path with zero value assertions — the fix was correct, the claim was
unproven, and a one-token regression would have shipped green.)

**Q3 — Revert test: what flips each test red?**
For every new or changed test, name the smallest production regression — ideally one token —
that makes it fail, and at which parameter. No answer = the test is theater. Controls must
assert the same properties as the red legs (a control that asserts less is decoration).
Watch for: assertions against cumulative shared state (vacuous in-suite — assert deltas),
tests reaching the code only through routing that a threshold change silently re-routes,
and oracle values printed in failure messages but never asserted.

**Q4 — Tolerance honesty: what hides inside the thresholds?**
For every accuracy/epsilon/tolerance: estimate the SMALLEST plausible regression — one lane,
one element, one slot, not the full defect — and check it against the threshold. Prefer
structural assertions that make regressions integer-visible: exact integers below 2^24 in
FP32, all-equal inputs whose aggregate is exact, poisoned buffers and sentinel tails
(unwritten output must read back as poison, never as a plausible stale value), bitwise
compares for max/min/count.

**Q5 — Masking patterns: how does this codebase get fooled?**
Universal list; the repo pattern file extends it:
- Correctness-by-geometry/config: code right only under the caller's exact shape (one
  threadgroup, pow2 width, fixed batch) with nothing enforcing the shape.
- Comments claiming robustness the code lacks — the comment is the mask.
- Silent fallback: rescue paths converting breakage into plausible results (GPU→CPU,
  cache-miss→recompute, error→default).
- Phantom names: identifiers resolved at runtime (function names, keys, routes) that match
  nothing — or worse, are rewritten to match the WRONG thing.
- Divergent dual-build paths: per-file vs combined compilation, debug vs release resource
  loading — code correct in the tested path and broken in the shipped one.
- Vacuous assertions: shared counters, order-dependent state, `>= N` against a pool warmed
  by earlier tests.
- Value-policy divergence across legs of one API: NaN/overflow/tie answers that differ by
  batch size, routing, or precision path.

**Q6 — Class closure: is this one bug or a family?**
Every accepted finding answers: is this an instance of a class? At the SECOND instance of a
class, propose the mechanical test that makes the whole class impossible — name round-trip
tests, literal-resolution closure against the real corpus, directory-tiling completeness
checks — with shrink-only allowlists for recorded open items. Classes end by construction,
not vigilance.

## Light tier procedure

1. Resolve scope; enumerate the diff AND untracked files in scope (diffs do not show
   untracked files — read them directly).
2. Read hunk-by-hunk plus the full post-change unit around every hunk.
3. Apply the Six Iron Questions. Verify each candidate finding against the actual code
   before reporting it.
4. Report (see Output contract). End by offering a remediation slice — do not start one.

## Deep tier procedure

1. Resolve scope. Partition the diff into disjoint reviewer scopes by concern: three
   partitions when the diff spans three or more concerns, else two. GPU/Metal repos:
   (A) kernel/shader corpus, (B) host/infra — dispatch, name resolution, caches, routing,
   (C) tests. Generic: (A) production logic, (B) interfaces/integration/config, (C) tests.
   **Never drop the test-honesty reviewer** — it is historically the highest-yield seat.
2. Dispatch the reviewers in parallel using the templates below, filling every {PARAM}.
   Reviewers are read-only and must not build, run tests, or touch git state.
3. **Verify-then-triage:** reviewer findings are hypotheses. Reproduce or refute every
   Critical and Important finding first-hand (read the code, trace the path, hand-derive
   the mechanism); spot-check Minors. Refuted findings appear in the report as refuted with
   the evidence — never silently dropped.
4. Report in owner-decision format: findings grouped for scoping (by fix-shape or
   subsystem), each group with severity span and effort note. If the repo keeps a
   `docs/audits/` ledger, record the review there in the ledger's format; if none exists,
   offer to start one. Then STOP — no production edits until the owner picks groups.

## Deep-tier reviewer templates

Fill: {SCOPE_FILES} = the partition's file list · {DIFF_CMD} = exact diff command(s) ·
{UNTRACKED} = untracked in-scope paths to Read directly · {PLAN_OR_LEDGER} = spec/ledger
path(s) if any · {PATTERN_FILE} = repo pattern library path if present · {GATE_STATUS} =
current test/gate results (e.g. "debug 1577/0, release 1577/0").

Common preamble for all three (prepend verbatim):

```
You are a senior adversarial code reviewer applying an audit-grade methodology. The changes
under review already pass their gates ({GATE_STATUS}) — passing tests are NOT evidence of
correctness; your job is to find what the gates cannot see. Review {DIFF_CMD} restricted to
{SCOPE_FILES}; also Read these untracked files directly (diffs do not show them):
{UNTRACKED}. Read every hunk AND the full post-change unit around it. If {PLAN_OR_LEDGER}
exists, verify in BOTH directions: does the code deliver what the record claims, and does
the record overclaim what the code does. If {PATTERN_FILE} exists, its entries are
mandatory checklist items.

READ-ONLY: do not modify files, do not build, do not run tests, do not touch git state.

Report format — your final message is the deliverable, plain data:
### Strengths (specific, file:line)
### Issues — #### Critical / #### Important / #### Minor
For each: file:line (post-change), the defect, severity + liveness
(LIVE/LIVE-cond/LATENT/DEAD), the mechanism, a CONCRETE failure scenario (inputs/state →
wrong output), and the fix.
### Claim-coverage table — every correctness/robustness claim the diff adds or touches →
covering test leg (file:line) or UNCOVERED.
### Assessment — ready to merge? Yes / No / With fixes + two sentences.
```

### Template A — production logic

```
Your seat: PRODUCTION LOGIC. For every algorithm the diff adds or rewrites, re-derive
correctness by hand at adversarial parameters — minimum and maximum sizes, non-powers-of-two,
one-element and empty inputs, boundary multiples of any internal block/tile/group size, and
shapes only a caller-you-haven't-seen could produce. Check: invariant preservation across
refactors; boundary and bounds arithmetic (including integer overflow in index math);
numeric policy drift (NaN/±Inf propagation, tie-breaking, rounding, accumulation order,
flush-to-zero sensitivity); concurrency and synchronization discipline (every barrier/lock
reached uniformly, no early return around synchronization, read-after-reduce reuse hazards);
and divergence between build/deployment variants of the same logic (per-file vs combined
compilation, debug vs release resource paths — code can be correct in the tested variant and
broken in the shipped one). For every robustness claim in new comments: prove it by
derivation or find the covering test; claims you cannot prove or find covered are findings.
```

### Template B — interfaces and infrastructure

```
Your seat: INTERFACES AND INFRASTRUCTURE. Hunt defects that live between units, not inside
them. (1) Name/key resolution: enumerate every runtime-resolved identifier in scope
(function-name lookups, cache keys, routes, registry entries) and resolve each against the
real corpus — flag phantoms (resolve to nothing) and hijacks (rewritten to the WRONG real
target); check collision semantics where two entry points share a cache slot expecting
different results. (2) Routing and gates: for every threshold, flag, or decision gate the
diff touches, re-derive reachability for everything the gate shields — a correct gate change
can arm a dormant defect elsewhere; do not trust recorded shielding claims, re-derive them.
(3) Caller/callee contracts: for every dispatch/call site of changed code, verify the
caller's geometry/arguments satisfy the unit's documented contract, including edge shapes
(zero, one, maximum). (4) Silent fallback: find rescue paths that convert breakage into
plausible results; verify claimed-loud paths actually throw/log to the caller. (5) Deletion
collateral: for everything the diff removes, grep (suffix wildcards, never exact-quoted full
names) for surviving references — including build manifests, warm lists, and registries.
```

### Template C — test honesty

```
Your seat: TEST HONESTY. Determine what these tests would FAIL to catch. Read the production
code each test claims to pin — you cannot judge a test's teeth without the code it bites.
Per test: (1) Revert test — name the smallest production regression (ideally one token) that
flips it red, and at which parameter; no answer = theater. (2) Tolerance honesty — estimate
the smallest plausible regression (one lane/element/slot) against every threshold; flag any
tolerance that swallows it. (3) Controls — do pow2/happy-path/control legs assert the SAME
properties as the red legs? (4) Coverage vs claims — build the claim-coverage table: every
robustness claim in the changed production code → the test leg exercising it, or UNCOVERED;
pay special attention to code paths reachable only through routing (would a threshold change
silently turn this into a self-comparison?). (5) Vacuous assertions — shared/cumulative
state, order dependence, assertions that pass with the feature deleted. (6) Oracle
independence — are reference values computed independently and ASSERTED, or merely printed
in failure messages? (7) Deleted tests — did anything with actual teeth get removed?
```

## Output contract

Light tier report:

```
## /audit report — <scope> (light)
### Verdict — one paragraph: overall risk, the one thing to fix first.
### Findings — P1 / P2 / P3 subsections; each finding: file:line, liveness, mechanism,
    concrete failure scenario, proposed fix.
### Claim-coverage table — claim → covering leg (file:line) | UNCOVERED.
### Class-closure proposals — for any finding family with ≥2 instances: the mechanical test.
### Pattern-file updates — new masking patterns appended (or "none").
### Offer — proposed remediation slice, awaiting scope decision.
```

Deep tier adds: per-reviewer verdicts, the verify-then-triage disposition per finding
(CONFIRMED first-hand / REFUTED + evidence), findings grouped for owner scoping with effort
notes, and the ledger entry (or the offer to start a ledger).

## Prevention duties (every invocation, both tiers)

- Apply the class-closure rule (Q6): second instance of any class → propose the closing test.
- Enforce the claim-coverage discipline (Q2) — and remember it also applies when WRITING
  code, per the global CLAUDE.md "Review & claim discipline" section: never write an
  unproven robustness claim as fact.
- Append newly discovered masking patterns to the repo's `docs/audits/REVIEW-PATTERNS.md`
  (create it on first pattern if absent, after telling the owner).
- Standing recommendation to surface when relevant (not a task of this skill): CI should run
  BOTH test legs (debug and release) — release-only code paths (combined-source compiles,
  stripped debug scaffolding) are otherwise invisible to automation.
```
````

- [ ] **Step 2: Verify structure**

Run:
```bash
grep -c "^\*\*Q[1-6] — " ~/.claude/commands/audit.md          # expect 6
grep -c "^### Template [ABC] — " ~/.claude/commands/audit.md   # expect 3
grep -n "{GATE_STATUS}\|{SCOPE_FILES}\|{DIFF_CMD}\|{UNTRACKED}\|{PLAN_OR_LEDGER}\|{PATTERN_FILE}" ~/.claude/commands/audit.md | head -3   # params present
grep -n "verify-then-triage\|Never drop the test-honesty\|shrink-only" ~/.claude/commands/audit.md | wc -l   # expect >= 3
```
Expected: counts as annotated; no zero results.

### Task 2: Seed `docs/audits/REVIEW-PATTERNS.md` (VectorAccelerate)

**Files:**
- Create: `<repo>/docs/audits/REVIEW-PATTERNS.md`

**Interfaces:**
- Consumes: the pattern-file path contract from Task 1 (`docs/audits/REVIEW-PATTERNS.md`, entry format: name / mechanism / tell / incident).
- Produces: the file `/audit` loads in this repo.

- [ ] **Step 1: Write the file with exactly this content**

````markdown
# REVIEW-PATTERNS — VectorAccelerate

How this codebase gets fooled. Loaded by `/audit` as mandatory checklist items; reviews that
discover a new masking pattern append it here (name / mechanism / tell / incident). Seeded
2026-08-18 from the AUDIT-2/AUDIT-3 epics and their slice-4 meta-review.

## 1. Dispatch-geometry shielding
- **Mechanism:** a kernel is correct only under the host's exact dispatch shape (one
  threadgroup, `min(256, dim)` width, pow2 size); nothing enforces the shape.
- **Tell:** `tgSize/2`-style strides, `id % 256`, guards before barriers, shared arrays
  indexed by raw thread id; hosts computing "safe" geometries with comments.
- **Incident:** VA3-002/-013/-014 — six kernels wrong at every geometry the hosts happened
  never to use.

## 2. Dual-compile divergence
- **Mechanism:** debug builds load the per-file-compiled metallib; release compiles all
  shaders as ONE concatenated source behind `KernelContext`'s hand-mirrored preamble.
  Symbol visibility, macro shadowing, and header edits behave differently per path.
- **Tell:** new shader helpers/macros used cross-file; edits to `Metal4Common.h` alone
  (plugin does not track header deps — touch `*.metal` after, VA2-013); any `#define`
  matching a preamble constant (`EPSILON`).
- **Incident:** VA2-001 (release suite red 37/1540 at pristine HEAD), VA3-012 (EPSILON
  1e-7/1e-8 drift, still open).

## 3. Routing-gate masking (and gate-change arming)
- **Mechanism:** GPU legs invisible to tests because thresholds route small test inputs to
  CPU; conversely, a gate RELAXED for a good reason arms every dormant defect it shielded.
- **Tell:** `dimension > 16`-style fallbacks; decision-engine gates; tests whose inputs sit
  below routing thresholds; any diff touching `shouldUseGPU`/threshold logic.
- **Incident:** VA3-002 (engine dotProduct GPU leg untested — all tests dim ≤ 16);
  VA3-032 (VA2-003's k-gate exemption armed a phantom-kernel throw on a public API).

## 4. Phantom names / resolution hijack
- **Mechanism:** runtime-resolved identifiers that match no kernel — or are REWRITTEN by a
  derivation to the wrong real kernel (`"dotProduct"` → `dot_product_kernel`).
- **Tell:** `getPipeline(functionName:)` / `makeFunction(name:)` literals; operation-string
  derivations; any new rewriting case in `PipelineCacheKey.functionName`.
- **Incident:** five instances (batchCosineDistance, tiledTransposeInPlace, vectorMultiply,
  batchDotProduct, batchManhattanDistance) + the VA3-031 hijack. Mechanically closed by
  `ShaderLibraryCompletenessTests` name round-trip + Swift-literal closure (shrink-only
  allowlists) — new instances should be impossible; if one appears, the closure test has a
  hole.

## 5. Pool stale-bytes readback
- **Mechanism:** a kernel that exits without writing its output leaves pooled-buffer bytes
  to be read back as the answer — wrong values with no error.
- **Tell:** any early `return` before output writes; capability caps (`k > MAX`) that no-op;
  result buffers acquired from pools without poisoning in tests.
- **Incident:** VA3-031 (stale dot products), VA3-008/-034 (warp-select tails and over-cap k).

## 6. Pow2-only test data
- **Mechanism:** reduction/tree defects invisible because every tested size is a power of
  two (or a multiple of the block size); comments may claim non-pow2 robustness the guard
  does not provide.
- **Tell:** test dims all in {16, 32, …, 256, 512}; "robustness check" comments on
  `size/2`-halving loops; hosts forcing pow2 widths "for correctness".
- **Incident:** VA3-014 — orphaning trees under a literal "Robustness check for
  non-power-of-2 tgSize" comment; hosts knew and compensated.

## 7. Silent CPU fallback
- **Mechanism:** provider rescues convert GPU breakage into correct CPU results — tests
  green while kernels rot (or never existed).
- **Tell:** `fallbackToCPU` defaults, `catch` → CPU result, `try?` on dispatch paths;
  differential tests that only ever run through the provider.
- **Incident:** AUDIT-2 foundational finding — `batchCosineDistance` was requested at three
  engine sites and NEVER EXISTED; every cosine batch dispatch silently ran CPU. Counter:
  RoutingProvenanceTests telemetry assertions + kernel-direct legs.

## 8. Fast-math idiom fragility
- **Mechanism:** `-ffast-math` reassociates `sqrt(a)*sqrt(b)` into `sqrt(a·b)` (overflow),
  rewrites `(x/a)/b` into subnormal-reciprocal multiplies (flush to zero), folds
  `isinf`/`isfinite` per compile path.
- **Tell:** those exact shapes; `== INFINITY` compares; new normalization/denominator math
  not using the established idioms (two-stage `precise::divide`, `> FLT_MAX` compares,
  pre-scaled rescues in `Metal4Common.h`).
- **Incident:** VA2-008 (measured corruptions), VA3-015 (stragglers, open).

## 9. Cumulative shared-state assertions
- **Mechanism:** process-shared pools/counters make `>= N` assertions vacuously true in
  full-suite runs (and order-dependently false in isolation — flake or theater, depending
  on direction).
- **Tell:** assertions on statistics without a pre-call snapshot; anything asserting on a
  shared singleton's totals.
- **Incident:** Priority2 buffer-pool test — flaked on `allocationCount`, then the fix was
  vacuous on cumulative hits+misses; final form asserts the delta.

## 10. Value-policy divergence across legs of one API
- **Mechanism:** one public operation, multiple implementations (GPU / SIMD / scalar CPU,
  or size-tiered legs) with different NaN/overflow/clamp/tie policies — answers flip at
  routing boundaries.
- **Tell:** size-threshold routing (`>= 100`); inline math in one leg where siblings call a
  shared core; NaN-swallowing guards (`norm > 0`).
- **Incident:** VA3-033 (cosine answers flipped at the batch-size-100 boundary), VA3-016
  (NaN/tie policy inconsistencies, open).
````

- [ ] **Step 2: Verify structure**

Run:
```bash
grep -c "^## [0-9]" docs/audits/REVIEW-PATTERNS.md    # expect 10
grep -c "\*\*Incident:\*\*" docs/audits/REVIEW-PATTERNS.md   # expect 10
git status --short docs/audits/REVIEW-PATTERNS.md      # untracked, NOT staged
```

### Task 3: Append the discipline block to `~/.claude/CLAUDE.md`

**Files:**
- Modify: `~/.claude/CLAUDE.md` (append at end)

**Interfaces:**
- Consumes: the verbatim block from the spec §6 (cross-referenced by audit.md's "Prevention duties").

- [ ] **Step 1: Idempotence check**

Run: `grep -n "Review & claim discipline" ~/.claude/CLAUDE.md`
Expected: no match (if it matches, STOP — the block exists; skip Step 2).

- [ ] **Step 2: Append exactly this block (preceded by one blank line)**

```markdown
  ## Review & claim discipline

  - Any correctness or robustness claim written in a comment ("works for any N",
  "thread-safe", "handles NaN") must name the test that exercises it, be derivable in one
  step from adjacent code, or be written as an explicit unverified contract — never as fact.
  Unproven claims are defects waiting to be believed.
  - When a defect is an instance of a class (second occurrence anywhere), add the mechanical
  test that closes the class, not just the instance fix.
  - When a routing/config gate changes, re-derive reachability for everything the gate
  shields — a correct gate change can arm a dormant defect elsewhere.
```

(Indentation note: this file's existing sections use two-space-indented body text; match it.)

- [ ] **Step 3: Verify**

Run: `grep -A2 "Review & claim discipline" ~/.claude/CLAUDE.md | head -5`
Expected: the heading plus first bullet, appearing exactly once (`grep -c` = 1).

### Task 4: Live smoke test of the light tier

**Files:** none (behavioral verification).

- [ ] **Step 1: Invoke `/audit` in light mode on a small real scope** — e.g.
  `/audit Sources/VectorAccelerate/Operations/BatchDistanceOperations.swift` (a file with
  known recent changes and a known pattern history).

- [ ] **Step 2: Check the output contract** — the report must contain: Verdict, Findings by
  severity with liveness + concrete scenarios, the Claim-coverage table, Class-closure
  proposals, Pattern-file updates, and the remediation Offer — and must apply NO fixes.

- [ ] **Step 3: Confirm the pattern file was loaded** — the review's checklist work should
  reference at least one REVIEW-PATTERNS entry where relevant (e.g. value-policy divergence
  for this file).

## Self-review record

- Spec coverage: §1 tiers/scope → Task 1 Usage; §2 partitioning → Task 1 deep procedure; §3
  six questions → Task 1; §4 templates/rules → Task 1 templates + preamble; §5 pattern seed
  → Task 2 (10 entries match); §6 CLAUDE.md verbatim → Task 3; ledger convention → Task 1
  deep procedure step 4; non-goals (TDD relationship, CI-note-only, coexistence with
  /code-review) → Task 1 rules/prevention duties. No gaps.
- Placeholders: none (full contents embedded).
- Consistency: pattern-file path, entry format, and CLAUDE.md section title match across
  Tasks 1/2/3.
