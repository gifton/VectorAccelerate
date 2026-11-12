# VectorAccelerate on VectorCore 0.4.0-alpha.1 — Integration Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a VectorAccelerate branch that builds and passes its full suite, Debug and Release, against the exact VectorCore `0.4.0-alpha.1` tag, so SwiftTopics can pin it by revision.

**Architecture:** Start from `gifton/shipped-metallib` (main plus four commits). Pin VectorCore exactly, carry over the migration already proven against the October candidate on `gifton/vectorcore-0.4.0-prep`, audit the rows the alpha added after that candidate, and record the evidence. Nothing here adds features.

**Tech Stack:** Swift 6.4.2 (Xcode 27.2, build 27B52), SwiftPM, Metal 4, XCTest and Swift Testing.

**Spec:** `/Users/goftin/dev/projects/GournalV2/SwiftTopics/.worktrees/foundation-port/docs/superpowers/specs/2026-11-10-foundation-port-design.md` — Part B. Read "Constraints" and "Anti-goals" too.

## Global Constraints

- VectorCore is pinned `exact: "0.4.0-alpha.1"`. The tag's commit is `44c9ffaf145a59688f420023b28cdcf87c158070`.
- The branch starts from current main including the four-commit `gifton/shipped-metallib` work.
- The branch is an integration revision. It is not merged to VectorAccelerate main. Pushing, opening a pull request, merging and tagging each need the owner's go-ahead at the time.
- Work happens in a separate worktree. No local path dependency and no `swift package edit` state in anything committed. `Package.resolved` is committed.
- Built with Xcode 27.2 / Swift 6.4.2. Release is part of acceptance. Do not suppress optimisation or weaken the compiler canary.
- No `inout` handoff of a borrowed pointer, buffer descriptor, span or pointer-bearing wrapper to a mutating helper, including through callbacks.
- A test that is changed, skipped or re-baselined is listed with its reason. A claim names the test that exercises it.
- Do not modify the VectorCore checkout at `/Users/goftin/dev/projects/VectorCore-archive/VectorCore-fixes-2026-10-27-84798d3`. Read it only.

## Review Focus

1. **A stale VectorCore is resolved instead of the tag** (SwiftPM cache, or a leftover `swift package edit`). Expected: `Package.resolved` names version `0.4.0-alpha.1` at the commit above. Pinned in Task 2, Step 3.
2. **The Release-only shader load path breaks while Debug passes**, as it did before the hardening work. Expected: the Release suite is run in full, not inferred from Debug. Pinned in Task 5.
3. **A behaviour row compiles silently and changes results.** Expected: every alpha-only row is mapped to a call site or to grep evidence that the API is unused. Pinned in Task 4.
4. **Tests pass because they skipped.** Expected: skip counts are compared with the baseline, and any new skip is explained. Pinned in Task 5, Step 3.
5. **A conflict resolution quietly drops reviewed work from `gifton/shipped-metallib`.** Expected: the guard and canary suites from that branch still exist and pass after the carry-over. Pinned in Task 3, Step 4.

---

### Task 1: Worktree, branch and baseline on VectorCore 0.3.3

**Files:**
- Create (untracked, under `.build`): `.build/alpha1-evidence/baseline-debug.log`, `.build/alpha1-evidence/baseline-release.log`

**Interfaces:**
- Produces: branch `gifton/vectorcore-0.4.0-alpha1` in worktree `.worktrees/vectorcore-alpha1`; baseline totals (executed / failed / skipped) for Debug and Release, used by Task 5.

- [ ] **Step 1: Create the worktree** (via superpowers:using-git-worktrees)

```bash
cd /Users/goftin/dev/gsuite/VSK/future/VectorAccelerate
git fetch origin
git worktree add .worktrees/vectorcore-alpha1 -b gifton/vectorcore-0.4.0-alpha1 gifton/shipped-metallib
cd .worktrees/vectorcore-alpha1
```

Expected: `git log -1 --format=%s` prints `docs: record the shipped-library work and the slice 32 review`.

- [ ] **Step 2: Confirm the toolchain and a clean package state**

Run: `swift --version && xcrun -f metal && git status --short && ls Packages 2>/dev/null`
Expected: Swift 6.4.2; a path to `metal` (if missing, run `xcodebuild -downloadComponent MetalToolchain`); empty status; no `Packages` directory.

- [ ] **Step 3: Record the baseline on the existing 0.3.3 pin**

Run:
```bash
mkdir -p .build/alpha1-evidence
swift test 2>&1 | tee .build/alpha1-evidence/baseline-debug.log
swift test -c release 2>&1 | tee .build/alpha1-evidence/baseline-release.log
```
Expected: zero failures in both. Write down executed, failed and skipped for each.

### Task 2: Pin the alpha and capture the first build

**Files:**
- Modify: `Package.swift` (the `VectorCore` dependency line and the comment above it)
- Modify: `Package.resolved`

**Interfaces:**
- Produces: `.build/alpha1-evidence/first-build.log`, the list of every error the alpha causes on this branch.

- [ ] **Step 1: Change the requirement**

Replace `.package(url: "https://github.com/gifton/VectorCore", from: "0.3.3"),` with `.package(url: "https://github.com/gifton/VectorCore", exact: "0.4.0-alpha.1"),` and rewrite the comment above it to say why the pin is exact.

- [ ] **Step 2: Resolve**

Run: `swift package resolve`
Expected: success.

- [ ] **Step 3: Verify what was resolved**

Run: `python3 -c "import json;p=[x for x in json.load(open('Package.resolved'))['pins'] if x['identity']=='vectorcore'][0]['state'];print(p['version'],p['revision'])"`
Expected: `0.4.0-alpha.1 44c9ffaf145a59688f420023b28cdcf87c158070`

- [ ] **Step 4: Build and keep the output**

Run: `swift build --build-tests 2>&1 | tee .build/alpha1-evidence/first-build.log`
Expected: either `Build complete!` or compile errors. Both are results. On `origin/main` the known break is `ComputeDevice.gpu().isAvailable`; this branch may already have replaced it.

- [ ] **Step 5: Commit**

```bash
git add Package.swift Package.resolved
git commit -m "build: pin VectorCore to exact 0.4.0-alpha.1"
```

### Task 3: Carry over the proven 0.4.0 migration

**Files:**
- Modify: whatever `git cherry-pick -n 7eaec19` touches. Twenty-one of its 42 files differ from this branch, among them `Sources/VectorAccelerate/Core/{FallbackProvider,KernelContext,Metal4ComputeEngine,MetalDevice}.swift`, `Integration/{MetalComputeProvider,SoACandidateSet}.swift`, `Metal/Shaders/{DistanceShaders,MinkowskiDistance}.metal`, `Operations/{BatchProcessor,MemoryMapManager}.swift`
- Create: `Tests/VectorAccelerateTests/Hardening/ChebyshevNaNPolicyTests.swift`, `Tests/VectorAccelerateTests/VectorCore040CompatTests.swift`, `docs/migrations/VECTORCORE-0.4.0-CANDIDATE.md` (all three arrive with the cherry-pick)

**Interfaces:**
- Consumes: commit `7eaec19` on `gifton/vectorcore-0.4.0-prep`; its `docs/migrations/VECTORCORE-0.4.0-CANDIDATE.md` is the map of what each change is for.
- Produces: a branch that compiles against the alpha with the nine 0.4.0-behaviour tests present.

- [ ] **Step 1: Apply without committing**

Run: `git cherry-pick -n 7eaec19`
Expected: conflicts. List them with `git diff --name-only --diff-filter=U`.

- [ ] **Step 2: Resolve by this rule, file by file**

- Hunks about the shipped `default.metallib`, the empty `SoACandidateSet`, the compiler canary or the lexical guard: keep this branch's side. They landed here in reviewed form on 2026-11-02.
- Hunks about `ComputeDevice` availability, the Chebyshev NaN policy, normalisation parity, or the 0.4.0 compatibility tests: take the cherry-picked side.
- `Package.swift`: keep `exact: "0.4.0-alpha.1"`.
- `CHANGELOG.md`: keep both sides' entries under `[Unreleased]`.

Write every hunk that fits neither rule into `.build/alpha1-evidence/unclassified-hunks.md` with the choice made and why.

- [ ] **Step 3: Build**

Run: `swift build --build-tests 2>&1 | tee .build/alpha1-evidence/post-migration-build.log`
Expected: `Build complete!`. Fix remaining errors using `DOWNSTREAM_UPGRADES.md` in the VectorCore checkout (sections 1 and 2) and note each fix beside its row ID.

- [ ] **Step 4: Check nothing reviewed was lost**

Run: `swift test --filter LostStoreShapeGuardTests --filter CompilerLostStoreCanaryTests --filter ShaderLibraryCompletenessTests --filter PerfGateCoverageTests`
Expected: all pass, and the output shows tests from all four suites ran. Use repeated `--filter` flags, not one pattern with `|`; alternation has silently widened scope in this suite before. The canary's expected non-strict failure appears only in Release.

- [ ] **Step 5: Run the nine 0.4.0-behaviour tests**

Run: `swift test --filter ChebyshevNaNPolicyTests --filter NormalizationParityTests --filter VectorCore040CompatTests`
Expected: all pass. Against 0.3.3 these failed: `ChebyshevNaNPolicyTests` (5), `NormalizationParityTests` (2), `VectorCore040CompatTests` (2).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: migrate to VectorCore 0.4.0-alpha.1"
```

### Task 4: Pin the resolved version and audit the alpha-only rows

**Files:**
- Modify: `Tests/VectorAccelerateTests/VectorCore040CompatTests.swift`
- Create: `docs/migrations/VECTORCORE-0.4.0-ALPHA1.md`

**Interfaces:**
- Consumes: `VectorCoreVersion.version` and `VectorCoreVersion.prerelease` (VectorCore `Sources/VectorCore/Version.swift`).
- Produces: the audit table in `VECTORCORE-0.4.0-ALPHA1.md`, completed by Task 5.

- [ ] **Step 1: Write the failing test**

```swift
func testLinkedVectorCoreIsTheAlpha() {
    XCTAssertEqual(VectorCoreVersion.version, "0.4.0-alpha.1")
    XCTAssertEqual(VectorCoreVersion.prerelease, "alpha.1")
}
```

- [ ] **Step 2: Run it**

Run: `swift test --filter VectorCore040CompatTests/testLinkedVectorCoreIsTheAlpha`
Expected: PASS. If it fails, the build is linking a different VectorCore; stop and return to Task 2, Step 3.

- [ ] **Step 3: Audit the rows the alpha added after the October candidate**

The rows are CB-43, CB-44, CB-45, BH-62, BH-63, BH-64, BH-65, BH-66, AB-09, AB-10 and NA-14 in `DOWNSTREAM_UPGRADES.md`. For each, write one line in `docs/migrations/VECTORCORE-0.4.0-ALPHA1.md`: the row ID, the `git grep` command run over `Sources` and `Tests`, its hit count, and either "not used" or the call site with the test that covers the new behaviour.

Known starting points: `git grep -nE 'AlignedDynamicArrayStorage|DynamicArrayStorage'` and `git grep -n 'SIMDStorage'` returned no hits in `Sources` on this branch on 2026-11-11, and the six `withUnsafeMutableBufferPointer` sites are calls on arrays, which CB-45 leaves available. Re-run them; do not copy these results.

- [ ] **Step 4: Commit**

```bash
git add Tests/VectorAccelerateTests/VectorCore040CompatTests.swift docs/migrations/VECTORCORE-0.4.0-ALPHA1.md
git commit -m "test: pin the linked VectorCore version; audit alpha-only upgrade rows"
```

### Task 5: Full suites and the record

**Files:**
- Modify: `docs/migrations/VECTORCORE-0.4.0-ALPHA1.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: baseline totals from Task 1.
- Produces: the branch tip hash that SwiftTopics pins.

- [ ] **Step 1: Run Debug**

Run: `swift test 2>&1 | tee .build/alpha1-evidence/alpha1-debug.log`
Expected: zero failures.

- [ ] **Step 2: Run Release**

Run: `swift test -c release 2>&1 | tee .build/alpha1-evidence/alpha1-release.log`
Expected: zero failures. `swift build -c release --build-tests` is known not to link the `@testable` target; `swift test -c release` is the working form.

- [ ] **Step 3: Compare with the baseline**

For Debug and Release, compare executed, failed and skipped with Task 1. Every test whose outcome differs, and every new skip, is listed by name with its cause.

- [ ] **Step 4: Write the record**

Complete `docs/migrations/VECTORCORE-0.4.0-ALPHA1.md` with: the pins and commit hashes; host, Xcode and Swift versions; each command and its totals; the contents of `first-build.log` summarised as a list of errors or "built clean"; the audit table; the outcome differences; any unclassified hunks from Task 3; and what was not checked. Add one `[Unreleased]` line to `CHANGELOG.md`.

- [ ] **Step 5: Commit and report**

```bash
git add docs/migrations/VECTORCORE-0.4.0-ALPHA1.md CHANGELOG.md
git commit -m "docs: record the VectorCore 0.4.0-alpha.1 integration revision"
git rev-parse HEAD
```

Report the hash to the owner and ask for the go-ahead to push the branch. Do not push without it.
