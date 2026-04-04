# Agent Prompt: Tier 1 Foundation Migration

Use this prompt to start a new Claude Code session for migrating GPUDecisionEngine and Write-Ahead Log from VectorIndexAccelerated to VectorAccelerate.

---

## Prompt

```
I need you to migrate two components from a deprecated package (VectorIndexAccelerated) into VectorAccelerate. This is a Metal 4 GPU-accelerated vector search library for Apple platforms.

## Project Context

- **Current repo**: /Users/goftin/dev/gsuite/VSK/future/VectorAccelerate
- **Source repo** (deprecated): /Users/goftin/dev/gsuite/VSK/future/VectorIndexAccelerated
- **Migration plan**: docs/migrations/VIA_MIGRATION_PLAN.md

VectorAccelerate is a Metal 4-only library (macOS 26+, iOS 26+) that provides GPU-accelerated vector operations. It uses:
- `Metal4Context` actor for GPU device management
- `BufferPool` for buffer allocation
- Actor-based concurrency throughout
- Swift 6.2 with strict concurrency

## Task 1: GPUDecisionEngine Migration

**Source**: `../VectorIndexAccelerated/Sources/VectorIndexAccelerated/GPU/GPUDecisionEngine.swift` (290 lines)

**Purpose**: Adaptive GPU/CPU routing based on workload characteristics and runtime performance history.

**Migration steps**:
1. Read the source file to understand the implementation
2. Create `Sources/VectorAccelerate/Core/GPUDecisionEngine.swift`
3. Update to work with VA's `Metal4Context` instead of VIA's `MetalDevice`
4. Key adaptations needed:
   - Use `Metal4Context` for device capability queries
   - Update `GPUOperation` enum to match VA's operation types (distance computation, top-k selection, quantization, etc.)
   - Ensure all types are `Sendable` for Swift 6 concurrency
5. Add integration hooks (but don't modify AcceleratedVectorIndex yet - just create the standalone component)

**Target location**: `Sources/VectorAccelerate/Core/GPUDecisionEngine.swift`

## Task 2: Write-Ahead Log Migration

**Source**: `../VectorIndexAccelerated/Sources/VectorIndexAccelerated/Persistence/WriteAheadLog.swift` (507 lines)

**Purpose**: Crash recovery and durability for index operations.

**Migration steps**:
1. Read the source file to understand the implementation
2. Create directory: `Sources/VectorAccelerate/Persistence/`
3. Create `Sources/VectorAccelerate/Persistence/WriteAheadLog.swift`
4. This component is mostly standalone (uses Foundation, CryptoKit, os.Logger)
5. Key adaptations:
   - Update logger subsystem to "VectorAccelerate"
   - Ensure all types are `Sendable`
   - The WALSegment class needs to be made thread-safe or converted to an actor
6. Create supporting types file if needed: `WALTypes.swift`

**Target location**: `Sources/VectorAccelerate/Persistence/WriteAheadLog.swift`

## Requirements

1. **DO NOT** modify existing VA files yet - just create the new standalone components
2. **DO** ensure Swift 6 strict concurrency compliance (no warnings)
3. **DO** maintain the same public API as the source where possible
4. **DO** add appropriate documentation comments
5. **DO** create any necessary supporting types

## Verification

After migration, verify:
1. `swift build` succeeds with no errors
2. No concurrency warnings with strict concurrency enabled
3. The new files are properly integrated into the module (can be imported)

## Notes

- VA uses `os.Logger` for logging - follow the same pattern
- VA prefers actors over classes with locks
- The WALSegment class in the source uses FileHandle which isn't Sendable - you may need to adapt this

Start by reading both source files, then proceed with the migrations one at a time. Ask me if you have questions about VA's architecture.
```

---

## Quick Start

1. Copy the prompt above
2. Start a new Claude Code session in the VectorAccelerate directory
3. Paste the prompt
4. The agent will read the source files and perform the migration

## Expected Output

After completion, you should have:

```
Sources/VectorAccelerate/
├── Core/
│   └── GPUDecisionEngine.swift    # NEW
│
└── Persistence/
    ├── WriteAheadLog.swift        # NEW
    └── WALTypes.swift             # NEW (if needed)
```

## Follow-up Tasks (after this migration)

1. Integration into `AcceleratedVectorIndex` (separate task)
2. Unit tests for both components
3. HNSW migration (Tier 1, high effort - separate focused effort)
