# VectorAccelerate Metal 4 Migration

This directory contains all documentation, task specifications, and context files for migrating VectorAccelerate from Metal 3 to Metal 4 (iOS 26+, macOS 26+).

## Directory Structure

```
docs/metal4/
├── README.md                    # This file - index and navigation
├── MASTER_PLAN.md              # Complete upgrade plan and timeline
│
├── 00-context/                  # Reference materials (READ-ONLY context)
│   ├── architecture.md         # Current VectorAccelerate architecture
│   ├── metal4-api-summary.md   # Metal 4 API quick reference
│   ├── file-inventory.md       # All files and their purposes
│   └── patterns.md             # Current code patterns to migrate
│
├── 01-foundation/               # Phase 1: Core infrastructure
│   ├── PHASE_OVERVIEW.md       # Phase summary and dependencies
│   ├── task-metal4context.md   # Metal4Context actor implementation
│   ├── task-residency.md       # ResidencyManager implementation
│   └── task-device-update.md   # MetalDevice updates
│
├── 02-command-encoding/         # Phase 2: Command encoding
│   ├── PHASE_OVERVIEW.md
│   ├── task-unified-encoder.md # Unified MTL4ComputeCommandEncoder
│   ├── task-argument-tables.md # ArgumentTable implementation
│   └── task-synchronization.md # Event-based synchronization
│
├── 03-shader-compilation/       # Phase 3: Compilation infrastructure
│   ├── PHASE_OVERVIEW.md
│   ├── task-compiler.md        # MTL4Compiler wrapper
│   ├── task-harvesting.md      # Pipeline harvesting
│   └── task-msl4-shaders.md    # MSL 4.0 shader updates
│
├── 04-ml-integration/           # Phase 4: ML/Tensor features
│   ├── PHASE_OVERVIEW.md
│   ├── task-tensor-infra.md    # MTLTensor infrastructure
│   ├── task-learned-distance.md # LearnedDistanceKernel
│   └── task-neural-quant.md    # NeuralQuantizationKernel
│
├── 05-kernel-migrations/        # Individual kernel migration specs
│   ├── KERNEL_INDEX.md         # Index of all kernels
│   ├── l2-distance.md
│   ├── cosine-similarity.md
│   ├── dot-product.md
│   └── ...
│
└── scripts/
    ├── bundle-task.sh          # Bundle context for a specific task
    ├── bundle-kernel.sh        # Bundle context for kernel migration
    └── validate-migration.sh   # Validate Metal 4 compatibility
```

## Quick Start for External Agent

### Option 1: Single Task Focus

To work on a specific task, share these files:
1. `00-context/architecture.md` - Understand the codebase
2. `00-context/metal4-api-summary.md` - Metal 4 reference
3. The specific task file (e.g., `01-foundation/task-metal4context.md`)

### Option 2: Use Bundle Script

```bash
# Generate a complete context bundle for a task
./scripts/bundle-task.sh 01-foundation/task-metal4context.md > context-bundle.md

# Generate context for kernel migration
./scripts/bundle-kernel.sh L2DistanceKernel > l2-kernel-bundle.md
```

### Option 3: Full Kernel Migration

For kernel work, share:
1. `00-context/patterns.md` - Current implementation patterns
2. `00-context/metal4-api-summary.md` - Metal 4 reference
3. The relevant kernel spec from `05-kernel-migrations/`
4. The actual source files (Swift + Metal shader)

## Task File Format

Each task file follows this structure:

```markdown
# Task: [Name]

## Objective
[Clear statement of what needs to be done]

## Current Implementation
[Code snippets from current files]

## Target Implementation (Metal 4)
[Expected code patterns for Metal 4]

## Files to Modify
- `path/to/file.swift` - [what changes]
- `path/to/shader.metal` - [what changes]

## Dependencies
- [Other tasks that must complete first]

## Acceptance Criteria
- [ ] [Specific testable criteria]
- [ ] [Build succeeds]
- [ ] [Tests pass]

## Context Files to Include
[List of files the agent needs to see]
```

## Migration Progress

| Phase | Status | Tasks | Complete |
|-------|--------|-------|----------|
| 1. Foundation | Not Started | 3 | 0/3 |
| 2. Command Encoding | Not Started | 3 | 0/3 |
| 3. Shader Compilation | Not Started | 3 | 0/3 |
| 4. ML Integration | Not Started | 3 | 0/3 |
| 5. Kernel Migrations | Not Started | 25+ | 0/25 |

## Key Files Reference

### Core Infrastructure (Phase 1-2)
| File | Lines | Purpose |
|------|-------|---------|
| `Core/MetalContext.swift` | 530 | Main context actor |
| `Core/MetalDevice.swift` | 445 | Device management |
| `Core/ComputeEngine.swift` | 640 | Compute execution |
| `Core/BufferPool.swift` | 515 | Buffer management |
| `Core/ShaderManager.swift` | 964 | Shader compilation |

### Kernels (Phase 5)
| Kernel | Swift File | Metal Shader |
|--------|------------|--------------|
| L2Distance | `Kernels/L2DistanceKernel.swift` | `Shaders/L2Distance.metal` |
| CosineSimilarity | `Kernels/CosineSimilarityKernel.swift` | `Shaders/CosineSimilarity.metal` |
| DotProduct | `Kernels/DotProductKernel.swift` | `Shaders/DotProduct.metal` |
| TopKSelection | `Kernels/TopKSelectionKernel.swift` | `Shaders/AdvancedTopK.metal` |
| ... | ... | ... |

## How to Use with Deep Research Agent

### Prompt Template

```
I'm migrating VectorAccelerate from Metal 3 to Metal 4.

## Context
[Paste contents of 00-context/architecture.md]

## Metal 4 API Reference
[Paste contents of 00-context/metal4-api-summary.md]

## Task Specification
[Paste contents of specific task file]

## Current Implementation
[Paste relevant source files]

## Request
Please implement [task name] following the Metal 4 patterns described.
Provide complete, production-ready code.
```

### For Kernel Migrations

```
I need to migrate [KernelName] to Metal 4.

## Current Patterns
[Paste 00-context/patterns.md]

## Kernel Specification
[Paste kernel migration spec]

## Current Swift Implementation
[Paste .swift file]

## Current Metal Shader
[Paste .metal file]

## Request
Migrate this kernel to Metal 4 with:
1. ArgumentTable binding
2. Residency set integration
3. Barrier-based synchronization
4. MSL 4.0 shader updates
```
