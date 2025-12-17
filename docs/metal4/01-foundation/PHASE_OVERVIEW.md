# Phase 1: Foundation

## Objective

Establish the core Metal 4 infrastructure that all other components depend on.

## Tasks

| Task | Description | Status | Dependencies |
|------|-------------|--------|--------------|
| [task-device-update.md](task-device-update.md) | Metal4Capabilities, GPUBackend, feature gating | **Complete** | None |
| [task-metal4context.md](task-metal4context.md) | Metal4Context actor with MTL4CommandQueue | **Complete** | task-device-update |
| [task-residency.md](task-residency.md) | ResidencyManager for explicit residency | **Complete** | task-metal4context |

## Execution Order

```
1. task-device-update.md   ─────┐
                                ├──► 2. task-metal4context.md ──► 3. task-residency.md
   (can start immediately)      │
                                │
   Metal4Capabilities           │      Metal4Context               ResidencyManager
   GPUBackend                   │      MTL4CommandQueue            MTLResidencySet
   BackendPreference            │      Event synchronization       BufferPool integration
```

## Completion Criteria

- [x] Metal4Context can create MTL4CommandQueue
- [x] Metal4Context can create MTL4CommandBuffer from device
- [x] ResidencyManager tracks all allocated buffers
- [x] ResidencySet properly attached to queue/buffers
- [x] MetalDevice supports Metal 4 capability detection
- [x] All existing tests pass

## Architecture After Phase 1

```
┌─────────────────────────────────────────────────────────────┐
│                     Metal4Context (NEW)                     │
│  ├─ MTL4CommandQueue                                        │
│  ├─ ResidencyManager (NEW)                                  │
│  │   └─ MTLResidencySet                                     │
│  ├─ Metal4Compiler (NEW)                                    │
│  └─ BufferPool (updated)                                    │
├─────────────────────────────────────────────────────────────┤
│                   MetalDevice (UPDATED)                     │
│  ├─ MSL 4.0 compilation                                     │
│  ├─ MTL4Compiler creation                                   │
│  └─ GPU address support                                     │
└─────────────────────────────────────────────────────────────┘
```

## Files Created/Modified

### New Files
- `Core/Metal4Capabilities.swift` - Feature detection struct with GPU family checks
- `Core/Metal4Error.swift` - Metal 4 specific error factory methods
- `Core/ResidencyManager.swift` - Explicit residency management actor
- `Core/Metal4Context.swift` - Metal 4 compute context actor

### Modified Files
- `Core/MetalDevice.swift` - Added Metal 4 capabilities property and factory methods

## Risk Mitigation

- Keep `MetalContext` working for Metal 3 fallback
- Feature-flag Metal 4 code paths initially
- Comprehensive testing at each step
