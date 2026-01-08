# Debuggable Metal Shaders Setup Guide

This guide explains how to enable Metal shader debugging in Xcode using MetalCompilerPlugin.

## Why This Matters

By default, SPM compiles `.metal` files into `default.metallib` without debug symbols. This means:
- **No shader source viewing** in Xcode Metal Debugger
- **No breakpoints** in Metal shaders
- **No variable inspection** during GPU debugging

MetalCompilerPlugin adds `-gline-tables-only` and `-frecord-sources` flags, creating a `debug.metallib` that works with Xcode's GPU debugging tools.

## Implementation Steps

### Step 1: Add MetalCompilerPlugin Dependency

Update `Package.swift`:

```swift
let package = Package(
    name: "VectorAccelerate",
    // ... existing config ...
    dependencies: [
        .package(url: "https://github.com/gifton/VectorCore", from: "0.1.6"),
        // Add MetalCompilerPlugin for debug builds
        .package(url: "https://github.com/schwa/MetalCompilerPlugin", branch: "main"),
    ],
    targets: [
        .target(
            name: "VectorAccelerate",
            dependencies: [
                .product(name: "VectorCore", package: "VectorCore"),
            ],
            resources: [
                .process("Metal/Shaders")
            ],
            plugins: [
                // Only apply in debug builds for shader debugging
                .plugin(name: "MetalCompilerPlugin", package: "MetalCompilerPlugin")
            ]
        ),
        // ... rest of targets ...
    ]
)
```

### Step 2: Create Plugin Configuration

Create `Sources/VectorAccelerate/metal-compiler-plugin.json`:

```json
{
    "output": "debug.metallib",
    "flags": [
        "-gline-tables-only",
        "-frecord-sources",
        "-std=metal4.0",
        "-mmacosx-version-min=26.0"
    ],
    "include-paths": [
        "Metal/Shaders"
    ],
    "plugin-logging": true
}
```

**Flag Explanation:**
- `-gline-tables-only`: Minimal debug info for line-level debugging
- `-frecord-sources`: Embeds shader source for Xcode viewing
- `-std=metal4.0`: Metal Shading Language 4.0 (required for Metal 4 features)
- `-mmacosx-version-min=26.0`: Target macOS 26 (Tahoe) for Metal 4 support

### Step 3: Update KernelContext for Debug Library Loading

Modify `Sources/VectorAccelerate/Core/KernelContext.swift` to prefer debug.metallib:

```swift
public static func loadMetalLibrary(device: any MTLDevice) -> (any MTLLibrary)? {
    // 1. Try device's default library (works in app bundles)
    if let library = device.makeDefaultLibrary() {
        return library
    }

    // 2. Find VectorAccelerate's resource bundle
    guard let resourceBundle = findVectorAccelerateBundle() else {
        #if DEBUG
        print("[VectorAccelerate] Warning: Could not locate resource bundle")
        #endif
        return nil
    }

    // 3. In DEBUG builds, prefer debug.metallib for Xcode Metal Debugger support
    #if DEBUG
    if let libraryURL = resourceBundle.url(forResource: "debug", withExtension: "metallib"),
       let library = try? device.makeLibrary(URL: libraryURL) {
        print("[VectorAccelerate] Loaded debug.metallib with shader debugging support")
        return library
    }
    #endif

    // 4. Try default.metallib (SPM auto-compiled, no debug symbols)
    if let libraryURL = resourceBundle.url(forResource: "default", withExtension: "metallib"),
       let library = try? device.makeLibrary(URL: libraryURL) {
        return library
    }

    // 5. Fallback: Runtime compile from sources
    if let library = compileMetalSourcesFromBundle(device: device, bundle: resourceBundle) {
        return library
    }

    return nil
}
```

### Step 4: Conditional Plugin Application (Optional)

If you want the plugin to only run in debug configurations, you can use environment-based configuration. However, SPM plugins don't have native debug/release awareness, so the common approach is:

**Option A: Always include debug.metallib, only load in DEBUG**
- Plugin always runs, creates debug.metallib
- Loading code only uses it when `#if DEBUG`

**Option B: Separate debug target**
- Create `VectorAccelerateDebug` target with plugin
- Use for development, switch to main target for release

### Step 5: Xcode Metal Debugger Usage

After setup, you can debug Metal shaders in Xcode:

1. **Capture GPU Frame**: Product → Debug → Capture GPU Frame (or ⌘⇧G)
2. **Navigate to Shader**: In the GPU trace, select a draw/compute call
3. **View Shader Source**: Click on the shader stage to see source code
4. **Set Breakpoints**: Click line numbers in shader source
5. **Inspect Variables**: Step through shader execution, inspect thread variables

## File Structure After Setup

```
VectorAccelerate/
├── Package.swift                              # Updated with plugin dependency
├── Sources/
│   └── VectorAccelerate/
│       ├── metal-compiler-plugin.json         # Plugin configuration
│       ├── Core/
│       │   └── KernelContext.swift            # Updated loading logic
│       └── Metal/
│           └── Shaders/
│               ├── L2Distance.metal
│               ├── L2Normalization.metal
│               └── ... (other shaders)
└── docs/
    └── METAL_DEBUG_SETUP.md                   # This file
```

## Build Output

After building with the plugin:
- `debug.metallib` - Created by MetalCompilerPlugin (with debug symbols)
- `default.metallib` - Created by SPM (no debug symbols, optimized)

## Troubleshooting

### Plugin Not Running
- Ensure `metal-compiler-plugin.json` is in the target source directory
- Check build logs for plugin output (`"plugin-logging": true`)

### Shaders Not Visible in Debugger
- Verify `debug.metallib` is being loaded (check debug print statements)
- Ensure you're using `device.makeLibrary(URL:)`, not `makeDefaultLibrary()`

### Include Path Issues
- Add paths to `include-paths` in JSON config
- For Metal4Common.h, ensure it's in an included directory

### Compilation Errors
- Check Metal language version matches your shaders
- Verify all dependencies are available

## Performance Considerations

- **Debug builds**: Slightly larger metallib, minimal runtime overhead
- **Release builds**: Use `default.metallib` (no debug symbols, smaller size)
- The loading code automatically selects appropriate library based on build config

## References

- [MetalCompilerPlugin GitHub](https://github.com/schwa/MetalCompilerPlugin)
- [Apple Metal Debugger Documentation](https://developer.apple.com/documentation/xcode/metal-debugger)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
