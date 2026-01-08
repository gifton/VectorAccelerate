#!/bin/bash
# =============================================================================
# VectorAccelerate Metal Shader Compilation Script
# =============================================================================
# Compiles all Metal shader source files into a single default.metallib
# for distribution with SPM packages.
#
# Usage: ./Scripts/compile_metallib.sh
#
# Requirements:
#   - Xcode 26+ with Metal 4.0 support
#   - macOS 26.0+ SDK
# =============================================================================

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths
SHADERS_DIR="$PROJECT_ROOT/Sources/VectorAccelerate/Metal/Shaders"
OUTPUT_FILE="$SHADERS_DIR/default.metallib"
TEMP_DIR=$(mktemp -d)

echo "=== VectorAccelerate Metal Shader Compilation ==="
echo "Shaders directory: $SHADERS_DIR"
echo "Output: $OUTPUT_FILE"
echo ""

# Verify shaders directory exists
if [ ! -d "$SHADERS_DIR" ]; then
    echo "Error: Shaders directory not found: $SHADERS_DIR"
    exit 1
fi

# Find all .metal files
METAL_FILES=$(find "$SHADERS_DIR" -name "*.metal" -type f | sort)
FILE_COUNT=$(echo "$METAL_FILES" | wc -l | tr -d ' ')

echo "Found $FILE_COUNT Metal shader files"
echo ""

# Compile each .metal file to .air (intermediate representation)
echo "Step 1: Compiling shaders to AIR..."
AIR_FILES=""
for METAL_FILE in $METAL_FILES; do
    FILENAME=$(basename "$METAL_FILE" .metal)
    AIR_FILE="$TEMP_DIR/$FILENAME.air"

    echo "  Compiling: $FILENAME.metal"

    xcrun -sdk macosx metal \
        -std=metal3.2 \
        -target air64-apple-macos26.0 \
        -I "$SHADERS_DIR" \
        -c "$METAL_FILE" \
        -o "$AIR_FILE"

    AIR_FILES="$AIR_FILES $AIR_FILE"
done

echo ""
echo "Step 2: Linking AIR files to metallib..."

# Link all AIR files into a single metallib
xcrun -sdk macosx metallib \
    -o "$OUTPUT_FILE" \
    $AIR_FILES

# Cleanup
rm -rf "$TEMP_DIR"

# Verify output
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    echo ""
    echo "=== Success ==="
    echo "Created: $OUTPUT_FILE"
    echo "Size: $SIZE"
    echo ""
    echo "The metallib is now ready for SPM distribution."
else
    echo "Error: Failed to create metallib"
    exit 1
fi
