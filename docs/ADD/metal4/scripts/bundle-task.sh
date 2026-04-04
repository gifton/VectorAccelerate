#!/bin/bash
#
# bundle-task.sh - Bundle all context needed for a specific Metal 4 migration task
#
# Usage: ./bundle-task.sh <task-file> [--include-source]
#
# Example: ./bundle-task.sh 01-foundation/task-metal4context.md --include-source
#
# Output: Prints a complete context bundle to stdout that can be shared with
#         an external agent. Redirect to file if needed:
#         ./bundle-task.sh 01-foundation/task-metal4context.md > bundle.md

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$DOCS_DIR")")"
CONTEXT_DIR="$DOCS_DIR/00-context"

TASK_FILE="$1"
INCLUDE_SOURCE="${2:-}"

if [ -z "$TASK_FILE" ]; then
    echo "Usage: $0 <task-file> [--include-source]" >&2
    echo "" >&2
    echo "Available tasks:" >&2
    find "$DOCS_DIR" -name "task-*.md" -type f | sed "s|$DOCS_DIR/||" | sort >&2
    exit 1
fi

TASK_PATH="$DOCS_DIR/$TASK_FILE"

if [ ! -f "$TASK_PATH" ]; then
    echo "Error: Task file not found: $TASK_PATH" >&2
    exit 1
fi

# Extract task name from file
TASK_NAME=$(basename "$TASK_FILE" .md | sed 's/task-//')

cat << 'HEADER'
# Metal 4 Migration Context Bundle

This document contains all context needed to complete a Metal 4 migration task
for VectorAccelerate. It is self-contained and can be used by an external agent.

---

HEADER

echo "# Task: $TASK_NAME"
echo ""
echo "Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""
echo "---"
echo ""

# Include architecture overview
echo "# Part 1: Architecture Overview"
echo ""
cat "$CONTEXT_DIR/architecture.md"
echo ""
echo "---"
echo ""

# Include Metal 4 API summary
echo "# Part 2: Metal 4 API Reference"
echo ""
cat "$CONTEXT_DIR/metal4-api-summary.md"
echo ""
echo "---"
echo ""

# Include current patterns
echo "# Part 3: Current Patterns (to migrate)"
echo ""
cat "$CONTEXT_DIR/patterns.md"
echo ""
echo "---"
echo ""

# Include the task specification
echo "# Part 4: Task Specification"
echo ""
cat "$TASK_PATH"
echo ""

# If --include-source, extract and include source files mentioned in task
if [ "$INCLUDE_SOURCE" = "--include-source" ]; then
    echo "---"
    echo ""
    echo "# Part 5: Source Files"
    echo ""

    # Extract file paths from "Files to Modify" section
    FILES=$(grep -A 50 "## Files to Modify" "$TASK_PATH" | grep -E "^\s*-\s*\`" | sed 's/.*`\([^`]*\)`.*/\1/' | head -20)

    for FILE in $FILES; do
        FULL_PATH="$PROJECT_ROOT/$FILE"
        if [ -f "$FULL_PATH" ]; then
            echo "## File: $FILE"
            echo ""
            echo '```swift'
            cat "$FULL_PATH"
            echo '```'
            echo ""
        fi
    done
fi

echo "---"
echo ""
echo "# Instructions for Agent"
echo ""
cat << 'INSTRUCTIONS'
Please implement the task described above following these guidelines:

1. **Follow Metal 4 patterns** as described in the API Reference
2. **Maintain backward compatibility** where specified in the task
3. **Use Swift 6 strict concurrency** - all new code must be Sendable-safe
4. **Preserve existing functionality** - this is a migration, not a rewrite
5. **Add documentation** for any new public APIs
6. **Include error handling** for all Metal operations

## Output Format

Provide your implementation as:
1. Complete file contents for each modified file
2. Brief explanation of key changes
3. Any assumptions or decisions made
4. Suggested test cases

## Quality Checklist

Before submitting, verify:
- [ ] Compiles without errors
- [ ] No new warnings introduced
- [ ] Follows existing code style
- [ ] All Metal 4 APIs used correctly
- [ ] Residency management is complete
- [ ] Synchronization is correct
INSTRUCTIONS
