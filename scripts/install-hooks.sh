#!/usr/bin/env bash
# install-hooks.sh — configure bare-metal repo to use version-controlled hooks
#
# Usage: bash scripts/install-hooks.sh

set -euo pipefail

HOOKS_DIR=".githooks"

echo "Installing custom hooks..."

if [ ! -d "$HOOKS_DIR" ]; then
    echo "ERROR: $HOOKS_DIR directory not found."
    echo "  This script must be run from the repository root."
    exit 1
fi

# Make hook scripts executable
chmod +x "$HOOKS_DIR"/* 2>/dev/null || true

# Tell git to look for hooks in version-controlled directory
git config core.hooksPath "$HOOKS_DIR"

echo "Hooks configured. Git now reads hooks from: $HOOKS_DIR"
echo "Currently installed hooks:"
for hook in "$HOOKS_DIR"/*; do
    name=$(basename "$hook")
    echo "  - $name"
done

echo ""
echo "To remove custom hooks and restore default git hooks:"
echo "  git config --unset core.hooksPath"
echo ""
echo "To bypass a hook for a single push:"
echo "  git push --no-verify"
