#!/bin/sh

# Install git hooks for this repository

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "${BLUE}Installing git hooks...${NC}\n"

# Get the directory where this script is located
HOOKS_DIR="$(cd "$(dirname "$0")" && pwd)"

# Make hooks executable
chmod +x "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-push"
chmod +x "$HOOKS_DIR/post-checkout"
chmod +x "$HOOKS_DIR/post-commit"
chmod +x "$HOOKS_DIR/post-merge"

echo "${GREEN}✓ Made hooks executable${NC}"

# Configure git to use .githooks directory
git config core.hooksPath .githooks

echo "${GREEN}✓ Configured git to use .githooks directory${NC}"

# Initialize Git LFS if available
if command -v git-lfs >/dev/null 2>&1; then
    git lfs install --skip-repo >/dev/null 2>&1 || true
    echo "${GREEN}✓ Initialized Git LFS filters${NC}\n"
else
    echo "${YELLOW}! git-lfs command not found. Please install git-lfs (e.g. 'brew install git-lfs').${NC}\n"
fi

echo "${GREEN}Git hooks installed successfully!${NC}\n"

echo "${BLUE}Installed hooks:${NC}"
echo "  - pre-commit: Format, lint checks, and auto-track JSON/MD >= 500KB with Git LFS"
echo "  - pre-push: Build, test checks, and Git LFS push sync"
echo "  - post-checkout: Git LFS checkout sync"
echo "  - post-commit: Git LFS commit sync"
echo "  - post-merge: Git LFS merge sync"
echo ""
echo "${YELLOW}Tips:${NC}"
echo "  - Skip hooks temporarily: git commit --no-verify"
echo "  - Or for push: git push --no-verify"
echo "  - Run pre-commit manually: .githooks/pre-commit"
echo "  - Run pre-push manually: .githooks/pre-push"
echo ""
