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

echo "${GREEN}✓ Made hooks executable${NC}"

# Configure git to use .githooks directory
git config core.hooksPath .githooks

echo "${GREEN}✓ Configured git to use .githooks directory${NC}\n"

echo "${GREEN}Git hooks installed successfully!${NC}\n"

echo "${BLUE}Installed hooks:${NC}"
echo "  - pre-commit: Format and lint checks"
echo "  - pre-push: Build and test checks"
echo ""
echo "${YELLOW}Tips:${NC}"
echo "  - Skip hooks temporarily: git commit --no-verify"
echo "  - Or for push: git push --no-verify"
echo "  - Run pre-commit manually: .githooks/pre-commit"
echo "  - Run pre-push manually: .githooks/pre-push"
echo ""
