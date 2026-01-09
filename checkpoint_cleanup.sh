#!/bin/bash

# LungGuard: Remove Large Checkpoint File from Git
# This script removes model.23-0.22.keras from Git history

set -e  # Exit on any error

echo "============================================"
echo "LungGuard: Git Checkpoint Cleanup Script"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Verify we're in the right directory
echo -e "${YELLOW}[1/8] Verifying repository...${NC}"
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: Not in a Git repository. Run this from MachineLearning root.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Git repository found${NC}"
echo ""

# Step 2: Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}[2/8] Current branch: ${CURRENT_BRANCH}${NC}"
if [ "$CURRENT_BRANCH" != "classification_model_improvements" ]; then
    echo -e "${YELLOW}Warning: You're not on classification_model_improvements branch${NC}"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi
echo ""

# Step 3: Remove the large file from Git tracking
echo -e "${YELLOW}[3/8] Removing checkpoint file from Git...${NC}"
git rm --cached ClassificationModel/testing/Checkpoints/ClassificationModel/model.23-0.22.keras 2>/dev/null || echo "File already removed from index"
echo -e "${GREEN}✓ File removed from Git tracking${NC}"
echo ""

# Step 4: Update .gitignore
echo -e "${YELLOW}[4/8] Updating .gitignore...${NC}"
cat >> .gitignore << 'EOF'

# Model weights and checkpoints (added by cleanup script)
*.keras
*.h5
*.weights.h5
*.hdf5
*.pt
*.pth
*.ckpt
*.pb
*.onnx

# Checkpoint directories
Checkpoints/
checkpoints/
ClassificationModel/testing/Checkpoints/
saved_model/
models/weights/

# Training artifacts
logs/
tensorboard/
runs/
outputs/
wandb/
EOF
echo -e "${GREEN}✓ .gitignore updated${NC}"
echo ""

# Step 5: Commit the changes
echo -e "${YELLOW}[5/8] Committing changes...${NC}"
git add .gitignore
git commit -m "Remove large checkpoint file from tracking and update .gitignore" || echo "Nothing to commit"
echo -e "${GREEN}✓ Changes committed${NC}"
echo ""

# Step 6: Clean Git history with BFG (if file exists in history)
echo -e "${YELLOW}[6/8] Cleaning Git history...${NC}"
echo "This will create a temporary mirror repository..."

# Create temp directory
TEMP_DIR="../MachineLearning-cleanup-temp"
rm -rf "$TEMP_DIR"

# Clone as mirror
git clone --mirror "$(git remote get-url origin)" "$TEMP_DIR"

# Run BFG
cd "$TEMP_DIR"
echo "Running BFG Repo Cleaner..."
bfg --delete-files "model.23-0.22.keras" --no-blob-protection

# Clean up
echo "Cleaning up repository..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Return to original directory
cd - > /dev/null
echo -e "${GREEN}✓ Git history cleaned${NC}"
echo ""

# Step 7: Force push
echo -e "${YELLOW}[7/8] Pushing changes to remote...${NC}"
echo -e "${RED}WARNING: This will force push and rewrite history!${NC}"
read -p "Continue with force push? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$TEMP_DIR"
    git push --force --all
    cd - > /dev/null
    
    # Update local repository
    git fetch origin
    git reset --hard "origin/$CURRENT_BRANCH"
    
    echo -e "${GREEN}✓ Changes pushed successfully${NC}"
else
    echo -e "${YELLOW}Skipped force push. You can manually push later with:${NC}"
    echo "  cd $TEMP_DIR"
    echo "  git push --force --all"
    echo "  cd -"
    echo "  git fetch origin"
    echo "  git reset --hard origin/$CURRENT_BRANCH"
fi
echo ""

# Step 8: Cleanup
echo -e "${YELLOW}[8/8] Cleaning up temporary files...${NC}"
rm -rf "$TEMP_DIR"
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# Verify
echo "============================================"
echo "Verification"
echo "============================================"
git count-objects -vH

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ Cleanup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo "1. Verify your repo size is smaller (see above)"
echo "2. Your checkpoint file is still on disk (not deleted)"
echo "3. It's just no longer tracked by Git"
echo ""
echo "The file is located at:"
echo "  ClassificationModel/testing/Checkpoints/ClassificationModel/model.23-0.22.keras"
echo ""
echo "Consider uploading it to cloud storage and adding download instructions to README"