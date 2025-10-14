#!/bin/bash
# ============================================================================
# Release Tagging Script
# 릴리스 태그 생성 및 모델 체크포인트 버전 기록
# ============================================================================
#
# Usage:
#   ./scripts/create_release.sh v0.2.0
#   ./scripts/create_release.sh v0.2.0 --push
#
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

prompt_confirm() {
    read -p "$(echo -e "${YELLOW}$1 (y/n):${NC} ")" -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# Parse arguments
VERSION="$1"
PUSH_TAG=false

if [ $# -lt 1 ]; then
    log_error "Usage: $0 <version> [--push]"
    echo ""
    echo "Examples:"
    echo "  $0 v0.2.0"
    echo "  $0 v0.2.0 --push"
    exit 1
fi

if [ $# -gt 1 ] && [ "$2" == "--push" ]; then
    PUSH_TAG=true
fi

# Validate version format
if ! [[ $VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
    log_error "Invalid version format: $VERSION"
    log_info "Expected format: vX.Y.Z or vX.Y.Z-suffix"
    exit 1
fi

log_info "========================================"
log_info "Creating Release: $VERSION"
log_info "========================================"

# Check if on main/master branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]] && [[ "$CURRENT_BRANCH" != "master" ]]; then
    log_warning "Current branch is '$CURRENT_BRANCH', not main/master"
    if ! prompt_confirm "Continue anyway?"; then
        log_info "Release cancelled"
        exit 0
    fi
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    log_error "Working directory has uncommitted changes"
    git status --short
    log_info "Commit or stash changes before creating a release"
    exit 1
fi

log_success "Working directory is clean"

# Check if tag already exists
if git rev-parse "$VERSION" >/dev/null 2>&1; then
    log_error "Tag $VERSION already exists"
    log_info "Delete existing tag with: git tag -d $VERSION"
    exit 1
fi

log_success "Tag $VERSION is available"

# Get current commit
COMMIT_HASH=$(git rev-parse HEAD)
COMMIT_SHORT=$(git rev-parse --short HEAD)
COMMIT_MESSAGE=$(git log -1 --pretty=%B)

log_info "Current commit: $COMMIT_SHORT"
log_info "Commit message: ${COMMIT_MESSAGE:0:50}..."

# Tag model checkpoints
log_info ""
log_info "Tagging model checkpoints..."

CHECKPOINT_DIR="checkpoints"
MODEL_DIR="models"
VERSION_TAGGED=false

if [ -d "$CHECKPOINT_DIR" ]; then
    # Count checkpoint files
    CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.bin" \) | wc -l)

    if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
        log_info "Found $CHECKPOINT_COUNT checkpoint files"

        # Create version tag file
        VERSION_TAG_FILE="$CHECKPOINT_DIR/VERSION_${VERSION}.txt"
        echo "$VERSION" > "$VERSION_TAG_FILE"
        echo "Commit: $COMMIT_SHORT" >> "$VERSION_TAG_FILE"
        echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$VERSION_TAG_FILE"
        echo "" >> "$VERSION_TAG_FILE"
        echo "Checkpoint Files:" >> "$VERSION_TAG_FILE"

        # List all checkpoint files with hashes
        while IFS= read -r checkpoint; do
            if [ -f "$checkpoint" ]; then
                HASH=$(shasum -a 256 "$checkpoint" | awk '{print $1}')
                FILENAME=$(basename "$checkpoint")
                echo "  $FILENAME: $HASH" >> "$VERSION_TAG_FILE"
            fi
        done < <(find "$CHECKPOINT_DIR" -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.bin" \))

        log_success "Created checkpoint version tag: $VERSION_TAG_FILE"
        VERSION_TAGGED=true

        # Add to git
        git add "$VERSION_TAG_FILE"
        git commit -m "chore: Tag checkpoints for $VERSION"
        log_success "Committed checkpoint version tag"
    else
        log_info "No checkpoint files found in $CHECKPOINT_DIR"
    fi
else
    log_info "Checkpoint directory not found: $CHECKPOINT_DIR"
fi

# Tag models if exists
if [ -d "$MODEL_DIR" ]; then
    MODEL_COUNT=$(find "$MODEL_DIR" -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.bin" \) | wc -l)

    if [ "$MODEL_COUNT" -gt 0 ]; then
        log_info "Found $MODEL_COUNT model files"

        MODEL_TAG_FILE="$MODEL_DIR/VERSION_${VERSION}.txt"
        echo "$VERSION" > "$MODEL_TAG_FILE"
        echo "Commit: $COMMIT_SHORT" >> "$MODEL_TAG_FILE"
        echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MODEL_TAG_FILE"
        echo "" >> "$MODEL_TAG_FILE"
        echo "Model Files:" >> "$MODEL_TAG_FILE"

        while IFS= read -r model; do
            if [ -f "$model" ]; then
                HASH=$(shasum -a 256 "$model" | awk '{print $1}')
                FILENAME=$(basename "$model")
                echo "  $FILENAME: $HASH" >> "$MODEL_TAG_FILE"
            fi
        done < <(find "$MODEL_DIR" -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.bin" \))

        log_success "Created model version tag: $MODEL_TAG_FILE"

        git add "$MODEL_TAG_FILE"
        git commit -m "chore: Tag models for $VERSION"
        log_success "Committed model version tag"
        VERSION_TAGGED=true
    fi
fi

# Create Git tag
log_info ""
log_info "Creating Git tag..."

# Prompt for release notes
log_info "Enter release notes (press Ctrl+D when done):"
RELEASE_NOTES=$(cat)

# Create annotated tag
if [ -n "$RELEASE_NOTES" ]; then
    git tag -a "$VERSION" -m "$RELEASE_NOTES"
else
    git tag -a "$VERSION" -m "Release $VERSION"
fi

log_success "Created Git tag: $VERSION"

# Show tag info
log_info ""
log_info "Tag information:"
git show "$VERSION" --no-patch

# Push to remote
if [ "$PUSH_TAG" = true ]; then
    log_info ""
    log_warning "Pushing tag to remote..."

    if prompt_confirm "Push tag $VERSION to origin?"; then
        git push origin "$VERSION"

        # Push model/checkpoint version tags if created
        if [ "$VERSION_TAGGED" = true ]; then
            git push origin HEAD
            log_success "Pushed model version tags"
        fi

        log_success "Pushed tag to origin"
    else
        log_info "Tag not pushed (use: git push origin $VERSION)"
    fi
else
    log_info ""
    log_info "Tag created locally. Push with:"
    log_info "  git push origin $VERSION"
fi

# Generate release notes
log_info ""
log_info "Generating release notes..."

RELEASE_NOTES_FILE="releases/RELEASE_NOTES_${VERSION}.md"
mkdir -p releases

if [ -f "scripts/generate_release_notes.py" ]; then
    python scripts/generate_release_notes.py --from $(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "HEAD~10") --to "$VERSION" --output "$RELEASE_NOTES_FILE" || {
        log_warning "Failed to generate release notes automatically"
    }

    if [ -f "$RELEASE_NOTES_FILE" ]; then
        log_success "Release notes generated: $RELEASE_NOTES_FILE"
    fi
else
    log_warning "Release notes generator not found: scripts/generate_release_notes.py"
fi

# Create artifact manifest
log_info ""
log_info "Creating artifact manifest..."

MANIFEST_DIR="releases"
MANIFEST_FILE="$MANIFEST_DIR/manifest_${VERSION}.json"
mkdir -p "$MANIFEST_DIR"

cat > "$MANIFEST_FILE" <<EOF
{
  "version": "$VERSION",
  "release_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git": {
    "commit": "$COMMIT_HASH",
    "branch": "$CURRENT_BRANCH",
    "tag": "$VERSION"
  },
  "artifacts": {
    "models": "$([ -f "$MODEL_DIR/VERSION_${VERSION}.txt" ] && echo "tagged" || echo "none")",
    "checkpoints": "$([ -f "$CHECKPOINT_DIR/VERSION_${VERSION}.txt" ] && echo "tagged" || echo "none")"
  },
  "docker_images": {
    "app": "fragrance-ai-app:$VERSION",
    "worker-llm": "fragrance-ai-worker-llm:$VERSION",
    "worker-rl": "fragrance-ai-worker-rl:$VERSION"
  }
}
EOF

log_success "Artifact manifest created: $MANIFEST_FILE"

# Calculate manifest hash
MANIFEST_HASH=$(shasum -a 256 "$MANIFEST_FILE" | awk '{print $1}')
echo "$MANIFEST_HASH  $(basename $MANIFEST_FILE)" > "$MANIFEST_DIR/manifest_${VERSION}.sha256"

log_success "Manifest hash: $MANIFEST_HASH"

# Summary
log_info ""
log_success "========================================"
log_success "Release $VERSION created successfully!"
log_success "========================================"
echo ""
log_info "Next steps:"
echo "  1. Review release notes: $RELEASE_NOTES_FILE"
echo "  2. Build Docker images: docker-compose build"
echo "  3. Run pre-deployment check: python scripts/pre_deployment_check.py --version $VERSION"
echo "  4. Deploy: ./scripts/deploy.sh production $VERSION"
echo ""

if [ "$PUSH_TAG" = false ]; then
    log_info "Push tag to remote:"
    echo "  git push origin $VERSION"
    echo ""
fi
