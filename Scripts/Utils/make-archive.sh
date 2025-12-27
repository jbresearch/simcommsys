#!/bin/bash

# Setup safety settings (exit on error, unset variables, or pipe failures)
set -euo pipefail

# Get the repo name from the current folder
REPO_NAME=$(basename "$PWD")

echo "Creating a temporary directory..."
# mktemp -d creates a secure, unique directory in /tmp
TEMP_DIR=$(mktemp -d)

# Setup a trap to delete the temp folder automatically when the script exits
# This runs whether the script succeeds or fails
cleanup() {
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    echo "Done."
}
trap cleanup EXIT

echo "Cloning the current repository into temporary location..."
# Clone the current directory ($PWD) to the temp directory
# To speed things up, only clone from the latest tag to the tip of this branch
# (this is what's needed to determine the version)
LATEST_TAG=$(git describe --tags --abbrev=0)
COMMIT_COUNT=$(git rev-list --count "${LATEST_TAG}..HEAD")
DEPTH=$((COMMIT_COUNT + 1))
git clone --depth "$DEPTH" "file://$PWD" "$TEMP_DIR"

echo "Working out version information..."
make -C "$TEMP_DIR" version.txt
VERSION=$(cat "$TEMP_DIR/version.txt")
echo $VERSION

echo "Removing build files..."
git -C "$TEMP_DIR" clean -dfX

echo "Removing git-specific files..."
# Remove the .git directory inside the temp folder
rm -rf "$TEMP_DIR/.git"
# Remove .gitignore or .gitattributes if you don't want them in the archive
rm -f "$TEMP_DIR/.git*"

echo "Creating the archive..."
# Create the tarball
# (change directory, so the archive doesn't contain the full /tmp/ path)
ARCHIVE_NAME="${REPO_NAME}_${VERSION}.tar.gz"
tar -czf "$ARCHIVE_NAME" -C "$TEMP_DIR" .

echo "---------------------------------------"
echo "Success! Archive created: $ARCHIVE_NAME"
