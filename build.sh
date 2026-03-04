#!/usr/bin/env bash
# Build the melanoma classifier pipeline.
# Runs cmake/make on the *host* OS so it uses the host's OpenCV installation,
# working around the VS Code Flatpak sandbox which has no OpenCV headers.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$REPO_DIR/build"

run_host() {
    # If inside a Flatpak sandbox, forward commands to the host shell.
    if [ -n "${FLATPAK_ID:-}" ]; then
        flatpak-spawn --host "$@"
    else
        "$@"
    fi
}

echo "==> Configuring (Release)..."
run_host cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Symlink compile_commands.json to repo root for VS Code IntelliSense
[ -f "$BUILD_DIR/compile_commands.json" ] && \
    ln -sf "$BUILD_DIR/compile_commands.json" "$REPO_DIR/compile_commands.json"

echo "==> Building..."
run_host cmake --build "$BUILD_DIR" --parallel "$(nproc)"

echo ""
echo "Build complete. Run the pipeline from the build/ directory so that"
echo "the relative data paths (../skin-cancer-mnist-ham10000/ etc.) resolve:"
echo ""
echo "  cd $BUILD_DIR && ./pipeline"
