#!/usr/bin/env bash

set -euo pipefail

REPO_URL="https://github.com/CrispStrobe/CrispASR"
SOURCE_DIR="${CRISPASR_SOURCE_DIR:-CrispASR}"
FINAL_BIN_DIR="bin"
FINAL_BIN_NAME="crispasr"

usage() {
	cat <<'EOF'
Usage: ./build_binary.sh [--help]

Builds CrispASR from source and installs the executable at bin/crispasr.
Set CRISPASR_SOURCE_DIR to use a different source checkout directory.
Set BUILD_JOBS to override the detected parallel build job count.
Set CRISPASR_REF to a tag or commit (e.g. v0.8.55) to build a pinned version.
Set BUILD_TARGET to build only some targets, space-separated (CI uses "crispasr-cli crispasr-lib":
the CLI and the shared library the session runner loads).
EOF
}

if [[ $# -gt 0 ]]; then
	if [[ "$#" -eq 1 && "$1" == "--help" ]]; then
		usage
		exit 0
	fi
	echo "Unknown option: $1" >&2
	usage
	exit 1
fi

for command_name in git cmake; do
	if ! command -v "$command_name" >/dev/null 2>&1; then
		echo "$command_name is required to build CrispASR." >&2
		exit 1
	fi
done

if [[ -e "$SOURCE_DIR" ]]; then
	if [[ ! -e "$SOURCE_DIR/.git" ]]; then
		echo "Source directory exists but is not a Git checkout: $SOURCE_DIR" >&2
		exit 1
	fi
	echo "[submodules] Initializing CrispASR submodules"
	git -C "$SOURCE_DIR" submodule update --init --recursive
else
	echo "[clone] $REPO_URL -> $SOURCE_DIR"
	git clone --recursive "$REPO_URL" "$SOURCE_DIR"
fi

if [[ -n "${CRISPASR_REF:-}" ]]; then
	echo "[checkout] $CRISPASR_REF"
	git -C "$SOURCE_DIR" fetch --tags --quiet origin
	git -C "$SOURCE_DIR" checkout --quiet "$CRISPASR_REF"
	git -C "$SOURCE_DIR" submodule update --init --recursive
fi

if [[ -n "${BUILD_JOBS:-}" ]]; then
	jobs="$BUILD_JOBS"
elif command -v nproc >/dev/null 2>&1; then
	jobs="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
	jobs="$(sysctl -n hw.ncpu)"
elif command -v getconf >/dev/null 2>&1; then
	jobs="$(getconf _NPROCESSORS_ONLN)"
else
	jobs=2
fi

echo "[configure] $SOURCE_DIR/build"
cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" -DCMAKE_BUILD_TYPE=Release
target_args=()
if [[ -n "${BUILD_TARGET:-}" ]]; then
	read -ra targets <<<"$BUILD_TARGET"
	target_args=(--target "${targets[@]}")
fi
echo "[build] CrispASR with $jobs jobs ${BUILD_TARGET:+(target: $BUILD_TARGET)}"
cmake --build "$SOURCE_DIR/build" --parallel "$jobs" ${target_args[@]+"${target_args[@]}"}

source_bin="$SOURCE_DIR/build/bin/$FINAL_BIN_NAME"
if [[ ! -f "$source_bin" && -f "$source_bin.exe" ]]; then
	source_bin="$source_bin.exe"
fi
if [[ ! -f "$source_bin" ]]; then
	echo "Built executable not found: $source_bin" >&2
	exit 1
fi

dest_bin="$FINAL_BIN_DIR/$FINAL_BIN_NAME"
mkdir -p "$FINAL_BIN_DIR"
echo "[install] Copying $source_bin -> $dest_bin"
cp -f "$source_bin" "$dest_bin"
chmod +x "$dest_bin"

echo "[done] Binary installed at: $dest_bin"
for lib in "$SOURCE_DIR/build/src/libcrispasr.dylib" "$SOURCE_DIR/build/src/libcrispasr.so"; do
	if [[ -f "$lib" ]]; then
		echo "[hint] For --runner session: export CRISPASR_LIB_PATH=$lib"
		break
	fi
done
