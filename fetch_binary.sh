#!/usr/bin/env bash

set -euo pipefail

REPO="CrispStrobe/CrispASR"
DEFAULT_VERSION="latest"
DEFAULT_OUT_DIR="src"
LINUX_EXTRACT_DIR_NAME="crispasr-linux-x86_64"
FINAL_BIN_DIR="src"
FINAL_BIN_NAME="crispasr"

ASSET_LINUX="crispasr-linux-x86_64.tar.gz"
ASSET_MACOS="crispasr-macos.tar.gz"
ASSET_WINDOWS_CUDA="crispasr-windows-x86_64-cuda.zip"

usage() {
	cat <<'EOF'
Usage:
	./fetch_binary.sh [options]

Options:
	--target <name>        Target alias: linux | macos | windows-cuda (default: inferred from OS)
	--version <tag>        Release tag (for example: v0.8.32). Use "latest" for newest release (default)

Examples:
	./fetch_binary.sh --target linux --version v0.8.32
	./fetch_binary.sh --target macos
	./fetch_binary.sh --target windows-cuda
	./fetch_binary.sh --target linux --version latest
EOF
}

finalize_linux=1

target=""
version="$DEFAULT_VERSION"
out_dir="$DEFAULT_OUT_DIR"
keep_archive=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		--target)
			[[ $# -ge 2 ]] || { echo "Missing value for --target" >&2; exit 1; }
			target="$2"
			shift 2
			;;
		--version)
			[[ $# -ge 2 ]] || { echo "Missing value for --version" >&2; exit 1; }
			version="$2"
			shift 2
			;;
		*)
			echo "Unknown option: $1" >&2
			usage
			exit 1
			;;
	esac
done

if [[ -z "$target" ]]; then
	case "$(uname -s)" in
		Linux)
			target="linux"
			;;
		Darwin)
			target="macos"
			;;
		*)
			echo "Could not infer target from OS. Use --target linux|macos|windows-cuda." >&2
			exit 1
			;;
	esac
fi

case "$target" in
	linux)
		asset_name="$ASSET_LINUX"
		;;
	macos)
		asset_name="$ASSET_MACOS"
		;;
	windows-cuda)
		asset_name="$ASSET_WINDOWS_CUDA"
		;;
	*)
		echo "Invalid target: $target. Expected linux, macos, or windows-cuda." >&2
		exit 1
		;;
esac

mkdir -p "$out_dir"
archive_path="$out_dir/$asset_name"

if [[ "$version" == "latest" ]]; then
	download_url="https://github.com/$REPO/releases/latest/download/$asset_name"
else
	download_url="https://github.com/$REPO/releases/download/$version/$asset_name"
fi

echo "[fetch] target=$target version=$version"
echo "[fetch] $download_url"
curl -fL "$download_url" -o "$archive_path"

echo "[extract] $archive_path -> $out_dir"
case "$asset_name" in
	*.tar.gz)
		tar -xzf "$archive_path" -C "$out_dir"
		;;
	*.zip)
		if ! command -v unzip >/dev/null 2>&1; then
			echo "unzip is required to extract ZIP archives. Install it and rerun." >&2
			exit 1
		fi
		unzip -o "$archive_path" -d "$out_dir"
		;;
	*)
		echo "Unsupported archive format: $asset_name" >&2
		exit 1
		;;
esac

if [[ "$target" == "linux" && $finalize_linux -eq 1 ]]; then
	source_dir="$out_dir/$LINUX_EXTRACT_DIR_NAME"
	source_bin="$source_dir/$FINAL_BIN_NAME"
	dest_bin="$FINAL_BIN_DIR/$FINAL_BIN_NAME"

	if [[ ! -d "$source_dir" ]]; then
		echo "Linux extract directory not found: $source_dir" >&2
		exit 1
	fi

	if [[ ! -f "$source_bin" ]]; then
		echo "Binary not found in extracted Linux directory: $source_bin" >&2
		exit 1
	fi

	echo "[finalize] Removing everything under $source_dir except $FINAL_BIN_NAME"
	find "$source_dir" -mindepth 1 ! -name "$FINAL_BIN_NAME" -exec rm -rf {} +

	mkdir -p "$FINAL_BIN_DIR"
	echo "[finalize] Copying $source_bin -> $dest_bin"
	cp -f "$source_bin" "$dest_bin"
	chmod +x "$dest_bin"

	echo "[finalize] Removing directory $source_dir"
	rm -rf "$source_dir"
fi

if [[ $keep_archive -eq 0 ]]; then
	rm -f "$archive_path"
fi

echo "[done] Files extracted to: $out_dir"