#!/bin/sh
# Run in Debian. Source must be a fresh pinned llama.cpp v0.8.0 tree.
# Usage: sh guest/build-inference-runtime.sh SOURCE STOCK_RUNTIME NEW_RUNTIME
set -eu
src=$(realpath "$1")
stock=$(realpath "$2")
out=$(realpath -m "$3")
repo=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
if [ -e "$out" ]; then echo "Output already exists: $out" >&2; exit 1; fi
source_hash=$(sha256sum "$src/ggml/src/ggml-vulkan/ggml-vulkan.cpp" | cut -c1-64)
for patch_name in core16 powervr-mmv powervr-rms core-fp16; do
    patch -d "$src" -p1 < "$repo/patches/llama-b11960-$patch_name.patch"
done
cmake -S "$src" -B "$src/build-spiralpixcel" -G Ninja     -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O1 -DNDEBUG" -DCMAKE_C_FLAGS_RELEASE="-O1 -DNDEBUG"     -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_VULKAN=ON -DGGML_NATIVE=OFF     -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF     -DLLAMA_BUILD_SERVER=OFF -DGGML_CCACHE=OFF
cmake --build "$src/build-spiralpixcel" --target ggml-vulkan -j1
mkdir -p "$(dirname "$out")"
stage=$(mktemp -d "$(dirname "$out")/runtime.partial.XXXXXX")
trap 'rm -rf "$stage"' EXIT
cp -a "$stock/." "$stage/"
cp "$src/build-spiralpixcel/bin/libggml-vulkan.so" "$stage/"
c++ -std=c++17 -O1 -I "$src/ggml/include" "$repo/guest/check_mmv.cpp"     -L "$stage" -lggml -lggml-base -o "$stage/check-mmv"
python3 - "$stage" "$source_hash" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
data={"source_vulkan_cpp_before_sha256":sys.argv[2],
      "libggml-vulkan_sha256":hashlib.sha256((p/"libggml-vulkan.so").read_bytes()).hexdigest(),
      "check-mmv_sha256":hashlib.sha256((p/"check-mmv").read_bytes()).hexdigest(),
      "patches":["core16","core-fp16","powervr-mmv-wg","powervr-rms-subgroups"],"build_cxx_optimization":"O1","jobs":1}
(p/"spiralpixcel-kernel.json").write_text(json.dumps(data,indent=2)+"\n")
PY
LD_LIBRARY_PATH="$stage" "$stage/check-mmv" "$stage" >"$stage/kernel-check.jsonl"
sync "$stage/libggml-vulkan.so" "$stage/spiralpixcel-kernel.json"
mv "$stage" "$out"
trap - EXIT
echo "Installed checked runtime: $out"
