#!/bin/sh
# Run in Debian. Source must be a fresh pinned llama.cpp v0.8.0 tree.
# Usage: sh guest/build-inference-runtime.sh SOURCE STOCK_RUNTIME NEW_RUNTIME [--output-row-split]
set -eu
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 SOURCE STOCK_RUNTIME NEW_RUNTIME [--output-row-split]" >&2
    exit 2
fi
split_output=0
if [ "$#" -eq 4 ]; then
    if [ "$4" != --output-row-split ]; then
        echo "Unknown option: $4" >&2
        exit 2
    fi
    split_output=1
fi
src=$(realpath "$1")
stock=$(realpath "$2")
out=$(realpath -m "$3")
repo=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
if [ -e "$out" ] || [ -L "$out" ]; then
    echo "Output already exists: $out" >&2
    exit 1
fi
source_hash=$(sha256sum "$src/ggml/src/ggml-vulkan/ggml-vulkan.cpp" | cut -c1-64)
for patch_name in core16 powervr-mmv powervr-rms core-fp16; do
    patch -d "$src" -p1 < "$repo/patches/llama-b11960-$patch_name.patch"
done
if [ "$split_output" -eq 1 ]; then
    for patch_name in output-row-split metadata-views bounded-upload host-read-barrier coherent-readback completion-wait; do
        patch -d "$src" -p1 < "$repo/patches/llama-b11960-$patch_name.patch"
    done
fi
build="$src/build-spiralpixcel"
build_tools=OFF
if [ "$split_output" -eq 1 ]; then
    build_tools=ON
fi
cmake -S "$src" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O1 -DNDEBUG" -DCMAKE_C_FLAGS_RELEASE="-O1 -DNDEBUG" \
    -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_VULKAN=ON -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES="$build_tools" -DLLAMA_BUILD_TOOLS="$build_tools" \
    -DLLAMA_BUILD_SERVER=OFF -DGGML_CCACHE=OFF
if [ "$split_output" -eq 1 ]; then
    cmake --build "$build" --target ggml-cpu llama-bench llama-completion llama ggml ggml-vulkan -j1
else
    cmake --build "$build" --target ggml ggml-vulkan -j1
fi
mkdir -p "$(dirname "$out")"
stage=$(mktemp -d "$(dirname "$out")/runtime.partial.XXXXXX")
trap 'rm -rf "$stage"' EXIT
cp -a "$stock/." "$stage/"

# Replace the entire family, including aliases, without following stock symlinks.
copy_library_family() {
    library_family=$1
    set -- "$build/bin/$library_family.so"*
    if [ ! -f "$1" ]; then
        echo "Built library missing: $library_family" >&2
        exit 1
    fi
    rm -f "$stage/$library_family.so" "$stage/$library_family.so."*
    cp -a "$@" "$stage/"
}
if [ "$split_output" -eq 1 ]; then
    # A split runtime must not load a stock CPU variant, RPC backend, or tool
    # dependency against the rebuilt core. Remove stock shared libraries first.
    find "$stage" \( -type f -o -type l \) -name '*.so*' -delete
    for library_family in libggml libggml-base libggml-cpu libggml-vulkan libllama; do
        if [ ! -f "$build/bin/$library_family.so" ]; then
            echo "Built library missing: $library_family" >&2
            exit 1
        fi
    done
    set -- "$build/bin/"*.so*
    cp -a "$@" "$stage/"
    for executable in llama-bench llama-completion; do
        if [ ! -x "$build/bin/$executable" ]; then
            echo "Built executable missing: $executable" >&2
            exit 1
        fi
        rm -f "$stage/$executable"
        cp -a "$build/bin/$executable" "$stage/"
    done
    rm -f "$stage/check-large" "$stage/check-model" "$stage/check-readback"
else
    copy_library_family libggml
    copy_library_family libggml-base
    copy_library_family libggml-vulkan
fi
rm -f "$stage/check-mmv"
c++ -std=c++17 -O1 -I "$src/ggml/include" "$repo/guest/check_mmv.cpp" \
    -L "$stage" -lggml -lggml-base -o "$stage/check-mmv"
if [ "$split_output" -eq 1 ]; then
    c++ -std=c++17 -O1 -I "$src/ggml/include" "$repo/guest/check_readback.cpp" \
        -L "$stage" -lggml -lggml-base -o "$stage/check-readback"
    c++ -std=c++17 -O1 -I "$src/ggml/include" "$repo/guest/check_large.cpp" \
        -L "$stage" -lggml -lggml-base -o "$stage/check-large"
    c++ -std=c++17 -O1 -I "$src/include" -I "$src/ggml/include" "$repo/guest/check_model.cpp" \
        -L "$stage" -lllama -lggml -lggml-base -o "$stage/check-model"
fi
python3 - "$stage" "$source_hash" "$split_output" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

p = Path(sys.argv[1])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

data = {
    "source_vulkan_cpp_before_sha256": sys.argv[2],
    "libggml-vulkan_sha256": digest(p / "libggml-vulkan.so"),
    "check-mmv_sha256": digest(p / "check-mmv"),
    "core_library_sha256": {path.name: digest(path) for family in ("libggml", "libggml-base")
                            for path in sorted(p.glob(family + ".so*")) if path.is_file()},
    "patches": ["core16", "core-fp16", "powervr-mmv-wg", "powervr-rms-subgroups"],
    "build_cxx_optimization": "O1", "jobs": 1,
}
if sys.argv[3] == "1":
    executables = ("llama-bench", "llama-completion", "check-mmv", "check-large", "check-model", "check-readback")
    data.update(
        output_row_split=True,
        experimental=True,
        validation_status="experimental_not_model_validated",
        model_correctness_validated=False,
        queue_idle_completion=True,
        validation_transfer_mode="GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1",
        libllama_sha256=digest(p / "libllama.so.0.8.0"),
        runtime_library_sha256={path.name: digest(path) for path in sorted(p.glob("*.so*")) if path.is_file()},
        runtime_executable_sha256={name: digest(p / name) for name in executables},
        **{"check-large_sha256": digest(p / "check-large"), "check-model_sha256": digest(p / "check-model")},
    )
    data["patches"] += ["output-row-split", "metadata-views", "bounded-upload", "host-read-barrier", "coherent-readback", "completion-wait"]
(p / "spiralpixcel-kernel.json").write_text(json.dumps(data, indent=2) + "\n")
PY
if [ "$split_output" -eq 1 ]; then
    GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 LD_LIBRARY_PATH="$stage" "$stage/check-readback" "$stage" >"$stage/readback-check.jsonl"
    GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 LD_LIBRARY_PATH="$stage" "$stage/check-mmv" "$stage" >"$stage/kernel-check.jsonl"
    GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 LD_LIBRARY_PATH="$stage" "$stage/check-large" "$stage" >"$stage/large-check.jsonl"
else
    LD_LIBRARY_PATH="$stage" "$stage/check-mmv" "$stage" >"$stage/kernel-check.jsonl"
fi
sync -f "$stage"
if [ -e "$out" ] || [ -L "$out" ]; then
    echo "Output appeared during build: $out" >&2
    exit 1
fi
mv -T "$stage" "$out"
trap - EXIT
if [ "$split_output" -eq 1 ]; then
    echo "Installed experimental runtime (model validation pending): $out"
else
    echo "Installed checked runtime: $out"
fi
