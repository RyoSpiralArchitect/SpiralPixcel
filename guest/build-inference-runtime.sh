#!/bin/sh
# Run in Debian. Source must be a fresh pinned llama.cpp v0.8.0 tree, or with --resume the
# tree of an interrupted run of this script with the same options and patch files.
# Usage: sh guest/build-inference-runtime.sh SOURCE STOCK_RUNTIME NEW_RUNTIME [--output-row-split] [--powervr-u8-mmvq]
#        [--powervr-decode] [--resume]
set -eu
# Each check below sets the switches it means to test; inherited ones would change the path.
unset GGML_VK_PWR_U8_MMVQ GGML_VK_PWR_MMV_WG GGML_VK_PWR_RMS_SUBGROUPS GGML_VK_FORCE_MMVQ GGML_VK_DISABLE_MMVQ \
    GGML_VK_DISABLE_F16 GGML_VK_PREFER_HOST_MEMORY GGML_VK_SPLIT_OUTPUT GGML_VK_DISABLE_ASYNC GGML_VK_WAIT_IDLE \
    GGML_VK_SERIALIZE_SUBMISSIONS GGML_VK_PERF_LOGGER GGML_VK_PWR_ATTN GGML_VK_PWR_BARRIER GGML_VK_PREFER_DEVICE_LOCAL
if [ "$#" -lt 3 ] || [ "$#" -gt 7 ]; then
    echo "Usage: $0 SOURCE STOCK_RUNTIME NEW_RUNTIME [--output-row-split] [--powervr-u8-mmvq] [--powervr-decode] [--resume]" >&2
    exit 2
fi
src=$(realpath "$1")
stock=$(realpath "$2")
out=$(realpath -m "$3")
shift 3
split_output=0
u8_mmvq=0
decode=0
resume=0
for option in "$@"; do
    case "$option" in
        --output-row-split) split_output=1 ;;
        --powervr-u8-mmvq) u8_mmvq=1 ;;
        --powervr-decode) decode=1 ;;
        --resume) resume=1 ;;
        *) echo "Unknown option: $option" >&2; exit 2 ;;
    esac
done
repo=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
if [ "$decode" -eq 1 ] && { [ "$split_output" -eq 0 ] || [ "$u8_mmvq" -eq 0 ]; }; then
    # The decode patch applies on top of both: it edits their staging copies and device flags.
    echo "--powervr-decode needs --output-row-split and --powervr-u8-mmvq" >&2
    exit 2
fi
if [ -e "$out" ] || [ -L "$out" ]; then
    echo "Output already exists: $out" >&2
    exit 1
fi
patch_names="core16 powervr-mmv powervr-rms core-fp16"
if [ "$split_output" -eq 1 ]; then
    patch_names="$patch_names output-row-split metadata-views bounded-upload host-read-barrier coherent-readback completion-wait"
fi
if [ "$u8_mmvq" -eq 1 ]; then
    # Opt-in at run time with GGML_VK_PWR_U8_MMVQ=1. Its new shader rebuilds every shader once.
    patch_names="$patch_names powervr-u8-mmvq"
fi
if [ "$decode" -eq 1 ]; then
    # Opt-in at run time with GGML_VK_PWR_ATTN=1, GGML_VK_PWR_BARRIER=1 and GGML_VK_PREFER_DEVICE_LOCAL=1.
    patch_names="$patch_names powervr-decode"
fi
patch_list=""
for patch_name in $patch_names; do
    patch_list="$patch_list $patch_name=$(sha256sum "$repo/patches/llama-b11960-$patch_name.patch" | cut -c1-64)"
done
# Android can stop the Terminal VM mid-build (MemoryLimiter); the marker lets a later run
# continue the same patched tree instead of patching it twice.
marker="$src/.spiralpixcel-build"
# The marker alone is no proof: reverse the patches on a copy of the files they touch and
# require the recorded pre-patch source.
verify_patched_tree() {
    check=$(mktemp -d)
    for patch_name in $patch_names; do
        sed -n 's|^+++ b/||p' "$repo/patches/llama-b11960-$patch_name.patch" | cut -f1
    done | sort -u | while read -r file; do
        mkdir -p "$check/$(dirname "$file")"
        cp "$src/$file" "$check/$file"
    done
    reversed=""
    for patch_name in $patch_names; do
        reversed="$patch_name $reversed"
    done
    for patch_name in $reversed; do
        if ! patch -d "$check" -p1 -R -s -f < "$repo/patches/llama-b11960-$patch_name.patch" >/dev/null; then
            rm -rf "$check"
            echo "Patch $patch_name is not applied as recorded in $src" >&2
            exit 1
        fi
    done
    restored=$(sha256sum "$check/ggml/src/ggml-vulkan/ggml-vulkan.cpp" | cut -c1-64)
    rm -rf "$check"
    if [ "$restored" != "$source_hash" ]; then
        echo "Reversing the patches does not give the recorded source in $src" >&2
        exit 1
    fi
}
if [ "$resume" -eq 1 ]; then
    if [ ! -f "$marker" ] || [ "$(sed -n 's/^patches=//p' "$marker")" != "$patch_list" ]; then
        echo "No interrupted build with these options and patches in $src" >&2
        exit 1
    fi
    if grep -q '^completed=' "$marker"; then
        echo "This tree already completed $(sed -n 's/^completed=//p' "$marker"); start from a fresh tree" >&2
        exit 1
    fi
    source_hash=$(sed -n 's/^source_vulkan_cpp_before_sha256=//p' "$marker")
    verify_patched_tree
else
    if [ -e "$marker" ]; then
        echo "Source was already patched by this script; use --resume if its build was interrupted: $src" >&2
        exit 1
    fi
    source_hash=$(sha256sum "$src/ggml/src/ggml-vulkan/ggml-vulkan.cpp" | cut -c1-64)
    for patch_name in $patch_names; do
        patch -d "$src" -p1 < "$repo/patches/llama-b11960-$patch_name.patch"
    done
    printf 'source_vulkan_cpp_before_sha256=%s\npatches=%s\n' "$source_hash" "$patch_list" > "$marker"
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
python3 - "$stage" "$source_hash" "$split_output" "$u8_mmvq" "$resume" "$patch_list" "$decode" <<'PY'
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
    "build_cxx_optimization": "O1", "jobs": 1, "resumed_build": sys.argv[5] == "1",
    "patch_sha256": dict(item.split("=", 1) for item in sys.argv[6].split()),
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
if sys.argv[4] == "1":
    data["patches"] += ["powervr-u8-mmvq"]
    data["powervr_u8_mmvq"] = True
    data["powervr_u8_mmvq_types"] = ["q4_K", "q5_0", "q6_K", "q8_0"]
    data["powervr_u8_mmvq_variants"] = ["q5_0/act_shared", "q5_0/tpr16", "q8_0/act_shared", "q8_0/tpr16"]
if sys.argv[7] == "1":
    data["patches"] += ["powervr-decode"]
    data["powervr_decode"] = True
(p / "spiralpixcel-kernel.json").write_text(json.dumps(data, indent=2) + "\n")
PY
if [ "$split_output" -eq 1 ]; then
    GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 LD_LIBRARY_PATH="$stage" "$stage/check-readback" "$stage" >"$stage/readback-check.jsonl"
    GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 LD_LIBRARY_PATH="$stage" "$stage/check-mmv" "$stage" >"$stage/kernel-check.jsonl"
    GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 LD_LIBRARY_PATH="$stage" "$stage/check-large" "$stage" >"$stage/large-check.jsonl"
else
    LD_LIBRARY_PATH="$stage" "$stage/check-mmv" "$stage" >"$stage/kernel-check.jsonl"
fi
if [ "$u8_mmvq" -eq 1 ] && [ "$split_output" -eq 1 ]; then
    GGML_VK_PWR_U8_MMVQ=1 GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 \
        LD_LIBRARY_PATH="$stage" "$stage/check-mmv" "$stage" >"$stage/kernel-check-u8.jsonl" 2>"$stage/kernel-check-u8.log"
elif [ "$u8_mmvq" -eq 1 ]; then
    GGML_VK_PWR_U8_MMVQ=1 LD_LIBRARY_PATH="$stage" "$stage/check-mmv" "$stage" \
        >"$stage/kernel-check-u8.jsonl" 2>"$stage/kernel-check-u8.log"
fi
if [ "$u8_mmvq" -eq 1 ]; then
    # Passing cases alone do not show the kernel ran: a missing pipeline falls back silently.
    for type_name in q4_K q5_0 q6_K q8_0; do
        if ! grep -q "ggml_vulkan: mul_mat_vecq_u8 selected for $type_name\$" "$stage/kernel-check-u8.log"; then
            echo "The unsigned mat-vec kernel did not run for $type_name; see kernel-check-u8.log" >&2
            exit 1
        fi
    done
    # The shape variants are picked by row count; the 4864- and 128-row cases reach both.
    for type_name in q5_0 q8_0; do
        for variant in act_shared tpr16; do
            if ! grep -q "ggml_vulkan: mul_mat_vecq_u8 variant $variant selected for $type_name\$" "$stage/kernel-check-u8.log"; then
                echo "The unsigned mat-vec kernel's $variant variant did not run for $type_name; see kernel-check-u8.log" >&2
                exit 1
            fi
        done
    done
fi
if [ "$decode" -eq 1 ]; then
    # The decode switches change barriers and buffer placement for every operation, so the
    # transfer, kernel and large-output checks run again with all of them on.
    for checker in check-readback check-mmv check-large; do
        case "$checker" in
            check-readback) receipt=readback-check-decode ;;
            check-mmv) receipt=kernel-check-decode ;;
            *) receipt=large-check-decode ;;
        esac
        GGML_VK_PWR_U8_MMVQ=1 GGML_VK_PWR_ATTN=1 GGML_VK_PWR_BARRIER=1 GGML_VK_PREFER_DEVICE_LOCAL=1 \
            GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 LD_LIBRARY_PATH="$stage" \
            "$stage/$checker" "$stage" >"$stage/$receipt.jsonl" 2>"$stage/$receipt.log"
    done
    if ! grep -q "ggml_vulkan: mul_mat_vec_p021_pvr selected\$" "$stage/kernel-check-decode.log"; then
        echo "The attention-score kernel did not run; see kernel-check-decode.log" >&2
        exit 1
    fi
fi
sync -f "$stage"
if [ -e "$out" ] || [ -L "$out" ]; then
    echo "Output appeared during build: $out" >&2
    exit 1
fi
mv -T "$stage" "$out"
trap - EXIT
printf 'completed=%s\n' "$out" >> "$marker"
if [ "$split_output" -eq 1 ]; then
    echo "Installed experimental runtime (model validation pending): $out"
else
    echo "Installed checked runtime: $out"
fi
