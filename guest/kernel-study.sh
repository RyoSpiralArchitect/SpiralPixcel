#!/bin/sh
# Run in Debian. Time ggml-vulkan's mat-vec shaders against the unsigned-dot kernels on the
# Qwen2.5-0.5B decode shapes, with roofline, ALU and reduction probes for context.
# Usage: sh guest/kernel-study.sh LLAMA_SOURCE RUNTIME OUTPUT_DIR
# LLAMA_SOURCE supplies ggml headers and shaders; RUNTIME supplies libggml-base.so.
set -eu
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 LLAMA_SOURCE RUNTIME OUTPUT_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
runtime=$(realpath "$2")
out=$(realpath -m "$3")
repo=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
if [ -e "$out" ]; then
    echo "Output already exists: $out" >&2
    exit 1
fi
mkdir -p "$out/spv"
ggml_shaders="$src/ggml/src/ggml-vulkan/vulkan-shaders"
c++ -std=c++17 -O2 -I "$src/ggml/include" "$repo/guest/kernel_bench.cpp" \
    -L "$runtime" -lggml-base -lvulkan -Wl,-rpath,"$runtime" -o "$out/kernel_bench"

# Same defines and flags as vulkan-shaders-gen for mul_mat_vec_<type>_f32_f32_subgroup_no_shmem.
glsl() { name=$1; shift; glslc -fshader-stage=compute --target-env=vulkan1.2 -O "$@" -o "$out/spv/$name.spv"; }
ggml_defs="-DFLOAT_TYPE=float -DFLOAT_TYPEV2=vec2 -DB_TYPE=float -DB_TYPEV2=vec2 -DB_TYPEV4=vec4 -DD_TYPE=float"
for type in q5_0 q8_0; do
    upper=$(echo "$type" | tr a-z A-Z)
    glsl "ggml_$type" $ggml_defs "-DDATA_A_$upper=1" -DUSE_SUBGROUP_ADD_NO_SHMEM=1 -I"$ggml_shaders" "$ggml_shaders/mul_mat_vec.comp"
    glsl "u8_$type" "-DDATA_A_$upper" "$repo/guest/shaders/mul_mat_vecq_u8.comp"
done
# K-quants use their own shaders and, on this driver, the shared-memory reduction variant.
for type in q4_k q6_k; do
    upper=$(echo "$type" | tr a-z A-Z)
    glsl "ggml_$type" $ggml_defs "-DDATA_A_$upper=1" -I"$ggml_shaders" "$ggml_shaders/mul_mat_vec_$type.comp"
done
glsl u8_q4_k -DDATA_A_Q4_K "$repo/guest/shaders/mul_mat_vecq_u8.comp"
glsl read_uvec4 -DWIDTH=4 "$repo/guest/shaders/bench/read_pattern.comp"
glsl read_uint -DWIDTH=1 "$repo/guest/shaders/bench/read_pattern.comp"
for dq in 1 3; do
    glsl "float_q5_0_dq$dq" -DTPR=4 "-DDQ=$dq" "$repo/guest/shaders/bench/mmv_q5_0_pairs.comp"
done
glsl empty -DTPR=4 -DDQ=-1 -DBENCH_NO_A -DBENCH_NO_B -DBENCH_NO_REDUCE "$repo/guest/shaders/bench/mmv_q5_0_pairs.comp"
for op in 0 1 2 3 4 5; do
    glsl "alu$op" "-DOP=$op" "$repo/guest/shaders/bench/alu_probe.comp"
done
for mode in 0 1; do
    glsl "rms$mode" "-DMODE=$mode" "$repo/guest/shaders/bench/rms_probe.comp"
done

failed=0
probe_errors=0
bench() { "$out/kernel_bench" "$@" >>"$out/results.jsonl" 2>>"$out/stderr.log"; }
kernel() { bench "$@" || failed=$((failed + 1)); }
# Probes do not compute the product, so their check fails by design (exit 3); any other
# exit means the probe did not run.
probe() { bench "$@" || { [ "$?" -eq 3 ] || probe_errors=$((probe_errors + 1)); }; }

# Decode shapes per token: gate/up 48, q/o 48, k/v 36 (Q5_0) and 12 (Q8_0), output head 2+1.
for shape in "q5_0 4864 896 64" "q5_0 896 896 128" "q5_0 128 896 256" \
             "q8_0 65536 896 8" "q8_0 20864 896 16" "q8_0 128 896 256"; do
    set -- $shape
    type=$1 m=$2 k=$3 reps=$4
    copies=8
    [ "$m" -gt 16384 ] && copies=2
    rows=2
    [ "$type" = q8_0 ] && rows=1
    kernel "$out/spv/ggml_$type.spv" "$type" "$m" "$k" "$rows" "128,$rows,1" --subgroup 128 \
        --copies "$copies" --reps "$reps" --label "ggml/$type/$m"
    kernel "$out/spv/u8_$type.spv" "$type" "$m" "$k" 32 128,32,1 --subgroup 128 --b-format q8_1x4 \
        --copies "$copies" --reps "$reps" --label "u8/$type/$m"
    block=22
    [ "$type" = q8_0 ] && block=34
    words=$((m * (k / 32) * block / 4))
    wg_rows=$(( (m + 1215) / 1216 ))
    probe "$out/spv/read_uvec4.spv" "$type" "$m" "$k" "$wg_rows" "128,$words" --copies "$copies" --reps "$reps" \
        --label "roofline_uvec4/$type/$m"
done
# ffn_down: 896x4864, Q4_K in half the layers and Q6_K in the other half.
kernel "$out/spv/ggml_q4_k.spv" q4_k 896 4864 2 128,2,1 --label ggml/q4_k/896
kernel "$out/spv/u8_q4_k.spv" q4_k 896 4864 4 128,4,1 --subgroup 128 --b-format q8_1x4 --label u8/q4_k/896
kernel "$out/spv/ggml_q6_k.spv" q6_k 896 4864 2 128,2,1 --label ggml/q6_k/896
kernel "$out/spv/float_q5_0_dq1.spv" q5_0 4864 896 32 128,32,1 --subgroup 128 --label float_dq1/q5_0/4864
kernel "$out/spv/float_q5_0_dq3.spv" q5_0 4864 896 32 128,32,1 --subgroup 128 --label float_dq3/q5_0/4864
probe "$out/spv/read_uint.spv" q5_0 4864 896 4 128,749056 --label roofline_uint/q5_0/4864
probe "$out/spv/empty.spv" q5_0 32 896 32 128,32,1 --subgroup 128 --reps 256 --label empty/1wg/memory_barrier
probe "$out/spv/empty.spv" q5_0 32 896 32 128,32,1 --subgroup 128 --reps 256 --barrier none --label empty/1wg/no_barrier
for op in 0 1 2 3 4 5; do
    probe "$out/spv/alu$op.spv" q5_0 4864 896 4 128 --reps 16 --label "alu/op$op/1216wg"
done
for mode in 0 1; do
    for block in 128 512; do
        probe "$out/spv/rms$mode.spv" f32 896 896 896 "$block" --subgroup 128 --copies 1 --reps 256 \
            --label "rms/mode$mode/block$block"
    done
done

python3 - "$out" "$repo" "$failed" "$probe_errors" "$src" "$runtime" <<'PY'
import hashlib, json, platform, subprocess, sys
from pathlib import Path
out, repo, failed, probe_errors = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
src, runtime = Path(sys.argv[5]), Path(sys.argv[6])
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources = [repo / "guest/kernel-study.sh", repo / "guest/kernel_bench.cpp", repo / "guest/shaders/mul_mat_vecq_u8.comp",
           *sorted((repo / "guest/shaders/bench").glob("*.comp"))]
# libggml-base quantizes the weights and supplies the reference dequantization.
record = {"failed_kernel_runs": failed, "probe_errors": probe_errors, "kernel": platform.release(),
          "llama_source": str(src), "runtime": str(runtime),
          "glslc": subprocess.run(["glslc", "--version"], capture_output=True, text=True).stdout.splitlines()[:2],
          "sha256": {**{str(p.relative_to(repo)): digest(p) for p in sources},
                     **{f"runtime/{p.name}": digest(p.resolve()) for p in sorted(runtime.glob("libggml-base.so*"))},
                     **{p.name: digest(p) for p in [out / "kernel_bench", out / "results.jsonl", out / "stderr.log",
                                                     *sorted((out / "spv").glob("*.spv"))]}}}
(out / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
PY
echo "Results: $out/results.jsonl ($failed failed kernel runs, $probe_errors probe errors)"
[ "$failed" -eq 0 ] && [ "$probe_errors" -eq 0 ]
