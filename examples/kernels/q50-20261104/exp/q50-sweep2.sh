# q50-sweep2.sh WORKDIR OUT ROUNDS: diagnose Q5_0 mat-vec variants on the decode shapes with kernel_bench.
# WORKDIR holds kernel_bench and guest/ (shaders); OUT gets spv/ and results.jsonl. Activations and
# outputs are device-local, as with GGML_VK_PREFER_DEVICE_LOCAL.
set -u
W=$1; O=$2; ROUNDS=$3
mkdir -p $O/spv
U8=$W/guest/shaders/mul_mat_vecq_u8.comp
glsl() { name=$1; shift; glslc -fshader-stage=compute --target-env=vulkan1.2 -O "$@" -o $O/spv/$name.spv || echo "COMPILE_FAIL $name"; }
# name defines rows_per_workgroup
VARIANTS="base_t4::32 diag_loads:-DDIAG_LOADS:32 diag_alu:-DDIAG_ALU:32 sh28_t4:-DACT_SHARED,-DMAX_ACT_BLOCKS=28:32 wsh_t4:-DW_SHARED:32 wsh_sh28:-DW_SHARED,-DACT_SHARED,-DMAX_ACT_BLOCKS=28:32 base_t16:-DTPR=16:8"
for v in $VARIANTS; do
    name=${v%%:*}; rest=${v#*:}; defs=${rest%:*}
    glsl $name -DDATA_A_Q5_0 $(echo $defs | tr ',' ' ') $U8
done
glsl read_uvec4 -DWIDTH=4 $W/guest/shaders/bench/read_pattern.comp
glsl read_uint -DWIDTH=1 $W/guest/shaders/bench/read_pattern.comp
run() { $W/kernel_bench "$@" --io-memory device >> $O/results.jsonl 2>> $O/stderr.log || echo "RUN_FAIL $*" >> $O/stderr.log; }
for round in $(seq 1 $ROUNDS); do
    for shape in "4864 896 64" "896 896 128" "128 896 256"; do
        set -- $shape; m=$1; k=$2; reps=$3
        for v in $VARIANTS; do
            name=${v%%:*}; rows=${v##*:}
            [ -f $O/spv/$name.spv ] || continue
            run $O/spv/$name.spv q5_0 $m $k $rows 128,$rows,1 --subgroup 128 --b-format q8_1x4 --copies 8 --reps $reps \
                --label "r$round/$name/$m"
        done
        words=$((m * (k / 32) * 22 / 4)); wg_rows=$(( (m + 1215) / 1216 ))
        for r in read_uvec4 read_uint; do
            $W/kernel_bench $O/spv/$r.spv q5_0 $m $k $wg_rows 128,$words --copies 8 --reps $reps --io-memory device \
                --label "r$round/roofline_${r#read_}/$m" >> $O/results.jsonl 2>> $O/stderr.log
        done
    done
done
echo "SWEEP_DONE $(wc -l < $O/results.jsonl) rows"
