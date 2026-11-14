# q50-sweep4.sh WORKDIR OUT ROUNDS: confirm the Q5_0 picks and try them on Q8_0, with kernel_bench.
set -u
W=$1; O=$2; ROUNDS=$3
mkdir -p $O/spv
U8=$W/guest/shaders/mul_mat_vecq_u8.comp
glsl() { name=$1; shift; glslc -fshader-stage=compute --target-env=vulkan1.2 -O "$@" -o $O/spv/$name.spv || echo "COMPILE_FAIL $name"; }
# name:type:defines:rows_per_workgroup
VARIANTS="q5_t4:q5_0::32 q5_alu_t4:q5_0:-DQ50_ALU:32 q5_alu_sh28:q5_0:-DQ50_ALU,-DACT_SHARED,-DMAX_ACT_BLOCKS=28:32
q5_alu_sh32:q5_0:-DQ50_ALU,-DACT_SHARED,-DMAX_ACT_BLOCKS=32:32 q5_alu_sh64:q5_0:-DQ50_ALU,-DACT_SHARED,-DMAX_ACT_BLOCKS=64:32
q5_alu_t16:q5_0:-DQ50_ALU,-DTPR=16:8 q8_t4:q8_0::32 q8_sh32:q8_0:-DACT_SHARED,-DMAX_ACT_BLOCKS=32:32 q8_t16:q8_0:-DTPR=16:8"
for v in $VARIANTS; do
    IFS=: read name type defs rows <<EOF
$v
EOF
    upper=$(echo $type | tr a-z A-Z)
    glsl $name -DDATA_A_$upper $(echo $defs | tr ',' ' ') $U8
done
run() { $W/kernel_bench "$@" --io-memory device >> $O/results.jsonl 2>> $O/stderr.log || echo "RUN_FAIL $*" >> $O/stderr.log; }
for round in $(seq 1 $ROUNDS); do
    for shape in "q5_0 4864 896 256" "q5_0 896 896 512" "q5_0 128 896 1024" "q8_0 65536 896 16" "q8_0 20864 896 64" "q8_0 128 896 1024"; do
        set -- $shape; type=$1; m=$2; k=$3; reps=$4
        copies=8; [ $m -gt 16384 ] && copies=2
        for v in $VARIANTS; do
            IFS=: read name vtype defs rows <<EOF
$v
EOF
            [ "$vtype" = "$type" ] || continue
            [ -f $O/spv/$name.spv ] || continue
            run $O/spv/$name.spv $type $m $k $rows 128,$rows,1 --subgroup 128 --b-format q8_1x4 --copies $copies --reps $reps \
                --label "r$round/$name/$m"
        done
    done
done
echo "SWEEP_DONE $(wc -l < $O/results.jsonl) rows"
