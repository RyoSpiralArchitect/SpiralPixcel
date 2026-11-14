# submit-scan.sh RUNTIME OUT "N1 N2 ...": per-token Vulkan call costs (timing shim, no GPU timestamps) and
# 128-token decode throughput for each GGML_VK_MAX_NODES_PER_SUBMIT value, all decode switches on.
set -u
R=$1; O=$2; VALUES=$3
M=~/.local/share/spiral-pixel/models/qwen2.5-0.5b-q4_k_m.gguf
SHIM=~/.local/share/SpiralPixcel/kernel-study/host-overhead-20261104/vkshim.so
ENVS="GGML_VK_PWR_U8_MMVQ=1 GGML_VK_PWR_ATTN=1 GGML_VK_PWR_BARRIER=1 GGML_VK_PREFER_DEVICE_LOCAL=1 GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1"
mkdir -p $O
for v in $VALUES; do
    for n in 32 160; do
        sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
        env LD_LIBRARY_PATH=$R LD_PRELOAD=$SHIM VKSHIM_OUT=$O/shim-$v-$n.json $ENVS GGML_VK_MAX_NODES_PER_SUBMIT=$v \
            timeout 200 $R/llama-bench -m $M -p 0 -n $n -b 16 -ub 16 -t 4 -fa 0 -ngl 99 -dev Vulkan0 -r 1 --repack 1 --no-warmup -o json \
            > $O/shim-$v-$n.out 2> $O/shim-$v-$n.err
    done
    sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
    env LD_LIBRARY_PATH=$R $ENVS GGML_VK_MAX_NODES_PER_SUBMIT=$v \
        timeout 300 $R/llama-bench -m $M -p 0 -n 128 -r 3 -b 16 -ub 16 -t 4 -fa 0 -ngl 99 -dev Vulkan0 --repack 1 --no-warmup -o json \
        > $O/tg128-$v.json 2> $O/tg128-$v.err
    python3 - $O $v <<'PY'
import json, sys
O, v = sys.argv[1], sys.argv[2]
a, b = (json.load(open(f"{O}/shim-{v}-{n}.json")) for n in (32, 160))
def per_token(name, i):
    return (b["funcs"].get(name, [0, 0])[i] - a["funcs"].get(name, [0, 0])[i]) / 128
t = json.load(open(f"{O}/tg128-{v}.json"))[0]["samples_ts"]
rec = {"max_nodes_per_submit": int(v), "submits_per_token": per_token("QueueSubmit", 0),
       "submit_ms_per_token": per_token("QueueSubmit", 1) / 1e6, "waitidle_ms_per_token": per_token("QueueWaitIdle", 1) / 1e6,
       "fence_wait_ms_per_token": per_token("WaitForFences", 1) / 1e6, "dispatches_per_token": per_token("CmdDispatch", 0),
       "barriers_per_token": per_token("CmdPipelineBarrier", 0),
       "shim_wall_ms_per_token": (b["wall_ns"] - a["wall_ns"]) / 128 / 1e6, "tg128_samples": t,
       "tg128_steady": sum(t[1:]) / len(t[1:])}
open(f"{O}/scan.jsonl", "a").write(json.dumps(rec) + "\n")
print(json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in rec.items()}))
PY
done
