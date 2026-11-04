#!/bin/zsh
# clockpair.sh OUT RT...: 480-token decodes on the given runtimes in order, each profiled with clockprofile.sh;
# prints the rate and the time-weighted GPU clock over the active intervals (less than half the time at 460 MHz).
SP=${0:A:h}
out=$1; shift
for rt in "$@"; do
    tag=$out-$rt-$(date +%H%M%S)
    $SP/clockprofile.sh $tag 25 "R=~/.local/share/SpiralPixcel/runtimes/llama-b11960-pvr-decode-$rt; M=~/.local/share/spiral-pixel/models/qwen2.5-0.5b-q4_k_m.gguf; sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; systemd-run --user --scope --quiet -p MemoryHigh=1900M -p MemoryMax=2100M -p MemorySwapMax=2500M env LD_LIBRARY_PATH=\$R GGML_VK_PWR_U8_MMVQ=1 GGML_VK_PWR_ATTN=1 GGML_VK_PWR_BARRIER=1 GGML_VK_PREFER_DEVICE_LOCAL=1 GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 timeout 200 \$R/llama-bench -m \$M -p 0 -n 480 -r 1 -t 4 -b 16 -ub 16 -fa 0 -ngl 99 -dev Vulkan0 --repack 1 --no-warmup -o json" > /dev/null
    /usr/bin/python3 -I - $tag $rt <<'PY'
import collections, json, re, sys
tag, rt = sys.argv[1], sys.argv[2]
rate = json.loads(re.search(r"\[.*\]", open(tag + ".stdout").read(), re.S).group(0))[0]["avg_ts"]
snaps = collections.defaultdict(dict)
for line in open(tag + ".snaps"):
    t, f, ms, tr = line.split()
    snaps[float(t)][int(f) // 1000000] = (int(ms), int(tr))
times = sorted(snaps)
ms_at, trans, active = collections.Counter(), 0, 0
for a, b in zip(times, times[1:]):
    d = {f: snaps[b][f][0] - snaps[a][f][0] for f in snaps[a]}
    total = sum(d.values())
    if total == 0 or d.get(460, 0) > total / 2:
        continue
    active += 1
    ms_at.update(d)
    trans += sum(snaps[b][f][1] - snaps[a][f][1] for f in snaps[a])
total = sum(ms_at.values())
mean = sum(f * ms for f, ms in ms_at.items()) / total
rec = {"runtime": rt, "tg480_tokens_s": round(rate, 3), "active_intervals": active, "active_ms": total,
       "mean_mhz": round(mean, 1), "share_1286": round(ms_at[1286] / total, 3), "share_460": round(ms_at[460] / total, 3),
       "transitions_per_s": round(trans / (total / 1000), 1), "ms_at_mhz": dict(sorted(ms_at.items()))}
open(tag.rsplit("-", 2)[0] + "-pairs.jsonl", "a").write(json.dumps(rec) + "\n")
print(json.dumps({k: v for k, v in rec.items() if k != "ms_at_mhz"}))
PY
done
