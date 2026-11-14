#!/bin/zsh
# freqrun.sh RUNTIME_LABEL OUT: a 480-token decode in the VM (all decode switches on, 4 threads) while the
# Mac samples the GPU clock over adb every ~0.25 s; prints the rate and the clock distribution during it.
SP=${0:A:h}
rt=$1; out=$2
( for i in $(seq 1 200); do f=$(perl -e 'alarm 5; exec @ARGV' adb -d shell cat /sys/class/devfreq/34f00000.gpu0/cur_freq 2>/dev/null | tr -d '\r'); echo "$(date +%s.%N) $f"; sleep 0.2; done > $out.freq ) &
sampler=$!
start=$(date +%s.%N)
$SP/px -t 300 "R=~/.local/share/SpiralPixcel/runtimes/llama-b11960-pvr-decode-$rt; M=~/.local/share/spiral-pixel/models/qwen2.5-0.5b-q4_k_m.gguf; sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; systemd-run --user --scope --quiet -p MemoryHigh=1900M -p MemoryMax=2100M -p MemorySwapMax=2500M env LD_LIBRARY_PATH=\$R GGML_VK_PWR_U8_MMVQ=1 GGML_VK_PWR_ATTN=1 GGML_VK_PWR_BARRIER=1 GGML_VK_PREFER_DEVICE_LOCAL=1 GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 timeout 200 \$R/llama-bench -m \$M -p 0 -n 480 -r 1 -t 4 -b 16 -ub 16 -fa 0 -ngl 99 -dev Vulkan0 --repack 1 --no-warmup -o json" < /dev/null > $out.bench.json 2> $out.bench.err
end=$(date +%s.%N)
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
/usr/bin/python3 -I - $out $start $end <<'PY'
import json, sys, collections
out, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rate = json.load(open(out + ".bench.json"))[0]["avg_ts"]
gen = 480 / rate
# the decode ends ~1 s before the command returns; keep samples from the generation window only
samples = [(float(t), int(f)) for t, f in (line.split() for line in open(out + ".freq") if len(line.split()) == 2 and line.split()[1].isdigit())]
window = [f for t, f in samples if end - 1.0 - gen <= t <= end - 1.0]
dist = collections.Counter(window)
mean = sum(window) / len(window) if window else 0
print(json.dumps({"tg480_tokens_s": round(rate, 2), "samples": len(window), "mean_mhz": round(mean / 1e6),
                  "mhz_counts": {str(k // 1000000): v for k, v in sorted(dist.items())}}))
PY
