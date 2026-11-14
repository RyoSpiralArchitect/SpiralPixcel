#!/bin/zsh
# hostcpu.sh RUNTIME_LABEL OUT: run a 480-token decode in the VM and sample, over adb, the CPU time of the
# Android-side crosvm thread that executes the guest's Vulkan commands (it carries the guest process name).
SP=${0:A:h}
rt=$1; out=$2
$SP/px -t 300 "R=~/.local/share/SpiralPixcel/runtimes/llama-b11960-pvr-decode-$rt; M=~/.local/share/spiral-pixel/models/qwen2.5-0.5b-q4_k_m.gguf; sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; systemd-run --user --scope --quiet -p MemoryHigh=1900M -p MemoryMax=2100M -p MemorySwapMax=2500M env LD_LIBRARY_PATH=\$R GGML_VK_PWR_U8_MMVQ=1 GGML_VK_PWR_ATTN=1 GGML_VK_PWR_BARRIER=1 GGML_VK_PREFER_DEVICE_LOCAL=1 GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 timeout 200 \$R/llama-bench -m \$M -p 0 -n 480 -r 1 -t 4 -b 16 -ub 16 -fa 0 -ngl 99 -dev Vulkan0 --repack 1 --no-warmup -o json" < /dev/null > $out.bench.json 2> $out.bench.err &
bench=$!
snap() { perl -e 'alarm 15; exec @ARGV' adb -d shell 'for p in $(pidof crosvm_debian); do for t in /proc/$p/task/*; do cat $t/stat; done; done' 2>/dev/null | awk '{ match($0, /\(.*\)/); name=substr($0, RSTART+1, RLENGTH-2); rest=substr($0, RSTART+RLENGTH+1); split(rest, f, " "); print name "\t" f[12] "\t" f[13] }'; }
sleep 9
date +%s.%N > $out.t0; snap > $out.snap0
sleep 8
date +%s.%N > $out.t1; snap > $out.snap1
wait $bench
python3 - $out <<'PY'
import json, sys
out = sys.argv[1]
def load(path):
    rows = {}
    for line in open(path):
        name, u, s = line.rstrip("\n").split("\t")
        rows.setdefault(name, [0, 0])
        rows[name][0] += int(u); rows[name][1] += int(s)
    return rows
a, b = load(out + ".snap0"), load(out + ".snap1")
dt = float(open(out + ".t1").read()) - float(open(out + ".t0").read())
rate = json.load(open(out + ".bench.json"))[0]["avg_ts"]
tok = rate * dt
thread = b.get("llama-bench", [0, 0]); thread0 = a.get("llama-bench", [0, 0])
du, ds = (thread[0] - thread0[0]) * 10.0, (thread[1] - thread0[1]) * 10.0  # ms (USER_HZ 100)
print(json.dumps({"interval_s": round(dt, 2), "tg480_tokens_s": rate, "tokens_in_interval": round(tok, 1),
                  "render_thread_user_ms_per_token": round(du / tok, 2), "render_thread_sys_ms_per_token": round(ds / tok, 2),
                  "render_thread_cpu_share": round((du + ds) / (dt * 1000), 3)}))
PY
