#!/bin/zsh
# tokenspan.sh OUT N RT...: one N-token llama-bench decode per runtime, in the order given, under the timing
# shim's token-span mode (VKSHIM_TOKEN_TS=1), with every decode switch on. Before and after each run the Mac
# reads the GPU's devfreq trans_stat and the thermal status over adb, then copies the run's span file and
# stderr from the VM. OUT is a local path prefix; files are OUT-RT-HHMMSS.{before,run,after,spans,err}.
# $PX is the SSH helper (default: px beside this script); the shim must already be built as $D/vkshim.so.
SP=${0:A:h}
PX=${PX:-$SP/px}
out=$1; n=$2; shift 2
D='~/.local/share/SpiralPixcel/kernel-study/token-span-v2-20261124'
snap() {
    perl -e 'alarm 8; exec @ARGV' adb -d shell 'date +%s.%N; cat /sys/class/devfreq/34f00000.gpu0/trans_stat; dumpsys thermalservice | grep -E "Thermal Status|VIRTUAL-SKIN"' 2>/dev/null | tr -d '\r'
}
for rt in "$@"; do
    base=${out:t}-$rt-$(date +%H%M%S)
    loc=${out:h}/$base
    snap > $loc.before
    $PX -t 200 "D=$D; R=~/.local/share/SpiralPixcel/runtimes/llama-b11960-pvr-decode-$rt; M=~/.local/share/spiral-pixel/models/qwen2.5-0.5b-q4_k_m.gguf; cd \$D; sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; systemd-run --user --scope --quiet -p MemoryHigh=1900M -p MemoryMax=2100M -p MemorySwapMax=2500M env LD_LIBRARY_PATH=\$R LD_PRELOAD=\$D/vkshim.so VKSHIM_TOKEN_TS=1 VKSHIM_OUT=\$D/$base.json VKSHIM_TOKEN_OUT=\$D/$base.spans GGML_VK_PWR_U8_MMVQ=1 GGML_VK_PWR_ATTN=1 GGML_VK_PWR_BARRIER=1 GGML_VK_PREFER_DEVICE_LOCAL=1 GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 timeout 120 \$R/llama-bench -m \$M -p 0 -n $n -r 1 -t 4 -b 16 -ub 16 -fa 0 -ngl 99 -dev Vulkan0 --repack 1 --no-warmup -o json > \$D/$base.stdout 2> \$D/$base.err; echo \"exit \$?\"; cat \$D/$base.json; cat \$D/$base.stdout" < /dev/null 2>&1 | grep -v '^MESA' > $loc.run
    snap > $loc.after
    $PX -t 60 "cat $D/$base.spans" < /dev/null > $loc.spans 2> /dev/null
    $PX -t 60 "grep -v '^MESA' $D/$base.err" < /dev/null > $loc.err 2> /dev/null
    print -r -- "$base $(grep -m1 '^exit' $loc.run) spans=$(wc -l < $loc.spans | tr -d ' ')"
    grep -q '^exit 0' $loc.run || { print -r -- "stopping: $base did not exit 0"; exit 1; }
done
