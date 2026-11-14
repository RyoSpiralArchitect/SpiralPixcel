#!/bin/zsh
# clockprofile.sh OUT SECONDS VM_COMMAND: run VM_COMMAND in the Pixel VM while the Mac reads the GPU's
# devfreq trans_stat (time in ms at each frequency, transition count) about once a second over adb.
SP=${0:A:h}
out=$1; secs=$2; shift 2
$SP/px -t 600 "$*" < /dev/null > $out.stdout 2> $out.stderr &
job=$!
: > $out.snaps
for i in $(seq 1 $secs); do
    t=$(date +%s.%N)
    perl -e 'alarm 5; exec @ARGV' adb -d shell cat /sys/class/devfreq/34f00000.gpu0/trans_stat 2>/dev/null | tr -d '\r' \
        | awk -v t=$t '/^[ *]+[0-9]+:/ { gsub(/\*/, ""); split($0, a, ":"); f=a[1]+0; n=split(a[2], c, " "); printf "%s %d %s", t, f, c[n]; s=0; for (j=1;j<n;j++) s+=c[j]; printf " %d\n", s }' >> $out.snaps
    sleep 0.7
done
wait $job
/usr/bin/python3 -I - $out <<'PY'
import sys, collections
out = sys.argv[1]
snaps = collections.defaultdict(dict)  # time -> {freq: (ms, transitions out)}
for line in open(out + ".snaps"):
    t, f, ms, tr = line.split()
    snaps[float(t)][int(f) // 1000000] = (int(ms), int(tr))
times = sorted(snaps)
print("interval  " + " ".join(f"{f:>5}" for f in sorted(snaps[times[0]])) + "   transitions  (ms at each MHz)")
for a, b in zip(times, times[1:]):
    d = {f: snaps[b][f][0] - snaps[a][f][0] for f in snaps[a]}
    tr = sum(snaps[b][f][1] - snaps[a][f][1] for f in snaps[a])
    print(f"{b - times[0]:6.1f}s  " + " ".join(f"{d[f]:5d}" for f in sorted(d)) + f"   {tr:5d}")
PY
