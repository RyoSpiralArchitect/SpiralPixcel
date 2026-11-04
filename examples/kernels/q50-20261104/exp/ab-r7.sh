#!/bin/zsh
# ab-r7.sh: six full-gate runs alternating the r6 and r7 runtimes, both with the u8 kernel and the decode switches.
SP=${0:A:h}
RT=/home/droid/.local/share/SpiralPixcel/runtimes
COMMON=(--gpu-layers 99 --repetitions 3 --split-output --completion-wait queue --u8-mmvq --powervr-decode)
: > $SP/ab-r7-runs.txt
for label in r6-1 r7-1 r7-2 r6-2 r6-3 r7-3; do
  runtime=$RT/llama-b11960-pvr-decode-${label%%-*}
  out=$($SP/prun -- --runtime $runtime "${COMMON[@]}" 2>&1 | tee /dev/stderr | grep -o '20[0-9]*T[0-9]*Z-[0-9a-f]*' | head -1)
  echo "$label $out --runtime $runtime ${COMMON[*]}" >> $SP/ab-r7-runs.txt
done
cat $SP/ab-r7-runs.txt
