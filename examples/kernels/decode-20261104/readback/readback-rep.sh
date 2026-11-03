# readback-rep.sh RUNTIME OUT N: run check-readback N times per switch combination, count failed runs and cases.
set -u
R=$1; O=$2; N=$3; mkdir -p $O
BASE="GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1 GGML_VK_PWR_U8_MMVQ=1"
for i in $(seq 1 $N); do
  for arm in control barrier devlocal both; do
    case $arm in
      control) extra="";;
      barrier) extra="GGML_VK_PWR_BARRIER=1";;
      devlocal) extra="GGML_VK_PREFER_DEVICE_LOCAL=1";;
      both) extra="GGML_VK_PWR_BARRIER=1 GGML_VK_PREFER_DEVICE_LOCAL=1 GGML_VK_PWR_ATTN=1";;
    esac
    env LD_LIBRARY_PATH=$R $BASE $extra timeout 60 $R/check-readback $R > $O/$arm-$i.jsonl 2> /dev/null
    echo "$arm $i rc=$? $(tail -1 $O/$arm-$i.jsonl | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d.get("status"), d.get("failed"))')"
  done
done | awk '{print} {runs[$1]++; if ($4 != "pass") fail[$1]++} END {for (a in runs) printf "SUMMARY %s runs=%d failed_runs=%d\n", a, runs[a], fail[a]+0}'
