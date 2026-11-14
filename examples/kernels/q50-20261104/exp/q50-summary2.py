"""Summarize q50-sweep4.sh results per weight type and shape (median over rounds).

usage: q50-summary2.py RESULTS_JSONL
"""
import json
import statistics
import sys
from collections import defaultdict

BLOCK = {"q5_0": 22, "q8_0": 34}
times, ok, nmse = defaultdict(list), defaultdict(list), defaultdict(list)
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    r = json.loads(line)
    _, name, m = r["label"].split("/")
    key = (r["type"], int(m), name)
    times[key].append(r["marginal_us"])
    ok[key].append(bool(r.get("pass")))
    nmse[key].append(r.get("nmse"))
for shape in sorted({k[:2] for k in times}, key=lambda s: (s[0], -s[1])):
    t, m = shape
    nbytes = m * 28 * BLOCK[t]
    keys = sorted((k for k in times if k[:2] == shape), key=lambda k: statistics.median(times[k]))
    base_key = (t, m, "q5_t4" if t == "q5_0" else "q8_t4")
    base = statistics.median(times[base_key]) if base_key in times else None
    print(f"== {t} {m} x 896 ({nbytes / 1e6:.2f} MB)")
    for k in keys:
        med = statistics.median(times[k])
        spread = "/".join(f"{x:.1f}" for x in times[k])
        rel = f"{base / med:5.2f}x" if base else ""
        print(f"  {k[2]:12s} {med:9.2f} us {nbytes / med / 1e3:6.1f} GB/s  {rel}  [{spread}]  "
              f"{'pass' if all(ok[k]) else 'FAIL'} nmse={max(nmse[k]):.2e}")
