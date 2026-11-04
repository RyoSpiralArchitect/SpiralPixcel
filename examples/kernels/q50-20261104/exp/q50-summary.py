"""Summarize q50-sweep.sh results: median marginal time per variant and shape over rounds.

usage: q50-summary.py RESULTS_JSONL
"""
import json
import statistics
import sys
from collections import defaultdict

BYTES = {4864: 4864 * 28 * 22, 896: 896 * 28 * 22, 128: 128 * 28 * 22}
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
times, ok, nmse = defaultdict(list), defaultdict(list), defaultdict(list)
for r in rows:
    _, name, m = r["label"].split("/")
    key = (int(m), name)
    times[key].append(r["marginal_us"])
    if not name.startswith("roofline"):
        ok[key].append(bool(r.get("pass")))
        nmse[key].append(r.get("nmse"))
for m in (4864, 896, 128):
    keys = sorted((k for k in times if k[0] == m), key=lambda k: statistics.median(times[k]))
    base = statistics.median(times[(m, "base_t4")]) if (m, "base_t4") in times else None
    print(f"== {m} x 896 ({BYTES[m] / 1e6:.2f} MB)")
    for k in keys:
        med = statistics.median(times[k])
        spread = "/".join(f"{t:.1f}" for t in times[k])
        rel = f"{base / med:5.2f}x" if base else ""
        check = "" if k[1].startswith("roofline") else ("pass" if all(ok[k]) else "FAIL") + f" nmse={max(nmse[k]):.2e}"
        print(f"  {k[1]:14s} {med:8.2f} us  {BYTES[m] / med / 1e3:6.1f} GB/s  vs base {rel}  [{spread}]  {check}")
