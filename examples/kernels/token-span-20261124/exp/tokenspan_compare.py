"""Compare runtimes from tokenspan_summary.py output: python3 -I tokenspan_compare.py SUMMARY.jsonl

Prints one row per run in order, then per runtime the median over its runs of each statistic, and the
r7 - r6 difference of neighbouring runs (ABBA order: runs 1-2, 3-4, ... are compared after sorting each pair).
"""
import json
import statistics
import sys

KEYS = [("gpu_ms", "min"), ("gpu_ms", "p10"), ("gpu_ms", "median"), ("gpu_ms", "mean"),
        ("wall_ms", "p10"), ("wall_ms", "median"), ("wall_ms", "mean"), ("outside_ms", "median")]


def label(k):
    return f"{k[0].replace('_ms', '')}.{k[1]}"


def main():
    runs = [json.loads(line) for line in open(sys.argv[1])]
    print("run                 rt   " + " ".join(f"{label(k):>12}" for k in KEYS) + "   steps/s  MHz(not 460)  1286 share  status")
    for r in runs:
        print(f"{r['tag']:<20}{r['runtime']:<5}" + " ".join(f"{r[k[0]][k[1]]:12.3f}" for k in KEYS)
              + f"   {r['steps_per_s']:7.2f}  {r['clock_mean_mhz_not_460']:12.1f}  {r['clock_share_1286_of_not_460']:10.3f}  {r['thermal_status']}")
    out = {}
    for rt in sorted(set(r["runtime"] for r in runs)):
        sel = [r for r in runs if r["runtime"] == rt]
        out[rt] = {label(k): round(statistics.median(r[k[0]][k[1]] for r in sel), 3) for k in KEYS}
        out[rt]["steps_per_s"] = round(statistics.median(r["steps_per_s"] for r in sel), 3)
        print(f"median of {len(sel)} {rt}: {json.dumps(out[rt])}")
    pairs = []
    for a, b in zip(runs[0::2], runs[1::2]):
        x = {r["runtime"]: r for r in (a, b)}
        if set(x) == {"r6", "r7"}:
            pairs.append({label(k): round(x["r7"][k[0]][k[1]] - x["r6"][k[0]][k[1]], 3) for k in KEYS})
    for i, p in enumerate(pairs, 1):
        print(f"pair {i} r7-r6: {json.dumps(p)}")
    if pairs:
        print("median r7-r6: " + json.dumps({k: round(statistics.median(p[k] for p in pairs), 3) for k in pairs[0]}))


if __name__ == "__main__":
    main()
