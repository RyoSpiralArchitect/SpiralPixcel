"""Summarize token-span runs: python3 -I tokenspan_summary.py SKIP TAG...

Each TAG names files from tokenspan.sh (TAG.run, TAG.before, TAG.after, TAG.spans). A span line is: GPU ticks,
submissions inside the span, host ns at its opening, host ns after the queue wait. Each decode step has three
spans: two input uploads (2 submissions each) and the graph (10 submissions; 6 on the first step). Graph spans
are those with at least 4 submissions; the first SKIP of them are left out. Per step:
  gpu      GPU time from the graph's first submission to its end (the work is not split by timestamps),
  host     host time from that submission to the return of vkQueueWaitIdle,
  wall     host time from one graph's opening to the next one's,
  outside  wall minus host: readback, sampling, inputs and their uploads until the next graph starts.
Prints one JSON line per run.
"""
import json
import re
import statistics
import sys


def trans_stat(path):
    """Unix time, ms at each GPU frequency (MHz), and the thermal status line from a saved snapshot."""
    lines = open(path).read().splitlines()
    ms, status = {}, None
    for line in lines[1:]:
        m = re.match(r"^\s*\*?\s*(\d+):.*\s(\d+)$", line)
        if m:
            ms[int(m.group(1)) // 1000000] = int(m.group(2))
        elif "Thermal Status" in line:
            status = int(line.split(":")[1])
    return float(lines[0]), ms, status


def stats(xs):
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=20)
    return {"min": round(xs[0], 3), "p10": round(q[1], 3), "p25": round(q[4], 3), "median": round(statistics.median(xs), 3),
            "mean": round(statistics.fmean(xs), 3), "p90": round(q[17], 3)}


def summarize(skip, tag):
    run = open(tag + ".run").read()
    summary = json.loads(next(line for line in run.splitlines() if line.startswith('{"wall_ns"')))
    bench = json.loads(re.search(r"\[\s*\{.*\}\s*\]", run, re.S).group(0))[0]
    period = summary["token"]["period_ns"]
    rows = [tuple(int(v) for v in line.split()) for line in open(tag + ".spans")]
    graphs = [r for r in rows if r[1] >= 4]
    g = graphs[skip:]
    gpu = [r[0] * period / 1e6 for r in g]
    host = [(r[3] - r[2]) / 1e6 for r in g]
    wall = [(b[2] - a[2]) / 1e6 for a, b in zip(g, g[1:])]
    outside = [w - h for w, h in zip(wall, host)]
    uploads = [r[0] * period / 1e6 for r in rows if r[1] < 4]
    t0, f0, therm0 = trans_stat(tag + ".before")
    t1, f1, therm1 = trans_stat(tag + ".after")
    clock = {f: f1[f] - f0.get(f, 0) for f in f1 if f1[f] - f0.get(f, 0) > 0}
    decode_s = (g[-1][3] - g[0][2]) / 1e9
    return {
        "tag": tag.rsplit("/", 1)[-1],
        "runtime": tag.rsplit("-", 2)[-2],
        "llama_bench_tokens_s": round(bench["avg_ts"], 3),
        "n_gen": bench["n_gen"],
        "token": summary["token"],
        "graph_spans": len(graphs),
        "graph_submissions": sorted(set(r[1] for r in g)),
        "steps_used": len(g),
        "steps_per_s": round((len(g) - 1) / (sum(wall) / 1000), 3),
        "gpu_ms": stats(gpu),
        "host_ms": stats(host),
        "wall_ms": stats(wall),
        "outside_ms": stats(outside),
        "upload_gpu_ms": stats(uploads),
        "gpu_sum_over_wall": round(sum(gpu[:-1]) / sum(wall), 4),
        "decode_s": round(decode_s, 3),
        "window_s": round(t1 - t0, 2),
        "clock_ms_at_mhz": dict(sorted(clock.items())),
        "thermal_status": [therm0, therm1],
    }


def main():
    skip = int(sys.argv[1])
    for tag in sys.argv[2:]:
        print(json.dumps(summarize(skip, tag)))


if __name__ == "__main__":
    main()
