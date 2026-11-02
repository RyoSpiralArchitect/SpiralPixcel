"""Record llama.cpp graph placement separately from the timed benchmark."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


_SPLIT = re.compile(r"^## SPLIT #(\d+):\s*(\S+)\s+#")
_NODE = re.compile(r"^node #\s*(\d+)\s+\(\s*([^)]*?)\s*\):\s*(.*?)\s+\([^)]*\)\s+\[([^]]*)\].*?:\s*(.*)$")
_REJECTED = re.compile(
    r"done_getting_tensors: tensor '([^']+)' \(([^)]+)\).*?"
    r"cannot be used with preferred buffer type (\S+), using (\S+) instead"
)


def parse_schedule(raw):
    """Parse the final placement dump, retaining full split backend names."""
    graphs = []
    nodes = []
    backend = None
    split_index = None
    rejected = {}
    for line in raw.splitlines():
        rejected_match = _REJECTED.search(line)
        if rejected_match:
            tensor, tensor_type, preferred, selected = rejected_match.groups()
            rejected[(tensor, preferred, selected)] = {
                "tensor": tensor, "type": tensor_type,
                "preferred_buffer": preferred, "selected_buffer": selected, "log_line": line,
            }
        split_match = _SPLIT.match(line)
        if split_match:
            index, backend = split_match.groups()
            split_index = int(index)
            if split_index == 0 and nodes:
                graphs.append(nodes)
                nodes = []
            continue
        node_match = _NODE.match(line)
        if node_match and backend is not None:
            index, op, name, buffer_name, sources = node_match.groups()
            nodes.append({"index": int(index), "op": op.strip(), "name": name.strip(),
                          "backend": backend, "buffer_label": buffer_name.strip(),
                          "split": split_index, "sources": sources.strip()})
    if backend is not None and not nodes:
        raise ValueError("scheduler log ends with an empty graph placement dump")
    if nodes:
        graphs.append(nodes)
    if not graphs:
        raise ValueError("scheduler log contains no graph placement dump")
    final = graphs[-1]
    counts = {}
    for node in final:
        counts[node["backend"]] = counts.get(node["backend"], 0) + 1
    return {
        "placement_scope": "last graph placement dump in scheduler.log",
        "logged_graphs": len(graphs), "final_logged_node_count": len(final),
        "nodes_per_backend": counts,
        "cpu_nodes": [node for node in final if node["backend"] == "CPU"],
        "output_nodes": [node for node in final if "output" in node["name"] or "output.weight" in node["sources"]],
        "rejected_weights": list(rejected.values()),
    }


def _validate_probe(raw, gpu_layers):
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("scheduler probe did not produce exactly one benchmark row")
    row = rows[0]
    if not isinstance(row, dict):
        raise ValueError("scheduler probe row is not an object")
    if (row.get("n_prompt"), row.get("n_gen"), row.get("n_gpu_layers")) != (0, 1, gpu_layers):
        raise ValueError("scheduler probe benchmark parameters changed")
    if gpu_layers and row.get("devices") != "Vulkan0":
        raise ValueError("scheduler probe did not select Vulkan0")
    samples = row.get("samples_ts")
    if (not isinstance(samples, list) or len(samples) != 1
            or not isinstance(samples[0], (int, float)) or not math.isfinite(samples[0]) or samples[0] <= 0):
        raise ValueError("scheduler probe has no completed decode sample")


def collect(runtime, model, gpu_layers, repack, flash_attn, environment, output):
    """Save diagnostic logs and a placement receipt, or fail with the receipt retained."""
    runtime, model, output = Path(runtime), Path(model), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command = [str(runtime / "llama-bench"), "-m", str(model), "-p", "0", "-n", "1", "-r", "1",
               "-b", "16", "-ub", "16", "-t", "4", "-fa", flash_attn, "-ngl", str(gpu_layers),
               "-dev", "Vulkan0" if gpu_layers else "none", "--repack", str(repack),
               "--no-warmup", "-v", "-o", "json"]
    if not gpu_layers:
        command += ["--no-op-offload", "1"]
    env = environment.copy()
    env["GGML_SCHED_DEBUG"] = "2"
    env.pop("GGML_VK_PERF_LOGGER", None)
    env.pop("GGML_VK_PERF_LOGGER_CONCURRENT", None)
    report = {"format": "SpiralPixcel.schedule.v1", "completed": False, "command": command,
              "diagnostic_only": True, "environment_overrides": {"GGML_SCHED_DEBUG": "2"},
              "disabled_environment_flags": ["GGML_VK_PERF_LOGGER", "GGML_VK_PERF_LOGGER_CONCURRENT"]}
    log, bench = output / "scheduler.log", output / "scheduler-bench.json"
    try:
        with bench.open("wb") as stdout, log.open("wb") as stderr:
            run = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=180)
        report["exit_code"] = run.returncode
        if run.returncode:
            raise ValueError(f"scheduler probe exited with {run.returncode}")
        _validate_probe(bench.read_text(), gpu_layers)
        report.update(parse_schedule(log.read_text(errors="replace")))
        report["completed"] = True
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
    report["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (log, bench) if p.exists()}
    temporary = output / "schedule.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "schedule.json")
    if not report["completed"]:
        raise ValueError("scheduler report failed; see schedule.json and scheduler.log")
    return report
