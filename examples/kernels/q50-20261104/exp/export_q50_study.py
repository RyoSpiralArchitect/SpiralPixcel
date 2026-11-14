"""Export the r6/r7 alternating runs into SpiralPixcel/examples/inference/q50-study/.

usage: export_q50_study.py RUNS_TXT
RUNS_TXT lists the runs in execution order as "label run_id args..." (ab-r7.sh). The r7 build receipt
and checks come from export/runtime-r7/ and its greedy continuations from export/generation-r7/.
"""
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path.home() / "SpiralPixcel"
RUNS = REPO / "runs"
SP = Path(__file__).resolve().parent
OUT = REPO / "examples/inference/q50-study"
WRAPPER = ("systemd-run --user --collect -p MemoryHigh=1900M -p MemoryMax=2100M -p MemorySwapMax=2500M "
           "--working-directory=<guest run dir> sh -c '{ python3 experiment.py doctor && cp result/result.json "
           "result/doctor.json && python3 inference.py ARGS; } > result/build-run.log 2>&1'")
NOTE = ("Same guest job as `pixel-lab run inference`, run in a memory-limited systemd user unit so the Terminal VM "
        "stays below Android's 4 GiB MemoryLimiter; collected with `pixel-lab collect`.")
MEANS = ("decode_tg32_tokens_s", "steady_tg32_tokens_s", "prefill_pp64_tokens_s", "steady_pp64_tokens_s")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gzip_copy(src, dst):
    with src.open("rb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=dst.open("wb"), mtime=0) as packed:
        shutil.copyfileobj(raw, packed)


def copy_tree(src, dst):
    dst.mkdir(parents=True)
    for path in sorted(src.iterdir()):
        if path.is_file():
            (gzip_copy(path, dst / (path.name + ".gz")) if path.suffix == ".log" else shutil.copyfile(path, dst / path.name))


def export_run(label, run_id, args, order):
    src, dst = RUNS / run_id, OUT / "runs" / f"{label}-{run_id.rsplit('-', 1)[1]}"
    dst.mkdir(parents=True)
    refs = {}
    for path in sorted(src.iterdir()):
        if path.name.startswith("model-") and path.suffix == ".bin":
            digest = sha256(path)
            vector = OUT / "vectors" / f"{digest}.bin"
            if not vector.exists():
                shutil.copyfile(path, vector)
            refs[path.name] = {"file": f"../../vectors/{digest}.bin", "sha256": digest,
                               "bytes": path.stat().st_size, "format": "float32-le"}
        elif path.suffix == ".log":
            gzip_copy(path, dst / (path.name + ".gz"))
        elif path.is_file():
            shutil.copyfile(path, dst / path.name)
    (dst / "vector-refs.json").write_text(json.dumps(refs, indent=2) + "\n")
    (dst / "job.json").write_text(json.dumps({"run_id": run_id, "order": order, "guest_args": args,
                                              "wrapper": WRAPPER, "note": NOTE}, indent=2) + "\n")
    result = json.loads((src / "result.json").read_text())
    bench = json.loads((src / "bench.json").read_text())
    rows = {("pp64" if row["n_prompt"] else "tg32"): row["samples_ts"] for row in bench}
    assert result["status"] == "pass", (label, result.get("error"))
    check = result["model_check"]
    tg, pp = rows["tg32"], rows["pp64"]
    return {"label": label, "runtime_label": label.split("-")[0], "run_id": run_id, "runtime": result["runtime"],
            "status": result["status"], "decode_tg32_tokens_s": result["decode_tg32_tokens_s"], "tg32_samples": tg,
            "steady_tg32_tokens_s": sum(tg[1:]) / len(tg[1:]), "prefill_pp64_tokens_s": result["prefill_pp64_tokens_s"],
            "pp64_samples": pp, "steady_pp64_tokens_s": sum(pp[1:]) / len(pp[1:]), "vm": result.get("vm"),
            "model_sha256": result.get("model_sha256"),
            "libggml_vulkan_sha256": result.get("runtime_sha256", {}).get("libggml-vulkan.so"),
            "model_nmse": check["nmse"], "model_max_abs_error": check["max_abs_error"], "top1_equal": check["top1_equal"],
            "candidate_sha256": refs["model-candidate.bin"]["sha256"], "reference_sha256": refs["model-reference.bin"]["sha256"],
            "kernel_check_cases": result["kernel_check"]["cases"], "readback_cases": result["readback_check"]["cases"],
            "large_check_cases": result["large_kernel_check"]["cases"],
            "nodes_per_backend": result["schedule"]["nodes_per_backend"],
            "sampled_peak_rss_kib": result["sampled_peak_rss_kib"], "peak_device_allocated_mib": result["peak_device_allocated_mib"],
            "u8_mmvq_selected": result.get("u8_mmvq_selected"), "u8_mmvq_variants_selected": result.get("u8_mmvq_variants_selected"),
            "attention_kernel_selected": result.get("attention_kernel_selected")}


def main():
    runs = []
    for line in Path(sys.argv[1]).read_text().splitlines():
        label, run_id, *args = line.split()
        runs.append((label, run_id, args))
    readme = (OUT / "README.md").read_text() if (OUT / "README.md").exists() else None
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "vectors").mkdir(parents=True)
    entries = [export_run(label, run_id, args, order) for order, (label, run_id, args) in enumerate(runs, 1)]
    means = {}
    for key in MEANS:
        old = [e[key] for e in entries if e["runtime_label"] == "r6"]
        new = [e[key] for e in entries if e["runtime_label"] == "r7"]
        means[key] = {"r6": sum(old) / len(old), "r7": sum(new) / len(new)}
        means[key]["ratio"] = means[key]["r7"] / means[key]["r6"]
    copy_tree(SP / "export/runtime-r7", OUT / "runtime" / "r7")
    copy_tree(SP / "export/generation-r7", OUT / "generation" / "r7")
    summary = {"date_local": "2026-11-04", "timezone": "Asia/Tokyo", "format": "SpiralPixcel.q50-study.v1",
               "model_scope": "Qwen2.5-0.5B-Instruct Q4_K_M, one fixed 33-token prompt for the logit gate; short timings",
               "vm_boot_ids": sorted({e["vm"]["boot_id"] for e in entries if e["vm"]}),
               "runs": entries, "means": means}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if readme is not None:
        (OUT / "README.md").write_text(readme)
    inventory = {str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob("*"))
                 if p.is_file() and p.name != "sha256.json"}
    (OUT / "sha256.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"files": len(inventory), "boots": summary["vm_boot_ids"], "means": means}, indent=1))


if __name__ == "__main__":
    main()
