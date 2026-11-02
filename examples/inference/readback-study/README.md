# Readback and completion-wait study

Pixel 11 Pro, standard Debian VM, PowerVR Vulkan, 2026-11-03 JST. This directory retains the failed controls, corrected runs, full fixed-prompt logits, and short timings. [Usage and interpretation](../../../docs/readback.md) describe the opt-in runtime and its limits.

## Final comparison using the same runtime

All three configurations used the same executable/library hashes and the same Qwen2.5-0.5B Q4_K_M file. Each completed doctor, 27 transfer cases, 56 kernel cases, full CPU/GPU logit comparison, placement, and timing. Split configurations also completed the large-head basis check. Conditions: four CPU threads, batch/ubatch 16, context 512 for the model checker, Flash Attention off, repack 1, no warmup. Each timing row has three pp64 or tg32 samples; each configuration has one completed pipeline, not a sustained-load study.

| Directory | Completion mode / placement | tg32 tok/s | pp64 tok/s | Sampled peak process RSS KiB |
| --- | --- | ---: | ---: | ---: |
| `serial-patched-full-4cea6ebb` | Per-operation wait, GPU request 99, split head | 2.173966 | 17.006230 | 223840 |
| `queue-full-0edc77dd` | Queue wait at backend synchronization, GPU request 99, split head | 8.143779 | 39.694103 | 223284 |
| `queue-cpu-control-4de7d7f2` | Queue wait, GPU request 1, unsplit head falls back to CPU | 35.536816 | 68.754369 | 607312 |

Queue-wait decode was about 3.75 times the per-operation-wait control. The CPU-majority control was faster than both. This does not establish that splitting itself improves speed, that maximum supported model size increased, or that RSS and Vulkan allocations can be added together on this UMA device.

The queue and serial full-GPU candidate vectors are byte-identical. Against the CPU reference, the complete 151936-logit vector for one fixed 33-token prompt has NMSE 0.00224630465, maximum absolute error 0.83569467, and matching top1 token. The pre-set gate is NMSE <= 0.005 plus top1 equality. This is bounded numeric validation, not a broad answer-quality evaluation. `summary.json` verifies the same-runtime and vector identities; `provenance.json` binds the pinned source, ten patches, and modified source files.

## Failures retained

- `async-control.jsonl`: two of 27 small transfer cases fail, including a previous value in ordinary host memory and zeros in the requested Vulkan host allocation. `sync-first.jsonl` and `readback-sync-repeat-*` record the initial synchronous-transfer control.
- `sync-only-*`: synchronous transfers alone still produced wrong full-GPU logits. Separate FP32, no-fusion, and batch-1 controls did not repair them. A checker metadata `status: pass` means the run completed with finite, nonzero values; the separate CPU/GPU `comparison.json` determines numerical agreement.
- `sync-only-head-calibration`, `sync-only-control-a1`, `sync-only-head-b1`, `sync-only-control-a2`, and `sync-only-head-b2-failed`: the last head attempt failed one F32 matrix case at the 56-case gate (NMSE about 0.05054). It has no timing row. This incomplete comparison must not be summarized as a successful paired study or replaced by a later passing run.
- `serial-original-full` and `serial-full-fixed-prompt`: the earlier per-operation-wait diagnostic repaired the fixed-prompt result, at low speed. These use the preceding runtime; the final table instead compares identical patched binaries.
- `kernel-serialized-repeat-*` and `queue-idle-*`: three repetitions of the indicated kernel/transfer gates, before the final completed pipelines. The later queue-wait runs passed all indicated cases; attribution of the original failure to driver, backend, or VM remains unresolved.

The [earlier zero-output and GUI-recovery snapshot](../output-row-study/) remains separate and unchanged in its raw data. No intentional GPU hang was used to validate the session guard.

## Files

Each pipeline directory includes `result.json`, raw check JSONL, placement, benchmark JSON, source-session receipts, and losslessly compressed logs (`.log.gz`). `vector-refs.json` points into `vectors/`; filenames are SHA-256 digests, values are little-endian float32, and duplicate vectors are stored once. Full vector hashes also match the original `model-check.json` receipts. Standalone model controls retain their original metadata and independently calculated comparisons. The small doctor input/output and SPIR-V files are retained with the pipelines. `sha256.json` inventories every exported file except itself.

Session/job paths are reproducibility context from the guest, not credentials. Device identifiers, account dumps, private keys, and screenshots are excluded.
