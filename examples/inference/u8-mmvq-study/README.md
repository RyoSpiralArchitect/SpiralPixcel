# Unsigned-dot mat-vec A/B

Pixel 11 Pro, standard Debian VM, PowerVR Vulkan, 2026-11-04 00:28–00:55 JST, one VM boot (`59b05043-d0ba-4add-8024-25975af6deff`). [Usage and interpretation](../../../docs/kernels.md) describe the opt-in kernel and its limits.

Every run completed doctor, 27 transfer cases, the kernel gate, the two large-head cases, the full CPU/GPU logit comparison, placement, and timing. Same Qwen2.5-0.5B Q4_K_M file (`74a4da8c…`), all-GPU request with split output head, batch/ubatch 16, four threads, Flash Attention off, repack 1, no warmup, three pp64 and tg32 samples. "Steady" averages samples 2 and 3; the first sample of each run carries one-time start-up such as pipeline creation.

## Same binary, kernel off and on (`same-binary-r3/`)

Runtime `runtimes/llama-b11960-u8-mmvq-r3` (patch `b6d4d7d7…`, receipt in `runtime/r3/`), with `--u8-mmvq` (`GGML_VK_PWR_U8_MMVQ=1`) or without it. Both arms passed the same 68-case kernel gate.

| Order | Directory | Kernel | tg32 tok/s | tg32 steady | pp64 tok/s | Logit NMSE | top1 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `off-1-6cb95a1b9449` | off | 8.103 | 8.26 | 39.66 | 0.002246 | equal |
| 2 | `on-1-6c91bc30eba8` | on | 19.206 | 20.44 | 43.14 | 0.002615 | equal |
| 3 | `on-2-35d2f7c594b0` | on | 18.263 | 19.70 | 42.94 | 0.002615 | equal |
| 4 | `off-2-9dc81c6f0210` | off | 8.035 | 8.20 | 39.85 | 0.002246 | equal |
| 5 | `off-3-bb6d3ddb5caa` | off | 8.030 | 8.24 | 39.71 | 0.002246 | equal |
| 6 | `on-3-2123b8ed250e` | on | 18.744 | 19.68 | 43.44 | 0.002615 | equal |

Mean tg32 8.056 → 18.738 tok/s (2.33), steady 8.235 → 19.941 (2.42); per pair 2.37, 2.27, 2.33. Placement was CPU 2 / Vulkan0 611 nodes in every run, and sampled RSS and logged Vulkan allocations did not change. The kernel-check and model-candidate logs of every "on" run record `mul_mat_vecq_u8 selected for` q4_K, q5_0 and q8_0; llama-bench drops library logs without `-v`, so `bench.log` has none.

The "off" candidate vector equals `../readback-study/queue-full-0edc77dd` byte for byte (`c0e7340f…`), and the CPU reference equals that study's (`433c8bc1…`). All "on" vectors are identical (`511616a0…`). The new path has NMSE 0.0026147 and maximum absolute error 0.966 against the CPU, inside the pre-set NMSE <= 0.005 and top1 gate. It quantizes activations to 8 bits per 32 values; the CPU also does so for Q5_0 and Q8_0 but uses 256-value Q8_K blocks for Q4_K. The logit gate runs the new kernel at one position (the 33rd token).

## Against the documented runtime (`runtime-r2-vs-current/`)

The documented queue-wait runtime `large-study/runtime-completion-wait` against `runtimes/llama-b11960-u8-mmvq-r2` (patch `df07510c…`, receipt in `runtime/r2/`). r3 adds only an instance-version check to r2, which does not change behaviour on this device. These runs predate the boot-id field; the boot's uptime covers them (see `summary.json`).

| Order | Directory | tg32 tok/s | tg32 steady | pp64 tok/s | Kernel gate cases |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `current-1-649c2bc06841` | 8.097 | 8.24 | 38.89 | 56 |
| 2 | `r2-1-418bf56124c5` | 18.630 | 19.72 | 43.22 | 68 |
| 3 | `r2-2-f4cca5ed2f93` | 19.364 | 21.00 | 41.62 | 68 |
| 4 | `current-2-e3d936a34742` | 7.784 | 7.97 | 36.99 | 56 |

## Rejected attempt (`rejected-r2-attempt/`)

The first r2 attempt also required a kernel-selection line in `bench.log`. llama-bench cannot print one without `-v`, so both u8 runs failed after timing (tg32 19.46 and 18.34 tok/s in their `bench.json`), while the two current-runtime runs passed. The requirement was removed and the set was rerun above. Only receipts are kept.

These are short timings on one prompt. They do not establish sustained performance, answer quality, other models, or longer contexts. The CPU-majority control in `../readback-study/` (35.5 tok/s) remains faster. An earlier revision (2026-11-03, patch `4ca787fb…`, before the review fixes) gave the same vectors and 7.84 → 18.98 tok/s in two alternating pairs; those runs are not exported.

## Other files

`generation/` holds 64-token greedy continuations of one prompt from the CPU and the current GPU path (2026-11-03) and the r2 runtime with the kernel on (2026-11-04); `commands.json` has the exact commands. The two GPU outputs are byte-identical, so the new kernel kept every argmax for 64 positions; the CPU diverges after the colon. `runtime/r2/` and `runtime/r3/` hold the build receipts (`spiralpixcel-kernel.json` with `patch_sha256`), the build-time checks including `kernel-check-u8.log`, and the build logs.

Each run directory has the receipts, raw check JSONL and logs as `.log.gz`, and `job.json` with the order, arguments and the memory-limited wrapper used instead of `pixel-lab session`. `vector-refs.json` points into `vectors/`, where filenames are SHA-256 digests of little-endian float32 logits stored once. `summary.json` collects the values above from the receipts. `sha256.json` inventories every exported file except itself.
