# Q5_0/Q8_0 shape variants: r6 against r7

Pixel 11 Pro, standard Debian VM, PowerVR Vulkan, 2026-11-04 17:46–17:50 JST, one VM boot (`2014be07-86aa-4abd-aad4-e166eb7b99cd`). [The write-up](../../../docs/matvec-variants.md) explains the variants; [the kernel study](../../kernels/q50-20261104/) holds the isolated measurements and the GPU-time and CPU-thread measurements quoted below.

Runtimes `runtimes/llama-b11960-pvr-decode-r6` (u8 patch `9828dd27…`, decode patch `a17395ba…`) and `runtimes/llama-b11960-pvr-decode-r7` (u8 patch `75942386…` with the shape variants, decode patch `bf4316a6…`, whose changes are only context lines). r7's receipt and build-time checks are in `runtime/r7/`; r6's are in `../decode-study/runtime/r6/`. Apart from `libggml-vulkan.so` and `check-mmv` (which gained the 4864-row cases), every r7 library and tool is byte-identical to r6's once the build directory name is replaced and the build-id note removed.

Every run used `--u8-mmvq --powervr-decode` with the split output head and queue completion wait, and completed doctor, 27 transfer cases, the kernel gate (76 cases on r6; 102 on r7, which also requires each variant's selection line), the two large-head cases, the full CPU/GPU logit comparison, placement, and timing. Same Qwen2.5-0.5B Q4_K_M file (`74a4da8c…`), batch/ubatch 16, four threads, Flash Attention off, repack 1, no warmup, three pp64 and tg32 samples. "Steady" averages samples 2 and 3.

| Order | Directory | Runtime | tg32 tok/s | tg32 samples | tg32 steady | pp64 tok/s | Logit NMSE | top1 |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | `runs/r6-1-f6fb3221b8ac` | r6 | 28.208 | 24.10 / 29.75 / 30.77 | 30.26 | 45.87 | 0.002360 | equal |
| 2 | `runs/r7-1-6928bdaf8eda` | r7 | 28.186 | 23.54 / 28.28 / 32.74 | 30.51 | 46.07 | 0.002360 | equal |
| 3 | `runs/r7-2-a7d2ba66c9d6` | r7 | 27.625 | 23.05 / 31.05 / 28.78 | 29.91 | 44.80 | 0.002360 | equal |
| 4 | `runs/r6-2-1f4fb6c0c845` | r6 | 24.048 | 21.73 / 25.40 / 25.02 | 25.21 | 46.49 | 0.002360 | equal |
| 5 | `runs/r6-3-56e0da6168d9` | r6 | 28.455 | 25.08 / 30.94 / 29.35 | 30.14 | 45.57 | 0.002360 | equal |
| 6 | `runs/r7-3-cc448c727176` | r7 | 28.250 | 24.57 / 32.41 / 27.77 | 30.09 | 45.58 | 0.002360 | equal |

The means are tg32 26.90 → 28.02 tok/s and steady 28.54 → 30.17, but the difference comes from the fourth run, which was slower throughout. Without it, r6 averages 28.33 (steady 30.20) against r7's 28.02 (30.17). Ungated 128-token runs alternating the two runtimes (`../../kernels/q50-20261104/end-to-end/tg128/`) gave steady decode 28.38 against 28.55 tok/s. The variants did not change decode throughput beyond the noise of these runs. pp64 is unchanged (46.0 against 45.5).

What changed is GPU time. The command-buffer GPU timestamps (`../../kernels/q50-20261104/end-to-end/q50r*-summary.jsonl`) put r6 at 32.3, 32.9 and 33.9 ms per token and r7 at 30.8, 31.1 and 29.5, about 2.6 ms less. Six further pairs of 480-token decodes with the GPU clock sampled (`../../kernels/q50-20261104/end-to-end/clock/`) averaged 25.47 (r6) and 25.95 tok/s (r7), per pair 0.98 to 1.10, with mean clocks of 1036 and 1010 MHz; about a fifth of the clock samples read 460 MHz in every run. The guest's main thread spends most of each token in the final `vkQueueWaitIdle` (`../../kernels/q50-20261104/end-to-end/submit-scan/`), so it waits on the GPU rather than limiting it; an earlier reading of its 100% CPU as a submission bottleneck was wrong.

All six candidate vectors are byte-identical (`13197ed7…`, in `vectors/`) and equal to the r4 and r6 candidates in `../decode-study/`; the CPU reference is `433c8bc1…`. Placement was CPU 2 / Vulkan0 611 nodes in every run, logged Vulkan allocations 404.48 MiB, sampled RSS 223,240–226,408 KiB. The kernel-check and model-candidate logs of every run record the unsigned kernel for q4_K, q5_0, q6_K and q8_0 and the attention-score kernel; the r7 logs also record `act_shared` and `tpr16` for q5_0 and q8_0 (`u8_mmvq_variants_selected` in each `result.json`).

`generation/r7/` holds 64-token greedy continuations of one prompt on r7 with the decode switches off and on, and `commands.json`. Both are byte-identical to each other and to the r2, r4 and r6 continuations. `llama-completion` reported 20.59 tok/s (off) and 32.76 tok/s (on) for those 63 decode steps.

These are short timings on one prompt and the default context; sustained load, answer quality, longer contexts and other models are not evaluated.

Each run directory has the receipts, raw check JSONL and logs as `.log.gz`, and `job.json` with the order, arguments and the memory-limited wrapper used instead of `pixel-lab session`. `vector-refs.json` points into `vectors/`. `summary.json` collects the values above (its means include the fourth run). `sha256.json` inventories every exported file except itself.
