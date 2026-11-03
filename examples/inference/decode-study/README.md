# PowerVR decode-path A/B

Pixel 11 Pro, standard Debian VM, PowerVR Vulkan, 2026-11-04 JST, one VM boot (`59b05043-d0ba-4add-8024-25975af6deff`). [Usage and interpretation](../../../docs/decode.md) describe the opt-in switches and their limits; [the kernel study](../../kernels/decode-20261104/) holds the probes that located the costs and the repeated transfer checks behind the readback fix.

Every run completed doctor, 27 transfer cases, the 76-case kernel gate, the two large-head cases, the full CPU/GPU logit comparison, placement, and timing. Same Qwen2.5-0.5B Q4_K_M file (`74a4da8c…`), all-GPU request with split output head, queue completion wait, batch/ubatch 16, four threads, Flash Attention off, repack 1, no warmup, three pp64 and tg32 samples. "Steady" averages samples 2 and 3; the first sample of each run carries one-time start-up such as pipeline creation.

## Final runtime, decode switches off and on (`same-binary-r6/`)

Runtime `runtimes/llama-b11960-pvr-decode-r6` (receipt in `runtime/r6/`), built with `--output-row-split --powervr-u8-mmvq --powervr-decode` from this repository's patches (decode patch `a17395ba…`, which includes the pinned-destination invalidate). Every run used `--u8-mmvq`; the decode runs added `--powervr-decode` (`GGML_VK_PWR_ATTN=1`, `GGML_VK_PWR_BARRIER=1`, `GGML_VK_PREFER_DEVICE_LOCAL=1`). 06:57–07:02 JST.

| Order | Directory | Decode switches | tg32 tok/s | tg32 samples | tg32 steady | pp64 tok/s | Logit NMSE | top1 |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | `u8-1-b5b31a321d67` | off | 18.477 | 16.01 / 19.78 / 19.65 | 19.71 | 42.96 | 0.002360 | equal |
| 2 | `decode-1-34e56d14889d` | on | 28.834 | 25.23 / 30.92 / 30.35 | 30.63 | 45.55 | 0.002360 | equal |
| 3 | `decode-2-d4de74553161` | on | 28.183 | 23.29 / 31.96 / 29.29 | 30.63 | 45.54 | 0.002360 | equal |
| 4 | `u8-2-c906646b5ee1` | off | 19.385 | 17.32 / 20.65 / 20.18 | 20.42 | 42.98 | 0.002360 | equal |
| 5 | `u8-3-f4c111af5d27` | off | 18.815 | 16.89 / 19.32 / 20.24 | 19.78 | 43.86 | 0.002360 | equal |
| 6 | `decode-3-87f8378e15c1` | on | 28.671 | 23.74 / 32.02 / 30.25 | 31.14 | 45.10 | 0.002360 | equal |

Mean tg32 18.89 → 28.56 tok/s (1.51), steady 19.97 → 30.80 (1.54); per pair 1.56, 1.45, 1.52. pp64 43.3 → 45.4 (1.05).

All six candidate vectors are byte-identical (`13197ed7…`, in `vectors/`), as is the CPU reference (`433c8bc1…`). The decode switches left the final logits unchanged. The new attention-score kernel sums in a different order, which changes the lowest bits of about 70% of its outputs in isolation. The differences vanish where the attention output is quantized to 8 bits for the output projection. The vector differs from the r3 candidate (`511616a0…`) because the unsigned kernel now also covers Q6_K, and the NMSE against the CPU fell from 0.002615 to 0.002360. Placement was CPU 2 / Vulkan0 611 nodes in every run. Logged Vulkan allocations (404.48 MiB) and sampled RSS (223,228–223,252 KiB) did not change with the switches.

The kernel-check and model-candidate logs of every run record `mul_mat_vecq_u8 selected for` q4_K, q5_0, q6_K and q8_0. Those of the decode runs also record `mul_mat_vec_p021_pvr selected`. llama-bench drops library logs without `-v`, so `bench.log` has none.

r6 differs from the r5 build only by the decode patch's new hunk. Apart from `libggml-vulkan.so`, every library and tool of r6 is byte-identical to r5's once the build directory name (`r6-src`, compiled in through `__FILE__`) is replaced and the `.note.gnu.build-id` section is removed (`runtime/r6/r5r6-compare.json`, written by `runtime/r6/r5r6-compare.sh`).

## Before the readback fix (`same-binary-r4/`)

Runtime `runtimes/llama-b11960-pvr-decode-r4` (receipt in `runtime/r4/`), decode patch `0925ed44…`: the same switches without the pinned-destination invalidate, built before the review fixes to the checkers (its 76-case gate had one GQA-8 case with a single KV head, which did not reach the new attention shader). 04:28–04:34 JST, same boot and commands.

| Order | Directory | Decode switches | tg32 tok/s | tg32 samples | tg32 steady | pp64 tok/s | Logit NMSE | top1 |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | `u8-1-dca125edf2c1` | off | 19.075 | 17.33 / 20.21 / 19.68 | 19.95 | 41.70 | 0.002360 | equal |
| 2 | `decode-1-293363434a94` | on | 28.521 | 24.40 / 31.42 / 29.75 | 30.58 | 42.60 | 0.002360 | equal |
| 3 | `decode-2-3cdb199d1aa1` | on | 28.853 | 24.21 / 30.38 / 31.97 | 31.17 | 45.85 | 0.002360 | equal |
| 4 | `u8-2-73c601d8f581` | off | 18.062 | 15.55 / 19.43 / 19.21 | 19.32 | 41.13 | 0.002360 | equal |
| 5 | `u8-3-9a1427912659` | off | 18.959 | 16.90 / 20.27 / 19.70 | 19.99 | 42.90 | 0.002360 | equal |
| 6 | `decode-3-543f5fec7b56` | on | 24.783 | 23.19 / 26.18 / 24.98 | 25.58 | 44.76 | 0.002360 | equal |

Mean tg32 18.70 → 27.39 tok/s (1.46), steady 19.75 → 29.11 (1.47); per pair 1.50, 1.60, 1.31. pp64 41.9 → 44.4 (1.06). The third decode run was slower throughout, which these short runs cannot separate from thermal or background variation. All six runs passed every gate and produced the same candidate vector as r6. Sampled RSS was 223,296–226,412 KiB.

The r5 rebuild (r4's patches, with the review fixes to the checkers and scripts) then failed its transfer gate in one decode run of the same A/B: two 64-byte reads into pinned host memory came back still holding the zeros written before them. Repeating the transfer check showed the failure only with device-local buffers, and the fix in r6 removed it; the failed run and all repetitions are in [`../../kernels/decode-20261104/readback/`](../../kernels/decode-20261104/readback/). With host-visible buffers the checker's writes and reads are CPU copies that never involve the GPU, so those arms say nothing about reads of GPU-written host-visible memory. The other five r5 runs passed and are not exported here, since r6 supersedes them. r4's transfer gates passing does not show that its reads were unaffected: the failure was intermittent.

These are short timings on one prompt and the default context. They do not establish sustained performance, answer quality, other models, or behaviour at longer contexts. A separate, ungated `llama-bench` run with a 1024-token context on r4 is in [`../../kernels/decode-20261104/depth/`](../../kernels/decode-20261104/depth/). The CPU-majority control in `../readback-study/` (35.5 tok/s) remains faster.

## Other files

`generation/r6/` and `generation/r4/` hold 64-token greedy continuations of one prompt with the decode switches off and on, and `commands.json` with the exact commands. All four are byte-identical to each other and to the r2 continuation in `../u8-mmvq-study/generation/gpu-u8.txt`, so the switches and the Q6_K kernel kept every argmax for 64 positions. `llama-completion` reported 19.93 tok/s (off) and 31.23 tok/s (on) on r6, and 19.27 and 27.58 tok/s on r4, for those 63 decode steps.

`runtime/r6/` and `runtime/r4/` hold the build receipts (`spiralpixcel-kernel.json` with `patch_sha256`, `powervr_u8_mmvq_types` and `powervr_decode`) and the build-time checks. The plain, u8 and decode runs of the kernel gate all pass 76 of 76. The transfer and large-output checks were repeated with every switch on (`*-decode.jsonl`; r6 keeps their logs as well). The build logs are included.

Each run directory has the receipts, raw check JSONL and logs as `.log.gz`, and `job.json` with the order, arguments and the memory-limited wrapper used instead of `pixel-lab session`. `vector-refs.json` points into `vectors/`, where filenames are SHA-256 digests of little-endian float32 logits stored once. `summary.json` collects the values above from the receipts, per set. `sha256.json` inventories every exported file except itself.
