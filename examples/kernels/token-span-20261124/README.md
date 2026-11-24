# Token-boundary GPU spans: r6 against r7

Pixel 11 Pro, standard Debian VM (boot `41119ca1-3818-4901-a08e-de3c859ff473`), PowerVR Vulkan, 2026-11-24 11:11–11:15 JST. The shape variants of the Q5_0/Q8_0 mat-vec (runtime r7) had cut per-token GPU time by 2.6 ms when every command buffer carried a timestamp at both ends ([`../q50-20261104/end-to-end/`](../q50-20261104/end-to-end/)). This study measures the same comparison with timestamps only at token boundaries, so the decode graph runs unsplit. [The write-up](../../../docs/matvec-variants.md) puts the result in context.

## Method

`guest/probes/vkshim.c` with `VKSHIM_TOKEN_TS=1` (source sha256 `277159b6…`, built in the VM with gcc 14.2.0 `-O2 -Wall -Wextra`, `vkshim.so` sha256 `99650058…`). The shim writes one GPU timestamp before the first submission to the waited queue after each wait and one before each `vkQueueWaitIdle` on that queue, each in a command buffer of its own; ggml's command buffers are untouched. A span covers everything the GPU does from the first submission to the end of the last, including any idle time between submissions.

With the completion wait (`GGML_VK_WAIT_IDLE=1`), ggml waits on the compute queue three times per decode step: two input uploads (two submissions each) and the graph (ten submissions; six on the first step). The summaries use the graph spans.

`exp/tokenspan.sh OUT N RT...` ran one `llama-bench -p 0 -n N -r 1 -t 4 -b 16 -ub 16 -fa 0 -ngl 99 -dev Vulkan0 --repack 1 --no-warmup` per runtime, with every decode switch on (`GGML_VK_PWR_U8_MMVQ=1 GGML_VK_PWR_ATTN=1 GGML_VK_PWR_BARRIER=1 GGML_VK_PREFER_DEVICE_LOCAL=1 GGML_VK_SPLIT_OUTPUT=1 GGML_VK_DISABLE_ASYNC=1 GGML_VK_WAIT_IDLE=1`), in a memory-limited user scope after dropping the VM's page cache. Before and after each run the Mac read the GPU's devfreq `trans_stat` and the thermal status over adb. The runtimes are those of [`../../inference/q50-study/`](../../inference/q50-study/): r6 without the variants, r7 with them. The model is the same Qwen2.5-0.5B Q4_K_M file.

## Before the GPU

The first version of this mode (not kept) resubmitted a single start command buffer while it was still pending. Its first run, on 2026-11-24 at 02:28 JST, hung the GPU, and Android's framework restarted 25 s later.

The present version:

- takes each span's command buffers from a ring of 16 pairs, reused only after `vkQueueWaitIdle` on the same queue has returned;
- marks each buffer busy from submission until then, and turns itself off rather than resubmit a busy one;
- creates its command pool for the waited queue's family, recorded from `vkGetDeviceQueue` and `vkGetDeviceQueue2`;
- stays off unless that family supports compute or graphics and has timestamp bits.

The PowerVR device has one queue family (graphics, compute, transfer; two queues; 64 timestamp bits; 26.0417 ns per tick).

Before running on the PowerVR GPU, the shim ran on lavapipe with r7 and 64 tokens (`rehearsal/`). The loader was limited to the CPU Vulkan driver (`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json`, `GGML_VK_VISIBLE_DEVICES=0`), so the PowerVR driver was not loaded.

- **First build** (`rehearsal1-lvp.*`): turned itself off (state −2, queue family unknown), because ggml retrieves its queues with `vkGetDeviceQueue2`.
- **Second build** (`lvp.*`), with that entry point recorded: 63 spans from 127 openings (the ring wrapped about eight times), none paused and none unread. Each GPU span equalled its host span, about 206 ms per step on the CPU driver.

## Results

| Run | Runtime | Graph GPU min (ms) | p10 | median | Wall per step, median (ms) | Steps/s | Mean MHz, non-460 time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ab1-r6-111320` | r6 | 25.617 | 25.958 | 35.966 | 51.794 | 17.83 | 1125 |
| `ab1-r7-111338` | r7 | 24.808 | 27.753 | 38.035 | 57.890 | 16.50 | 1120 |
| `ab1-r7-111355` | r7 | 24.787 | 26.202 | 37.388 | 55.516 | 17.05 | 1039 |
| `ab1-r6-111411` | r6 | 25.848 | 26.103 | 38.674 | 55.960 | 16.87 | 1134 |
| `ab1-r6-111430` | r6 | 25.778 | 27.203 | 37.238 | 55.044 | 17.14 | 1136 |
| `ab1-r7-111448` | r7 | 24.819 | 25.879 | 35.103 | 51.781 | 17.81 | 1106 |
| `ab1-r7-111504` | r7 | 24.782 | 26.551 | 38.085 | 56.104 | 17.11 | 1100 |
| `ab1-r6-111520` | r6 | 25.792 | 25.885 | 35.312 | 51.848 | 18.44 | 1194 |

Each run decoded 128 tokens; the first four graph spans are skipped. Wall per step is the host time from one graph's first submission to the next one's. All eight runs exited 0, with the shim active (state 1) and no span paused or unread. The thermal status was 0 before and after every run.

- **The floor.** The fastest step of each run (graph GPU minimum) is 25.62–25.85 ms on r6 and 24.78–24.82 ms on r7. r7 minus r6 per ABBA pair: −0.81, −1.06, −0.96 and −1.01 ms (median −0.98 ms).
  - At the top clock the variants save about 1.0 ms of GPU time per step (3.8%). That is close to the 1.2 ms the isolated kernel timings predicted, not the 2.6 ms of the per-command-buffer measurement. That measurement averaged over steps run at different clocks, with the graph split by its timestamps.
- **The medians compare nothing.** When checked during the series (11:14 and 11:15 JST), the phone was in its screensaver (Android's "Dreaming"; the device admin caps the screen timeout at 60 s) with Terminal behind it, and all eight runs show the same noise:
  - steps took a median 52–58 ms, of which a median 11.7–15.3 ms fell outside the graph span;
  - the GPU spent 37–72% of its non-460 MHz time at 1286 MHz;
  - the per-pair difference of median GPU spans ranges from −2.1 to +2.8 ms.
- **r7's clock was lower** in this session (mean over non-460 MHz time 1039–1120 MHz against 1125–1194 MHz on r6). The earlier [`trans_stat` profiles](../q50-20261104/end-to-end/clockprof/), with the screen on, had found the two within 15 MHz. Four runs each are too few to decide.
- **Screen on.** `smoke-r7-111127` (r7, 32 tokens) started before the screensaver. Step 1 is the first graph (six submissions, 84 ms of GPU time) and step 2 still took 30.9 ms. Steps 3 to 10 ran the graph in 24.8–25.0 ms of GPU time (one step 26.5 ms) and the whole step in 28.7–30.9 ms; 3.4–4.6 ms per step fell outside the graph. With about 86% of a step in the graph's GPU time, 1.0 ms less GPU time would make a step about 3.5% shorter at an unchanged clock. That is smaller than the spread of the earlier throughput comparisons (0.98–1.10 per pair).
- **llama-bench's rate.** The `avg_ts` that llama-bench reports includes about 1.6–1.7 s between the first input upload and the first graph (the first step's setup; no warm-up run), so it is not used here.

## Files

- `exp/tokenspan.sh`, `exp/tokenspan_summary.py` and `exp/tokenspan_compare.py`: the runs, the per-run summary (`python3 -I exp/tokenspan_summary.py SKIP gpu-runs/TAG...`) and the table above.
- `gpu-runs/smoke-r7-111127.*` and `gpu-runs/ab1-<runtime>-<HHMMSS>.*`, one set per run:
  - `.before`, `.after`: Unix time, `trans_stat` and the thermal lines;
  - `.run`: exit status, the shim's summary JSON and llama-bench's JSON;
  - `.spans`: one line per span with GPU ticks, submissions in the span, and host ns since load at the opening and after the wait;
  - `.err`: llama-bench's stderr without the MESA lines.
- `smoke-summary.jsonl` (2 graph spans skipped) and `ab1-summary.jsonl` (4 skipped).
- `rehearsal/`: the two lavapipe runs (shim summary, llama-bench output, spans).
- `sha256.json`: hashes of every file here except itself and this README.
