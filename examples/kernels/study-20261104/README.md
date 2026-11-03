# Mat-vec kernel study

Output of `sh guest/kernel-study.sh` on the Pixel 11 Pro Debian VM, 2026-11-04 00:42 JST. `results.jsonl` has one JSON line per run of `guest/kernel_bench.cpp`, including the specialization constants it used; `provenance.json` records the source, shader, binary, result and `libggml-base` hashes, the source and runtime paths, and the count of failed kernel runs and probe errors (both 0); `stderr.log` is empty.

Each line times one shader on ggml-quantized random weights rotated across copies, after a warm-up, as the marginal cost between a run of `reps` dispatches and one of `5 * reps` (`marginal_us`). Dispatches are separated by a compute-to-compute memory barrier, as in ggml-vulkan, and only the ends of a run are timestamped: a timestamp between dispatches costs about 100 us on this GPU. `nmse` compares against ggml's own dequantization in double precision, using the quantized activations when `b_format` is `q8_1x4`; `nmse_vs_f32_activations` measures the activation quantization itself. Values that are not finite are written as `null`.

Label prefixes:

- `ggml/` — ggml-vulkan's own mat-vec shader, compiled with the defines and specialization constants it uses on this driver.
- `u8/` — `guest/shaders/mul_mat_vecq_u8.comp`.
- `roofline_uvec4/`, `roofline_uint/` — streaming reads of the same number of bytes; their `nmse` is meaningless.
- `float_dq1/`, `float_dq3/` — the floating-point Q5_0 comparison kernel (`guest/shaders/bench/mmv_q5_0_pairs.comp`).
- `empty/`, `alu/`, `rms/` — calibration probes for dispatch cost, ALU throughput (`alu_probe.comp`, OP 0–5) and RMS-norm reductions; their `nmse` is meaningless.

Small shapes (128 rows, and to a lesser degree 896 rows) are dominated by the fixed dispatch and barrier cost, so their ratios are not kernel comparisons. Timings shift with the GPU clock and temperature; compare lines from the same run.
