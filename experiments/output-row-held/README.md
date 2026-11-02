# Held readback proposal

`pinned-read-fallback.patch` is an **untested follow-up**, excluded from `build-inference-runtime.sh`. It applies after the published output-row patch stack but has not been compiled or run on the Pixel. It is not a confirmed fix for the zero logits or black GUI.

The proposal routes pinned async destinations and the context's tensor-get path through staging when `GGML_VK_SPLIT_OUTPUT=1`. The hypothesis is that the model's Vulkan_Host logits destination differs from the ordinary host vector used by the isolated checker. It does not change the synchronous UMA-direct source mapping path. A minimal transfer-only reproducer should come before another full-model run.

See [observations, recovery and adoption criteria](../../docs/large-output.md).
