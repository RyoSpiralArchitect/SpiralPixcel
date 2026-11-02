# Readback proposals and diagnostic controls

`pinned-read-fallback.patch` is an **untested follow-up**, excluded from `build-inference-runtime.sh`. It applies after the published output-row patch stack but has not been compiled or run on the Pixel. It is not a confirmed fix for the zero logits or black GUI.

The proposal routes pinned async destinations and the context's tensor-get path through staging when `GGML_VK_SPLIT_OUTPUT=1`. The original hypothesis was that the model's Vulkan_Host logits destination differed from the ordinary host vector used by the isolated checker. It does not change the synchronous UMA-direct source mapping path. The later transfer-only reproducer also failed for an ordinary host destination, so this proposal was not selected.

`batch1-diagnostic.patch` changes only the fixed-prompt checker to batch/ubatch 1. It was compiled and run as a separate control with synchronous transfers, before the completion-wait repair; the full GPU logits still failed the CPU comparison. Its output and comparison are retained in [the readback study](../../examples/inference/readback-study/sync-only-full-batch1/). It is excluded from the runtime builder and does not change the published checker's batch/ubatch 16.

The selected, compiled repair is the opt-in [completion-wait patch](../../patches/llama-b11960-completion-wait.patch). [Current conditions and bounded results](../../docs/readback.md) describe it.

See [observations, recovery and adoption criteria](../../docs/large-output.md).
