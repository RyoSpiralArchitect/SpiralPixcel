"""Add the Q5_0 and Q8_0 shape variants of the unsigned mat-vec kernel to a llama.cpp tree at the u8 patch.

usage: q50_u8_edit.py TREE SHADER
"""
import re
import shutil
import sys
from pathlib import Path

tree, shader = Path(sys.argv[1]), Path(sys.argv[2])
vk = tree / "ggml/src/ggml-vulkan"
shutil.copyfile(shader, vk / "vulkan-shaders/mul_mat_vecq_u8.comp")


def edit(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, (path.name, old[:70], text.count(old))
    path.write_text(text.replace(old, new, 1))


edit(vk / "ggml-vulkan-types.h",
     "    std::atomic<uint32_t> pvr_u8_mmvq_logged_types {};  // bit per ggml_type, for the one-time log\n",
     "    std::atomic<uint32_t> pvr_u8_mmvq_logged_types {};  // bit per ggml_type, for the one-time log\n"
     "    std::atomic<uint32_t> pvr_u8_mmvq_logged_variants {};  // bit per type and shape variant\n")
edit(vk / "ggml-vulkan-types.h",
     "    vk_pipeline pipeline_mul_mat_vecq_u8_f32[GGML_TYPE_COUNT];\n",
     "    vk_pipeline pipeline_mul_mat_vecq_u8_f32[GGML_TYPE_COUNT];\n"
     "    // Shape variants: activations staged in shared memory for tall matrices, sixteen invocations\n"
     "    // per row for short ones.\n"
     "    vk_pipeline pipeline_mul_mat_vecq_u8_act_shared_f32[GGML_TYPE_COUNT];\n"
     "    vk_pipeline pipeline_mul_mat_vecq_u8_tpr16_f32[GGML_TYPE_COUNT];\n")

cpp = vk / "ggml-vulkan.cpp"
text = cpp.read_text()
q6k = re.search(r"^ *ggml_vk_create_pipeline\(device, device->pipeline_mul_mat_vecq_u8_f32\[GGML_TYPE_Q6_K\].*\n", text, re.M)
assert q6k, "Q6_K u8 pipeline line not found"
variants = (
    "        // Shape variants of the Q5_0 and Q8_0 kernels: 32 rows per workgroup with the shared\n"
    "        // activation stage, 8 with sixteen invocations per row.\n"
    + "".join(
        f"        ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vecq_u8_{v}_f32[GGML_TYPE_{t.upper()}], "
        f"\"mul_mat_vecq_u8_{t}_{v}_f32\", mul_mat_vecq_u8_{t}_{v}_f32_len, mul_mat_vecq_u8_{t}_{v}_f32_data, \"main\", "
        f"mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {{{rows}, 1, 1}}, {{128, {rows}, 1}}, 1, true, true, 128);\n"
        for t in ("q5_0", "q8_0") for v, rows in (("act_shared", 32), ("tpr16", 8))))
text = text[:q6k.end()] + variants + text[q6k.end():]
cpp.write_text(text)
edit(cpp,
     """        dmmv_u8 = ctx->device->pipeline_mul_mat_vecq_u8_f32[src0->type];
        quantize_y = dmmv_u8 != nullptr;
        // Checkers look for this line to confirm the kernel ran rather than a silent fallback.
        const uint32_t bit = 1u << (src0->type % 32);
        if (dmmv_u8 && !(ctx->device->pvr_u8_mmvq_logged_types.fetch_or(bit) & bit)) {
            GGML_LOG_INFO("ggml_vulkan: mul_mat_vecq_u8 selected for %s\\n", ggml_type_name(src0->type));
        }
""",
     """        dmmv_u8 = ctx->device->pipeline_mul_mat_vecq_u8_f32[src0->type];
        // Shape variants, measured on 896-column Q5_0 and Q8_0 rows: the shared activation stage
        // (at most 32 blocks) pays off from about 2048 rows, sixteen invocations per row below 256.
        const char * variant = nullptr;
        uint32_t variant_index = 0;
        if (ne01 >= 2048 && ne00 <= 32 * 32 && ctx->device->pipeline_mul_mat_vecq_u8_act_shared_f32[src0->type]) {
            dmmv_u8 = ctx->device->pipeline_mul_mat_vecq_u8_act_shared_f32[src0->type];
            variant = "act_shared";
        } else if (ne01 <= 256 && ctx->device->pipeline_mul_mat_vecq_u8_tpr16_f32[src0->type]) {
            dmmv_u8 = ctx->device->pipeline_mul_mat_vecq_u8_tpr16_f32[src0->type];
            variant = "tpr16";
            variant_index = 1;
        }
        quantize_y = dmmv_u8 != nullptr;
        // Checkers look for these lines to confirm the kernel ran rather than a silent fallback.
        const uint32_t bit = 1u << (src0->type % 32);
        if (dmmv_u8 && !(ctx->device->pvr_u8_mmvq_logged_types.fetch_or(bit) & bit)) {
            GGML_LOG_INFO("ggml_vulkan: mul_mat_vecq_u8 selected for %s\\n", ggml_type_name(src0->type));
        }
        const uint32_t variant_bit = 1u << ((2 * src0->type + variant_index) % 32);
        if (variant && !(ctx->device->pvr_u8_mmvq_logged_variants.fetch_or(variant_bit) & variant_bit)) {
            GGML_LOG_INFO("ggml_vulkan: mul_mat_vecq_u8 variant %s selected for %s\\n", variant, ggml_type_name(src0->type));
        }
""")
edit(vk / "vulkan-shaders/vulkan-shaders-gen.cpp",
     """            string_to_spv("mul_mat_vecq_u8_" + tname + "_f32", "mul_mat_vecq_u8.comp", {{data_a_key, "1"}});
""",
     """            string_to_spv("mul_mat_vecq_u8_" + tname + "_f32", "mul_mat_vecq_u8.comp", {{data_a_key, "1"}});
            if (tname == "q5_0" || tname == "q8_0") {
                string_to_spv("mul_mat_vecq_u8_" + tname + "_act_shared_f32", "mul_mat_vecq_u8.comp", {{data_a_key, "1"}, {"ACT_SHARED", "1"}});
                string_to_spv("mul_mat_vecq_u8_" + tname + "_tpr16_f32", "mul_mat_vecq_u8.comp", {{data_a_key, "1"}, {"TPR", "16"}});
            }
""")
print("u8 variants applied")
