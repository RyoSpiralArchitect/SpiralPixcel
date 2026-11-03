// p021_test: run ggml's p021 attention-score shader and the PowerVR variant on the same random
// f16 K and f32 q, compare the outputs bit for bit, and time each one (marginal GPU time per
// dispatch, compute-only barrier between dispatches).
// usage: p021_test stock.spv pvr.spv [nrows] [ncols] [kv_heads] [gqa] [passes]
// Build: gcc -O2 p021_test.c -o p021_test -lvulkan -lm; the .spv files come from a ggml-vulkan build directory.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CK(x)                                                                                      \
    do {                                                                                           \
        VkResult r_ = (x);                                                                         \
        if (r_ != VK_SUCCESS) {                                                                    \
            fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_);                       \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static VkDevice dev;
static VkQueue queue;
static VkPhysicalDeviceMemoryProperties mp;
static double ts_ns;

static uint16_t f2h(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
    if (e < 113) return sign;  // flush tiny values (inputs are in [-1, 1] and not tiny)
    uint32_t h = sign | ((e - 112) << 10) | (m >> 13);
    if ((m >> 12) & 1) h++;    // round half up on the dropped bits (good enough for test data)
    return (uint16_t)h;
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; void * ptr; } buffer_t;

static buffer_t make_buffer(VkDeviceSize size) {
    buffer_t b;
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    CK(vkCreateBuffer(dev, &bci, NULL, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t t = 0;
    while (t < mp.memoryTypeCount && !((mr.memoryTypeBits >> t & 1) && (mp.memoryTypes[t].propertyFlags & want) == want)) t++;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, t };
    CK(vkAllocateMemory(dev, &mai, NULL, &b.mem));
    CK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    CK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.ptr));
    return b;
}

static VkShaderModule load_module(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t * code = malloc(n);
    if (fread(code, 1, n, f) != (size_t)n) exit(1);
    fclose(f);
    VkShaderModuleCreateInfo ci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)n, code };
    VkShaderModule m;
    CK(vkCreateShaderModule(dev, &ci, NULL, &m));
    free(code);
    return m;
}

static VkPipeline make_pipeline(VkShaderModule m, VkPipelineLayout layout, const uint32_t * spec, uint32_t nspec) {
    VkSpecializationMapEntry me[4];
    for (uint32_t i = 0; i < nspec; i++) me[i] = (VkSpecializationMapEntry){ i, 4 * i, 4 };
    VkSpecializationInfo si = { nspec, me, 4 * nspec, spec };
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, NULL, 128 };
    VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                       { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &rs,
                                         VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
                                         VK_SHADER_STAGE_COMPUTE_BIT, m, "main", &si },
                                       layout, VK_NULL_HANDLE, 0 };
    VkPipeline p;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
    return p;
}

static VkCommandBuffer cb;
static VkFence fence;
static VkQueryPool qp;

// GPU time of `reps` dispatches of one pipeline, each followed by a compute-only memory barrier.
static double run(VkPipeline p, VkPipelineLayout layout, VkDescriptorSet set, const uint32_t * pc, uint32_t gx,
                  uint32_t gy, uint32_t gz, int reps) {
    CK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
    CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, qp, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, pc);
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    for (int r = 0; r < reps; r++) {
        vkCmdDispatch(cb, gx, gy, gz);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo s = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL };
    CK(vkQueueSubmit(queue, 1, &s, fence));
    CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    CK(vkResetFences(dev, 1, &fence));
    uint64_t ts[2];
    CK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return (double)(ts[1] - ts[0]) * ts_ns / 1e3;
}

static int cmp_d(const void * a, const void * b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double marginal(VkPipeline p, VkPipelineLayout l, VkDescriptorSet s, const uint32_t * pc, uint32_t gx, uint32_t gy, uint32_t gz) {
    double v[7];
    for (int i = 0; i < 7; i++) v[i] = (run(p, l, s, pc, gx, gy, gz, 100) - run(p, l, s, pc, gx, gy, gz, 20)) / 80;
    qsort(v, 7, sizeof(double), cmp_d);
    return v[3];
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s stock.spv pvr.spv [nrows] [ncols] [kv_heads] [gqa] [passes]\n", argv[0]);
        return 2;
    }
    const uint32_t nrows = argc > 3 ? atoi(argv[3]) : 256, ncols = argc > 4 ? atoi(argv[4]) : 64;
    const uint32_t kvh = argc > 5 ? atoi(argv[5]) : 2, gqa = argc > 6 ? atoi(argv[6]) : 7;
    const uint32_t passes = argc > 7 ? atoi(argv[7]) : 4;
    const uint32_t qh = kvh * gqa;

    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "p021_test", 1, NULL, 0, VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai, 0, NULL, 0, NULL };
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t npd = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &npd, &pd);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    ts_ns = props.limits.timestampPeriod;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
    VkQueueFamilyProperties qfp[16];
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
    uint32_t qf = 0;
    while (qf < nqf && !(qfp[qf].queueFlags & VK_QUEUE_COMPUTE_BIT)) qf++;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio };
    VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13 };
    f12.shaderFloat16 = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12 };
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f11, 0, 1, &qci, 0, NULL, 0, NULL, NULL };
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qf, 0, &queue);

    buffer_t A = make_buffer((VkDeviceSize)nrows * kvh * ncols * 2);
    buffer_t B = make_buffer((VkDeviceSize)qh * ncols * 4);
    buffer_t D0 = make_buffer((VkDeviceSize)qh * nrows * 4), D1 = make_buffer((VkDeviceSize)qh * nrows * 4);
    buffer_t F = make_buffer(16);
    srand(1234);
    for (uint32_t i = 0; i < nrows * kvh * ncols; i++) ((uint16_t *)A.ptr)[i] = f2h((float)rand() / RAND_MAX * 2 - 1);
    for (uint32_t i = 0; i < qh * ncols; i++) ((float *)B.ptr)[i] = (float)rand() / RAND_MAX * 2 - 1;
    memset(D0.ptr, 0xff, qh * nrows * 4);
    memset(D1.ptr, 0xff, qh * nrows * 4);

    VkDescriptorSetLayoutBinding binds[5];
    for (int i = 0; i < 5; i++) binds[i] = (VkDescriptorSetLayoutBinding){ i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 5, binds };
    VkDescriptorSetLayout dsl;
    CK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 28 };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr };
    VkPipelineLayout layout;
    CK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 2, 1, &ps };
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSet sets[2];
    VkDescriptorSetLayout dsls[2] = { dsl, dsl };
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 2, dsls };
    CK(vkAllocateDescriptorSets(dev, &dsai, sets));
    for (int s = 0; s < 2; s++) {
        VkDescriptorBufferInfo infos[5] = { { A.buf, 0, VK_WHOLE_SIZE }, { B.buf, 0, VK_WHOLE_SIZE },
                                            { s ? D1.buf : D0.buf, 0, VK_WHOLE_SIZE }, { F.buf, 0, VK_WHOLE_SIZE },
                                            { F.buf, 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, sets[s], 0, 0, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, infos, NULL };
        vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
    }
    const uint32_t spec_stock[2] = { 128, gqa }, spec_pvr[3] = { 128, gqa, passes };
    VkPipeline stock = make_pipeline(load_module(argv[1]), layout, spec_stock, 2);
    VkPipeline pvr = make_pipeline(load_module(argv[2]), layout, spec_pvr, 3);

    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf };
    VkCommandPool pool;
    CK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
    CK(vkCreateFence(dev, &fci, NULL, &fence));
    VkQueryPoolCreateInfo qpci = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2, 0 };
    CK(vkCreateQueryPool(dev, &qpci, NULL, &qp));

    const uint32_t pc[7] = { ncols, nrows, kvh, qh, 0, 0, 0 };
    const uint32_t rows_per_wg = 128 / 8 * passes;
    run(stock, layout, sets[0], pc, 1, nrows, kvh, 1);
    run(pvr, layout, sets[1], pc, 1, (nrows + rows_per_wg - 1) / rows_per_wg, kvh, 1);

    const float * d0 = D0.ptr, * d1 = D1.ptr;
    uint32_t differ = 0, unwritten = 0;
    double max_abs = 0, max_ref = 0;
    for (uint32_t i = 0; i < qh * nrows; i++) {
        uint32_t a, b;
        memcpy(&a, &d0[i], 4);
        memcpy(&b, &d1[i], 4);
        if (b == 0xffffffffu) unwritten++;
        if (a != b) differ++;
        double e = fabs((double)d0[i] - (double)d1[i]);
        if (e > max_abs) max_abs = e;
        if (fabs(d0[i]) > max_ref) max_ref = fabs(d0[i]);
    }
    // Double-precision reference for both outputs.
    double err0 = 0, err1 = 0, ref2 = 0;
    for (uint32_t h = 0; h < qh; h++) {
        const uint32_t kv = h / gqa;
        for (uint32_t r = 0; r < nrows; r++) {
            double s = 0;
            for (uint32_t c = 0; c < ncols; c++) {
                uint16_t x = ((uint16_t *)A.ptr)[(r * kvh + kv) * ncols + c];
                uint32_t sign = (x & 0x8000) << 16, e = (x >> 10) & 0x1f, m = x & 0x3ff;
                uint32_t fbits = e ? sign | ((e + 112) << 23) | (m << 13) : sign;
                float xf;
                memcpy(&xf, &fbits, 4);
                s += (double)xf * ((float *)B.ptr)[h * ncols + c];
            }
            err0 += (d0[h * nrows + r] - s) * (d0[h * nrows + r] - s);
            err1 += (d1[h * nrows + r] - s) * (d1[h * nrows + r] - s);
            ref2 += s * s;
        }
    }
    const double t_stock = marginal(stock, layout, sets[0], pc, 1, nrows, kvh);
    const double t_pvr = marginal(pvr, layout, sets[1], pc, 1, (nrows + rows_per_wg - 1) / rows_per_wg, kvh);
    printf("{\"nrows\": %u, \"ncols\": %u, \"kv_heads\": %u, \"gqa\": %u, \"passes\": %u, \"outputs\": %u, "
           "\"bitwise_differ\": %u, \"unwritten\": %u, \"max_abs_diff\": %.3g, \"max_abs_value\": %.3g, "
           "\"nmse_stock_vs_f64\": %.3g, \"nmse_pvr_vs_f64\": %.3g, \"stock_us\": %.2f, \"pvr_us\": %.2f}\n",
           nrows, ncols, kvh, gqa, passes, qh * nrows, differ, unwritten, max_abs, max_ref, err0 / ref2, err1 / ref2,
           t_stock, t_pvr);
    return 0;
}
