// dispatch_probe: GPU cost of one small compute dispatch inside a command buffer, depending on
// pipeline and descriptor-set changes, barriers, push-constant size and timestamps.
// usage: dispatch_probe tiny.spv
// Build: gcc -O2 dispatch_probe.c -o dispatch_probe -lvulkan; shader: glslc --target-env=vulkan1.3 shaders/bench/dispatch_tiny.comp
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CK(x)                                                                                      \
    do {                                                                                           \
        VkResult r_ = (x);                                                                         \
        if (r_ != VK_SUCCESS) {                                                                    \
            fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_);                       \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define NPIPE 8
#define NSET 8
#define WG 128

static VkDevice dev;
static VkQueue queue;
static VkPipeline pipes[NPIPE];
static VkPipelineLayout layout;
static VkDescriptorSet sets[NSET];
static VkCommandBuffer cb;
static VkFence fence;
static VkQueryPool qpool;
static double ts_ns;
static VkBuffer g_buf;

typedef struct {
    const char * label;
    int pipelines;   // distinct pipelines cycled per dispatch
    int sets;        // distinct descriptor sets cycled per dispatch
    int rebind;      // bind pipeline and set before every dispatch even if unchanged
    int barrier;     // after each dispatch: 0 none, 1 compute memory barrier, 2 ggml's barrier
                     // (compute|transfer stages, all four access bits), 3 execution only,
                     // 4 buffer barrier on the written slice, 5 synchronization2 memory barrier
    int pc_bytes;    // push-constant bytes per dispatch (4 or 128)
    int timestamps;  // timestamp after each dispatch
    uint32_t groups; // workgroups per dispatch
} cfg_t;

static double run(const cfg_t * c, int n) {
    CK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
    CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, qpool, 0, 2 + n);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    uint32_t pc[32] = { 0 };
    int bound_p = -1, bound_s = -1;
    for (int d = 0; d < n; d++) {
        int p = d % c->pipelines, s = d % c->sets;
        if (c->rebind || p != bound_p) vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[p]);
        if (c->rebind || s != bound_s || p != bound_p)
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[s], 0, NULL);
        bound_p = p;
        bound_s = s;
        pc[0] = 4;
        vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, c->pc_bytes, pc);
        vkCmdDispatch(cb, c->groups, 1, 1);
        switch (c->barrier) {
        case 1:
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            break;
        case 2: {
            const VkAccessFlags all = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            const VkPipelineStageFlags st = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkMemoryBarrier gb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, all, all };
            vkCmdPipelineBarrier(cb, st, st, 0, 1, &gb, 0, NULL, 0, NULL);
        } break;
        case 3:
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 0, NULL);
            break;
        case 4: {
            VkBufferMemoryBarrier bb = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT,
                                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED,
                                         VK_QUEUE_FAMILY_IGNORED, g_buf, (VkDeviceSize)s * (64 << 10), 64 << 10 };
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &bb, 0, NULL);
        } break;
        case 5: {
            VkMemoryBarrier2 m2 = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
            VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &m2, 0, NULL, 0, NULL };
            vkCmdPipelineBarrier2(cb, &di);
        } break;
        case 6: case 7: case 8: case 9: case 10: case 11: {
            const VkAccessFlags sh = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            const VkAccessFlags tr = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            const VkPipelineStageFlags cs = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags src = cs, dst = cs;
            VkAccessFlags sa = VK_ACCESS_SHADER_WRITE_BIT, da = sh;
            if (c->barrier == 6) { dst = cs | ts; da = sh | tr; }                                    // compute -> compute|transfer
            if (c->barrier == 7) { src = cs | ts; sa = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT; }  // compute|transfer -> compute
            if (c->barrier == 8) { sa = VK_ACCESS_SHADER_WRITE_BIT; da = VK_ACCESS_SHADER_READ_BIT; }  // write -> read only
            if (c->barrier == 9) { src = cs | ts; dst = cs | ts; sa = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT; da = sh | tr; }  // ggml minus read bits in src
            if (c->barrier == 10) { src = cs | ts; dst = cs | ts; sa = sh; da = sh; }               // ggml stages, shader access only
            if (c->barrier == 11) { sa = sh | tr; da = sh | tr; }                                     // ggml access, compute stages only
            VkMemoryBarrier gb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, sa, da };
            vkCmdPipelineBarrier(cb, src, dst, 0, 1, &gb, 0, NULL, 0, NULL);
        } break;
        case 12: {
            VkMemoryBarrier2 m2 = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                    VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT };
            VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &m2, 0, NULL, 0, NULL };
            vkCmdPipelineBarrier2(cb, &di);
        } break;
        case 13: {
            // synchronization2 with sync1-equivalent masks: compute|all-transfer stages, all four access kinds.
            VkMemoryBarrier2 m2 = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                    VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                                    VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT };
            VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &m2, 0, NULL, 0, NULL };
            vkCmdPipelineBarrier2(cb, &di);
        } break;
        default:
            break;
        }
        if (c->timestamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 2 + d);
    }
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL };
    CK(vkQueueSubmit(queue, 1, &si, fence));
    CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    CK(vkResetFences(dev, 1, &fence));
    uint64_t ts[2];
    CK(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return (double)(ts[1] - ts[0]) * ts_ns / 1e3;
}

static int cmp_d(const void * a, const void * b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

// Marginal GPU time per dispatch: (T(5n) - T(n)) / 4n, median over reps.
static void measure(const cfg_t * c) {
    const int n = 40, reps = 7;
    double v[7];
    for (int r = 0; r < reps; r++) v[r] = (run(c, 5 * n) - run(c, n)) / (4 * n);
    qsort(v, reps, sizeof(double), cmp_d);
    printf("{\"test\": \"%s\", \"pipelines\": %d, \"sets\": %d, \"rebind\": %d, \"barrier\": %d, \"push_bytes\": %d, "
           "\"timestamps\": %d, \"groups\": %u, \"us_per_dispatch\": %.2f, \"min\": %.2f, \"max\": %.2f}\n",
           c->label, c->pipelines, c->sets, c->rebind, c->barrier, c->pc_bytes, c->timestamps, c->groups, v[reps / 2], v[0],
           v[reps - 1]);
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s tiny.spv\n", argv[0]);
        return 2;
    }
    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "dispatch_probe", 1, NULL, 0, VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai, 0, NULL, 0, NULL };
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t npd = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &npd, &pd);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    ts_ns = props.limits.timestampPeriod;
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
    VkQueueFamilyProperties qfp[16];
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
    uint32_t qf = 0;
    while (qf < nqf && !(qfp[qf].queueFlags & VK_QUEUE_COMPUTE_BIT)) qf++;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio };
    VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    f13.synchronization2 = VK_TRUE;
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f13, 0, 1, &qci, 0, NULL, 0, NULL, NULL };
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qf, 0, &queue);

    // One device-local buffer; each descriptor set sees its own 64 KiB slice.
    const VkDeviceSize slice = 64 << 10;
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, slice * NSET, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    VkBuffer buf;
    CK(vkCreateBuffer(dev, &bci, NULL, &buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = 0;
    while (mt < mp.memoryTypeCount && !((mr.memoryTypeBits >> mt & 1) && mp.memoryTypes[mt].propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        mt++;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mt };
    VkDeviceMemory mem;
    CK(vkAllocateMemory(dev, &mai, NULL, &mem));
    CK(vkBindBufferMemory(dev, buf, mem, 0));
    g_buf = buf;

    FILE * f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t * code = malloc(n);
    if (fread(code, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)n, code };
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkDescriptorSetLayoutBinding b0 = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b0 };
    VkDescriptorSetLayout dsl;
    CK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 128 };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr };
    CK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
    // Distinct pipelines: the specialization constant changes the shader's arithmetic.
    for (int p = 0; p < NPIPE; p++) {
        uint32_t k = 3 + 2 * p;
        VkSpecializationMapEntry me = { 0, 0, 4 };
        VkSpecializationInfo spi = { 1, &me, 4, &k };
        VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                             { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", &spi },
                                             layout, VK_NULL_HANDLE, 0 };
        CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipes[p]));
    }
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NSET };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, NSET, 1, &ps };
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    for (int s = 0; s < NSET; s++) {
        VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl };
        CK(vkAllocateDescriptorSets(dev, &dsai, &sets[s]));
        VkDescriptorBufferInfo dbi = { buf, slice * s, slice };
        VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, sets[s], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi, NULL };
        vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
    }
    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf };
    VkCommandPool pool;
    CK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
    CK(vkCreateFence(dev, &fci, NULL, &fence));
    VkQueryPoolCreateInfo qpci = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2 + 5 * 40, 0 };
    CK(vkCreateQueryPool(dev, &qpci, NULL, &qpool));

    cfg_t warm = { "warmup", 1, 1, 0, 1, 4, 0, 64 };
    for (int i = 0; i < 30; i++) run(&warm, 200);

    cfg_t cfgs[] = {
        { "mem_compute", 1, 1, 0, 1, 4, 0, 1 },
        { "ggml_barrier", 1, 1, 0, 2, 4, 0, 1 },
        { "c_to_ct", 1, 1, 0, 6, 4, 0, 1 },
        { "ct_to_c", 1, 1, 0, 7, 4, 0, 1 },
        { "write_to_read", 1, 1, 0, 8, 4, 0, 1 },
        { "ggml_no_src_reads", 1, 1, 0, 9, 4, 0, 1 },
        { "ggml_stages_shader_access", 1, 1, 0, 10, 4, 0, 1 },
        { "compute_stages_all_access", 1, 1, 0, 11, 4, 0, 1 },
        { "sync2_compute", 1, 1, 0, 5, 4, 0, 1 },
        { "sync2_compute_copy_clear", 1, 1, 0, 12, 4, 0, 1 },
        { "sync2_all_transfer", 1, 1, 0, 13, 4, 0, 1 },
        { "ggml_barrier", 1, 1, 0, 2, 4, 0, 1 },
        { "sync2_compute", 1, 1, 0, 5, 4, 0, 1 },
    };
    for (size_t i = 0; i < sizeof(cfgs) / sizeof(cfgs[0]); i++) measure(&cfgs[i]);
    return 0;
}
