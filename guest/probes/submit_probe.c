// submit_probe: per-submission latency, pipelining and fence behaviour of one Vulkan compute queue.
// usage: submit_probe spin.spv [iters_heavy]
// Build: gcc -O2 submit_probe.c -o submit_probe -lvulkan; shader: glslc --target-env=vulkan1.3 shaders/bench/submit_spin.comp
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

#define NCB 32
#define GROUPS 64

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static VkDevice dev;
static VkQueue queue;
static VkPipeline pipe;
static VkPipelineLayout layout;
static VkDescriptorSet dset;
static VkCommandBuffer cbs[NCB];
static VkFence fences[NCB];
static VkQueryPool qpool;
static double ts_period_ns;

static void record(int i, uint32_t dispatches, uint32_t iters) {
    VkCommandBuffer cb = cbs[i];
    CK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
    CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, qpool, 2 * i, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 2 * i);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, NULL);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &iters);
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    for (uint32_t d = 0; d < dispatches; d++) {
        vkCmdDispatch(cb, GROUPS, 1, 1);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             NULL, 0, NULL);
    }
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 2 * i + 1);
    CK(vkEndCommandBuffer(cb));
}

static double submit(int i, VkFence f) {
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cbs[i], 0, NULL };
    double t0 = now_us();
    CK(vkQueueSubmit(queue, 1, &si, f));
    return now_us() - t0;
}

static double wait_fence(VkFence f) {
    double t0 = now_us();
    CK(vkWaitForFences(dev, 1, &f, VK_TRUE, UINT64_MAX));
    double dt = now_us() - t0;
    CK(vkResetFences(dev, 1, &f));
    return dt;
}

// GPU start/end (us, relative to the first cb's start) for cbs [0, n).
static void gpu_times(int n, double * start, double * end) {
    uint64_t ts[2 * NCB];
    CK(vkGetQueryPoolResults(dev, qpool, 0, 2 * n, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    for (int i = 0; i < n; i++) {
        start[i] = (double)(ts[2 * i] - ts[0]) * ts_period_ns / 1e3;
        end[i] = (double)(ts[2 * i + 1] - ts[0]) * ts_period_ns / 1e3;
    }
}

static int cmp_d(const void * a, const void * b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double median(double * v, int n) {
    qsort(v, n, sizeof(double), cmp_d);
    return v[n / 2];
}

// K submits back to back, fence on the last; one JSON line with host and GPU timings.
static int quiet;
static void chain(const char * label, int K, uint32_t dispatches, uint32_t iters, int reps) {
    double total[16], sub_sum[16], gpu_busy[16], gpu_gap[16], gpu_span[16];
    for (int r = 0; r < reps; r++) {
        for (int k = 0; k < K; k++) record(k, dispatches, iters);
        double t0 = now_us(), s = 0;
        for (int k = 0; k < K; k++) s += submit(k, k == K - 1 ? fences[0] : VK_NULL_HANDLE);
        wait_fence(fences[0]);
        total[r] = now_us() - t0;
        sub_sum[r] = s;
        double st[NCB], en[NCB];
        gpu_times(K, st, en);
        double busy = 0, gap = 0;
        for (int k = 0; k < K; k++) {
            busy += en[k] - st[k];
            if (k) gap += st[k] - en[k - 1];
        }
        gpu_busy[r] = busy;
        gpu_gap[r] = K > 1 ? gap / (K - 1) : 0;
        gpu_span[r] = en[K - 1] - st[0];
    }
    if (quiet) return;
    printf("{\"test\": \"%s\", \"submits\": %d, \"dispatches_per_submit\": %u, \"iters\": %u, \"reps\": %d, "
           "\"host_total_us\": %.1f, \"host_submit_calls_us\": %.1f, \"gpu_busy_us\": %.1f, \"gpu_gap_between_submits_us\": %.1f, "
           "\"gpu_span_us\": %.1f}\n",
           label, K, dispatches, iters, reps, median(total, reps), median(sub_sum, reps), median(gpu_busy, reps),
           median(gpu_gap, reps), median(gpu_span, reps));
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s spin.spv [heavy_iters]\n", argv[0]);
        return 2;
    }
    uint32_t heavy = argc > 2 ? (uint32_t)atoi(argv[2]) : 20000;

    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "submit_probe", 1, NULL, 0, VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai, 0, NULL, 0, NULL };
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t npd = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &npd, &pd);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    ts_period_ns = props.limits.timestampPeriod;
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
    VkQueueFamilyProperties qfp[16];
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
    uint32_t qf = 0;
    while (qf < nqf && !(qfp[qf].queueFlags & VK_QUEUE_COMPUTE_BIT)) qf++;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio };
    VkPhysicalDeviceHostQueryResetFeatures hqr = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, NULL, VK_TRUE };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &hqr, 0, 1, &qci, 0, NULL, 0, NULL, NULL };
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qf, 0, &queue);

    // Storage buffer in device-local memory.
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, GROUPS * 128 * 4,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    VkBuffer buf;
    CK(vkCreateBuffer(dev, &bci, NULL, &buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    // PROBE_EXTRA_TYPE/PROBE_EXTRA_MIB/PROBE_EXTRA_COUNT: extra allocations of one memory type, mapped if
    // host-visible. PROBE_BUF_IN_EXTRA=1 places the dispatch buffer at the start of the first one.
    // PROBE_BUF_TYPE picks the dispatch buffer's own memory type otherwise (default: first device-local).
    const char * et = getenv("PROBE_EXTRA_TYPE");
    int extra_count = getenv("PROBE_EXTRA_COUNT") ? atoi(getenv("PROBE_EXTRA_COUNT")) : (et ? 1 : 0);
    VkDeviceSize extra_size = (VkDeviceSize)(getenv("PROBE_EXTRA_MIB") ? atoi(getenv("PROBE_EXTRA_MIB")) : 16) << 20;
    VkDeviceMemory extra[64];
    for (int e = 0; e < extra_count && e < 64; e++) {
        VkMemoryAllocateInfo eai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, extra_size, (uint32_t)atoi(et) };
        CK(vkAllocateMemory(dev, &eai, NULL, &extra[e]));
        if (mp.memoryTypes[atoi(et)].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            void * ptr;
            CK(vkMapMemory(dev, extra[e], 0, VK_WHOLE_SIZE, 0, &ptr));
            memset(ptr, 1, extra_size);
        }
    }
    if (extra_count && getenv("PROBE_BUF_IN_EXTRA") && atoi(getenv("PROBE_BUF_IN_EXTRA"))) {
        CK(vkBindBufferMemory(dev, buf, extra[0], 0));
    } else {
        uint32_t mt = 0;
        if (getenv("PROBE_BUF_TYPE")) {
            mt = (uint32_t)atoi(getenv("PROBE_BUF_TYPE"));
        } else {
            while (mt < mp.memoryTypeCount &&
                   !((mr.memoryTypeBits >> mt & 1) && (mp.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
                mt++;
        }
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mt };
        VkDeviceMemory mem;
        CK(vkAllocateMemory(dev, &mai, NULL, &mem));
        CK(vkBindBufferMemory(dev, buf, mem, 0));
    }

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
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr };
    CK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                         { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
                                         layout, VK_NULL_HANDLE, 0 };
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps };
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl };
    CK(vkAllocateDescriptorSets(dev, &dsai, &dset));
    VkDescriptorBufferInfo dbi = { buf, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, dset, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi, NULL };
    vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);

    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
                                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf };
    VkCommandPool pool;
    CK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool,
                                         VK_COMMAND_BUFFER_LEVEL_PRIMARY, NCB };
    CK(vkAllocateCommandBuffers(dev, &cbai, cbs));
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
    for (int i = 0; i < NCB; i++) CK(vkCreateFence(dev, &fci, NULL, &fences[i]));
    VkQueryPoolCreateInfo qpci = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2 * NCB, 0 };
    CK(vkCreateQueryPool(dev, &qpci, NULL, &qpool));

    // Warm up the clock.
    quiet = 1;
    for (int i = 0; i < 20; i++) chain("warmup", 4, 4, heavy, 1);
    quiet = 0;

    // Round trip of one trivial submission.
    {
        double sub[40], wt[40], tot[40];
        for (int r = 0; r < 40; r++) {
            record(0, 1, 0);
            double t0 = now_us();
            sub[r] = submit(0, fences[0]);
            wt[r] = wait_fence(fences[0]);
            tot[r] = now_us() - t0;
        }
        double lo = tot[0];
        for (int r = 1; r < 40; r++) lo = tot[r] < lo ? tot[r] : lo;
        printf("{\"test\": \"roundtrip_trivial\", \"reps\": 40, \"submit_call_us\": %.1f, \"fence_wait_us\": %.1f, "
               "\"total_us\": %.1f, \"min_total_us\": %.1f}\n",
               median(sub, 40), median(wt, 40), median(tot, 40), lo);
    }
    if (getenv("PROBE_QUICK")) {
        chain("chain_trivial", 8, 1, 0, 7);
        chain("decode_like", 1, 400, 64, 7);
        chain("decode_like", 10, 40, 64, 7);
        return 0;
    }
    // Back-to-back submissions of trivial work.
    int ks[] = { 1, 2, 4, 8, 16 };
    for (int i = 0; i < 5; i++) chain("chain_trivial", ks[i], 1, 0, 7);
    // Heavy work: is the per-submission cost overlapped with GPU execution?
    for (int i = 0; i < 4; i++) chain("chain_heavy", ks[i], 1, heavy, 7);
    // The same total work in one submission.
    chain("one_submit_heavy", 1, 8, heavy, 7);
    chain("chain_heavy", 8, 1, heavy, 7);
    // 400 small dispatches split into 1, 4 or 10 submissions (decode-like).
    chain("decode_like", 1, 400, 64, 7);
    chain("decode_like", 4, 100, 64, 7);
    chain("decode_like", 10, 40, 64, 7);

    // Fence on an empty submission after heavy work, and queue-idle wait.
    {
        double empty_fence[7], direct[7], idle[7], gpu[7];
        for (int r = 0; r < 7; r++) {
            record(0, 8, heavy);
            double t0 = now_us();
            submit(0, VK_NULL_HANDLE);
            CK(vkQueueSubmit(queue, 0, NULL, fences[0]));
            wait_fence(fences[0]);
            empty_fence[r] = now_us() - t0;
            double st[1], en[1];
            gpu_times(1, st, en);
            gpu[r] = en[0] - st[0];
            CK(vkQueueWaitIdle(queue));

            record(0, 8, heavy);
            t0 = now_us();
            submit(0, fences[0]);
            wait_fence(fences[0]);
            direct[r] = now_us() - t0;

            record(0, 8, heavy);
            t0 = now_us();
            submit(0, VK_NULL_HANDLE);
            CK(vkQueueWaitIdle(queue));
            idle[r] = now_us() - t0;
        }
        printf("{\"test\": \"completion_wait\", \"reps\": 7, \"gpu_busy_us\": %.1f, \"empty_submit_fence_us\": %.1f, "
               "\"fence_on_work_us\": %.1f, \"queue_wait_idle_us\": %.1f}\n",
               median(gpu, 7), median(empty_fence, 7), median(direct, 7), median(idle, 7));
    }
    return 0;
}
