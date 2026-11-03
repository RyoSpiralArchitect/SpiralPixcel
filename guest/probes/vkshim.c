// LD_PRELOAD timing shim for Vulkan calls made through vkGet*ProcAddr (vulkan-hpp dynamic dispatch).
// Build: gcc -O2 -shared -fPIC vkshim.c -o vkshim.so -ldl. Run any ggml-vulkan program with
// LD_PRELOAD=vkshim.so VKSHIM_OUT=summary.json [VKSHIM_GPU_TS=1] [VKSHIM_TRACE_FROM=N ...].
// Counts and times selected entry points; optionally records a trace window between two
// vkQueueWaitIdle calls. Output goes to $VKSHIM_OUT (summary) and $VKSHIM_TRACE (events).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

#define FUNCS(X)                                                                                   \
    X(QueueSubmit) X(QueueSubmit2) X(QueueWaitIdle) X(DeviceWaitIdle) X(WaitForFences)             \
    X(ResetFences) X(GetFenceStatus) X(WaitSemaphores) X(GetSemaphoreCounterValue) X(MapMemory)   \
    X(UnmapMemory) X(FlushMappedMemoryRanges) X(InvalidateMappedMemoryRanges)                    \
    X(UpdateDescriptorSets) X(BeginCommandBuffer) X(EndCommandBuffer) X(ResetCommandBuffer)       \
    X(ResetCommandPool) X(AllocateCommandBuffers) X(CmdDispatch) X(CmdPipelineBarrier)            \
    X(CmdCopyBuffer) X(CmdFillBuffer) X(CmdPushConstants) X(CmdBindPipeline)                      \
    X(CmdBindDescriptorSets) X(CmdWriteTimestamp) X(AllocateMemory) X(FreeMemory)                 \
    X(CreateBuffer) X(DestroyBuffer)

enum {
#define X(n) F_##n,
    FUNCS(X)
#undef X
    F_COUNT
};
static const char * names[F_COUNT] = {
#define X(n) #n,
    FUNCS(X)
#undef X
};
static uint64_t cnt[F_COUNT], tot[F_COUNT], mx[F_COUNT];

typedef struct { uint64_t t0, dt; int f; uint32_t arg; } ev_t;
static ev_t * trace;
static size_t ntrace, captrace = 1 << 20;
static int tracing;
static long waitidle_seen, trace_from = -1, trace_len = 1;
static uint64_t first_ns;
static VkDevice g_dev;
static void * real_sym(const char * name);

static inline void rec(int f, uint64_t t0, uint32_t arg) {
    uint64_t dt = now_ns() - t0;
    cnt[f]++;
    tot[f] += dt;
    if (dt > mx[f]) mx[f] = dt;
    if (tracing && ntrace < captrace) trace[ntrace++] = (ev_t){ t0, dt, f, arg };
}

#define DEF(ret, n, params, args, arg)                                                             \
    static PFN_vk##n real_##n;                                                                     \
    static VKAPI_ATTR ret VKAPI_CALL w_##n params {                                               \
        uint64_t t0 = now_ns();                                                                    \
        ret res_ = real_##n args;                                                                   \
        rec(F_##n, t0, arg);                                                                       \
        return res_;                                                                               \
    }
#define DEFV(n, params, args, arg)                                                                 \
    static PFN_vk##n real_##n;                                                                     \
    static VKAPI_ATTR void VKAPI_CALL w_##n params {                                              \
        uint64_t t0 = now_ns();                                                                    \
        real_##n args;                                                                             \
        rec(F_##n, t0, arg);                                                                       \
    }

DEF(VkResult, QueueSubmit2, (VkQueue q, uint32_t n, const VkSubmitInfo2 * s, VkFence f), (q, n, s, f), n)
DEF(VkResult, DeviceWaitIdle, (VkDevice d), (d), 0)
DEF(VkResult, ResetFences, (VkDevice d, uint32_t n, const VkFence * f), (d, n, f), n)
DEF(VkResult, GetFenceStatus, (VkDevice d, VkFence f), (d, f), 0)
DEF(VkResult, WaitSemaphores, (VkDevice d, const VkSemaphoreWaitInfo * i, uint64_t t), (d, i, t), 0)
DEF(VkResult, GetSemaphoreCounterValue, (VkDevice d, VkSemaphore s, uint64_t * v), (d, s, v), 0)
DEF(VkResult, MapMemory, (VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize s, VkMemoryMapFlags fl, void ** p),
    (d, m, o, s, fl, p), (uint32_t)(s >> 10))
DEFV(UnmapMemory, (VkDevice d, VkDeviceMemory m), (d, m), 0)
DEF(VkResult, FlushMappedMemoryRanges, (VkDevice d, uint32_t n, const VkMappedMemoryRange * r), (d, n, r),
    n ? (uint32_t)(r[0].size >> 10) : 0)
DEF(VkResult, InvalidateMappedMemoryRanges, (VkDevice d, uint32_t n, const VkMappedMemoryRange * r), (d, n, r),
    n ? (uint32_t)(r[0].size >> 10) : 0)
DEFV(UpdateDescriptorSets, (VkDevice d, uint32_t nw, const VkWriteDescriptorSet * w, uint32_t nc, const VkCopyDescriptorSet * c),
     (d, nw, w, nc, c), nw)
DEF(VkResult, ResetCommandBuffer, (VkCommandBuffer b, VkCommandBufferResetFlags f), (b, f), 0)
DEF(VkResult, ResetCommandPool, (VkDevice d, VkCommandPool p, VkCommandPoolResetFlags f), (d, p, f), 0)
DEF(VkResult, AllocateCommandBuffers, (VkDevice d, const VkCommandBufferAllocateInfo * i, VkCommandBuffer * b), (d, i, b), 0)
DEFV(CmdPipelineBarrier,
     (VkCommandBuffer b, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags f, uint32_t nm,
      const VkMemoryBarrier * m, uint32_t nb, const VkBufferMemoryBarrier * bb, uint32_t ni, const VkImageMemoryBarrier * ib),
     (b, s, d, f, nm, m, nb, bb, ni, ib), nm + nb)
DEFV(CmdCopyBuffer, (VkCommandBuffer b, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy * r), (b, s, d, n, r),
     n ? (uint32_t)r[0].size : 0)
DEFV(CmdFillBuffer, (VkCommandBuffer b, VkBuffer d, VkDeviceSize o, VkDeviceSize s, uint32_t v), (b, d, o, s, v), (uint32_t)s)
DEFV(CmdPushConstants, (VkCommandBuffer b, VkPipelineLayout l, VkShaderStageFlags s, uint32_t o, uint32_t n, const void * v),
     (b, l, s, o, n, v), n)
DEFV(CmdBindPipeline, (VkCommandBuffer b, VkPipelineBindPoint p, VkPipeline pl), (b, p, pl), 0)
DEFV(CmdBindDescriptorSets,
     (VkCommandBuffer b, VkPipelineBindPoint p, VkPipelineLayout l, uint32_t f, uint32_t n, const VkDescriptorSet * s,
      uint32_t nd, const uint32_t * d),
     (b, p, l, f, n, s, nd, d), n)
DEFV(CmdWriteTimestamp, (VkCommandBuffer b, VkPipelineStageFlagBits s, VkQueryPool p, uint32_t q), (b, s, p, q), q)
// Allocations are also logged to $VKSHIM_ALLOC (type index, KiB, pNext sType, result).
static PFN_vkAllocateMemory real_AllocateMemory;
static VKAPI_ATTR VkResult VKAPI_CALL w_AllocateMemory(VkDevice d, const VkMemoryAllocateInfo * i,
                                                       const VkAllocationCallbacks * a, VkDeviceMemory * m) {
    uint64_t t0 = now_ns();
    VkResult res_ = real_AllocateMemory(d, i, a, m);
    rec(F_AllocateMemory, t0, (uint32_t)(i->allocationSize >> 10));
    // vulkan-hpp resolves device functions through the instance, so the device is first seen here.
    if (!g_dev) g_dev = d;
    static FILE * log;
    const char * path = getenv("VKSHIM_ALLOC");
    if (path && !log) log = fopen(path, "w");
    if (log) {
        const VkBaseInStructure * next = (const VkBaseInStructure *)i->pNext;
        fprintf(log, "alloc type=%u kib=%llu next=%d res=%d us=%.1f\n", i->memoryTypeIndex,
                (unsigned long long)(i->allocationSize >> 10), next ? (int)next->sType : 0, res_, (now_ns() - t0) / 1e3);
        fflush(log);
    }
    return res_;
}
DEFV(FreeMemory, (VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks * a), (d, m, a), 0)
DEF(VkResult, CreateBuffer, (VkDevice d, const VkBufferCreateInfo * i, const VkAllocationCallbacks * a, VkBuffer * b),
    (d, i, a, b), (uint32_t)(i->size >> 10))
DEFV(DestroyBuffer, (VkDevice d, VkBuffer b, const VkAllocationCallbacks * a), (d, b, a), 0)


// GPU timestamps around every command buffer (VKSHIM_GPU_TS=1), read back after fence and queue waits.
static int gpu_ts;
static PFN_vkGetDeviceProcAddr real_gdpa;
static VkQueryPool g_pool;
static PFN_vkCmdWriteTimestamp p_write_ts;
static PFN_vkCmdResetQueryPool p_reset_qp;
static PFN_vkGetQueryPoolResults p_get_qp;
#define QSLOTS 1024
typedef struct { VkCommandBuffer cb; uint32_t slot, dispatches; } cbmap_t;
static cbmap_t cbmap[256];
static uint32_t next_slot;
typedef struct { uint64_t cpu_ns; uint32_t slot, dispatches; } pend_t;
static pend_t pend[512];
static int npend;
typedef struct { uint64_t cpu_ns, g0, g1; uint32_t dispatches; } gev_t;
static gev_t * gtrace;
static size_t ngtrace, capg = 1 << 16;
static uint64_t gpu_busy_ticks, gpu_batches;

static cbmap_t * cb_entry(VkCommandBuffer cb, int create) {
    static cbmap_t * last;
    if (last && last->cb == cb) return last;
    for (int i = 0; i < 256; i++)
        if (cbmap[i].cb == cb) return last = &cbmap[i];
    if (!create) return NULL;
    for (int i = 0; i < 256; i++)
        if (!cbmap[i].cb) {
            cbmap[i].cb = cb;
            return last = &cbmap[i];
        }
    return NULL;
}

static int gpu_ts_ready(void) {
    if (!gpu_ts || !g_dev) return 0;
    if (!real_gdpa) real_gdpa = (PFN_vkGetDeviceProcAddr)real_sym("vkGetDeviceProcAddr");
    if (g_pool) return 1;
    PFN_vkCreateQueryPool create = (PFN_vkCreateQueryPool)real_gdpa(g_dev, "vkCreateQueryPool");
    p_write_ts = (PFN_vkCmdWriteTimestamp)real_gdpa(g_dev, "vkCmdWriteTimestamp");
    p_reset_qp = (PFN_vkCmdResetQueryPool)real_gdpa(g_dev, "vkCmdResetQueryPool");
    p_get_qp = (PFN_vkGetQueryPoolResults)real_gdpa(g_dev, "vkGetQueryPoolResults");
    VkQueryPoolCreateInfo ci = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2 * QSLOTS, 0 };
    if (!create || create(g_dev, &ci, NULL, &g_pool) != VK_SUCCESS) {
        gpu_ts = 0;
        return 0;
    }
    gtrace = calloc(capg, sizeof(gev_t));
    return 1;
}

static void resolve_pending(void) {
    for (int i = 0; i < npend; i++) {
        uint64_t v[2];
        if (p_get_qp(g_dev, g_pool, 2 * pend[i].slot, 2, sizeof(v), v, 8,
                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
            continue;
        gpu_busy_ticks += v[1] - v[0];
        gpu_batches++;
        if (tracing && gtrace && ngtrace < capg) gtrace[ngtrace++] = (gev_t){ pend[i].cpu_ns, v[0], v[1], pend[i].dispatches };
    }
    npend = 0;
}

static PFN_vkBeginCommandBuffer real_BeginCommandBuffer;
static VKAPI_ATTR VkResult VKAPI_CALL w_BeginCommandBuffer(VkCommandBuffer b, const VkCommandBufferBeginInfo * i) {
    uint64_t t0 = now_ns();
    VkResult res_ = real_BeginCommandBuffer(b, i);
    if (res_ == VK_SUCCESS && gpu_ts_ready()) {
        cbmap_t * e = cb_entry(b, 1);
        if (e) {
            e->slot = next_slot++ % QSLOTS;
            e->dispatches = 0;
            p_reset_qp(b, g_pool, 2 * e->slot, 2);
            p_write_ts(b, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_pool, 2 * e->slot);
        }
    }
    rec(F_BeginCommandBuffer, t0, 0);
    return res_;
}
static PFN_vkEndCommandBuffer real_EndCommandBuffer;
static VKAPI_ATTR VkResult VKAPI_CALL w_EndCommandBuffer(VkCommandBuffer b) {
    uint64_t t0 = now_ns();
    if (gpu_ts && g_pool) {
        cbmap_t * e = cb_entry(b, 0);
        if (e) p_write_ts(b, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_pool, 2 * e->slot + 1);
    }
    VkResult res_ = real_EndCommandBuffer(b);
    rec(F_EndCommandBuffer, t0, 0);
    return res_;
}
static PFN_vkCmdDispatch real_CmdDispatch;
static VKAPI_ATTR void VKAPI_CALL w_CmdDispatch(VkCommandBuffer b, uint32_t x, uint32_t y, uint32_t z) {
    uint64_t t0 = now_ns();
    real_CmdDispatch(b, x, y, z);
    if (gpu_ts && g_pool) {
        cbmap_t * e = cb_entry(b, 0);
        if (e) e->dispatches++;
    }
    rec(F_CmdDispatch, t0, x);
}
static PFN_vkQueueSubmit real_QueueSubmit;
static VKAPI_ATTR VkResult VKAPI_CALL w_QueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo * s, VkFence f) {
    uint64_t t0 = now_ns();
    VkResult res_ = real_QueueSubmit(q, n, s, f);
    rec(F_QueueSubmit, t0, n ? s[0].commandBufferCount : 0);
    if (gpu_ts && g_pool)
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j < s[i].commandBufferCount; j++) {
                cbmap_t * e = cb_entry(s[i].pCommandBuffers[j], 0);
                if (e && npend < 512) pend[npend++] = (pend_t){ t0, e->slot, e->dispatches };
            }
    return res_;
}
static PFN_vkWaitForFences real_WaitForFences;
static VKAPI_ATTR VkResult VKAPI_CALL w_WaitForFences(VkDevice d, uint32_t n, const VkFence * f, VkBool32 a, uint64_t t) {
    uint64_t t0 = now_ns();
    VkResult res_ = real_WaitForFences(d, n, f, a, t);
    rec(F_WaitForFences, t0, n);
    if (res_ == VK_SUCCESS && gpu_ts && g_pool) resolve_pending();
    return res_;
}


// VKSHIM_HIDE_DL_TYPE=N: report memory type N without DEVICE_LOCAL, so that device buffers fall back to
// plain device-local memory while staging buffers can still use N.
static void hide_device_local(VkPhysicalDeviceMemoryProperties * mp) {
    const char * t = getenv("VKSHIM_HIDE_DL_TYPE");
    if (!t) return;
    uint32_t i = (uint32_t)atoi(t);
    if (i < mp->memoryTypeCount) mp->memoryTypes[i].propertyFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}
static PFN_vkGetPhysicalDeviceMemoryProperties real_GetPhysicalDeviceMemoryProperties;
static VKAPI_ATTR void VKAPI_CALL w_GetPhysicalDeviceMemoryProperties(VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties * mp) {
    real_GetPhysicalDeviceMemoryProperties(pd, mp);
    hide_device_local(mp);
}
static PFN_vkGetPhysicalDeviceMemoryProperties2 real_GetPhysicalDeviceMemoryProperties2;
static VKAPI_ATTR void VKAPI_CALL w_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2 * mp) {
    real_GetPhysicalDeviceMemoryProperties2(pd, mp);
    hide_device_local(&mp->memoryProperties);
}

// vkQueueWaitIdle also drives the trace window.
static PFN_vkQueueWaitIdle real_QueueWaitIdle;
static VKAPI_ATTR VkResult VKAPI_CALL w_QueueWaitIdle(VkQueue q) {
    if (trace_from >= 0 && waitidle_seen == trace_from) tracing = 1;
    uint64_t t0 = now_ns();
    VkResult r = real_QueueWaitIdle(q);
    rec(F_QueueWaitIdle, t0, 0);
    if (r == VK_SUCCESS && gpu_ts && g_pool) resolve_pending();
    waitidle_seen++;
    if (tracing && waitidle_seen >= trace_from + trace_len) tracing = 0;
    return r;
}

// Lookups made by the Vulkan loader itself must stay unwrapped.
static int from_loader(void * ret) {
    Dl_info info;
    return dladdr(ret, &info) && info.dli_fname && strstr(info.dli_fname, "libvulkan.so");
}

static PFN_vkVoidFunction lookup(const char * name, PFN_vkVoidFunction p) {
    if (!p) return p;
#define X(n)                                                                                       \
    if (!strcmp(name, "vk" #n)) {                                                                  \
        real_##n = (PFN_vk##n)p;                                                                   \
        return (PFN_vkVoidFunction)w_##n;                                                          \
    }
    FUNCS(X)
    X(GetPhysicalDeviceMemoryProperties) X(GetPhysicalDeviceMemoryProperties2)
    if (!strcmp(name, "vkGetPhysicalDeviceMemoryProperties2KHR")) {
        real_GetPhysicalDeviceMemoryProperties2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)p;
        return (PFN_vkVoidFunction)w_GetPhysicalDeviceMemoryProperties2;
    }
#undef X
    return p;
}

// ggml may dlopen its backend, which leaves libvulkan outside the global lookup scope.
static void * real_sym(const char * name) {
    static void * lib;
    if (!lib) lib = dlopen("libvulkan.so.1", RTLD_NOW);
    void * p = lib ? dlsym(lib, name) : NULL;
    if (!p) {
        fprintf(stderr, "vkshim: cannot resolve %s\n", name);
        abort();
    }
    return p;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice d, const char * name);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance i, const char * name) {
    static PFN_vkGetInstanceProcAddr real;
    if (!real) real = (PFN_vkGetInstanceProcAddr)real_sym("vkGetInstanceProcAddr");
    if (from_loader(__builtin_return_address(0))) return real(i, name);
    if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    return lookup(name, real(i, name));
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice d, const char * name) {
    static PFN_vkGetDeviceProcAddr real;
    if (!real) real = (PFN_vkGetDeviceProcAddr)real_sym("vkGetDeviceProcAddr");
    if (from_loader(__builtin_return_address(0))) return real(d, name);

    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    return lookup(name, real(d, name));
}
// libggml-vulkan also imports vkCmdCopyBuffer directly.
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer b, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy * r) {
    static PFN_vkCmdCopyBuffer real;
    if (!real) real = (PFN_vkCmdCopyBuffer)real_sym("vkCmdCopyBuffer");
    uint64_t t0 = now_ns();
    real(b, s, d, n, r);
    rec(F_CmdCopyBuffer, t0, n ? (uint32_t)r[0].size : 0);
}

__attribute__((constructor)) static void init(void) {
    first_ns = now_ns();
    gpu_ts = getenv("VKSHIM_GPU_TS") && atoi(getenv("VKSHIM_GPU_TS"));
    const char * from = getenv("VKSHIM_TRACE_FROM");
    const char * len = getenv("VKSHIM_TRACE_LEN");
    if (from) {
        trace_from = atol(from);
        trace_len = len ? atol(len) : 1;
        trace = calloc(captrace, sizeof(ev_t));
    }
}

__attribute__((destructor)) static void fini(void) {
    // Wrappers such as timeout(1) inherit LD_PRELOAD; only a process that made Vulkan calls reports.
    uint64_t calls = 0;
    for (int i = 0; i < F_COUNT; i++) calls += cnt[i];
    if (!calls) return;
    const char * out = getenv("VKSHIM_OUT");
    FILE * f = out ? fopen(out, "w") : stderr;
    if (!f) return;
    fprintf(f, "{\"wall_ns\": %llu, \"waitidle_calls\": %ld, \"funcs\": {", (unsigned long long)(now_ns() - first_ns), waitidle_seen);
    int first = 1;
    for (int i = 0; i < F_COUNT; i++) {
        if (!cnt[i]) continue;
        fprintf(f, "%s\"%s\": [%llu, %llu, %llu]", first ? "" : ", ", names[i], (unsigned long long)cnt[i],
                (unsigned long long)tot[i], (unsigned long long)mx[i]);
        first = 0;
    }
    fprintf(f, "}, \"gpu_batches\": %llu, \"gpu_busy_ticks\": %llu}\n", (unsigned long long)gpu_batches,
            (unsigned long long)gpu_busy_ticks);
    if (f != stderr) fclose(f);
    const char * gout = getenv("VKSHIM_GPU_TRACE");
    if (gtrace && gout) {
        FILE * g = fopen(gout, "w");
        if (g) {
            for (size_t i = 0; i < ngtrace; i++)
                fprintf(g, "%llu %llu %llu %u\n", (unsigned long long)gtrace[i].cpu_ns, (unsigned long long)gtrace[i].g0,
                        (unsigned long long)gtrace[i].g1, gtrace[i].dispatches);
            fclose(g);
        }
    }
    const char * tout = getenv("VKSHIM_TRACE");
    if (trace && tout) {
        FILE * t = fopen(tout, "w");
        if (t) {
            for (size_t i = 0; i < ntrace; i++)
                fprintf(t, "%llu %llu %s %u\n", (unsigned long long)trace[i].t0, (unsigned long long)trace[i].dt,
                        names[trace[i].f], trace[i].arg);
            fclose(t);
        }
    }
}
