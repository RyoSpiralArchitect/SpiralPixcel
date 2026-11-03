// Time one ggml-style mat-vec compute shader on the Pixel GPU and check it on the CPU.
// Weights are random rows quantized by the pinned ggml; the reference is ggml's own
// dequantization in double precision. Bindings and push constants follow ggml-vulkan's
// mul_mat_vec interface, so its shaders and new candidates run unchanged.
#include <vulkan/vulkan.h>
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(VkResult r, const char* operation) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": VkResult " + std::to_string(r));
}
#define VK(call) check((call), #call)

static std::vector<uint32_t> read_words(const std::string& name) {
    std::ifstream in(name, std::ios::binary | std::ios::ate);
    const auto size = in.tellg();
    if (size <= 0 || size % 4) throw std::runtime_error("invalid shader: " + name);
    std::vector<uint32_t> words(size / 4);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(words.data()), size)) throw std::runtime_error("short read");
    return words;
}

static std::vector<float> values(size_t count, uint32_t seed) {
    std::vector<float> out(count);
    for (auto& value : out) {
        seed = seed * 1664525u + 1013904223u;
        value = (static_cast<int32_t>(seed >> 8) - 8388608) / 8388608.0f;
    }
    return out;
}

struct MatVecPush {
    uint32_t ncols, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d;
    uint32_t fusion_flags, base_work_group_y, ne02, ne12, broadcast2, broadcast3;
};

struct Buffer {
    VkBuffer handle{};
    VkDeviceMemory memory{};
    void* mapped{};
    VkDeviceSize size{};
};

struct Gpu {
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkPhysicalDeviceProperties props{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t family = 0;
    VkCommandPool pool{};

    Buffer allocate(VkDeviceSize bytes, VkMemoryPropertyFlags wanted, VkBufferUsageFlags usage) {
        Buffer b;
        b.size = bytes;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        VK(vkCreateBuffer(device, &info, nullptr, &b.handle));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.handle, &req);
        VkPhysicalDeviceMemoryProperties memory;
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        uint32_t chosen = UINT32_MAX;
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & wanted) == wanted &&
                !(memory.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT))) {
                chosen = i;
                break;
            }
        }
        if (chosen == UINT32_MAX) throw std::runtime_error("no matching memory type");
        VkMemoryAllocateInfo mem{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mem.allocationSize = req.size;
        mem.memoryTypeIndex = chosen;
        VK(vkAllocateMemory(device, &mem, nullptr, &b.memory));
        VK(vkBindBufferMemory(device, b.handle, b.memory, 0));
        if (wanted & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) VK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
        return b;
    }
    void release(Buffer& b) {
        if (b.mapped) vkUnmapMemory(device, b.memory);
        vkDestroyBuffer(device, b.handle, nullptr);
        vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }
    VkCommandBuffer begin() {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool; cai.commandBufferCount = 1; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        VkCommandBuffer cb;
        VK(vkAllocateCommandBuffers(device, &cai, &cb));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK(vkBeginCommandBuffer(cb, &begin));
        return cb;
    }
    void submit(VkCommandBuffer cb) {
        VK(vkEndCommandBuffer(cb));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1; submit.pCommandBuffers = &cb;
        VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        VK(vkQueueWaitIdle(queue));
        vkFreeCommandBuffers(device, pool, 1, &cb);
    }
    // Device-local memory is filled through a host-visible staging copy.
    void upload(Buffer& dst, const void* data, VkDeviceSize bytes, VkDeviceSize offset = 0) {
        Buffer staging = allocate(bytes, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memcpy(staging.mapped, data, bytes);
        VkCommandBuffer cb = begin();
        VkBufferCopy region{0, offset, bytes};
        vkCmdCopyBuffer(cb, staging.handle, dst.handle, 1, &region);
        submit(cb);
        release(staging);
    }
};

static void open_gpu(Gpu& g) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "SpiralPixcel kernel bench"; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    VK(vkCreateInstance(&ii, nullptr, &g.instance));
    uint32_t count = 0;
    VK(vkEnumeratePhysicalDevices(g.instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    if (count) VK(vkEnumeratePhysicalDevices(g.instance, &count, devices.data()));
    for (auto candidate : devices) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(candidate, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
        uint32_t n = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, nullptr);
        std::vector<VkQueueFamilyProperties> queues(n);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, queues.data());
        for (uint32_t i = 0; i < n; ++i) {
            if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && queues[i].timestampValidBits) {
                g.physical = candidate; g.props = p; g.family = i; break;
            }
        }
        if (g.physical) break;
    }
    if (!g.physical) throw std::runtime_error("no hardware Vulkan compute queue with timestamps");
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f12.pNext = &f13; f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(g.physical, &features);
    // Enable only what ggml-vulkan's mat-vec shaders use, plus subgroup size control.
    VkPhysicalDeviceVulkan13Features e13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    e13.subgroupSizeControl = f13.subgroupSizeControl; e13.computeFullSubgroups = f13.computeFullSubgroups;
    e13.shaderIntegerDotProduct = f13.shaderIntegerDotProduct;  // the unsigned-dot kernels and probes
    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    e12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess; e12.shaderFloat16 = f12.shaderFloat16;
    e12.shaderInt8 = f12.shaderInt8; e12.uniformAndStorageBuffer8BitAccess = f12.uniformAndStorageBuffer8BitAccess;
    e12.pNext = &e13;
    VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    e11.storageBuffer16BitAccess = f11.storageBuffer16BitAccess;
    e11.uniformAndStorageBuffer16BitAccess = f11.uniformAndStorageBuffer16BitAccess;
    e11.pNext = &e12;
    VkPhysicalDeviceFeatures2 enabled{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    enabled.features.shaderInt16 = features.features.shaderInt16;
    enabled.features.shaderInt64 = features.features.shaderInt64;
    enabled.pNext = &e11;
    const float priority = 1;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = g.family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.pNext = &enabled; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    VK(vkCreateDevice(g.physical, &di, nullptr, &g.device));
    vkGetDeviceQueue(g.device, g.family, 0, &g.queue);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = g.family;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK(vkCreateCommandPool(g.device, &cpi, nullptr, &g.pool));
}

static ggml_type parse_type(const std::string& name) {
    static const std::map<std::string, ggml_type> types{
        {"f32", GGML_TYPE_F32}, {"f16", GGML_TYPE_F16}, {"q4_0", GGML_TYPE_Q4_0}, {"q5_0", GGML_TYPE_Q5_0},
        {"q8_0", GGML_TYPE_Q8_0}, {"q4_k", GGML_TYPE_Q4_K}, {"q5_k", GGML_TYPE_Q5_K}, {"q6_k", GGML_TYPE_Q6_K}};
    auto it = types.find(name);
    if (it == types.end()) throw std::runtime_error("unsupported type: " + name);
    return it->second;
}

static std::vector<uint32_t> parse_list(const std::string& text) {
    std::vector<uint32_t> out;
    std::stringstream in(text);
    for (std::string item; std::getline(in, item, ',');) {
        if (!item.empty()) out.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    return out;
}

static int run(int argc, char** argv) {
    if (argc < 7) {
        throw std::runtime_error("expected: shader.spv type M K rows_per_workgroup spec0,spec1,... "
                                 "[--subgroup N] [--copies C] [--reps R] [--seed S] [--label TEXT]");
    }
    const std::string shader_path = argv[1];
    const ggml_type type = parse_type(argv[2]);
    const uint32_t m = std::stoul(argv[3]), k = std::stoul(argv[4]), rows = std::stoul(argv[5]);
    const std::vector<uint32_t> spec = parse_list(argv[6]);
    uint32_t subgroup = 0, copies = 8, reps = 64, seed = 1;
    std::string label = shader_path, barrier_kind = "memory";
    bool device_io = false;
    std::string b_format = "f32", dump_path;
    for (int i = 7; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--subgroup") subgroup = std::stoul(argv[i + 1]);
        else if (flag == "--copies") copies = std::stoul(argv[i + 1]);
        else if (flag == "--reps") reps = std::stoul(argv[i + 1]);
        else if (flag == "--seed") seed = std::stoul(argv[i + 1]);
        else if (flag == "--label") label = argv[i + 1];
        else if (flag == "--barrier") barrier_kind = argv[i + 1];
        else if (flag == "--io-memory") device_io = std::string(argv[i + 1]) == "device";
        else if (flag == "--b-format") b_format = argv[i + 1];
        else if (flag == "--dump") dump_path = argv[i + 1];
        else throw std::runtime_error("unknown option: " + flag);
    }
    if (!m || !k || !rows || !copies || !reps || k % ggml_blck_size(type)) throw std::runtime_error("invalid shape");

    // Random weights quantized by ggml; the reference uses ggml's own dequantization.
    const size_t row_bytes = ggml_row_size(type, k);
    const size_t a_bytes = row_bytes * m;
    std::vector<uint8_t> quantized(a_bytes);
    {
        const auto weights = values(size_t(m) * k, seed);
        if (type == GGML_TYPE_F32) std::memcpy(quantized.data(), weights.data(), a_bytes);
        else ggml_quantize_chunk(type, weights.data(), quantized.data(), 0, m, k, nullptr);
    }
    const auto x = values(k, seed * 7919u + 17u);
    // Activations as the kernel sees them: f32, or ggml-vulkan's block_q8_1_x4 (four f16vec2
    // (d, d*sum) words, then four blocks of 32 int8 values: 144 bytes per 128 values).
    std::vector<float> effective = x;
    std::vector<uint8_t> b_bytes(reinterpret_cast<const uint8_t*>(x.data()),
                                 reinterpret_cast<const uint8_t*>(x.data() + k));
    if (b_format == "q8_1x4") {
        if (k % 32) throw std::runtime_error("q8_1x4 activations need K divisible by 32");
        const uint32_t blocks = k / 32;
        b_bytes.assign(size_t((blocks + 3) / 4) * 144, 0);
        for (uint32_t blk = 0; blk < blocks; ++blk) {
            float amax = 0;
            for (uint32_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32 * blk + j]));
            const float d = amax / 127.0f, inv = d != 0 ? 1.0f / d : 0.0f;
            const ggml_fp16_t dh = ggml_fp32_to_fp16(d);
            int8_t q[32];
            int sum = 0;
            for (uint32_t j = 0; j < 32; ++j) {
                q[j] = static_cast<int8_t>(std::lround(x[32 * blk + j] * inv));
                sum += q[j];
                effective[32 * blk + j] = ggml_fp16_to_fp32(dh) * q[j];
            }
            const ggml_fp16_t ds[2] = {dh, ggml_fp32_to_fp16(d * sum)};
            uint8_t* group = b_bytes.data() + size_t(blk / 4) * 144;
            std::memcpy(group + (blk % 4) * 4, ds, 4);
            std::memcpy(group + 16 + (blk % 4) * 32, q, 32);
        }
    } else if (b_format != "f32") {
        throw std::runtime_error("--b-format must be f32 or q8_1x4");
    }
    std::vector<double> reference(m), reference_f32(m);
    {
        const auto* traits = ggml_get_type_traits(type);
        std::vector<float> row(k);
        for (uint32_t r = 0; r < m; ++r) {
            const uint8_t* src = quantized.data() + size_t(r) * row_bytes;
            if (type == GGML_TYPE_F32) std::memcpy(row.data(), src, k * sizeof(float));
            else traits->to_float(src, row.data(), k);
            double sum = 0, sum_f32 = 0;
            for (uint32_t c = 0; c < k; ++c) {
                sum += double(row[c]) * effective[c];
                sum_f32 += double(row[c]) * x[c];
            }
            reference[r] = sum;
            reference_f32[r] = sum_f32;
        }
    }

    Gpu g;
    open_gpu(g);
    // Each copy is its own allocation so the 128 MiB descriptor range applies per copy, and
    // rotating through copies keeps one dispatch's weights from staying cached for the next.
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    std::vector<Buffer> a(copies);
    const VkDeviceSize a_alloc = (a_bytes + 255) & ~VkDeviceSize(255);
    for (auto& buffer : a) {
        buffer = g.allocate(a_alloc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storage);
        g.upload(buffer, quantized.data(), a_bytes);
    }
    // Activations either share host-coherent memory with the CPU or stay device-local.
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags io = device_io ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : host;
    const VkBufferUsageFlags io_usage = storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    Buffer b = g.allocate(std::max<VkDeviceSize>(b_bytes.size(), 256), io, io_usage);
    Buffer d = g.allocate(std::max<VkDeviceSize>(m * sizeof(float), 256), io, io_usage);
    Buffer fuse = g.allocate(std::max<VkDeviceSize>(m * sizeof(float), 256), io, io_usage);
    Buffer readback = g.allocate(std::max<VkDeviceSize>(m * sizeof(float), 256), host, io_usage);
    {
        const std::vector<float> poison(m, std::nanf("")), zeros(m, 0.0f);
        g.upload(b, b_bytes.data(), b_bytes.size());
        g.upload(d, poison.data(), m * sizeof(float));
        g.upload(fuse, zeros.data(), m * sizeof(float));
    }

    VkDescriptorSetLayoutBinding bindings[5];
    for (uint32_t i = 0; i < 5; ++i) bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 5; li.pBindings = bindings;
    VkDescriptorSetLayout layout;
    VK(vkCreateDescriptorSetLayout(g.device, &li, nullptr, &layout));
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MatVecPush)};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1; pi.pSetLayouts = &layout; pi.pushConstantRangeCount = 1; pi.pPushConstantRanges = &push;
    VkPipelineLayout pipe_layout;
    VK(vkCreatePipelineLayout(g.device, &pi, nullptr, &pipe_layout));
    const auto code = read_words(shader_path);
    VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    si.codeSize = code.size() * 4; si.pCode = code.data();
    VkShaderModule shader;
    VK(vkCreateShaderModule(g.device, &si, nullptr, &shader));
    std::vector<VkSpecializationMapEntry> entries(spec.size());
    for (uint32_t i = 0; i < spec.size(); ++i) entries[i] = {i, uint32_t(i * sizeof(uint32_t)), sizeof(uint32_t)};
    VkSpecializationInfo spec_info{uint32_t(entries.size()), entries.data(), spec.size() * sizeof(uint32_t), spec.data()};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    required.requiredSubgroupSize = subgroup;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = shader; ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = spec.empty() ? nullptr : &spec_info;
    if (subgroup) {
        ci.stage.pNext = &required;
        ci.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    ci.layout = pipe_layout;
    VkPipeline pipeline;
    VK(vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline));

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * copies};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = copies; dpi.poolSizeCount = 1; dpi.pPoolSizes = &pool_size;
    VkDescriptorPool pool;
    VK(vkCreateDescriptorPool(g.device, &dpi, nullptr, &pool));
    std::vector<VkDescriptorSetLayout> layouts(copies, layout);
    std::vector<VkDescriptorSet> sets(copies);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool; ai.descriptorSetCount = copies; ai.pSetLayouts = layouts.data();
    VK(vkAllocateDescriptorSets(g.device, &ai, sets.data()));
    for (uint32_t c = 0; c < copies; ++c) {
        const VkDescriptorBufferInfo infos[5]{{a[c].handle, 0, a_bytes}, {b.handle, 0, b_bytes.size()},
                                              {d.handle, 0, m * sizeof(float)}, {fuse.handle, 0, m * sizeof(float)},
                                              {fuse.handle, 0, m * sizeof(float)}};
        VkWriteDescriptorSet writes[5];
        for (uint32_t i = 0; i < 5; ++i) {
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = sets[c]; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(g.device, 5, writes, 0, nullptr);
    }

    // Same grid as ggml_vk_mul_mat_vec_q_f16: ne01 rows over x, split into z beyond the limit.
    uint32_t groups_x = m, groups_z = 1;
    if (m > g.props.limits.maxComputeWorkGroupCount[0]) {
        groups_z = 64;
        groups_x = (m + groups_z - 1) / groups_z;
    }
    groups_x = (groups_x + rows - 1) / rows;
    const MatVecPush pc{k, k, k, m, k * m, k, m, 0, 0, 1, 1, 1, 1};

    VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qp.queryType = VK_QUERY_TYPE_TIMESTAMP; qp.queryCount = 2;
    VkQueryPool queries;
    VK(vkCreateQueryPool(g.device, &qp, nullptr, &queries));
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    // Only the ends of a sequence are stamped: a PowerVR timestamp between dispatches splits
    // the work into separate jobs and costs about 100 us by itself.
    struct Timing { double gpu_us, wall_us; };
    auto measure = [&](uint32_t count) {
        VkCommandBuffer cb = g.begin();
        vkCmdResetQueryPool(cb, queries, 0, 2);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cb, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queries, 0);
        for (uint32_t r = 0; r < count; ++r) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &sets[r % copies], 0, nullptr);
            vkCmdDispatch(cb, groups_x, 1, groups_z);
            // ggml-vulkan separates dependent dispatches with a global memory barrier.
            if (barrier_kind == "memory") {
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &barrier, 0, nullptr, 0, nullptr);
            } else if (barrier_kind == "execution") {
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 0, nullptr, 0, nullptr, 0, nullptr);
            } else if (barrier_kind != "none") {
                throw std::runtime_error("--barrier must be memory, execution, or none");
            }
        }
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queries, 1);
        VkMemoryBarrier to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &to_host, 0, nullptr, 0, nullptr);
        const auto start = std::chrono::steady_clock::now();
        g.submit(cb);
        const double wall = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        uint64_t t[2];
        VK(vkGetQueryPoolResults(g.device, queries, 0, 2, sizeof(t), t, sizeof(uint64_t),
                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        return Timing{(t[1] - t[0]) * double(g.props.limits.timestampPeriod) / 1e3, wall};
    };
    // The GPU clock ramps for several milliseconds after idle, which dominates short runs.
    // After a warm-up, the marginal cost between a short and a five-times longer run is the
    // steady per-dispatch time.
    measure(reps);
    const Timing short_run = measure(reps), long_run = measure(5 * reps);
    const double marginal = (long_run.gpu_us - short_run.gpu_us) / (4.0 * reps);
    const double per_dispatch = long_run.gpu_us / (5.0 * reps);
    {
        VkCommandBuffer cb = g.begin();
        VkBufferCopy region{0, 0, m * sizeof(float)};
        vkCmdCopyBuffer(cb, d.handle, readback.handle, 1, &region);
        VkMemoryBarrier to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, nullptr, 0, nullptr);
        g.submit(cb);
    }
    const float* out = static_cast<const float*>(readback.mapped);
    if (!dump_path.empty()) {  // raw little-endian float32 outputs, for probes
        std::ofstream dump(dump_path, std::ios::binary);
        dump.write(reinterpret_cast<const char*>(out), m * sizeof(float));
        if (!dump) throw std::runtime_error("cannot write " + dump_path);
    }
    double err2 = 0, ref2 = 0, max_abs = 0, err2_f32 = 0, ref2_f32 = 0;
    bool finite = true;
    for (uint32_t r = 0; r < m; ++r) {
        if (!std::isfinite(out[r])) finite = false;
        const double e = double(out[r]) - reference[r];
        err2 += e * e; ref2 += reference[r] * reference[r];
        max_abs = std::max(max_abs, std::fabs(e));
        const double e32 = double(out[r]) - reference_f32[r];
        err2_f32 += e32 * e32; ref2_f32 += reference_f32[r] * reference_f32[r];
    }
    const double nmse = ref2 > 0 ? err2 / ref2 : err2;
    const double nmse_f32 = ref2_f32 > 0 ? err2_f32 / ref2_f32 : err2_f32;
    // JSON has no nan or inf: probes that do not compute the product report null.
    auto num = [](double v) { std::ostringstream o; o.precision(6); if (std::isfinite(v)) o << v; else o << "null"; return o.str(); };
    std::cout.precision(6);
    std::cout << "{\"label\":\"" << label << "\",\"type\":\"" << ggml_type_name(type) << "\",\"m\":" << m << ",\"k\":" << k
              << ",\"rows_per_workgroup\":" << rows << ",\"spec\":[" << argv[6] << "]"
              << ",\"groups\":[" << groups_x << ",1," << groups_z << "]"
              << ",\"subgroup\":" << subgroup << ",\"copies\":" << copies << ",\"reps\":" << reps
              << ",\"barrier\":\"" << barrier_kind << "\",\"io_memory\":\"" << (device_io ? "device" : "host") << "\""
              << ",\"marginal_us\":" << num(marginal) << ",\"us_per_dispatch\":" << num(per_dispatch)
              << ",\"wall_us_per_dispatch\":" << num(long_run.wall_us / (5.0 * reps))
              << ",\"weights_gb_s\":" << num(a_bytes / (marginal * 1e3))
              << ",\"b_format\":\"" << b_format << "\",\"nmse\":" << num(nmse) << ",\"nmse_vs_f32_activations\":" << num(nmse_f32)
              << ",\"max_abs\":" << num(max_abs)
              << ",\"finite\":" << (finite ? "true" : "false") << ",\"pass\":" << (finite && nmse < 1e-6 ? "true" : "false")
              << "}" << std::endl;

    vkDestroyQueryPool(g.device, queries, nullptr);
    vkDestroyDescriptorPool(g.device, pool, nullptr);
    vkDestroyPipeline(g.device, pipeline, nullptr);
    vkDestroyShaderModule(g.device, shader, nullptr);
    vkDestroyPipelineLayout(g.device, pipe_layout, nullptr);
    vkDestroyDescriptorSetLayout(g.device, layout, nullptr);
    for (auto& buffer : a) g.release(buffer);
    g.release(b); g.release(d); g.release(fuse); g.release(readback);
    vkDestroyCommandPool(g.device, g.pool, nullptr);
    vkDestroyDevice(g.device, nullptr);
    vkDestroyInstance(g.instance, nullptr);
    return finite && nmse < 1e-6 ? 0 : 3;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
