// A headless Vulkan 1.0 compute dispatch. The caller supplies two-buffer GLSL.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void check(VkResult r, const char* operation) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": VkResult " + std::to_string(r));
}
#define VK(call) check((call), #call)

static std::vector<uint32_t> read_words(const char* name) {
    std::ifstream in(name, std::ios::binary | std::ios::ate);
    const auto size = in.tellg();
    if (size <= 0 || size % 4) throw std::runtime_error(std::string("invalid input: ") + name);
    std::vector<uint32_t> words(size / 4);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(words.data()), size)) throw std::runtime_error("short read");
    return words;
}

struct Buffer {
    VkBuffer handle{};
    VkDeviceMemory memory{};
    void* mapped{};
    bool coherent{};
};
static Buffer allocate(VkDevice dev, VkPhysicalDevice gpu, VkDeviceSize bytes) {
    Buffer b;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VK(vkCreateBuffer(dev, &info, nullptr, &b.handle));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b.handle, &req);
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(gpu, &props);
    uint32_t chosen = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        auto flags = props.memoryTypes[i].propertyFlags;
        if (!(req.memoryTypeBits & (1u << i)) || !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        chosen = i;
        if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) { b.coherent = true; break; }
    }
    if (chosen == UINT32_MAX) throw std::runtime_error("no host-visible storage memory");
    VkMemoryAllocateInfo mem{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mem.allocationSize = req.size;
    mem.memoryTypeIndex = chosen;
    VK(vkAllocateMemory(dev, &mem, nullptr, &b.memory));
    VK(vkBindBufferMemory(dev, b.handle, b.memory, 0));
    VK(vkMapMemory(dev, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    return b;
}

struct Params { uint32_t width, height; float rate, decay; uint32_t tick, drive_ticks; };
static_assert(sizeof(Params) == 24);

static int run(char** a) {
    Params params{static_cast<uint32_t>(std::stoul(a[4])), static_cast<uint32_t>(std::stoul(a[5])),
                        std::stof(a[7]), std::stof(a[8]), 0, 0};
    const auto steps = std::stoul(a[6]);
    const auto input = read_words(a[2]);
    const auto code = read_words(a[1]);
    if (!steps || steps > 1225 || !params.width || !params.height ||
        uint64_t(params.width) * params.height != input.size()) throw std::runtime_error("invalid grid or steps");
    const VkDeviceSize bytes = input.size() * 4;
    std::vector<uint32_t> masks;
    if (a[10]) {
        masks = read_words(a[10]);
        const size_t stride = (input.size() + 31) / 32;
        if (masks.size() % stride || masks.size() / stride > 201) throw std::runtime_error("invalid touch mask");
        params.drive_ticks = static_cast<uint32_t>(masks.size() / stride);
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "SpiralPixcel"; app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    VkInstance instance;
    VK(vkCreateInstance(&ii, nullptr, &instance));
    uint32_t count = 0;
    VK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    if (count) VK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    VkPhysicalDevice gpu{};
    VkPhysicalDeviceProperties props{};
    uint32_t family = 0, timestamp_bits = 0;
    bool allow_cpu = std::string(a[9]) == "allow-software";
    for (auto candidate : devices) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(candidate, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !allow_cpu) continue;
        uint32_t n = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, nullptr);
        std::vector<VkQueueFamilyProperties> queues(n);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, queues.data());
        for (uint32_t i = 0; i < n; ++i) {
            if (!queues[i].queueCount || !(queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
            gpu = candidate; props = p; family = i; timestamp_bits = queues[i].timestampValidBits; break;
        }
        if (gpu) break;
    }
    if (!gpu) throw std::runtime_error("no hardware Vulkan compute queue (check VM GPU acceleration)");
    std::cout << "device\t" << props.deviceName << "\napi\t" << VK_VERSION_MAJOR(props.apiVersion)
              << '.' << VK_VERSION_MINOR(props.apiVersion) << '.' << VK_VERSION_PATCH(props.apiVersion)
              << "\ndriver_version\t" << props.driverVersion << "\nqueue_family\t" << family
              << "\ndevice_type\t" << props.deviceType << "\ntimestamp_valid_bits\t" << timestamp_bits << '\n';
    VK(vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, nullptr));
    std::vector<VkExtensionProperties> extensions(count);
    if (count) VK(vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, extensions.data()));
    for (const auto& e : extensions) std::cout << "extension\t" << e.extensionName << '\n';

    const float priority = 1;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    VkDevice dev;
    VK(vkCreateDevice(gpu, &di, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, family, 0, &queue);
    std::vector<Buffer> buffers{allocate(dev, gpu, bytes), allocate(dev, gpu, bytes)};
    if (!masks.empty()) {
        buffers.push_back(allocate(dev, gpu, masks.size()*4));
        std::memcpy(buffers[2].mapped, masks.data(), masks.size()*4);
    }
    std::memcpy(buffers[0].mapped, input.data(), bytes);
    auto range = [](const Buffer& b) {
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        r.memory = b.memory; r.size = VK_WHOLE_SIZE; return r;
    };
    for (const auto& b : buffers) {
        if (!b.coherent) { auto r = range(b); VK(vkFlushMappedMemoryRanges(dev, 1, &r)); }
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(buffers.size());
    for (uint32_t i = 0; i < bindings.size(); ++i)
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = static_cast<uint32_t>(bindings.size()); li.pBindings = bindings.data();
    VkDescriptorSetLayout layout;
    VK(vkCreateDescriptorSetLayout(dev, &li, nullptr, &layout));
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params)};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1; pi.pSetLayouts = &layout; pi.pushConstantRangeCount = 1; pi.pPushConstantRanges = &push;
    VkPipelineLayout pipe_layout;
    VK(vkCreatePipelineLayout(dev, &pi, nullptr, &pipe_layout));
    VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    si.codeSize = code.size() * 4; si.pCode = code.data();
    VkShaderModule shader;
    VK(vkCreateShaderModule(dev, &si, nullptr, &shader));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = shader; ci.stage.pName = "main";
    ci.layout = pipe_layout;
    VkPipeline pipeline;
    VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline));
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * li.bindingCount};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 2; dpi.poolSizeCount = 1; dpi.pPoolSizes = &pool_size;
    VkDescriptorPool pool;
    VK(vkCreateDescriptorPool(dev, &dpi, nullptr, &pool));
    VkDescriptorSetLayout layouts[]{layout, layout};
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool; ai.descriptorSetCount = 2; ai.pSetLayouts = layouts;
    VkDescriptorSet sets[2];
    VK(vkAllocateDescriptorSets(dev, &ai, sets));
    for (unsigned j = 0; j < 2; ++j) {
        for (unsigned k = 0; k < bindings.size(); ++k) {
            const auto index = k < 2 ? j ^ k : k;
            VkDescriptorBufferInfo bi{buffers[index].handle, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = sets[j]; w.dstBinding = k; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
            vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
        }
    }

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = family;
    VkCommandPool commands;
    VK(vkCreateCommandPool(dev, &cpi, nullptr, &commands));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = commands; cai.commandBufferCount = 1; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VkCommandBuffer cb;
    VK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkQueryPool queries{};
    if (timestamp_bits) {
        VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP; qp.queryCount = 2;
        VK(vkCreateQueryPool(dev, &qp, nullptr, &queries));
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK(vkBeginCommandBuffer(cb, &begin));
    if (queries) {
        vkCmdResetQueryPool(cb, queries, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cb, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
    for (unsigned i = 0; i < steps; ++i) {
        if (params.drive_ticks) {
            vkCmdPushConstants(cb, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, offsetof(Params, tick), sizeof(i), &i);
        }
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &sets[i % 2], 0, nullptr);
        vkCmdDispatch(cb, (params.width + 7) / 8, (params.height + 7) / 8, 1);
        const bool last = i + 1 == steps;
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = last ? VK_ACCESS_HOST_READ_BIT : VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             last ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    if (queries) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
    VK(vkEndCommandBuffer(cb));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &cb;
    auto start = std::chrono::steady_clock::now();
    VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    VK(vkQueueWaitIdle(queue));
    std::cout << "submit_wait_ms\t" << std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() << '\n';
    if (queries) {
        uint64_t t[2];
        VK(vkGetQueryPoolResults(dev, queries, 0, 2, sizeof(t), t, sizeof(*t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        const uint64_t mask = std::numeric_limits<uint64_t>::max() >> (64 - timestamp_bits);
        std::cout << "gpu_ms\t" << ((t[1] - t[0]) & mask) * double(props.limits.timestampPeriod) / 1e6 << '\n';
        vkDestroyQueryPool(dev, queries, nullptr);
    }
    Buffer& output = buffers[steps % 2];
    if (!output.coherent) { auto r = range(output); VK(vkInvalidateMappedMemoryRanges(dev, 1, &r)); }
    std::ofstream file(a[3], std::ios::binary);
    file.write(static_cast<const char*>(output.mapped), bytes);
    file.close();
    if (!file) throw std::runtime_error("write output failed");
    vkDestroyCommandPool(dev, commands, nullptr); vkDestroyDescriptorPool(dev, pool, nullptr);
    vkDestroyPipeline(dev, pipeline, nullptr); vkDestroyShaderModule(dev, shader, nullptr);
    vkDestroyPipelineLayout(dev, pipe_layout, nullptr); vkDestroyDescriptorSetLayout(dev, layout, nullptr);
    for (const auto& b : buffers) {
        vkUnmapMemory(dev, b.memory); vkDestroyBuffer(dev, b.handle, nullptr); vkFreeMemory(dev, b.memory, nullptr);
    }
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(instance, nullptr);
    return 0;
}
int main(int argc, char** argv) {
    try {
        if (argc != 10 && argc != 11) throw std::runtime_error("expected: shader input output width height steps rate decay hardware|allow-software [touch-mask]");
        return run(argv);
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
