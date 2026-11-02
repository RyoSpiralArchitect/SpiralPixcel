// A transfer-only reproducer. No model, shader graph, or large allocation.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static int inspect(const std::vector<uint8_t>& expected, const uint8_t* actual,
                   size_t bytes, int trial, const char* route) {
    size_t mismatches = 0, zeros = 0, first = bytes;
    for (size_t i = 0; i < bytes; ++i) {
        zeros += actual[i] == 0;
        if (actual[i] != expected[i]) { ++mismatches; first = std::min(first, i); }
    }
    std::cout << "{\"route\":\"" << route << "\",\"bytes\":" << bytes
              << ",\"offset\":64,\"trial\":" << trial << ",\"mismatched_bytes\":" << mismatches
              << ",\"zero_bytes\":" << zeros << ",\"status\":\"" << (mismatches ? "fail" : "pass") << "\"";
    if (mismatches) std::cout << ",\"first_mismatch\":" << first << ",\"expected\":" << unsigned(expected[first])
                              << ",\"actual\":" << unsigned(actual[first]);
    std::cout << "}" << std::endl;
    return mismatches ? 1 : 0;
}

int main(int argc, char** argv) {
    ggml_backend_t gpu = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t device_buffer = nullptr, host_buffer = nullptr;
    try {
        if (argc != 2) throw std::runtime_error("expected RUNTIME");
        ggml_backend_load_all_from_path(argv[1]);
        auto dev = ggml_backend_dev_by_name("Vulkan0");
        if (!dev) throw std::runtime_error("Vulkan0 required");
        gpu = ggml_backend_dev_init(dev, nullptr);
        auto host_type = ggml_backend_dev_host_buffer_type(dev);
        if (!gpu || !host_type) throw std::runtime_error("Vulkan backend and its host buffer type required");
        int cases = 0, failed = 0;
        for (size_t bytes : {size_t(64), size_t(4096), size_t(1048576)}) {
            ggml_init_params init{2*ggml_tensor_overhead(), nullptr, true};
            ctx = ggml_init(init);
            if (!ctx) throw std::runtime_error("metadata allocation failed");
            auto tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, bytes + 128);
            device_buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu);
            host_buffer = ggml_backend_buft_alloc_buffer(host_type, bytes);
            if (!device_buffer || !host_buffer) throw std::runtime_error("bounded buffer allocation failed");
            auto pinned = static_cast<uint8_t*>(ggml_backend_buffer_get_base(host_buffer));
            if (!pinned) throw std::runtime_error("host buffer has no base pointer");
            for (int trial = 0; trial < 3; ++trial) {
                std::vector<uint8_t> payload(bytes+128, 0xa5), expected(bytes), normal(bytes), synchronous(bytes);
                for (size_t i = 0; i < bytes; ++i) {
                    expected[i] = uint8_t(((i*73 + (trial+1)*29) ^ (i >> 5)) & 255);
                }
                std::copy(expected.begin(), expected.end(), payload.begin()+64);
                ggml_backend_tensor_set(tensor, payload.data(), 0, payload.size());
                ggml_backend_tensor_get(tensor, synchronous.data(), 64, bytes);
                failed += inspect(expected, synchronous.data(), bytes, trial, "sync_normal"); ++cases;
                ggml_backend_tensor_get_async(gpu, tensor, normal.data(), 64, bytes);
                ggml_backend_synchronize(gpu);
                failed += inspect(expected, normal.data(), bytes, trial, "async_api_normal"); ++cases;
                std::memset(pinned, 0, bytes);
                ggml_backend_tensor_get_async(gpu, tensor, pinned, 64, bytes);
                ggml_backend_synchronize(gpu);
                failed += inspect(expected, pinned, bytes, trial, "async_api_vulkan_host"); ++cases;
            }
            ggml_backend_buffer_free(host_buffer); host_buffer = nullptr;
            ggml_backend_buffer_free(device_buffer); device_buffer = nullptr;
            ggml_free(ctx); ctx = nullptr;
        }
        std::cout << "{\"status\":\"" << (failed ? "fail" : "pass") << "\",\"cases\":" << cases
                  << ",\"failed\":" << failed << ",\"transfer_only\":true,\"max_payload_bytes\":1048576"
                  << ",\"host_buffer_requested\":\"Vulkan host type (allocation may fall back)\""
                  << ",\"split_output_env\":\"" << (std::getenv("GGML_VK_SPLIT_OUTPUT") ? std::getenv("GGML_VK_SPLIT_OUTPUT") : "unset") << "\""
                  << ",\"disable_async_env\":" << (std::getenv("GGML_VK_DISABLE_ASYNC") ? "true" : "false") << "}" << std::endl;
        ggml_backend_free(gpu);
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        if (host_buffer) ggml_backend_buffer_free(host_buffer);
        if (device_buffer) ggml_backend_buffer_free(device_buffer);
        if (ctx) ggml_free(ctx);
        if (gpu) ggml_backend_free(gpu);
        std::cerr << "check-readback: " << e.what() << std::endl;
        return 2;
    }
}
