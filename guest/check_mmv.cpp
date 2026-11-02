// Numerical gate for a pinned ggml Vulkan runtime. Compile with the matching ggml headers.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static constexpr double MMV_NMSE_LIMIT = 5e-4;
static constexpr double RMS_NMSE_LIMIT = 1e-5;
enum class Operation { MatMul, RmsNorm, RmsNormMul };

static std::vector<float> values(size_t count, uint32_t seed) {
    std::vector<float> out(count);
    for (auto& value : out) {
        seed = seed * 1664525u + 1013904223u;
        value = (static_cast<int32_t>(seed >> 8) - 8388608) / 16777216.0f;
    }
    return out;
}

struct Graph {
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor* weights = nullptr;
    ggml_tensor* activations = nullptr;
    ggml_tensor* output = nullptr;
    ggml_cgraph* graph = nullptr;

    Graph(ggml_backend_t backend, ggml_type type, int64_t k, int64_t m, int64_t n, Operation operation = Operation::MatMul) {
        ggml_init_params init{8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true};
        context = ggml_init(init);
        if (!context) throw std::runtime_error("ggml context allocation failed");
        weights = ggml_new_tensor_2d(context, type, k, m);
        activations = ggml_new_tensor_2d(context, GGML_TYPE_F32, k, n);
        ggml_set_input(weights);
        ggml_set_input(activations);
        if (operation == Operation::MatMul) {
            output = ggml_mul_mat(context, weights, activations);
        } else {
            output = ggml_rms_norm(context, activations, 1e-5f);
            if (operation == Operation::RmsNormMul) output = ggml_mul(context, output, weights);
        }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(context, 16, false);
        ggml_build_forward_expand(graph, output);
        if (!ggml_backend_supports_op(backend, output)) {
            ggml_free(context);
            context = nullptr;
            throw std::runtime_error("backend does not support requested operation");
        }
        buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        if (!buffer) {
            ggml_free(context);
            context = nullptr;
            throw std::runtime_error("backend tensor allocation failed");
        }
    }
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    ~Graph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
};

static std::vector<float> compute(Graph& g, ggml_backend_t backend,
                                  const std::vector<uint8_t>& weights, const std::vector<float>& input) {
    ggml_backend_tensor_set(g.weights, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(g.activations, input.data(), 0, input.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, g.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("backend graph computation failed");
    }
    std::vector<float> output(ggml_nelements(g.output));
    ggml_backend_tensor_get(g.output, output.data(), 0, output.size() * sizeof(float));
    return output;
}

// The callers print dimensions first. This appends the common whole-output error gate.
static bool report_errors(const std::vector<float>& expected, const std::vector<float>& actual, double limit) {
    double squared_error = 0., energy = 0., max_abs = 0.;
    bool finite = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
            finite = false;
            continue;
        }
        const double diff = double(expected[i]) - actual[i];
        squared_error += diff * diff;
        energy += double(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(diff));
    }
    const double nmse = energy > 0 ? squared_error / energy : -1.;
    const bool pass = finite && energy > 0 && nmse <= limit;
    std::cout << ",\"elements\":" << actual.size()
              << ",\"max_abs_error\":" << max_abs << ",\"nmse\":" << nmse
              << ",\"nmse_limit\":" << limit
              << ",\"finite\":" << (finite ? "true" : "false")
              << ",\"status\":\"" << (pass ? "pass" : "fail") << "\"}" << std::endl;
    return pass;
}

static bool check_case(ggml_backend_t cpu, ggml_backend_t gpu, ggml_type type, int64_t k, int64_t m, int64_t n) {
    const auto weights = values(static_cast<size_t>(k*m), 73);
    const auto activations = values(static_cast<size_t>(k*n), 149);
    std::vector<uint8_t> quantized(ggml_row_size(type, k) * m);
    const size_t written = ggml_quantize_chunk(type, weights.data(), quantized.data(), 0, m, k, nullptr);
    if (written != quantized.size()) throw std::runtime_error("unexpected quantized weight size");
    Graph reference(cpu, type, k, m, n);
    Graph candidate(gpu, type, k, m, n);
    const auto expected = compute(reference, cpu, quantized, activations);
    const auto actual = compute(candidate, gpu, quantized, activations);
    std::cout << "{\"op\":\"MUL_MAT\",\"type\":\"" << ggml_type_name(type) << "\",\"k\":" << k
              << ",\"m\":" << m << ",\"n\":" << n;
    return report_errors(expected, actual, MMV_NMSE_LIMIT);
}

static bool check_rms(ggml_backend_t cpu, ggml_backend_t gpu, int64_t width, int64_t batch, Operation operation) {
    const auto activations = values(static_cast<size_t>(width*batch), 149);
    const auto gains = values(static_cast<size_t>(width), 73);
    std::vector<uint8_t> gain_bytes(gains.size() * sizeof(float));
    std::memcpy(gain_bytes.data(), gains.data(), gain_bytes.size());
    Graph reference(cpu, GGML_TYPE_F32, width, 1, batch, operation);
    Graph candidate(gpu, GGML_TYPE_F32, width, 1, batch, operation);
    const auto expected = compute(reference, cpu, gain_bytes, activations);
    const auto actual = compute(candidate, gpu, gain_bytes, activations);
    std::cout << "{\"op\":\"" << (operation == Operation::RmsNorm ? "RMS_NORM" : "RMS_NORM_MUL")
              << "\",\"type\":\"f32\",\"width\":" << width << ",\"batch\":" << batch;
    return report_errors(expected, actual, RMS_NMSE_LIMIT);
}

int main(int argc, char** argv) {
    ggml_backend_t cpu = nullptr, gpu = nullptr;
    try {
        if (argc != 2) throw std::runtime_error("expected runtime directory as the only argument");
        ggml_backend_load_all_from_path(argv[1]);
        const auto cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        const auto gpu_device = ggml_backend_dev_by_name("Vulkan0");
        if (!cpu_device || !gpu_device) throw std::runtime_error("CPU and Vulkan0 are both required");
        cpu = ggml_backend_dev_init(cpu_device, nullptr);
        gpu = ggml_backend_dev_init(gpu_device, nullptr);
        if (!cpu || !gpu) throw std::runtime_error("backend initialization failed");
        int cases = 0, failed = 0;
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K}) {
            for (int64_t k : {768, 896, 1024}) {
                if (k % ggml_blck_size(type)) continue;
                for (int64_t m : {128, 896}) {
                    for (int64_t n : {1, 2}) {
                        if (!check_case(cpu, gpu, type, k, m, n)) ++failed;
                        ++cases;
                    }
                }
            }
        }
        for (auto operation : {Operation::RmsNorm, Operation::RmsNormMul}) {
            for (int64_t width : {128, 256, 896, 1536}) {
                for (int64_t batch : {1, 2}) {
                    if (!check_rms(cpu, gpu, width, batch, operation)) ++failed;
                    ++cases;
                }
            }
        }
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        ggml_quantize_free();
        std::cout << "{\"status\":\"" << (failed ? "fail" : "pass")
                  << "\",\"cases\":" << cases << ",\"failed\":" << failed
                  << ",\"mmv_nmse_limit\":" << MMV_NMSE_LIMIT
                  << ",\"rms_nmse_limit\":" << RMS_NMSE_LIMIT << "}" << std::endl;
        return failed ? 1 : 0;
    } catch (const std::exception& error) {
        if (gpu) ggml_backend_free(gpu);
        if (cpu) ggml_backend_free(cpu);
        std::cerr << "check_mmv: " << error.what() << std::endl;
        return 2;
    }
}
