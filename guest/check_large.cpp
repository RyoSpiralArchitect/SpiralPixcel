// A >128 MiB Q8_0 weight tested against an unsplit CPU MUL_MAT.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

static std::vector<float> values(int64_t count, uint32_t seed) {
    std::vector<float> result(count);
    for (float& value : result) {
        seed = seed * 1664525u + 1013904223u;
        value = (int32_t(seed >> 8) - 8388608) / 16777216.f;
    }
    return result;
}

struct Graph {
    ggml_context* context = nullptr;
    ggml_context* weight_context = nullptr;
    ggml_backend_buffer_t memory = nullptr;
    ggml_backend_buffer_t weight_memory = nullptr;
    ggml_tensor* weights = nullptr;
    ggml_tensor* input = nullptr;
    ggml_tensor* output = nullptr;
    ggml_cgraph* graph = nullptr;

    Graph(ggml_backend_t backend, int64_t k, int64_t m, int64_t n, int64_t chunk_rows) {
        ggml_init_params params{32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true};
        context = ggml_init(params);
        if (!context) throw std::runtime_error("metadata allocation failed");
        // Model weights and compute tensors use separate buffers in llama.cpp.
        ggml_init_params weight_params{4 * ggml_tensor_overhead(), nullptr, true};
        weight_context = ggml_init(weight_params);
        if (!weight_context) throw std::runtime_error("weight metadata allocation failed");
        weights = ggml_new_tensor_2d(weight_context, GGML_TYPE_Q8_0, k, m);
        input = ggml_new_tensor_2d(context, GGML_TYPE_F32, k, n);
        ggml_set_input(weights);
        ggml_set_input(input);
        if (!chunk_rows) {
            output = ggml_mul_mat(context, weights, input);
        } else {
            for (int64_t first = 0; first < m; first += chunk_rows) {
                const int64_t count = std::min(chunk_rows, m - first);
                auto view = ggml_view_2d(context, weights, k, count, weights->nb[1], first * weights->nb[1]);
                auto part = ggml_mul_mat(context, view, input);
                output = output ? ggml_concat(context, output, part, 0) : part;
            }
        }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(context, 64, false);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                std::cerr << "unsupported " << ggml_op_name(node->op) << " bytes=" << ggml_nbytes(node) << '\n';
                ggml_free(context);
                context = nullptr;
                throw std::runtime_error("backend rejected graph node");
            }
        }
        weight_memory = ggml_backend_alloc_ctx_tensors(weight_context, backend);
        if (!weight_memory) throw std::runtime_error("weight tensor allocation failed");
        memory = ggml_backend_alloc_ctx_tensors(context, backend);
        if (!memory) {
            ggml_free(context);
            context = nullptr;
            throw std::runtime_error("tensor allocation failed");
        }
    }
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    ~Graph() {
        if (memory) ggml_backend_buffer_free(memory);
        if (weight_memory) ggml_backend_buffer_free(weight_memory);
        if (context) ggml_free(context);
        if (weight_context) ggml_free(weight_context);
    }
};

struct TransferCheck {
    size_t probes = 0;
    size_t mismatched_weight_bytes = 0;
    size_t mismatched_input_bytes = 0;
};

static void check_transfer(Graph& graph, const std::vector<uint8_t>& weights,
                           const std::vector<float>& input, TransferCheck& check) {
    std::vector<size_t> offsets{0, 16u*1024u*1024u-32, 16u*1024u*1024u,
                                64u*1024u*1024u, 128u*1024u*1024u, weights.size()-64};
    for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
        auto node = ggml_graph_node(graph.graph, i);
        if (node->op == GGML_OP_VIEW && node->view_src == graph.weights) {
            offsets.push_back(node->view_offs);
            if (node->view_offs >= 64) offsets.push_back(node->view_offs - 64);
        }
    }
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    // Read distinct regions in both directions. A staging copy that trails one
    // request behind must fail this gate even when some repeated reads agree.
    for (int direction = 0; direction < 2; ++direction) {
        for (size_t offset : offsets) {
            if (offset + 64 > weights.size()) continue;
            uint8_t actual[64];
            ggml_backend_tensor_get(graph.weights, actual, offset, sizeof(actual));
            ++check.probes;
            for (size_t i = 0; i < sizeof(actual); ++i) {
                check.mismatched_weight_bytes += actual[i] != weights[offset+i];
            }
        }
        std::reverse(offsets.begin(), offsets.end());
    }
    std::vector<uint8_t> actual_input(input.size()*sizeof(float));
    const auto expected_input = reinterpret_cast<const uint8_t*>(input.data());
    for (int repeat = 0; repeat < 2; ++repeat) {
        ggml_backend_tensor_get(graph.input, actual_input.data(), 0, actual_input.size());
        ++check.probes;
        for (size_t i = 0; i < actual_input.size(); ++i) {
            check.mismatched_input_bytes += actual_input[i] != expected_input[i];
        }
    }
}

static std::vector<float> compute(Graph& graph, ggml_backend_t backend,
                                  const std::vector<uint8_t>& weights, const std::vector<float>& input,
                                  TransferCheck* transfer = nullptr) {
    // Exercise the public whole-tensor upload as well as the split compute graph.
    // The candidate backend must bound its coherent staging allocation internally.
    ggml_backend_tensor_set(graph.weights, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(graph.input, input.data(), 0, input.size() * sizeof(float));
    if (transfer) check_transfer(graph, weights, input, *transfer);
    if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("compute failed");
    }
    std::vector<float> output(ggml_nelements(graph.output));
    ggml_backend_tensor_get(graph.output, output.data(), 0, output.size() * sizeof(float));
    return output;
}

int main(int argc, char** argv) {
    ggml_backend_t cpu = nullptr;
    ggml_backend_t gpu = nullptr;
    try {
        if (argc != 2) throw std::runtime_error("expected runtime directory");
        ggml_backend_load_all_from_path(argv[1]);
        const auto cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        const auto gpu_device = ggml_backend_dev_by_name("Vulkan0");
        if (!cpu_device || !gpu_device) throw std::runtime_error("CPU and Vulkan0 are required");
        cpu = ggml_backend_dev_init(cpu_device, nullptr);
        gpu = ggml_backend_dev_init(gpu_device, nullptr);
        if (!cpu || !gpu) throw std::runtime_error("backend initialization failed");

        constexpr int64_t k = 896;
        constexpr int64_t m = 151937; // An odd final row exercises the chunk tail.
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, k);
        std::vector<uint8_t> weights(row_bytes * m);
        for (int64_t row = 0; row < m; ++row) {
            // Generate one row at a time, avoiding a >500 MiB floating-point copy.
            const auto source = values(k, 73 + uint32_t(row % 127));
            const size_t written = ggml_quantize_chunk(GGML_TYPE_Q8_0, source.data(),
                weights.data() + row * row_bytes, 0, 1, k, nullptr);
            if (written != row_bytes) throw std::runtime_error("quantized row size differs");
        }
        if (weights.size() <= 128u * 1024 * 1024) {
            throw std::runtime_error("test does not cross the 128 MiB range");
        }
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(gpu));
        const int64_t aligned_rows = alignment / std::gcd(row_bytes, alignment);
        const int64_t row_multiple = std::lcm<int64_t>(256, aligned_rows);
        int64_t chunk_rows = std::min<int64_t>(65536, 64u * 1024 * 1024 / row_bytes);
        chunk_rows -= chunk_rows % row_multiple;
        if (!chunk_rows) throw std::runtime_error("chunk budget cannot satisfy alignment");

        int cases = 0;
        int failed = 0;
        for (int64_t n : {1, 16}) {
            const int64_t basis_indices[16] = {0, 1, 7, 31, 32, 63, 127, 128,
                                               255, 256, 511, 512, 767, 768, 894, 895};
            std::vector<float> input(k * n, 0.f);
            for (int64_t column = 0; column < n; ++column) input[column*k + basis_indices[column]] = 1.f;
            std::vector<float> cpu_output;
            {
                Graph reference(cpu, k, m, n, 0);
                cpu_output = compute(reference, cpu, weights, input);
            }
            std::vector<float> expected(m*n);
            std::vector<float> decoded(k);
            const auto traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
            for (int64_t row = 0; row < m; ++row) {
                traits->to_float(weights.data() + row*row_bytes, decoded.data(), k);
                for (int64_t column = 0; column < n; ++column) {
                    expected[column*m + row] = decoded[basis_indices[column]];
                }
            }
            double cpu_max_abs_vs_basis = 0;
            for (size_t i = 0; i < expected.size(); ++i) {
                cpu_max_abs_vs_basis = std::max(cpu_max_abs_vs_basis, std::abs(double(cpu_output[i]) - expected[i]));
            }
            std::vector<float> actual;
            TransferCheck transfer;
            {
                Graph candidate(gpu, k, m, n, chunk_rows);
                actual = compute(candidate, gpu, weights, input, &transfer);
            }
            double squared_error = 0;
            double energy = 0;
            double max_abs = 0;
            double boundary_max_abs = 0;
            constexpr double absolute_tolerance = 1e-3;
            constexpr double relative_tolerance = 1e-3;
            size_t mismatched_elements = 0;
            size_t boundary_elements = 0;
            bool finite = true;
            for (size_t i = 0; i < expected.size(); ++i) {
                const int64_t row = i % m;
                const int64_t chunk_row = row % chunk_rows;
                const bool boundary = row == 0 || row + 1 == m ||
                                      chunk_row <= 1 || chunk_row + 1 == chunk_rows;
                if (boundary) ++boundary_elements;
                if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
                    finite = false;
                    ++mismatched_elements;
                    continue;
                }
                const double delta = double(expected[i]) - actual[i];
                const double absolute_error = std::abs(delta);
                const double tolerance = absolute_tolerance + relative_tolerance * std::abs(expected[i]);
                if (absolute_error > tolerance) ++mismatched_elements;
                squared_error += delta * delta;
                energy += double(expected[i]) * expected[i];
                max_abs = std::max(max_abs, absolute_error);
                if (boundary) boundary_max_abs = std::max(boundary_max_abs, absolute_error);
            }
            const double nmse = energy > 0 ? squared_error / energy : -1;
            const bool pass = finite && energy > 0 && nmse <= 5e-4 && mismatched_elements == 0 &&
                              transfer.mismatched_weight_bytes == 0 && transfer.mismatched_input_bytes == 0;
            ++cases;
            if (!pass) ++failed;
            std::cout << "{\"op\":\"SPLIT_MUL_MAT_CONCAT_BASIS\",\"type\":\"q8_0\",\"k\":" << k
                      << ",\"m\":" << m << ",\"n\":" << n << ",\"weight_bytes\":" << weights.size()
                      << ",\"chunk_rows\":" << chunk_rows
                      << ",\"reference\":\"dequantized_q8_0_basis_values\""
                      << ",\"separate_weight_buffer\":true,\"cpu_unsplit_max_abs_vs_basis\":" << cpu_max_abs_vs_basis
                      << ",\"max_abs_error\":" << max_abs
                      << ",\"nmse\":" << nmse << ",\"nmse_limit\":0.0005"
                      << ",\"absolute_tolerance\":" << absolute_tolerance
                      << ",\"relative_tolerance\":" << relative_tolerance
                      << ",\"mismatched_elements\":" << mismatched_elements
                      << ",\"boundary_elements\":" << boundary_elements
                      << ",\"boundary_max_abs_error\":" << boundary_max_abs
                      << ",\"readback_probes\":" << transfer.probes
                      << ",\"mismatched_weight_bytes\":" << transfer.mismatched_weight_bytes
                      << ",\"mismatched_input_bytes\":" << transfer.mismatched_input_bytes
                      << ",\"finite\":" << (finite ? "true" : "false")
                      << ",\"status\":\"" << (pass ? "pass" : "fail") << "\"}" << std::endl;
        }
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        ggml_quantize_free();
        std::cout << "{\"status\":\"" << (failed ? "fail" : "pass") << "\",\"cases\":" << cases
                  << ",\"failed\":" << failed << "}" << std::endl;
        return failed ? 1 : 0;
    } catch (const std::exception& error) {
        if (gpu) ggml_backend_free(gpu);
        if (cpu) ggml_backend_free(cpu);
        std::cerr << error.what() << '\n';
        return 2;
    }
}
