// Technical full-logit gate using the pinned public llama API.
// The fixed prompt SHA-256 below identifies these exact UTF-8 bytes.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// A token piece may end inside a UTF-8 sequence. Encode bytes without emitting
// malformed JSON text or turning a rendering issue into a numerical failure.
static void json_bytes_hex(const std::string& text) {
    std::cout << '"';
    for (const unsigned char c : text) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
    }
    std::cout << std::dec << '"';
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("expected RUNTIME MODEL OUTPUT.bin GPU_LAYERS");
        ggml_backend_load_all_from_path(argv[1]);
        const std::string layer_arg = argv[4];
        size_t used = 0;
        const int gpu_layers = std::stoi(layer_arg, &used);
        if (used != layer_arg.size() || gpu_layers < 0 || gpu_layers > 999) {
            throw std::runtime_error("GPU_LAYERS must be an integer from 0 to 999");
        }
        const auto device = gpu_layers ? ggml_backend_dev_by_name("Vulkan0") : nullptr;
        if (gpu_layers && !device) throw std::runtime_error("Vulkan0 is required");
        if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            throw std::runtime_error("CPU backend is required");
        }
        llama_backend_init();
        ggml_backend_dev_t devices[]{device, nullptr};
        auto model_params = llama_model_default_params();
        model_params.devices = devices;
        model_params.n_gpu_layers = gpu_layers;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.main_gpu = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], model_params), llama_model_free);
        if (!model) throw std::runtime_error("model loading failed");
        const llama_vocab* vocab = llama_model_get_vocab(model.get());
        const std::string prompt =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nWhich planet is known as the Red Planet? Answer in one word.<|im_end|>\n"
            "<|im_start|>assistant\n";
        const int32_t needed = llama_tokenize(vocab, prompt.data(), int32_t(prompt.size()), nullptr, 0, true, true);
        if (needed >= 0 || needed == INT32_MIN) throw std::runtime_error("invalid tokenizer size");
        std::vector<llama_token> tokens(size_t(-needed));
        const int32_t count = llama_tokenize(vocab, prompt.data(), int32_t(prompt.size()), tokens.data(), int32_t(tokens.size()), true, true);
        if (count <= 0 || count > 512) throw std::runtime_error("tokenization failed or context exceeded");
        tokens.resize(size_t(count));

        auto context_params = llama_context_default_params();
        context_params.n_ctx = 512;
        context_params.n_batch = 16;
        context_params.n_ubatch = 16;
        context_params.n_seq_max = 1;
        context_params.n_outputs_max = 1;
        context_params.n_threads = 4;
        context_params.n_threads_batch = 4;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.no_perf = true;
        context_params.offload_kqv = gpu_layers != 0;
        context_params.op_offload = gpu_layers != 0;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), llama_free);
        if (!context) throw std::runtime_error("context creation failed");
        llama_batch batch = llama_batch_init(16, 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            throw std::runtime_error("batch allocation failed");
        }
        int decode_error = 0;
        for (int32_t first = 0; first < count; first += 16) {
            batch.n_tokens = std::min<int32_t>(16, count - first);
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                batch.token[i] = tokens[size_t(first+i)];
                batch.pos[i] = first+i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = first+i+1 == count;
            }
            decode_error = llama_decode(context.get(), batch);
            if (decode_error) break;
        }
        llama_batch_free(batch);
        if (decode_error) throw std::runtime_error("llama_decode failed: " + std::to_string(decode_error));
        llama_synchronize(context.get());
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const float* logits = llama_get_logits_ith(context.get(), -1);
        if (!logits || n_vocab <= 0) throw std::runtime_error("no final logits");
        std::vector<float> values(logits, logits+n_vocab);
        const size_t nonzero = std::count_if(values.begin(), values.end(), [](float v) { return v != 0.f; });
        const size_t nonfinite = std::count_if(values.begin(), values.end(), [](float v) { return !std::isfinite(v); });
        const bool valid = nonzero != 0 && nonfinite == 0;
        const uint32_t endian_probe = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian_probe) != 1) {
            throw std::runtime_error("this gate writes little-endian float32 only");
        }
        static_assert(sizeof(float) == 4, "expected binary32 logits");
        std::ofstream out(argv[3], std::ios::binary);
        out.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size()*sizeof(float)));
        out.close();
        if (!out) throw std::runtime_error("logit output write failed");
        llama_token top_id = -1;
        float top_logit = 0;
        std::string top_piece;
        if (valid) {
            const auto top = std::max_element(values.begin(), values.end());
            top_id = llama_token(std::distance(values.begin(), top));
            top_logit = *top;
            std::vector<char> piece(256);
            int32_t piece_size = llama_token_to_piece(vocab, top_id, piece.data(), int32_t(piece.size()), 0, true);
            if (piece_size < 0) {
                piece.resize(size_t(-piece_size));
                piece_size = llama_token_to_piece(vocab, top_id, piece.data(), int32_t(piece.size()), 0, true);
            }
            if (piece_size < 0) throw std::runtime_error("top token rendering failed");
            top_piece.assign(piece.data(), size_t(piece_size));
        }
        std::cout << "{\"status\":\"" << (valid ? "pass" : "fail")
                  << "\",\"technical_validation_only\":true,\"format\":\"float32-le\""
                  << ",\"gpu_layers\":" << gpu_layers
                  << ",\"prompt_sha256\":\"3ae35bc99e935fea913bbae40ac007931405b8242c50f63bd06b32aac923a762\""
                  << ",\"prompt_tokens\":" << count
                  << ",\"logit_count\":" << n_vocab
                  << ",\"nonzero_count\":" << nonzero
                  << ",\"nonfinite_count\":" << nonfinite
                  << ",\"finite\":" << (nonfinite == 0 ? "true" : "false");
        if (valid) {
            std::cout << ",\"top1_token\":" << top_id
                      << ",\"top1_logit\":" << std::setprecision(9) << top_logit << ",\"top1_piece_hex\":";
            json_bytes_hex(top_piece);
        } else {
            std::cout << ",\"top1_token\":null,\"top1_logit\":null,\"top1_piece_hex\":null";
        }
        std::cout << "}" << std::endl;
        return valid ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "check_model: " << error.what() << std::endl;
        std::cout << "{\"status\":\"fail\",\"error_bytes_hex\":";
        json_bytes_hex(error.what());
        std::cout << "}" << std::endl;
        return 1;
    }
}
