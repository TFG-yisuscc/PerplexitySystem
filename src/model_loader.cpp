// Reference convention: llama.cpp b9030, commit e7f1e8d5e1
// (tools/perplexity/perplexity.cpp, function perplexity())

#include "model_loader.h"

#include <llama.h>
#include <fmt/core.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <cstring>
#include <sstream>
#include <algorithm>

// ── logging ──────────────────────────────────────────────────────────────────

static void log_callback(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fmt::print(stderr, "[llama.cpp ERROR] {}", text);
    }
    // suppress INFO / WARN / DEBUG to keep output clean
}

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string meta_str(const llama_model* model, const char* key) {
    char buf[256] = {};
    int rc = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    return rc >= 0 ? std::string(buf) : "";
}

static ModelInfo extract_model_info(const llama_model* model) {
    ModelInfo info;

    // Description string, e.g. "llama 7B Q4_K_M"
    char desc[512] = {};
    llama_model_desc(model, desc, sizeof(desc));
    const std::string desc_str(desc);

    // Architecture from GGUF metadata; fall back to first word of desc.
    info.architecture = meta_str(model, "general.architecture");
    if (info.architecture.empty() && !desc_str.empty()) {
        info.architecture = desc_str.substr(0, desc_str.find(' '));
    }

    // Quantization: everything after the first two tokens (arch + model-size).
    // desc format: "<arch> <size> <ftype_name>" e.g. "qwen2 1.5B Q4_K - Medium"
    // → info.quantization = "Q4_K - Medium"
    {
        std::string d = desc_str;
        auto skip_token = [&d]() {
            auto pos = d.find(' ');
            d = (pos != std::string::npos) ? d.substr(pos + 1) : "";
        };
        skip_token(); // skip arch
        skip_token(); // skip model-size
        info.quantization = d.empty() ? desc_str : d;
    }

    info.n_params         = static_cast<int64_t>(llama_model_n_params(model));
    info.embedding_length = llama_model_n_embd(model);
    info.n_layers         = llama_model_n_layer(model);
    info.max_context      = llama_model_n_ctx_train(model);

    return info;
}

// ── public API ────────────────────────────────────────────────────────────────

LoadedModel load_model(const Config& cfg) {
    // Silence llama.cpp info/debug logs; surface only errors.
    llama_log_set(log_callback, nullptr);

    llama_backend_init();

    // Model params: CPU-only (n_gpu_layers=0 for Pi compatibility).
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model* model = llama_model_load_from_file(cfg.model_path.c_str(), mparams);
    if (!model)
        throw std::runtime_error("failed to load model: " + cfg.model_path);

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int n_vocab  = llama_vocab_n_tokens(vocab);

    // Context params.
    int n_threads = cfg.n_threads > 0
                    ? cfg.n_threads
                    : static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads < 1) n_threads = 1;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = static_cast<uint32_t>(cfg.context_size);
    cparams.n_batch        = static_cast<uint32_t>(cfg.batch_size);
    cparams.n_ubatch       = static_cast<uint32_t>(cfg.batch_size);
    cparams.n_threads      = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.no_perf        = true;  // don't collect perf timings

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("failed to create llama context");
    }

    ModelInfo info = extract_model_info(model);

    return LoadedModel{model, ctx, info, add_bos, n_vocab};
}

void free_loaded_model(LoadedModel& m) {
    if (m.ctx)   { llama_free(m.ctx);        m.ctx   = nullptr; }
    if (m.model) { llama_model_free(m.model); m.model = nullptr; }
    llama_backend_free();
}
