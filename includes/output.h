#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include "config.h"
#include "perplexity.h"

struct ModelInfo {
    std::string architecture;
    std::string quantization;
    int64_t     n_params        = -1;
    int64_t     embedding_length = -1;
    int64_t     n_layers        = -1;
    int64_t     max_context     = -1;
};

struct RunMeta {
    Config      cfg;
    ModelInfo   model_info;
    std::string corpus_sha256;
    int         n_tokens_total  = 0;
    int         n_vocab         = 0;
    int64_t     ts_run_start_ns = 0;
    int64_t     ts_run_end_ns   = 0;
};

// Creates results/<timestamp_ns>/ directory and writes resumen.json + .jsonl.
// Returns the output directory path.
std::string write_results(const RunMeta& meta, const PerplexityResult& result);
