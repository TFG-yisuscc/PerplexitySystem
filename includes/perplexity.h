#pragma once
#include <cstdint>
#include <vector>
#include <string>

struct llama_context;

struct PerplexityResult {
    double ppl             = 0.0;
    double nll_mean        = 0.0;   // mean cross-entropy in nats
    double ppl_stderr      = -1.0;  // <0 means not computed (→ null in JSON)
    int    n_tokens_scored = 0;
    int    n_chunks        = 0;
    bool   add_bos         = false;
    std::vector<double> per_chunk_ppl;   // running PPL after each chunk

    // OLLAMA engine only (always 0 / 100.0 for LLAMA)
    int    n_tokens_missing = 0;
    double coverage_pct     = 100.0;
};

// Runs teacher-forcing perplexity via direct llama_decode (LLAMA engine).
// batch_size must be >= context_size (one decode per chunk, no logit buffering).
// Throws std::runtime_error on invalid parameters.
PerplexityResult run_perplexity(
    llama_context*          ctx,
    const std::vector<int>& tokens,
    int context_size,
    int batch_size,
    int max_tokens_scored = 0   // 0 = score full corpus
);
