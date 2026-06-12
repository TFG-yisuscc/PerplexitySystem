#pragma once
#include "config.h"
#include "perplexity.h"
#include <vector>

struct llama_model;

// Runs teacher-forcing perplexity using Ollama's /api/generate endpoint.
// For each scored position in [n_ctx/2, n_ctx-2] per chunk, builds the
// detokenised prefix via model and searches the target token in top_logprobs.
// Requires model != nullptr (needed for llama_token_to_piece).
// Throws std::runtime_error on connection failure or unsupported Ollama version.
PerplexityResult run_ollama_perplexity(
    const Config&           cfg,
    const std::vector<int>& tokens,
    llama_model*            model
);
