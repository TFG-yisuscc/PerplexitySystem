#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

enum class InferenceEngine { LLAMA, OLLAMA };

struct Config {
    // ── shared ────────────────────────────────────────────────────────────────
    InferenceEngine engine       = InferenceEngine::LLAMA;
    std::string     corpus_path;               // empty → use embedded corpus
    int             context_size = 512;
    int             seed         = 0;
    nlohmann::json  annotations  = nullptr;
    std::string     output_dir   = "results";
    std::string     raw_json;                  // original JSON for provenance

    // ── LLAMA engine ──────────────────────────────────────────────────────────
    // model_path: single-model (required if model_paths empty).
    // model_paths: batch run — each entry is an independent GGUF; overrides model_path.
    std::string              model_path;
    std::vector<std::string> model_paths;
    int                      n_threads  = 0;   // 0 → hardware_concurrency()
    int                      batch_size = 0;   // 0 → context_size

    // ── OLLAMA engine ─────────────────────────────────────────────────────────
    // ollama_model_name  / ollama_model_names: analogous to model_path / model_paths.
    // model_path (above): GGUF used for tokenisation — shared across all ollama_model_names.
    std::string              ollama_model_name;
    std::vector<std::string> ollama_model_names;
    std::string              ollama_url        = "http://localhost:11434";
    int                      top_logprobs_k    = 20;
    int                      max_tokens_scored = 0;   // 0 = unlimited
    bool                     ollama_raw        = true;
};

// Parse and validate a JSON string; throws std::runtime_error on invalid input.
Config parse_config(const std::string& json_str);
