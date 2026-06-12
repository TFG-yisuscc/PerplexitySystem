#include "output.h"

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using json   = nlohmann::json;

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string ns_to_iso(int64_t ns) {
    const int64_t sec = ns / 1'000'000'000LL;
    const int64_t rem = ns % 1'000'000'000LL;
    std::time_t tt = static_cast<std::time_t>(sec);
    std::tm gmt{};
    gmtime_r(&tt, &gmt);
    std::ostringstream ss;
    ss << std::put_time(&gmt, "%Y-%m-%dT%H:%M:%S");
    ss << '.' << std::setw(9) << std::setfill('0') << rem << 'Z';
    return ss.str();
}

// Replace characters that are illegal in filenames.
static std::string sanitize_for_filename(std::string s) {
    for (char& c : s)
        if (c == '/' || c == '\\' || c == ':' || c == ' ' || c == '\t'
            || c == '*' || c == '?' || c == '"' || c == '<' || c == '>')
            c = '_';
    return s;
}

// Short model identifier used in the JSONL filename.
static std::string model_id(const RunMeta& m) {
    if (m.cfg.engine == InferenceEngine::OLLAMA)
        return sanitize_for_filename(m.cfg.ollama_model_name);
    // LLAMA: basename of model_path without extension.
    return sanitize_for_filename(fs::path(m.cfg.model_path).stem().string());
}

// ── public API ────────────────────────────────────────────────────────────────

std::string write_results(const RunMeta& meta, const PerplexityResult& result) {
    const std::string run_id    = std::to_string(meta.ts_run_start_ns);
    const bool        is_ollama = (meta.cfg.engine == InferenceEngine::OLLAMA);
    const double      duration  =
        (meta.ts_run_end_ns - meta.ts_run_start_ns) * 1e-9;

    // Create output directory: <output_dir>/<run_id>/
    const fs::path out_dir = fs::path(meta.cfg.output_dir) / run_id;
    try {
        fs::create_directories(out_dir);
    } catch (const fs::filesystem_error& e) {
        throw std::runtime_error(
            std::string("cannot create output directory: ") + e.what());
    }

    // ── build model object ────────────────────────────────────────────────────
    json model_obj = json::object();
    if (!meta.cfg.model_path.empty()) {
        model_obj = {
            {"path",             meta.cfg.model_path},
            {"architecture",     meta.model_info.architecture},
            {"quantization",     meta.model_info.quantization},
            {"n_params",         meta.model_info.n_params},
            {"embedding_length", meta.model_info.embedding_length},
            {"n_layers",         meta.model_info.n_layers},
            {"max_context",      meta.model_info.max_context},
            {"n_vocab",          meta.n_vocab}
        };
    }

    // ── build ollama_meta object ──────────────────────────────────────────────
    const json ollama_meta = is_ollama
        ? json{
            {"ollama_model_name", meta.cfg.ollama_model_name},
            {"ollama_url",        meta.cfg.ollama_url},
            {"top_logprobs_k",    meta.cfg.top_logprobs_k},
            {"ollama_raw",        meta.cfg.ollama_raw}
          }
        : json(nullptr);

    // ── build result object ───────────────────────────────────────────────────
    const json result_obj = {
        {"perplexity",       result.ppl},
        {"nll_mean",         result.nll_mean},
        {"ppl_stderr",       result.ppl_stderr >= 0.0
                                 ? json(result.ppl_stderr)
                                 : json(nullptr)},
        {"n_tokens_scored",  result.n_tokens_scored},
        {"n_chunks",         result.n_chunks},
        {"add_bos",          result.add_bos},
        {"n_tokens_missing", result.n_tokens_missing},
        {"coverage_pct",     result.coverage_pct},
        {"per_chunk_ppl",    result.per_chunk_ppl}
    };

    // ── resumen.json ──────────────────────────────────────────────────────────
    json resumen;
    resumen["run_id"]       = run_id;
    resumen["ts_start_ns"]  = meta.ts_run_start_ns;
    resumen["ts_end_ns"]    = meta.ts_run_end_ns;
    resumen["ts_start_iso"] = ns_to_iso(meta.ts_run_start_ns);
    resumen["ts_end_iso"]   = ns_to_iso(meta.ts_run_end_ns);
    resumen["duration_sec"] = duration;
    resumen["metric"]       = "wikitext2_teacher_forcing_perplexity";
    resumen["convention"]   =
        "llama.cpp b9030 non-overlapping chunks of n_ctx, "
        "score positions [n_ctx/2, n_ctx-2]";
    resumen["engine"]       = is_ollama ? "OLLAMA" : "LLAMA";
    resumen["corpus"] = {
        {"path",            meta.cfg.corpus_path},
        {"sha256",          meta.corpus_sha256},
        {"n_tokens_total",  meta.n_tokens_total}
    };
    resumen["model"]       = model_obj;
    resumen["ollama_meta"] = ollama_meta;
    resumen["result"]      = result_obj;
    resumen["annotations"] = meta.cfg.annotations;
    try {
        resumen["config"]  = json::parse(meta.cfg.raw_json);
    } catch (...) {
        resumen["config"]  = meta.cfg.raw_json;  // fallback: raw string
    }

    const fs::path resumen_path = out_dir / "resumen.json";
    {
        std::ofstream f(resumen_path);
        if (!f) throw std::runtime_error(
            "cannot write " + resumen_path.string());
        f << resumen.dump(2) << '\n';
    }

    // ── <run_id>_perplexity_<model>.jsonl ─────────────────────────────────────
    const std::string mid       = model_id(meta);
    const std::string jsonl_name = run_id + "_perplexity_" + mid + ".jsonl";
    const fs::path    jsonl_path = out_dir / jsonl_name;

    json record;
    record["run_id"]           = run_id;
    record["ts_ns"]            = meta.ts_run_start_ns;
    record["ts_iso"]           = ns_to_iso(meta.ts_run_start_ns);
    record["metric"]           = "wikitext2_teacher_forcing_perplexity";
    record["engine"]           = is_ollama ? "OLLAMA" : "LLAMA";
    record["perplexity"]       = result.ppl;
    record["nll_mean"]         = result.nll_mean;
    record["ppl_stderr"]       = result.ppl_stderr >= 0.0
                                     ? json(result.ppl_stderr)
                                     : json(nullptr);
    record["n_tokens_scored"]  = result.n_tokens_scored;
    record["n_chunks"]         = result.n_chunks;
    record["n_tokens_missing"] = result.n_tokens_missing;
    record["coverage_pct"]     = result.coverage_pct;
    record["context_size"]     = meta.cfg.context_size;
    record["corpus_sha256"]    = meta.corpus_sha256;
    record["n_tokens_total"]   = meta.n_tokens_total;
    record["model_id"]         = mid;
    record["architecture"]     = meta.model_info.architecture;
    record["quantization"]     = meta.model_info.quantization;
    record["n_params"]         = meta.model_info.n_params;
    record["n_vocab"]          = meta.n_vocab;
    record["ollama_model"]     = is_ollama
                                     ? json(meta.cfg.ollama_model_name)
                                     : json(nullptr);
    record["ollama_url"]       = is_ollama
                                     ? json(meta.cfg.ollama_url)
                                     : json(nullptr);
    record["top_logprobs_k"]   = is_ollama
                                     ? json(meta.cfg.top_logprobs_k)
                                     : json(nullptr);
    record["annotations"]      = meta.cfg.annotations;

    {
        std::ofstream f(jsonl_path);
        if (!f) throw std::runtime_error(
            "cannot write " + jsonl_path.string());
        f << record.dump() << '\n';
    }

    fmt::print("Output written to: {}/\n", out_dir.string());
    fmt::print("  {}\n", resumen_path.filename().string());
    fmt::print("  {}\n", jsonl_name);

    return out_dir.string();
}
