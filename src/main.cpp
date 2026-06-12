#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <llama.h>
#include <nlohmann/json.hpp>

#include "config.h"
#include "corpus.h"
#include "model_loader.h"
#include "ollama_engine.h"
#include "output.h"
#include "perplexity.h"

using json  = nlohmann::json;
using sys_clock = std::chrono::system_clock;

static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               sys_clock::now().time_since_epoch())
        .count();
}

// ── config parsing ────────────────────────────────────────────────────────────

Config parse_config(const std::string& json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::exception& e) {
        throw std::runtime_error(fmt::format("JSON parse error: {}", e.what()));
    }

    Config cfg;
    cfg.raw_json = json_str;

    // inference_engine
    if (j.contains("inference_engine")) {
        if (!j["inference_engine"].is_string())
            throw std::runtime_error("'inference_engine' must be \"LLAMA\" or \"OLLAMA\"");
        const std::string eng = j["inference_engine"].get<std::string>();
        if (eng == "LLAMA")       cfg.engine = InferenceEngine::LLAMA;
        else if (eng == "OLLAMA") cfg.engine = InferenceEngine::OLLAMA;
        else throw std::runtime_error("'inference_engine' must be \"LLAMA\" or \"OLLAMA\", got: " + eng);
    }

    // corpus_path — optional; empty → embedded corpus used automatically
    if (j.contains("corpus_path")) {
        if (!j["corpus_path"].is_string())
            throw std::runtime_error("'corpus_path' must be a string");
        cfg.corpus_path = j["corpus_path"].get<std::string>();
        if (cfg.corpus_path.empty())
            throw std::runtime_error("'corpus_path' must not be empty if provided");
    }

    // model_path (single) and model_paths (batch) for LLAMA / OLLAMA tokeniser
    if (j.contains("model_path")) {
        if (!j["model_path"].is_string())
            throw std::runtime_error("'model_path' must be a string");
        cfg.model_path = j["model_path"].get<std::string>();
        if (cfg.model_path.empty())
            throw std::runtime_error("'model_path' must not be empty");
    }
    if (j.contains("model_paths")) {
        if (!j["model_paths"].is_array())
            throw std::runtime_error("'model_paths' must be an array of strings");
        for (const auto& p : j["model_paths"]) {
            if (!p.is_string())
                throw std::runtime_error("each entry in 'model_paths' must be a string");
            const std::string s = p.get<std::string>();
            if (s.empty()) throw std::runtime_error("'model_paths' entries must not be empty");
            cfg.model_paths.push_back(s);
        }
    }

    // LLAMA validation: need at least one model source
    if (cfg.engine == InferenceEngine::LLAMA &&
        cfg.model_path.empty() && cfg.model_paths.empty())
        throw std::runtime_error(
            "'model_path' or 'model_paths' is required for engine=LLAMA");

    // ollama_model_name and ollama_model_names
    if (j.contains("ollama_model_name")) {
        if (!j["ollama_model_name"].is_string())
            throw std::runtime_error("'ollama_model_name' must be a string");
        cfg.ollama_model_name = j["ollama_model_name"].get<std::string>();
        if (cfg.ollama_model_name.empty())
            throw std::runtime_error("'ollama_model_name' must not be empty");
    }
    if (j.contains("ollama_model_names")) {
        if (!j["ollama_model_names"].is_array())
            throw std::runtime_error("'ollama_model_names' must be an array of strings");
        for (const auto& n : j["ollama_model_names"]) {
            if (!n.is_string())
                throw std::runtime_error("each entry in 'ollama_model_names' must be a string");
            const std::string s = n.get<std::string>();
            if (s.empty()) throw std::runtime_error("'ollama_model_names' entries must not be empty");
            cfg.ollama_model_names.push_back(s);
        }
    }

    // OLLAMA validation
    if (cfg.engine == InferenceEngine::OLLAMA &&
        cfg.ollama_model_name.empty() && cfg.ollama_model_names.empty())
        throw std::runtime_error(
            "'ollama_model_name' or 'ollama_model_names' is required for engine=OLLAMA");

    // shared optional fields
    if (j.contains("context_size")) {
        if (!j["context_size"].is_number_integer())
            throw std::runtime_error("'context_size' must be a positive integer");
        cfg.context_size = j["context_size"].get<int>();
        if (cfg.context_size <= 0)
            throw std::runtime_error("'context_size' must be > 0");
    }
    if (j.contains("seed")) {
        if (!j["seed"].is_number_integer())
            throw std::runtime_error("'seed' must be an integer");
        cfg.seed = j["seed"].get<int>();
    }
    if (j.contains("annotations"))
        cfg.annotations = j["annotations"];
    if (j.contains("output_dir")) {
        if (!j["output_dir"].is_string())
            throw std::runtime_error("'output_dir' must be a string");
        cfg.output_dir = j["output_dir"].get<std::string>();
        if (cfg.output_dir.empty())
            throw std::runtime_error("'output_dir' must not be empty");
    }

    // LLAMA-specific optional
    if (j.contains("n_threads")) {
        if (!j["n_threads"].is_number_integer())
            throw std::runtime_error("'n_threads' must be a positive integer");
        cfg.n_threads = j["n_threads"].get<int>();
        if (cfg.n_threads <= 0)
            throw std::runtime_error("'n_threads' must be > 0");
    }
    if (j.contains("batch_size")) {
        if (!j["batch_size"].is_number_integer())
            throw std::runtime_error("'batch_size' must be a positive integer");
        cfg.batch_size = j["batch_size"].get<int>();
        if (cfg.batch_size <= 0)
            throw std::runtime_error("'batch_size' must be > 0");
    }
    if (cfg.batch_size == 0) cfg.batch_size = cfg.context_size;

    // OLLAMA-specific optional
    if (j.contains("ollama_url")) {
        if (!j["ollama_url"].is_string())
            throw std::runtime_error("'ollama_url' must be a string");
        cfg.ollama_url = j["ollama_url"].get<std::string>();
        if (cfg.ollama_url.empty())
            throw std::runtime_error("'ollama_url' must not be empty");
    }
    if (j.contains("top_logprobs_k")) {
        if (!j["top_logprobs_k"].is_number_integer())
            throw std::runtime_error("'top_logprobs_k' must be an integer");
        cfg.top_logprobs_k = j["top_logprobs_k"].get<int>();
        if (cfg.top_logprobs_k < 1 || cfg.top_logprobs_k > 20)
            throw std::runtime_error("'top_logprobs_k' must be in [1, 20]");
    }
    if (j.contains("max_tokens_scored")) {
        if (!j["max_tokens_scored"].is_number_integer())
            throw std::runtime_error("'max_tokens_scored' must be a positive integer");
        cfg.max_tokens_scored = j["max_tokens_scored"].get<int>();
        if (cfg.max_tokens_scored <= 0)
            throw std::runtime_error("'max_tokens_scored' must be > 0");
    }
    if (j.contains("ollama_raw")) {
        if (!j["ollama_raw"].is_boolean())
            throw std::runtime_error("'ollama_raw' must be a boolean");
        cfg.ollama_raw = j["ollama_raw"].get<bool>();
    }

    return cfg;
}

// ── usage ─────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    fmt::print(stderr,
        "PerplexitySystem — teacher-forcing perplexity on WikiText-2\n\n"
        "Usage:\n"
        "  {} <config.json>\n"
        "  {} --json '<json_string>'\n\n"
        "Shared fields:\n"
        "  corpus_path      (string, optional)   Path to wiki.test.raw; omit to use embedded corpus\n"
        "  inference_engine (string, \"LLAMA\")    \"LLAMA\" or \"OLLAMA\"\n"
        "  context_size     (int, 512)            Evaluation window in tokens\n"
        "  seed             (int, 0)              Metadata only\n"
        "  annotations      (any)                 Free-form metadata\n"
        "  output_dir       (string, \"results\")   Results root directory\n\n"
        "engine=LLAMA fields:\n"
        "  model_path       (string)              Single GGUF model\n"
        "  model_paths      (array of strings)    Batch: runs each model in sequence\n"
        "  n_threads        (int, all cores)       CPU decode threads\n"
        "  batch_size       (int, context_size)    llama_decode batch size\n\n"
        "engine=OLLAMA fields:\n"
        "  ollama_model_name  (string)             Single Ollama model name\n"
        "  ollama_model_names (array of strings)   Batch: shared tokeniser from model_path\n"
        "  model_path         (string)             GGUF for tokenisation (shared across batch)\n"
        "  ollama_url         (string, \"http://localhost:11434\")\n"
        "  top_logprobs_k     (int [1,20], 20)     Candidates per position\n"
        "  max_tokens_scored  (int, unlimited)      Cap for quick tests\n"
        "  ollama_raw         (bool, true)          Bypass chat template\n\n"
        "Output: results/<timestamp_ns>/resumen.json  +  <ts>_perplexity_<model>.jsonl\n"
        "Metric: wikitext2_teacher_forcing_perplexity\n",
        prog, prog);
}

// ── helpers ───────────────────────────────────────────────────────────────────

static void print_model_info(const LoadedModel& lm) {
    const ModelInfo& mi = lm.info;
    fmt::print("  architecture:     {}\n", mi.architecture.empty() ? "(unknown)" : mi.architecture);
    fmt::print("  quantization:     {}\n", mi.quantization.empty() ? "(unknown)" : mi.quantization);
    fmt::print("  n_params:         {}\n", mi.n_params);
    fmt::print("  embedding_length: {}\n", mi.embedding_length);
    fmt::print("  n_layers:         {}\n", mi.n_layers);
    fmt::print("  max_context:      {}\n", mi.max_context);
    fmt::print("  vocab_size:       {}\n", lm.n_vocab);
    fmt::print("  add_bos:          {}\n", lm.add_bos ? "true" : "false");
}

// Load corpus (file or embedded) given a loaded model (for tokenisation).
// If lm.model is nullptr: returns raw CorpusData (no tokens) using embedded or file.
static CorpusData load_corpus_for(const LoadedModel& lm, const Config& cfg) {
    const bool use_embedded = cfg.corpus_path.empty();

    if (lm.model) {
        const llama_vocab* vocab = llama_model_get_vocab(lm.model);
        return use_embedded ? load_corpus_embedded(vocab)
                            : load_corpus(vocab, cfg.corpus_path);
    } else {
        return use_embedded ? load_corpus_raw_embedded()
                            : load_corpus_raw(cfg.corpus_path);
    }
}

static PerplexityResult run_engine(const Config& run_cfg,
                                    const CorpusData& corpus,
                                    const LoadedModel& lm) {
    if (run_cfg.engine == InferenceEngine::LLAMA) {
        return run_perplexity(lm.ctx, corpus.tokens,
                               run_cfg.context_size, run_cfg.batch_size,
                               run_cfg.max_tokens_scored);
    } else {
        return run_ollama_perplexity(run_cfg, corpus.tokens, lm.model);
    }
}

static void print_result(const std::string& model_id,
                          const PerplexityResult& r,
                          bool is_ollama) {
    fmt::print("\n=== {} ===\n", model_id);
    fmt::print("  perplexity:       {:.4f}\n", r.ppl);
    fmt::print("  nll_mean (nats):  {:.6f}\n", r.nll_mean);
    if (r.ppl_stderr >= 0.0)
        fmt::print("  ppl_stderr:       {:.5f}\n", r.ppl_stderr);
    fmt::print("  n_tokens_scored:  {}\n", r.n_tokens_scored);
    fmt::print("  n_chunks:         {}\n", r.n_chunks);
    if (is_ollama) {
        fmt::print("  n_tokens_missing: {}\n", r.n_tokens_missing);
        fmt::print("  coverage_pct:     {:.2f}%\n", r.coverage_pct);
    }
}

// ── entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::string json_str;
    if (std::string(argv[1]) == "--json") {
        if (argc < 3) {
            fmt::print(stderr, "Error: --json requires a JSON string argument\n\n");
            print_usage(argv[0]);
            return 1;
        }
        json_str = argv[2];
    } else {
        std::ifstream f(argv[1]);
        if (!f) { fmt::print(stderr, "Error: cannot open config file: {}\n", argv[1]); return 1; }
        std::ostringstream ss; ss << f.rdbuf(); json_str = ss.str();
    }

    Config cfg;
    try {
        cfg = parse_config(json_str);
    } catch (const std::exception& e) {
        fmt::print(stderr, "Error: invalid configuration: {}\n", e.what());
        return 1;
    }

    const bool is_ollama     = (cfg.engine == InferenceEngine::OLLAMA);
    const bool use_embedded  = cfg.corpus_path.empty();

    // ── build model list ──────────────────────────────────────────────────────
    // model_paths / ollama_model_names override the single-value fields.
    std::vector<std::string> model_list;
    if (is_ollama) {
        model_list = cfg.ollama_model_names.empty()
                     ? std::vector<std::string>{cfg.ollama_model_name}
                     : cfg.ollama_model_names;
    } else {
        model_list = cfg.model_paths.empty()
                     ? std::vector<std::string>{cfg.model_path}
                     : cfg.model_paths;
    }
    const int n_models = static_cast<int>(model_list.size());

    // ── startup summary ───────────────────────────────────────────────────────
    fmt::print("PerplexitySystem starting\n");
    fmt::print("  engine:       {}\n", is_ollama ? "OLLAMA" : "LLAMA");
    fmt::print("  corpus:       {}\n",
               use_embedded ? "embedded (wiki.test.raw)" : cfg.corpus_path);
    fmt::print("  context_size: {}\n", cfg.context_size);
    fmt::print("  output_dir:   {}\n", cfg.output_dir);
    if (!is_ollama) {
        fmt::print("  batch_size:   {}\n", cfg.batch_size);
        fmt::print("  n_threads:    {}\n", cfg.n_threads == 0 ? -1 : cfg.n_threads);
        if (cfg.max_tokens_scored > 0)
            fmt::print("  max_scored:   {}\n", cfg.max_tokens_scored);
    } else {
        fmt::print("  ollama_url:   {}\n", cfg.ollama_url);
        fmt::print("  top_k:        {}\n", cfg.top_logprobs_k);
        if (cfg.max_tokens_scored > 0)
            fmt::print("  max_scored:   {}\n", cfg.max_tokens_scored);
        if (!cfg.model_path.empty())
            fmt::print("  tokeniser:    {}\n", cfg.model_path);
    }
    fmt::print("  models ({}):\n", n_models);
    for (int i = 0; i < n_models; ++i)
        fmt::print("    [{}] {}\n", i, model_list[i]);

    // ── OLLAMA optimisation: load shared tokeniser once ───────────────────────
    // For OLLAMA batch, the tokeniser GGUF (model_path) is the same for all model
    // names, so we load it once and reuse the corpus tokenisation across iterations.
    LoadedModel shared_lm;
    CorpusData  shared_corpus;
    const bool  reuse_tokenizer = is_ollama;

    if (reuse_tokenizer) {
        if (!cfg.model_path.empty()) {
            fmt::print("\nLoading shared tokeniser:\n");
            try {
                shared_lm = load_model(cfg);
                print_model_info(shared_lm);
            } catch (const std::exception& e) {
                fmt::print(stderr, "Error loading tokeniser model: {}\n", e.what());
                return 1;
            }
        }
        fmt::print("\n");
        try {
            shared_corpus = load_corpus_for(shared_lm, cfg);
        } catch (const std::exception& e) {
            fmt::print(stderr, "Error loading corpus: {}\n", e.what());
            free_loaded_model(shared_lm);
            return 1;
        }
        if (!shared_corpus.tokens.empty() &&
            static_cast<int>(shared_corpus.tokens.size()) < 2 * cfg.context_size) {
            fmt::print(stderr,
                "Error: corpus has only {} tokens, need >= {} for context_size={}\n",
                shared_corpus.tokens.size(), 2 * cfg.context_size, cfg.context_size);
            free_loaded_model(shared_lm);
            return 1;
        }
        fmt::print("  sha256: {}\n", shared_corpus.sha256);
        if (!shared_corpus.tokens.empty())
            fmt::print("  n_tokens: {}\n", static_cast<int>(shared_corpus.tokens.size()));
    }

    // ── multi-model loop ──────────────────────────────────────────────────────
    struct RunSummary { std::string id; PerplexityResult result; };
    std::vector<RunSummary> all_results;
    all_results.reserve(n_models);
    int n_failed = 0;

    for (int mi = 0; mi < n_models; ++mi) {
        const std::string& model_id = model_list[mi];

        fmt::print("\n──── Model {}/{}: {} ────\n", mi + 1, n_models, model_id);

        // Build per-iteration config with the current model
        Config run_cfg = cfg;
        if (is_ollama) run_cfg.ollama_model_name = model_id;
        else           run_cfg.model_path        = model_id;

        // Model and corpus: either shared (OLLAMA) or freshly loaded (LLAMA)
        LoadedModel* lm_ptr     = nullptr;
        const CorpusData* cp    = nullptr;
        LoadedModel  own_lm;
        CorpusData   own_corpus;

        if (reuse_tokenizer) {
            lm_ptr = &shared_lm;
            cp     = &shared_corpus;
        } else {
            // LLAMA: each model has its own tokeniser — reload both
            try {
                own_lm = load_model(run_cfg);
            } catch (const std::exception& e) {
                fmt::print(stderr, "  Error loading model: {} — skipped\n", e.what());
                ++n_failed; continue;
            }
            print_model_info(own_lm);

            try {
                own_corpus = load_corpus_for(own_lm, cfg);
            } catch (const std::exception& e) {
                fmt::print(stderr, "  Error loading corpus: {} — skipped\n", e.what());
                free_loaded_model(own_lm);
                ++n_failed; continue;
            }
            if (!own_corpus.tokens.empty() &&
                static_cast<int>(own_corpus.tokens.size()) < 2 * cfg.context_size) {
                fmt::print(stderr,
                    "  Error: corpus too small ({} tokens) for context_size={} — skipped\n",
                    own_corpus.tokens.size(), cfg.context_size);
                free_loaded_model(own_lm);
                ++n_failed; continue;
            }
            fmt::print("  sha256: {}\n", own_corpus.sha256);
            if (!own_corpus.tokens.empty())
                fmt::print("  n_tokens: {}\n", static_cast<int>(own_corpus.tokens.size()));

            lm_ptr = &own_lm;
            cp     = &own_corpus;
        }

        // Run
        const int64_t ts_start = now_ns();
        PerplexityResult result;
        try {
            result = run_engine(run_cfg, *cp, *lm_ptr);
        } catch (const std::exception& e) {
            fmt::print(stderr, "  Error: {} — skipped\n", e.what());
            if (!reuse_tokenizer) free_loaded_model(own_lm);
            ++n_failed; continue;
        }
        const int64_t ts_end = now_ns();

        print_result(model_id, result, is_ollama);

        // Write output
        RunMeta meta;
        meta.cfg             = run_cfg;
        meta.model_info      = lm_ptr->info;
        meta.corpus_sha256   = cp->sha256;
        meta.n_tokens_total  = static_cast<int>(cp->tokens.size());
        meta.n_vocab         = lm_ptr->n_vocab;
        meta.ts_run_start_ns = ts_start;
        meta.ts_run_end_ns   = ts_end;
        try {
            write_results(meta, result);
        } catch (const std::exception& e) {
            fmt::print(stderr, "  Warning: could not write results: {}\n", e.what());
        }

        all_results.push_back({model_id, result});
        if (!reuse_tokenizer) free_loaded_model(own_lm);
    }

    if (reuse_tokenizer) free_loaded_model(shared_lm);

    // ── comparison table (only for batch runs) ────────────────────────────────
    if (n_models > 1 && !all_results.empty()) {
        const int col = 48;
        fmt::print("\n\n{:=<{}}\n", "= Comparison ", 74);
        fmt::print("{:<{}} {:>10} {:>10} {:>10} {:>10}\n",
                   "Model", col, "PPL", "NLL", "stderr", "scored");
        fmt::print("{:-<74}\n", "");
        for (const auto& r : all_results) {
            const std::string short_id = r.id.size() > (std::size_t)col
                ? "…" + r.id.substr(r.id.size() - col + 1)
                : r.id;
            const std::string stderr_str = r.result.ppl_stderr >= 0.0
                ? fmt::format("{:.5f}", r.result.ppl_stderr)
                : "N/A";
            fmt::print("{:<{}} {:>10.4f} {:>10.6f} {:>10} {:>10}\n",
                       short_id, col,
                       r.result.ppl, r.result.nll_mean,
                       stderr_str, r.result.n_tokens_scored);
        }
        if (n_failed > 0)
            fmt::print("({} model(s) failed and were skipped)\n", n_failed);
    }

    return n_failed > 0 && all_results.empty() ? 1 : 0;
}
