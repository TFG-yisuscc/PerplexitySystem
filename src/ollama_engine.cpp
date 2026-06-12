// Teacher-forcing perplexity via Ollama /api/generate with top_logprobs.
// Convention: same chunk/scoring window as the LLAMA engine (b9030):
//   non-overlapping chunks of n_ctx tokens; score positions [first, n_ctx-2]
//   where first = n_ctx/2.

#include "ollama_engine.h"

// Include nlohmann/json (vcpkg 3.12.0) before ollama.hpp so the INCLUDE guard
// prevents the older 3.11.3 version bundled inside ollama.hpp from being used.
#include <nlohmann/json.hpp>

// ollama.hpp bundles httplib 0.15.3 (no OpenSSL/brotli by default).
// We don't use the Ollama wrapper class — it lacks logprobs support.
// We use only httplib::Client for raw /api/generate requests.
#include <third_party/ollama.hpp>

#include <llama.h>
#include <fmt/core.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

// ── URL parsing ───────────────────────────────────────────────────────────────

static std::pair<std::string, int> parse_ollama_url(const std::string& url) {
    std::string s = url;
    if (s.rfind("http://", 0) == 0)       s = s.substr(7);
    else if (s.rfind("https://", 0) == 0) s = s.substr(8);
    // Strip trailing path
    auto slash = s.find('/');
    if (slash != std::string::npos) s = s.substr(0, slash);
    // Split host:port
    auto colon = s.rfind(':');
    if (colon != std::string::npos) {
        try { return {s.substr(0, colon), std::stoi(s.substr(colon + 1))}; }
        catch (...) {}
    }
    return {s, 11434};
}

// ── detokenisation ────────────────────────────────────────────────────────────

static std::string token_to_piece(const llama_vocab* vocab, int tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, (int)sizeof(buf) - 1, 0, true);
    if (n < 0) {
        // Buffer too small (rare for normal tokens)
        std::string s(-n + 1, '\0');
        n = llama_token_to_piece(vocab, tok, s.data(), (int)s.size(), 0, true);
        if (n > 0) { s.resize(n); return s; }
        return "";
    }
    return std::string(buf, n);
}

// Concatenates piece strings for tokens[from..to_inc] (inclusive).
static std::string build_prefix(const llama_vocab* vocab,
                                  const std::vector<int>& tokens,
                                  int from, int to_inc) {
    std::string result;
    result.reserve((to_inc - from + 1) * 5);
    for (int i = from; i <= to_inc; ++i)
        result += token_to_piece(vocab, tokens[i]);
    return result;
}

// ── token matching ────────────────────────────────────────────────────────────

// U+2581 LOWER ONE EIGHTH BLOCK (▁) encodes as UTF-8 bytes E2 96 81.
// SentencePiece uses it to mark a leading space. Normalize to ASCII space.
static std::string normalize_piece(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ) {
        if ((unsigned char)s[i] == 0xE2 && i + 2 < s.size() &&
            (unsigned char)s[i + 1] == 0x96 && (unsigned char)s[i + 2] == 0x81) {
            out += ' '; i += 3;
        } else {
            out += s[i++];
        }
    }
    return out;
}

// Returns the log-prob of target from the top_logprobs JSON array, or NaN.
// Matching order:
//   1. Exact string equality
//   2. After replacing ▁ with ASCII space on both sides
//   3. Byte array comparison using the "bytes" field (if present)
static double find_logprob(const json& top_lp, const std::string& target) {
    if (!top_lp.is_array()) return std::numeric_limits<double>::quiet_NaN();

    const std::string t_norm = normalize_piece(target);
    const std::vector<uint8_t> t_bytes(target.begin(), target.end());

    for (const auto& e : top_lp) {
        if (!e.is_object() || !e.contains("token")) continue;
        const std::string cand = e["token"].get<std::string>();
        const double lp = e.value("logprob", std::numeric_limits<double>::quiet_NaN());

        if (cand == target)                    return lp;  // 1: exact
        if (normalize_piece(cand) == t_norm)   return lp;  // 2: normalized

        if (e.contains("bytes") && e["bytes"].is_array()) {          // 3: bytes
            if (e["bytes"].get<std::vector<uint8_t>>() == t_bytes)   return lp;
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

// ── HTTP call ─────────────────────────────────────────────────────────────────

// Sends one /api/generate request with num_predict=1, returns the top_logprobs
// array for the first generated token. Throws on fatal errors.
// first_call: if true, raise hard errors on missing/unsupported logprobs.
static json post_generate(httplib::Client& cli,
                           const Config&   cfg,
                           const std::string& prefix,
                           bool first_call) {
    json body = {
        {"model",        cfg.ollama_model_name},
        {"prompt",       prefix},
        {"raw",          cfg.ollama_raw},
        {"stream",       false},
        {"logprobs",     true},
        {"top_logprobs", cfg.top_logprobs_k},
        {"options", {
            {"num_predict", 1},
            {"temperature", 0},
            {"seed",        0}
        }}
    };

    auto res = cli.Post("/api/generate", body.dump(), "application/json");

    if (!res) {
        throw std::runtime_error(
            "cannot connect to Ollama at " + cfg.ollama_url +
            " — is the server running?");
    }
    if (res->status == 404) {
        throw std::runtime_error(
            "model not found in Ollama: '" + cfg.ollama_model_name + "'");
    }
    if (res->status != 200) {
        const std::string body_excerpt = res->body.substr(0, 200);
        throw std::runtime_error(
            "Ollama HTTP " + std::to_string(res->status) + ": " + body_excerpt);
    }

    json resp;
    try { resp = json::parse(res->body); }
    catch (...) {
        throw std::runtime_error("Ollama response is not valid JSON");
    }

    // Locate top_logprobs — handle the three response shapes seen in the wild.
    if (!resp.contains("logprobs") || resp["logprobs"].is_null()) {
        if (first_call)
            throw std::runtime_error(
                "Ollama response has no 'logprobs' field. "
                "Requires Ollama >= 0.12.11 built with logprobs support.");
        return json::array();
    }

    const json& lp = resp["logprobs"];

    // Shape A: logprobs is an array → lp[0]["top_logprobs"]
    if (lp.is_array() && !lp.empty() && lp[0].contains("top_logprobs"))
        return lp[0]["top_logprobs"];

    // Shape B: logprobs is object with content[] → content[0]["top_logprobs"]
    if (lp.is_object() && lp.contains("content") &&
        lp["content"].is_array() && !lp["content"].empty() &&
        lp["content"][0].contains("top_logprobs"))
        return lp["content"][0]["top_logprobs"];

    // Shape C: logprobs is object with direct top_logprobs key
    if (lp.is_object() && lp.contains("top_logprobs"))
        return lp["top_logprobs"];

    if (first_call)
        throw std::runtime_error(
            "Unexpected 'logprobs' structure in Ollama response. "
            "Check Ollama version (>= 0.12.11 required).");
    return json::array();
}

// ── public API ────────────────────────────────────────────────────────────────

PerplexityResult run_ollama_perplexity(
    const Config&           cfg,
    const std::vector<int>& tokens,
    llama_model*            model)
{
    if (!model || tokens.empty())
        throw std::runtime_error(
            "OLLAMA engine requires 'model_path' in config "
            "for tokenisation and detokenisation");

    const llama_vocab* vocab = llama_model_get_vocab(model);

    auto [host, port] = parse_ollama_url(cfg.ollama_url);
    httplib::Client cli(host, port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(60, 0);
    cli.set_write_timeout(10, 0);

    const int n_ctx    = cfg.context_size;
    const int first    = n_ctx / 2;              // first scored position in chunk
    const int n_scored = n_ctx - 1 - first;      // positions scored per chunk
    const int n_total  = (int)tokens.size();
    const int n_chunks = n_total / n_ctx;

    if (n_chunks == 0)
        throw std::runtime_error(
            "corpus too small for context_size=" + std::to_string(n_ctx));

    fmt::print("Ollama engine: {} chunks, n_ctx={}, scored/chunk={}\n",
               n_chunks, n_ctx, n_scored);
    fmt::print("  endpoint: {}/api/generate\n  model:    {}\n  top_k:    {}\n",
               cfg.ollama_url, cfg.ollama_model_name, cfg.top_logprobs_k);
    if (cfg.max_tokens_scored > 0)
        fmt::print("  max_tokens_scored: {}\n", cfg.max_tokens_scored);

    double nll_sum  = 0.0;
    double nll2_sum = 0.0;
    int    count    = 0;      // tokens successfully scored (found in top_k)
    int    missing  = 0;      // tokens not found in top_k
    int    calls    = 0;      // total HTTP requests
    bool   first_call = true;
    bool   done     = false;

    PerplexityResult result;
    result.n_chunks = n_chunks;
    result.per_chunk_ppl.reserve(n_chunks);

    for (int ci = 0; ci < n_chunks && !done; ++ci) {
        const int cs = ci * n_ctx;   // chunk start token index

        for (int j = first; j < n_ctx - 1 && !done; ++j) {
            // prefix: tokens[cs .. cs+j] detokenised
            const std::string prefix = build_prefix(vocab, tokens, cs, cs + j);
            // target: the next token after the prefix
            const std::string target = token_to_piece(vocab, tokens[cs + j + 1]);

            json top_lp;
            try {
                top_lp = post_generate(cli, cfg, prefix, first_call);
                first_call = false;
            } catch (const std::runtime_error&) {
                // Fatal — propagate to main()
                throw;
            }
            ++calls;

            const double lp = find_logprob(top_lp, target);
            if (std::isnan(lp)) {
                ++missing;
            } else {
                const double v = -lp;   // NLL contribution (nats)
                nll_sum  += v;
                nll2_sum += v * v;
                ++count;
            }

            // Per-position progress every 100 calls
            if (calls % 100 == 0) {
                fmt::print(stderr,
                    "[chunk {}/{} pos {}/{}] scored={} missing={} calls={}",
                    ci + 1, n_chunks, j - first + 1, n_scored,
                    count, missing, calls);
                if (count > 0)
                    fmt::print(stderr, " running_ppl={:.4f}", std::exp(nll_sum / count));
                fmt::print(stderr, "\n");
            }

            if (cfg.max_tokens_scored > 0 &&
                (count + missing) >= cfg.max_tokens_scored)
                done = true;
        }

        // Per-chunk progress (matches LLAMA engine output format)
        const double rppl = count > 0 ? std::exp(nll_sum / count) : -1.0;
        result.per_chunk_ppl.push_back(rppl);
        fmt::print(stderr, "[{}/{}]{:.4f},", ci + 1, n_chunks, rppl);
        if ((ci + 1) % 10 == 0) fmt::print(stderr, "\n");
    }
    fmt::print(stderr, "\n");

    if (count == 0) {
        fmt::print(stderr,
            "Warning: 0 tokens scored — all {} positions had no match in "
            "top_{} candidates\n", missing, cfg.top_logprobs_k);
        return result;
    }

    result.nll_mean         = nll_sum / count;
    result.ppl              = std::exp(result.nll_mean);
    result.n_tokens_scored  = count;
    result.n_tokens_missing = missing;
    result.coverage_pct     = 100.0 * count / (count + missing);

    if (result.coverage_pct < 90.0)
        fmt::print(stderr,
            "WARNING: coverage_pct={:.1f}% — many targets not in top_{} "
            "candidates. Consider increasing top_logprobs_k.\n",
            result.coverage_pct, cfg.top_logprobs_k);

    // PPL stderr: delta method, same formula as LLAMA engine
    if (count > 1) {
        const double pop_var = nll2_sum / count - result.nll_mean * result.nll_mean;
        if (pop_var > 0.0)
            result.ppl_stderr = result.ppl * std::sqrt(pop_var / (count - 1));
    }

    return result;
}
