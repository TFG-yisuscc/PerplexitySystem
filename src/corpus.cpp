// Reference convention: llama.cpp b9030 uses common_tokenize(ctx, text, add_special=true).
// We replicate that with llama_tokenize(vocab, ..., add_special=true, parse_special=false).

#include "corpus.h"
#include "sha256.h"

// corpus_embedded.h is generated at build time by cmake/gen_corpus_header.py.
// It defines kCorpusData[], kCorpusSize, and kCorpusSHA256[].
#include "corpus_embedded.h"

#include <llama.h>
#include <fmt/core.h>
#include <fstream>
#include <iterator>
#include <stdexcept>

// ── shared tokenisation helper ────────────────────────────────────────────────

static std::vector<int> tokenize(const llama_vocab* vocab,
                                  const char* text, std::size_t text_len) {
    int max_tokens = static_cast<int>(text_len) + 2;
    std::vector<int> tokens(max_tokens);

    int n = llama_tokenize(vocab, text, static_cast<int>(text_len),
                           tokens.data(), max_tokens,
                           /*add_special=*/true, /*parse_special=*/false);
    if (n < 0) {
        max_tokens = -n + 1;
        tokens.resize(max_tokens);
        n = llama_tokenize(vocab, text, static_cast<int>(text_len),
                           tokens.data(), max_tokens, true, false);
        if (n < 0)
            throw std::runtime_error(
                "tokenisation failed (llama_tokenize returned " +
                std::to_string(n) + ")");
    }
    tokens.resize(n);
    return tokens;
}

// ── file-based loading ────────────────────────────────────────────────────────

CorpusData load_corpus(const llama_vocab* vocab, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open corpus file: " + path);

    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    if (text.empty()) throw std::runtime_error("corpus file is empty: " + path);

    fmt::print("Corpus: {} bytes from {}\n", text.size(), path);

    const std::string sha = sha256_impl::hash_bytes(text.data(), text.size());
    auto tokens = tokenize(vocab, text.data(), text.size());
    fmt::print("  tokenised: {} tokens (vocab {})\n", tokens.size(),
               llama_vocab_n_tokens(vocab));
    return {std::move(tokens), sha};
}

CorpusData load_corpus_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open corpus file: " + path);

    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    if (text.empty()) throw std::runtime_error("corpus file is empty: " + path);

    fmt::print("Corpus: {} bytes from {} (raw, no tokenisation)\n",
               text.size(), path);
    return {{}, sha256_impl::hash_bytes(text.data(), text.size())};
}

// ── embedded corpus ───────────────────────────────────────────────────────────

CorpusData load_corpus_embedded(const llama_vocab* vocab) {
    fmt::print("Corpus: {} bytes (embedded, sha256={}...)\n",
               kCorpusSize, std::string(kCorpusSHA256).substr(0, 16));
    auto tokens = tokenize(vocab, reinterpret_cast<const char*>(kCorpusData),
                           kCorpusSize);
    fmt::print("  tokenised: {} tokens (vocab {})\n", tokens.size(),
               llama_vocab_n_tokens(vocab));
    return {std::move(tokens), kCorpusSHA256};
}

CorpusData load_corpus_raw_embedded() {
    fmt::print("Corpus: {} bytes (embedded, raw, no tokenisation)\n", kCorpusSize);
    return {{}, kCorpusSHA256};
}
