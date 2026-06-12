#pragma once
#include <string>
#include <vector>

struct llama_vocab;

struct CorpusData {
    std::vector<int> tokens;   // llama_token = int32_t; empty for raw mode
    std::string      sha256;
};

// Load corpus from a file path and tokenise it.
CorpusData load_corpus(const llama_vocab* vocab, const std::string& path);

// Load corpus from file, compute SHA-256, but do NOT tokenise (tokens stays empty).
// Used for OLLAMA without a tokeniser model.
CorpusData load_corpus_raw(const std::string& path);

// Load the corpus that was embedded at compile time (corpus/wiki.test.raw).
// Tokenises with the provided vocab.
CorpusData load_corpus_embedded(const llama_vocab* vocab);

// Embedded corpus without tokenisation (for OLLAMA without model_path).
CorpusData load_corpus_raw_embedded();
