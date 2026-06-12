// Reference: llama.cpp b9030, commit e7f1e8d5e1
// tools/perplexity/perplexity.cpp, function perplexity()
//
// Convention replicated:
//  - Non-overlapping chunks of n_ctx tokens: n_chunks = tokens.size() / n_ctx
//  - BOS token replaces position 0 of each chunk's batch (not the tokens vector)
//    when llama_vocab_get_add_bos() is true; identical to the reference which
//    temporarily overwrites tokens[chunk_start] then restores it.
//  - Only the second half is scored: positions j = [n_ctx/2, n_ctx-2]
//  - Scored tokens per chunk: n_ctx - n_ctx/2 - 1  (e.g. 255 for n_ctx=512)
//  - Log-softmax computed in double precision (subtract max, sum exp, log)
//  - NLL and NLL² accumulated in double; ppl_stderr via delta method

#include "perplexity.h"

#include <llama.h>
#include <fmt/core.h>

#include <cmath>
#include <stdexcept>

// ── double-precision log-softmax ──────────────────────────────────────────────

static double log_softmax_double(const float* logits, int n_vocab, int tok) {
    float max_val = logits[0];
    for (int i = 1; i < n_vocab; ++i)
        if (logits[i] > max_val) max_val = logits[i];

    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i)
        sum_exp += std::exp(static_cast<double>(logits[i] - max_val));

    return static_cast<double>(logits[tok] - max_val) - std::log(sum_exp);
}

// ── public API ────────────────────────────────────────────────────────────────

PerplexityResult run_perplexity(
    llama_context*          ctx,
    const std::vector<int>& tokens,
    int context_size,
    int batch_size,
    int max_tokens_scored)
{
    if (batch_size < context_size)
        throw std::runtime_error(
            "batch_size (" + std::to_string(batch_size) +
            ") must be >= context_size (" + std::to_string(context_size) + ")");

    const llama_model* model   = llama_get_model(ctx);
    const llama_vocab* vocab   = llama_model_get_vocab(model);
    const bool         add_bos = llama_vocab_get_add_bos(vocab);
    const int          n_vocab = llama_vocab_n_tokens(vocab);
    const int          n_ctx   = context_size;
    const int          first   = n_ctx / 2;          // ref: "calculate over last half"
    const int          n_chunks = static_cast<int>(tokens.size()) / n_ctx;

    if (n_chunks == 0)
        throw std::runtime_error(
            "corpus too small for context_size=" + std::to_string(n_ctx));

    fmt::print("Computing perplexity: {} chunks, n_ctx={}, first={}, "
               "scored/chunk={}{}\n",
               n_chunks, n_ctx, first, n_ctx - first - 1,
               max_tokens_scored > 0
                   ? fmt::format(", max_tokens={}", max_tokens_scored) : "");

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    double nll_sum  = 0.0;
    double nll2_sum = 0.0;
    int    count    = 0;
    bool   done     = false;

    PerplexityResult result;
    result.n_chunks = n_chunks;
    result.add_bos  = add_bos;
    result.per_chunk_ppl.reserve(n_chunks);

    for (int ci = 0; ci < n_chunks && !done; ++ci) {
        const int chunk_start = ci * n_ctx;

        // How many positions to score in this chunk (may be less than full half
        // for the final chunk when max_tokens_scored is set).
        const int full_score   = n_ctx - first - 1;
        const int to_score     = (max_tokens_scored > 0)
                                 ? std::min(full_score, max_tokens_scored - count)
                                 : full_score;

        // Clear KV cache before each chunk (same as ref: llama_memory_clear)
        llama_memory_clear(llama_get_memory(ctx), true);

        // Build batch — inject BOS at position 0 without touching tokens[].
        // Only request logits for the positions we will actually score.
        batch.n_tokens = n_ctx;
        for (int k = 0; k < n_ctx; ++k) {
            batch.token   [k]    = (k == 0 && add_bos)
                                   ? llama_vocab_bos(vocab)
                                   : tokens[chunk_start + k];
            batch.pos     [k]    = static_cast<llama_pos>(k);
            batch.n_seq_id[k]    = 1;
            batch.seq_id  [k][0] = 0;
            batch.logits  [k]    = (k >= first && k < first + to_score) ? 1 : 0;
        }

        if (llama_decode(ctx, batch) != 0) {
            fmt::print(stderr, "\nWarning: llama_decode failed at chunk {}/{} — skipped\n",
                       ci + 1, n_chunks);
            result.per_chunk_ppl.push_back(-1.0);
            continue;
        }

        // Score positions [first, first + to_score): logit at j predicts j+1
        for (int j = first; j < first + to_score; ++j) {
            const float* row    = llama_get_logits_ith(ctx, j);
            const int    target = tokens[chunk_start + j + 1];
            const double log_p  = log_softmax_double(row, n_vocab, target);
            const double v      = -log_p;
            nll_sum  += v;
            nll2_sum += v * v;
        }
        count += to_score;

        // Running PPL (matches llama-perplexity's per-chunk output)
        const double running_ppl = std::exp(nll_sum / count);
        result.per_chunk_ppl.push_back(running_ppl);
        fmt::print(stderr, "[{}/{}]{:.4f},", ci + 1, n_chunks, running_ppl);
        if ((ci + 1) % 10 == 0) fmt::print(stderr, "\n");

        if (max_tokens_scored > 0 && count >= max_tokens_scored)
            done = true;
    }
    fmt::print(stderr, "\n");

    llama_batch_free(batch);

    if (count == 0) {
        fmt::print(stderr, "Error: no tokens were scored\n");
        return result;
    }

    result.nll_mean        = nll_sum / count;
    result.ppl             = std::exp(result.nll_mean);
    result.n_tokens_scored = count;

    // PPL stderr via delta method (replicates reference b9030):
    //   pop_var  = E[v²] − E[v]²   (population variance of per-token NLL)
    //   sem_nll  = sqrt(pop_var / (N-1))   (standard error of mean NLL)
    //   ppl_stderr = ppl × sem_nll
    if (count > 1) {
        const double pop_var = nll2_sum / count - result.nll_mean * result.nll_mean;
        if (pop_var > 0.0)
            result.ppl_stderr = result.ppl * std::sqrt(pop_var / (count - 1));
    }

    return result;
}
