#pragma once
#include <string>
#include "config.h"
#include "output.h"

// Forward-declare opaque llama types to avoid including llama.h everywhere.
struct llama_model;
struct llama_context;

struct LoadedModel {
    llama_model*   model   = nullptr;
    llama_context* ctx     = nullptr;
    ModelInfo      info;
    bool           add_bos = false;
    int            n_vocab = 0;
};

// Initialises llama backend, loads the model and creates a context.
// Throws std::runtime_error on failure.
// Call free_loaded_model() when done.
LoadedModel load_model(const Config& cfg);

// Releases context, model and llama backend.
void free_loaded_model(LoadedModel& m);
