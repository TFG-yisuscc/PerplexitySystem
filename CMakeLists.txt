cmake_minimum_required(VERSION 3.25.1)
project(PerplexitySystem)

set(CMAKE_CXX_STANDARD 20)

find_package(llama CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ── embedded corpus header (generated at build time) ─────────────────────────
set(CORPUS_SOURCE  "${CMAKE_SOURCE_DIR}/corpus/wiki.test.raw")
set(CORPUS_GEN_DIR "${CMAKE_BINARY_DIR}/generated")
set(CORPUS_HEADER  "${CORPUS_GEN_DIR}/corpus_embedded.h")

add_custom_command(
    OUTPUT  "${CORPUS_HEADER}"
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/cmake/gen_corpus_header.py"
            "${CORPUS_SOURCE}"
            "${CORPUS_HEADER}"
    DEPENDS "${CORPUS_SOURCE}"
            "${CMAKE_SOURCE_DIR}/cmake/gen_corpus_header.py"
    COMMENT "Embedding corpus/wiki.test.raw → corpus_embedded.h"
)
add_custom_target(corpus_embedded_header DEPENDS "${CORPUS_HEADER}")

# ── main executable ───────────────────────────────────────────────────────────
include_directories(includes)

add_executable(PerplexitySystem
    src/main.cpp
    src/model_loader.cpp
    src/corpus.cpp
    src/perplexity.cpp
    src/output.cpp
    src/ollama_engine.cpp
    includes/config.h
    includes/model_loader.h
    includes/corpus.h
    includes/sha256.h
    includes/perplexity.h
    includes/output.h
    includes/ollama_engine.h
)

add_dependencies(PerplexitySystem corpus_embedded_header)
target_include_directories(PerplexitySystem PRIVATE "${CORPUS_GEN_DIR}")

target_link_libraries(PerplexitySystem PRIVATE
    llama
    fmt::fmt
    nlohmann_json::nlohmann_json
    Threads::Threads
)

target_link_options(PerplexitySystem PRIVATE -static)

configure_file(${CMAKE_SOURCE_DIR}/config_example.json
               ${CMAKE_BINARY_DIR}/config_example.json COPYONLY)
