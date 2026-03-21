#pragma once

#include "llama.h"

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>

struct llama_layer_info {
    int layer_id;
    std::string name;
    size_t tensor_size;
    bool loaded;
    bool in_gpu;
    int64_t last_access;
};

class LLAMA_API LlamaWeightStreaming {
public:
    LlamaWeightStreaming();
    ~LlamaWeightStreaming();

    void set_model_path(const std::string& path);
    void set_max_gpu_layers(int32_t layers);
    void set_cache_size(int32_t size);
    void enable(bool enable);

    bool is_enabled() const { return enabled_; }

    llama_layer_info get_layer_info(int layer_id);
    std::vector<llama_layer_info> get_all_layers();

    void on_layer_access(int layer_id);
    void evict_layer(int layer_id);
    void load_layer(int layer_id);

    size_t get_gpu_memory_used() const { return gpu_memory_used_; }
    size_t get_cpu_memory_used() const { return cpu_memory_used_; }

    int32_t get_optimal_gpu_layers() const;

    bool should_offload_layer(int layer_id) const;
    bool should_evict_layer(int layer_id) const;

private:
    bool enabled_;
    std::string model_path_;
    int32_t max_gpu_layers_;
    int32_t cache_size_;
    size_t gpu_memory_used_;
    size_t cpu_memory_used_;

    std::vector<llama_layer_info> layers_;
    std::unordered_map<int, llama_layer_info> layer_map_;

    std::mutex mutex_;
    std::atomic<int64_t> access_counter_{0};

    void analyze_model_layers();
    void update_layer_priority(int layer_id);
};

extern "C" {

LLAMA_API LlamaWeightStreaming* llama_weight_streaming_create();
LLAMA_API void llama_weight_streaming_destroy(LlamaWeightStreaming* ws);

LLAMA_API void llama_weight_streaming_set_model_path(LlamaWeightStreaming* ws, const char* path);
LLAMA_API void llama_weight_streaming_set_max_gpu_layers(LlamaWeightStreaming* ws, int32_t layers);
LLAMA_API void llama_weight_streaming_set_cache_size(LlamaWeightStreaming* ws, int32_t size);
LLAMA_API void llama_weight_streaming_enable(LlamaWeightStreaming* ws, bool enable);

LLAMA_API bool llama_weight_streaming_is_enabled(LlamaWeightStreaming* ws);

LLAMA_API int32_t llama_weight_streaming_get_optimal_gpu_layers(LlamaWeightStreaming* ws);

}
