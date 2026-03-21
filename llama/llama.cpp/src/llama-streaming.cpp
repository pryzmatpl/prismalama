#include "llama-streaming.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "ggml.h"
#include "gguf.h"

LlamaWeightStreaming::LlamaWeightStreaming()
    : enabled_(false)
    , model_path_("")
    , max_gpu_layers_(0)
    , cache_size_(8)
    , gpu_memory_used_(0)
    , cpu_memory_used_(0) {
}

LlamaWeightStreaming::~LlamaWeightStreaming() {
}

void LlamaWeightStreaming::set_model_path(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_path_ = path;
    analyze_model_layers();
}

void LlamaWeightStreaming::set_max_gpu_layers(int32_t layers) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_gpu_layers_ = layers;
}

void LlamaWeightStreaming::set_cache_size(int32_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_size_ = size;
}

void LlamaWeightStreaming::enable(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enable;
}

llama_layer_info LlamaWeightStreaming::get_layer_info(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_id >= 0 && layer_id < (int)layers_.size()) {
        return layers_[layer_id];
    }
    return llama_layer_info{-1, "", 0, false, false, 0};
}

std::vector<llama_layer_info> LlamaWeightStreaming::get_all_layers() {
    std::lock_guard<std::mutex> lock(mutex_);
    return layers_;
}

void LlamaWeightStreaming::on_layer_access(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_id >= 0 && layer_id < (int)layers_.size()) {
        layers_[layer_id].last_access = access_counter_++;
        update_layer_priority(layer_id);
    }
}

void LlamaWeightStreaming::evict_layer(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_id >= 0 && layer_id < (int)layers_.size()) {
        if (layers_[layer_id].in_gpu) {
            layers_[layer_id].in_gpu = false;
            gpu_memory_used_ -= layers_[layer_id].tensor_size;
        }
    }
}

void LlamaWeightStreaming::load_layer(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_id >= 0 && layer_id < (int)layers_.size()) {
        if (!layers_[layer_id].in_gpu) {
            layers_[layer_id].in_gpu = true;
            gpu_memory_used_ += layers_[layer_id].tensor_size;
        }
    }
}

int32_t LlamaWeightStreaming::get_optimal_gpu_layers() const {
    if (max_gpu_layers_ > 0) {
        return max_gpu_layers_;
    }

    if (layers_.empty()) {
        return 0;
    }

    return (int32_t)layers_.size() / 4;
}

bool LlamaWeightStreaming::should_offload_layer(int layer_id) const {
    if (!enabled_) {
        return true;
    }

    if (max_gpu_layers_ == 0) {
        return true;
    }

    if (layer_id < max_gpu_layers_) {
        return true;
    }

    return false;
}

bool LlamaWeightStreaming::should_evict_layer(int layer_id) const {
    if (!enabled_ || layer_id < 0 || layer_id >= (int)layers_.size()) {
        return false;
    }

    const auto& layer = layers_[layer_id];
    if (!layer.in_gpu) {
        return false;
    }

    int gpu_layers = 0;
    for (const auto& l : layers_) {
        if (l.in_gpu) {
            gpu_layers++;
        }
    }

    return gpu_layers > cache_size_;
}

void LlamaWeightStreaming::analyze_model_layers() {
    layers_.clear();
    layer_map_.clear();

    if (model_path_.empty()) {
        return;
    }

    std::vector<std::string> gguf_files;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(model_path_)) {
            if (entry.is_regular_file()) {
                std::string name = entry.path().filename().string();
                if (name.size() > 5 && name.substr(name.size() - 5) == ".gguf") {
                    gguf_files.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning model directory: " << e.what() << std::endl;
        return;
    }

    if (gguf_files.empty()) {
        return;
    }

    std::sort(gguf_files.begin(), gguf_files.end());

    int layer_id = 0;
    size_t total_size = 0;

    for (const auto& file : gguf_files) {
        struct gguf_context* ctx = gguf_init_from_file(file.c_str(), (struct gguf_init_params){
            .no_alloc = true,
            .ctx = nullptr
        });
        if (!ctx) {
            continue;
        }

        int n_tensors = gguf_get_n_tensors(ctx);
        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(ctx, i);
            if (!name) {
                continue;
            }

            std::string tensor_name(name);

            if (tensor_name.find("blk.") == 0) {
                int blk_layer = -1;
                if (sscanf(tensor_name.c_str(), "blk.%d.", &blk_layer) == 1) {
                    if (blk_layer >= 0) {
                        if (blk_layer >= (int)layers_.size()) {
                            layers_.resize(blk_layer + 1);
                        }
                        if (layers_[blk_layer].name.empty()) {
                            layers_[blk_layer].layer_id = blk_layer;
                            layers_[blk_layer].name = tensor_name;
                            layers_[blk_layer].loaded = true;
                            layers_[blk_layer].in_gpu = false;
                            layers_[blk_layer].last_access = 0;
                        }
                    }
                }
            }

            if (tensor_name.find("expert") != std::string::npos) {
                if ((int)layers_.size() <= layer_id) {
                    layers_.resize(layer_id + 1);
                }
                if (layers_[layer_id].name.empty()) {
                    layers_[layer_id].layer_id = layer_id;
                    layers_[layer_id].name = tensor_name;
                    layers_[layer_id].tensor_size = 0;
                    layers_[layer_id].loaded = true;
                    layers_[layer_id].in_gpu = false;
                    layers_[layer_id].last_access = 0;
                }
                layer_id++;
            }
        }

        gguf_free(ctx);
    }

    std::cout << "WEIGHT_STREAMING: Model analyzed, " << layers_.size()
              << " layers, total size: " << (total_size / (1024*1024*1024)) << " GB" << std::endl;
}

void LlamaWeightStreaming::update_layer_priority(int layer_id) {
    if (layer_id < 0 || layer_id >= (int)layers_.size()) {
        return;
    }

    for (int i = 0; i < (int)layers_.size(); i++) {
        if (i != layer_id && layers_[i].in_gpu) {
            layers_[i].last_access--;
        }
    }
}

extern "C" {

LlamaWeightStreaming* llama_weight_streaming_create() {
    return new LlamaWeightStreaming();
}

void llama_weight_streaming_destroy(LlamaWeightStreaming* ws) {
    delete ws;
}

void llama_weight_streaming_set_model_path(LlamaWeightStreaming* ws, const char* path) {
    if (ws) {
        ws->set_model_path(path);
    }
}

void llama_weight_streaming_set_max_gpu_layers(LlamaWeightStreaming* ws, int32_t layers) {
    if (ws) {
        ws->set_max_gpu_layers(layers);
    }
}

void llama_weight_streaming_set_cache_size(LlamaWeightStreaming* ws, int32_t size) {
    if (ws) {
        ws->set_cache_size(size);
    }
}

void llama_weight_streaming_enable(LlamaWeightStreaming* ws, bool enable) {
    if (ws) {
        ws->enable(enable);
    }
}

bool llama_weight_streaming_is_enabled(LlamaWeightStreaming* ws) {
    return ws && ws->is_enabled();
}

int32_t llama_weight_streaming_get_optimal_gpu_layers(LlamaWeightStreaming* ws) {
    return ws ? ws->get_optimal_gpu_layers() : 0;
}

}
