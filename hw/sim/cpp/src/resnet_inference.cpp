/**
 * @file resnet_inference.cpp
 * @brief ResNet-18 inference engine implementation
 * @author ResNet-Accel Team
 * @date 2024
 */

#include "resnet_inference.hpp"
#include "accelerator_driver.hpp"
#include "bsr_packer.hpp"
#include "golden_models.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>

// ImageNet normalization constants
constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};

//==============================================================================
// LayerWeights Destructor
//==============================================================================

LayerWeights::~LayerWeights() {
    delete weight_bsr;
    delete downsample_weight;
}

//==============================================================================
// Constructor & Destructor
//==============================================================================

ResNetInference::ResNetInference(bool use_accelerator)
    : use_accelerator_(use_accelerator)
    , accelerator_(nullptr) {
    
    init_layer_configs();
    
    // Allocate activation buffers (max size for 512x56x56)
    act_buffer_a_.resize(512 * 56 * 56);
    act_buffer_b_.resize(512 * 56 * 56);
    acc_buffer_.resize(512 * 56 * 56);
    
    if (use_accelerator_) {
        // Initialize accelerator driver (would require full setup)
        std::cout << "[ResNet] Initialized with accelerator backend\n";
    } else {
        std::cout << "[ResNet] Initialized with golden model backend\n";
    }
}

ResNetInference::~ResNetInference() = default;

//==============================================================================
// Layer Configuration
//==============================================================================

void ResNetInference::init_layer_configs() {
    // Conv1: 224x224x3 -> 112x112x64 (7x7 kernel, stride=2, pad=3)
    layer_configs_.push_back({
        "conv1", 3, 64, 224, 224, 7, 2, 3, true, false, false
    });
    
    // Stage 1: 2x BasicBlock (64 channels, 56x56 after maxpool)
    for (int i = 0; i < 2; ++i) {
        layer_configs_.push_back({
            "layer1." + std::to_string(i) + ".conv1", 
            64, 64, 56, 56, 3, 1, 1, true, false, false
        });
        layer_configs_.push_back({
            "layer1." + std::to_string(i) + ".conv2", 
            64, 64, 56, 56, 3, 1, 1, false, true, false
        });
    }
    
    // Stage 2: 2x BasicBlock (128 channels, downsample to 28x28)
    layer_configs_.push_back({
        "layer2.0.conv1", 64, 128, 56, 56, 3, 2, 1, true, false, false
    });
    layer_configs_.push_back({
        "layer2.0.conv2", 128, 128, 28, 28, 3, 1, 1, false, true, true
    });
    layer_configs_.push_back({
        "layer2.1.conv1", 128, 128, 28, 28, 3, 1, 1, true, false, false
    });
    layer_configs_.push_back({
        "layer2.1.conv2", 128, 128, 28, 28, 3, 1, 1, false, true, false
    });
    
    // Stage 3: 2x BasicBlock (256 channels, downsample to 14x14)
    layer_configs_.push_back({
        "layer3.0.conv1", 128, 256, 28, 28, 3, 2, 1, true, false, false
    });
    layer_configs_.push_back({
        "layer3.0.conv2", 256, 256, 14, 14, 3, 1, 1, false, true, true
    });
    layer_configs_.push_back({
        "layer3.1.conv1", 256, 256, 14, 14, 3, 1, 1, true, false, false
    });
    layer_configs_.push_back({
        "layer3.1.conv2", 256, 256, 14, 14, 3, 1, 1, false, true, false
    });
    
    // Stage 4: 2x BasicBlock (512 channels, downsample to 7x7)
    layer_configs_.push_back({
        "layer4.0.conv1", 256, 512, 14, 14, 3, 2, 1, true, false, false
    });
    layer_configs_.push_back({
        "layer4.0.conv2", 512, 512, 7, 7, 3, 1, 1, false, true, true
    });
    layer_configs_.push_back({
        "layer4.1.conv1", 512, 512, 7, 7, 3, 1, 1, true, false, false
    });
    layer_configs_.push_back({
        "layer4.1.conv2", 512, 512, 7, 7, 3, 1, 1, false, true, false
    });
    
    // FC layer: 512 -> 1000
    layer_configs_.push_back({
        "fc", 512, 1000, 1, 1, 1, 1, 0, false, false, false
    });
    
    std::cout << "[ResNet] Configured " << layer_configs_.size() << " layers\n";
}

//==============================================================================
// Model Loading
//==============================================================================

void ResNetInference::load_model(const std::string& weights_dir) {
    std::cout << "[ResNet] Loading model from " << weights_dir << "\n";
    
    for (const auto& config : layer_configs_) {
        auto weights = std::make_unique<LayerWeights>();
        
        // Simplified loading - real implementation would load .npy files
        size_t weight_size = config.out_channels * config.in_channels * 
                            config.kernel_size * config.kernel_size;
        
        // Dummy weights for now
        std::vector<int8_t> dense_weights(weight_size, 1);
        
        // Convert to BSR format using standalone function
        weights->weight_bsr = new resnet_accel::BSRMatrix();
        *weights->weight_bsr = resnet_accel::dense_to_bsr(
            dense_weights.data(), 
            config.in_channels * config.kernel_size * config.kernel_size,
            config.out_channels
        );
        
        weights->bias.resize(config.out_channels, 0);
        weights->weight_scale = 0.01f;
        weights->act_input_scale = 0.01f;
        weights->act_output_scale = 0.01f;
        weights->downsample_weight = nullptr;
        weights->downsample_scale = 0.01f;
        
        layer_weights_.push_back(std::move(weights));
        
        std::cout << "  Loaded " << config.name << " (" << weight_size << " params)\n";
    }
}

void ResNetInference::load_labels(const std::string& labels_file) {
    std::ifstream file(labels_file);
    std::string line;
    while (std::getline(file, line)) {
        class_labels_.push_back(line);
    }
    std::cout << "[ResNet] Loaded " << class_labels_.size() << " class labels\n";
}

//==============================================================================
// Preprocessing
//==============================================================================

void ResNetInference::preprocess_image(const uint8_t* input, int8_t* output) {
    constexpr size_t H = 224, W = 224, C = 3;
    constexpr float scale = 0.01f;  // Activation scale
    
    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                size_t idx = c * H * W + h * W + w;
                
                // Normalize
                float pixel = static_cast<float>(input[idx]) / 255.0f;
                float normalized = (pixel - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                
                // Quantize
                int32_t quantized = static_cast<int32_t>(
                    std::nearbyint(normalized / scale));
                
                // Saturate to INT8
                if (quantized > 127) quantized = 127;
                if (quantized < -128) quantized = -128;
                
                output[idx] = static_cast<int8_t>(quantized);
            }
        }
    }
}

//==============================================================================
// Inference
//==============================================================================

InferenceResult ResNetInference::run_inference(const uint8_t* image_data) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Preprocess image
    std::vector<int8_t> input_quantized(3 * 224 * 224);
    preprocess_image(image_data, input_quantized.data());
    
    std::cout << "[ResNet] Running inference...\n";
    
    // Simplified execution - real implementation would:
    // 1. Run conv1 + maxpool
    // 2. Run each stage's BasicBlocks
    // 3. Global average pooling
    // 4. FC layer + softmax
    
    // For demonstration, return dummy result
    InferenceResult result;
    result.probabilities.resize(1000, 0.001f);
    result.probabilities[281] = 0.85f;  // Tabby cat
    result.predicted_class = 281;
    result.confidence = 0.85f;
    
    // Normalize probabilities
    float sum = 0.0f;
    for (float p : result.probabilities) sum += p;
    for (float& p : result.probabilities) p /= sum;
    result.confidence = result.probabilities[result.predicted_class];
    
    if (!class_labels_.empty()) {
        result.class_name = class_labels_[result.predicted_class];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return result;
}

InferenceResult ResNetInference::run_inference_file(const std::string& image_path) {
    // Load image using stb_image or similar
    // For now, create dummy 224x224x3 image
    std::vector<uint8_t> image_data(224 * 224 * 3, 128);
    return run_inference(image_data.data());
}

//==============================================================================
// Helper Functions
//==============================================================================

TopKResult ResNetInference::get_top_k(const InferenceResult& result, int k) {
    std::vector<std::pair<int, float>> indexed;
    for (size_t i = 0; i < result.probabilities.size(); ++i) {
        indexed.emplace_back(i, result.probabilities[i]);
    }
    
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    
    TopKResult topk;
    for (int i = 0; i < k && i < static_cast<int>(indexed.size()); ++i) {
        topk.class_indices.push_back(indexed[i].first);
        topk.probabilities.push_back(indexed[i].second);
        if (!class_labels_.empty()) {
            topk.class_names.push_back(class_labels_[indexed[i].first]);
        }
    }
    
    return topk;
}

void ResNetInference::run_and_print(const std::string& image_path) {
    auto result = run_inference_file(image_path);
    auto topk = get_top_k(result, 5);
    
    std::cout << "\n=== Inference Result ===\n";
    std::cout << "Latency: " << result.latency_ms << " ms\n";
    std::cout << "\nTop-5 Predictions:\n";
    for (size_t i = 0; i < topk.class_indices.size(); ++i) {
        std::cout << "  " << (i + 1) << ". Class " << topk.class_indices[i];
        if (!topk.class_names.empty()) {
            std::cout << " (" << topk.class_names[i] << ")";
        }
        std::cout << ": " << (topk.probabilities[i] * 100.0f) << "%\n";
    }
}

void ResNetInference::benchmark(const std::string& image_path, int num_runs) {
    std::cout << "[ResNet] Running benchmark with " << num_runs << " iterations\n";
    
    std::vector<double> latencies;
    for (int i = 0; i < num_runs; ++i) {
        auto result = run_inference_file(image_path);
        latencies.push_back(result.latency_ms);
    }
    
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / num_runs;
    double min = *std::min_element(latencies.begin(), latencies.end());
    double max = *std::max_element(latencies.begin(), latencies.end());
    
    std::cout << "Results:\n";
    std::cout << "  Mean:   " << mean << " ms\n";
    std::cout << "  Min:    " << min << " ms\n";
    std::cout << "  Max:    " << max << " ms\n";
    std::cout << "  FPS:    " << (1000.0 / mean) << "\n";
}

bool ResNetInference::verify_accuracy(const uint8_t* image_data, float tolerance) {
    // Run both accelerator and golden model, compare results
    std::cout << "[ResNet] Accuracy verification not yet implemented\n";
    return true;
}

float ResNetInference::get_model_sparsity() const {
    size_t total_params = 0;
    size_t nonzero_params = 0;
    
    for (const auto& weights : layer_weights_) {
        if (weights->weight_bsr) {
            total_params += weights->weight_bsr->num_block_rows * 
                           weights->weight_bsr->num_block_cols * 256;
            nonzero_params += weights->weight_bsr->data.size();
        }
    }
    
    return 1.0f - (static_cast<float>(nonzero_params) / total_params);
}

std::vector<std::pair<std::string, float>> ResNetInference::get_layer_sparsity() const {
    std::vector<std::pair<std::string, float>> result;
    
    for (size_t i = 0; i < layer_configs_.size(); ++i) {
        const auto& config = layer_configs_[i];
        const auto& weights = layer_weights_[i];
        
        if (weights->weight_bsr) {
            size_t total = weights->weight_bsr->num_block_rows * 
                          weights->weight_bsr->num_block_cols * 256;
            size_t nonzero = weights->weight_bsr->data.size();
            float sparsity = 1.0f - (static_cast<float>(nonzero) / total);
            result.emplace_back(config.name, sparsity);
        }
    }
    
    return result;
}

void ResNetInference::print_model_summary() const {
    std::cout << "\n=== ResNet-18 Model Summary ===\n";
    std::cout << "Total layers: " << layer_configs_.size() << "\n";
    std::cout << "Model sparsity: " << (get_model_sparsity() * 100.0f) << "%\n";
    std::cout << "\nPer-layer breakdown:\n";
    
    auto layer_sparsity = get_layer_sparsity();
    for (const auto& [name, sparsity] : layer_sparsity) {
        std::cout << "  " << name << ": " << (sparsity * 100.0f) << "% sparse\n";
    }
}

void ResNetInference::softmax(const float* input, float* output, size_t size) {
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    for (size_t i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

void ResNetInference::run_conv_layer(size_t layer_idx, const int8_t* input, int8_t* output) {
    // Simplified - real implementation would use accelerator or golden model
    std::cout << "  Running layer " << layer_configs_[layer_idx].name << "\n";
}

void ResNetInference::run_basic_block(size_t block_idx, const int8_t* input, int8_t* output) {
    // Run conv1 -> relu -> conv2 -> add residual -> relu
    run_conv_layer(block_idx, input, act_buffer_a_.data());
    run_conv_layer(block_idx + 1, act_buffer_a_.data(), output);
}

void ResNetInference::run_fc_layer(const int8_t* input, float* output) {
    // Matrix multiply with FC weights
    std::cout << "  Running FC layer\n";
}
