/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                      RESNET_INFERENCE.CPP                                 ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  IMPLEMENTS: resnet_inference.hpp                                         ║
 * ║  NEW FILE - combines logic from multiple Python files                    ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  WHAT YOU NEED TO IMPLEMENT:                                              ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  Constructor / Destructor                                                 ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  ResNetInference(use_accelerator):                                        ║
 * ║    use_accelerator_ = use_accelerator;                                    ║
 * ║    if (use_accelerator) {                                                 ║
 * ║        accelerator_ = std::make_unique<AcceleratorDriver>(                ║
 * ║            AcceleratorDriver::Mode::SIMULATION);  // or FPGA              ║
 * ║    }                                                                      ║
 * ║    bsr_packer_ = std::make_unique<BSRPacker>();                           ║
 * ║    init_layer_configs();                                                  ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  init_layer_configs() - ResNet-18 architecture                            ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  ResNet-18 layers:                                                        ║
 * ║    conv1:     3 -> 64, 7x7, stride 2, pad 3                              ║
 * ║    maxpool:   3x3, stride 2, pad 1                                       ║
 * ║                                                                           ║
 * ║    layer1.0.conv1: 64 -> 64, 3x3, stride 1                               ║
 * ║    layer1.0.conv2: 64 -> 64, 3x3, stride 1  + residual                   ║
 * ║    layer1.1.conv1: 64 -> 64, 3x3, stride 1                               ║
 * ║    layer1.1.conv2: 64 -> 64, 3x3, stride 1  + residual                   ║
 * ║                                                                           ║
 * ║    layer2.0.conv1: 64 -> 128, 3x3, stride 2  (downsample)                ║
 * ║    layer2.0.conv2: 128 -> 128, 3x3, stride 1  + residual + downsample    ║
 * ║    layer2.1.conv1: 128 -> 128, 3x3, stride 1                             ║
 * ║    layer2.1.conv2: 128 -> 128, 3x3, stride 1  + residual                 ║
 * ║                                                                           ║
 * ║    layer3.0.conv1: 128 -> 256, 3x3, stride 2  (downsample)               ║
 * ║    layer3.0.conv2: 256 -> 256, 3x3, stride 1  + residual + downsample    ║
 * ║    layer3.1.conv1: 256 -> 256, 3x3, stride 1                             ║
 * ║    layer3.1.conv2: 256 -> 256, 3x3, stride 1  + residual                 ║
 * ║                                                                           ║
 * ║    layer4.0.conv1: 256 -> 512, 3x3, stride 2  (downsample)               ║
 * ║    layer4.0.conv2: 512 -> 512, 3x3, stride 1  + residual + downsample    ║
 * ║    layer4.1.conv1: 512 -> 512, 3x3, stride 1                             ║
 * ║    layer4.1.conv2: 512 -> 512, 3x3, stride 1  + residual                 ║
 * ║                                                                           ║
 * ║    avgpool:  Global average pooling                                      ║
 * ║    fc:       512 -> 1000                                                 ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  load_model(weights_dir) - Load quantized weights                        ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  For each layer:                                                          ║
 * ║    1. Load {layer}_weight_int8.npy                                       ║
 * ║    2. Load {layer}_weight_scales.npy                                     ║
 * ║    3. Load {layer}_bias_int32.npy (if exists)                            ║
 * ║    4. Convert weight to BSR format                                       ║
 * ║    5. Store in layer_weights_                                            ║
 * ║                                                                           ║
 * ║  Also load quantization_metadata.json for activation scales              ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  preprocess_image(input, output) - ImageNet preprocessing                ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  ImageNet normalization (per-channel):                                    ║
 * ║    mean = [0.485, 0.456, 0.406]                                          ║
 * ║    std  = [0.229, 0.224, 0.225]                                          ║
 * ║                                                                           ║
 * ║  for c in [0, 1, 2]:  // R, G, B                                         ║
 * ║    for pixel in channel[c]:                                               ║
 * ║      normalized = (pixel / 255.0 - mean[c]) / std[c]                     ║
 * ║      quantized = round(normalized / activation_scale)                     ║
 * ║      output = clamp(quantized, -128, 127)                                 ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  run_inference(image_data) - Full inference pipeline                     ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  1. preprocess_image(image_data, act_buffer_a_)                          ║
 * ║  2. conv1 + relu                                                         ║
 * ║  3. maxpool                                                               ║
 * ║  4. For each BasicBlock in layer1-4:                                     ║
 * ║       run_basic_block()                                                   ║
 * ║  5. Global average pooling                                                ║
 * ║  6. FC layer                                                              ║
 * ║  7. Softmax                                                               ║
 * ║  8. Return InferenceResult                                                ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  run_basic_block() - ResNet BasicBlock (2 convs + residual)              ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  identity = input  (or downsample if stride != 1)                        ║
 * ║  x = conv1(input)                                                         ║
 * ║  x = relu(x)                                                              ║
 * ║  x = conv2(x)                                                             ║
 * ║  x = x + identity  (add residual)                                        ║
 * ║  x = relu(x)                                                              ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  softmax(input, output, size) - Convert logits to probabilities          ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  // Find max for numerical stability                                      ║
 * ║  float max_val = *max_element(input, input + size);                       ║
 * ║                                                                           ║
 * ║  // Compute exp and sum                                                   ║
 * ║  float sum = 0;                                                           ║
 * ║  for (i = 0; i < size; i++) {                                             ║
 * ║      output[i] = exp(input[i] - max_val);                                 ║
 * ║      sum += output[i];                                                    ║
 * ║  }                                                                        ║
 * ║                                                                           ║
 * ║  // Normalize                                                             ║
 * ║  for (i = 0; i < size; i++) {                                             ║
 * ║      output[i] /= sum;                                                    ║
 * ║  }                                                                        ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include "resnet_inference.hpp"
#include "accelerator_driver.hpp"
#include "bsr_packer.hpp"
#include "golden_models.hpp"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

// ImageNet normalization constants
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

// =============================================================================
// LayerWeights destructor
// =============================================================================

LayerWeights::~LayerWeights() {
    // TODO: Clean up owned BSRMatrix if dynamically allocated
    // delete weight_bsr;
    // delete downsample_weight;
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

ResNetInference::ResNetInference(bool use_accelerator)
    : use_accelerator_(use_accelerator) {
    
    // TODO: Implement
    //
    // if (use_accelerator) {
    //     // Choose mode based on environment or build config
    //     #ifdef FPGA_MODE
    //     accelerator_ = std::make_unique<AcceleratorDriver>(
    //         AcceleratorDriver::Mode::FPGA);
    //     #else
    //     accelerator_ = std::make_unique<AcceleratorDriver>(
    //         AcceleratorDriver::Mode::SIMULATION);
    //     #endif
    //     accelerator_->initialize();
    // }
    //
    // bsr_packer_ = std::make_unique<BSRPacker>();
    // init_layer_configs();
    //
    // // Allocate intermediate buffers
    // // Max activation size: 64 * 56 * 56 = 200704
    // act_buffer_a_.resize(512 * 56 * 56);
    // act_buffer_b_.resize(512 * 56 * 56);
    // acc_buffer_.resize(512 * 56 * 56);
}

ResNetInference::~ResNetInference() = default;

// =============================================================================
// Layer Configuration
// =============================================================================

void ResNetInference::init_layer_configs() {
    // TODO: Implement - define all ResNet-18 layers
    //
    // layer_configs_.clear();
    //
    // // conv1: 3 -> 64, 7x7, stride 2, pad 3
    // layer_configs_.push_back({
    //     .name = "conv1",
    //     .in_channels = 3,
    //     .out_channels = 64,
    //     .in_height = 224,
    //     .in_width = 224,
    //     .kernel_size = 7,
    //     .stride = 2,
    //     .padding = 3,
    //     .has_relu = true,
    //     .has_residual = false,
    //     .has_downsample = false
    // });
    //
    // // layer1.0.conv1: 64 -> 64, 3x3
    // layer_configs_.push_back({
    //     .name = "layer1.0.conv1",
    //     .in_channels = 64,
    //     .out_channels = 64,
    //     .in_height = 56,
    //     .in_width = 56,
    //     .kernel_size = 3,
    //     .stride = 1,
    //     .padding = 1,
    //     .has_relu = true,
    //     .has_residual = false,
    //     .has_downsample = false
    // });
    //
    // // ... continue for all layers ...
}

// =============================================================================
// Model Loading
// =============================================================================

void ResNetInference::load_model(const std::string& weights_dir) {
    // TODO: Implement - load weights for each layer
    //
    // layer_weights_.clear();
    //
    // for (const auto& cfg : layer_configs_) {
    //     auto weights = std::make_unique<LayerWeights>();
    //
    //     // Load weight file
    //     std::string weight_path = weights_dir + "/" + cfg.name + "_weight_int8.npy";
    //     std::vector<int8_t> weight_data; // = load_npy_int8(weight_path);
    //
    //     // Convert to BSR
    //     size_t rows = cfg.out_channels;
    //     size_t cols = cfg.in_channels * cfg.kernel_size * cfg.kernel_size;
    //     BSRMatrix bsr = bsr_packer_->dense_to_bsr(weight_data.data(), rows, cols, 1.0f);
    //     weights->weight_bsr = new BSRMatrix(std::move(bsr));
    //
    //     // Load scales
    //     std::string scale_path = weights_dir + "/" + cfg.name + "_weight_scales.npy";
    //     // weights->weight_scale = load_npy_float(scale_path)[0];
    //
    //     // Load bias if exists
    //     std::string bias_path = weights_dir + "/" + cfg.name + "_bias_int32.npy";
    //     // if (file_exists(bias_path)) {
    //     //     weights->bias = load_npy_int32(bias_path);
    //     // }
    //
    //     layer_weights_.push_back(std::move(weights));
    // }
}

void ResNetInference::load_labels(const std::string& labels_file) {
    // TODO: Implement
    //
    // std::ifstream file(labels_file);
    // if (!file) return;
    //
    // class_labels_.clear();
    // std::string line;
    // while (std::getline(file, line)) {
    //     class_labels_.push_back(line);
    // }
}

// =============================================================================
// Preprocessing
// =============================================================================

void ResNetInference::preprocess_image(const uint8_t* input, int8_t* output) {
    // TODO: Implement ImageNet preprocessing
    //
    // float act_scale = 0.02f;  // Get from quantization metadata
    //
    // for (int c = 0; c < 3; c++) {
    //     for (int h = 0; h < 224; h++) {
    //         for (int w = 0; w < 224; w++) {
    //             // Input is HWC format (from image loader)
    //             uint8_t pixel = input[h * 224 * 3 + w * 3 + c];
    //
    //             // Normalize
    //             float normalized = (pixel / 255.0f - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
    //
    //             // Quantize
    //             int32_t quantized = static_cast<int32_t>(std::round(normalized / act_scale));
    //
    //             // Clamp
    //             if (quantized > 127) quantized = 127;
    //             if (quantized < -128) quantized = -128;
    //
    //             // Output is CHW format (for convolution)
    //             output[c * 224 * 224 + h * 224 + w] = static_cast<int8_t>(quantized);
    //         }
    //     }
    // }
}

// =============================================================================
// Inference
// =============================================================================

InferenceResult ResNetInference::run_inference(const uint8_t* image_data) {
    InferenceResult result;
    
    // TODO: Implement full inference pipeline
    //
    // // 1. Preprocess
    // preprocess_image(image_data, act_buffer_a_.data());
    //
    // // 2. Run conv1 + relu
    // run_conv_layer(0, act_buffer_a_.data(), act_buffer_b_.data());
    //
    // // 3. Maxpool
    // std::vector<int8_t> pooled(64 * 56 * 56);
    // golden::maxpool2d_int8(act_buffer_b_.data(), pooled.data(),
    //                        64, 112, 112, 3, 2);
    //
    // // 4. Run layer1-4 (8 BasicBlocks total)
    // // ... swap buffers appropriately ...
    //
    // // 5. Global average pooling
    // std::vector<int32_t> pooled_fc(512);
    // golden::avgpool_global_int8(current_act, pooled_fc.data(), 512, 7, 7);
    //
    // // 6. FC layer
    // std::vector<float> logits(1000);
    // run_fc_layer(pooled_act, logits.data());
    //
    // // 7. Softmax
    // result.probabilities.resize(1000);
    // softmax(logits.data(), result.probabilities.data(), 1000);
    //
    // // 8. Find top prediction
    // auto max_it = std::max_element(result.probabilities.begin(),
    //                                result.probabilities.end());
    // result.predicted_class = std::distance(result.probabilities.begin(), max_it);
    // result.confidence = *max_it;
    //
    // if (!class_labels_.empty() && result.predicted_class < class_labels_.size()) {
    //     result.class_name = class_labels_[result.predicted_class];
    // }
    
    return result;
}

InferenceResult ResNetInference::run_inference_file(const std::string& image_path) {
    // TODO: Implement - load image from file
    //
    // // Use stb_image or similar
    // int width, height, channels;
    // unsigned char* data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    //
    // if (!data) {
    //     throw std::runtime_error("Failed to load image: " + image_path);
    // }
    //
    // // Resize to 224x224 if needed
    // std::vector<uint8_t> resized(224 * 224 * 3);
    // // ... resize logic ...
    //
    // auto result = run_inference(resized.data());
    //
    // stbi_image_free(data);
    // return result;
    
    return InferenceResult{};
}

// =============================================================================
// Layer Execution
// =============================================================================

void ResNetInference::run_conv_layer(size_t layer_idx, 
                                      const int8_t* input, 
                                      int8_t* output) {
    // TODO: Implement
    //
    // const auto& cfg = layer_configs_[layer_idx];
    // const auto& weights = layer_weights_[layer_idx];
    //
    // if (use_accelerator_) {
    //     // Use hardware accelerator
    //     accelerator_->run_layer(layer_idx, input, acc_buffer_.data());
    //
    //     // Requantize
    //     float in_scale = weights->act_input_scale * weights->weight_scale;
    //     float out_scale = weights->act_output_scale;
    //     golden::requantize_int32_to_int8(acc_buffer_.data(), output,
    //                                       output_size, in_scale, out_scale);
    // } else {
    //     // Use golden model
    //     golden::conv2d_bsr_int8(input, *weights->weight_bsr,
    //                             weights->bias.data(), acc_buffer_.data(),
    //                             1, cfg.in_channels, cfg.out_channels,
    //                             cfg.in_height, cfg.in_width,
    //                             cfg.kernel_size, cfg.stride, cfg.padding);
    //
    //     // Requantize + ReLU
    //     // ...
    // }
    //
    // if (cfg.has_relu) {
    //     golden::relu_int8(output, output_size);
    // }
}

void ResNetInference::run_basic_block(size_t block_idx,
                                       const int8_t* input,
                                       int8_t* output) {
    // TODO: Implement
    //
    // // Save identity for residual connection
    // const int8_t* identity = input;
    // std::vector<int8_t> identity_ds;  // For downsample case
    //
    // // First conv + relu
    // size_t conv1_idx = /* calculate from block_idx */;
    // run_conv_layer(conv1_idx, input, act_buffer_b_.data());
    //
    // // Second conv (no relu before residual add)
    // size_t conv2_idx = conv1_idx + 1;
    // run_conv_layer(conv2_idx, act_buffer_b_.data(), output);
    //
    // // Downsample identity if needed
    // if (layer_configs_[conv2_idx].has_downsample) {
    //     // Run 1x1 conv on identity
    //     // ...
    //     identity = identity_ds.data();
    // }
    //
    // // Add residual
    // golden::add_residual(acc_buffer_.data(), identity, acc_buffer_.data(),
    //                      output_size, main_scale, residual_scale);
    //
    // // Final ReLU
    // golden::relu_int8(output, output_size);
}

void ResNetInference::run_fc_layer(const int8_t* input, float* output) {
    // TODO: Implement
    //
    // // FC layer: 512 -> 1000
    // // Can use dense matmul or also BSR if sparse
    //
    // std::vector<int32_t> fc_acc(1000);
    // // golden::matmul_int8(input, fc_weight, fc_acc.data(), 1, 512, 1000);
    //
    // // Dequantize to float
    // // golden::dequantize_int32_to_float(fc_acc.data(), output, 1000, scale);
}

// =============================================================================
// Softmax
// =============================================================================

void ResNetInference::softmax(const float* input, float* output, size_t size) {
    // TODO: Implement
    //
    // // Find max for numerical stability
    // float max_val = *std::max_element(input, input + size);
    //
    // // Compute exp and sum
    // float sum = 0.0f;
    // for (size_t i = 0; i < size; i++) {
    //     output[i] = std::exp(input[i] - max_val);
    //     sum += output[i];
    // }
    //
    // // Normalize
    // for (size_t i = 0; i < size; i++) {
    //     output[i] /= sum;
    // }
}

// =============================================================================
// Utility Functions
// =============================================================================

TopKResult ResNetInference::get_top_k(const InferenceResult& result, int k) {
    TopKResult topk;
    
    // TODO: Implement
    //
    // // Create index array
    // std::vector<size_t> indices(result.probabilities.size());
    // std::iota(indices.begin(), indices.end(), 0);
    //
    // // Partial sort to get top K
    // std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
    //     [&](size_t a, size_t b) {
    //         return result.probabilities[a] > result.probabilities[b];
    //     });
    //
    // for (int i = 0; i < k; i++) {
    //     topk.class_indices.push_back(indices[i]);
    //     topk.probabilities.push_back(result.probabilities[indices[i]]);
    //     if (indices[i] < class_labels_.size()) {
    //         topk.class_names.push_back(class_labels_[indices[i]]);
    //     }
    // }
    
    return topk;
}

void ResNetInference::run_and_print(const std::string& image_path) {
    // TODO: Implement
    //
    // auto result = run_inference_file(image_path);
    // auto topk = get_top_k(result, 5);
    //
    // std::cout << "Top 5 predictions:" << std::endl;
    // for (int i = 0; i < 5; i++) {
    //     std::cout << "  " << i+1 << ". " << topk.class_names[i]
    //               << " (" << topk.probabilities[i] * 100 << "%)" << std::endl;
    // }
    // std::cout << "Latency: " << result.latency_ms << " ms" << std::endl;
}

void ResNetInference::benchmark(const std::string& image_path, int num_runs) {
    // TODO: Implement
    //
    // // Warm up
    // run_inference_file(image_path);
    //
    // // Benchmark
    // double total_time = 0;
    // for (int i = 0; i < num_runs; i++) {
    //     auto result = run_inference_file(image_path);
    //     total_time += result.latency_ms;
    // }
    //
    // std::cout << "Benchmark results (" << num_runs << " runs):" << std::endl;
    // std::cout << "  Average latency: " << total_time / num_runs << " ms" << std::endl;
    // std::cout << "  Throughput: " << 1000.0 / (total_time / num_runs) << " FPS" << std::endl;
}

bool ResNetInference::verify_accuracy(const uint8_t* image_data, float tolerance) {
    // TODO: Implement
    //
    // // Run with accelerator
    // use_accelerator_ = true;
    // auto hw_result = run_inference(image_data);
    //
    // // Run with golden model
    // use_accelerator_ = false;
    // auto sw_result = run_inference(image_data);
    //
    // // Compare
    // float max_diff = 0;
    // for (size_t i = 0; i < 1000; i++) {
    //     float diff = std::abs(hw_result.probabilities[i] - sw_result.probabilities[i]);
    //     max_diff = std::max(max_diff, diff);
    // }
    //
    // return max_diff < tolerance && hw_result.predicted_class == sw_result.predicted_class;
    return true;
}

float ResNetInference::get_model_sparsity() const {
    // TODO: Implement
    //
    // size_t total_blocks = 0;
    // size_t nnz_blocks = 0;
    //
    // for (const auto& weights : layer_weights_) {
    //     if (weights->weight_bsr) {
    //         total_blocks += weights->weight_bsr->num_block_rows *
    //                        weights->weight_bsr->num_block_cols;
    //         nnz_blocks += weights->weight_bsr->nnz_blocks;
    //     }
    // }
    //
    // return 1.0f - static_cast<float>(nnz_blocks) / total_blocks;
    return 0.0f;
}

std::vector<std::pair<std::string, float>> ResNetInference::get_layer_sparsity() const {
    std::vector<std::pair<std::string, float>> sparsities;
    
    // TODO: Implement
    //
    // for (size_t i = 0; i < layer_configs_.size(); i++) {
    //     if (i < layer_weights_.size() && layer_weights_[i]->weight_bsr) {
    //         float sparsity = layer_weights_[i]->weight_bsr->sparsity();
    //         sparsities.emplace_back(layer_configs_[i].name, sparsity);
    //     }
    // }
    
    return sparsities;
}

void ResNetInference::print_model_summary() const {
    // TODO: Implement
    //
    // std::cout << "ResNet-18 Model Summary" << std::endl;
    // std::cout << "=======================" << std::endl;
    // std::cout << "Total layers: " << layer_configs_.size() << std::endl;
    // std::cout << "Model sparsity: " << get_model_sparsity() * 100 << "%" << std::endl;
    // std::cout << std::endl;
    //
    // std::cout << "Layer-by-layer:" << std::endl;
    // for (const auto& [name, sparsity] : get_layer_sparsity()) {
    //     std::cout << "  " << name << ": " << sparsity * 100 << "% sparse" << std::endl;
    // }
}
