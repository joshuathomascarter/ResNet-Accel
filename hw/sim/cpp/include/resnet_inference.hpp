/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                      RESNET_INFERENCE.HPP                                 ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  REPLACES: Nothing directly - this is NEW functionality                  ║
 * ║            Combines: sw/host/accel.py inference logic                    ║
 * ║                      sw/training/export_bsr.py weight loading            ║
 * ║                      sw/golden_models/*.py for CPU fallback              ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║    Complete ResNet-18 inference engine for the sparse accelerator.       ║
 * ║    This is the TOP-LEVEL interface that ties everything together.        ║
 * ║    Load a model, run inference on images, get classification results.    ║
 * ║                                                                           ║
 * ║  WHY THIS FILE IS CRITICAL:                                               ║
 * ║    • This is what you DEMO at the hackathon                              ║
 * ║    • Judges want to see: load model → run image → get result             ║
 * ║    • Encapsulates all complexity behind simple API                       ║
 * ║    • Handles layer-by-layer execution with BSR sparsity                  ║
 * ║                                                                           ║
 * ║  RESNET-18 ARCHITECTURE:                                                  ║
 * ║                                                                           ║
 * ║    Input Image (224x224x3)                                               ║
 * ║         │                                                                ║
 * ║         ▼                                                                ║
 * ║    ┌─────────────┐                                                       ║
 * ║    │ Conv1 7x7   │ 64 filters, stride 2                                  ║
 * ║    └──────┬──────┘                                                       ║
 * ║           │                                                              ║
 * ║    ┌──────▼──────┐                                                       ║
 * ║    │ MaxPool 3x3 │ stride 2                                              ║
 * ║    └──────┬──────┘                                                       ║
 * ║           │                                                              ║
 * ║    ┌──────▼──────┐                                                       ║
 * ║    │ Stage 1     │ 2 BasicBlocks, 64 channels                            ║
 * ║    │ ┌─────────┐ │                                                       ║
 * ║    │ │  3x3    │◄├─────┐ (residual)                                      ║
 * ║    │ │  3x3    │─┼─────┘                                                 ║
 * ║    │ └─────────┘ │                                                       ║
 * ║    └──────┬──────┘                                                       ║
 * ║           │                                                              ║
 * ║    ┌──────▼──────┐                                                       ║
 * ║    │ Stage 2     │ 2 BasicBlocks, 128 channels (downsample)              ║
 * ║    └──────┬──────┘                                                       ║
 * ║           │                                                              ║
 * ║    ┌──────▼──────┐                                                       ║
 * ║    │ Stage 3     │ 2 BasicBlocks, 256 channels (downsample)              ║
 * ║    └──────┬──────┘                                                       ║
 * ║           │                                                              ║
 * ║    ┌──────▼──────┐                                                       ║
 * ║    │ Stage 4     │ 2 BasicBlocks, 512 channels (downsample)              ║
 * ║    └──────┬──────┘                                                       ║
 * ║           │                                                              ║
 * ║    ┌──────▼──────┐                                                       ║
 * ║    │ AvgPool     │ Global average pooling to 512x1x1                     ║
 * ║    └──────┬──────┘                                                       ║
 * ║           │                                                              ║
 * ║    ┌──────▼──────┐                                                       ║
 * ║    │ FC 1000     │ Fully connected to 1000 ImageNet classes              ║
 * ║    └──────┬──────┘                                                       ║
 * ║           │                                                              ║
 * ║           ▼                                                              ║
 * ║    Class Probabilities (1000)                                            ║
 * ║                                                                           ║
 * ║  LAYER EXECUTION FLOW:                                                    ║
 * ║    For each layer:                                                       ║
 * ║      1. Load BSR weights from file (first run) or cache                  ║
 * ║      2. Load activations to DMA buffer                                   ║
 * ║      3. Configure hardware registers (channels, size, stride, etc.)      ║
 * ║      4. Start accelerator                                                ║
 * ║      5. Wait for completion                                              ║
 * ║      6. Apply activation function (ReLU)                                 ║
 * ║      7. Handle residual connection (for BasicBlocks)                     ║
 * ║      8. Requantize output for next layer                                 ║
 * ║                                                                           ║
 * ║  SPARSE ACCELERATION:                                                     ║
 * ║    • Weights are pruned to ~50-70% block sparsity                        ║
 * ║    • Only non-zero 16x16 blocks are computed                             ║
 * ║    • Speedup = 1 / (1 - sparsity) theoretically                          ║
 * ║    • Actual speedup depends on memory efficiency                         ║
 * ║                                                                           ║
 * ║  KEY DATA STRUCTURES:                                                     ║
 * ║                                                                           ║
 * ║    LayerWeights:                                                          ║
 * ║      - BSRMatrix weight       Sparse weight matrix                       ║
 * ║      - vector<int32_t> bias   Bias vector                                ║
 * ║      - float weight_scale     Quantization scale                         ║
 * ║      - float act_scale        Activation scale for this layer            ║
 * ║                                                                           ║
 * ║    InferenceResult:                                                       ║
 * ║      - vector<float> probs    1000 class probabilities                   ║
 * ║      - int top_class          Predicted class index                      ║
 * ║      - float confidence       Top class probability                      ║
 * ║      - float latency_ms       Inference time                             ║
 * ║      - PerfMetrics perf       Detailed performance                       ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef RESNET_INFERENCE_HPP
#define RESNET_INFERENCE_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <array>

// Forward declarations
class AcceleratorDriver;
class BSRPacker;
struct BSRMatrix;
struct PerfMetrics;

/**
 * Layer configuration for ResNet-18
 */
struct ResNetLayerConfig {
    std::string name;           // e.g., "layer1.0.conv1"
    size_t in_channels;
    size_t out_channels;
    size_t in_height;
    size_t in_width;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    bool has_relu;              // Apply ReLU after this layer
    bool has_residual;          // Add residual connection after this layer
    bool has_downsample;        // This layer's residual needs 1x1 downsample
};

/**
 * Loaded weights for one layer
 */
struct LayerWeights {
    BSRMatrix* weight_bsr;      // Sparse weight in BSR format (owned)
    std::vector<int32_t> bias;  // Bias vector (INT32)
    float weight_scale;         // Weight quantization scale
    float act_input_scale;      // Input activation scale
    float act_output_scale;     // Output activation scale
    
    // For downsample in residual blocks
    BSRMatrix* downsample_weight;
    float downsample_scale;
    
    ~LayerWeights();            // Clean up owned BSRMatrix
};

/**
 * Result from inference
 */
struct InferenceResult {
    std::vector<float> probabilities;  // 1000 class probabilities
    int predicted_class;               // Index of highest probability
    float confidence;                  // Value of highest probability
    double latency_ms;                 // Total inference time
    
    // Per-layer performance (optional)
    std::vector<PerfMetrics> layer_perf;
    
    // Human-readable class name (if labels loaded)
    std::string class_name;
};

/**
 * Top-K predictions
 */
struct TopKResult {
    std::vector<int> class_indices;
    std::vector<float> probabilities;
    std::vector<std::string> class_names;
};

/**
 * ResNet-18 inference engine
 */
class ResNetInference {
public:
    /**
     * Constructor
     * 
     * @param use_accelerator  True for FPGA/Verilator, false for CPU golden model
     */
    explicit ResNetInference(bool use_accelerator = true);
    ~ResNetInference();
    
    /**
     * Load model weights from directory
     * 
     * Expected directory structure:
     *   weights_dir/
     *     conv1_weight_int8.npy
     *     conv1_weight_scales.npy
     *     conv1_bias_int32.npy
     *     layer1.0.conv1_weight_int8.npy
     *     ...
     *     fc_weight_int8.npy
     *     fc_bias_int32.npy
     *     quantization_metadata.json
     * 
     * TODO: Implement - for each layer:
     *   1. Load .npy files using numpy reader
     *   2. Convert weights to BSR format
     *   3. Store in layer_weights_ vector
     */
    void load_model(const std::string& weights_dir);
    
    /**
     * Load ImageNet class labels for human-readable output
     */
    void load_labels(const std::string& labels_file);
    
    /**
     * Run inference on a single image
     * 
     * @param image_data  RGB image data, 224x224x3, uint8_t [0-255]
     * @return            InferenceResult with predictions
     * 
     * TODO: Implement - full pipeline:
     *   1. Preprocess image (normalize, quantize)
     *   2. Run each layer through accelerator
     *   3. Handle residual connections
     *   4. Apply softmax to final output
     *   5. Return top predictions
     */
    InferenceResult run_inference(const uint8_t* image_data);
    
    /**
     * Run inference on image file (JPEG, PNG)
     * Uses stb_image or similar for loading
     */
    InferenceResult run_inference_file(const std::string& image_path);
    
    /**
     * Get top-K predictions
     */
    TopKResult get_top_k(const InferenceResult& result, int k = 5);
    
    /**
     * Run inference and print result
     */
    void run_and_print(const std::string& image_path);
    
    /**
     * Benchmark: run N inferences and report statistics
     */
    void benchmark(const std::string& image_path, int num_runs = 100);
    
    /**
     * Compare accelerator result to golden model
     * Returns true if results match within tolerance
     */
    bool verify_accuracy(const uint8_t* image_data, float tolerance = 0.01f);
    
    /**
     * Get total model sparsity
     */
    float get_model_sparsity() const;
    
    /**
     * Get per-layer sparsity breakdown
     */
    std::vector<std::pair<std::string, float>> get_layer_sparsity() const;
    
    /**
     * Print model summary
     */
    void print_model_summary() const;
    
private:
    bool use_accelerator_;
    std::unique_ptr<AcceleratorDriver> accelerator_;
    std::unique_ptr<BSRPacker> bsr_packer_;
    
    // Layer configurations (fixed for ResNet-18)
    std::vector<ResNetLayerConfig> layer_configs_;
    
    // Loaded weights for each layer
    std::vector<std::unique_ptr<LayerWeights>> layer_weights_;
    
    // ImageNet class labels
    std::vector<std::string> class_labels_;
    
    // Intermediate activation buffers
    std::vector<int8_t> act_buffer_a_;
    std::vector<int8_t> act_buffer_b_;
    std::vector<int32_t> acc_buffer_;
    
    /**
     * Initialize layer configurations for ResNet-18
     */
    void init_layer_configs();
    
    /**
     * Preprocess image: normalize and quantize
     * 
     * ImageNet normalization:
     *   mean = [0.485, 0.456, 0.406]
     *   std = [0.229, 0.224, 0.225]
     *   normalized = (pixel/255.0 - mean) / std
     *   quantized = round(normalized / act_scale)
     */
    void preprocess_image(const uint8_t* input, int8_t* output);
    
    /**
     * Run single convolution layer
     */
    void run_conv_layer(size_t layer_idx, const int8_t* input, int8_t* output);
    
    /**
     * Run single BasicBlock (2 conv layers + residual)
     */
    void run_basic_block(size_t block_idx, const int8_t* input, int8_t* output);
    
    /**
     * Run fully connected layer
     */
    void run_fc_layer(const int8_t* input, float* output);
    
    /**
     * Apply softmax to get probabilities
     */
    void softmax(const float* input, float* output, size_t size);
};

#endif // RESNET_INFERENCE_HPP
